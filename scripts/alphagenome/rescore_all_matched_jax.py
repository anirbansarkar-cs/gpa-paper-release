#!/usr/bin/env python3
"""Rescore all matched-seed pools with the new JAX AlphaGenome encoder.

JAX analogue of ``rescore_all_matched.py``. Uses ``score_sequences_jax`` which
works around two bugs in ``alphagenome_ft_mpra.oracle.load_oracle``:
  1. Saved ``config.json`` has ``use_encoder_output: false`` when it should be True.
  2. ``load_oracle`` does not forward ``init_seq_len``; defaults to 2**14.

Validated against the labmate's PyT reference (`alphagenome-encoder-ft`) — same
construct (left+insert+right+promoter+barcode = 281bp), same adapter/promoter/
barcode strings. Validated against existing K562 JAX scores
(``ag_k562_scores_jax``): Spearman = 0.96 on a non-saturated pool.

Writes:
  - GPA H5: ``ag_{k562,hepg2,wtc11}_scores_jax_v2`` datasets. Existing
            ``ag_*_scores_v2`` (PyT) and ``ag_k562_scores_jax`` (old JAX,
            pre-relocation) are left untouched.
  - ISM/LEDIDI CSV: appends ``{cell}_ag_jax`` columns to the existing
                    ``<input>_ag.csv`` (which already has PyT ``{cell}_ag``).
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, "${GPA_REPO_ROOT}")
from scripts.alphagenome.rescore_all_matched import collect_inputs, read_seqs
from scripts.alphagenome.score_sequences_jax import (
    JAX_CHECKPOINTS, load_jax_oracle, score, sequences_to_onehot,
)
from scripts.alphagenome.score_sequences_torch import pad_to_230


def write_h5(path: Path, cell_preds: dict):
    with h5py.File(path, "a") as f:
        for cell, preds in cell_preds.items():
            key = f"ag_{cell}_scores_jax_v2"
            if key in f:
                del f[key]
            f.create_dataset(key, data=np.asarray(preds, dtype=np.float32))


def write_csv_ag_jax(path: Path, cell_preds: dict):
    """Append _jax columns to the existing <base>_ag.csv (which has PyT
    columns from rescore_all_matched.py). If that file does not exist yet,
    fall back to writing a fresh <base>_ag_jax.csv."""
    pyt_csv = path.with_name(path.stem + "_ag.csv")
    if pyt_csv.exists():
        df = pd.read_csv(pyt_csv)
        for cell, preds in cell_preds.items():
            df[f"{cell}_ag_jax"] = preds
        df.to_csv(pyt_csv, index=False)
        return pyt_csv
    df = pd.read_csv(path)
    for cell, preds in cell_preds.items():
        df[f"{cell}_ag_jax"] = preds
    out = path.with_name(path.stem + "_ag_jax.csv")
    df.to_csv(out, index=False)
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Score only the first N inputs (smoke test).")
    p.add_argument("--dry_run", action="store_true",
                   help="Score but skip writing to H5/CSV.")
    args = p.parse_args()

    items = collect_inputs()
    if args.limit is not None:
        # Pick a representative slice: 1 GPA H5 + 1 ISM CSV + 1 LEDIDI CSV.
        # Ensures we exercise both H5 and CSV write paths under any limit >= 3.
        if args.limit >= 3:
            h5s = [it for it in items if it[2] == "h5"][:1]
            isms = [it for it in items if "ISM::" in it[0]][:1]
            ledidis = [it for it in items if "LEDIDI::" in it[0]][:1]
            items = h5s + isms + ledidis + [
                it for it in items if it not in h5s + isms + ledidis
            ][: max(0, args.limit - 3)]
        else:
            items = items[: args.limit]
    print(f"[jax_rescore] {len(items)} inputs to score"
          + (f" (LIMITED, dry_run={args.dry_run})" if args.limit else ""),
          flush=True)
    cells = ["k562", "hepg2", "wtc11"]

    # Pre-load + onehot every input once. JAX scorer wants 200bp inserts
    # (not pre-padded with adapters); MPRAOracle.predict(mode='core') adds
    # left/right/promoter/barcode internally.
    seqs_per_item = []
    onehot_per_item = []
    for label, path, kind, seq_col in items:
        seqs = read_seqs(path, kind, seq_col)
        # Strip adapters if caller pre-padded to 230. JAX 'core' mode adds them.
        normalized = []
        for s in seqs:
            if len(s) == 230:
                s = s[15:215]
            elif len(s) != 200:
                raise ValueError(f"{label}: unexpected seq len {len(s)} (expected 200 or 230)")
            normalized.append(s)
        seqs_per_item.append(normalized)
        onehot_per_item.append(sequences_to_onehot(normalized))
        print(f"  loaded {label}: N={len(normalized)}", flush=True)

    # cell -> { item_index: predictions }
    preds_by_cell = {c: {} for c in cells}
    for cell in cells:
        t0 = time.time()
        print(f"\n[jax_rescore] loading {cell} from {JAX_CHECKPOINTS[cell]}", flush=True)
        oracle = load_jax_oracle(cell)
        print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
        for idx, ((label, _, _, _), oh) in enumerate(zip(items, onehot_per_item)):
            t1 = time.time()
            preds = score(oracle, oh, batch_size=64)
            preds_by_cell[cell][idx] = preds
            print(f"  {cell:>5} {label:<60s} N={len(preds):>5}  "
                  f"mean={preds.mean():+.3f} max={preds.max():+.3f}  "
                  f"({time.time() - t1:.1f}s)", flush=True)
        del oracle
        gc.collect()

    # Write outputs
    if args.dry_run:
        print(f"\n[jax_rescore] dry_run, skipping writes", flush=True)
    else:
        print(f"\n[jax_rescore] writing outputs", flush=True)
        for idx, (label, path, kind, _) in enumerate(items):
            cell_preds = {c: preds_by_cell[c][idx] for c in cells}
            if kind == "h5":
                write_h5(path, cell_preds)
                print(f"  H5 update -> {path}", flush=True)
            else:
                out = write_csv_ag_jax(path, cell_preds)
                print(f"  CSV append -> {out}", flush=True)

    print("[jax_rescore] done", flush=True)


if __name__ == "__main__":
    main()
