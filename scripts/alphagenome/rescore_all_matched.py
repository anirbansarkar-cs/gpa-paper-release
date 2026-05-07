#!/usr/bin/env python3
"""Rescore all matched-seed pools (GPA + ISM + LEDIDI) with the new PyTorch
AlphaGenome encoder, K562/HepG2/WTC11.

Loads each cell's model ONCE and sweeps all inputs through it (avoids paying
the 30 s ckpt-load tax 42 times). Writes:
  - GPA H5: ag_{cell}_scores_v2 datasets, ag_k562_scores replaced (old stashed
            as ag_k562_scores_jax).
  - ISM/LEDIDI CSV: <input>_ag.csv with k562_ag/hepg2_ag/wtc11_ag columns.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "${GPA_REPO_ROOT}")
from scripts.alphagenome.score_sequences_torch import (
    CHECKPOINTS, pad_to_230, IDX_TO_BASE, load_seqs_from_h5,
)
from alphagenome_encoder_ft import EncoderMPRAModel


PROJECT = Path("${GPA_REPO_ROOT}")

GPA_RUNS_50178 = [
    "run_v13_nodps_bf10_b1k_cap10_ism50178",
    "run_v13_nodps_bf10_b1k_cap20_ism50178",
    "run_v13b_nodps_bf10_b2k_cap30_ism50178",
    "run_v13c_nodps_bf10_b1k_cap50_ism50178",
    "run_v13_nodps_bf10_b1k_uncapped_ism50178",
    "run_v13b_dps_kl_bf10_b2k_cap10_ism50178",
    "run_v13b_dps_kl_bf10_b2k_cap20_ism50178",
    "run_v13c_dps_kl_bf10_b2k_cap30_nogcdps_ism50178",
    "run_v13c_dps_kl_bf10_b2k_cap50_ism50178",
    "run_v13c_dps_std_bf1_eta1k_b1k_uncap_ism50178",
]
GPA_RUNS_181692 = [r.replace("ism50178", "ism181692").replace("v13c", "v14")
                                                     .replace("v13b", "v14")
                                                     .replace("v13", "v14")
                                                     .replace("uncapped", "uncap")
                   for r in GPA_RUNS_50178]
ALL_GPA_RUNS = GPA_RUNS_50178 + GPA_RUNS_181692


def collect_inputs():
    """Return list of (label, path, kind, seq_col) tuples."""
    items = []

    for r in ALL_GPA_RUNS:
        h5 = PROJECT / "results" / "k562_mdlm_gpa" / r / "gpa_output_pool.h5"
        if h5.exists():
            items.append((f"GPA::{r}", h5, "h5", None))

    for uid in (50178, 181692):
        c = PROJECT / "results" / "ism_baseline" / f"ism_trajectory_uid{uid}.csv"
        if c.exists():
            items.append((f"ISM::uid{uid}", c, "csv", "sequence"))

    for d in sorted((PROJECT / "results" / "ledidi_comparison" / "matched").glob("uid*")):
        for c in sorted(d.glob("ledidi_uid*_t*_l*.csv")):
            if c.stem.endswith("_ag"):
                continue
            items.append((f"LEDIDI::{c.parent.name}::{c.stem}", c, "csv", "edited_sequence"))
    return items


def read_seqs(path: Path, kind: str, seq_col: str | None) -> list[str]:
    if kind == "h5":
        return load_seqs_from_h5(path)
    df = pd.read_csv(path)
    if seq_col not in df.columns:
        for cand in ("sequence", "edited_sequence"):
            if cand in df.columns:
                seq_col = cand; break
    return df[seq_col].astype(str).tolist()


def score_batch(model, seqs: list[str], batch_size: int) -> np.ndarray:
    out = []
    for i in range(0, len(seqs), batch_size):
        with torch.no_grad():
            preds = model.predict_sequences(seqs[i:i+batch_size], construct_mode="promoter_barcode")
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        out.append(preds.reshape(-1))
    return np.concatenate(out)


def write_h5(path: Path, cell_preds: dict, stash_old_k562: bool):
    with h5py.File(path, "a") as f:
        for cell, preds in cell_preds.items():
            key = f"ag_{cell}_scores_v2"
            if key in f: del f[key]
            f.create_dataset(key, data=np.asarray(preds, dtype=np.float32))
        if stash_old_k562 and "ag_k562_scores" in f and "ag_k562_scores_jax" not in f:
            f["ag_k562_scores_jax"] = f["ag_k562_scores"][...]


def write_csv_ag(path: Path, cell_preds: dict):
    df = pd.read_csv(path)
    for cell, preds in cell_preds.items():
        df[f"{cell}_ag"] = preds
    out = path.with_name(path.stem + "_ag.csv")
    df.to_csv(out, index=False)
    return out


def main():
    items = collect_inputs()
    print(f"[rescore] {len(items)} inputs to score", flush=True)
    cells = ["k562", "hepg2", "wtc11"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    # Pre-load all sequences (paying once) so we don't re-read per cell
    seqs_per_item = []
    for label, path, kind, seq_col in items:
        seqs = read_seqs(path, kind, seq_col)
        seqs = [pad_to_230(s) for s in seqs]
        seqs_per_item.append(seqs)
        print(f"  loaded {label}: N={len(seqs)}", flush=True)

    # cell -> { item_index: predictions }
    preds_by_cell = {c: {} for c in cells}
    for cell in cells:
        ckpt = CHECKPOINTS[cell]
        t0 = time.time()
        print(f"\n[rescore] loading {cell} from {ckpt}", flush=True)
        model = EncoderMPRAModel.from_checkpoint(ckpt, device=device)
        model.eval()
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        for idx, ((label, _, _, _), seqs) in enumerate(zip(items, seqs_per_item)):
            t1 = time.time()
            preds = score_batch(model, seqs, batch_size)
            preds_by_cell[cell][idx] = preds
            print(f"  {cell:>5} {label:<60s} N={len(seqs):>5}  "
                  f"mean={preds.mean():.3f} max={preds.max():.3f}  "
                  f"({time.time()-t1:.1f}s)", flush=True)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    # Write outputs
    print(f"\n[rescore] writing outputs", flush=True)
    for idx, (label, path, kind, _) in enumerate(items):
        cell_preds = {c: preds_by_cell[c][idx] for c in cells}
        if kind == "h5":
            write_h5(path, cell_preds, stash_old_k562=True)
            print(f"  H5 update -> {path}", flush=True)
        else:
            out = write_csv_ag(path, cell_preds)
            print(f"  CSV write -> {out}", flush=True)

    print("[rescore] done", flush=True)


if __name__ == "__main__":
    main()
