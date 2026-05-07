#!/usr/bin/env python3
"""Bundle AG (K562) rescore over all GPA / ISM / LEDIDI outputs of v2.

Discovers in a v2 root dir:
  - GPA h5s (``gpa_output_pool.h5`` and ``gpa_output.h5``) under ``gpa/*/``
  - ISM trajectories (``trajectories.csv``) under ``ism/*/``
  - LEDIDI outputs (``ledidi_edited_*.csv``) under ``ledidi/*/``

For each input scores K562 only (the comparison's target cell) with the JAX
fine-tuned encoder; writes:
  - GPA h5: ``ag_k562_scores_jax_v2`` dataset (overwrites if present)
  - CSV: parallel ``<base>_ag.csv`` with a ``k562_ag_jax`` column appended

Designed to be called from inside the JAX (``alphagenome``) conda env. See
``score_ag_pool.sh`` for the env-setup wrapper.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, "${GPA_REPO_ROOT}")
from scripts.alphagenome.score_sequences_jax import (
    load_jax_oracle, score, sequences_to_onehot,
)

CELL = "k562"
DEFAULT_BATCH = 64
SEQ_LEN = 200
# AG IS strand-sensitive (empirically: fwd vs RC mean diff ~0.23, Spearman 0.93
# on 200 K562 seqs from a smoke ISM trajectory). Worth the 2x inference cost to
# average fwd + RC predictions, both for the headline AG number and for the
# hacking-metric Pearson(LN, AG). Same TTA convention as score_ln_pool.py.


def reverse_complement_onehot(onehot: np.ndarray) -> np.ndarray:
    """RC of (N, L, 4) one-hot in A=0,C=1,G=2,T=3 channel order.
    Reverse positions, swap A<->T (channels 0<->3) and C<->G (channels 1<->2).
    """
    rc = onehot[:, ::-1, :].copy()
    rc = rc[:, :, [3, 2, 1, 0]]
    return rc


def discover(v2_root: Path) -> list[tuple[str, Path, str, str | None]]:
    """Return list of (label, path, kind, seq_col)."""
    items: list[tuple[str, Path, str, str | None]] = []

    # GPA: both the pool. Also walk gpa_v3/
    # (low-LN-high-AG hunt with archive thresholding) and any future gpa_*.
    for gpa_subdir in sorted(v2_root.glob("gpa*")):
        if not gpa_subdir.is_dir():
            continue
        for run_dir in sorted(gpa_subdir.glob("*")):
            if not run_dir.is_dir():
                continue
            for fname in ("gpa_output_pool.h5", "gpa_output.h5",
                          "gpa_output_filtered.h5"):
                h5 = run_dir / fname
                if h5.exists():
                    items.append((f"GPA::{gpa_subdir.name}::{run_dir.name}::{fname}",
                                  h5, "h5", None))

    # ISM: trajectories.csv per pool
    for pool_dir in sorted((v2_root / "ism").glob("*")):
        if not pool_dir.is_dir():
            continue
        c = pool_dir / "trajectories.csv"
        if c.exists():
            items.append((f"ISM::{pool_dir.name}", c, "csv", "sequence"))

    # LEDIDI: ledidi_edited_*.csv per (pool, l) dir
    for pool_dir in sorted((v2_root / "ledidi").glob("*")):
        if not pool_dir.is_dir():
            continue
        for c in sorted(pool_dir.glob("ledidi_edited_*.csv")):
            if c.stem.endswith("_ag"):
                continue
            items.append((f"LEDIDI::{pool_dir.name}::{c.stem}", c, "csv",
                          "edited_sequence"))
    return items


def load_h5_seqs_as_onehot(h5p: Path) -> np.ndarray:
    """Read sequence indices from GPA h5. pool h5 uses ``indices``;
    archive uses ``seqs`` (per the ARCHIVE_THRESHOLD path in run_k562_mdlm_gpa.py).
    Returns (N, 200, 4) one-hot.
    """
    with h5py.File(h5p, "r") as f:
        if "indices" in f:
            idx = f["indices"][:].astype(np.int64)
        elif "seqs" in f:
            idx = f["seqs"][:].astype(np.int64)
        else:
            raise KeyError(f"Neither 'indices' nor 'seqs' in {h5p}")
    eye = np.eye(4, dtype=np.float32)
    return eye[idx]


def load_csv_seqs_as_onehot(csv: Path, seq_col: str | None) -> tuple[np.ndarray,
                                                                     pd.DataFrame]:
    df = pd.read_csv(csv)
    if seq_col is None or seq_col not in df.columns:
        for cand in ("sequence", "edited_sequence"):
            if cand in df.columns:
                seq_col = cand; break
        else:
            raise KeyError(f"No sequence column in {csv}")
    seqs = df[seq_col].astype(str).tolist()
    onehot = sequences_to_onehot(seqs)              # (N, L, 4)
    return onehot, df


def write_h5_jax(h5p: Path, preds_fwd: np.ndarray, preds_rc: np.ndarray,
                 stage: str = "stage1") -> None:
    """Write JAX AG scores to h5. stage1 → ag_{cell}_scores_jax_v2{,_rc};
    stage2 → ag_{cell}_scores_jax_v2_stage2{,_rc}. Existing keys for the
    OTHER stage are left untouched."""
    stage_tag = "" if stage == "stage1" else f"_{stage}"
    with h5py.File(h5p, "a") as f:
        for suffix, arr in (("", preds_fwd), ("_rc", preds_rc)):
            key = f"ag_{CELL}_scores_jax_v2{stage_tag}{suffix}"
            if key in f:
                del f[key]
            f.create_dataset(key, data=np.asarray(arr, dtype=np.float32))


def write_csv_jax(csv: Path, df: pd.DataFrame,
                  preds_fwd: np.ndarray, preds_rc: np.ndarray,
                  stage: str = "stage1") -> Path:
    """Write/append ``<base>_ag.csv``. stage1 → k562_ag_jax{,_rc};
    stage2 → k562_ag_jax_stage2{,_rc}."""
    stage_tag = "" if stage == "stage1" else f"_{stage}"
    out = csv.with_name(csv.stem + "_ag.csv")
    fwd_col = f"{CELL}_ag_jax{stage_tag}"
    rc_col = f"{CELL}_ag_jax{stage_tag}_rc"
    if out.exists():
        existing = pd.read_csv(out)
        if len(existing) != len(df):
            print(f"  [warn] {out} length mismatch ({len(existing)} vs {len(df)}); "
                  f"overwriting with new", flush=True)
            existing = df.copy()
        existing[fwd_col] = preds_fwd
        existing[rc_col] = preds_rc
        existing.to_csv(out, index=False)
    else:
        df = df.copy()
        df[fwd_col] = preds_fwd
        df[rc_col] = preds_rc
        df.to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_root", required=True,
                    help="gpa_vs_ism_ledidi_v2 root")
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--filter", default=None,
                    help="Optional substring to filter labels (e.g. 'pool_A', 'GPA')")
    ap.add_argument("--shard", type=int, default=0,
                    help="Shard index (0-based). With --num_shards N, this "
                         "process scores every Nth input starting from shard.")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Total number of shards. Set to N to split the work "
                         "across N parallel jobs.")
    ap.add_argument("--stage", default="stage1", choices=["stage1", "stage2"],
                    help="JAX checkpoint stage: stage1 (frozen base) or "
                         "stage2 (full fine-tune). Stage2 writes to "
                         "*_stage2 keys, leaving stage1 keys untouched.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print discovered inputs without scoring")
    args = ap.parse_args()

    v2_root = Path(args.v2_root)
    items = discover(v2_root)
    if args.filter:
        items = [it for it in items if args.filter in it[0]]
    if args.num_shards > 1:
        items = [it for i, it in enumerate(items)
                 if i % args.num_shards == args.shard]
        print(f"[score_ag_pool] shard {args.shard}/{args.num_shards} "
              f"-> {len(items)} inputs", flush=True)
    else:
        print(f"[score_ag_pool] {len(items)} inputs", flush=True)
    for it in items:
        print(f"  - {it[0]}  ({it[1]})", flush=True)
    if args.dry_run or not items:
        return

    print(f"\n[score_ag_pool] loading JAX K562 oracle ({args.stage}) ...", flush=True)
    t0 = time.time()
    oracle = load_jax_oracle(CELL, stage=args.stage)
    print(f"[score_ag_pool] oracle loaded in {time.time()-t0:.1f}s", flush=True)

    for label, path, kind, seq_col in items:
        t0 = time.time()
        try:
            if kind == "h5":
                onehot = load_h5_seqs_as_onehot(path)
            else:
                onehot, df = load_csv_seqs_as_onehot(path, seq_col)
            preds_fwd = score(oracle, onehot, batch_size=args.batch_size)
            onehot_rc = reverse_complement_onehot(onehot)
            preds_rc_only = score(oracle, onehot_rc, batch_size=args.batch_size)
            preds_rc = (preds_fwd + preds_rc_only) / 2.0
            stage_tag = "" if args.stage == "stage1" else f"_{args.stage}"
            if kind == "h5":
                write_h5_jax(path, preds_fwd, preds_rc, stage=args.stage)
                out_str = (f"  [h5] wrote ag_{CELL}_scores_jax_v2{stage_tag}"
                           f"{{,_rc}} to {path.name}")
            else:
                out = write_csv_jax(path, df, preds_fwd, preds_rc, stage=args.stage)
                out_str = f"  [csv] wrote {out.name} ({CELL}_ag_jax{stage_tag}{{,_rc}})"
            dt = time.time() - t0
            print(f"{label}: N={len(preds_fwd)} fwd_mean={preds_fwd.mean():.3f} "
                  f"rcavg_mean={preds_rc.mean():.3f} fwd_max={preds_fwd.max():.3f} "
                  f"rcavg_max={preds_rc.max():.3f}  {out_str}  ({dt:.1f}s)", flush=True)
        except Exception as e:
            print(f"{label}: FAILED — {e!r}", flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
