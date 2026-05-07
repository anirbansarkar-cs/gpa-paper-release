#!/usr/bin/env python3
"""Bundle LegNet K562 rescore over all GPA / ISM / LEDIDI outputs of v2.

Mirror of ``score_ag_pool.py`` but for the K562 LegNet oracle. Runs in the
``d3_cuda118`` conda env (PyTorch). For each input writes both fwd-only and
fwd+RC-averaged LN scores (the LegNet paper / Penzar 2023 convention is to
average fwd and reverse-complement predictions at inference time; the trained
model is approximately strand-invariant via training augmentation, but
inference-time TTA tightens the prediction).

Outputs:
  - GPA h5: ``ln_k562_scores_v2{,_rc}`` datasets
  - ISM/LEDIDI CSV: appends ``k562_ln{,_rc}`` columns to existing
    ``<base>_ag.csv`` (which has AG cols), or writes fresh ``<base>_ag.csv``
    if AG hasn't been scored yet.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "${GPA_REPO_ROOT}")
from scripts.k562_mdlm_gpa.k562_oracle import K562Oracle, load_k562_oracle
from scripts.ledidi_comparison.score_ag_pool import discover  # reuse discovery

CELL = "k562"
DEFAULT_BATCH = 256
SEQ_LEN = 200
DEFAULT_ORACLE = (
    "${GPA_REPO_ROOT}/model_zoo/lentimpra/oracle_models/"
    "best_model-epoch=24-val_pearson=0.814.ckpt"
)


def reverse_complement_indices(idx: np.ndarray) -> np.ndarray:
    """RC of (N, L) indices in A=0,C=1,G=2,T=3.
    complement = 3 - x; reverse positions; .copy() per project negative-stride rule.
    """
    return (3 - idx)[:, ::-1].copy()


def load_h5_indices(h5p: Path) -> np.ndarray:
    """Read sequence indices. paper pool uses ``indices``."""
    with h5py.File(h5p, "r") as f:
        if "indices" in f:
            return f["indices"][:].astype(np.int64)
        elif "seqs" in f:
            return f["seqs"][:].astype(np.int64)
        else:
            raise KeyError(f"Neither 'indices' nor 'seqs' in {h5p}")


def _unused_old_load_h5_indices(h5p: Path) -> np.ndarray:
    with h5py.File(h5p, "r") as f:
        return f["indices"][:].astype(np.int64)


def load_csv_indices(csv: Path, seq_col: str | None) -> tuple[np.ndarray,
                                                              pd.DataFrame]:
    df = pd.read_csv(csv)
    if seq_col is None or seq_col not in df.columns:
        for cand in ("sequence", "edited_sequence"):
            if cand in df.columns:
                seq_col = cand; break
        else:
            raise KeyError(f"No sequence column in {csv}")
    seqs = df[seq_col].astype(str).tolist()
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    idx = np.array(
        [[base_to_idx[c] for c in s] for s in seqs], dtype=np.int64
    )
    return idx, df


def write_h5_ln(h5p: Path, preds_fwd: np.ndarray, preds_rc: np.ndarray) -> None:
    with h5py.File(h5p, "a") as f:
        for suffix, arr in (("", preds_fwd), ("_rc", preds_rc)):
            key = f"ln_{CELL}_scores_v2{suffix}"
            if key in f:
                del f[key]
            f.create_dataset(key, data=np.asarray(arr, dtype=np.float32))


def write_csv_ln(csv: Path, df: pd.DataFrame,
                 preds_fwd: np.ndarray, preds_rc: np.ndarray) -> Path:
    out = csv.with_name(csv.stem + "_ag.csv")
    if out.exists():
        existing = pd.read_csv(out)
        if len(existing) != len(df):
            print(f"  [warn] {out} length mismatch ({len(existing)} vs {len(df)}); "
                  f"overwriting with new", flush=True)
            existing = df.copy()
        existing[f"{CELL}_ln"] = preds_fwd
        existing[f"{CELL}_ln_rc"] = preds_rc
        existing.to_csv(out, index=False)
    else:
        df = df.copy()
        df[f"{CELL}_ln"] = preds_fwd
        df[f"{CELL}_ln_rc"] = preds_rc
        df.to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_root", required=True)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--oracle_ckpt", default=DEFAULT_ORACLE)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    v2_root = Path(args.v2_root)
    items = discover(v2_root)
    if args.filter:
        items = [it for it in items if args.filter in it[0]]
    print(f"[score_ln_pool] {len(items)} inputs", flush=True)
    for it in items:
        print(f"  - {it[0]}  ({it[1]})", flush=True)
    if args.dry_run or not items:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[score_ln_pool] loading K562 LegNet oracle ({device}) ...", flush=True)
    t0 = time.time()
    lit = load_k562_oracle(args.oracle_ckpt, device=device)
    oracle = K562Oracle(lit, device=device)
    print(f"[score_ln_pool] loaded in {time.time()-t0:.1f}s", flush=True)

    for label, path, kind, seq_col in items:
        t0 = time.time()
        try:
            if kind == "h5":
                idx = load_h5_indices(path)
            else:
                idx, df = load_csv_indices(path, seq_col)

            idx_rc = reverse_complement_indices(idx)
            preds_fwd, _ = oracle.score(torch.from_numpy(idx).long(),
                                        batch_size=args.batch_size)
            preds_rc_only, _ = oracle.score(torch.from_numpy(idx_rc).long(),
                                            batch_size=args.batch_size)
            preds_fwd = np.asarray(preds_fwd, dtype=np.float64).reshape(-1)
            preds_rc_only = np.asarray(preds_rc_only, dtype=np.float64).reshape(-1)
            preds_rc = (preds_fwd + preds_rc_only) / 2.0

            if kind == "h5":
                write_h5_ln(path, preds_fwd, preds_rc)
                out_str = f"  [h5] wrote ln_{CELL}_scores_v2{{,_rc}} to {path.name}"
            else:
                out = write_csv_ln(path, df, preds_fwd, preds_rc)
                out_str = f"  [csv] wrote {out.name} ({CELL}_ln{{,_rc}})"
            dt = time.time() - t0
            print(f"{label}: N={len(preds_fwd)} fwd_mean={preds_fwd.mean():.3f} "
                  f"rc_mean={preds_rc.mean():.3f} fwd_max={preds_fwd.max():.3f} "
                  f"rc_max={preds_rc.max():.3f}  {out_str}  ({dt:.1f}s)", flush=True)
        except Exception as e:
            print(f"{label}: FAILED — {e!r}", flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
