#!/usr/bin/env python3
"""Generate 500-seq subsamples of GPA pools using three strategies.

Modes:
  uniform           — RNG-shuffled 500 from N.
  stratified_ag     — 10 AG-score quantile bins x 50 seqs/bin.
  diversity_greedy  — highest-AG start, greedy max-min Hamming distance.

Input H5s must have /indices (N,200) and /ag_k562_scores (N,).
Writes one H5 per (pool, mode) with /indices, /ag_k562_scores, /oracle_preds,
plus the source H5 path in an attribute.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np


def uniform(n: int, k: int, rng) -> np.ndarray:
    if k >= n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)


def stratified_ag(ag: np.ndarray, k: int, rng, n_bins: int = 10) -> np.ndarray:
    n = len(ag)
    if k >= n:
        return np.arange(n)
    per_bin = k // n_bins
    leftover = k - per_bin * n_bins
    quantiles = np.linspace(0, 1, n_bins + 1)
    # numpy quantile returns n_bins+1 edges
    edges = np.quantile(ag, quantiles)
    edges[-1] += 1e-9
    # Assign each seq to a bin; use np.digitize
    bin_idx = np.clip(np.digitize(ag, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        pool = np.where(bin_idx == b)[0]
        take = per_bin + (1 if b < leftover else 0)
        if len(pool) == 0:
            continue
        if len(pool) <= take:
            out.append(pool)
        else:
            out.append(rng.choice(pool, size=take, replace=False))
    arr = np.concatenate(out)
    # pad if under-quota (sparse bins)
    if len(arr) < k:
        remain = np.setdiff1d(np.arange(n), arr, assume_unique=True)
        need = k - len(arr)
        if len(remain) >= need:
            arr = np.concatenate([arr, rng.choice(remain, need, replace=False)])
    return arr[:k]


def diversity_greedy(indices: np.ndarray, ag: np.ndarray, k: int) -> np.ndarray:
    """Greedy max-min Hamming distance — start at argmax(AG)."""
    n, L = indices.shape
    if k >= n:
        return np.arange(n)
    # Convert to uint8 for memory savings
    idx_u8 = indices.astype(np.uint8)
    order = []
    start = int(np.argmax(ag))
    order.append(start)
    # min_dist[i] = min Hamming distance from seq i to selected set
    min_dist = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    # update with the starting seq
    while len(order) < k:
        last = idx_u8[order[-1]]
        # vectorized Hamming
        # (N,L) != (L,) broadcasting
        d = (idx_u8 != last[None, :]).sum(axis=1, dtype=np.int32)
        np.minimum(min_dist, d, out=min_dist)
        # prevent re-selection
        min_dist_mask = min_dist.copy()
        min_dist_mask[order] = -1
        nxt = int(np.argmax(min_dist_mask))
        order.append(nxt)
    return np.asarray(order, dtype=np.int64)


def load_pool(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        indices = f["indices"][:]
        ag = f["ag_k562_scores"][:] if "ag_k562_scores" in f else None
        ln = f["oracle_preds"][:] if "oracle_preds" in f else None
        # Optionally the 3-head rescoring artifacts
        extra = {}
        for k in ("hepg2_legnet_scores", "wtc11_legnet_scores",
                  "k562_legnet_scores", "spec_k562_legnet"):
            if k in f:
                extra[k] = f[k][:]
    return indices, ag, ln, extra


def write_sub(out_path: Path, src: Path, sel: np.ndarray,
              indices, ag, ln, extra, mode: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("indices", data=indices[sel])
        if ag is not None:
            f.create_dataset("ag_k562_scores", data=ag[sel])
        if ln is not None:
            f.create_dataset("oracle_preds", data=ln[sel])
        for k, arr in extra.items():
            f.create_dataset(k, data=arr[sel])
        f.attrs["source_pool"] = str(src)
        f.attrs["mode"] = mode
        f.attrs["n_subsample"] = len(sel)
        f.attrs["n_full"] = len(indices)
        f.create_dataset("source_idx", data=sel.astype(np.int64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pools", nargs="+", required=True,
                   help="H5 paths or dirs containing gpa_output_pool.h5.")
    p.add_argument("--k", type=int, default=500)
    p.add_argument("--modes", nargs="+",
                   default=["uniform", "stratified_ag", "diversity_greedy"])
    p.add_argument("--out_root",
                   default="results/k562_mdlm_gpa/subsamples")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    files = []
    for pth in args.pools:
        p_ = Path(pth)
        if p_.is_dir():
            files.extend(sorted(p_.rglob("gpa_output_pool.h5")))
        else:
            files.append(p_)
    print(f"Pools: {len(files)}  modes={args.modes}  k={args.k}",
          flush=True)

    rng_master = np.random.default_rng(args.seed)
    out_root = Path(args.out_root)
    summary = {}

    for h5_path in files:
        pool_tag = h5_path.parent.name
        indices, ag, ln, extra = load_pool(h5_path)
        n = len(indices)
        if ag is None:
            print(f"  SKIP (no AG): {h5_path}")
            continue
        summary[pool_tag] = {"n_full": n,
                              "full_ag_mean": float(ag.mean()),
                              "full_ag_max": float(ag.max()),
                              "full_ln_mean": float(ln.mean()) if ln is not None else None}
        for mode in args.modes:
            # deterministic per-mode seed
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            t0 = time.perf_counter()
            if mode == "uniform":
                sel = uniform(n, args.k, rng)
            elif mode == "stratified_ag":
                sel = stratified_ag(ag, args.k, rng)
            elif mode == "diversity_greedy":
                sel = diversity_greedy(indices, ag, args.k)
            else:
                raise ValueError(mode)
            dt = time.perf_counter() - t0
            out_path = out_root / pool_tag / f"{mode}_k{args.k}.h5"
            write_sub(out_path, h5_path, sel, indices, ag, ln, extra, mode)
            summary[pool_tag][f"{mode}_ag_mean"] = float(ag[sel].mean())
            summary[pool_tag][f"{mode}_ag_max"] = float(ag[sel].max())
            summary[pool_tag][f"{mode}_elapsed_s"] = dt
            print(f"  {pool_tag:55s} {mode:18s} n={len(sel)} "
                  f"ag_mean={ag[sel].mean():.3f}  "
                  f"dt={dt:.2f}s -> {out_path.name}", flush=True)

    out_json = out_root / f"subsample_summary_seed{args.seed}.json"
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {out_json}")


if __name__ == "__main__":
    main()
