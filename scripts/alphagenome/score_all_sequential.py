#!/usr/bin/env python3
"""
Score ALL GPA runs with AlphaGenome HepG2 oracle sequentially on a single GPU.
Loads the model once, then iterates over all H5 files.

Usage:
    python scripts/alphagenome/score_all_sequential.py \
        --results_dir results/rerd_comparison \
        --output results/alphagenome/all_runs_hepg2_summary.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from alphagenome_ft_mpra.oracle import load_oracle

HEPG2_CKPT = "${GPA_SHARED_ROOT}/alphagenome_encoder/mpra-HepG2-optimal/stage1"
K562_CKPT = "${GPA_SHARED_ROOT}/alphagenome_encoder/mpra-K562-optimal/stage1"

IDX_TO_BASE = "ACGT"


def indices_to_onehot(indices):
    N, L = indices.shape
    onehot = np.zeros((N, L, 4), dtype=np.float32)
    for i in range(4):
        onehot[:, :, i] = (indices == i).astype(np.float32)
    return onehot


def score_h5(oracle, h5_path, batch_size=64):
    """Score a single H5 file. Returns (scores, n_seqs) or (None, 0) on error."""
    import h5py
    try:
        with h5py.File(h5_path, "r") as f:
            if "indices" in f:
                indices = f["indices"][:]
                onehot = indices_to_onehot(indices)
            elif "arr_0" in f:
                arr = f["arr_0"][:]
                onehot = arr.transpose(0, 2, 1).astype(np.float32)
            else:
                return None, 0
    except Exception as e:
        print(f"    ERROR reading {h5_path}: {e}")
        return None, 0

    N = onehot.shape[0]
    all_scores = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = onehot[start:end]
        scores = oracle.predict(batch, mode="core", batch_size=batch_size)
        all_scores.append(np.asarray(scores).astype(np.float64))

    return np.concatenate(all_scores, axis=0), N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/rerd_comparison")
    parser.add_argument("--output", default="results/alphagenome/all_runs_hepg2_summary.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cell_types", nargs="+", default=["hepg2"],
                        choices=["k562", "hepg2"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all gpa_output.h5 files, skip pareto variants
    h5_files = sorted(results_dir.glob("run_*/gpa_output.h5"))
    h5_files = [f for f in h5_files if "_pareto" not in f.parent.name]
    print(f"Found {len(h5_files)} runs to score")

    # Load oracles once
    oracles = {}
    ckpt_map = {"hepg2": HEPG2_CKPT, "k562": K562_CKPT}
    for cell in args.cell_types:
        print(f"\n[Load] Loading {cell.upper()} oracle from {ckpt_map[cell]}")
        oracles[cell] = load_oracle(
            ckpt_map[cell],
            left_adapter="AGGACCGGATCAACT",
            right_adapter="CATTGCGTGAACCGA",
        )
        print(f"  Loaded OK")

    # Score all runs
    rows = []
    t0 = time.time()
    for i, h5_path in enumerate(h5_files):
        run_name = h5_path.parent.name.replace("run_", "")
        row = {"run": run_name}

        for cell, oracle in oracles.items():
            scores, n_seqs = score_h5(oracle, h5_path, batch_size=args.batch_size)
            if scores is not None:
                row["N"] = n_seqs
                row[f"{cell}_ag_mean"] = float(scores.mean())
                row[f"{cell}_ag_median"] = float(np.median(scores))
                row[f"{cell}_ag_max"] = float(scores.max())
                row[f"{cell}_ag_p95"] = float(np.percentile(scores, 95))
                row[f"{cell}_ag_std"] = float(scores.std())

        rows.append(row)
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed * 60
        print(f"  [{i+1}/{len(h5_files)}] {run_name}: "
              + ", ".join(f"{c}_ag={row.get(f'{c}_ag_mean', 'ERR'):.3f}"
                          if isinstance(row.get(f'{c}_ag_mean'), float)
                          else f"{c}_ag=ERR"
                          for c in args.cell_types)
              + f"  ({rate:.1f} runs/min)")

    # Save full results
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n[Save] Wrote {out_path} ({len(df)} runs)")

    # Print top 20 by HepG2-AG mean
    if "hepg2_ag_mean" in df.columns:
        print(f"\n{'='*80}")
        print("TOP 20 RUNS BY HepG2-AG MEAN")
        print(f"{'='*80}")
        top = df.nlargest(20, "hepg2_ag_mean")
        print(f"{'Run':<50} {'N':>5} {'HepG2-AG mean':>14} {'HepG2-AG max':>13} {'HepG2-AG p95':>13}")
        print("-" * 100)
        for _, r in top.iterrows():
            print(f"{r['run']:<50} {r.get('N', 0):>5.0f} {r['hepg2_ag_mean']:>14.4f} {r['hepg2_ag_max']:>13.4f} {r.get('hepg2_ag_p95', 0):>13.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min for {len(h5_files)} runs")


if __name__ == "__main__":
    main()
