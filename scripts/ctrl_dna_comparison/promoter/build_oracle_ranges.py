#!/usr/bin/env python3
"""
Compute per-cell oracle prediction ranges (min/max) for the promoter
comparison, for fitness normalization in run_ctrldna_promoter.py and
evaluation. Runs each of the 3 regLM EnformerModel checkpoints on a
5K-sample subset of the training sequences.

Mirrors scripts/ctrl_dna_comparison/prepare_data.py:compute_oracle_ranges
which writes oracle_ranges.json for the enhancer gReLU oracle.

Usage:
    python scripts/ctrl_dna_comparison/promoter/build_oracle_ranges.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from oracle_adapter import (  # noqa: E402
    PROMOTER_CELLS, PromoterOraclePool, PromoterCellAdapter,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_csv",
        default="scripts/ctrl_dna_comparison/promoter/data/finetuning_data.csv",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="scripts/ctrl_dna_comparison/promoter/checkpoints",
    )
    parser.add_argument(
        "--output",
        default="scripts/ctrl_dna_comparison/promoter/data/oracle_ranges.json",
    )
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA — oracle inference will be slow")

    print(f"Reading {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    train = df.loc[df["is_train"].astype(bool)].reset_index(drop=True)
    n = min(args.sample_size, len(train))
    sample = train.sample(n=n, random_state=args.seed).reset_index(drop=True)
    print(f"  Sampled {n:,} train rows")

    print(f"\nLoading 3 EnformerModel checkpoints from {args.ckpt_dir}")
    pool = PromoterOraclePool(args.ckpt_dir, device=device)

    seqs = sample["sequence"].tolist()
    ranges = {}
    for cell in PROMOTER_CELLS:
        adapter = PromoterCellAdapter(pool, cell)
        preds = []
        for start in range(0, n, args.batch_size):
            chunk = seqs[start:start + args.batch_size]
            p = adapter(chunk).detach().cpu().numpy()
            preds.append(p)
        preds = np.concatenate(preds)
        r_min, r_max = float(preds.min()), float(preds.max())
        r_mean = float(preds.mean())
        ranges[cell] = {"min": r_min, "max": r_max, "mean": r_mean}
        print(f"  {cell:<6} min={r_min:.4f} max={r_max:.4f} mean={r_mean:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(ranges, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
