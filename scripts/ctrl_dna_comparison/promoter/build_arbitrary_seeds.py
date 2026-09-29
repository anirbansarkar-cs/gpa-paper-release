#!/usr/bin/env python3
"""
Build 5 arbitrary (uniformly random, un-filtered) seed sets for promoter GPA.

Each set is an independent uniform sample of 5000 sequences from the TRAIN
split of `finetuning_data.csv`. Not cell-specific, not filtered by any
activity metric — neutral starting pools for GPA.

Usage:
    python scripts/ctrl_dna_comparison/promoter/build_arbitrary_seeds.py
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_csv",
        default="scripts/ctrl_dna_comparison/promoter/data/finetuning_data.csv",
    )
    ap.add_argument(
        "--output_dir",
        default="scripts/ctrl_dna_comparison/promoter/data",
    )
    ap.add_argument("--n_sets", type=int, default=5)
    ap.add_argument("--set_size", type=int, default=5000,
                    help="Sequences per set. If > train rows, sample with replacement.")
    ap.add_argument("--master_seed", type=int, default=20260416)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    train = df.loc[df["is_train"].astype(bool)].reset_index(drop=True)
    print(f"Reading {args.input_csv}: {len(train):,} train rows")
    assert "sequence" in train.columns

    replace = args.set_size > len(train)
    print(f"Sampling {args.n_sets} sets of {args.set_size} each  replace={replace}")

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.master_seed)
    for set_id in range(args.n_sets):
        # Fresh seed per set so the 5 draws are independent, but deterministic
        # given master_seed.
        subrng = np.random.default_rng(args.master_seed + 1 + set_id)
        idx = subrng.choice(len(train), args.set_size, replace=replace)
        out = pd.DataFrame({
            "sequence": train["sequence"].values[idx],
            "JURKAT": train["JURKAT"].values[idx],
            "K562": train["K562"].values[idx],
            "THP1": train["THP1"].values[idx],
        })
        path = Path(args.output_dir) / f"seeds_arbitrary_set{set_id}.csv"
        out.to_csv(path, index=False)
        activity_means = out[["JURKAT", "K562", "THP1"]].mean().to_dict()
        print(f"  set{set_id}: {path} ({len(out):,} rows, "
              f"mean activity J={activity_means['JURKAT']:+.3f} "
              f"K={activity_means['K562']:+.3f} T={activity_means['THP1']:+.3f})")


if __name__ == "__main__":
    main()
