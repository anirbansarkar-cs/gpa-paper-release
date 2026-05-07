#!/usr/bin/env python3
"""
Build binary-labeled CSV for HyenaDNA fine-tune on Reddy 2024 promoter MPRA.

Matches Ctrl-DNA's published promoter conditioning scheme exactly: 3-digit
label "<JURKAT><K562><THP1>" where each digit is 1 if that cell's activity
is >= the cell's median (computed on TRAIN rows only) and 0 otherwise.
Ctrl-DNA's `reinforce_multi_lagrange.py:get_prefix_label()` hardcodes
"100" / "010" / "001" prompts for JURKAT / K562 / THP1 targets, which are
exactly the 3 "this cell only high" cases under this scheme.

Mirrors the enhancer pipeline (scripts/ctrl_dna_comparison/prepare_data.py
:make_labeled_csv), adapted for promoter column names.

Usage:
    python scripts/ctrl_dna_comparison/promoter/build_hyenadna_labels.py
"""
import argparse
import os

import numpy as np
import pandas as pd

CELLS = ["JURKAT", "K562", "THP1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="scripts/ctrl_dna_comparison/promoter/data/finetuning_data.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="scripts/ctrl_dna_comparison/data/promoter_labeled.csv",
    )
    args = parser.parse_args()

    print(f"Reading {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"  {len(df):,} rows")

    train_mask = df["is_train"].astype(bool).values
    print(f"  Train rows: {train_mask.sum():,} (medians computed on train only)")

    bits = np.zeros((len(df), len(CELLS)), dtype=int)
    for i, cell in enumerate(CELLS):
        median = float(np.median(df.loc[train_mask, cell].values))
        bits[:, i] = (df[cell].values >= median).astype(int)
        print(f"  {cell}: median={median:+.4f}, train 1-fraction="
              f"{bits[train_mask, i].mean():.3f}")

    labels = [f"{b[0]}{b[1]}{b[2]}" for b in bits]
    out = pd.DataFrame({"sequence": df["sequence"].values, "label": labels})
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    label_counts = out["label"].value_counts()
    print(f"\nWrote {args.output_csv}: {len(out):,} rows")
    print(f"  Label distribution (8 classes): {label_counts.to_dict()}")


if __name__ == "__main__":
    main()
