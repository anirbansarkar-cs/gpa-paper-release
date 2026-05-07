#!/usr/bin/env python3
"""Deterministic 50/50 split of Gosai MPRA for the DNA-CRAFT comparison.

Produces:
    results/dna_craft_comparison/enhancer/gosai_splits/half_A.csv  (Design-Model half)
    results/dna_craft_comparison/enhancer/gosai_splits/half_B.csv  (Evaluation-Model half)

Both halves retain all three log2FC columns so downstream per-cell training can
reuse the same split across cells.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path("${GPA_REPO_ROOT}")
MPRA_DATA = PROJECT_DIR / "data/gosai_mpra/Table_S2_MPRA_dataset.txt"
DEFAULT_OUT = PROJECT_DIR / "results/dna_craft_comparison/enhancer/gosai_splits"

LABEL_COLS = ["HepG2_log2FC", "K562_log2FC", "SKNSH_log2FC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MPRA_DATA}")
    df = pd.read_csv(MPRA_DATA, sep="\t",
                     usecols=["sequence"] + LABEL_COLS)
    print(f"  total rows:            {len(df)}")

    df["seq_len"] = df["sequence"].str.len()
    df = df[df["seq_len"] == 200].copy()
    df = df.drop(columns=["seq_len"])
    print(f"  rows at 200 bp:        {len(df)}")

    df = df.dropna(subset=LABEL_COLS)
    print(f"  rows with all 3 cells: {len(df)}")

    half_a, half_b = train_test_split(df, test_size=0.5,
                                      random_state=args.seed, shuffle=True)
    half_a = half_a.reset_index(drop=True)
    half_b = half_b.reset_index(drop=True)
    print(f"  half_A (Design-Model):     {len(half_a)}")
    print(f"  half_B (Evaluation-Model): {len(half_b)}")

    a_path = out_dir / "half_A.csv"
    b_path = out_dir / "half_B.csv"
    half_a.to_csv(a_path, index=False)
    half_b.to_csv(b_path, index=False)
    print(f"wrote {a_path}")
    print(f"wrote {b_path}")

    for cell_col in LABEL_COLS:
        print(f"  {cell_col}: "
              f"A mean={half_a[cell_col].mean():+.3f}, "
              f"B mean={half_b[cell_col].mean():+.3f}")


if __name__ == "__main__":
    main()
