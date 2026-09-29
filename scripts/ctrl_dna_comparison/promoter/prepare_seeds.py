"""Build per-cell seed CSVs for CtrlDNA / GPA promoter runs.

Ctrl-DNA's pipeline starts from low-activity sequences and optimizes upward.
We pick the bottom-N rows by target-cell expression from the held-out test
split of `finetuning_data.csv`, writing one CSV per cell.

Output columns: sequence, <CELL>
"""
import argparse
from pathlib import Path

import pandas as pd

CELLS = ["JURKAT", "K562", "THP1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_seeds", type=int, default=256)
    ap.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split to draw seeds from.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_csv)
    split_col = f"is_{args.split}"
    pool = df.loc[df[split_col]].reset_index(drop=True)
    print(f"split={args.split} pool={len(pool)}")

    for cell in CELLS:
        ordered = pool.sort_values(by=cell, ascending=True).head(args.n_seeds)
        out = ordered[["sequence", cell]].reset_index(drop=True)
        dest = out_dir / f"seeds_{cell}.csv"
        out.to_csv(dest, index=False)
        print(f"  {cell}: wrote {len(out)} seeds -> {dest} "
              f"({cell} range [{out[cell].min():.3f}, {out[cell].max():.3f}])")


if __name__ == "__main__":
    main()
