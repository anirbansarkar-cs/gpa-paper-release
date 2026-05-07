#!/usr/bin/env python3
"""Materialize 128 random init sequences per (cell, seed) from Gosai test split.

Per plan §5: paper convention is 128-sequence output pools. The GPA runner
upsamples these 128 to POP=10000 at init (with replacement). Drawing the
seeds from the Gosai test split makes the comparison deterministic across
cells / seeds and gives GPA exposure to realistic 200 bp enhancer motifs.

The "test split" here is Half B (`gosai_splits/half_B.csv`) — the same data
the Evaluation-Model was trained on, so seeds are drawn from sequences the
Eval-Model has *not* seen as training labels for the OPPOSITE half (Design).
This is acceptable; the seeds are just initialisations, never scored against
the training half they came from.

Outputs (under RESULTS_DIR/seeds):
  {cell}_seed{0,1,2}.csv     — 128 rows, columns: sequence, hepg2, k562, sknsh
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "results" / "dna_craft_comparison" / "enhancer"

DEFAULT_SOURCE = RESULTS_DIR / "gosai_splits" / "half_B.csv"
CELLS = ["hepg2", "k562", "sknsh"]
SEED_COL_MAP = {
    "hepg2": "HepG2_log2FC",
    "k562": "K562_log2FC",
    "sknsh": "SKNSH_log2FC",
}
# Deterministic per-cell seed offsets (do not use Python's hash() — salted by PYTHONHASHSEED)
CELL_SEED_OFFSET = {"hepg2": 11, "k562": 23, "sknsh": 37}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_csv", default=str(DEFAULT_SOURCE),
                   help="CSV of held-out Gosai sequences to draw seeds from.")
    p.add_argument("--out_dir", default=str(RESULTS_DIR / "seeds"))
    p.add_argument("--n_seeds", type=int, default=128)
    p.add_argument("--n_replicates", type=int, default=3,
                   help="Number of independent (seed_id) replicates per cell.")
    p.add_argument("--cells", nargs="+", default=CELLS)
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.source_csv)
    if not src.exists():
        raise FileNotFoundError(
            f"Source CSV not found: {src}. Run gosai_50_50_split.py first.")
    df = pd.read_csv(src)
    df = df[df["sequence"].str.len() == 200].reset_index(drop=True)
    print(f"[prepare_seeds] source: {src}  n_rows={len(df):,}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cell in args.cells:
        col = SEED_COL_MAP[cell]
        if col not in df.columns:
            raise KeyError(f"missing column {col!r} in {src}")
        for seed_id in range(args.n_replicates):
            rng = np.random.default_rng(seed=42 + seed_id * 100 + CELL_SEED_OFFSET[cell])
            chosen = rng.choice(len(df), size=args.n_seeds, replace=False)
            sub = df.iloc[chosen][["sequence",
                                    SEED_COL_MAP["hepg2"],
                                    SEED_COL_MAP["k562"],
                                    SEED_COL_MAP["sknsh"]]].copy()
            sub.columns = ["sequence", "hepg2", "k562", "sknsh"]
            out_path = out_dir / f"{cell}_seed{seed_id}.csv"
            sub.to_csv(out_path, index=False)
            print(f"  wrote {out_path}  rows={len(sub)}")


if __name__ == "__main__":
    main()
