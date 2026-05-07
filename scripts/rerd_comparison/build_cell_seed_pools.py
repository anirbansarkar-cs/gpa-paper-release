#!/usr/bin/env python3
"""Build K562 and SK-N-SH seed pools (top-K Gosai by cell activity, GC 45-55%)
mirroring `gosai_seeds_hepg2_gc45_55.h5`. Run-once helper for DNA-CRAFT
K562/SKNSH-targeted GPA submissions.

Outputs:
    results/rerd_comparison/gosai_seeds_k562_gc45_55.h5
    results/rerd_comparison/gosai_seeds_sknsh_gc45_55.h5

Schema (matches HepG2 pool):
    arr_0          (N, 4, 200)  one-hot
    indices        (N, 200)     int64 (A=0, C=1, G=2, T=3)
    oracle_preds   (N,)         per-cell measured activity
    gc_fractions   (N,)         GC fraction
    labels         (N, 3)       [hepg2, k562, sknsh] measured
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT = Path("${GPA_REPO_ROOT}")
GOSAI_CSV = "${GPA_EXTERNAL_ROOT}/DRAKES_data/data_and_model/mdlm/gosai_data/processed_data/gosai_all.csv"
DNA_ALPHABET = {"A": 0, "C": 1, "G": 2, "T": 3}
CELL_TO_COL = {"hepg2": "hepg2", "k562": "k562", "sknsh": "sknsh"}


def encode_seqs(seqs: list) -> np.ndarray:
    n = len(seqs)
    L = len(seqs[0])
    out = np.zeros((n, L), dtype=np.int64)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            out[i, j] = DNA_ALPHABET.get(c, 0)
    return out


def build_pool(cell: str, top_k: int, gc_low: float, gc_high: float, out_path: Path):
    print(f"\n=== Building {cell.upper()} pool ===")
    print(f"Loading {GOSAI_CSV} ...")
    df = pd.read_csv(GOSAI_CSV)
    print(f"  rows: {len(df):,}")
    print(f"  columns: {list(df.columns)[:10]}...")

    cell_col = CELL_TO_COL[cell]
    if cell_col not in df.columns:
        raise KeyError(f"Column '{cell_col}' not in CSV; available: {list(df.columns)}")
    if "seq" not in df.columns:
        raise KeyError(f"Column 'seq' not in CSV; available: {list(df.columns)}")

    seqs = df["seq"].astype(str).tolist()
    L = len(seqs[0])
    print(f"  seq len: {L}")
    labels = df[["hepg2", "k562", "sknsh"]].values.astype(np.float32)

    print("Encoding ...")
    indices = encode_seqs(seqs)
    cell_idx = {"hepg2": 0, "k562": 1, "sknsh": 2}[cell]
    oracle_preds = labels[:, cell_idx].copy()
    gc_fracs = ((indices == 1) | (indices == 2)).mean(axis=1).astype(np.float32)

    print(f"  full pool: {len(indices):,}, {cell} mean={oracle_preds.mean():.3f}, "
          f"max={oracle_preds.max():.3f}")

    # Filter by GC band first (matches HepG2-pool construction)
    gc_mask = (gc_fracs >= gc_low) & (gc_fracs <= gc_high)
    print(f"  GC filter [{gc_low:.2f}, {gc_high:.2f}]: {gc_mask.sum():,} pass")
    indices = indices[gc_mask]
    labels = labels[gc_mask]
    oracle_preds = oracle_preds[gc_mask]
    gc_fracs = gc_fracs[gc_mask]

    # Top-K by cell activity
    if top_k > 0 and top_k < len(indices):
        print(f"Selecting top {top_k:,} by {cell} activity ...")
        ranked = np.argsort(oracle_preds)[::-1].copy()
        chosen = ranked[:top_k]
        indices = indices[chosen]
        labels = labels[chosen]
        oracle_preds = oracle_preds[chosen]
        gc_fracs = gc_fracs[chosen]

    n = len(indices)
    print(f"  final: {n:,} seqs, {cell} score range "
          f"[{oracle_preds.min():.3f}, {oracle_preds.max():.3f}], "
          f"GC mean={gc_fracs.mean()*100:.1f}%")

    onehot = np.eye(4, dtype=np.float32)[indices].transpose(0, 2, 1)
    print(f"Writing {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("arr_0", data=onehot, compression="gzip")
        f.create_dataset("indices", data=indices, compression="gzip")
        f.create_dataset("oracle_preds", data=oracle_preds, compression="gzip")
        f.create_dataset("gc_fractions", data=gc_fracs, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
        f.attrs["target_cell"] = cell
        f.attrs["n_seqs"] = n
        f.attrs["seq_len"] = indices.shape[1]
        f.attrs["gc_low"] = gc_low
        f.attrs["gc_high"] = gc_high
    print(f"  done: {n:,} seqs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["k562", "sknsh"],
                    choices=["hepg2", "k562", "sknsh"])
    ap.add_argument("--top_k", type=int, default=10000,
                    help="Top-K by cell activity (post-GC filter)")
    ap.add_argument("--gc_low", type=float, default=0.45)
    ap.add_argument("--gc_high", type=float, default=0.55)
    ap.add_argument("--out_dir", default=str(PROJECT / "results/rerd_comparison"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for cell in args.cells:
        out_path = out_dir / f"gosai_seeds_{cell}_gc{int(args.gc_low*100)}_{int(args.gc_high*100)}.h5"
        build_pool(cell, args.top_k, args.gc_low, args.gc_high, out_path)


if __name__ == "__main__":
    main()
