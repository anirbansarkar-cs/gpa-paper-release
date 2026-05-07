#!/usr/bin/env python3
"""
Score GPA-generated sequences with fine-tuned AlphaGenome MPRA oracles.

Loads K562 and/or HepG2 stage1 checkpoints (JAX/Orbax) and scores
sequences from H5 (indices key) or CSV (sequence column).

Usage:
    # Score an H5 file with both cell types
    python scripts/alphagenome/score_sequences.py \
        --input results/rerd_comparison/run_v9_mean/gpa_output.h5 \
        --output results/alphagenome/v9_mean_scores.csv

    # Score with only K562
    python scripts/alphagenome/score_sequences.py \
        --input results/rerd_comparison/run_v9_mean/gpa_output.h5 \
        --cell_types k562 \
        --output results/alphagenome/v9_mean_k562.csv

    # Score a CSV with sequence column
    python scripts/alphagenome/score_sequences.py \
        --input results/ism_high_oracle_scored.csv \
        --seq_column sequence \
        --output results/alphagenome/ism_scores.csv

    # Also append scores back to H5
    python scripts/alphagenome/score_sequences.py \
        --input results/rerd_comparison/run_v9_mean/gpa_output.h5 \
        --output results/alphagenome/v9_mean_scores.csv \
        --append_h5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from alphagenome_ft_mpra.oracle import load_oracle, MPRAOracle

# Default checkpoint paths (relocated 2026-04-25 to ${GPA_SHARED_ROOT}/models/alphagenome_encoder/jax/)
CHECKPOINTS = {
    "k562":  "${GPA_SHARED_ROOT}/models/alphagenome_encoder/jax/mpra-K562-optimal/stage1",
    "hepg2": "${GPA_SHARED_ROOT}/models/alphagenome_encoder/jax/mpra-HepG2-optimal/stage1",
    "wtc11": "${GPA_SHARED_ROOT}/models/alphagenome_encoder/jax/mpra-WTC11-optimal/stage1",
}

# Encoding: A=0, C=1, G=2, T=3 (matches GPA/diffusion convention)
IDX_TO_BASE = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(IDX_TO_BASE)}


def indices_to_onehot(indices: np.ndarray) -> np.ndarray:
    """Convert integer indices (N, L) to one-hot (N, L, 4)."""
    N, L = indices.shape
    onehot = np.zeros((N, L, 4), dtype=np.float32)
    for i in range(4):
        onehot[:, :, i] = (indices == i).astype(np.float32)
    return onehot


def sequences_to_onehot(sequences: list[str]) -> np.ndarray:
    """Convert DNA string sequences to one-hot (N, L, 4)."""
    indices = np.array(
        [[BASE_TO_IDX.get(c.upper(), 0) for c in seq] for seq in sequences],
        dtype=np.int64,
    )
    return indices_to_onehot(indices)


def load_input(input_path: str, seq_column: str = "sequence"):
    """Load sequences from H5 or CSV. Returns (onehot, metadata_dict)."""
    path = Path(input_path)

    if path.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            if "indices" in f:
                indices = f["indices"][:]
                onehot = indices_to_onehot(indices)
            elif "arr_0" in f:
                # One-hot in (N, 4, L) format -> (N, L, 4)
                arr = f["arr_0"][:]
                onehot = arr.transpose(0, 2, 1).astype(np.float32)
            else:
                raise KeyError(f"No 'indices' or 'arr_0' in {path}")

            # Gather existing oracle scores for comparison
            metadata = {}
            for key in f.keys():
                if key.startswith("oracle_") or key in ("gc_fractions",):
                    metadata[key] = f[key][:]

        return onehot, metadata

    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        if seq_column not in df.columns:
            raise KeyError(f"Column '{seq_column}' not found in {path}. "
                           f"Available: {list(df.columns)}")
        sequences = df[seq_column].tolist()
        onehot = sequences_to_onehot(sequences)
        metadata = {col: df[col].values for col in df.columns if col != seq_column}
        return onehot, metadata

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def score_with_oracle(
    oracle: MPRAOracle,
    onehot: np.ndarray,
    batch_size: int = 64,
    mode: str = "core",
) -> np.ndarray:
    """Score sequences with AlphaGenome oracle. Returns (N,) scores."""
    N = onehot.shape[0]
    all_scores = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = onehot[start:end]  # (B, L, 4)
        scores = oracle.predict(batch, mode=mode, batch_size=batch_size)
        all_scores.append(np.asarray(scores))

        if (start // batch_size) % 10 == 0:
            print(f"  Scored {end}/{N} sequences", flush=True)

    return np.concatenate(all_scores, axis=0).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="Score sequences with AlphaGenome MPRA oracles")
    parser.add_argument("--input", required=True,
                        help="Input H5 (with indices) or CSV (with sequence column)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path for scores")
    parser.add_argument("--cell_types", nargs="+", default=["k562", "hepg2"],
                        choices=["k562", "hepg2"],
                        help="Cell types to score (default: both)")
    parser.add_argument("--seq_column", default="sequence",
                        help="Column name for sequences in CSV input")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Inference batch size")
    parser.add_argument("--mode", default="core",
                        choices=["core", "flanked", "full"],
                        help="Construct mode for adapters/promoter/barcode")
    parser.add_argument("--append_h5", action="store_true",
                        help="Also append AlphaGenome scores back to input H5")
    parser.add_argument("--k562_ckpt", default=CHECKPOINTS["k562"],
                        help="K562 checkpoint directory")
    parser.add_argument("--hepg2_ckpt", default=CHECKPOINTS["hepg2"],
                        help="HepG2 checkpoint directory")
    args = parser.parse_args()

    # Load input sequences
    print(f"[Load] Reading {args.input}")
    onehot, metadata = load_input(args.input, seq_column=args.seq_column)
    N, L, _ = onehot.shape
    print(f"  Loaded {N:,} sequences of length {L}")

    # Score with each cell type
    ckpt_map = {"k562": args.k562_ckpt, "hepg2": args.hepg2_ckpt}
    ag_scores = {}

    for cell_type in args.cell_types:
        ckpt_dir = ckpt_map[cell_type]
        print(f"\n[Oracle] Loading {cell_type.upper()} AlphaGenome oracle: {ckpt_dir}")
        # Training used init_seq_len=281 (max random shift of 250bp construct).
        # Flatten pooling needs ceil(L/128)=3 encoder positions → L>256bp.
        # mode="core" constructs: left(15) + payload(200) + right(15) + promoter(35) + barcode(15) = 280bp
        # → ceil(280/128)=3 encoder positions → flatten=3*1536=4608 ✓
        oracle = load_oracle(
            ckpt_dir,
            left_adapter="AGGACCGGATCAACT",
            right_adapter="CATTGCGTGAACCGA",
        )
        print(f"  Pooling: {oracle.pooling_type}, center_bp: {oracle.center_bp}")

        print(f"\n[Score] Scoring {N:,} sequences with {cell_type.upper()} oracle...")
        scores = score_with_oracle(oracle, onehot, batch_size=args.batch_size,
                                   mode=args.mode)
        ag_scores[f"{cell_type}_ag"] = scores
        print(f"  {cell_type.upper()}: mean={scores.mean():.4f}, "
              f"median={np.median(scores):.4f}, max={scores.max():.4f}")

    # Build output DataFrame
    out_df = pd.DataFrame({"seq_idx": np.arange(N)})
    for col, vals in ag_scores.items():
        out_df[col] = vals

    # Add existing Enformer eval scores for correlation analysis
    for cell in ("k562", "hepg2", "sknsh"):
        eval_key = f"oracle_{cell}_eval"
        if eval_key in metadata:
            out_df[f"{cell}_enformer_eval"] = metadata[eval_key]
        ft_key = f"oracle_{cell}_ft"
        if ft_key in metadata:
            out_df[f"{cell}_enformer_ft"] = metadata[ft_key]

    if "gc_fractions" in metadata:
        out_df["gc_fraction"] = metadata["gc_fractions"]

    # Save CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\n[Save] Wrote {out_path}")

    # Optionally append to H5
    if args.append_h5 and Path(args.input).suffix in (".h5", ".hdf5"):
        import h5py
        print(f"\n[H5] Appending AlphaGenome scores to {args.input}")
        with h5py.File(args.input, "a") as f:
            for col, vals in ag_scores.items():
                key = f"oracle_{col}"
                if key in f:
                    del f[key]
                f.create_dataset(key, data=vals, compression="gzip")
            f.attrs["alphagenome_scored"] = True

    # Cross-oracle correlation analysis
    print(f"\n{'=' * 72}")
    print("CROSS-ORACLE CORRELATION: Enformer-eval vs AlphaGenome")
    print(f"{'=' * 72}")

    from scipy import stats

    for cell in ("k562", "hepg2"):
        ag_col = f"{cell}_ag"
        enf_col = f"{cell}_enformer_eval"
        if ag_col in out_df.columns and enf_col in out_df.columns:
            ag = out_df[ag_col].values
            enf = out_df[enf_col].values
            mask = np.isfinite(ag) & np.isfinite(enf)
            if mask.sum() > 2:
                r_pearson, p_pearson = stats.pearsonr(ag[mask], enf[mask])
                r_spearman, p_spearman = stats.spearmanr(ag[mask], enf[mask])
                print(f"\n  {cell.upper()}:")
                print(f"    Pearson  r={r_pearson:.4f}  (p={p_pearson:.2e})")
                print(f"    Spearman r={r_spearman:.4f}  (p={p_spearman:.2e})")
                print(f"    AG:  mean={ag[mask].mean():.4f}, max={ag[mask].max():.4f}")
                print(f"    Enf: mean={enf[mask].mean():.4f}, max={enf[mask].max():.4f}")
            else:
                print(f"\n  {cell.upper()}: insufficient data for correlation")
        elif ag_col in out_df.columns:
            print(f"\n  {cell.upper()}: No Enformer eval scores available for comparison")

    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    main()
