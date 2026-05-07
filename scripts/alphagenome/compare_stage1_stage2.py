#!/usr/bin/env python3
"""Quick comparison of AG K562 stage1 vs stage2 scores on existing GPA output."""

import sys
import time
import numpy as np
import h5py

from alphagenome_ft_mpra.oracle import load_oracle

H5_PATH = "results/rerd_comparison/run_v22b_rnd_K1/gpa_output.h5"
STAGE1 = "${GPA_SHARED_ROOT}/alphagenome_encoder/mpra-K562-optimal/stage1"
STAGE2 = "${GPA_SHARED_ROOT}/alphagenome_encoder/mpra-K562-optimal/stage2"
N_SEQS = 100  # score first 100 for speed

LEFT_ADAPTER = "AGGACCGGATCAACT"
RIGHT_ADAPTER = "CATTGCGTGAACCGA"

IDX_TO_BASE = "ACGT"


def indices_to_onehot(indices):
    N, L = indices.shape
    onehot = np.zeros((N, L, 4), dtype=np.float32)
    for i in range(4):
        onehot[:, :, i] = (indices == i).astype(np.float32)
    return onehot


def main():
    # Load sequences
    with h5py.File(H5_PATH, "r") as f:
        indices = f["indices"][:N_SEQS]
    onehot = indices_to_onehot(indices)
    print(f"Loaded {N_SEQS} sequences from {H5_PATH}")

    # Score with stage1
    print(f"\nLoading stage1 from {STAGE1}...")
    t0 = time.time()
    oracle1 = load_oracle(STAGE1, left_adapter=LEFT_ADAPTER, right_adapter=RIGHT_ADAPTER)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    scores1 = np.asarray(oracle1.predict(onehot), dtype=np.float64)
    print(f"  Stage1 scores: mean={scores1.mean():.4f}, std={scores1.std():.4f}, "
          f"min={scores1.min():.4f}, max={scores1.max():.4f}")

    # Score with stage2
    print(f"\nLoading stage2 from {STAGE2}...")
    t0 = time.time()
    oracle2 = load_oracle(STAGE2, left_adapter=LEFT_ADAPTER, right_adapter=RIGHT_ADAPTER)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    scores2 = np.asarray(oracle2.predict(onehot), dtype=np.float64)
    print(f"  Stage2 scores: mean={scores2.mean():.4f}, std={scores2.std():.4f}, "
          f"min={scores2.min():.4f}, max={scores2.max():.4f}")

    # Compare
    diff = scores1 - scores2
    print(f"\n--- Stage1 - Stage2 ---")
    print(f"  Mean diff:   {diff.mean():.4f}")
    print(f"  Std diff:    {diff.std():.4f}")
    print(f"  Max |diff|:  {np.abs(diff).max():.4f}")
    print(f"  Correlation: {np.corrcoef(scores1, scores2)[0,1]:.6f}")

    # Check if existing oracle_k562 matches stage2
    with h5py.File(H5_PATH, "r") as f:
        if "oracle_k562" in f:
            existing = f["oracle_k562"][:N_SEQS]
            existing = np.asarray(existing, dtype=np.float64)
            match = np.allclose(existing, scores2, atol=1e-3)
            print(f"\n  Existing oracle_k562 matches stage2? {match}")
            if not match:
                ediff = existing - scores2
                print(f"  Existing vs stage2: mean diff={ediff.mean():.4f}, max |diff|={np.abs(ediff).max():.4f}")
            # Also compare existing (LegNet) vs stage1
            ediff1 = existing - scores1
            print(f"  Existing (LegNet) vs stage1: mean diff={ediff1.mean():.4f}, max |diff|={np.abs(ediff1).max():.4f}")
            print(f"  Existing (LegNet) scores: mean={existing.mean():.4f}, std={existing.std():.4f}")


if __name__ == "__main__":
    main()
