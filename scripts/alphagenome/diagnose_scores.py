#!/usr/bin/env python3
"""Diagnose AG oracle scoring: check encoding, predict vs predict_sequences, etc."""

import numpy as np
import h5py
from alphagenome_ft_mpra.oracle import load_oracle

STAGE1 = "${GPA_SHARED_ROOT}/alphagenome_encoder/mpra-K562-optimal/stage1"
H5_PATH = "results/rerd_comparison/run_v22b_rnd_K1/gpa_output.h5"
LEFT_ADAPTER = "AGGACCGGATCAACT"
RIGHT_ADAPTER = "CATTGCGTGAACCGA"

IDX_TO_BASE = "ACGT"  # A=0, C=1, G=2, T=3


def indices_to_sequence(indices_1d):
    """Convert index array to DNA string."""
    return "".join(IDX_TO_BASE[i] for i in indices_1d)


def indices_to_onehot(indices):
    N, L = indices.shape
    onehot = np.zeros((N, L, 4), dtype=np.float32)
    for i in range(4):
        onehot[:, :, i] = (indices == i).astype(np.float32)
    return onehot


def seq_to_onehot(seq):
    """Convert DNA string to one-hot (1, L, 4)."""
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    L = len(seq)
    onehot = np.zeros((1, L, 4), dtype=np.float32)
    for i, b in enumerate(seq):
        onehot[0, i, base_to_idx[b]] = 1.0
    return onehot


def main():
    oracle = load_oracle(STAGE1, left_adapter=LEFT_ADAPTER, right_adapter=RIGHT_ADAPTER)

    # 1. Trivial repeat
    repeat_seq = "ACGT" * 50
    score_repeat = np.asarray(oracle.predict_sequences([repeat_seq], mode="core"), dtype=np.float64)
    print(f"ACGT*50 via predict_sequences: {score_repeat[0]:.4f}")

    score_repeat_oh = np.asarray(oracle.predict(seq_to_onehot(repeat_seq), mode="core"), dtype=np.float64)
    print(f"ACGT*50 via predict(onehot):   {score_repeat_oh[0]:.4f}")

    # 2. Load first 10 GPA sequences from H5
    with h5py.File(H5_PATH, "r") as f:
        indices = f["indices"][:10]
        if "oracle_k562" in f:
            legnet_scores = np.asarray(f["oracle_k562"][:10], dtype=np.float64)
            print(f"\nLegNet oracle_k562 (first 10): {legnet_scores}")

    # 3. Score via one-hot (my original method)
    onehot = indices_to_onehot(indices)
    scores_oh = np.asarray(oracle.predict(onehot, mode="core"), dtype=np.float64)
    print(f"\nGPA seqs via predict(onehot):      mean={scores_oh.mean():.4f}, scores={scores_oh}")

    # 4. Score via predict_sequences (string method) to cross-check
    sequences = [indices_to_sequence(indices[i]) for i in range(10)]
    scores_str = np.asarray(oracle.predict_sequences(sequences, mode="core"), dtype=np.float64)
    print(f"GPA seqs via predict_sequences:    mean={scores_str.mean():.4f}, scores={scores_str}")

    # 5. Check if they match
    diff = scores_oh - scores_str
    print(f"\npredict vs predict_sequences diff: mean={diff.mean():.6f}, max={np.abs(diff).max():.6f}")

    # 6. Print first sequence to verify encoding
    print(f"\nFirst GPA sequence (first 30 bases): {sequences[0][:30]}")
    print(f"First GPA indices (first 30):        {indices[0][:30]}")

    # 7. Check GC content of first few
    for i in range(3):
        seq = sequences[i]
        gc = (seq.count("G") + seq.count("C")) / len(seq)
        print(f"  Seq {i}: GC={gc:.2%}, score_oh={scores_oh[i]:.4f}, score_str={scores_str[i]:.4f}")

    # 8. Random sequences for baseline
    rng = np.random.default_rng(42)
    random_seqs = ["".join(rng.choice(list("ACGT"), 200)) for _ in range(10)]
    scores_random = np.asarray(oracle.predict_sequences(random_seqs, mode="core"), dtype=np.float64)
    print(f"\nRandom seqs (10):  mean={scores_random.mean():.4f}, std={scores_random.std():.4f}, "
          f"min={scores_random.min():.4f}, max={scores_random.max():.4f}")


if __name__ == "__main__":
    main()
