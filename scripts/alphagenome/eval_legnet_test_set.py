#!/usr/bin/env python3
"""Evaluate LegNet on the K562 lentiMPRA test set.

Computes Pearson/Spearman correlation between LegNet predictions and
ground-truth MPRA activity.
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
DATA_H5 = "model_zoo/lentimpra/lenti_MPRA_K562_data.h5"
LEGNET_CKPT = "${GPA_REPO_ROOT}/model_zoo/lentimpra/oracle_models/best_model-epoch=24-val_pearson=0.814.ckpt"
BATCH_SIZE = 256
OUTDIR = Path("results/ag_vs_legnet_analysis")

# Add d3_evaluation_pipeline to path for mpralegnet
sys.path.insert(0, str(Path.home() / "d3_evaluation_pipeline"))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    print(f"Loading test data from {DATA_H5}...")
    with h5py.File(DATA_H5, "r") as f:
        onehot_test = f["onehot_test"][:]  # (N, 230, 4) -- A=0, C=1, G=2, T=3
        y_test = f["y_test"][:].flatten().astype(np.float64)

    N = len(y_test)
    print(f"  Test set: {N} sequences, shape={onehot_test.shape}")
    print(f"  y_test: mean={y_test.mean():.4f}, std={y_test.std():.4f}")

    # H5 stores (N, 230, 4) already in A=0, G=1, C=2, T=3 order (matches LegNet's CODES).
    # HDF5Dataset just transposes to (N, 4, 230) — no channel swap needed.
    onehot_legnet = onehot_test.transpose(0, 2, 1)  # (N, 4, 230)
    print(f"  Transposed to LegNet format: {onehot_legnet.shape} (A=0,G=1,C=2,T=3)")

    # Load LegNet
    print(f"\nLoading LegNet from {LEGNET_CKPT}...")
    from mpralegnet import load_model
    t0 = time.time()
    model, config = load_model(LEGNET_CKPT)
    model.eval()
    model.cuda()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Score
    print(f"  Scoring {N} sequences (batch_size={BATCH_SIZE})...")
    t0 = time.time()
    preds_list = []
    with torch.no_grad():
        for i in range(0, N, BATCH_SIZE):
            batch = torch.tensor(
                onehot_legnet[i:i + BATCH_SIZE], dtype=torch.float32
            ).cuda()
            out = model(batch).cpu().numpy().flatten()
            preds_list.append(out)
    preds = np.concatenate(preds_list).astype(np.float64)
    print(f"  Scored in {time.time() - t0:.1f}s")

    r_pearson, p_pearson = stats.pearsonr(y_test, preds)
    r_spearman, p_spearman = stats.spearmanr(y_test, preds)

    print(f"\n{'='*60}")
    print(f"LegNet K562 lentiMPRA Test Set Performance (N={N})")
    print(f"{'='*60}")
    print(f"  Pearson:  r = {r_pearson:.4f} (p = {p_pearson:.2e})")
    print(f"  Spearman: r = {r_spearman:.4f} (p = {p_spearman:.2e})")
    print(f"  Pred mean={np.mean(preds):.4f}, std={np.std(preds):.4f}")
    print(f"  True mean={y_test.mean():.4f}, std={y_test.std():.4f}")
    print(f"{'='*60}")

    # Append to existing results file
    out_file = OUTDIR / "k562_test_set_performance.txt"
    with open(out_file, "a") as fout:
        fout.write(f"\nLegNet:     Pearson={r_pearson:.4f}, Spearman={r_spearman:.4f}\n")
    print(f"\nAppended to: {out_file}")


if __name__ == "__main__":
    main()
