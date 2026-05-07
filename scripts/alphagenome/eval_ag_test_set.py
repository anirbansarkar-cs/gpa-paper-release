#!/usr/bin/env python3
"""Evaluate AlphaGenome oracle on the K562 lentiMPRA test set.

Computes Pearson/Spearman correlation between AG predictions and ground-truth
MPRA activity to verify concordance with published benchmarks (~0.83).

Also evaluates LegNet on the same test set for direct comparison.
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
DATA_H5 = "model_zoo/lentimpra/lenti_MPRA_K562_data.h5"
AG_CKPT_STAGE1 = "${GPA_SHARED_ROOT}/alphagenome_encoder/mpra-K562-optimal/stage1"
AG_CKPT_STAGE2 = "${GPA_SHARED_ROOT}/alphagenome_encoder/mpra-K562-optimal/stage2"
LEFT_ADAPTER = "AGGACCGGATCAACT"
RIGHT_ADAPTER = "CATTGCGTGAACCGA"
BATCH_SIZE = 64
OUTDIR = Path("results/ag_vs_legnet_analysis")


def eval_ag(onehot_payload, y_true, ckpt_path, label):
    """Score with AlphaGenome and compute correlations."""
    from alphagenome_ft_mpra.oracle import load_oracle

    print(f"\nLoading AG oracle ({label}) from {ckpt_path}...")
    t0 = time.time()
    oracle = load_oracle(
        ckpt_path,
        left_adapter=LEFT_ADAPTER,
        right_adapter=RIGHT_ADAPTER,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print(f"  Scoring {len(onehot_payload)} sequences...")
    t0 = time.time()
    preds = oracle.predict(onehot_payload, mode="core", batch_size=BATCH_SIZE)
    preds = np.asarray(preds, dtype=np.float64).flatten()
    print(f"  Scored in {time.time() - t0:.1f}s")

    r_pearson, p_pearson = stats.pearsonr(y_true, preds)
    r_spearman, p_spearman = stats.spearmanr(y_true, preds)

    print(f"\n  AG {label} K562 test-set performance:")
    print(f"    Pearson:  r = {r_pearson:.4f} (p = {p_pearson:.2e})")
    print(f"    Spearman: r = {r_spearman:.4f} (p = {p_spearman:.2e})")
    print(f"    Pred mean={np.mean(preds):.4f}, std={np.std(preds):.4f}")
    print(f"    True mean={np.mean(y_true):.4f}, std={np.std(y_true):.4f}")

    return preds, r_pearson, r_spearman


def eval_legnet(onehot_full_230, y_true):
    """Score with LegNet and compute correlations."""
    try:
        from d3_evaluation_pipeline.mpralegnet import load_model
    except ImportError:
        print("\n  LegNet not available, skipping.")
        return None, None, None

    print("\nLoading LegNet K562 oracle...")
    t0 = time.time()
    model = load_model("K562")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # LegNet expects (N, 4, L) with A=0, G=1, C=2, T=3 channel order
    # Input is (N, L, 4) with A=0, C=1, G=2, T=3
    # Need: transpose + swap channels 1<->2
    onehot_t = onehot_full_230.transpose(0, 2, 1)  # (N, 4, L)
    # Swap C (ch1) and G (ch2) to get A,G,C,T
    onehot_legnet = onehot_t.copy()
    onehot_legnet[:, 1, :] = onehot_t[:, 2, :]  # G
    onehot_legnet[:, 2, :] = onehot_t[:, 1, :]  # C

    import torch
    print(f"  Scoring {len(onehot_legnet)} sequences...")
    t0 = time.time()

    preds_list = []
    with torch.no_grad():
        for i in range(0, len(onehot_legnet), BATCH_SIZE):
            batch = torch.tensor(
                onehot_legnet[i:i + BATCH_SIZE], dtype=torch.float32
            ).cuda()
            out = model(batch).cpu().numpy().flatten()
            preds_list.append(out)
    preds = np.concatenate(preds_list).astype(np.float64)
    print(f"  Scored in {time.time() - t0:.1f}s")

    r_pearson, p_pearson = stats.pearsonr(y_true, preds)
    r_spearman, p_spearman = stats.spearmanr(y_true, preds)

    print(f"\n  LegNet K562 test-set performance:")
    print(f"    Pearson:  r = {r_pearson:.4f} (p = {p_pearson:.2e})")
    print(f"    Spearman: r = {r_spearman:.4f} (p = {p_spearman:.2e})")
    print(f"    Pred mean={np.mean(preds):.4f}, std={np.std(preds):.4f}")

    return preds, r_pearson, r_spearman


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    print(f"Loading test data from {DATA_H5}...")
    with h5py.File(DATA_H5, "r") as f:
        onehot_test = f["onehot_test"][:]  # (39340, 230, 4)
        y_test = f["y_test"][:].flatten().astype(np.float64)  # (39340,)

    print(f"  Test set: {len(y_test)} sequences, shape={onehot_test.shape}")
    print(f"  y_test: mean={y_test.mean():.4f}, std={y_test.std():.4f}")

    # The test data is 230bp = 15(left) + 200(payload) + 15(right)
    # AG oracle expects payload only (200bp), then adds adapters in "core" mode
    onehot_payload = onehot_test[:, 15:215, :]  # Strip adapters -> (N, 200, 4)
    print(f"  Payload shape (stripped adapters): {onehot_payload.shape}")

    # Evaluate AG stage1
    ag1_preds, ag1_pearson, ag1_spearman = eval_ag(
        onehot_payload, y_test, AG_CKPT_STAGE1, "stage1"
    )

    # Evaluate AG stage2
    ag2_preds, ag2_pearson, ag2_spearman = eval_ag(
        onehot_payload, y_test, AG_CKPT_STAGE2, "stage2"
    )

    # Evaluate LegNet
    ln_preds, ln_pearson, ln_spearman = eval_legnet(onehot_test, y_test)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: K562 lentiMPRA Test Set Performance")
    print("=" * 70)
    print(f"  {'Oracle':<20} {'Pearson r':>10} {'Spearman r':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12}")
    print(f"  {'AG stage1':<20} {ag1_pearson:>10.4f} {ag1_spearman:>12.4f}")
    print(f"  {'AG stage2':<20} {ag2_pearson:>10.4f} {ag2_spearman:>12.4f}")
    if ln_pearson is not None:
        print(f"  {'LegNet':<20} {ln_pearson:>10.4f} {ln_spearman:>12.4f}")
    print(f"\n  Blog reference: AG Pearson r ~ 0.83 (K562)")
    print("=" * 70)

    # AG stage1 vs stage2 correlation
    r_s1s2, _ = stats.pearsonr(ag1_preds, ag2_preds)
    print(f"\n  AG stage1-stage2 correlation: Pearson r = {r_s1s2:.4f}")

    # AG vs LegNet correlation on test set
    if ln_preds is not None:
        r_ag_ln, _ = stats.pearsonr(ag1_preds, ln_preds)
        print(f"  AG stage1 vs LegNet correlation: Pearson r = {r_ag_ln:.4f}")

    # Save results
    out_file = OUTDIR / "k562_test_set_performance.txt"
    with open(out_file, "w") as fout:
        fout.write("K562 lentiMPRA Test Set Performance\n")
        fout.write(f"N = {len(y_test)} sequences\n\n")
        fout.write(f"AG stage1:  Pearson={ag1_pearson:.4f}, Spearman={ag1_spearman:.4f}\n")
        fout.write(f"AG stage2:  Pearson={ag2_pearson:.4f}, Spearman={ag2_spearman:.4f}\n")
        if ln_pearson is not None:
            fout.write(f"LegNet:     Pearson={ln_pearson:.4f}, Spearman={ln_spearman:.4f}\n")
        fout.write(f"\nAG s1-s2 corr: {r_s1s2:.4f}\n")
        if ln_preds is not None:
            fout.write(f"AG s1 vs LN:   {r_ag_ln:.4f}\n")
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
