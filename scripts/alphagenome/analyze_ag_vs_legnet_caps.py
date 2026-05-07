#!/usr/bin/env python3
"""Analyze AG oracle agreement with LegNet on capped-edit GPA runs.

Hypothesis: edit-capped GPA runs stay closer to natural sequences, so AG
(AlphaGenome) should be a more reliable evaluator — and LegNet-AG correlation
should be higher for capped vs uncapped runs.

28 runs already have both oracle_k562 (LegNet) and oracle_k562_ag (AG) scores.
No new scoring needed.
"""

import re
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── Run registry ─────────────────────────────────────────────────────────────
# All 28 runs with AG scores, tagged with metadata.
# Format: (run_dir, is_capped, has_dps, cap_value, hc_value, init_type, label)

RESULTS_DIR = Path("results/rerd_comparison")

RUN_REGISTRY = [
    # --- Uncapped runs (7) ---
    ("run_v7_cascade_pw035_beta100", False, True, None, None, "bio", "v7 spec"),
    ("run_v9_pw0_smooth01_hard4555", False, True, None, None, "bio", "v9 mean"),
    ("run_v24a_kl_hc1", False, True, None, 1, "bio", "v24a KL hc1"),
    ("run_v24c_nodps_ref", False, False, None, None, "bio", "v24c noDPS ref"),
    ("run_v24c_nodps_s0_hc1", False, False, None, 1, "bio", "v24c noDPS s0 hc1"),
    ("run_v24c_nodps_s0_hc2", False, False, None, 2, "bio", "v24c noDPS s0 hc2"),
    ("run_v24c_nodps_s2_hc1", False, False, None, 1, "bio", "v24c noDPS s2 hc1"),
    # --- Capped + DPS runs (6) ---
    ("run_v24b_kl_cap10_nf05", True, True, 10, None, "bio", "v24b cap10 nf05"),
    ("run_v24b_kl_cap10_nf10", True, True, 10, None, "bio", "v24b cap10 nf10"),
    ("run_v24b_kl_cap10_nf10_hc2", True, True, 10, 2, "bio", "v24b cap10 nf10 hc2"),
    ("run_v24b_kl_cap10_nf15", True, True, 10, None, "bio", "v24b cap10 nf15"),
    ("run_v24b_kl_cap15_nf10", True, True, 15, None, "bio", "v24b cap15 nf10"),
    ("run_v25b_kl_hc1_cap07_nf05", True, True, 7, 1, "bio", "v25b cap07 nf05"),
    # --- Capped + DPS runs continued (4) ---
    ("run_v25b_kl_hc1_cap10_nf05", True, True, 10, 1, "bio", "v25b cap10 nf05"),
    ("run_v25b_kl_hc1_cap10_nf10", True, True, 10, 1, "bio", "v25b cap10 nf10"),
    ("run_v25b_kl_hc2_cap10_nf05", True, True, 10, 2, "bio", "v25b hc2 cap10 nf05"),
    # --- Capped + noDPS runs (8) ---
    ("run_v25a_nodps_hc1_cap10", True, False, 10, 1, "bio", "v25a noDPS hc1 cap10"),
    ("run_v25a_nodps_hc2_cap10", True, False, 10, 2, "bio", "v25a noDPS hc2 cap10"),
    ("run_v25a_nodps_hc2_cap15", True, False, 15, 2, "bio", "v25a noDPS hc2 cap15"),
    ("run_v25c_nodps_hc2_cap07", True, False, 7, 2, "bio", "v25c noDPS hc2 cap07"),
    ("run_v25c_nodps_hc3_cap10", True, False, 10, 3, "bio", "v25c noDPS hc3 cap10"),
    ("run_v25c_nodps_hc4_cap10", True, False, 10, 4, "bio", "v25c noDPS hc4 cap10"),
    ("run_v25d_rand_nodps_hc1_cap10", True, False, 10, 1, "rand", "v25d rand noDPS hc1 cap10"),
    ("run_v25d_rand_nodps_hc1_cap10_r2", True, False, 10, 1, "rand", "v25d rand noDPS hc1 cap10 r2"),
    # --- Capped + random init + DPS (3) ---
    ("run_v25d_rand_kl_hc1_cap10_nf05", True, True, 10, 1, "rand", "v25d rand cap10 nf05"),
    ("run_v25d_rand_kl_hc1_cap10_nf05_r2", True, True, 10, 1, "rand", "v25d rand cap10 nf05 r2"),
    ("run_v25d_rand_kl_hc1_cap10_nf10", True, True, 10, 1, "rand", "v25d rand cap10 nf10"),
    # --- Capped + random init + noDPS (1) ---
    ("run_v25d_rand_nodps_hc2_cap10", True, False, 10, 2, "rand", "v25d rand noDPS hc2 cap10"),
]

OUTDIR = Path("results/ag_vs_legnet_analysis")


def load_run(run_dir):
    """Load LegNet and AG scores from a run's H5 file."""
    h5_path = RESULTS_DIR / run_dir / "gpa_output.h5"
    with h5py.File(h5_path, "r") as f:
        data = {
            "legnet": f["oracle_k562"][:].astype(np.float64),
            "ag": f["oracle_k562_ag"][:].astype(np.float64),
            "gc": f["gc_fractions"][:].astype(np.float64),
        }
        if "oracle_k562_eval" in f:
            data["eval"] = f["oracle_k562_eval"][:].astype(np.float64)
        if "indices" in f:
            data["indices"] = f["indices"][:]
        if "arr_0" in f:
            data["seeds"] = f["arr_0"][:]
    return data


def hamming_distance(seeds, indices):
    """Mean per-sequence Hamming distance. Seeds may be one-hot (N,4,L) or indices (N,L)."""
    if seeds.ndim == 3:
        # One-hot (N, 4, L) -> indices (N, L)
        seeds = np.argmax(seeds, axis=1)
    return np.sum(seeds != indices, axis=1).astype(np.float64)


def compute_run_stats(data):
    """Compute summary statistics for a single run."""
    legnet = data["legnet"]
    ag = data["ag"]

    r_pearson, p_pearson = stats.pearsonr(legnet, ag)
    r_spearman, p_spearman = stats.spearmanr(legnet, ag)

    result = {
        "n": len(legnet),
        "legnet_mean": np.mean(legnet),
        "legnet_std": np.std(legnet),
        "legnet_max": np.max(legnet),
        "ag_mean": np.mean(ag),
        "ag_std": np.std(ag),
        "ag_max": np.max(ag),
        "gc_mean": np.mean(data["gc"]),
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "spearman_r": r_spearman,
        "spearman_p": p_spearman,
    }

    # Note: arr_0 == indices in all runs (final state stored in both), so
    # edit distance from seeds is not recoverable from the H5 files.

    return result


def print_summary_table(all_stats):
    """Print formatted per-run summary table."""
    print("\n" + "=" * 140)
    print("PER-RUN SUMMARY TABLE")
    print("=" * 140)
    header = (
        f"{'Run':<40} {'Cap':>3} {'DPS':>3} {'N':>5} "
        f"{'LegNet μ':>9} {'LegNet σ':>9} {'LegNet max':>10} "
        f"{'AG μ':>7} {'AG σ':>6} {'AG max':>7} "
        f"{'Pearson':>8} {'Spearman':>9} {'GC μ':>6}"
    )
    print(header)
    print("-" * 135)

    for run_dir, is_capped, has_dps, cap_val, hc, init, label in RUN_REGISTRY:
        s = all_stats[run_dir]
        cap_str = str(cap_val) if cap_val else "-"
        dps_str = "Y" if has_dps else "N"
        print(
            f"{label:<40} {cap_str:>3} {dps_str:>3} {s['n']:>5} "
            f"{s['legnet_mean']:>9.3f} {s['legnet_std']:>9.3f} {s['legnet_max']:>10.3f} "
            f"{s['ag_mean']:>7.3f} {s['ag_std']:>7.3f} {s['ag_max']:>7.3f} "
            f"{s['pearson_r']:>8.3f} {s['spearman_r']:>9.3f} {s['gc_mean']:>6.1%}"
        )


def plot_legnet_vs_ag_scatter(all_stats, all_data):
    """Cross-run scatter: LegNet mean vs AG mean, colored by capped/uncapped."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel 1: Run-level means ---
    ax = axes[0]
    for run_dir, is_capped, has_dps, cap_val, hc, init, label in RUN_REGISTRY:
        s = all_stats[run_dir]
        color = "tab:blue" if is_capped else "tab:red"
        marker = "o" if has_dps else "s"
        ax.scatter(
            s["legnet_mean"], s["ag_mean"],
            c=color, marker=marker, s=60, alpha=0.8,
            edgecolors="k", linewidths=0.5,
        )
    # Legend
    ax.scatter([], [], c="tab:blue", marker="o", label="Capped + DPS")
    ax.scatter([], [], c="tab:blue", marker="s", label="Capped + noDPS")
    ax.scatter([], [], c="tab:red", marker="o", label="Uncapped + DPS")
    ax.scatter([], [], c="tab:red", marker="s", label="Uncapped + noDPS")
    ax.set_xlabel("LegNet mean (K562)")
    ax.set_ylabel("AG mean (K562)")
    ax.set_title("Run-level: LegNet vs AG means")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Correlation across runs
    legnet_means = [all_stats[r[0]]["legnet_mean"] for r in RUN_REGISTRY]
    ag_means = [all_stats[r[0]]["ag_mean"] for r in RUN_REGISTRY]
    r_cross, p_cross = stats.pearsonr(legnet_means, ag_means)
    ax.text(
        0.05, 0.95, f"r={r_cross:.3f} (p={p_cross:.2e})",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # --- Panel 2: Within-run correlations ---
    ax = axes[1]
    capped_corrs = []
    uncapped_corrs = []
    for run_dir, is_capped, has_dps, cap_val, hc, init, label in RUN_REGISTRY:
        s = all_stats[run_dir]
        corr = s["spearman_r"]
        color = "tab:blue" if is_capped else "tab:red"
        if is_capped:
            capped_corrs.append(corr)
        else:
            uncapped_corrs.append(corr)
        ax.scatter(
            1 if is_capped else 0, corr,
            c=color, s=60, alpha=0.7, edgecolors="k", linewidths=0.5,
        )
    # Mean lines
    ax.axhline(np.mean(capped_corrs), color="tab:blue", ls="--", alpha=0.7,
               label=f"Capped mean: {np.mean(capped_corrs):.3f}")
    ax.axhline(np.mean(uncapped_corrs), color="tab:red", ls="--", alpha=0.7,
               label=f"Uncapped mean: {np.mean(uncapped_corrs):.3f}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Uncapped", "Capped"])
    ax.set_ylabel("Within-run Spearman(LegNet, AG)")
    ax.set_title("LegNet-AG agreement: Capped vs Uncapped")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Mann-Whitney test
    u_stat, u_p = stats.mannwhitneyu(capped_corrs, uncapped_corrs, alternative="greater")
    ax.text(
        0.05, 0.05, f"MW U p={u_p:.3f} (capped > uncapped)",
        transform=ax.transAxes, fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # --- Panel 3: Cap value effect ---
    ax = axes[2]
    cap_groups = {7: [], 10: [], 15: []}
    for run_dir, is_capped, has_dps, cap_val, hc, init, label in RUN_REGISTRY:
        if is_capped and cap_val in cap_groups:
            cap_groups[cap_val].append(all_stats[run_dir]["spearman_r"])
    positions = []
    data_boxes = []
    labels_box = []
    # Add uncapped
    positions.append(0)
    data_boxes.append(uncapped_corrs)
    labels_box.append("uncap")
    for i, cap in enumerate(sorted(cap_groups.keys())):
        if cap_groups[cap]:
            positions.append(i + 1)
            data_boxes.append(cap_groups[cap])
            labels_box.append(f"cap={cap}")
    bp = ax.boxplot(data_boxes, positions=positions, widths=0.5, patch_artist=True)
    colors_box = ["tab:red"] + ["tab:blue"] * (len(positions) - 1)
    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_box)
    ax.set_ylabel("Within-run Spearman(LegNet, AG)")
    ax.set_title("Effect of cap value on LegNet-AG agreement")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(OUTDIR / "ag_vs_legnet_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTDIR / 'ag_vs_legnet_overview.png'}")


def plot_dps_effect(all_stats):
    """Compare AG scores and correlations: DPS vs noDPS."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: AG mean by DPS status (capped only)
    ax = axes[0]
    dps_ag = []
    nodps_ag = []
    for run_dir, is_capped, has_dps, cap_val, hc, init, label in RUN_REGISTRY:
        if not is_capped:
            continue
        s = all_stats[run_dir]
        if has_dps:
            dps_ag.append(s["ag_mean"])
        else:
            nodps_ag.append(s["ag_mean"])
    bp = ax.boxplot([nodps_ag, dps_ag], labels=["noDPS", "DPS"], patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("tab:green")
    bp["boxes"][0].set_alpha(0.4)
    bp["boxes"][1].set_facecolor("tab:orange")
    bp["boxes"][1].set_alpha(0.4)
    ax.set_ylabel("AG mean (K562)")
    ax.set_title("AG scores: DPS vs noDPS (capped runs only)")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Spearman by DPS status (capped only)
    ax = axes[1]
    dps_corr = []
    nodps_corr = []
    for run_dir, is_capped, has_dps, cap_val, hc, init, label in RUN_REGISTRY:
        if not is_capped:
            continue
        s = all_stats[run_dir]
        if has_dps:
            dps_corr.append(s["spearman_r"])
        else:
            nodps_corr.append(s["spearman_r"])
    bp = ax.boxplot([nodps_corr, dps_corr], labels=["noDPS", "DPS"], patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("tab:green")
    bp["boxes"][0].set_alpha(0.4)
    bp["boxes"][1].set_facecolor("tab:orange")
    bp["boxes"][1].set_alpha(0.4)
    ax.set_ylabel("Within-run Spearman(LegNet, AG)")
    ax.set_title("LegNet-AG agreement: DPS vs noDPS (capped only)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(OUTDIR / "ag_dps_effect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTDIR / 'ag_dps_effect.png'}")


def plot_disagreement(all_stats):
    """Identify runs with high LegNet but low AG (oracle disagreement)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for run_dir, is_capped, has_dps, cap_val, hc, init, label in RUN_REGISTRY:
        s = all_stats[run_dir]
        color = "tab:blue" if is_capped else "tab:red"
        marker = "o" if has_dps else "s"
        ax.scatter(
            s["legnet_mean"], s["ag_mean"],
            c=color, marker=marker, s=80, alpha=0.8,
            edgecolors="k", linewidths=0.5,
        )
        # Label outliers (high LegNet, low AG)
        if s["legnet_mean"] > 8 and s["ag_mean"] < 0.5:
            ax.annotate(
                label, (s["legnet_mean"], s["ag_mean"]),
                textcoords="offset points", xytext=(5, 5), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.5),
            )
        # Label best AG
        if s["ag_mean"] > 1.5:
            ax.annotate(
                label, (s["legnet_mean"], s["ag_mean"]),
                textcoords="offset points", xytext=(5, 5), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.5),
            )

    ax.scatter([], [], c="tab:blue", marker="o", label="Capped + DPS")
    ax.scatter([], [], c="tab:blue", marker="s", label="Capped + noDPS")
    ax.scatter([], [], c="tab:red", marker="o", label="Uncapped + DPS")
    ax.scatter([], [], c="tab:red", marker="s", label="Uncapped + noDPS")

    # Reference lines
    ax.axhline(-0.43, color="gray", ls=":", alpha=0.5, label="Random baseline (AG≈-0.43)")
    ax.axhline(0, color="gray", ls="-", alpha=0.3)

    ax.set_xlabel("LegNet mean (K562)")
    ax.set_ylabel("AG mean (K562)")
    ax.set_title("Oracle Disagreement Map: LegNet vs AG")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTDIR / "ag_disagreement_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTDIR / 'ag_disagreement_map.png'}")


def print_key_findings(all_stats):
    """Print summary answering the key questions."""
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # 1. Capped vs uncapped AG-LegNet correlation
    capped_corrs = []
    uncapped_corrs = []
    capped_ag = []
    uncapped_ag = []
    for run_dir, is_capped, has_dps, cap_val, hc, init, label in RUN_REGISTRY:
        s = all_stats[run_dir]
        if is_capped:
            capped_corrs.append(s["spearman_r"])
            capped_ag.append(s["ag_mean"])
        else:
            uncapped_corrs.append(s["spearman_r"])
            uncapped_ag.append(s["ag_mean"])

    print(f"\n1. CAPPED vs UNCAPPED LegNet-AG agreement:")
    print(f"   Capped   (n={len(capped_corrs)}): Spearman mean={np.mean(capped_corrs):.3f} ± {np.std(capped_corrs):.3f}")
    print(f"   Uncapped (n={len(uncapped_corrs)}): Spearman mean={np.mean(uncapped_corrs):.3f} ± {np.std(uncapped_corrs):.3f}")
    u_stat, u_p = stats.mannwhitneyu(capped_corrs, uncapped_corrs, alternative="greater")
    print(f"   Mann-Whitney U test (capped > uncapped): p={u_p:.4f}")
    verdict = "YES" if u_p < 0.05 else "NO (not significant)"
    print(f"   → Does capping improve LegNet-AG agreement? {verdict}")

    print(f"\n2. CAPPED vs UNCAPPED AG scores:")
    print(f"   Capped   AG mean: {np.mean(capped_ag):.3f} ± {np.std(capped_ag):.3f}")
    print(f"   Uncapped AG mean: {np.mean(uncapped_ag):.3f} ± {np.std(uncapped_ag):.3f}")

    # 3. DPS effect on AG (capped only)
    dps_ag = []
    nodps_ag = []
    dps_corr = []
    nodps_corr = []
    for run_dir, is_capped, has_dps, cap_val, hc, init, label in RUN_REGISTRY:
        if not is_capped:
            continue
        s = all_stats[run_dir]
        if has_dps:
            dps_ag.append(s["ag_mean"])
            dps_corr.append(s["spearman_r"])
        else:
            nodps_ag.append(s["ag_mean"])
            nodps_corr.append(s["spearman_r"])

    print(f"\n3. DPS effect (capped runs only):")
    print(f"   DPS   AG mean: {np.mean(dps_ag):.3f} ± {np.std(dps_ag):.3f}, Spearman: {np.mean(dps_corr):.3f}")
    print(f"   noDPS AG mean: {np.mean(nodps_ag):.3f} ± {np.std(nodps_ag):.3f}, Spearman: {np.mean(nodps_corr):.3f}")

    # 4. Cap value effect
    print(f"\n4. Cap value effect on AG-LegNet Spearman:")
    for cap in [7, 10, 15]:
        corrs = [
            all_stats[r[0]]["spearman_r"]
            for r in RUN_REGISTRY
            if r[1] and r[3] == cap
        ]
        if corrs:
            print(f"   cap={cap:2d}: Spearman mean={np.mean(corrs):.3f} ± {np.std(corrs):.3f} (n={len(corrs)})")

    # 5. Disagreement: high LegNet, low AG
    print(f"\n5. ORACLE DISAGREEMENT (high LegNet, low AG):")
    print(f"   {'Run':<40} {'LegNet μ':>9} {'AG μ':>7} {'Gap':>6}")
    for run_dir, is_capped, has_dps, cap_val, hc, init, label in RUN_REGISTRY:
        s = all_stats[run_dir]
        # Flag runs where LegNet is high but AG is relatively low
        if s["legnet_mean"] > 7 and s["ag_mean"] < 0.5:
            gap = s["legnet_mean"] - s["ag_mean"]
            print(f"   {label:<40} {s['legnet_mean']:>9.3f} {s['ag_mean']:>7.3f} {gap:>6.2f}")

    # 6. Cross-run correlation
    legnet_means = [all_stats[r[0]]["legnet_mean"] for r in RUN_REGISTRY]
    ag_means = [all_stats[r[0]]["ag_mean"] for r in RUN_REGISTRY]
    r_cross, p_cross = stats.pearsonr(legnet_means, ag_means)
    r_sp, p_sp = stats.spearmanr(legnet_means, ag_means)
    print(f"\n6. CROSS-RUN correlation (run-level means, n=28):")
    print(f"   Pearson:  r={r_cross:.3f}, p={p_cross:.2e}")
    print(f"   Spearman: r={r_sp:.3f}, p={p_sp:.2e}")

    print("\n" + "=" * 80)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Loading 28 runs...")
    all_data = {}
    all_stats = {}
    for run_dir, *_ in RUN_REGISTRY:
        h5_path = RESULTS_DIR / run_dir / "gpa_output.h5"
        if not h5_path.exists():
            print(f"  WARNING: {h5_path} not found, skipping")
            continue
        data = load_run(run_dir)
        all_data[run_dir] = data
        all_stats[run_dir] = compute_run_stats(data)
        print(f"  {run_dir}: n={len(data['legnet'])}, LegNet μ={np.mean(data['legnet']):.2f}, AG μ={np.mean(data['ag']):.3f}")

    print(f"\nLoaded {len(all_stats)} / {len(RUN_REGISTRY)} runs")

    # Summary table
    print_summary_table(all_stats)

    # Plots
    print("\nGenerating plots...")
    plot_legnet_vs_ag_scatter(all_stats, all_data)
    plot_dps_effect(all_stats)
    plot_disagreement(all_stats)

    # Key findings
    print_key_findings(all_stats)

    print(f"\nAll outputs saved to: {OUTDIR}/")


if __name__ == "__main__":
    main()
