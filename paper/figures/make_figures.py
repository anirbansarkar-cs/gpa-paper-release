#!/usr/bin/env python3
"""Generate paper figures for GPA paper. Run once; outputs PDFs to figures/.

Figures:
  fig1_pareto.pdf      — pw Pareto frontier (GPA vs RERD vs CSMC)
  fig2_gc.pdf          — GC distribution histogram (GPA vs CtrlDNA vs Real)
  fig3_K_saturation.pdf — Branch factor K target/composite/shannon trajectory
  fig4_beta_traj.pdf   — ESS-adaptive β annealing trajectory
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import pandas as pd

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "legend.fontsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# ─── Figure 1: Pareto frontier (one-knob pw curve, GPA-only) ────────────
def fig1_pareto():
    # From v17b one-knob curve (GPA-only, no DPS, GC<50%):
    pws =     [0.00, 0.10, 0.20, 0.35, 0.50, 0.70]
    H_mean =  [8.97, 9.42, 9.20, 8.17, 7.98, 8.21]
    H_std =   [0.50, 0.04, 0.11, 0.08, 0.10, 0.44]
    sp_mean = [1.69, 4.21, 6.07, 7.31, 7.35, 7.51]
    sp_std =  [0.15, 0.09, 1.08, 0.31, 0.02, 0.14]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.errorbar(sp_mean, H_mean, xerr=sp_std, yerr=H_std, fmt="o-",
                color="#1f77b4", lw=2, ms=7, capsize=3, label="GPA (no DPS), pw sweep", zorder=3)
    for i, pw in enumerate(pws):
        ax.annotate(f"pw={pw}", (sp_mean[i], H_mean[i]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9, color="#1f77b4")
    # GPA + DPS final recipes (4-seed means)
    ax.errorbar([3.06], [10.06], xerr=[0.55], yerr=[0.19], fmt="s",
                color="#d62728", ms=10, capsize=3, label="GPA+DPS Mean (pw=0)", zorder=4)
    ax.errorbar([7.61], [9.35], xerr=[0.25], yerr=[0.20], fmt="s",
                color="#2ca02c", ms=10, capsize=3, label="GPA+DPS Spec (pw=0.35)", zorder=4)
    # RERD baselines
    rerd_sp = [1.64, 1.87, -0.94]
    rerd_H = [8.37, 7.42, 5.78]
    ax.scatter(rerd_sp, rerd_H, marker="x", color="gray", s=80, lw=2.5, label="RERD (α∈{0.5,1,2})", zorder=2)
    # CSMC literature
    ax.axhline(10.09, ls="--", color="purple", lw=1.2, alpha=0.6, label="CSMC (literature)")
    # DRAKES literature
    ax.axhline(8.02, ls=":", color="orange", lw=1.2, alpha=0.6, label="DRAKES (literature)")

    ax.set_xlabel("Specificity (HepG2 − ½(K562 + SKNSH))")
    ax.set_ylabel(r"$H_{\mathrm{eval}}$ (HepG2 oracle activity)")
    ax.set_title("GPA Pareto frontier on Gosai HepG2 enhancer benchmark")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_pareto.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Wrote fig1_pareto.pdf")

# ─── Figure 2: GC distribution histogram ────────────────────────────────
def fig2_gc():
    # Pull GC distributions from real GPA + CtrlDNA + reference pools
    ROOT = Path("${GPA_REPO_ROOT}/results/ctrl_dna_comparison")
    gc_gpa, gc_ctrl, gc_real = [], [], []
    # GPA v5 HepG2 pools (top-128 by composite)
    for s in range(5):
        d = ROOT / f"gpa_v5_K8_hepg2_seed{s}"
        if not (d / "gpa_output.csv").exists(): continue
        df = pd.read_csv(d / "gpa_output.csv")
        # Compute composite for top-128 selection
        df['composite'] = (
            2*((df['hepg2']-(-2.7347))/(7.4500-(-2.7347)))
            - ((df['k562']-(-1.847))/(8.4833-(-1.847)))
            - ((df['sknsh']-(-2.7112))/(9.3032-(-2.7112)))
            + 1.0)
        top = df.nlargest(128, 'composite')
        gc_gpa.extend([(seq.count('G')+seq.count('C'))/len(seq) for seq in top['sequence']])
    # CtrlDNA HepG2 R200 (top-128)
    for s in range(5):
        f = ROOT / "mse_oracle_scores" / f"ctrldna_mse_r200_seed{s}.csv"
        if not f.exists(): continue
        df = pd.read_csv(f)
        gc_ctrl.extend([(seq.count('G')+seq.count('C'))/len(seq) for seq in df['sequence']])
    # Reference: Gosai natural sequences
    real_path = Path("${GPA_REPO_ROOT}/data/gosai_mpra/Table_S2_MPRA_dataset.txt")
    if real_path.exists():
        # Sample 5000 random sequences for histogram
        try:
            rdf = pd.read_csv(real_path, sep='\t', usecols=['sequence'], nrows=20000)
            rdf = rdf.sample(min(5000, len(rdf)), random_state=42)
            gc_real = [(s.count('G')+s.count('C'))/len(s) for s in rdf['sequence']]
        except Exception as e:
            print(f"  warn: real Gosai sample failed: {e}")

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    bins = np.linspace(0.30, 0.75, 36)
    if gc_real: ax.hist(gc_real, bins=bins, density=True, alpha=0.45, color="gray", label=f"Gosai natural (n={len(gc_real)})")
    if gc_gpa:  ax.hist(gc_gpa, bins=bins, density=True, alpha=0.65, color="#1f77b4", label=f"GPA v5 (n={len(gc_gpa)})")
    if gc_ctrl: ax.hist(gc_ctrl, bins=bins, density=True, alpha=0.55, color="#d62728", label=f"CtrlDNA R200 (n={len(gc_ctrl)})")
    ax.axvspan(0.45, 0.55, color="green", alpha=0.10, label="bio_filter [0.45, 0.55]")
    ax.set_xlabel("GC content")
    ax.set_ylabel("Density")
    ax.set_title("GC distribution: GPA preserves naturalness; CtrlDNA drifts")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(OUT / "fig2_gc.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Wrote fig2_gc.pdf")

# ─── Figure 3: Branch factor K saturation ───────────────────────────────
def fig3_K():
    # From gpa_branch_factor_theory.md (Step 5 K∈{1,2,4,8} ablation, 3 seeds, promoter)
    K_vals = [1, 2, 4, 8]
    k562_tgt = [5.26, 5.42, 5.57, 5.60]
    k562_comp = [4.82, 4.99, 5.13, 5.25]
    k562_shan = [1.18, 1.22, 1.15, 1.07]
    thp1_tgt = [2.94, 3.05, 3.23, 3.31]
    thp1_comp = [2.72, 2.85, 2.97, 3.10]
    thp1_shan = [1.39, 1.35, 1.45, 1.14]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    for ax, ylab, k562_y, thp1_y in [
        (axes[0], "Target activity", k562_tgt, thp1_tgt),
        (axes[1], "Composite", k562_comp, thp1_comp),
        (axes[2], "Shannon entropy", k562_shan, thp1_shan),
    ]:
        ax.plot(K_vals, k562_y, "o-", lw=2, ms=8, label="K562", color="#1f77b4")
        ax.plot(K_vals, thp1_y, "s-", lw=2, ms=8, label="THP1", color="#d62728")
        ax.set_xlabel("Branch factor K")
        ax.set_ylabel(ylab)
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_vals); ax.set_xticklabels([str(k) for k in K_vals])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[0].set_title("Target rises with K; saturates on K562 by K=4-8")
    axes[1].set_title("Composite rises monotonically with K")
    axes[2].set_title("Shannon decreases (K↔β duality)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_K_saturation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Wrote fig3_K_saturation.pdf")

# ─── Figure 4: ESS-adaptive β trajectory ────────────────────────────────
def fig4_beta_traj():
    """Pull β trajectory from a real run's snapshot_summary.json."""
    ROOT = Path("${GPA_REPO_ROOT}/results/ctrl_dna_comparison")
    snap_file = ROOT / "gpa_v5_K8_hepg2_seed0/snapshot_summary.json"
    if not snap_file.exists():
        print(f"  skip fig4: snapshot summary not found ({snap_file})")
        return
    snaps = json.load(open(snap_file))
    steps = [s["step"] for s in snaps]
    betas = [s["beta"] for s in snaps]
    oracle_means = [s.get("oracle_mean", np.nan) for s in snaps]
    diversity = [s.get("diversity", np.nan) for s in snaps]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.5))
    color1 = "#d62728"
    ax1.plot(steps, betas, "o-", color=color1, lw=2, ms=4, label=r"$\beta$ (annealing)")
    ax1.set_xlabel("SMC step")
    ax1.set_ylabel(r"$\beta$ (inverse temperature)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    color2 = "#1f77b4"
    ax2.plot(steps, oracle_means, "s-", color=color2, lw=1.5, ms=4, alpha=0.8, label="Oracle mean")
    ax2.set_ylabel("Population mean reward", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    # Combined legend
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="lower right", fontsize=9)
    fig.suptitle(r"ESS-adaptive $\beta$ schedule (GPA v5, HepG2 seed 0)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_beta_traj.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Wrote fig4_beta_traj.pdf")

# ─── Figure 5/6: distribution demo (paper Figure 1 right panel) ─────────
# Two variants: 2-panel (in-paper) and 3-panel (sidecar for user review).
# Source: K562 GPA pools + ISM/LEDIDI scored under the held-out AG (AlphaGenome
# JAX MPRA-optimal/stage1) cross-oracle. LN = K562 LegNet (training oracle,
# easy-to-hack); AG = held-out cross-oracle, used as the "naturalness" axis.

K562_GPA_DIR = Path("${GPA_REPO_ROOT}/results/k562_mdlm_gpa")
K562_ISM = Path("${GPA_REPO_ROOT}/results/ism_baseline")
K562_LED = Path("${GPA_REPO_ROOT}/results/ledidi_comparison/matched")


def _load_gpa_pool(run_name):
    """Return (ag, ln) arrays for a GPA pool h5."""
    fp = K562_GPA_DIR / run_name / "gpa_output.h5"
    if not fp.exists():
        return None, None
    with h5py.File(fp, "r") as h:
        ag = h["ag_k562_scores_jax_v2"][:] if "ag_k562_scores_jax_v2" in h else h["ag_k562_scores"][:]
        ln = h["oracle_preds"][:] if "oracle_preds" in h else None
    return ag, ln


def _load_ism_endpoint(uid, gen=115):
    """Return (ag, ln) single-point of ISM at a generation (default last)."""
    fp = K562_ISM / f"ism_trajectory_uid{uid}_ag.csv"
    df = pd.read_csv(fp)
    df = df[df["ism_generation"] == gen]
    if len(df) == 0:
        df = pd.read_csv(fp).iloc[[-1]]
    return float(df["k562_ag_jax"].iloc[0]), float(df["oracle_pred"].iloc[0])


def _load_ledidi_best(uid, target_l):
    """Return (ag, ln) of the argmax-LN LEDIDI run for one (target, l) config."""
    fp = K562_LED / f"uid{uid}" / f"ledidi_uid{uid}_{target_l}_ag.csv"
    df = pd.read_csv(fp)
    best = int(np.argmax(df["edited_legnet"].values))
    return float(df["k562_ag_jax"].iloc[best]), float(df["edited_legnet"].iloc[best])


def fig5_gpa_distribution_2panel():
    """In-paper figure (Sec. Method, Figure 1 right). Two compact subpanels:
       (A) 1D histogram over AG (held-out activity); prior pool vs GPA pool;
           ISM/LEDIDI argmax-LN points marked as dashed verticals.
       (B) 2D scatter (LN, AG); method clouds; demonstrates the
           high-LN+low-AG corner is reward-hacking, the high-LN+high-AG corner
           is what GPA reaches without leaving the manifold.
    """
    # Prior proxy = lightly-annealed GPA at cap10 (≈20 edits, very close to seed)
    ag_prior, ln_prior = _load_gpa_pool("run_v13_nodps_bf10_b1k_cap10_ism50178")
    # GPA-final = GPA + DPS, cap50, the "high-fitness pool" the paper headlines
    ag_gpa, ln_gpa = _load_gpa_pool("run_v13c_dps_kl_bf10_b2k_cap50_ism50178")
    # ISM and LEDIDI single-point endpoints (uid=50178 random seed)
    ag_ism, ln_ism = _load_ism_endpoint(50178, gen=115)
    ag_led, ln_led = _load_ledidi_best(50178, "t15_l0.05")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

    # Panel A: AG histogram
    a = axes[0]
    bins = np.linspace(-1.5, 4.5, 40)
    a.hist(ag_prior, bins=bins, density=True, color="#888888", alpha=0.55,
           label=r"prior $p_\theta$ (n=5000)")
    a.hist(ag_gpa,   bins=bins, density=True, color="#1f77b4", alpha=0.7,
           label=r"GPA pool $\pi_{\beta_*}$ (n=5000)")
    a.axvline(ag_ism, color="#ff7f0e", lw=2.0, ls="--",
              label=f"ISM argmax-LN  (AG={ag_ism:+.2f})")
    a.axvline(ag_led, color="#2ca02c", lw=2.0, ls="--",
              label=f"LEDIDI argmax-LN (AG={ag_led:+.2f})")
    a.set_xlabel("Held-out oracle reward (AlphaGenome K562)")
    a.set_ylabel("Density")
    a.set_title("(A) GPA shifts the pool toward high reward")
    a.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    a.grid(alpha=0.2, axis="y")

    # Panel B: LN x AG scatter
    b = axes[1]
    b.scatter(ln_prior, ag_prior, s=4, alpha=0.25, color="#888888",
              label=r"prior $p_\theta$", rasterized=True)
    b.scatter(ln_gpa, ag_gpa, s=4, alpha=0.35, color="#1f77b4",
              label=r"GPA pool", rasterized=True)
    b.scatter([ln_ism], [ag_ism], s=140, marker="o",
              edgecolor="#ff7f0e", facecolor="white", lw=2.5,
              label="ISM argmax-LN", zorder=5)
    b.scatter([ln_led], [ag_led], s=160, marker="X",
              color="#2ca02c", edgecolor="black", lw=1.0,
              label="LEDIDI argmax-LN", zorder=5)
    b.axhline(0, color="black", lw=0.6, ls=":", alpha=0.5)
    b.set_xlabel("Training oracle reward (LegNet K562) — exploitable")
    b.set_ylabel("Held-out reward (AlphaGenome) — naturalness proxy")
    b.set_title("(B) GPA: high reward AND high naturalness")
    b.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    b.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT / "fig_gpa_distribution_2panel.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  Wrote fig_gpa_distribution_2panel.pdf")


def fig6_gpa_distribution_3panel():
    """Sidecar figure: 3-panel evolution of the GPA population over annealing.
       Saved to figures/ for user review; NOT included in main.tex by default."""
    stages = [
        ("run_v13_nodps_bf10_b1k_cap10_ism50178",   r"$\beta\!\approx\!0$ (cap10, ~20 edits)"),
        ("run_v13_nodps_bf10_b1k_cap20_ism50178",   r"mid-anneal (cap20, ~40 edits)"),
        ("run_v13_nodps_bf10_b1k_uncapped_ism50178", r"$\beta=\beta_*$ (uncapped pool)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    bins = np.linspace(-1.5, 4.5, 40)
    for ax, (run, lab) in zip(axes, stages):
        ag, ln = _load_gpa_pool(run)
        if ag is None:
            ax.set_visible(False)
            continue
        ax.hist(ag, bins=bins, density=True, color="#1f77b4", alpha=0.75)
        ax.axvline(float(np.mean(ag)), color="#d62728", lw=1.6, ls="--",
                   label=f"mean = {ag.mean():+.2f}")
        ax.set_xlabel("AG K562 reward")
        ax.set_title(lab, fontsize=10)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("Density")
    fig.suptitle(r"GPA population evolves rightward as $\beta$ anneals", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gpa_distribution_3panel.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  Wrote fig_gpa_distribution_3panel.pdf (sidecar — not in main.tex)")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_pareto()
    fig2_gc()
    fig3_K()
    fig4_beta_traj()
    fig5_gpa_distribution_2panel()
    fig6_gpa_distribution_3panel()
    print(f"All figures in {OUT}")
