#!/usr/bin/env python3
"""Final 5-seed GPA-vs-CtrlDNA promoter analysis.

Produces the publishable table: 3 cells × 2 GPA recipes × 2 selections
vs CtrlDNA R200. Reports target / composite / shannon / motif_corr with
mean±σ across up-to-5 seeds. Flags the universal-recipe winner if one exists.

Recipes probed (all theoretically grounded — see gpa_branch_factor_theory.md):
  * argmax_K=4  (effective-β reparameterization, K=4 tournament)
  * argmax_K=8  (effective-β reparameterization, K=8 tournament — canonical)
  * POP=20k argmax K=8 (K562 only, variance reduction)
  * nodps_recA argmax K=8 (K562 only, alt proposal)
  * JURKAT bio_filter=True K=8 (hedge)

Reads from gpa_output_pool.h5 per user rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROM = Path("${GPA_REPO_ROOT}/scripts/ctrl_dna_comparison/promoter")
sys.path.insert(0, str(_PROM))

from selection_sweep import (  # noqa: E402
    apply_selection,
    load_archive,
    load_best,
    metrics_from_subset,
)
from evaluate_promoter_pools import build_reference_kmer_profiles  # noqa: E402

RESULTS_DIR = Path("${GPA_REPO_ROOT}/results/ctrl_dna_comparison/promoter")

# (label, suffix, cell-list, seed-list, dps_tag)
CONFIGS = [
    ("argmax_K4",            "nobio_tuned_pop10k_K4", ["K562", "THP1"],        [0, 1, 2, 3, 4], "dps"),
    ("argmax_K8",            "nobio_tuned_pop10k_K8", ["K562", "THP1"],        [0, 1, 2, 3, 4], "dps"),
    ("argmax_K8_JURKAT",     "nobio_tuned_pop10k_K8", ["JURKAT"],              [0, 1, 2, 3, 4], "dps"),
    ("argmax_K8_biofilt_JK", "biofilt_pop10k_K8",     ["JURKAT"],              [0, 1, 2, 3, 4], "dps"),
    ("argmax_K8_POP20k",     "nobio_tuned_pop20k_K8", ["K562"],                [0, 1, 2],       "dps"),
    ("argmax_K8_nodpsA",     "nodpsA_K8",             ["K562", "THP1"],        [0, 1, 2, 3, 4], "nodps"),
    ("argmax_K8_nodpsA_JK",  "nodpsA_K8",             ["JURKAT"],              [0, 1, 2, 3, 4], "nodps"),
    # K ablation under universal recipe (noDPS + no_bio + pw=0.35 + eta=3000)
    ("univ_K1",              "univ_K1",               ["JURKAT", "K562", "THP1"], [0, 1, 2],   "nodps"),
    ("univ_K4",              "univ_K4",               ["JURKAT", "K562", "THP1"], [0, 1, 2],   "nodps"),
    # Shannon rescue arms (noDPS + K=8 universal base)
    ("univ_K8_slowanneal",   "univ_K8_slowanneal",    ["JURKAT", "K562", "THP1"], [0, 1, 2],   "nodps"),
    ("univ_K8_substeps2",    "univ_K8_substeps2",     ["JURKAT", "K562", "THP1"], [0, 1, 2],   "nodps"),
    ("univ_K8_substeps3",    "univ_K8_substeps3",     ["JURKAT", "K562", "THP1"], [0, 1, 2],   "nodps"),
    ("univ_K8_substeps4",    "univ_K8_substeps4",     ["JURKAT", "K562", "THP1"], [0, 1, 2],   "nodps"),
    ("univ_K8_beta50",       "univ_K8_beta50",        ["JURKAT", "K562", "THP1"], [0, 1, 2],   "nodps"),
    # L7 K-annealing: K(t) schedules on universal recipe — Shannon rescue hypothesis
    ("kanneal_late_80",      "univ_K8_kanneal_late_8_1_80",   ["JURKAT", "K562", "THP1"], [0, 1, 2], "nodps"),
    ("kanneal_late_60",      "univ_K8_kanneal_late_8_1_60",   ["JURKAT", "K562", "THP1"], [0, 1, 2], "nodps"),
    ("kanneal_linear",       "univ_K8_kanneal_linear_8_1",    ["JURKAT", "K562", "THP1"], [0, 1, 2], "nodps"),
    ("kanneal_step",         "univ_K8_kanneal_step_8_4_2_1",  ["JURKAT", "K562", "THP1"], [0, 1, 2], "nodps"),
    # Existing 5-seed JURKAT winner (for reference)
    ("dps_recB_JURKAT",      "recB_arch",             ["JURKAT"],              [0, 1, 2, 3, 4], "dps"),
    # L8 Phase 1 — reward regularization (diversity_lambda) on nodpsA_K8
    ("nodpsA_K8_div0p3",     "nodpsA_K8_div0p3",      ["JURKAT", "K562", "THP1"], [0, 1, 2], "nodps"),
    ("nodpsA_K8_div1p0",     "nodpsA_K8_div1p0",      ["JURKAT", "K562", "THP1"], [0, 1, 2], "nodps"),
    ("nodpsA_K8_div3p0",     "nodpsA_K8_div3p0",      ["JURKAT", "K562", "THP1"], [0, 1, 2], "nodps"),
    ("nodpsA_K8_div10p0",    "nodpsA_K8_div10p0",     ["JURKAT", "K562", "THP1"], [0, 1, 2], "nodps"),
    # L8 Phase 2a — late-K (8→1 @ 80%) × MAX_BETA on nodpsA_K8
    ("nodpsA_K8_latek80_b50", "nodpsA_K8_latek80_b50", ["JURKAT", "K562", "THP1"], [0, 1, 2], "nodps"),
    ("nodpsA_K8_latek80_b25", "nodpsA_K8_latek80_b25", ["JURKAT", "K562", "THP1"], [0, 1, 2], "nodps"),
    ("nodpsA_K8_latek80_b10", "nodpsA_K8_latek80_b10", ["JURKAT", "K562", "THP1"], [0, 1, 2], "nodps"),
]

SELECTIONS = [
    "top128_by_target",
    "top128_unique_archive_by_composite",
    "top128_greedy_hamming_archive",    # explicit diversity selection (heuristic)
    "top128_map_dpp_archive",           # MAP-DPP (Chen et al. 2018, principled)
]

CTRL = {
    "JURKAT": dict(target=5.68, composite=4.04, shannon=1.64, motif_corr=0.78),
    "K562":   dict(target=6.02, composite=5.01, shannon=1.69, motif_corr=0.80),
    "THP1":   dict(target=4.16, composite=2.16, shannon=1.79, motif_corr=0.70),
}


def main():
    print("Building reference k-mer profiles...")
    data_csv = str(_PROM / "data" / "finetuning_data.csv")
    ref_kmer = build_reference_kmer_profiles(data_csv, top_k=1000)

    rows = []
    missing = []
    for label, suffix, cells, seeds, dps_tag in CONFIGS:
        for cell in cells:
            for seed in seeds:
                name = f"gpa_full_{cell}_{dps_tag}_seed{seed}_{suffix}"
                pool_dir = RESULTS_DIR / name
                if not pool_dir.is_dir():
                    missing.append(name)
                    continue
                best = load_best(pool_dir)
                if best is None:
                    missing.append(f"{name} (no best_eval.h5)")
                    continue
                archive = load_archive(pool_dir)
                for sel in SELECTIONS:
                    try:
                        seqs, sc = apply_selection(sel, best, archive, cell)
                        m = metrics_from_subset(seqs, sc, cell, ref_kmer)
                    except Exception as ex:
                        print(f"  [err] {name} {sel}: {ex}")
                        continue
                    rows.append(dict(
                        recipe=label, cell=cell, seed=seed, selection=sel,
                        N=m["N"], target=m[cell.lower()],
                        composite=m["composite"], shannon=m["shannon"],
                        motif_corr=m["motif_corr"],
                    ))
    if missing:
        print(f"\n{len(missing)} missing pool dirs (skipped — still running):")
        for n in missing:
            print(f"  {n}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data yet.")
        return

    agg = (df.groupby(["recipe", "cell", "selection"])
             [["target", "composite", "shannon", "motif_corr"]]
             .agg(["mean", "std", "count"])
             .reset_index())
    agg.columns = [f"{a}_{b}" if b else a for a, b in agg.columns]
    out_csv = RESULTS_DIR / "FINAL_gpa_vs_ctrldna_5seed.csv"
    agg.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  rows={len(agg)}")

    # Print per-cell × selection
    for sel in SELECTIONS:
        print(f"\n{'='*90}")
        print(f"SELECTION: {sel}")
        print('='*90)
        for cell in ["JURKAT", "K562", "THP1"]:
            sub = agg[(agg["cell"] == cell) & (agg["selection"] == sel)]
            if sub.empty:
                continue
            ref = CTRL[cell]
            print(f"\n{cell}  (CtrlDNA: tgt={ref['target']:.2f}  "
                  f"comp={ref['composite']:.2f}  H={ref['shannon']:.2f}  "
                  f"motif={ref['motif_corr']:.2f})")
            print(f"  {'recipe':<26s} {'N':>3s} {'target':>13s} "
                  f"{'composite':>13s} {'shannon':>13s} {'motif':>13s}  "
                  f"{'Δtgt':>7s} {'Δcomp':>7s} {'ΔH':>7s} {'Δmot':>7s}  win")
            order = {c[0]: i for i, c in enumerate(CONFIGS)}
            sub = sub.sort_values("recipe",
                                  key=lambda s: s.map(lambda v: order.get(v, 99)))
            for _, r in sub.iterrows():
                n = int(r["target_count"])
                t = f"{r['target_mean']:.2f}±{r['target_std']:.2f}"
                c = f"{r['composite_mean']:.2f}±{r['composite_std']:.2f}"
                h = f"{r['shannon_mean']:.2f}±{r['shannon_std']:.2f}"
                mc = f"{r['motif_corr_mean']:.2f}±{r['motif_corr_std']:.2f}"
                dt = r["target_mean"] - ref["target"]
                dc = r["composite_mean"] - ref["composite"]
                dh = r["shannon_mean"] - ref["shannon"]
                dm = r["motif_corr_mean"] - ref["motif_corr"]
                win = "YES" if (dt > 0 and dc > 0 and dh > 0 and dm > 0) else (
                    "partial" if (dc > 0 and dm > 0) else "no")
                print(f"  {r['recipe']:<26s} {n:>3d} {t:>13s} {c:>13s} "
                      f"{h:>13s} {mc:>13s}  {dt:>+7.2f} {dc:>+7.2f} "
                      f"{dh:>+7.2f} {dm:>+7.2f}  {win}")

    # Summary: identify universal recipe
    print(f"\n{'='*90}")
    print("UNIVERSAL RECIPE CHECK (top128_unique_archive_by_composite, n>=3)")
    print('='*90)
    best_sel = "top128_unique_archive_by_composite"
    univ = agg[(agg["selection"] == best_sel) & (agg["target_count"] >= 3)]
    for recipe_label in ["argmax_K8", "argmax_K8_JURKAT", "argmax_K8_biofilt_JK",
                         "argmax_K8_nodpsA"]:
        sub = univ[univ["recipe"] == recipe_label]
        if len(sub) == 0:
            continue
        print(f"\n{recipe_label}:")
        for _, r in sub.iterrows():
            c = r["cell"]
            ref = CTRL[c]
            dc = r["composite_mean"] - ref["composite"]
            dm = r["motif_corr_mean"] - ref["motif_corr"]
            status = "WIN" if dc > 0 and dm > 0 else "LOSS" if dc < 0 else "partial"
            print(f"  {c}: Δcomp={dc:+.2f}  Δmotif={dm:+.2f}  [{status}]")


if __name__ == "__main__":
    main()
