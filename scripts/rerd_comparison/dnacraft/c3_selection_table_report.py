#!/usr/bin/env python3
"""Build paper-format table comparing GPA across 5 selection rules:
by_mingap, by_composite, by_motif, by_kmer3, by_diversity.

Source = pool. Recipes:
  - HepG2 → run_dps_pw030_b25_gct060_nobio
  - K562  → run_dps_pw030_b25_gct060_nobio_k562
  - SKNSH → run_dps_pw030_b25_gct060_nobio_sknsh
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("${GPA_REPO_ROOT}")
DCRAFT_DIR = PROJECT / "results/rerd_comparison/dnacraft"

# Canonical paper baselines (means-only view of arXiv 2604.20488 Table 2)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from paper_baselines import PAPER_BASELINES  # noqa: E402

METRIC_SPECS = [
    ("MinGap Eval ↑",  "mingap_eval_mean",     "mingap"),
    ("Motif Spr ↑",    "motif_corr_spearman",  "motif"),
    ("3-mer Pearson ↑","kmer3_corr_pearson",   "kmer3"),
    ("Diversity ↑",    "diversity_bits",       "div"),
]
CELLS = ["HepG2", "K562", "SK-N-SH"]
CELL_INTERNAL = {"HepG2": "HepG2", "K562": "K562", "SK-N-SH": "SKNSH"}
SELECTIONS = ["by_mingap", "by_composite", "by_motif", "by_kmer3", "by_diversity"]
SEL_LABEL = {
    "by_mingap": "by MinGap",
    "by_composite": "by composite",
    "by_motif": "by Motif",
    "by_kmer3": "by 3-mer",
    "by_diversity": "by Diversity",
}

C3_RECIPE = {
    "HepG2": "run_dps_pw030_b25_gct060_nobio",
    "K562": "run_dps_pw030_b25_gct060_nobio_k562",
    "SK-N-SH": "run_dps_pw030_b25_gct060_nobio_sknsh",
}


def aggregate(source: str) -> pd.DataFrame:
    df = pd.read_csv(DCRAFT_DIR / f"dnacraft_per_pool_{source}.csv")
    g = df.groupby(["recipe", "cell", "selection"], sort=True)
    cols = [m[1] for m in METRIC_SPECS]
    out = g[cols].agg(["mean", "std", "count"]).reset_index()
    out.columns = [f"{a}_{b}" if b else a for a, b in out.columns]
    return out


def lookup(agg: pd.DataFrame, recipe: str, cell_internal: str, selection: str):
    sub = agg[(agg["recipe"] == recipe) &
              (agg["cell"] == cell_internal) &
              (agg["selection"] == selection)]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


def fmt(m, s, paper_key):
    if pd.isna(m):
        return "—"
    if paper_key == "mingap":
        return f"{m:+.3f}±{s:.3f}" if not pd.isna(s) else f"{m:+.3f}"
    return f"{m:.3f}±{s:.3f}" if not pd.isna(s) else f"{m:.3f}"


def fmt_paper(v, paper_key):
    if v is None:
        return "—"
    if paper_key == "mingap":
        return f"{v:+.3f}"
    return f"{v:.3f}"


def validate_selection_sanity(agg: pd.DataFrame, recipes: list) -> None:
    """Tripwire: each `by_X` selection rule should produce the highest mean
    of the metric X among the single-criterion rules, on every (recipe, cell)
    where X is the corresponding column.

    Selection→metric pairs (skip by_composite — multi-metric blend):
      by_mingap    → mingap_eval_mean
      by_motif     → motif_corr_spearman
      by_kmer3     → kmer3_corr_pearson
      by_diversity → diversity_bits

    Raises AssertionError listing every (recipe, cell, sel) where the rule
    failed to dominate its target metric.
    """
    sel_to_metric = {
        "by_mingap": "mingap_eval_mean",
        "by_motif": "motif_corr_spearman",
        "by_kmer3": "kmer3_corr_pearson",
        "by_diversity": "diversity_bits",
    }
    failures = []
    for recipe in recipes:
        for cell in agg["cell"].unique():
            sub = agg[(agg["recipe"] == recipe) & (agg["cell"] == cell) &
                      (agg["selection"].isin(sel_to_metric.keys()))]
            if len(sub) == 0:
                continue
            for sel, metric_col in sel_to_metric.items():
                rec_sel = sub[sub["selection"] == sel]
                if len(rec_sel) == 0:
                    continue
                target_val = float(rec_sel.iloc[0][f"{metric_col}_mean"])
                # Find the best selection for this metric
                best_idx = sub[f"{metric_col}_mean"].idxmax()
                best_sel = sub.loc[best_idx, "selection"]
                best_val = float(sub.loc[best_idx, f"{metric_col}_mean"])
                if best_sel != sel:
                    failures.append(
                        f"  {recipe[:50]:50}  {cell:6}  '{sel}' should be best for "
                        f"'{metric_col}' but '{best_sel}' wins ({best_val:.4f} vs {target_val:.4f})"
                    )
    if failures:
        raise AssertionError(
            "Selection-metric sanity check FAILED on the following rows:\n" +
            "\n".join(failures) +
            "\n→ The selection rule for the failing 'by_X' likely does not "
            "actually optimize metric X. Investigate before publishing."
        )
    print(f"✓ selection-metric sanity check passed for "
          f"{len(recipes)} recipe(s) × {len(agg['cell'].unique())} cell(s).")


def main():
    agg = aggregate("pool")
    validate_selection_sanity(agg, recipes=list(C3_RECIPE.values()))

    md = []
    md.append("# GPA — selection-rule sensitivity (paper Table 2 format)\n")
    md.append("Pool source = `pool`. Recipe = "
              "`dps_pw030_b25_gct060_nobio` (DPS, GC≤60%, no bio filter).")
    md.append("Each cell uses the cell-targeted pool. "
              "Top-128 selected by 5 different rules from the candidate pool.")
    md.append("")
    md.append("Selection rules:")
    md.append("- **by MinGap** [paper-faithful]: top-128 by per-seq MinGap = target − max(off-targets)")
    md.append("- **by composite**: top-128 by mean of 3 ranks (per-seq motif Pearson + 3-mer Pearson + diversity)")
    md.append("- **by Motif**: greedy max-Spearman(pool-mean motif counts vs ref) — set-level, directly optimizes the reported metric")
    md.append("- **by 3-mer**: top-128 by per-seq 3-mer Pearson against ref (sum-decomposable: per-seq selection ≈ pool optimum)")
    md.append("- **by Diversity**: greedy max-Shannon entropy (bits) of selected set — set-level, directly optimizes the reported metric")
    md.append("")

    methods = ["Ctrl-DNA", "DNA-CRAFT"] + [f"GPA ({SEL_LABEL[s]})" for s in SELECTIONS]
    header = ["Cell", "Metric"] + methods
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * len(header)) + " |")

    for cell in CELLS:
        cell_int = CELL_INTERNAL[cell]
        recipe = C3_RECIPE[cell]
        for metric_label, metric_col, paper_key in METRIC_SPECS:
            row = [cell, metric_label]
            for pm in ["Ctrl-DNA", "DNA-CRAFT"]:
                v = PAPER_BASELINES.get((cell, pm), {}).get(paper_key)
                row.append(fmt_paper(v, paper_key))
            for sel in SELECTIONS:
                rec = lookup(agg, recipe, cell_int, sel)
                if rec is None:
                    row.append("—")
                else:
                    row.append(fmt(rec[f"{metric_col}_mean"],
                                   rec[f"{metric_col}_std"], paper_key))
            md.append("| " + " | ".join(row) + " |")
    md.append("")

    out_md = DCRAFT_DIR / "FINAL_gpa_c3_selection_sensitivity.md"
    out_md.write_text("\n".join(md))
    print(f"wrote {out_md}\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
