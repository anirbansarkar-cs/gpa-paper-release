#!/usr/bin/env python3
"""Build paper-format DNA-CRAFT Table 1 row for GPA, comparing across the
pool source.

For each (cell, metric) we pick the cell-targeted GPA recipe(s) and report
mean ± std across 3 seeds, alongside the transcribed paper baselines.

Selected GPA recipes:
  - no-GC-penalty:    run_dps_pw030_b25
                      run_dps_pw030_b25_k562
                      run_dps_pw030_b25_sknsh
  - standard (with GC penalty):
                      run_dps_pw030_b25_gct060_nobio
                      run_dps_pw030_b25_gct060_nobio_k562
                      run_dps_pw030_b25_gct060_nobio_sknsh
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("${GPA_REPO_ROOT}")
DCRAFT_DIR = PROJECT / "results/rerd_comparison/dnacraft"

# Canonical baselines (single source of truth)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from paper_baselines import PAPER_BASELINES  # noqa: E402

PAPER_METHODS = ["SMC", "CG", "TDS", "DRAKES", "D3", "Ledidi", "Ctrl-DNA", "DNA-CRAFT"]
CELLS = ["HepG2", "K562", "SK-N-SH"]
METRIC_SPECS = [
    ("MinGap Eval ↑",  "mingap_eval_mean",     "mingap"),
    ("Motif Spr ↑",    "motif_corr_spearman",  "motif"),
    ("3-mer Pearson ↑","kmer3_corr_pearson",   "kmer3"),
    ("Diversity ↑",    "diversity_bits",       "div"),
]
CELL_INTERNAL = {"HepG2": "HepG2", "K562": "K562", "SK-N-SH": "SKNSH"}

# Per cell: which recipe to pull the GPA row from. The pool name has been
# trained against the right cell.
GPA_RECIPE_BY_CELL = {
    "HepG2": {  # pools were HepG2-targeted by default
        "no_gc_penalty": "run_dps_pw030_b25",
        "standard":      "run_dps_pw030_b25_gct060_nobio",
    },
    "K562": {
        "no_gc_penalty": "run_dps_pw030_b25_k562",
        "standard":      "run_dps_pw030_b25_gct060_nobio_k562",
    },
    "SK-N-SH": {
        "no_gc_penalty": "run_dps_pw030_b25_sknsh",
        "standard":      "run_dps_pw030_b25_gct060_nobio_sknsh",
    },
}


def load_per_pool(source: str) -> pd.DataFrame:
    fp = DCRAFT_DIR / f"dnacraft_per_pool_{source}.csv"
    df = pd.read_csv(fp)
    return df


def aggregate_by_recipe_cell_sel(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["recipe", "cell", "selection"], sort=True)
    metric_cols = ["mingap_eval_mean", "motif_corr_spearman",
                   "kmer3_corr_pearson", "diversity_bits"]
    out = g[metric_cols].agg(["mean", "std", "count"]).reset_index()
    out.columns = [f"{a}_{b}" if b else a for a, b in out.columns]
    return out


def lookup(agg: pd.DataFrame, recipe: str, cell_internal: str,
           selection: str = "by_mingap") -> dict | None:
    sub = agg[(agg["recipe"] == recipe) &
              (agg["cell"] == cell_internal) &
              (agg["selection"] == selection)]
    if len(sub) == 0:
        # fall back to any selection
        sub = agg[(agg["recipe"] == recipe) & (agg["cell"] == cell_internal)]
    if len(sub) == 0:
        return None
    r = sub.iloc[0]
    return {col: r[col] for col in r.index}


def fmt_val(m, s, metric_col):
    if pd.isna(m):
        return "—"
    if metric_col == "mingap_eval_mean":
        return f"{m:+.3f}±{s:.3f}" if not pd.isna(s) else f"{m:+.3f}"
    return f"{m:.3f}±{s:.3f}" if not pd.isna(s) else f"{m:.3f}"


def build_table(sources: list[str]) -> str:
    aggs = {src: aggregate_by_recipe_cell_sel(load_per_pool(src)) for src in sources}

    lines = []
    methods = PAPER_METHODS + [f"GPA ({s})" for s in sources] + \
              [f"GPA ({s})" for s in sources]
    header = ["Cell", "Metric"] + methods
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for cell in CELLS:
        cell_int = CELL_INTERNAL[cell]
        for metric_label, metric_col, paper_key in METRIC_SPECS:
            row = [cell, metric_label]
            for pm in PAPER_METHODS:
                v = PAPER_BASELINES.get((cell, pm), {}).get(paper_key)
                row.append(f"{v:+.3f}" if (paper_key == "mingap" and v is not None)
                           else (f"{v:.3f}" if v is not None else "—"))
            for tag in ["no_gc_penalty", "standard"]:
                recipe = GPA_RECIPE_BY_CELL[cell][tag]
                for src in sources:
                    rec = lookup(aggs[src], recipe, cell_int)
                    if rec is None:
                        row.append("—")
                        continue
                    m = rec[f"{metric_col}_mean"]
                    s = rec[f"{metric_col}_std"]
                    row.append(fmt_val(m, s, metric_col))
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_compact_table(sources: list[str]) -> str:
    """One source at a time, narrower table."""
    out = []
    for src in sources:
        agg = aggregate_by_recipe_cell_sel(load_per_pool(src))
        out.append(f"\n### Source: `{src}`\n")
        methods = PAPER_METHODS + ["**GPA**", "**GPA**"]
        header = ["Cell", "Metric"] + methods
        out.append("| " + " | ".join(header) + " |")
        out.append("| " + " | ".join(["---"] * len(header)) + " |")
        for cell in CELLS:
            cell_int = CELL_INTERNAL[cell]
            for metric_label, metric_col, paper_key in METRIC_SPECS:
                row = [cell, metric_label]
                for pm in PAPER_METHODS:
                    v = PAPER_BASELINES.get((cell, pm), {}).get(paper_key)
                    if v is None:
                        row.append("—")
                    elif paper_key == "mingap":
                        row.append(f"{v:+.3f}")
                    else:
                        row.append(f"{v:.3f}")
                for tag in ["no_gc_penalty", "standard"]:
                    recipe = GPA_RECIPE_BY_CELL[cell][tag]
                    rec = lookup(agg, recipe, cell_int)
                    if rec is None:
                        row.append("—")
                    else:
                        row.append(fmt_val(rec[f"{metric_col}_mean"],
                                           rec[f"{metric_col}_std"], metric_col))
                out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main():
    sources = ["pool"]
    md = []
    md.append("# GPA vs DNA-CRAFT — Paper-format Comparison\n")
    md.append("Top-128 selected by MinGap Eval (paper-faithful selection).")
    md.append("GPA recipes: no GC penalty = `dps_pw030_b25` (DPS); "
              "standard = `dps_pw030_b25_gct060_nobio` (DPS, GC≤60%, no bio filter).\n")
    md.append("## Per-source comparison\n")
    md.append(build_compact_table(sources))
    md.append("\n## Combined (all 3 sources side-by-side)\n")
    md.append(build_table(sources))
    md.append("")

    out_md = DCRAFT_DIR / "FINAL_gpa_vs_dnacraft_table2.md"
    out_md.write_text("\n".join(md))
    print(f"wrote {out_md}")
    print("\n" + "=" * 100)
    print("\n".join(md))


if __name__ == "__main__":
    main()
