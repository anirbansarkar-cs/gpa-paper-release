#!/usr/bin/env python3
"""Collate per-pool DNA-CRAFT-protocol JSONs into a Table-2-format CSV.

Reads:
  results/rerd_comparison/dnacraft/per_pool_jsons/<pool>__<CELL>__<sel>.json

Outputs:
  results/rerd_comparison/dnacraft/dnacraft_table2.csv

Aggregates mean ± std across the 3 seeds per (recipe, cell, selection),
matching DNA-CRAFT Table 2 layout. Also prints transcribed paper-side
baselines side-by-side for visual comparison.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Strip _r{seed} suffix (3 replicate seeds in RERD pipeline) so aggregation
# groups r1/r2/r3 of the same recipe together.
_SEED_SUFFIX_RE = re.compile(r"_r(\d+)$")


def parse_recipe_seed(pool_name: str) -> tuple:
    """Extract (recipe_label, seed) from a pool name like
    'run_dps_pw000_b25_r1'. Returns (recipe='run_dps_pw000_b25', seed=1).
    Falls back to (pool_name, 0) if no _r{N} suffix.
    """
    m = _SEED_SUFFIX_RE.search(pool_name)
    if m:
        return pool_name[: m.start()], int(m.group(1))
    return pool_name, 0

PROJECT = Path("${GPA_REPO_ROOT}")
DEFAULT_JSON_DIR = PROJECT / "results/rerd_comparison/dnacraft/per_pool_jsons"
DEFAULT_OUT_CSV = PROJECT / "results/rerd_comparison/dnacraft/dnacraft_table2.csv"

# DNA-CRAFT Table 2 paper-side baselines — canonical source in paper_baselines.py
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from paper_baselines import PAPER_BASELINES  # noqa: E402


def collate(json_dir: Path) -> pd.DataFrame:
    rows = []
    for jp in sorted(json_dir.glob("*.json")):
        try:
            d = json.loads(jp.read_text())
        except Exception as ex:
            print(f"  [skip] {jp.name}: {ex}")
            continue
        # Re-derive recipe + seed from pool_name (the upstream scorer's regex was
        # broken and stored full pool name as 'recipe'); strip _r{N} suffix here.
        pool_name = d.get("pool_name", jp.stem.split("__")[0])
        recipe, seed = parse_recipe_seed(pool_name)
        rows.append({
            "pool_name": pool_name,
            "recipe": recipe,
            "seed": seed,
            "cell": d.get("target_cell"),
            "selection": d.get("selection"),
            "n_seqs": int(d.get("n_seqs", 0)),
            "mingap_eval_mean": d.get("mingap_eval_mean"),
            "motif_corr_spearman": d.get("motif_corr_spearman"),
            "kmer3_corr_pearson": d.get("kmer3_corr_pearson"),
            "diversity_bits": d.get("diversity_bits"),
        })
    return pd.DataFrame(rows)


def aggregate_by_recipe(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (recipe, cell, selection); aggregate mean ± std across seeds."""
    g = df.groupby(["recipe", "cell", "selection"], sort=True)
    metric_cols = ["mingap_eval_mean", "motif_corr_spearman",
                   "kmer3_corr_pearson", "diversity_bits"]
    out = g[metric_cols].agg(["mean", "std", "count"]).reset_index()
    # Flatten MultiIndex columns
    out.columns = [f"{a}_{b}" if b else a for a, b in out.columns]
    return out


def fmt_paper_table(agg: pd.DataFrame, recipes: list = None) -> str:
    """Render in DNA-CRAFT Table 2 layout: rows = (cell, metric), cols = method."""
    if recipes is None:
        recipes = sorted(agg["recipe"].unique())
    cells = ["HepG2", "K562", "SK-N-SH"]
    metric_specs = [
        ("MinGap Eval ↑",  "mingap_eval_mean"),
        ("Motif Corr. ↑",  "motif_corr_spearman"),
        ("3-mer Corr. ↑",  "kmer3_corr_pearson"),
        ("Diversity ↑",    "diversity_bits"),
    ]
    # For our outputs, cell label is "SKNSH" not "SK-N-SH"; map both directions.
    cell_alias = {"HepG2": "HepG2", "K562": "K562", "SK-N-SH": "SKNSH"}
    paper_alias = {"HepG2": "HepG2", "K562": "K562", "SKNSH": "SK-N-SH"}

    lines = []
    paper_methods = ["SMC", "CG", "TDS", "DRAKES", "D3", "Ledidi", "Ctrl-DNA", "DNA-CRAFT"]
    header = ["Cell", "Metric"] + paper_methods + recipes
    lines.append(" | ".join(header))
    lines.append(" | ".join(["---"] * len(header)))

    for cell_disp in cells:
        cell_internal = cell_alias[cell_disp]
        for metric_label, metric_col in metric_specs:
            row = [cell_disp, metric_label]
            # paper baselines
            for pm in paper_methods:
                v = PAPER_BASELINES.get((cell_disp, pm), {}).get(
                    {"mingap_eval_mean": "mingap", "motif_corr_spearman": "motif",
                     "kmer3_corr_pearson": "kmer3", "diversity_bits": "div"}[metric_col])
                row.append(f"{v:.3f}" if v is not None else "—")
            # our recipes
            for r in recipes:
                sub = agg[(agg["recipe"] == r) & (agg["cell"] == cell_internal)]
                if len(sub) == 0:
                    row.append("—")
                    continue
                # Aggregate over selections (separate column per selection if you want)
                # Default: show 'by_mingap' (paper-faithful)
                mg_row = sub[sub["selection"] == "by_mingap"]
                if len(mg_row) == 0:
                    mg_row = sub.iloc[[0]]
                m = mg_row.iloc[0][f"{metric_col}_mean"]
                s = mg_row.iloc[0][f"{metric_col}_std"]
                if pd.isna(m):
                    row.append("—")
                else:
                    row.append(f"{m:+.3f}±{s:.3f}" if metric_col == "mingap_eval_mean"
                               else f"{m:.3f}±{s:.3f}")
            lines.append(" | ".join(row))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", type=Path, default=DEFAULT_JSON_DIR)
    ap.add_argument("--output_csv", type=Path, default=DEFAULT_OUT_CSV)
    args = ap.parse_args()

    print(f"Reading {args.json_dir} ...")
    df = collate(args.json_dir)
    print(f"  {len(df)} per-pool rows")
    if len(df) == 0:
        return

    # Per-pool wide CSV (one row per pool). Derived from output_csv stem so
    # different sources don't clobber each other.
    stem = args.output_csv.stem  # e.g. "dnacraft_table2_pool"
    suffix = stem.replace("dnacraft_table2", "").lstrip("_")
    raw_name = f"dnacraft_per_pool{('_' + suffix) if suffix else ''}.csv"
    raw_csv = args.output_csv.parent / raw_name
    df.to_csv(raw_csv, index=False, float_format="%.4f")
    print(f"  wrote {raw_csv}")

    # Aggregate
    agg = aggregate_by_recipe(df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.output_csv, index=False, float_format="%.4f")
    print(f"  wrote {args.output_csv}")

    # Pretty paper-format print
    print("\n" + "=" * 100)
    print("DNA-CRAFT Table-2-format comparison (paper baselines + our GPA recipes)")
    print("=" * 100)
    print(fmt_paper_table(agg))


if __name__ == "__main__":
    main()
