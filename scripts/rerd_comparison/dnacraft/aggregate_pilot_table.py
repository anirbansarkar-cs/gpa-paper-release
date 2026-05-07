#!/usr/bin/env python3
"""Aggregate the dnacraft_pilot_* per-pool JSONs into a DNA-CRAFT Table-2-style
per-arm × per-cell summary.

Matches each pool to its target cell (the cell it was conditioned on) and
aggregates 4 metrics across 5 seeds: MinGap Eval, Motif Spearman,
3-mer Pearson, Diversity (bits).

Headline selection = by_composite (per user preference), with by_mingap
reported alongside as the paper-faithful reference.

Also pulls the standard GPA recipe (run_*_dps_pw030_b25_gct060_nobio) for
context, as established in FINAL_gpa_vs_dnacraft_table2.md.

Outputs:
  results/rerd_comparison/dnacraft/dnacraft_pilot_table2.csv
  results/rerd_comparison/dnacraft/dnacraft_pilot_table2.md
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("${GPA_REPO_ROOT}")
JSON_DIR = PROJECT / "results/rerd_comparison/dnacraft/per_pool_jsons"
OUT_CSV = PROJECT / "results/rerd_comparison/dnacraft/dnacraft_pilot_table2.csv"
OUT_MD = PROJECT / "results/rerd_comparison/dnacraft/dnacraft_pilot_table2.md"

CELL_LC_TO_CANON = {"hepg2": "HepG2", "k562": "K562", "sknsh": "SKNSH"}
ARMS = ["ucb_stacked", "uniform_stacked", "uniform_dialed"]
SELECTIONS = ["by_composite", "by_mingap"]

# NOTE (2026-05-03): ucb_stacked was promoted from i=12 → i=18 after the
# validation sweep (see dnacraft_i12_vs_i18.md). The headline ucb_stacked rows
# below now come from `dnacraft_ucb_iter_i18_K8_d2_*`, NOT the original
# `dnacraft_pilot_ucb_stacked_*` (i=12) pools. uniform_stacked / uniform_dialed
# remain on their original i=9 pilot pools.
UCB_I18_RE = re.compile(
    r"^dnacraft_ucb_iter_i18_K8_d2_(?P<cell>hepg2|k562|sknsh)_s(?P<seed>\d+)$"
)
PILOT_RE = re.compile(
    r"^dnacraft_pilot_(?P<arm>uniform_stacked|uniform_dialed)"
    r"_(?P<cell>hepg2|k562|sknsh)_s(?P<seed>\d+)$"
)
# standard GPA recipe pool names:
#   HepG2: run_dps_pw030_b25_gct060_nobio_r{1,2,3}      (no cell suffix)
#   K562:  run_dps_pw030_b25_gct060_nobio_k562_r{1,2,3}
#   SKNSH: run_dps_pw030_b25_gct060_nobio_sknsh_r{1,2,3}
GPA_HEPG2_RE = re.compile(r"^run_dps_pw030_b25_gct060_nobio_r(?P<seed>\d+)$")
GPA_OTHER_RE = re.compile(
    r"^run_dps_pw030_b25_gct060_nobio_(?P<cell>k562|sknsh)_r(?P<seed>\d+)$"
)


def load_jsons():
    rows = []
    for jp in sorted(JSON_DIR.glob("*.json")):
        try:
            d = json.loads(jp.read_text())
        except Exception:
            continue
        rows.append({
            "pool_name": d["pool_name"],
            "scored_cell": d["target_cell"],
            "selection": d["selection"],
            "mingap_eval": d["mingap_eval_mean"],
            "motif_spr": d["motif_corr_spearman"],
            "kmer3_pearson": d["kmer3_corr_pearson"],
            "diversity_bits": d["diversity_bits"],
            "n_seqs": d["n_seqs"],
        })
    return pd.DataFrame(rows)


def label_method(pool_name):
    # ucb_stacked headline source: i=18 validation pools (promoted 2026-05-03).
    m = UCB_I18_RE.match(pool_name)
    if m:
        return "pilot_ucb_stacked", CELL_LC_TO_CANON[m.group("cell")], int(m.group("seed"))
    m = PILOT_RE.match(pool_name)
    if m:
        return f"pilot_{m.group('arm')}", CELL_LC_TO_CANON[m.group("cell")], int(m.group("seed"))
    m = GPA_HEPG2_RE.match(pool_name)
    if m:
        return "GPA_baseline", "HepG2", int(m.group("seed"))
    m = GPA_OTHER_RE.match(pool_name)
    if m:
        return "GPA_baseline", CELL_LC_TO_CANON[m.group("cell")], int(m.group("seed"))
    return None, None, None


def aggregate(df):
    rows = []
    for _, r in df.iterrows():
        method, pool_cell, seed = label_method(r["pool_name"])
        if method is None:
            continue
        # Match scored cell to the pool's conditioning cell (matched-cell only).
        if pool_cell != r["scored_cell"]:
            continue
        rows.append({
            "method": method,
            "cell": r["scored_cell"],
            "selection": r["selection"],
            "seed": seed,
            "mingap_eval": r["mingap_eval"],
            "motif_spr": r["motif_spr"],
            "kmer3_pearson": r["kmer3_pearson"],
            "diversity_bits": r["diversity_bits"],
            "n_seqs": r["n_seqs"],
        })
    long = pd.DataFrame(rows)

    # Aggregate mean ± std across seeds
    agg = long.groupby(["method", "cell", "selection"]).agg(
        n_seeds=("seed", "count"),
        n=("n_seqs", "first"),
        mingap_mean=("mingap_eval", "mean"),
        mingap_std=("mingap_eval", "std"),
        motif_mean=("motif_spr", "mean"),
        motif_std=("motif_spr", "std"),
        kmer_mean=("kmer3_pearson", "mean"),
        kmer_std=("kmer3_pearson", "std"),
        div_mean=("diversity_bits", "mean"),
        div_std=("diversity_bits", "std"),
    ).reset_index()
    return long, agg


def fmt(mean, std, dec=3, sign=False):
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:+.{dec}f}" if sign else f"{mean:.{dec}f}"
    return (f"{mean:+.{dec}f}±{std:.{dec}f}" if sign
            else f"{mean:.{dec}f}±{std:.{dec}f}")


# Paper baselines (verbatim from FINAL_gpa_vs_dnacraft_table2.md, pool source)
PAPER = {
    ("HepG2", "mingap"): {"Ctrl-DNA": "+7.786", "DNA-CRAFT": "+4.346"},
    ("HepG2", "motif"):  {"Ctrl-DNA": "0.629",  "DNA-CRAFT": "0.921"},
    ("HepG2", "kmer"):   {"Ctrl-DNA": "0.494",  "DNA-CRAFT": "0.980"},
    ("HepG2", "div"):    {"Ctrl-DNA": "1.897",  "DNA-CRAFT": "1.979"},
    ("K562",  "mingap"): {"Ctrl-DNA": "+9.067", "DNA-CRAFT": "+5.686"},
    ("K562",  "motif"):  {"Ctrl-DNA": "0.634",  "DNA-CRAFT": "0.933"},
    ("K562",  "kmer"):   {"Ctrl-DNA": "0.413",  "DNA-CRAFT": "0.976"},
    ("K562",  "div"):    {"Ctrl-DNA": "1.896",  "DNA-CRAFT": "1.981"},
    ("SKNSH", "mingap"): {"Ctrl-DNA": "+3.720", "DNA-CRAFT": "+3.230"},
    ("SKNSH", "motif"):  {"Ctrl-DNA": "0.477",  "DNA-CRAFT": "0.881"},
    ("SKNSH", "kmer"):   {"Ctrl-DNA": "0.201",  "DNA-CRAFT": "0.969"},
    ("SKNSH", "div"):    {"Ctrl-DNA": "1.855",  "DNA-CRAFT": "1.976"},
}

METHOD_LABEL = {
    "GPA_baseline":          "GPA (baseline)",
    "pilot_ucb_stacked":     "ucb_stacked",
    "pilot_uniform_stacked": "uniform_stacked",
    "pilot_uniform_dialed":  "uniform_dialed",
}
METHOD_ORDER = ["GPA_baseline", "pilot_ucb_stacked",
                "pilot_uniform_stacked", "pilot_uniform_dialed"]
CELL_ORDER = ["HepG2", "K562", "SKNSH"]


def render_cell_table(agg, cell, selection):
    """One narrow 6-col table for a single (cell, selection): rows = methods,
    cols = 4 metrics. Includes paper baselines (Ctrl-DNA, DNA-CRAFT) at top."""
    sub = agg[(agg["selection"] == selection) & (agg["cell"] == cell)].copy()
    rows = []
    rows.append(f"#### {cell} — selection={selection}")
    rows.append("")
    rows.append("| Method | MinGap Eval ↑ | Motif Spr ↑ | 3-mer Pearson ↑ | Diversity (bits) ↑ |")
    rows.append("| --- | ---: | ---: | ---: | ---: |")
    # Paper baselines
    for paper_method in ["Ctrl-DNA", "DNA-CRAFT"]:
        rows.append("| " + " | ".join([
            paper_method,
            PAPER[(cell, "mingap")][paper_method],
            PAPER[(cell, "motif")][paper_method],
            PAPER[(cell, "kmer")][paper_method],
            PAPER[(cell, "div")][paper_method],
        ]) + " |")
    # GPA methods
    for m in METHOD_ORDER:
        r = sub[sub["method"] == m]
        if r.empty:
            cells = [METHOD_LABEL[m], "—", "—", "—", "—"]
        else:
            r = r.iloc[0]
            cells = [
                METHOD_LABEL[m],
                fmt(r["mingap_mean"], r["mingap_std"], dec=3, sign=True),
                fmt(r["motif_mean"],  r["motif_std"],  dec=3),
                fmt(r["kmer_mean"],   r["kmer_std"],   dec=3),
                fmt(r["div_mean"],    r["div_std"],    dec=3),
            ]
        rows.append("| " + " | ".join(cells) + " |")
    rows.append("")
    return "\n".join(rows)


def render_md(agg, selection):
    if agg[agg["selection"] == selection].empty:
        return f"_(no data for selection={selection})_\n"
    out = [f"### Selection: {selection}\n"]
    for cell in CELL_ORDER:
        out.append(render_cell_table(agg, cell, selection))
    return "\n".join(out)


def main():
    df = load_jsons()
    long, agg = aggregate(df)

    # CSV: long-form aggregated
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"Wrote {OUT_CSV} ({len(agg)} rows)")

    md = []
    md.append("# DNA-CRAFT MCTS Pilot — Table-2 Format (pool, top-128)\n")
    md.append("Pilot arms: `ucb_stacked` (GPA + UCB-MCTS d=2 **i=18** K=8 — promoted from i=12 after the 2026-05-03 validation sweep), "
              "`uniform_stacked` (GPA + uniform-MCTS d=2 i=9 K=8), "
              "`uniform_dialed` (no-DPS, max_beta=10, max_steps=12, K=8 + uniform-MCTS d=2 i=9).\n")
    md.append("Each cell aggregates 5 seeds. Pool source: `gpa_output_pool.h5`. "
              "Scored against matched conditioning cell only.\n")
    md.append("Paper baselines (Ctrl-DNA, DNA-CRAFT) verbatim from `FINAL_gpa_vs_dnacraft_table2.md`. "
              "standard GPA = `run_dps_pw030_b25_gct060_nobio` (3 seeds).\n")

    md.append(render_md(agg, "by_composite"))
    md.append(render_md(agg, "by_mingap"))

    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")
    print()
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
