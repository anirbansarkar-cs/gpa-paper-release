#!/usr/bin/env python3
"""Build the DNA-CRAFT-style top-64 comparison table:

  Methods (rows): Ctrl-DNA, DNA-CRAFT (verbatim paper numbers),
                  GPA, GPA-ucb_stacked i=12, GPA-ucb_stacked i=18.
  Cells (super-row groups): HepG2, K562, SK-N-SH.
  Metrics (columns): MinGap (eval), Motif Pearson, 3-mer Pearson, Diversity (bits).
  Selections (separate sub-tables): by_mingap, by_composite.

Reads from results/rerd_comparison/dnacraft/per_pool_jsons_n64/
(produced by `run_batch_score.py --source pool --n 64`).

Usage: python scripts/rerd_comparison/dnacraft/build_top64_table.py [--md path]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path("${GPA_REPO_ROOT}")
JSON_DIR = PROJECT / "results/rerd_comparison/dnacraft/per_pool_jsons_n64"

# Pool name → (GPA method label, target cell display name)
GPA_PATTERNS = [
    # HepG2 = nobio recipe (universal pick), 3 reps r1/r2/r3
    (re.compile(r"^run_dps_pw030_b25_gct060_nobio_r\d+$"),
     "GPA", "HepG2"),
    # K562 / SK-N-SH = nobio recipe, 3 reps
    (re.compile(r"^run_dps_pw030_b25_gct060_nobio_(k562)_r\d+$"),
     "GPA", "K562"),
    (re.compile(r"^run_dps_pw030_b25_gct060_nobio_(sknsh)_r\d+$"),
     "GPA", "SK-N-SH"),
    # ucb_stacked i=12 (pilot), 5 seeds per cell
    (re.compile(r"^dnacraft_pilot_ucb_stacked_(hepg2|k562|sknsh)_s\d+$"),
     "GPA-ucb_stacked i=12", None),
    # ucb_stacked i=18 (validation), 5 seeds per cell
    (re.compile(r"^dnacraft_ucb_iter_i18_K8_d2_(hepg2|k562|sknsh)_s\d+$"),
     "GPA-ucb_stacked i=18", None),
]
CELL_NORMAL = {"hepg2": "HepG2", "k562": "K562", "sknsh": "SK-N-SH"}

# Verbatim from paper_baselines.py (paper Table 2; selection per their protocol)
PAPER = {
    ("HepG2", "Ctrl-DNA"):   {"mingap": (7.786, 0.070), "motif": (0.629, 0.045), "kmer3": (0.494, 0.028), "div": (1.897, 0.026)},
    ("HepG2", "DNA-CRAFT"):  {"mingap": (4.346, 0.050), "motif": (0.921, 0.006), "kmer3": (0.980, 0.009), "div": (1.979, 0.000)},
    ("K562",  "Ctrl-DNA"):   {"mingap": (9.067, 0.170), "motif": (0.634, 0.084), "kmer3": (0.413, 0.058), "div": (1.896, 0.021)},
    ("K562",  "DNA-CRAFT"):  {"mingap": (5.686, 0.043), "motif": (0.933, 0.010), "kmer3": (0.976, 0.000), "div": (1.981, 0.001)},
    ("SK-N-SH","Ctrl-DNA"):  {"mingap": (3.720, 0.179), "motif": (0.477, 0.037), "kmer3": (0.201, 0.172), "div": (1.855, 0.091)},
    ("SK-N-SH","DNA-CRAFT"): {"mingap": (3.230, 0.022), "motif": (0.881, 0.031), "kmer3": (0.969, 0.007), "div": (1.976, 0.002)},
}

EVAL_CELL_TAG = {"HepG2": "HepG2", "K562": "K562", "SK-N-SH": "SKNSH"}
ROW_ORDER = ["Ctrl-DNA", "DNA-CRAFT",
             "GPA", "GPA-ucb_stacked i=12", "GPA-ucb_stacked i=18"]
CELL_ORDER = ["HepG2", "K562", "SK-N-SH"]


def classify_pool(pool_name: str):
    """Return (method, target_cell_display) or (None, None) if unmatched."""
    for pat, label, fixed_cell in GPA_PATTERNS:
        m = pat.match(pool_name)
        if not m:
            continue
        if fixed_cell is not None:
            return label, fixed_cell
        return label, CELL_NORMAL[m.group(1)]
    return None, None


def load_gpa_metrics():
    """Group pool JSONs by (method, target_cell, selection); each group
    holds one float per seed/rep for each metric."""
    groups = defaultdict(lambda: defaultdict(list))
    if not JSON_DIR.exists():
        raise SystemExit(f"Missing JSON dir {JSON_DIR}; run "
                         f"run_batch_score.py --n 64 first.")
    for path in sorted(JSON_DIR.glob("*.json")):
        # name = <pool>__<EvalCell>__<selection>.json
        stem = path.stem
        parts = stem.rsplit("__", 2)
        if len(parts) != 3:
            continue
        pool_name, eval_cell, selection = parts
        method, target_cell = classify_pool(pool_name)
        if method is None:
            continue
        # Only keep on-target eval (run cell == eval cell).
        if EVAL_CELL_TAG[target_cell] != eval_cell:
            continue
        with path.open() as f:
            d = json.load(f)
        rec = groups[(method, target_cell, selection)]
        rec["mingap"].append(d["mingap_eval_mean"])
        rec["motif"].append(d["motif_corr_spearman"])
        rec["kmer3"].append(d["kmer3_corr_pearson"])
        rec["div"].append(d["diversity_bits"])
    return groups


def fmt(mean, std):
    if std is None or not np.isfinite(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ({std:.3f})"


def render_subtable(groups, selection: str) -> list[str]:
    lines = []
    lines.append(f"### Selection: `{selection}` (top-64)")
    lines.append("")
    header = ("| Cell | Method | MinGap (eval) ↑ | Motif Pearson ↑ "
              "| 3-mer Pearson ↑ | Diversity (bits) ↑ | n |")
    sep = "|---|---|---|---|---|---|---:|"
    lines.extend([header, sep])

    for cell in CELL_ORDER:
        for method in ROW_ORDER:
            row_cells = [cell, method]
            n = "-"
            if method in ("Ctrl-DNA", "DNA-CRAFT"):
                p = PAPER.get((cell, method))
                if p is None:
                    row_cells += ["—"] * 4
                    n = "—"
                else:
                    row_cells.append(fmt(*p["mingap"]))
                    row_cells.append(fmt(*p["motif"]))
                    row_cells.append(fmt(*p["kmer3"]))
                    row_cells.append(fmt(*p["div"]))
                    n = "3 (paper)"
            else:
                rec = groups.get((method, cell, selection))
                if not rec or not rec["mingap"]:
                    row_cells += ["—"] * 4
                    n = "0"
                else:
                    for k in ("mingap", "motif", "kmer3", "div"):
                        arr = np.asarray(rec[k], dtype=float)
                        mu = float(arr.mean())
                        sd = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
                        row_cells.append(fmt(mu, sd))
                    n = str(len(rec["mingap"]))
            row_cells.append(n)
            lines.append("| " + " | ".join(row_cells) + " |")
        if cell != CELL_ORDER[-1]:
            lines.append("|   |   |   |   |   |   |   |")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=Path,
                    default=PROJECT / "results/dna_craft_comparison/top64_table.md",
                    help="Output Markdown path.")
    args = ap.parse_args()
    args.md.parent.mkdir(parents=True, exist_ok=True)

    groups = load_gpa_metrics()
    out = []
    out.append("# DNA-CRAFT-style comparison @ top-64 (matched to G* output volume)")
    out.append("")
    out.append("Pool source = `gpa_output_pool.h5`. "
               "Ctrl-DNA / DNA-CRAFT rows are verbatim from the DNA-CRAFT paper "
               "(arXiv 2604.20488 Table 2). GPA rows are rescored at top-64 from "
               "the same pools used in the existing top-128 tables. "
               "Mean (std). GPA = standard `dps_pw030_b25_gct060_nobio` "
               "recipe, no MCTS, 3 reps. GPA-ucb_stacked uses the standard GPA base "
               "+ depth-2 UCB-MCTS at i=12 (pilot) / i=18 (validation), 5 seeds.")
    out.append("")
    out.extend(render_subtable(groups, "by_pool_composite"))
    out.extend(render_subtable(groups, "by_composite"))
    out.extend(render_subtable(groups, "by_mingap"))

    args.md.write_text("\n".join(out) + "\n")
    print(f"Wrote {args.md}")
    print()
    print("\n".join(out))


if __name__ == "__main__":
    main()
