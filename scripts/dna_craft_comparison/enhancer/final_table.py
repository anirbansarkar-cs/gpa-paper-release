#!/usr/bin/env python3
"""Build the publication-ready GPA-vs-DNA-CRAFT comparison table.

Consumes `eval/aggregated.csv` (GPA's row, our experiment) and a hardcoded
table of paper Table 2 numbers (8 other methods × 3 cells), producing
`FINAL_gpa_vs_dnacraft.md` with the canonical DNA-CRAFT columns:

  | Method | Cell | MinGap (Eval) | ΔMinGap | Motif Corr | 3-mer Corr | Diversity |

Per plan §6 step 4, the primary number is **ΔMinGap = MinGap_eval −
MinGap_real_top_99.9%**. Because the paper trained an Eval-Model on a
different 50% of Gosai, raw absolute MinGaps are not directly comparable —
the delta vs the real-top anchor under each Eval-Model is portable.

Paper numbers below are placeholders; they will be filled in once the
camera-ready Table 2 is finalized. Empty cells render as `—`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "dna_craft_comparison" / "enhancer"

# --- Paper Table 2 (DNA-CRAFT, MLGenX @ ICLR 2026) — populate from camera-ready ---
# Each entry: (mingap_eval_mean, mingap_eval_std, motif_corr, kmer_corr, diversity)
# Use None for missing values; rendered as "—" in markdown.
PAPER_TABLE: dict[tuple[str, str], dict] = {
    # ("method", "cell"): {...}
    # ("smc", "hepg2"):    {"mingap_eval": (X, Y), "motif_corr": Z, "kmer_corr": W, "diversity": D},
    # Filled in later; left empty so final_table.py runs immediately.
}

PAPER_REAL_TOP_MINGAP: dict[str, float | None] = {
    "hepg2": None,  # Fill from paper Fig 7 once available.
    "k562": None,
    "sknsh": None,
}

METHODS_ORDER = ["smc", "cg", "tds", "drakes", "d3", "ledidi",
                 "ctrldna", "dna-craft",
                 "GPA-HyenaDNA (ours)", "GPA-DiMamba (ours)"]
CELL_ORDER = ["hepg2", "k562", "sknsh"]
COLS = ["MinGap (Eval)", "ΔMinGap", "Motif Corr", "3-mer Corr", "Diversity"]

# Mapping from a "GPA-*" method label to the eval/ subdir to read its
# aggregated.csv from. The HyenaDNA row reads from `eval/` (legacy default,
# scored under our Track E first wave). The DiMamba row reads from
# `eval/dimamba/` once that pipeline lands. If a subdir doesn't exist yet,
# the row stays as "—".
GPA_ROW_SOURCES = {
    "GPA-HyenaDNA (ours)": "eval",
    "GPA-DiMamba (ours)":  "eval/dimamba",
}


def fmt_pair(mean: float | None, std: float | None,
             precision: int = 3) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:+.{precision}f}"
    return f"{mean:+.{precision}f} ± {std:.{precision}f}"


def fmt_one(x: float | None, precision: int = 3) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:+.{precision}f}"


def load_gpa_row(eval_dir: Path) -> dict:
    agg_path = eval_dir / "aggregated.csv"
    if not agg_path.exists():
        return {}
    df = pd.read_csv(agg_path, header=[0, 1], index_col=0)
    out: dict[str, dict] = {}
    for cell in CELL_ORDER:
        if cell not in df.index:
            continue
        row = df.loc[cell]
        out[cell] = {
            "mingap_eval": (row.get(("mingap_eval", "mean"), None),
                             row.get(("mingap_eval", "std"), None)),
            "delta_mingap": (row.get(("delta_mingap", "mean"), None),
                             row.get(("delta_mingap", "std"), None)),
            "motif_corr": row.get(("motif_corr", "mean"), None),
            "kmer_corr": row.get(("kmer_corr", "mean"), None),
            "diversity": row.get(("diversity", "mean"), None),
        }
    return out


def to_markdown(gpa_rows_by_method: dict[str, dict],
                anchors: dict[str, float]) -> str:
    out: list[str] = []
    out.append("# GPA vs DNA-CRAFT — Gosai MPRA Enhancer\n")
    out.append("Paper-faithful 50/50 split-model evaluation. ΔMinGap is "
               "the primary cross-study number (anchor = mean MinGap on "
               "real top-99.9% under each Eval-Model).\n")
    out.append("| Method | Cell | " + " | ".join(COLS) + " |")
    out.append("|" + "|".join(["---"] * (2 + len(COLS))) + "|")
    for method in METHODS_ORDER:
        for cell in CELL_ORDER:
            if method in GPA_ROW_SOURCES:
                d = gpa_rows_by_method.get(method, {}).get(cell, {})
                mingap = fmt_pair(*(d.get("mingap_eval") or (None, None)))
                delta = fmt_pair(*(d.get("delta_mingap") or (None, None)))
                motif = fmt_one(d.get("motif_corr"))
                kmer = fmt_one(d.get("kmer_corr"))
                div = fmt_one(d.get("diversity"))
            else:
                paper = PAPER_TABLE.get((method, cell), {})
                mingap_pair = paper.get("mingap_eval")
                mingap = fmt_pair(*mingap_pair) if mingap_pair else "—"
                anchor = PAPER_REAL_TOP_MINGAP.get(cell)
                if mingap_pair and anchor is not None:
                    delta = fmt_one(mingap_pair[0] - anchor)
                else:
                    delta = "—"
                motif = fmt_one(paper.get("motif_corr"))
                kmer = fmt_one(paper.get("kmer_corr"))
                div = fmt_one(paper.get("diversity"))
            out.append(f"| {method} | {cell} | {mingap} | {delta} | "
                       f"{motif} | {kmer} | {div} |")
    out.append("")
    out.append("## Anchors (this Eval-Model)")
    for cell in CELL_ORDER:
        anc = anchors.get(cell)
        out.append(f"- **{cell}**: real-top-99.9% mean MinGap = "
                   f"{anc:+.3f}" if anc is not None else f"- **{cell}**: —")
    return "\n".join(out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_dir", default=str(RESULTS_DIR / "eval"))
    p.add_argument("--out", default=str(RESULTS_DIR / "FINAL_gpa_vs_dnacraft.md"))
    return p.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    # Anchors come from the HyenaDNA-stage eval (same Eval-Model is used for
    # both backbones, so the real-top-99.9% MinGap anchor is identical).
    anchors_path = eval_dir / "real_top_mingap.json"
    anchors = json.loads(anchors_path.read_text()) if anchors_path.exists() else {}
    # Load each GPA-* row from its dedicated subdir; absent subdirs render "—".
    gpa_rows_by_method: dict[str, dict] = {}
    for method, subdir in GPA_ROW_SOURCES.items():
        method_eval = (eval_dir / Path(subdir).relative_to("eval")
                       if subdir.startswith("eval/") else eval_dir)
        gpa_rows_by_method[method] = load_gpa_row(method_eval)
    md = to_markdown(gpa_rows_by_method, anchors)
    Path(args.out).write_text(md)
    print(f"[final_table] wrote {args.out}")


if __name__ == "__main__":
    main()
