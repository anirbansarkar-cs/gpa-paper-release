#!/usr/bin/env python3
"""Pareto plot: per-run MinGap (Eval) vs Diversity (Shannon) for the GPA pools.

Reads `eval/per_run_metrics.csv` (9 rows: 3 cells × 3 seeds), overlays the
real-top-99.9% MinGap anchor from `eval/real_top_mingap.json` as a vertical
guide line per cell, and emits a PNG to `eval/pareto_mingap_diversity.png`.

Each point is one (cell, seed) GPA run. A run is "Pareto-good" if it is to
the right of (higher MinGap than) the real-anchor line AND has high Shannon.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = PROJECT_ROOT / "results" / "dna_craft_comparison" / "enhancer" / "eval"

CELL_COLORS = {"hepg2": "#1f77b4", "k562": "#d62728", "sknsh": "#2ca02c"}
CELL_LABELS = {"hepg2": "HepG2", "k562": "K562", "sknsh": "SK-N-SH"}


def main() -> None:
    df = pd.read_csv(EVAL_DIR / "per_run_metrics.csv")
    anchors = json.loads((EVAL_DIR / "real_top_mingap.json").read_text())

    fig, ax = plt.subplots(figsize=(7, 5))
    for cell, group in df.groupby("cell"):
        ax.scatter(
            group["mingap_eval"], group["diversity"],
            color=CELL_COLORS[cell], s=70,
            label=f"GPA — {CELL_LABELS[cell]}",
            edgecolors="black", linewidths=0.5)
        # Per-cell real-top-99.9% anchor: vertical dashed line.
        ax.axvline(
            anchors[cell], color=CELL_COLORS[cell], linestyle="--",
            alpha=0.6, linewidth=1.2,
            label=f"real top-99.9% — {CELL_LABELS[cell]} ({anchors[cell]:.2f})")

    ax.set_xlabel("MinGap (Eval-Model)")
    ax.set_ylabel("Diversity (per-position Shannon)")
    ax.set_title(
        "GPA Pareto: MinGap vs Diversity, top-128 by MinGap_eval\n"
        f"(n={len(df)} runs = 3 cells × 3 seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    out = EVAL_DIR / "pareto_mingap_diversity.png"
    fig.savefig(out, dpi=150)
    print(f"[pareto] wrote {out}")


if __name__ == "__main__":
    main()
