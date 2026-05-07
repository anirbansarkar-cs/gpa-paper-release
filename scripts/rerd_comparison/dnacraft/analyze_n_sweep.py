#!/usr/bin/env python3
"""Aggregate the DNA-CRAFT N-sweep ablation (standard GPA, no MCTS).

Reads per-pool JSONs produced by run_batch_score.py from
results/rerd_comparison/dnacraft/per_pool_jsons_{source}/
matching dnacraft_nsweep_n{N}_{cell}_s{seed}__{CELL}__by_composite.json,
where source is the candidate pool. Groups by
(source, N, target_cell), and emits:

  (1) results/dna_craft_comparison/n_sweep_table.csv
      One row per (N, cell): mean and std over seeds for the four
      DNA-CRAFT Table-2 metrics (MinGap, Motif Pearson, 3-mer Pearson, Diversity).

  (2) results/dna_craft_comparison/n_sweep_table.md
      Markdown version of the same table for the appendix.

  (3) results/dna_craft_comparison/n_sweep_curves.pdf
      Four-panel metric-vs-N plot, log-x, three cell lines per panel,
      shaded mean +/- std bands. Visual companion to the table.

Usage: python scripts/rerd_comparison/dnacraft/analyze_n_sweep.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("${GPA_REPO_ROOT}")
# Headline = top-64 by_pool_composite (locked 2026-05-05). Other
# sources/selections are read for supplementary tables only.
JSON_DIRS = {
    "pool":           PROJECT / "results/rerd_comparison/dnacraft/per_pool_jsons_n64",
    "mingap_archive":      PROJECT / "results/rerd_comparison/dnacraft/per_pool_jsons_mingap_archive",
    "perstep_archive":     PROJECT / "results/rerd_comparison/dnacraft/per_pool_jsons_perstep_archive",
}
HEADLINE_SOURCE = "pool"
HEADLINE_SELECTION = "by_pool_composite"

OUT_DIR = PROJECT / "results/dna_craft_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATTERN = re.compile(
    r"^dnacraft_nsweep_n(?P<N>\d+)_(?P<cell>hepg2|k562|sknsh)_s(?P<seed>\d+)"
    r"__(?P<eval_cell>HepG2|K562|SKNSH)__(?P<sel>by_composite|by_mingap|by_pool_composite)\.json$"
)

CELL_DISPLAY = {"hepg2": "HepG2", "k562": "K562", "sknsh": "SK-N-SH"}
EVAL_CELL_MATCH = {"hepg2": "HepG2", "k562": "K562", "sknsh": "SKNSH"}

METRICS = [
    ("mingap_eval_mean", "MinGap"),
    ("motif_corr_spearman", "Motif"),
    ("kmer3_corr_pearson", "3-mer"),
    ("diversity_bits", "Diversity"),
]


def collect():
    rows = []
    for source, json_dir in JSON_DIRS.items():
        if not json_dir.exists():
            print(f"  [skip] {source}: dir does not exist ({json_dir})")
            continue
        for path in sorted(json_dir.glob("dnacraft_nsweep_*.json")):
            m = PATTERN.match(path.name)
            if not m:
                continue
            # Only keep on-target eval (run cell == eval cell).
            if EVAL_CELL_MATCH[m["cell"]] != m["eval_cell"]:
                continue
            with path.open() as f:
                data = json.load(f)
            rows.append({
                "source": source,
                "selection": m["sel"],
                "N": int(m["N"]),
                "cell": m["cell"],
                "seed": int(m["seed"]),
                **{key: data.get(key) for key, _ in METRICS},
            })
    if not rows:
        raise SystemExit(
            "No matching JSONs in any of: "
            + ", ".join(str(d) for d in JSON_DIRS.values()))
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["source", "selection", "N", "cell"])
    agg = {}
    for key, _ in METRICS:
        agg[f"{key}_mean"] = grouped[key].mean()
        agg[f"{key}_std"] = grouped[key].std(ddof=1)
    out = pd.DataFrame(agg).reset_index()
    out["n_seeds"] = grouped.size().values
    return out.sort_values(
        ["source", "selection", "cell", "N"]
    ).reset_index(drop=True)


def write_markdown(agg: pd.DataFrame, path: Path):
    lines = ["# DNA-CRAFT N-sweep — GPA (no MCTS) headline\n"]
    lines.append(
        f"**Headline** = `{HEADLINE_SOURCE}` source × `{HEADLINE_SELECTION}` "
        "selection at top-64. This matches the methodology used for the "
        "GPA row in the DNA-CRAFT paper comparison: same pool, "
        "same greedy pool-aggregate selection. Mean (std) over seeds.\n")

    headline = agg[(agg["source"] == HEADLINE_SOURCE)
                   & (agg["selection"] == HEADLINE_SELECTION)]
    if not headline.empty:
        lines.append("## Headline N-sweep")
        lines.append("")
        header = ("| Cell | N | seeds | MinGap | Motif | 3-mer | Diversity |")
        sep = "|---|---:|---:|---|---|---|---|"
        lines.extend([header, sep])
        for _, r in headline.iterrows():
            cells = [CELL_DISPLAY[r["cell"]],
                     f"{int(r['N'])}",
                     f"{int(r['n_seeds'])}"]
            for key, _ in METRICS:
                mu, sd = r[f"{key}_mean"], r[f"{key}_std"]
                cells.append(f"{mu:.3f} ({sd:.3f})"
                             if np.isfinite(sd) else f"{mu:.3f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    other = agg[~((agg["source"] == HEADLINE_SOURCE)
                  & (agg["selection"] == HEADLINE_SELECTION))]
    if not other.empty:
        lines.append("## Supplementary (other sources / selections)")
        lines.append("")
        header = ("| Source | Selection | Cell | N | seeds | "
                  "MinGap | Motif | 3-mer | Diversity |")
        sep = "|---|---|---|---:|---:|---|---|---|---|"
        lines.extend([header, sep])
        for _, r in other.iterrows():
            cells = [r["source"], r["selection"], CELL_DISPLAY[r["cell"]],
                     f"{int(r['N'])}", f"{int(r['n_seeds'])}"]
            for key, _ in METRICS:
                mu, sd = r[f"{key}_mean"], r[f"{key}_std"]
                cells.append(f"{mu:.3f} ({sd:.3f})"
                             if np.isfinite(sd) else f"{mu:.3f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def plot_curves(agg: pd.DataFrame, source: str, selection: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub_all = agg[(agg["source"] == source) & (agg["selection"] == selection)]
    if sub_all.empty:
        print(f"  [skip plot] no rows for source={source} selection={selection}")
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharex=True)
    color = {"hepg2": "#1f77b4", "k562": "#d62728", "sknsh": "#2ca02c"}

    for ax, (key, label) in zip(axes, METRICS):
        for cell in ("hepg2", "k562", "sknsh"):
            sub = sub_all[sub_all["cell"] == cell].sort_values("N")
            if sub.empty:
                continue
            mu = sub[f"{key}_mean"].to_numpy()
            sd = sub[f"{key}_std"].fillna(0).to_numpy()
            ax.plot(sub["N"], mu, "o-", color=color[cell],
                    label=CELL_DISPLAY[cell])
            ax.fill_between(sub["N"], mu - sd, mu + sd,
                            color=color[cell], alpha=0.2)
        ax.set_xscale("log")
        ax.set_xlabel("Population size $N$")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"metric value (top-64 {selection})")
    axes[-1].legend(loc="best", fontsize=9)

    fig.suptitle(
        f"GPA metric vs. population size $N$ — source={source}, "
        f"selection={selection} (DNA-CRAFT bench, MDLM backbone)",
        y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"Wrote {path}")


def main():
    df = collect()
    print(f"Loaded {len(df)} on-target rows; "
          f"unique (source, selection, N, cell): "
          f"{df.groupby(['source', 'selection', 'N', 'cell']).ngroups}")
    agg = aggregate(df)

    agg.to_csv(OUT_DIR / "n_sweep_table.csv", index=False)
    print(f"Wrote {OUT_DIR / 'n_sweep_table.csv'}")
    write_markdown(agg, OUT_DIR / "n_sweep_table.md")

    # Headline curve first; then any supplementary (source, selection) pairs.
    headline_pdf = (OUT_DIR
                    / f"n_sweep_curves_{HEADLINE_SOURCE}_{HEADLINE_SELECTION}.pdf")
    plot_curves(agg, HEADLINE_SOURCE, HEADLINE_SELECTION, headline_pdf)
    for (source, selection), _ in agg.groupby(["source", "selection"]):
        if source == HEADLINE_SOURCE and selection == HEADLINE_SELECTION:
            continue
        plot_curves(
            agg, source, selection,
            OUT_DIR / f"n_sweep_curves_{source}_{selection}_supplementary.pdf",
        )


if __name__ == "__main__":
    main()
