#!/usr/bin/env python3
"""Per-cell head-to-head: ucb_stacked i=12 vs i=18 (K=8, d=2).

Inputs (per_pool_jsons/):
  i=12: dnacraft_pilot_ucb_stacked_{cell_lc}_s{seed}__{Cell}__{selection}.json   (5 seeds × 3 cells)
  i=18: dnacraft_ucb_iter_i18_K8_d2_{cell_lc}_s{seed}__{Cell}__{selection}.json  (5 seeds × 3 cells)

Matched-cell only (cell_lc must equal target_cell).

Decision rule for headline swap (per user instruction):
  i=18 wins on a cell ⇔ ΔDiversity > 0  AND  |ΔMinGap| ≤ flat_thresh.
  Default flat_thresh = max(combined std, 0.20).

Outputs:
  results/rerd_comparison/dnacraft/dnacraft_i12_vs_i18.md
  results/rerd_comparison/dnacraft/dnacraft_i12_vs_i18.csv
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("${GPA_REPO_ROOT}")
JSON_DIR = PROJECT / "results/rerd_comparison/dnacraft/per_pool_jsons"
OUT_MD = PROJECT / "results/rerd_comparison/dnacraft/dnacraft_i12_vs_i18.md"
OUT_CSV = PROJECT / "results/rerd_comparison/dnacraft/dnacraft_i12_vs_i18.csv"

CELL_LC_TO_CANON = {"hepg2": "HepG2", "k562": "K562", "sknsh": "SKNSH"}
CELL_ORDER = ["HepG2", "K562", "SKNSH"]
SELECTIONS = ["by_composite", "by_mingap"]

I12_RE = re.compile(r"^dnacraft_pilot_ucb_stacked_(?P<cell>hepg2|k562|sknsh)_s(?P<seed>\d+)$")
I18_RE = re.compile(r"^dnacraft_ucb_iter_i18_K8_d2_(?P<cell>hepg2|k562|sknsh)_s(?P<seed>\d+)$")


def label_method(pool_name):
    """Return (i_value, cell_canon, seed) or (None, None, None)."""
    m = I12_RE.match(pool_name)
    if m:
        return 12, CELL_LC_TO_CANON[m.group("cell")], int(m.group("seed"))
    m = I18_RE.match(pool_name)
    if m:
        return 18, CELL_LC_TO_CANON[m.group("cell")], int(m.group("seed"))
    return None, None, None


def load_long():
    rows = []
    for jp in sorted(JSON_DIR.glob("*.json")):
        try:
            d = json.loads(jp.read_text())
        except Exception:
            continue
        i_val, pool_cell, seed = label_method(d["pool_name"])
        if i_val is None:
            continue
        # Matched-cell only
        if pool_cell != d["target_cell"]:
            continue
        rows.append({
            "i": i_val,
            "cell": pool_cell,
            "seed": seed,
            "selection": d["selection"],
            "pool_name": d["pool_name"],
            "mingap_eval": d["mingap_eval_mean"],
            "motif_spr": d["motif_corr_spearman"],
            "kmer3_pearson": d["kmer3_corr_pearson"],
            "diversity_bits": d["diversity_bits"],
        })
    return pd.DataFrame(rows)


def aggregate(long):
    return long.groupby(["i", "cell", "selection"]).agg(
        n_seeds=("seed", "count"),
        mingap_mean=("mingap_eval", "mean"),
        mingap_std=("mingap_eval", "std"),
        motif_mean=("motif_spr", "mean"),
        motif_std=("motif_spr", "std"),
        kmer_mean=("kmer3_pearson", "mean"),
        kmer_std=("kmer3_pearson", "std"),
        div_mean=("diversity_bits", "mean"),
        div_std=("diversity_bits", "std"),
    ).reset_index()


def fmt(mean, std=None, dec=3, sign=False):
    if pd.isna(mean):
        return "—"
    base = f"{mean:+.{dec}f}" if sign else f"{mean:.{dec}f}"
    if std is None or pd.isna(std):
        return base
    return f"{base}±{std:.{dec}f}"


def fmt_delta(d, dec=3):
    if pd.isna(d):
        return "—"
    return f"{d:+.{dec}f}"


def decide_swap(i12_row, i18_row, flat_thresh_floor=0.20):
    """Return (verdict_str, delta_dict). Verdict in {win, tie, loss}.

    win  : ΔDiv > 0 AND |ΔMinGap| ≤ flat_thresh
    tie  : ΔDiv approx 0 OR  ΔMinGap ambiguous
    loss : ΔDiv ≤ 0  OR  |ΔMinGap| > flat_thresh against i=18
    """
    d_div = i18_row["div_mean"] - i12_row["div_mean"]
    d_mg = i18_row["mingap_mean"] - i12_row["mingap_mean"]
    d_motif = i18_row["motif_mean"] - i12_row["motif_mean"]
    d_kmer = i18_row["kmer_mean"] - i12_row["kmer_mean"]

    combined_mg_std = float(np.sqrt(
        (i18_row.get("mingap_std", np.nan) or 0) ** 2
        + (i12_row.get("mingap_std", np.nan) or 0) ** 2
    ))
    flat_thresh = max(combined_mg_std, flat_thresh_floor)

    if d_div > 0 and abs(d_mg) <= flat_thresh:
        verdict = "WIN"
    elif d_div <= 0 and d_mg < -flat_thresh:
        verdict = "LOSS"
    else:
        verdict = "MIXED"

    return verdict, {
        "d_mingap": d_mg, "d_motif": d_motif, "d_kmer": d_kmer, "d_div": d_div,
        "flat_thresh": flat_thresh,
    }


def render_md(agg, long):
    md = []
    md.append("# DNA-CRAFT MCTS Pilot — i=12 vs i=18 (K=8, d=2) head-to-head\n")
    md.append("Pool source: `gpa_output_pool.h5`, top-128, matched-conditioning-cell only.")
    md.append("- i=12 = original `dnacraft_pilot_ucb_stacked_*` (5 seeds/cell)")
    md.append("- i=18 = `dnacraft_ucb_iter_i18_K8_d2_*` (5 seeds/cell — 3 fresh + 2 from K562 scaling sweep)")
    md.append("")
    md.append("**Headline-swap rule**: per cell, i=18 wins ⇔ ΔDiversity > 0 AND |ΔMinGap| ≤ flat_thresh "
              "(flat_thresh = max(combined std, 0.20)). Swap headline only if i=18 holds K562 win on **HepG2 + SK-N-SH**.")
    md.append("")

    swap_summary = {sel: {} for sel in SELECTIONS}

    for selection in SELECTIONS:
        md.append(f"## Selection: {selection}\n")
        for cell in CELL_ORDER:
            sub = agg[(agg["selection"] == selection) & (agg["cell"] == cell)]
            i12 = sub[sub["i"] == 12]
            i18 = sub[sub["i"] == 18]
            if i12.empty or i18.empty:
                md.append(f"### {cell} — selection={selection}\n")
                md.append(f"_(missing data: i=12 rows={len(i12)}, i=18 rows={len(i18)})_\n")
                swap_summary[selection][cell] = "MISSING"
                continue
            i12_r = i12.iloc[0]
            i18_r = i18.iloc[0]
            verdict, deltas = decide_swap(i12_r, i18_r)
            swap_summary[selection][cell] = verdict

            md.append(f"### {cell} — selection={selection}  →  **{verdict}**\n")
            md.append("| i | n | MinGap Eval ↑ | Motif Spr ↑ | 3-mer Pearson ↑ | Diversity (bits) ↑ |")
            md.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for r in (i12_r, i18_r):
                md.append("| " + " | ".join([
                    f"i={int(r['i'])}",
                    str(int(r["n_seeds"])),
                    fmt(r["mingap_mean"], r["mingap_std"], dec=3, sign=True),
                    fmt(r["motif_mean"],  r["motif_std"],  dec=3),
                    fmt(r["kmer_mean"],   r["kmer_std"],   dec=3),
                    fmt(r["div_mean"],    r["div_std"],    dec=3),
                ]) + " |")
            md.append("| **Δ (i=18 − i=12)** | — | "
                      f"{fmt_delta(deltas['d_mingap'])} | "
                      f"{fmt_delta(deltas['d_motif'])} | "
                      f"{fmt_delta(deltas['d_kmer'])} | "
                      f"{fmt_delta(deltas['d_div'])} |")
            md.append(f"\n_flat_thresh (MinGap) = {deltas['flat_thresh']:.3f}; "
                      f"ΔDiv = {deltas['d_div']:+.3f}; ΔMinGap = {deltas['d_mingap']:+.3f}_\n")

    # Decision summary
    md.append("---\n")
    md.append("## Decision summary\n")
    md.append("| selection | HepG2 | K562 | SK-N-SH | Headline swap? |")
    md.append("| --- | :---: | :---: | :---: | :---: |")
    for selection in SELECTIONS:
        h = swap_summary[selection].get("HepG2", "—")
        k = swap_summary[selection].get("K562", "—")
        s = swap_summary[selection].get("SKNSH", "—")
        # Swap rule from user: K562 win must hold on HepG2 + SK-N-SH
        non_k562_wins = sum(1 for v in (h, s) if v == "WIN")
        if k == "WIN" and non_k562_wins == 2:
            swap_str = "**YES** (full)"
        elif k == "WIN" and non_k562_wins == 1:
            swap_str = "PARTIAL"
        else:
            swap_str = "no"
        md.append(f"| {selection} | {h} | {k} | {s} | {swap_str} |")
    md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")
    return swap_summary


def main():
    long = load_long()
    print(f"Loaded {len(long)} matched-cell rows from {JSON_DIR}")
    if long.empty:
        print("No data — scoring may not be complete yet.")
        return
    agg = aggregate(long)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"Wrote {OUT_CSV} ({len(agg)} aggregate rows)")
    swap = render_md(agg, long)
    print()
    print(OUT_MD.read_text())
    print()
    print("Decision summary:", json.dumps(swap, indent=2))


if __name__ == "__main__":
    main()
