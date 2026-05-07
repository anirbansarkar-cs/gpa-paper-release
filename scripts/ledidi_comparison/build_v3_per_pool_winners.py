#!/usr/bin/env python3
"""Per-pool winner tables: best DPS GPA + best noDPS GPA + ISM + LEDIDI for
each (cap, pool) cell, with mean/max LN-RC and mean/max AG-RC.

GPA winners picked via Pareto (high AG, low LN) over 16 full-coverage v3 recipes.
ISM single recipe (ism_greedy_115). LEDIDI single recipe per pool (pool_X_l015).
ISM cap 115 is mapped onto the uncap (1000) row as the trajectory's natural endpoint.

Output:
  - results/ledidi_comparison/gpa_vs_ism_ledidi_v2/tables/FINAL_v3_per_pool_winners.md
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

V2_ROOT = Path("results/ledidi_comparison/gpa_vs_ism_ledidi_v2")
OUT_DIR = V2_ROOT / "tables"
HEADLINE = OUT_DIR / "headline_metrics.csv"
V3_METRICS = OUT_DIR / "v3_winners_metrics.csv"

CAP_LABEL = {20: "cap10 (20 ed)", 40: "cap20 (40 ed)", 60: "cap30 (60 ed)",
             100: "cap50 (100 ed)", 1000: "uncap"}
CAPS = [20, 40, 60, 100, 1000]


def pareto_winner(sub: pd.DataFrame) -> pd.Series:
    """Return Pareto-best row (max AG, min LN — sum of ranks)."""
    sub = sub.copy()
    sub["ag_rank"] = sub["mean_ag_rc"].rank(ascending=False, method="min")
    sub["ln_rank"] = sub["mean_ln_rc"].rank(ascending=True,  method="min")
    sub["score"]  = sub["ag_rank"] + sub["ln_rank"]
    return sub.sort_values("score").iloc[0]


def main():
    df = pd.read_csv(HEADLINE)
    v3 = pd.read_csv(V3_METRICS)
    # v3 has only full-coverage recipes; restrict GPA dataset to those
    full_recipes = set(v3["recipe"].unique())
    df_gpa_full = df[(df["method"] == "GPA") & (df["recipe"].isin(full_recipes))]

    md = ["# v3 — per-pool method comparison (low LN, high AG)",
          "",
          "Each row = one method's best variant at that (cap, pool) cell.",
          "DPS GPA + noDPS GPA = Pareto-best v3 recipe (max AG, min LN — sum of ranks).",
          "ISM = `ism_greedy_115` trajectory snapshot at the matched cap (uncap row uses gen 115 endpoint).",
          "LEDIDI = `pool_X_l015` (the only completed production sweep, 5K seeds, l=0.15), edit-distance-binned.",
          "All AG/LN values RC-averaged.",
          ""]

    for pool in ["pool_A", "pool_B"]:
        md.append(f"## {pool}")
        md.append("")
        md.append("| cap | method | recipe | n | mean LN | max LN | mean AG | max AG |")
        md.append("|---|---|---|---:|---:|---:|---:|---:|")

        for cap in CAPS:
            # DPS winner
            sub = df_gpa_full[(df_gpa_full["pool"] == pool) &
                              (df_gpa_full["cap_edits"] == cap)]
            dps_sub = sub[sub["recipe"].isin(
                v3[v3["arm"] == "DPS"]["recipe"].unique()
            )]
            nodps_sub = sub[sub["recipe"].isin(
                v3[v3["arm"] == "noDPS"]["recipe"].unique()
            )]
            if len(dps_sub):
                w = pareto_winner(dps_sub)
                md.append(f"| {CAP_LABEL[cap]} | GPA (DPS) | {w['recipe']} | "
                          f"{int(w['n_seqs'])} | {w['mean_ln_rc']:.3f} | "
                          f"{w['max_ln_rc']:.3f} | {w['mean_ag_rc']:.3f} | "
                          f"{w['max_ag_rc']:.3f} |")
            if len(nodps_sub):
                w = pareto_winner(nodps_sub)
                md.append(f"| {CAP_LABEL[cap]} | GPA (noDPS) | {w['recipe']} | "
                          f"{int(w['n_seqs'])} | {w['mean_ln_rc']:.3f} | "
                          f"{w['max_ln_rc']:.3f} | {w['mean_ag_rc']:.3f} | "
                          f"{w['max_ag_rc']:.3f} |")

            # ISM
            ism_cap = cap if cap < 1000 else 115
            ism = df[(df["method"] == "ISM") &
                     (df["pool"] == pool) &
                     (df["cap_edits"] == ism_cap)]
            if len(ism):
                w = ism.iloc[0]
                tag = f"gen={ism_cap}" + (" (uncap)" if cap == 1000 else "")
                md.append(f"| {CAP_LABEL[cap]} | ISM | ism_greedy ({tag}) | "
                          f"{int(w['n_seqs'])} | {w['mean_ln_rc']:.3f} | "
                          f"{w['max_ln_rc']:.3f} | {w['mean_ag_rc']:.3f} | "
                          f"{w['max_ag_rc']:.3f} |")

            # LEDIDI
            led = df[(df["method"] == "LEDIDI") &
                     (df["pool"] == pool) &
                     (df["recipe"] == f"{pool}_l015") &
                     (df["cap_edits"] == cap)]
            if len(led):
                w = led.iloc[0]
                md.append(f"| {CAP_LABEL[cap]} | LEDIDI | pool_l015 | "
                          f"{int(w['n_seqs'])} | {w['mean_ln_rc']:.3f} | "
                          f"{w['max_ln_rc']:.3f} | {w['mean_ag_rc']:.3f} | "
                          f"{w['max_ag_rc']:.3f} |")
            else:
                md.append(f"| {CAP_LABEL[cap]} | LEDIDI | pool_l015 | — | — | — | — | — |")
            md.append("|  |  |  |  |  |  |  |  |")  # cap separator
        md.append("")

    md_p = OUT_DIR / "FINAL_v3_per_pool_winners.md"
    md_p.write_text("\n".join(md) + "\n")
    print(f"Wrote {md_p}")


if __name__ == "__main__":
    main()
