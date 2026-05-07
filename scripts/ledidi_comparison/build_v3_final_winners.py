#!/usr/bin/env python3
"""Build the FINAL v3 DPS-vs-noDPS winner table.

Reads gpa_output_pool.h5 for every v3 run with full (5 caps × 2 pools = 10
runs) coverage. Reports mean/max AG-RC and mean LN-RC per (recipe, pool, cap).

Picks ONE DPS winner and ONE noDPS winner using two criteria:
  - by_ag_minus_ln: rank by mean(AG_RC) - mean(LN_RC) — combined "high AG, low LN"
  - by_pareto_count: count cells where recipe is on the Pareto frontier (high AG, low LN)

Output:
  - results/ledidi_comparison/gpa_vs_ism_ledidi_v2/tables/FINAL_v3_winners.md
  - results/ledidi_comparison/gpa_vs_ism_ledidi_v2/tables/v3_winners_metrics.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

V2_ROOT = Path("results/ledidi_comparison/gpa_vs_ism_ledidi_v2")
GPA_V3 = V2_ROOT / "gpa_v3"
OUT_DIR = V2_ROOT / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL = "k562"
CAPS_DISPLAY = [20, 40, 60, 100, 1000]
CAP_LABEL = {20: "cap10 (20 ed)", 40: "cap20 (40 ed)", 60: "cap30 (60 ed)",
             100: "cap50 (100 ed)", 1000: "uncap"}


def parse_run_dir(name: str):
    m = re.match(r"^(.+?)_pool_([AB])_(cap\d+|uncap)$", name)
    if not m:
        return None
    recipe = m.group(1)
    pool = f"pool_{m.group(2)}"
    tok = m.group(3)
    cap_edits = 1000 if tok == "uncap" else int(tok.replace("cap", "")) * 2
    return recipe, pool, cap_edits


def stats_block(h5p: Path) -> dict | None:
    if not h5p.exists():
        return None
    with h5py.File(h5p, "r") as f:
        ag = f.get("ag_k562_scores_jax_v2_rc")
        ln = f.get("ln_k562_scores_v2_rc")
        if ag is None or ln is None:
            return None
        ag = ag[:].astype(np.float64)
        ln = ln[:].astype(np.float64)
    return dict(n=int(len(ag)),
                mean_ag=float(ag.mean()), max_ag=float(ag.max()),
                mean_ln=float(ln.mean()))


def is_dps_recipe(recipe: str, run_dir: Path) -> bool:
    """Read run config to determine DPS arm."""
    h = run_dir / "gpa_history.json"
    if h.exists():
        try:
            import json
            return bool(json.load(open(h)).get("args", {}).get("use_dps", False))
        except Exception:
            pass
    # Fallback: name-based
    return ("nodps" not in recipe) and ("dpsKL" in recipe or "dps" in recipe.lower()
                                         or "v3a" in recipe or "v3j" in recipe
                                         or "v3k" in recipe or "v3l" in recipe
                                         or "v3m" in recipe)


def main():
    rows = []
    for d in sorted(GPA_V3.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_run_dir(d.name)
        if parsed is None:
            continue
        recipe, pool, cap = parsed
        s = stats_block(d / "gpa_output_pool.h5")
        if s is None:
            continue
        s.update(dict(recipe=recipe, pool=pool, cap_edits=cap,
                      arm="DPS" if is_dps_recipe(recipe, d) else "noDPS",
                      run_dir=str(d)))
        rows.append(s)
    df = pd.DataFrame(rows)
    if not len(df):
        print("No scored runs.")
        return

    # Restrict to recipes with full coverage (5 caps × 2 pools = 10 runs)
    cov = df.groupby("recipe")["cap_edits"].count()
    full = cov[cov == 10].index.tolist()
    partial = cov[cov < 10].index.tolist()
    print(f"Recipes with full coverage (n=10): {len(full)}")
    for r in full: print(f"  {r}  arm={df[df.recipe==r].arm.iloc[0]}")
    print(f"Partial-coverage recipes: {len(partial)}")
    for r in partial:
        n = cov.loc[r]
        print(f"  {r}: n={n}")

    df_full = df[df["recipe"].isin(full)].copy()

    # Per-recipe overall stats (averaged across all 10 cells)
    agg = (df_full.groupby(["recipe","arm"])
           .agg(mean_ag=("mean_ag","mean"),
                max_ag=("max_ag","mean"),
                mean_ln=("mean_ln","mean"))
           .reset_index())
    agg["ag_minus_ln"] = agg["mean_ag"] - agg["mean_ln"]

    # Pareto-cell count per recipe (counted across the 10 cells)
    pareto_counts = {r: 0 for r in full}
    for cap in CAPS_DISPLAY:
        for pool in ["pool_A","pool_B"]:
            sub = df_full[(df_full["cap_edits"]==cap) & (df_full["pool"]==pool)]
            if not len(sub): continue
            for _, r in sub.iterrows():
                dom = False
                for _, s in sub.iterrows():
                    if s.recipe == r.recipe: continue
                    if (s.mean_ag >= r.mean_ag and s.mean_ln <= r.mean_ln and
                        (s.mean_ag > r.mean_ag or s.mean_ln < r.mean_ln)):
                        dom = True; break
                if not dom:
                    pareto_counts[r.recipe] += 1
    agg["pareto_cells"] = agg["recipe"].map(pareto_counts)

    # Pick winners per arm
    dps_rank   = agg[agg["arm"]=="DPS"].sort_values(["pareto_cells","ag_minus_ln"], ascending=[False, False])
    nodps_rank = agg[agg["arm"]=="noDPS"].sort_values(["pareto_cells","ag_minus_ln"], ascending=[False, False])

    print("\n=== DPS recipes ranked (full coverage, by Pareto-cells then AG-LN) ===")
    print(dps_rank.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== noDPS recipes ranked ===")
    print(nodps_rank.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    dps_winner   = dps_rank.iloc[0]["recipe"]   if len(dps_rank)   else None
    nodps_winner = nodps_rank.iloc[0]["recipe"] if len(nodps_rank) else None

    # Save CSV
    csv_p = OUT_DIR / "v3_winners_metrics.csv"
    df_full.sort_values(["arm","recipe","pool","cap_edits"]).to_csv(csv_p, index=False)
    print(f"\nWrote {csv_p}")

    # Markdown final
    md = ["# v3 final winners — best DPS + best noDPS recipes (low LN, high AG)",
          "",
          "All values from `gpa_output_pool.h5` (5K AG-ranked best-pool snapshot).",
          "AG = K562 AlphaGenome JAX v2, RC-averaged. LN = K562 LegNet, RC-averaged.",
          ""]

    md.append("## Selection criteria")
    md.append("- Primary: count of Pareto-frontier cells (a recipe is Pareto on a "
              "(cap, pool) cell if no other full-coverage recipe has both higher "
              "AG and lower LN there).")
    md.append("- Tiebreak: mean(AG-RC) − mean(LN-RC) over all 10 cells.")
    md.append("")
    md.append(f"## DPS arm winner: **`{dps_winner}`**")
    md.append("")
    md.append(f"## noDPS arm winner: **`{nodps_winner}`**")
    md.append("")

    md.append("## Full-coverage recipe ranking")
    md.append("")
    md.append("| arm | recipe | mean AG-RC | max AG-RC | mean LN-RC | AG−LN | Pareto cells |")
    md.append("|:--:|---|---:|---:|---:|---:|---:|")
    rank_all = pd.concat([dps_rank, nodps_rank])
    for _, r in rank_all.iterrows():
        md.append(f"| {r['arm']} | {r['recipe']} | {r['mean_ag']:.3f} | "
                  f"{r['max_ag']:.3f} | {r['mean_ln']:.3f} | "
                  f"{r['ag_minus_ln']:.3f} | {int(r['pareto_cells'])}/10 |")
    md.append("")

    # Per-recipe per-cell tables for the 2 winners
    for winner_label, recipe in [("DPS winner", dps_winner),
                                  ("noDPS winner", nodps_winner)]:
        if recipe is None: continue
        sub = df_full[df_full["recipe"] == recipe]
        md.append(f"## {winner_label}: `{recipe}` — per cell")
        md.append("")
        md.append("| pool | metric | " + " | ".join(CAP_LABEL[c] for c in CAPS_DISPLAY) + " |")
        md.append("|---|---|" + "---:|" * len(CAPS_DISPLAY))
        for pool in ["pool_A","pool_B"]:
            for metric, key, fmt in [("mean AG-RC", "mean_ag", "{:.3f}"),
                                      ("max AG-RC",  "max_ag",  "{:.3f}"),
                                      ("mean LN-RC", "mean_ln", "{:.3f}")]:
                vals = []
                for c in CAPS_DISPLAY:
                    v = sub[(sub["pool"]==pool) & (sub["cap_edits"]==c)]
                    vals.append(fmt.format(v[key].iloc[0]) if len(v) else "—")
                md.append(f"| {pool} | {metric} | " + " | ".join(vals) + " |")
        md.append("")

    md.append("## Partial-coverage recipes (excluded from winner pick)")
    md.append("")
    md.append("| recipe | n_cells | tested caps |")
    md.append("|---|---:|---|")
    for r in partial:
        sub = df[df["recipe"]==r]
        caps_seen = sorted(sub["cap_edits"].unique().tolist())
        md.append(f"| {r} | {len(sub)} | {caps_seen} |")
    md.append("")

    md_p = OUT_DIR / "FINAL_v3_winners.md"
    md_p.write_text("\n".join(md) + "\n")
    print(f"Wrote {md_p}")


if __name__ == "__main__":
    main()
