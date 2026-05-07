#!/usr/bin/env python3
"""Build v3 nobio+nogc DPS vs noDPS comparison table.

Reads gpa_output_pool.h5 for every v3*nobio_nogc* run under gpa_v3/.
Reports mean/max AG (RC) and mean/max LN (RC) per (recipe, pool, cap).
Picks the single winning recipe by mean AG-RC averaged across caps and pools.

Output: results/ledidi_comparison/gpa_vs_ism_ledidi_v2/tables/
        FINAL_v3_nobio_nogc.md  +  v3_nobio_nogc_metrics.csv
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
CAPS_DISPLAY = [20, 40, 60, 100, 1000]   # cap10/20/30/50/uncap → edit budgets
CAP_LABEL = {20: "20 (cap10)", 40: "40 (cap20)", 60: "60 (cap30)",
             100: "100 (cap50)", 1000: "uncap"}

RECIPES = {
    "v3i_dpsKL_nobio_nogc":           dict(dps=True,  kl=0.5, eta=3000, beta=2000, bf=10),
    "v3u_dpsKL_nobio_nogc_b2k_eta500":  dict(dps=True, kl=0.1, eta=500,  beta=2000, bf=10),
    "v3u_dpsKL_nobio_nogc_b2k_eta1000": dict(dps=True, kl=0.1, eta=1000, beta=2000, bf=10),
    "v3x_dpsKL_nobio_nogc_b2k_BF20":  dict(dps=True,  kl=0.1, eta=3000, beta=2000, bf=20),
    "v3s_nodps_nobio_nogc":           dict(dps=False, kl=None, eta=None, beta=1000, bf=10),
    "v3t_nodps_nobio_nogc_b2k":       dict(dps=False, kl=None, eta=None, beta=2000, bf=10),
    "v3w_nodps_nobio_nogc_b2k_BF20":  dict(dps=False, kl=None, eta=None, beta=2000, bf=20),
}


def parse_run_dir(name: str):
    """Return (recipe, pool, cap_edits) or None if not a known v3 nobio_nogc dir."""
    for r in RECIPES:
        m = re.match(rf"^{re.escape(r)}_pool_([AB])_(cap\d+|uncap)$", name)
        if m:
            pool = f"pool_{m.group(1)}"
            tok = m.group(2)
            cap_edits = 1000 if tok == "uncap" else int(tok.replace("cap", "")) * 2
            return r, pool, cap_edits
    return None


def stats_block(h5p: Path) -> dict | None:
    if not h5p.exists():
        return None
    with h5py.File(h5p, "r") as f:
        ag = f.get("ag_k562_scores_jax_v2_rc")
        ln = f.get("ln_k562_scores_v2_rc")
        ag_fwd = f.get("ag_k562_scores_jax_v2")
        ln_fwd = f.get("ln_k562_scores_v2")
        if ag is None or ln is None:
            # Not yet rescored
            return None
        ag = ag[:].astype(np.float64)
        ln = ln[:].astype(np.float64)
        out = dict(
            n=int(len(ag)),
            mean_ag_rc=float(ag.mean()), max_ag_rc=float(ag.max()),
            mean_ln_rc=float(ln.mean()), max_ln_rc=float(ln.max()),
        )
        if ag_fwd is not None:
            ag_fwd = ag_fwd[:].astype(np.float64)
            out["mean_ag_fwd"] = float(ag_fwd.mean())
        if ln_fwd is not None:
            ln_fwd = ln_fwd[:].astype(np.float64)
            out["mean_ln_fwd"] = float(ln_fwd.mean())
    return out


def main():
    rows = []
    skipped = []
    for d in sorted(GPA_V3.iterdir()):
        parsed = parse_run_dir(d.name)
        if parsed is None:
            continue
        recipe, pool, cap_edits = parsed
        s = stats_block(d / "gpa_output_pool.h5")
        if s is None:
            skipped.append(d.name)
            continue
        s.update(dict(recipe=recipe, pool=pool, cap_edits=cap_edits,
                      arm="DPS" if RECIPES[recipe]["dps"] else "noDPS"))
        rows.append(s)
    df = pd.DataFrame(rows)
    if not len(df):
        print("No rows. Has scoring run?")
        return
    print(f"[build_v3_nobio_nogc] {len(df)} rows; {len(skipped)} skipped (unscored)")
    if skipped:
        print("  unscored:")
        for s in skipped: print(f"    {s}")

    csv_p = OUT_DIR / "v3_nobio_nogc_metrics.csv"
    df.sort_values(["arm", "recipe", "pool", "cap_edits"]).to_csv(csv_p, index=False)
    print(f"  wrote {csv_p}")

    # Headline winner: mean AG-RC averaged across all (pool, cap) cells, by recipe
    pivot_ag = (df.pivot_table(index="recipe", values="mean_ag_rc", aggfunc="mean")
                  .sort_values("mean_ag_rc", ascending=False))
    pivot_ag_max = (df.pivot_table(index="recipe", values="max_ag_rc", aggfunc="mean")
                      .sort_values("max_ag_rc", ascending=False))
    print("\nRanking by mean AG-RC (averaged over all caps × pools):")
    print(pivot_ag.to_string())
    print("\nRanking by mean of max AG-RC:")
    print(pivot_ag_max.to_string())

    # Build markdown
    md = ["# v3 nobio+nogc — DPS vs noDPS final table",
          "",
          "Pool A = LentiMPRA random 5K seed pool. Pool B = LentiMPRA-test 5K seed pool.",
          "All values from `gpa_output_pool.h5` (5K AG-ranked best-pool snapshot).",
          "AG = K562 AlphaGenome JAX v2, RC-averaged. LN = K562 LegNet, RC-averaged.",
          ""]

    md.append("## Recipe knob summary")
    md.append("")
    md.append("| recipe | arm | KL budget | DPS η | β | BF |")
    md.append("|---|:---:|:---:|---:|---:|---:|")
    for r, knobs in sorted(RECIPES.items(),
                           key=lambda kv: (kv[1]["dps"], kv[0])):
        md.append("| {r} | {arm} | {kl} | {eta} | {b} | {bf} |".format(
            r=r, arm="DPS" if knobs["dps"] else "noDPS",
            kl="—" if knobs["kl"] is None else knobs["kl"],
            eta="—" if knobs["eta"] is None else knobs["eta"],
            b=knobs["beta"], bf=knobs["bf"]))
    md.append("")

    for pool in sorted(df["pool"].unique()):
        md.append(f"## Pool {pool} — mean AG (RC) by cap")
        md.append("")
        md.append("| recipe | arm | "
                  + " | ".join(CAP_LABEL[c] for c in CAPS_DISPLAY)
                  + " | row mean |")
        md.append("|---|:---:|" + "---:|" * (len(CAPS_DISPLAY) + 1))
        sub = df[df["pool"] == pool]
        # Sort: DPS first then noDPS, then by mean across caps within arm
        recipe_order = (sub.groupby(["arm", "recipe"])["mean_ag_rc"]
                          .mean().reset_index()
                          .sort_values(["arm", "mean_ag_rc"], ascending=[True, False]))
        for _, rr in recipe_order.iterrows():
            r, arm = rr["recipe"], rr["arm"]
            cells = []
            for c in CAPS_DISPLAY:
                v = sub[(sub["recipe"] == r) & (sub["cap_edits"] == c)]
                cells.append(f"{v['mean_ag_rc'].iloc[0]:.3f}" if len(v) else "—")
            row_mean = (sub[sub["recipe"] == r]["mean_ag_rc"].mean())
            md.append(f"| {r} | {arm} | " + " | ".join(cells)
                      + f" | **{row_mean:.3f}** |")
        md.append("")

        md.append(f"### Pool {pool} — max AG (RC) by cap")
        md.append("")
        md.append("| recipe | arm | "
                  + " | ".join(CAP_LABEL[c] for c in CAPS_DISPLAY) + " |")
        md.append("|---|:---:|" + "---:|" * len(CAPS_DISPLAY))
        for _, rr in recipe_order.iterrows():
            r, arm = rr["recipe"], rr["arm"]
            cells = []
            for c in CAPS_DISPLAY:
                v = sub[(sub["recipe"] == r) & (sub["cap_edits"] == c)]
                cells.append(f"{v['max_ag_rc'].iloc[0]:.3f}" if len(v) else "—")
            md.append(f"| {r} | {arm} | " + " | ".join(cells) + " |")
        md.append("")

        md.append(f"### Pool {pool} — mean LN (RC) by cap (LN-hacking diagnostic; lower = less hacking)")
        md.append("")
        md.append("| recipe | arm | "
                  + " | ".join(CAP_LABEL[c] for c in CAPS_DISPLAY) + " |")
        md.append("|---|:---:|" + "---:|" * len(CAPS_DISPLAY))
        for _, rr in recipe_order.iterrows():
            r, arm = rr["recipe"], rr["arm"]
            cells = []
            for c in CAPS_DISPLAY:
                v = sub[(sub["recipe"] == r) & (sub["cap_edits"] == c)]
                cells.append(f"{v['mean_ln_rc'].iloc[0]:.3f}" if len(v) else "—")
            md.append(f"| {r} | {arm} | " + " | ".join(cells) + " |")
        md.append("")

    md.append("## Overall ranking — mean AG (RC), averaged across all caps × both pools")
    md.append("")
    md.append("| rank | recipe | arm | mean AG (RC) | mean of max AG (RC) |")
    md.append("|---:|---|:---:|---:|---:|")
    overall = (df.groupby(["recipe"])
                 .agg(mean_ag_rc=("mean_ag_rc", "mean"),
                      max_ag_rc=("max_ag_rc", "mean"))
                 .sort_values("mean_ag_rc", ascending=False)
                 .reset_index())
    for i, r in overall.iterrows():
        arm = "DPS" if RECIPES[r["recipe"]]["dps"] else "noDPS"
        md.append(f"| {i+1} | {r['recipe']} | {arm} | "
                  f"{r['mean_ag_rc']:.3f} | {r['max_ag_rc']:.3f} |")
    md.append("")

    winner = overall.iloc[0]
    winner_arm = "DPS" if RECIPES[winner["recipe"]]["dps"] else "noDPS"
    md.append(f"## Winner: **`{winner['recipe']}`** ({winner_arm})")
    md.append("")
    md.append(f"- Mean AG-RC across all caps × both pools: **{winner['mean_ag_rc']:.3f}**")
    md.append(f"- Mean of max AG-RC: **{winner['max_ag_rc']:.3f}**")
    knobs = RECIPES[winner["recipe"]]
    md.append(f"- Knobs: arm={winner_arm}, "
              f"KL_budget={knobs['kl']}, DPS_η={knobs['eta']}, "
              f"β={knobs['beta']}, BF={knobs['bf']}")
    md.append("")

    md_p = OUT_DIR / "FINAL_v3_nobio_nogc.md"
    md_p.write_text("\n".join(md) + "\n")
    print(f"  wrote {md_p}")


if __name__ == "__main__":
    main()
