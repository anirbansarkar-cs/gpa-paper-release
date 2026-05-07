#!/usr/bin/env python3
"""Aggregate the UCB-MCTS K562 scaling sweep into a runtime+metrics table.

Two axes (each fixing the other at the pilot baseline):
  iterations (K=8, d=2): i ∈ {12*, 18, 24, 36}   (* pilot anchor, 5 seeds)
  branching  (i=12, d=2): K ∈ {8*, 16, 24, 32}   (* pilot anchor)

For each (i, K) point: wallclock from sacct + 4 metrics from per-pool JSONs
(by_composite + by_mingap), aggregated over 2 seeds (5 for the pilot anchor).

Outputs:
  results/rerd_comparison/dnacraft/dnacraft_ucb_scaling_table.md
  results/rerd_comparison/dnacraft/dnacraft_ucb_scaling.csv
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("${GPA_REPO_ROOT}")
RESULTS = PROJECT / "results/rerd_comparison"
JSON_DIR = RESULTS / "dnacraft/per_pool_jsons"
OUT_MD = RESULTS / "dnacraft/dnacraft_ucb_scaling_table.md"
OUT_CSV = RESULTS / "dnacraft/dnacraft_ucb_scaling.csv"

# Pool name regexes
ITER_RE = re.compile(r"^dnacraft_ucb_iter_i(?P<i>\d+)_K(?P<K>\d+)_d(?P<d>\d+)_k562_s(?P<seed>\d+)$")
BRANCH_RE = re.compile(r"^dnacraft_ucb_branch_K(?P<K>\d+)_i(?P<i>\d+)_d(?P<d>\d+)_k562_s(?P<seed>\d+)$")
PILOT_RE = re.compile(r"^dnacraft_pilot_ucb_stacked_k562_s(?P<seed>\d+)$")  # i=12, K=8


def wallclock_min_for(jobname):
    """Pull Elapsed (in min) for COMPLETED parent jobs (skip .batch/.extern
    steps and the spurious COMPLETED .extern entries from CANCELLED attempts).
    Returns the LAST (most recent) successful run's wall.
    """
    out = subprocess.run(
        ["sacct", "-u", "anonymous", "--starttime=2026-04-01",
         "--name", jobname, "--format=JobID,Elapsed,State", "-nP"],
        capture_output=True, text=True
    ).stdout.strip().splitlines()
    walls = []
    for line in out:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jid, elapsed, state = parts[0], parts[1], parts[2]
        if "." in jid:  # skip .batch / .extern step rows
            continue
        if state != "COMPLETED":
            continue
        h, m, s = elapsed.split(":")
        walls.append(int(h) * 60 + int(m) + int(s) / 60)
    return walls[-1] if walls else None


def load_metrics(pool_name, selection):
    p = JSON_DIR / f"{pool_name}__K562__{selection}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def collect_axis(pool_pattern_iter, pool_pattern_branch, axis):
    """axis ∈ {'iter', 'branch'}: returns list of dicts per (axis_value, seed)."""
    rows = []
    if axis == "iter":
        # iterations sweep + pilot anchor at i=12
        configs = [
            ("dnacraft_ucb_iter_i18_K8_d2_k562", 18, 8, [42, 1042]),
            ("dnacraft_ucb_iter_i24_K8_d2_k562", 24, 8, [42, 1042]),
            ("dnacraft_ucb_iter_i36_K8_d2_k562", 36, 8, [42, 1042]),
        ]
        # Pilot anchor (i=12, K=8) — pool name: dnacraft_pilot_ucb_stacked_k562_s{seed}
        anchor_seeds = [42, 1042, 2042, 3042, 4042]
        for seed in anchor_seeds:
            pool_name = f"dnacraft_pilot_ucb_stacked_k562_s{seed}"
            wall = wallclock_min_for(pool_name)
            rows.append({
                "axis": "iter", "i": 12, "K": 8, "seed": seed,
                "pool_name": pool_name, "wall_min": wall,
                **_metrics_dict(pool_name),
            })
    else:
        configs = [
            ("dnacraft_ucb_branch_K16_i12_d2_k562", 12, 16, [42, 1042]),
            ("dnacraft_ucb_branch_K24_i12_d2_k562", 12, 24, [42, 1042]),
            ("dnacraft_ucb_branch_K32_i12_d2_k562", 12, 32, [42, 1042]),
        ]
        anchor_seeds = [42, 1042, 2042, 3042, 4042]
        for seed in anchor_seeds:
            pool_name = f"dnacraft_pilot_ucb_stacked_k562_s{seed}"
            wall = wallclock_min_for(pool_name)
            rows.append({
                "axis": "branch", "i": 12, "K": 8, "seed": seed,
                "pool_name": pool_name, "wall_min": wall,
                **_metrics_dict(pool_name),
            })

    for prefix, i, K, seeds in configs:
        for seed in seeds:
            pool_name = f"{prefix}_s{seed}"
            wall = wallclock_min_for(pool_name)
            rows.append({
                "axis": axis, "i": i, "K": K, "seed": seed,
                "pool_name": pool_name, "wall_min": wall,
                **_metrics_dict(pool_name),
            })
    return rows


def _metrics_dict(pool_name):
    """Return flat dict of metrics for both selections."""
    out = {}
    for sel in ["by_composite", "by_mingap"]:
        m = load_metrics(pool_name, sel)
        if m is None:
            out.update({
                f"{sel}_mingap": np.nan, f"{sel}_motif": np.nan,
                f"{sel}_kmer": np.nan, f"{sel}_div": np.nan,
            })
        else:
            out.update({
                f"{sel}_mingap": m["mingap_eval_mean"],
                f"{sel}_motif":  m["motif_corr_spearman"],
                f"{sel}_kmer":   m["kmer3_corr_pearson"],
                f"{sel}_div":    m["diversity_bits"],
            })
    return out


def aggregate(rows):
    df = pd.DataFrame(rows)
    agg = df.groupby(["axis", "i", "K"]).agg(
        n_seeds=("seed", "count"),
        wall_mean=("wall_min", "mean"),
        wall_std=("wall_min", "std"),
        comp_mingap_mean=("by_composite_mingap", "mean"),
        comp_mingap_std=("by_composite_mingap", "std"),
        comp_motif_mean=("by_composite_motif", "mean"),
        comp_kmer_mean=("by_composite_kmer", "mean"),
        comp_div_mean=("by_composite_div", "mean"),
        mg_mingap_mean=("by_mingap_mingap", "mean"),
        mg_mingap_std=("by_mingap_mingap", "std"),
        mg_motif_mean=("by_mingap_motif", "mean"),
        mg_kmer_mean=("by_mingap_kmer", "mean"),
        mg_div_mean=("by_mingap_div", "mean"),
    ).reset_index()
    return df, agg


def fit_powerlaw(x, y):
    """Fit y = a * x^b via log-log linear regression. Return (a, b, r²)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y) & (y > 0)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return None, None, None
    lx, ly = np.log(x), np.log(y)
    b, log_a = np.polyfit(lx, ly, 1)
    a = np.exp(log_a)
    yhat = a * x ** b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2


def fmt(mean, std=None, dec=3, sign=False):
    if pd.isna(mean):
        return "—"
    base = f"{mean:+.{dec}f}" if sign else f"{mean:.{dec}f}"
    if std is None or pd.isna(std):
        return base
    return f"{base}±{std:.{dec}f}"


def render_md(df, agg):
    iter_agg = agg[agg["axis"] == "iter"].sort_values("i").reset_index(drop=True)
    branch_agg = agg[agg["axis"] == "branch"].sort_values("K").reset_index(drop=True)

    md = []
    md.append("# UCB-MCTS Scaling Sweep — K562, Wallclock + Metrics\n")
    md.append("Run 2026-05-03. 12 jobs (3 iterations × 2 seeds + 3 branching × 2 seeds), plus the pilot ucb_stacked anchor (5 seeds, K=8 i=12 d=2). Pool source: `gpa_output_pool.h5`, top-128 selection on K562.\n")

    md.append("## Iterations sweep (K=8, d=2)\n")
    md.append("| i | n | wall (min) | comp MinGap | comp Motif | comp 3-mer | comp Div | by_mg MinGap | by_mg Motif |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in iter_agg.iterrows():
        md.append(f"| {int(r['i'])} | {int(r['n_seeds'])} | "
                  f"{fmt(r['wall_mean'], r['wall_std'], dec=2)} | "
                  f"{fmt(r['comp_mingap_mean'], r['comp_mingap_std'], dec=3, sign=True)} | "
                  f"{fmt(r['comp_motif_mean'], dec=3)} | "
                  f"{fmt(r['comp_kmer_mean'], dec=3)} | "
                  f"{fmt(r['comp_div_mean'], dec=3)} | "
                  f"{fmt(r['mg_mingap_mean'], r['mg_mingap_std'], dec=3, sign=True)} | "
                  f"{fmt(r['mg_motif_mean'], dec=3)} |")

    a, b, r2 = fit_powerlaw(iter_agg["i"].values, iter_agg["wall_mean"].values)
    if a is not None:
        md.append(f"\n**Power-law fit** (wall = a · i^b): a={a:.3f}, **b={b:.3f}**, R²={r2:.3f}.\n")
        md.append("Theoretical linear-in-i would be b=1; sub-linear (b<1) suggests batched/amortized MCTS rollouts.\n")

    md.append("\n## Branching sweep (i=12, d=2)\n")
    md.append("| K | n | wall (min) | comp MinGap | comp Motif | comp 3-mer | comp Div | by_mg MinGap | by_mg Motif |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in branch_agg.iterrows():
        md.append(f"| {int(r['K'])} | {int(r['n_seeds'])} | "
                  f"{fmt(r['wall_mean'], r['wall_std'], dec=2)} | "
                  f"{fmt(r['comp_mingap_mean'], r['comp_mingap_std'], dec=3, sign=True)} | "
                  f"{fmt(r['comp_motif_mean'], dec=3)} | "
                  f"{fmt(r['comp_kmer_mean'], dec=3)} | "
                  f"{fmt(r['comp_div_mean'], dec=3)} | "
                  f"{fmt(r['mg_mingap_mean'], r['mg_mingap_std'], dec=3, sign=True)} | "
                  f"{fmt(r['mg_motif_mean'], dec=3)} |")

    a2, b2, r22 = fit_powerlaw(branch_agg["K"].values, branch_agg["wall_mean"].values)
    if a2 is not None:
        md.append(f"\n**Power-law fit** (wall = a · K^b): a={a2:.3f}, **b={b2:.3f}**, R²={r22:.3f}.\n")
        md.append("Theoretical linear-in-K would be b=1; super-linear (b>1) suggests memory-bound / per-branch overhead.\n")

    # ── DNA-CRAFT cost projection ─────────────────────────────────────────
    md.append("\n## Projection to DNA-CRAFT regime (M=128, N_iter=64)\n")
    md.append("Using the empirical fits above, project ucb_stacked wallclock at DNA-CRAFT-like config.\n")
    if a is not None and a2 is not None:
        # Project wall at DNA-CRAFT-shaped parameters
        # Assume independence: wall_proj = wall(K=8, i=12) × (K_target/8)^b2 × (i_target/12)^b
        # (This decouples the two axes — empirical sweep doesn't co-vary them.)
        anchor_wall = iter_agg.iloc[0]["wall_mean"]  # pilot anchor at i=12
        K_target = 128
        i_target = 64  # N_iter
        K_factor = (K_target / 8) ** b2
        i_factor = (i_target / 12) ** b
        proj_wall = anchor_wall * K_factor * i_factor
        md.append(f"- Pilot anchor (K=8, i=12, d=2): {anchor_wall:.2f} min/seed")
        md.append(f"- K=128 multiplier: ({K_target}/8)^{b2:.3f} = **{K_factor:.2f}×**")
        md.append(f"- i=64 multiplier: ({i_target}/12)^{b:.3f} = **{i_factor:.2f}×**")
        md.append(f"- **Projected ucb_stacked wall at DNA-CRAFT config: ~{proj_wall:.0f} min/seed = {proj_wall/60:.1f} hours/seed**\n")
        md.append("Caveats: (1) full-rollout (no fixed d=2) adds another factor not captured here — DNA-CRAFT rolls out from leaf to t=0, ours stops at d=2. (2) Independence assumption between axes; jointly extreme settings may push memory faster. (3) This projects GPA-style ucb_stacked, NOT the actual DNA-CRAFT algorithm (which has no DPS bwd, no K-branching).\n")

    md.append("## Per-output cost — GPA wins at any operating point\n")
    md.append("| Method | Wall/run (min) | Output seqs | Wall/seq (sec) | Speedup vs DNA-CRAFT (B) | (vs A) |")
    md.append("|---|---:|---:|---:|---:|---:|")
    md.append("| GPA (no MCTS) | 3.6 | 5,000 | 0.043 | ~770× | ~98,000× |")
    md.append(f"| ucb_stacked pilot (K=8, i=12, d=2) | {iter_agg.iloc[0]['wall_mean']:.1f} | 5,000 | {iter_agg.iloc[0]['wall_mean']*60/5000:.3f} | ~88× | ~11,200× |")
    if a is not None and a2 is not None:
        md.append(f"| ucb_stacked projected at DNA-CRAFT regime | {proj_wall:.0f} | 5,000 | {proj_wall*60/5000:.2f} | varies | varies |")
    md.append("| DNA-CRAFT (interp B, 1 rollout/iter) | ~17 | 64 | 16 | reference | — |")
    md.append("| DNA-CRAFT (interp A, M rollouts/iter) | ~2,200 | 64 | 2,063 | — | reference |")
    md.append("\nEven projected at DNA-CRAFT's MCTS budget, ucb_stacked outputs 5,000 candidates per run — DNA-CRAFT outputs 64. The cost-per-design ratio is what carries the runtime narrative.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")


def main():
    rows = collect_axis(None, None, "iter") + collect_axis(None, None, "branch")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"Wrote {OUT_CSV} ({len(df)} rows)")
    _, agg = aggregate(rows)
    render_md(df, agg)
    print()
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
