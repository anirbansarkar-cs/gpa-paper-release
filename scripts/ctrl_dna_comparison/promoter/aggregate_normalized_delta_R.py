#!/usr/bin/env python3
"""Compute normalized ΔR (paper Table 1 scale) for CtrlDNA and GPA promoter pools.

Per-cell normalization to [0,1] using oracle_ranges.json (training-data activity range).
ΔR_norm = target_norm − mean(off_norm)  per pool, then aggregated mean ± std over seeds.

Outputs both raw-scale and normalized side-by-side.

CtrlDNA: per-seed pool means from eval_summary_fair_with_specificity.csv
GPA:     per-pool per-selection from targeted_per_pool_selection.csv (universal recipe)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("${GPA_REPO_ROOT}/scripts/ctrl_dna_comparison/promoter/data")
RESULTS = Path("${GPA_REPO_ROOT}/results/ctrl_dna_comparison/promoter")
ORACLE_RANGES = json.loads((DATA_DIR / "oracle_ranges.json").read_text())
CELLS = ["JURKAT", "K562", "THP1"]


def normalize(value, cell):
    r = ORACLE_RANGES[cell]
    return (value - r["min"]) / (r["max"] - r["min"])


def per_seed_delta_R_raw_norm(jurkat, k562, thp1, target_cell):
    """Compute (raw, normalized) ΔR for one seed given its 3 cell-mean values."""
    raw = {"JURKAT": jurkat, "K562": k562, "THP1": thp1}
    norm = {c: normalize(raw[c], c) for c in CELLS}
    off = [c for c in CELLS if c != target_cell]
    delta_R_raw  = raw[target_cell]  - np.mean([raw[c]  for c in off])
    delta_R_norm = norm[target_cell] - np.mean([norm[c] for c in off])
    return float(delta_R_raw), float(delta_R_norm), float(raw[target_cell]), float(norm[target_cell])


def aggregate_ctrldna():
    df = pd.read_csv(RESULTS / "eval_summary_fair_with_specificity.csv")
    df = df[(df["method"] == "ctrldna") & (df["selection"] == "top128_by_reward") & (df["suffix"] == "e15")]
    rows = []
    for cell in CELLS:
        sub = df[df["task"] == cell]
        for _, r in sub.iterrows():
            dR_raw, dR_norm, t_raw, t_norm = per_seed_delta_R_raw_norm(
                r["jurkat"], r["k562"], r["thp1"], cell)
            rows.append({"method": "CtrlDNA", "cell": cell, "seed": int(r["seed"]),
                         "target_raw": t_raw, "target_norm": t_norm,
                         "delta_R_raw": dR_raw, "delta_R_norm": dR_norm})
    return pd.DataFrame(rows)


def aggregate_gpa():
    src = RESULTS / "targeted_per_pool_selection.csv"
    if not src.exists():
        return pd.DataFrame()
    df = pd.read_csv(src)
    # Universal recipe = nodpsA_K8_div1p0 (or argmax_K8_nodpsA_JK for JURKAT)
    universal = {"JURKAT": "nodpsA_K8_div1p0", "K562": "nodpsA_K8_div1p0", "THP1": "nodpsA_K8_div1p0"}
    sel = "top128_by_target"
    rows = []
    for cell in CELLS:
        recipe = universal[cell]
        sub = df[(df["recipe"] == recipe) & (df["cell"] == cell) & (df["selection"] == sel)]
        for _, r in sub.iterrows():
            dR_raw, dR_norm, t_raw, t_norm = per_seed_delta_R_raw_norm(
                r["jurkat"], r["k562"], r["thp1"], cell)
            rows.append({"method": "GPA_universal", "cell": cell, "seed": int(r["seed"]),
                         "target_raw": t_raw, "target_norm": t_norm,
                         "delta_R_raw": dR_raw, "delta_R_norm": dR_norm})
    return pd.DataFrame(rows)


def main():
    ctrl = aggregate_ctrldna()
    gpa  = aggregate_gpa()
    df = pd.concat([ctrl, gpa], ignore_index=True) if len(gpa) else ctrl

    out_raw = RESULTS / "normalized_delta_R_per_seed.csv"
    df.to_csv(out_raw, index=False, float_format="%.4f")

    print("\n" + "=" * 96)
    print("ΔR — both raw oracle scale AND normalized [0,1] scale (paper Table 1)")
    print("=" * 96)
    fmt = lambda v: f"{v:+.3f}"
    for cell in CELLS:
        print(f"\n--- {cell} (off-targets = {[c for c in CELLS if c != cell]}) ---")
        for method in ("CtrlDNA", "GPA_universal"):
            sub = df[(df["method"] == method) & (df["cell"] == cell)]
            if len(sub) == 0:
                print(f"  {method:<14}  (no data)")
                continue
            tr  = sub["target_raw"]
            tn  = sub["target_norm"]
            dR  = sub["delta_R_raw"]
            dRn = sub["delta_R_norm"]
            print(f"  {method:<14} n={len(sub)}  "
                  f"target_raw={tr.mean():+.2f}±{tr.std():.2f}  "
                  f"target_norm={tn.mean():.3f}±{tn.std():.3f}  "
                  f"ΔR_raw={dR.mean():+.2f}±{dR.std():.2f}  "
                  f"ΔR_norm={dRn.mean():+.3f}±{dRn.std():.3f}")

    print(f"\nSaved per-seed: {out_raw}")


if __name__ == "__main__":
    main()
