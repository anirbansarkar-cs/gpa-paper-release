#!/usr/bin/env python3
"""Quick: re-select top-128 by ΔR_norm (paper-aligned) instead of target,
for both CtrlDNA and GPA universal recipe. Uses already-cached per-seq scores
(CSV rewards for CtrlDNA, h5 scores for GPA). Adds FIMO + JASPAR-2024
motif_corr against the same paper-aligned reference cache.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT = Path("${GPA_REPO_ROOT}")
DATA_DIR = PROJECT / "scripts/ctrl_dna_comparison/promoter/data"
RESULTS = PROJECT / "results/ctrl_dna_comparison/promoter"
ORACLE = json.loads((DATA_DIR / "oracle_ranges.json").read_text())
CELLS = ["JURKAT", "K562", "THP1"]
BASE_MAP = {0: "A", 1: "C", 2: "G", 3: "T"}

sys.path.insert(0, str(PROJECT / "scripts/ctrl_dna_comparison/promoter"))
import score_motif_corr_promoter as M


def normalize(v, cell):
    r = ORACLE[cell]
    return (v - r["min"]) / (r["max"] - r["min"])


def per_position_shannon_int(arr):
    n, L = arr.shape
    out = []
    for pos in range(L):
        c = np.array([(arr[:, pos] == b).sum() for b in [0, 1, 2, 3]], dtype=float)
        f = c / n
        f = f[f > 0]
        out.append(-np.sum(f * np.log2(f)))
    return float(np.mean(out))


def per_position_shannon_str(seqs):
    arr = np.array([list(s) for s in seqs])
    n, L = arr.shape
    out = []
    for pos in range(L):
        c = np.array([(arr[:, pos] == b).sum() for b in "ACGT"], dtype=float)
        f = c / n
        f = f[f > 0]
        out.append(-np.sum(f * np.log2(f)))
    return float(np.mean(out))


_MOTIFS = None
_BG = None
_REF_FREQ = {}


def init_fimo():
    global _MOTIFS, _BG
    print("Loading JASPAR motifs...")
    _MOTIFS, _BG = M.read_meme(M.JASPAR_MEME)
    print(f"  {len(_MOTIFS)} motifs")
    for cell in CELLS:
        cache = DATA_DIR / f"promoter_motif_ref_freq_{cell}.csv"
        _REF_FREQ[cell] = pd.read_csv(cache, index_col=0).squeeze("columns")


def fimo_motif_corr(seqs, target_cell):
    pool_counts = M.scan_sequences(seqs, _MOTIFS, _BG)
    corrs = M.compute_motif_correlation(pool_counts, _REF_FREQ[target_cell])
    return float(corrs.mean()), float(corrs.std())


def ctrldna_top128_by_delta_R(cell, seed):
    csv = RESULTS / f"ctrldna_full_{cell}_seed{seed}_e15" / f"ctrldna_{cell}_seed{seed}.csv"
    df = pd.read_csv(csv)
    df = df[df["round"] == df["round"].max()].copy()
    cell_to_col = {"JURKAT": "reward_1", "K562": "reward_2", "THP1": "reward_3"}
    tgt = df[cell_to_col[cell]]
    off_cols = [df[cell_to_col[c]] for c in CELLS if c != cell]
    df["delta_R_norm"] = tgt - (off_cols[0] + off_cols[1]) / 2
    df = df.sort_values("delta_R_norm", ascending=False).head(128)
    seqs = df["sequence"].tolist()
    raw = {c: df[cell_to_col[c]].values * (ORACLE[c]["max"] - ORACLE[c]["min"]) + ORACLE[c]["min"]
           for c in CELLS}
    norm = {c: df[cell_to_col[c]].values for c in CELLS}
    off = [c for c in CELLS if c != cell]
    mc_mean, _ = fimo_motif_corr(seqs, cell)
    return {
        "target_raw":  float(raw[cell].mean()),
        "target_norm": float(norm[cell].mean()),
        "delta_R_raw": float(raw[cell].mean() - 0.5 * sum(raw[c].mean() for c in off)),
        "delta_R_norm": float(norm[cell].mean() - 0.5 * sum(norm[c].mean() for c in off)),
        "shannon": per_position_shannon_str(seqs),
        "motif_corr": mc_mean,
    }


def gpa_top128_by_delta_R(cell, seed):
    pool_dir = RESULTS / f"gpa_full_{cell}_nodps_seed{seed}_nodpsA_K8_div1p0"
    h5_path = pool_dir / "gpa_output_pool.h5"
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as h:
        seqs_int = h["sequences"][:]
        scores = {c: h[f"score_{c}"][:].astype(np.float64) for c in CELLS}
    norm = {c: normalize(scores[c], c) for c in CELLS}
    off = [c for c in CELLS if c != cell]
    delta_R_norm_per_seq = norm[cell] - (norm[off[0]] + norm[off[1]]) / 2
    top_idx = np.argsort(delta_R_norm_per_seq)[::-1][:128]
    sub_seqs = seqs_int[top_idx]
    sub_norm = {c: norm[c][top_idx] for c in CELLS}
    sub_raw  = {c: scores[c][top_idx] for c in CELLS}
    seqs_str = ["".join(BASE_MAP[b] for b in s) for s in sub_seqs]
    mc_mean, _ = fimo_motif_corr(seqs_str, cell)
    return {
        "target_raw":  float(sub_raw[cell].mean()),
        "target_norm": float(sub_norm[cell].mean()),
        "delta_R_raw": float(sub_raw[cell].mean() - 0.5 * sum(sub_raw[c].mean() for c in off)),
        "delta_R_norm": float(sub_norm[cell].mean() - 0.5 * sum(sub_norm[c].mean() for c in off)),
        "shannon": per_position_shannon_int(sub_seqs),
        "motif_corr": mc_mean,
    }


def main():
    init_fimo()
    print(f"\n{'Cell':<8} {'Method':<14} target_raw     ΔR_norm        shannon       motif_corr_FIMO")
    print("-" * 105)
    out_rows = []
    for cell in CELLS:
        for method, fn in [("CtrlDNA", ctrldna_top128_by_delta_R), ("GPA universal", gpa_top128_by_delta_R)]:
            seed_results = []
            for seed in range(5):
                r = fn(cell, seed)
                if r:
                    seed_results.append(r)
            if not seed_results:
                continue
            agg = {k: (np.mean([r[k] for r in seed_results]),
                       np.std([r[k] for r in seed_results]))
                   for k in seed_results[0].keys()}
            f = lambda k: f"{agg[k][0]:+.2f}±{agg[k][1]:.2f}"
            print(f"{cell:<8} {method:<14} {f('target_raw')}  {f('delta_R_norm')}  {f('shannon')}  {f('motif_corr')}", flush=True)
            out_rows.append({"cell": cell, "method": method, **{k: v[0] for k, v in agg.items()},
                             **{f"{k}_std": v[1] for k, v in agg.items()}})
    df = pd.DataFrame(out_rows)
    out_csv = RESULTS / "top128_by_deltaR_norm_universal.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
