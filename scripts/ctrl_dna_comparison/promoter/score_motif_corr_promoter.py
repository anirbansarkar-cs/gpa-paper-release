#!/usr/bin/env python3
"""Paper-aligned FIMO + JASPAR-2024 motif_corr for promoter pools.

Implements the metric from Ctrl-DNA paper §3.3, §4.1:
  - Reference seqs: real promoters in top-50% target activity AND bottom-50%
    off-target activity (selecting cell-discriminative natural seqs).
  - FIMO scan with JASPAR-2024 PPMs over reference seqs → reference motif
    frequency vector q_real (mean over reference seqs).
  - Same FIMO scan over pool seqs → q_gen.
  - motif_corr = Pearson(q_gen, q_real). Mean ± std over the pool's seqs.

Two modes:
  build-ref:   build per-cell reference motif freq cache (one-time)
  score-pool:  score one pool's top-128 sequences and emit a JSON

Caches: promoter_motif_ref_freq_<cell>.csv (alongside enhancer caches)
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path("${GPA_REPO_ROOT}")
DATA_DIR = PROJECT_DIR / "scripts/ctrl_dna_comparison/promoter/data"
JASPAR_MEME = PROJECT_DIR / "data/JASPAR2024_CORE_vertebrates.meme"
PROMOTER_CELLS = ["JURKAT", "K562", "THP1"]
BASE_MAP = {0: "A", 1: "C", 2: "G", 3: "T"}


# ── FIMO + JASPAR ──────────────────────────────────────────────────────────
def read_meme(meme_file):
    from pymemesuite.common import MotifFile
    motifs = []
    mf = MotifFile(str(meme_file))
    while True:
        m = mf.read()
        if m is None:
            break
        motifs.append(m)
    return motifs, mf.background


def scan_sequences(sequences, motifs, bg, threshold=0.001):
    from pymemesuite.common import Sequence as MSequence
    from pymemesuite.fimo import FIMO
    fimo = FIMO(both_strands=True, threshold=threshold)
    seq_objs = [MSequence(str(s), name=str(i).encode()) for i, s in enumerate(sequences)]
    d = defaultdict(list)
    n_hits = 0
    for motif in motifs:
        match = fimo.score_motif(motif, seq_objs, bg).matched_elements
        for m in match:
            d["Matrix_id"].append(motif.name.decode())
            d["SeqID"].append(m.source.accession.decode())
            d["start"].append(m.start)
            n_hits += 1
    if n_hits == 0:
        return pd.DataFrame(0, index=[str(i) for i in range(len(sequences))], columns=["no_hits"])
    df = pd.DataFrame(d)
    counts = pd.pivot_table(df, values="start", index="SeqID", columns="Matrix_id",
                            aggfunc="count").fillna(0)
    counts = counts.reindex([str(i) for i in range(len(sequences))], fill_value=0)
    return counts


def compute_motif_correlation(gen_counts, ref_freq):
    common = gen_counts.columns.intersection(ref_freq.index)
    if len(common) == 0:
        return np.zeros(len(gen_counts))
    gen = gen_counts[common].values
    ref = ref_freq[common].values
    corrs = []
    for i in range(len(gen)):
        if gen[i].sum() == 0:
            corrs.append(0.0)
        else:
            gf = gen[i] / max(gen[i].sum(), 1)
            r = np.corrcoef(gf, ref)[0, 1]
            corrs.append(r if not np.isnan(r) else 0.0)
    return np.array(corrs)


def per_position_shannon(seqs):
    arr = np.array([list(s) for s in seqs])
    n, L = arr.shape
    out = []
    for pos in range(L):
        c = np.array([(arr[:, pos] == b).sum() for b in "ACGT"], dtype=float)
        f = c / n
        f = f[f > 0]
        out.append(-np.sum(f * np.log2(f)))
    return float(np.mean(out))


# ── Reference-seq selection (Ctrl-DNA paper rule) ─────────────────────────
def get_paper_reference_seqs(target_cell, n_max=2000, seed=42):
    """Top 50th percentile target activity ∩ bottom 50th percentile off-target.

    Paper §4.1: 'identify real sequences by selecting those in the top 50th
    percentile for fitness in the target cell type and the bottom 50th
    percentile in off-target cell types.'
    """
    df = pd.read_csv(DATA_DIR / "finetuning_data.csv")
    off = [c for c in PROMOTER_CELLS if c != target_cell]

    tgt_threshold = df[target_cell].quantile(0.5)
    off1_threshold = df[off[0]].quantile(0.5)
    off2_threshold = df[off[1]].quantile(0.5)

    mask = (df[target_cell] >= tgt_threshold) & \
           (df[off[0]] <= off1_threshold) & \
           (df[off[1]] <= off2_threshold)
    sub = df[mask]
    print(f"  {target_cell}: {len(sub)}/{len(df)} seqs match top-50% target ∩ bot-50% off-target")
    if len(sub) > n_max:
        sub = sub.sample(n=n_max, random_state=seed)
    return sub["sequence"].tolist()


def build_ref_freq(target_cell, motifs, bg, n_sample=2000, seed=42):
    cache = DATA_DIR / f"promoter_motif_ref_freq_{target_cell}.csv"
    if cache.exists():
        print(f"Loading cached ref freq: {cache}")
        return pd.read_csv(cache, index_col=0).squeeze("columns")
    print(f"Building ref freq for {target_cell} (no cache)")
    refs = get_paper_reference_seqs(target_cell, n_max=n_sample, seed=seed)
    print(f"  scanning {len(refs)} reference seqs with FIMO over {len(motifs)} motifs...")
    ref_counts = scan_sequences(refs, motifs, bg)
    ref_freq = ref_counts.sum(axis=0)
    ref_freq = ref_freq / max(ref_freq.sum(), 1)
    ref_freq.to_csv(cache)
    print(f"  Saved cache: {cache} ({len(ref_freq)} motifs)")
    return ref_freq


# ── Pool input loading ─────────────────────────────────────────────────────
def load_gpa_top128(pool_dir, target_cell):
    """Read GPA best_eval h5, take top-128 by target cell score, decode ACGT."""
    pd_ = Path(pool_dir)
    for fname in ("gpa_output_pool.h5", "gpa_output_best.h5", "gpa_output_final.h5"):
        f = pd_ / fname
        if f.exists():
            with h5py.File(f, "r") as h:
                seqs = h["sequences"][:]
                target_scores = h[f"score_{target_cell}"][:]
            top_idx = np.argsort(target_scores)[::-1][:128].copy()
            top_seqs = seqs[top_idx]
            return ["".join(BASE_MAP[b] for b in s) for s in top_seqs]
    raise FileNotFoundError(f"No GPA pool h5 in {pool_dir}")


def load_ctrldna_top128(pool_dir, target_cell, seed):
    """Read CtrlDNA per-seed CSV, take top-128 by true_score (target reward)."""
    pd_ = Path(pool_dir)
    csv = pd_ / f"ctrldna_{target_cell}_seed{seed}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No CtrlDNA CSV: {csv}")
    df = pd.read_csv(csv)
    # last round + sort by true_score
    last_round = df["round"].max()
    df = df[df["round"] == last_round].copy()
    df = df.sort_values("true_score", ascending=False).head(128)
    return df["sequence"].tolist()


# ── Main ───────────────────────────────────────────────────────────────────
def cmd_build_ref(args):
    print(f"Loading JASPAR motifs: {JASPAR_MEME}")
    motifs, bg = read_meme(JASPAR_MEME)
    print(f"  {len(motifs)} motifs")
    cells = args.cells.split(",") if args.cells else PROMOTER_CELLS
    for c in cells:
        build_ref_freq(c, motifs, bg)
    print("Done.")


def cmd_score_pool(args):
    motifs, bg = read_meme(JASPAR_MEME)
    ref_freq = build_ref_freq(args.target_cell, motifs, bg)
    if args.method == "gpa":
        seqs = load_gpa_top128(args.pool_dir, args.target_cell)
    elif args.method == "ctrldna":
        seqs = load_ctrldna_top128(args.pool_dir, args.target_cell, args.seed)
    elif args.method == "csv":
        seqs = pd.read_csv(args.input_csv)["sequence"].tolist()[:128]
    else:
        raise ValueError(args.method)
    print(f"  scanning {len(seqs)} pool seqs with FIMO...")
    pool_counts = scan_sequences(seqs, motifs, bg)
    corrs = compute_motif_correlation(pool_counts, ref_freq)
    out = {
        "target_cell": args.target_cell,
        "n_seqs": len(seqs),
        "motif_corr_mean": float(corrs.mean()),
        "motif_corr_std": float(corrs.std()),
        "shannon": per_position_shannon(seqs),
    }
    if args.method == "gpa":
        out["pool_dir"] = str(args.pool_dir)
    elif args.method == "ctrldna":
        out["pool_dir"] = str(args.pool_dir)
        out["seed"] = args.seed
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  motif_corr: {out['motif_corr_mean']:.3f} ± {out['motif_corr_std']:.3f}")
    print(f"  Saved {args.output_json}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("build-ref")
    pb.add_argument("--cells", default="")
    pb.set_defaults(fn=cmd_build_ref)
    ps = sub.add_parser("score-pool")
    ps.add_argument("--method", choices=["gpa", "ctrldna", "csv"], required=True)
    ps.add_argument("--pool_dir", type=str, default=None)
    ps.add_argument("--target_cell", type=str, required=True, choices=PROMOTER_CELLS)
    ps.add_argument("--seed", type=int, default=None)
    ps.add_argument("--input_csv", type=str, default=None)
    ps.add_argument("--output_json", type=str, required=True)
    ps.set_defaults(fn=cmd_score_pool)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
