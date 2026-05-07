#!/usr/bin/env python3
"""Build DNA-CRAFT Table 2 reference set: top-99.9 percentile of real Gosai
sequences ranked by their per-cell MinGap score of TRUE MPRA activity.

Per DNA-CRAFT paper §3.3:
  "For each desired cell type, we selected the top 99.9th percentile of real
   sequences ranked by their MinGap score of true MPRA activity. We scanned
   these top sequences with FIMO and the JASPAR 2024 core vertebrate database
   to compute TFBS frequency distribution."

This file deliberately avoids importing ANY DRAKES module (which transitively
loads lightning + torch + timm — 20+ min on slow NFS). It loads the Gosai
data directly via pandas and reproduces `count_kmers` / `batch_dna_detokenize`
inline (<10 lines each).

Output (one cache file per target cell):
  results/rerd_comparison/dnacraft/_ref_cache/dnacraft_ref_<CELL>.npz
"""
import argparse
import os
import sys
import time as _time
from collections import Counter, defaultdict
from pathlib import Path

print("[buildref] startup t=0", flush=True)
_t0 = _time.time()

import numpy as np
import pandas as pd
print(f"[buildref] numpy/pandas imported t={_time.time()-_t0:.1f}s", flush=True)

print(f"[buildref] importing pymemesuite t={_time.time()-_t0:.1f}s", flush=True)
from pymemesuite.common import MotifFile, Sequence as MSequence  # noqa: E402
from pymemesuite.fimo import FIMO  # noqa: E402
print(f"[buildref] pymemesuite imported t={_time.time()-_t0:.1f}s", flush=True)

# ── Constants ──────────────────────────────────────────────────────────────
PROJECT = Path("${GPA_REPO_ROOT}")
JASPAR_MEME = PROJECT / "data/JASPAR2024_CORE_vertebrates.meme"
GOSAI_CSV = Path("${GPA_EXTERNAL_ROOT}/DRAKES_data/data_and_model/mdlm/gosai_data/processed_data/gosai_all.csv")
REF_DIR = PROJECT / "results/rerd_comparison/dnacraft/_ref_cache"
CELL_TO_COL = {"HepG2": "hepg2", "K562": "k562", "SKNSH": "sknsh"}
ALL_CELLS = ["HepG2", "K562", "SKNSH"]


# ── Inline DRAKES helpers (avoid importing drakes_dna which pulls torch+lightning) ──
def count_kmers_inline(seqs: list, k: int = 3) -> dict:
    """Replicates `oracle.count_kmers` from DRAKES: returns dict[kmer→int]
    of total occurrences across all seqs."""
    counter = Counter()
    for s in seqs:
        for i in range(len(s) - k + 1):
            counter[s[i:i + k]] += 1
    return dict(counter)


# ── FIMO helpers (pure pymemesuite) ───────────────────────────────────────
def read_meme_motifs(meme_file: Path):
    motifs = []
    mf = MotifFile(str(meme_file))
    while True:
        m = mf.read()
        if m is None:
            break
        motifs.append(m)
    return motifs, mf.background


def fimo_scan_pool(seqs: list, motifs, bg, threshold: float = 0.001) -> pd.Series:
    """FIMO over all motifs against `seqs`. Returns per-motif count Series."""
    fimo = FIMO(both_strands=True, threshold=threshold)
    seq_objs = [MSequence(str(s), name=str(i).encode()) for i, s in enumerate(seqs)]
    counts = defaultdict(int)
    for motif in motifs:
        match = fimo.score_motif(motif, seq_objs, bg).matched_elements
        n_hits = sum(1 for _ in match)
        if n_hits > 0:
            counts[motif.name.decode()] += n_hits
    return pd.Series(counts, dtype=float)


# ── Reference build per cell ──────────────────────────────────────────────
def build_one_cell(target_cell: str, motifs, bg, gosai_df: pd.DataFrame,
                   percentile: float = 0.999, k_mer: int = 3) -> dict:
    target_col = CELL_TO_COL[target_cell]
    off_cols = [c for c in CELL_TO_COL.values() if c != target_col]
    target_act = gosai_df[target_col].to_numpy()
    off_act = gosai_df[off_cols].to_numpy()
    mingap_true = target_act - off_act.max(axis=1)

    threshold = float(np.quantile(mingap_true, percentile))
    keep_mask = mingap_true > threshold
    n_kept = int(keep_mask.sum())
    print(f"  [{target_cell}] {n_kept} seqs above {percentile*100:.1f}-pctile "
          f"(MinGap > {threshold:.3f})", flush=True)

    kept_seqs = gosai_df.loc[keep_mask, "seq"].tolist()

    print(f"  [{target_cell}] computing 3-mer counts ...", flush=True)
    _t = _time.time()
    kmer3_counts = count_kmers_inline(kept_seqs, k=k_mer)
    print(f"  [{target_cell}]   3-mer done ({_time.time()-_t:.1f}s, {len(kmer3_counts)} kmers)",
          flush=True)

    print(f"  [{target_cell}] FIMO scan over {n_kept} reference seqs ...", flush=True)
    _t = _time.time()
    motif_counts = fimo_scan_pool(kept_seqs, motifs, bg)
    print(f"  [{target_cell}]   FIMO done ({_time.time()-_t:.1f}s, "
          f"{len(motif_counts)} motifs with hits)", flush=True)

    return {
        "seqs": kept_seqs,
        "motif_counts": motif_counts,
        "kmer3_counts": kmer3_counts,
        "n_seqs": n_kept,
        "mingap_threshold": threshold,
        "target_cell": target_cell,
        "percentile": percentile,
    }


def save_ref(ref: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        seqs=np.array(ref["seqs"], dtype=object),
        motif_counts_index=np.array(ref["motif_counts"].index.tolist(), dtype=object),
        motif_counts_values=ref["motif_counts"].values.astype(np.float64),
        kmer3_counts=np.array([ref["kmer3_counts"]], dtype=object),
        n_seqs=int(ref["n_seqs"]),
        mingap_threshold=float(ref["mingap_threshold"]),
        target_cell=str(ref["target_cell"]),
        percentile=float(ref["percentile"]),
    )
    print(f"  → cached {out_path}", flush=True)


def load_ref(target_cell: str, ref_dir: Path = REF_DIR) -> dict:
    p = ref_dir / f"dnacraft_ref_{target_cell}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Reference cache missing: {p}")
    z = np.load(p, allow_pickle=True)
    motif_counts = pd.Series(
        z["motif_counts_values"],
        index=z["motif_counts_index"].tolist(),
        dtype=float,
    )
    kmer3_counts = z["kmer3_counts"].item()
    return {
        "seqs": z["seqs"].tolist(),
        "motif_counts": motif_counts,
        "kmer3_counts": kmer3_counts,
        "n_seqs": int(z["n_seqs"]),
        "mingap_threshold": float(z["mingap_threshold"]),
        "target_cell": str(z["target_cell"]),
        "percentile": float(z["percentile"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=ALL_CELLS + ["all"], default="all")
    ap.add_argument("--percentile", type=float, default=0.999)
    ap.add_argument("--ref_dir", type=Path, default=REF_DIR)
    ap.add_argument("--meme_file", type=Path, default=JASPAR_MEME)
    ap.add_argument("--gosai_csv", type=Path, default=GOSAI_CSV)
    args = ap.parse_args()

    cells = ALL_CELLS if args.cell == "all" else [args.cell]
    print(f"\nBuilding DNA-CRAFT references for cells: {cells}", flush=True)
    print(f"  percentile = {args.percentile}", flush=True)
    print(f"  ref_dir    = {args.ref_dir}", flush=True)
    print(f"  meme_file  = {args.meme_file}", flush=True)
    print(f"  gosai_csv  = {args.gosai_csv}", flush=True)

    print(f"\nLoading JASPAR motifs from {args.meme_file} ...", flush=True)
    _t = _time.time()
    motifs, bg = read_meme_motifs(args.meme_file)
    print(f"  Loaded {len(motifs)} motifs ({_time.time()-_t:.1f}s).", flush=True)

    print(f"\nLoading Gosai full dataset from {args.gosai_csv} ...", flush=True)
    _t = _time.time()
    gosai_df = pd.read_csv(args.gosai_csv)
    print(f"  Loaded {len(gosai_df)} Gosai seqs ({_time.time()-_t:.1f}s).", flush=True)
    # Sanity: required columns
    for col in ["seq", "hepg2", "k562", "sknsh"]:
        if col not in gosai_df.columns:
            raise ValueError(f"Gosai CSV missing column '{col}'; got {list(gosai_df.columns)}")

    args.ref_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        out_path = args.ref_dir / f"dnacraft_ref_{cell}.npz"
        if out_path.exists():
            print(f"\n[{cell}] cache exists at {out_path} — skipping (delete to rebuild)",
                  flush=True)
            continue
        print(f"\n[{cell}] building reference …", flush=True)
        ref = build_one_cell(cell, motifs, bg, gosai_df, percentile=args.percentile)
        save_ref(ref, out_path)

    print("\nALL_DONE", flush=True)


if __name__ == "__main__":
    main()
