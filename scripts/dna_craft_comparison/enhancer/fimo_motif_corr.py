#!/usr/bin/env python3
"""FIMO-based motif correlation for the DNA-CRAFT comparison.

For each (cell, pool) pair:
  1. Build (or load cached) reference motif-frequency vector from the
     real Gosai sequences in the top-99.9% by Eval-Model MinGap for `cell`.
  2. Write `pool.fa` to a tmpdir.
  3. Run `conda run -n meme fimo --thresh 1e-4 --text JASPAR2024_CORE_vertebrates.meme pool.fa`.
  4. Count TFBS hits per motif, normalize → frequency vector.
  5. Spearman correlation against the cell-specific reference.

Reference vector is cached at:
  results/dna_craft_comparison/enhancer/eval/motif_ref_{cell}.npy
  results/dna_craft_comparison/enhancer/eval/motif_names.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "dna_craft_comparison" / "enhancer"
JASPAR_MEME = PROJECT_ROOT / "data" / "JASPAR2024_CORE_vertebrates.meme"

IDX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


def indices_to_fasta(indices: np.ndarray, fa_path: Path,
                     name_prefix: str = "seq") -> None:
    with open(fa_path, "w") as fh:
        for i, row in enumerate(indices):
            seq = "".join(IDX_TO_BASE[int(b)] for b in row)
            fh.write(f">{name_prefix}_{i}\n{seq}\n")


def run_fimo(fasta: Path, threshold: float = 1e-4,
             meme_env: str = "meme") -> dict[str, int]:
    """Run FIMO and return motif_id -> hit_count."""
    cmd = ["conda", "run", "-n", meme_env, "fimo",
           "--thresh", str(threshold), "--text",
           str(JASPAR_MEME), str(fasta)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    counts: dict[str, int] = {}
    for line in out.stdout.splitlines():
        if not line or line.startswith("#") or line.startswith("motif_id"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        motif_id = parts[0]
        counts[motif_id] = counts.get(motif_id, 0) + 1
    return counts


def hits_to_freq(counts: dict[str, int],
                 motif_names: list[str]) -> np.ndarray:
    """Map raw hit counts to a frequency vector (one entry per motif)."""
    arr = np.array([counts.get(m, 0) for m in motif_names], dtype=np.float64)
    s = arr.sum()
    if s > 0:
        arr = arr / s
    return arr


def list_jaspar_motifs() -> list[str]:
    motifs: list[str] = []
    with open(JASPAR_MEME) as fh:
        for line in fh:
            if line.startswith("MOTIF"):
                motifs.append(line.split()[1])
    return motifs


def build_reference(cell: str,
                    top_indices: np.ndarray,
                    motif_names: list[str],
                    cache_dir: Path) -> np.ndarray:
    cache = cache_dir / f"motif_ref_{cell}.npy"
    if cache.exists():
        return np.load(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / f"ref_{cell}.fa"
        indices_to_fasta(top_indices, fa, name_prefix=f"ref_{cell}")
        counts = run_fimo(fa)
    freq = hits_to_freq(counts, motif_names)
    np.save(cache, freq)
    print(f"  [motif-ref] cached → {cache}")
    return freq


def motif_corr_for_pool(indices: np.ndarray,
                        ref_freq: np.ndarray,
                        motif_names: list[str]) -> float:
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "pool.fa"
        indices_to_fasta(indices, fa)
        counts = run_fimo(fa)
    pool_freq = hits_to_freq(counts, motif_names)
    rho, _ = spearmanr(pool_freq, ref_freq)
    return float(rho) if rho is not None and not np.isnan(rho) else 0.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True,
                   choices=["build_ref", "score_pool"])
    p.add_argument("--cell", required=True, choices=["hepg2", "k562", "sknsh"])
    p.add_argument("--ref_indices_npy", default=None,
                   help="(build_ref) Path to (N, L) int array of top-99.9% reference indices.")
    p.add_argument("--pool_indices_npy", default=None,
                   help="(score_pool) Path to (N, L) int array for one pool.")
    p.add_argument("--cache_dir", default=str(RESULTS_DIR / "eval"))
    return p.parse_args()


def main():
    args = parse_args()
    motif_names = list_jaspar_motifs()
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    names_path = Path(args.cache_dir) / "motif_names.json"
    if not names_path.exists():
        with open(names_path, "w") as fh:
            json.dump(motif_names, fh)
    if args.mode == "build_ref":
        if args.ref_indices_npy is None:
            raise ValueError("--ref_indices_npy required")
        idx = np.load(args.ref_indices_npy)
        ref = build_reference(args.cell, idx, motif_names, Path(args.cache_dir))
        print(f"  [ref] cell={args.cell} non-zero motifs: {(ref > 0).sum()}/{len(motif_names)}")
    else:
        if args.pool_indices_npy is None:
            raise ValueError("--pool_indices_npy required")
        ref = np.load(Path(args.cache_dir) / f"motif_ref_{args.cell}.npy")
        idx = np.load(args.pool_indices_npy)
        rho = motif_corr_for_pool(idx, ref, motif_names)
        print(f"motif_corr={rho:.4f}")


if __name__ == "__main__":
    main()
