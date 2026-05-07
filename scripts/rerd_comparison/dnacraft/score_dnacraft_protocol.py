#!/usr/bin/env python3
"""Score one GPA pool × target cell × selection-mode under the DNA-CRAFT
Table-2 protocol (paper-faithful).

Metrics (per DNA-CRAFT §3.3):
  1. MinGap Eval     — mean of per-seq MinGap = target − max(off-targets)
                       under the DRAKES Evaluation-Model
  2. Motif Corr      — Spearman ρ between pool-mean FIMO+JASPAR2024 TFBS
                       freq vector and the DNA-CRAFT reference motif freq
                       (top-99.9-percentile real Gosai by MinGap of true MPRA)
  3. 3-mer Corr      — Pearson ρ between pool 3-mer freq and reference 3-mer
                       freq (same reference)
  4. Diversity       — mean per-position Shannon entropy in BITS (log_2)

Selection: top-128 from the candidate pool,
either by per-seq MinGap (paper-faithful) or by composite-rank-of-motif/3-mer/diversity.

Reuses helpers from scripts/rerd_comparison/score_drakes_protocol.py:
  _scan_count_vector, metric_kmer_corr (replaced with reference-cache version),
  load_eval_oracle, _grelu_batch_score, load_gpa_pool.

Usage:
  python score_dnacraft_protocol.py \\
      --pool_dir results/rerd_comparison/run_dps_pw030_b25_r1 \\
      --cell HepG2 \\
      --selection by_mingap \\
      --output_json results/rerd_comparison/dnacraft/per_pool_jsons/<basename>.json
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import h5py
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Pure pymemesuite for FIMO — avoid grelu/lightning/torch (slow NFS imports)
from pymemesuite.common import MotifFile, Sequence as MSequence  # noqa: E402
from pymemesuite.fimo import FIMO  # noqa: E402

PROJECT = Path("${GPA_REPO_ROOT}")
sys.path.insert(0, str(PROJECT / "scripts/rerd_comparison/dnacraft"))

import build_dnacraft_reference as BR  # noqa: E402
import selection as SEL  # noqa: E402

CELL_TO_IDX = {"HepG2": 0, "K562": 1, "SKNSH": 2}
ALL_CELLS = ["HepG2", "K562", "SKNSH"]
JASPAR_MEME = PROJECT / "data/JASPAR2024_CORE_vertebrates.meme"
_BASE_LOOKUP = np.array(list("ACGT"))


# Inline replacements for DRAKES helpers (avoid heavy imports)
def _detokenize_indices(indices: np.ndarray) -> list:
    """Convert (N, L) int8/int64 [0..3] array → list of N ACGT strings."""
    return ["".join(_BASE_LOOKUP[row]) for row in indices.astype(np.int64)]


def _count_kmers_inline(seqs: list, k: int = 3) -> dict:
    counter = Counter()
    for s in seqs:
        for i in range(len(s) - k + 1):
            counter[s[i:i + k]] += 1
    return dict(counter)


def _read_motifs(meme_file: Path = JASPAR_MEME):
    motifs = []
    mf = MotifFile(str(meme_file))
    while True:
        m = mf.read()
        if m is None:
            break
        motifs.append(m)
    return motifs, mf.background


def _fimo_scan_pool(seqs: list, motifs, bg, threshold: float = 0.001) -> pd.Series:
    """Pool-level FIMO: per-motif total count across `seqs`."""
    fimo = FIMO(both_strands=True, threshold=threshold)
    seq_objs = [MSequence(str(s), name=str(i).encode()) for i, s in enumerate(seqs)]
    counts = defaultdict(int)
    for motif in motifs:
        match = fimo.score_motif(motif, seq_objs, bg).matched_elements
        n_hits = sum(1 for _ in match)
        if n_hits > 0:
            counts[motif.name.decode()] += n_hits
    return pd.Series(counts, dtype=float)


def _fimo_scan_per_seq(seqs: list, motifs, bg, threshold: float = 0.001):
    """Per-seq FIMO: returns (csr_matrix, motif_names_list).

    csr_matrix: shape (N_seqs, M_motifs), int values = # FIMO hits per (seq, motif).
    motif_names_list: list of M motif names matching csr columns.
    """
    from scipy.sparse import lil_matrix, csr_matrix
    fimo = FIMO(both_strands=True, threshold=threshold)
    seq_objs = [MSequence(str(s), name=str(i).encode()) for i, s in enumerate(seqs)]
    motif_names = [m.name.decode() for m in motifs]
    M = len(motif_names)
    N = len(seqs)
    mat = lil_matrix((N, M), dtype=np.int32)
    for j, motif in enumerate(motifs):
        match = fimo.score_motif(motif, seq_objs, bg).matched_elements
        for elem in match:
            try:
                seq_idx = int(elem.source.accession.decode())
            except Exception:
                continue
            if 0 <= seq_idx < N:
                mat[seq_idx, j] += 1
    return mat.tocsr(), motif_names


# ── Pool loading ───────────────────────────────────────────────────────────
SOURCE_FILES = {
    "final":           ("gpa_output.h5", "gpa_output_best.h5", "gpa_output_final.h5"),
    "pool":            ("gpa_output_pool.h5",),
    "archive":         ("gpa_output_filtered.h5",),
    "mingap_archive":  ("gpa_output_mingap_archive.h5",),
    "perstep_archive": ("gpa_output_perstep_archive.h5",),
}


def find_input_h5(pool_dir: Path, source: str = "final") -> Path:
    """Pick the candidate-pool h5 by source.
    - final:           gpa_output.h5 (5K final-step pop, w/ 3-cell scores cached)
    - pool:            gpa_output_pool.h5 (5K best by eval-model, w/ 3-cell scores cached)
    - archive:         gpa_output_filtered.h5 (~178K cumulative; 3-cell eval scores must
                       be precomputed separately — scorer expects oracle_{hepg2,k562,sknsh}_eval)
    - mingap_archive:  gpa_output_mingap_archive.h5 (DNA-CRAFT G*-style design A:
                       spec-bounded with eviction, capacity = N_max sequences)
    - perstep_archive: gpa_output_perstep_archive.h5 (design B: per-step top-K
                       by spec, accumulated, no eviction. ~T_ckpt × K sequences)
    """
    candidates = SOURCE_FILES.get(source)
    if candidates is None:
        raise ValueError(f"Unknown source '{source}'; expected one of {list(SOURCE_FILES)}")
    for fname in candidates:
        p = pool_dir / fname
        if p.exists():
            return p
    raise FileNotFoundError(f"No GPA pool h5 found in {pool_dir} (source={source})")


def load_pool_with_cell_scores(h5_path: Path) -> tuple:
    """Load (indices_int, seqs_str, cell_scores_3) from a GPA pool h5.

    cell_scores_3 has shape (N, 3) = [HepG2, K562, SKNSH] in EVAL-Model space.
    Falls back to re-scoring with the eval oracle if 3-cell scores aren't cached.
    """
    with h5py.File(h5_path, "r") as f:
        indices = f["indices"][:].astype(np.int64)
        cell_keys = ["oracle_hepg2_eval", "oracle_k562_eval", "oracle_sknsh_eval"]
        if not all(k in f for k in cell_keys):
            raise ValueError(
                f"Pool {h5_path} is missing cached 3-cell eval scores ({cell_keys}). "
                "Re-scoring with Eval-Model from this script is disabled to keep "
                "imports lightweight; use the existing score_drakes_protocol pipeline "
                "to populate those columns first."
            )
        cell_scores = np.stack([f[k][:] for k in cell_keys], axis=1)
    seqs = _detokenize_indices(indices)
    cell_scores = np.asarray(cell_scores, dtype=np.float64)
    return indices, seqs, cell_scores


# ── Per-seq features needed for composite selection ───────────────────────
_BASES = "ACGT"
_KMER3_TO_IDX = {a + b + c: 16 * i + 4 * j + k
                 for i, a in enumerate(_BASES)
                 for j, b in enumerate(_BASES)
                 for k, c in enumerate(_BASES)}


def per_seq_kmer3_freq(seqs: list) -> np.ndarray:
    """Returns (N, 64) L1-normalized 3-mer frequency per sequence."""
    out = np.zeros((len(seqs), 64), dtype=np.float64)
    for i, s in enumerate(seqs):
        for j in range(len(s) - 2):
            km = s[j:j + 3]
            if km in _KMER3_TO_IDX:
                out[i, _KMER3_TO_IDX[km]] += 1.0
    row_sums = out.sum(axis=1, keepdims=True)
    out = np.where(row_sums > 0, out / np.maximum(row_sums, 1e-12), 0.0)
    return out


def reference_kmer3_freq(kmer3_counts: dict) -> np.ndarray:
    """Convert reference dict[kmer→count] into a 64-dim L1-normalized vector."""
    vec = np.zeros(64, dtype=np.float64)
    for km, cnt in kmer3_counts.items():
        if km in _KMER3_TO_IDX:
            vec[_KMER3_TO_IDX[km]] += float(cnt)
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec


# ── Main metric computation on selected pool ──────────────────────────────
def per_position_shannon_bits(seqs: list[str]) -> float:
    """Mean per-position Shannon entropy in bits (log_2) across pool."""
    arr = np.array([list(s) for s in seqs])
    n, L = arr.shape
    out = []
    for pos in range(L):
        col = arr[:, pos]
        counts = np.array([(col == b).sum() for b in "ACGT"], dtype=np.float64)
        f = counts / max(n, 1)
        f = f[f > 0]
        out.append(float(-np.sum(f * np.log2(f))))
    return float(np.mean(out))


def motif_spearman(pool_motif_counts: pd.Series, ref_motif_counts: pd.Series) -> float:
    """Spearman ρ between pool motif freq vector and reference motif freq vector,
    with the union of motifs and zero-padding for absent motifs in either.
    Per DNA-CRAFT paper §3.3."""
    if len(pool_motif_counts) == 0:
        return float("nan")
    all_motifs = sorted(set(pool_motif_counts.index) | set(ref_motif_counts.index))
    pool_vec = np.array([pool_motif_counts.get(m, 0.0) for m in all_motifs], dtype=float)
    ref_vec = np.array([ref_motif_counts.get(m, 0.0) for m in all_motifs], dtype=float)
    if len(pool_vec) < 2 or pool_vec.std() == 0 or ref_vec.std() == 0:
        return float("nan")
    return float(spearmanr(pool_vec, ref_vec)[0])


def kmer3_pearson_pool(kmer3_pool_counts: dict, kmer3_ref_counts: dict) -> float:
    """Pearson ρ between pool 3-mer freq and reference 3-mer freq (pool-level)."""
    all_kmers = set(kmer3_pool_counts.keys()) | set(kmer3_ref_counts.keys())
    pool_v = np.array([kmer3_pool_counts.get(k, 0) for k in all_kmers], dtype=float)
    ref_v = np.array([kmer3_ref_counts.get(k, 0) for k in all_kmers], dtype=float)
    if pool_v.std() == 0 or ref_v.std() == 0:
        return float("nan")
    return float(pearsonr(pool_v, ref_v)[0])


# ── Recipe / seed parsing from pool_dir name ──────────────────────────────
RUN_RE = re.compile(r"^(?P<recipe>run_[^_]+_(?:dps|nodps)_[^_]+)(?:_r(?P<seed>\d+))?$")


def parse_recipe_seed(pool_name: str) -> tuple[str, int]:
    """Returns (recipe_label, seed) for a run_ pool dir name.
    Falls back to (pool_name, 0) if no match."""
    m = RUN_RE.match(pool_name)
    if m:
        return m.group("recipe"), int(m.group("seed") or "0")
    # default
    return pool_name, 0


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dir", type=Path, required=True,
                    help="GPA pool dir (must contain gpa_output.h5 or similar)")
    ap.add_argument("--cell", required=True, choices=ALL_CELLS)
    ap.add_argument("--selection", required=True,
                    choices=["by_mingap", "by_composite",
                             "by_motif", "by_kmer3", "by_diversity",
                             "by_pool_composite"])
    ap.add_argument("--n", type=int, default=128, help="Top-N to select (paper: 128)")
    ap.add_argument("--ref_dir", type=Path,
                    default=PROJECT / "results/rerd_comparison/dnacraft/_ref_cache")
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--source", default="final",
                    choices=["final", "pool", "filtered"],
                    help="Which candidate pool h5 to score from")
    args = ap.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    pool_name = args.pool_dir.name
    recipe_label, seed = parse_recipe_seed(pool_name)
    print(f"[score_dnacraft] pool={pool_name}  cell={args.cell}  selection={args.selection}")
    print(f"  recipe_label={recipe_label}  seed={seed}")

    t0 = time.time()
    h5_path = find_input_h5(args.pool_dir, source=args.source)
    indices, seqs, cell_scores = load_pool_with_cell_scores(h5_path)
    print(f"  Loaded {len(seqs)} sequences (length {len(seqs[0])}). t={time.time()-t0:.1f}s")

    # ── Reference ─────────────────────────────────────────────────────────
    ref = BR.load_ref(args.cell, ref_dir=args.ref_dir)
    print(f"  Loaded {args.cell} reference: n={ref['n_seqs']}, "
          f"MinGap≥{ref['mingap_threshold']:.3f}, motifs={len(ref['motif_counts'])}")
    ref_kmer_vec = reference_kmer3_freq(ref["kmer3_counts"])

    # ── Selection ─────────────────────────────────────────────────────────
    target_idx = CELL_TO_IDX[args.cell]
    if args.selection == "by_mingap":
        keep_idx = SEL.select_top_n_by_mingap(cell_scores, target_idx, n=args.n)
    else:
        # by_composite / by_motif / by_kmer3 / by_diversity all need per-seq features
        print(f"  Computing per-seq features for {args.selection} ...")
        seq_kmer = per_seq_kmer3_freq(seqs)
        motifs, bg = _read_motifs()
        csr, motif_names = _fimo_scan_per_seq(seqs, motifs, bg)
        seq_motif_counts = csr.toarray()
        ref_motif_freq = np.array([ref["motif_counts"].get(m, 0.0) for m in motif_names], dtype=float)
        if args.selection == "by_composite":
            keep_idx = SEL.select_top_n_by_composite(
                seq_kmer, seq_motif_counts, ref_kmer_vec, ref_motif_freq, indices, n=args.n
            )
        elif args.selection == "by_motif":
            keep_idx = SEL.select_top_n_by_motif(seq_motif_counts, ref_motif_freq, n=args.n)
        elif args.selection == "by_kmer3":
            keep_idx = SEL.select_top_n_by_kmer3(seq_kmer, ref_kmer_vec, n=args.n)
        elif args.selection == "by_diversity":
            keep_idx = SEL.select_top_n_by_diversity(indices, n=args.n)
        elif args.selection == "by_pool_composite":
            # Greedy joint maximization of pool-aggregate metrics
            # (motif Spearman + kmer Pearson + Shannon + MinGap-mean).
            per_seq_mg = SEL.per_seq_mingap(cell_scores, target_idx)
            keep_idx = SEL.select_top_n_by_pool_composite(
                seq_motif_counts=seq_motif_counts,
                seq_kmer_freq=seq_kmer,
                seqs_int=indices,
                ref_motif_freq=ref_motif_freq,
                ref_kmer_freq=ref_kmer_vec,
                per_seq_mingap=per_seq_mg,
                n=args.n,
                weights=getattr(args, "pool_composite_weights",
                                (1.0, 1.0, 1.0, 1.0)),
                mingap_floor_pct=getattr(args, "mingap_floor_pct", 0.0),
            )
        else:
            raise ValueError(f"unknown selection: {args.selection}")

    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    sub_seqs = [seqs[i] for i in keep_idx]
    sub_indices = indices[keep_idx]
    sub_cell_scores = cell_scores[keep_idx]
    print(f"  Selected top-{len(keep_idx)} (selection={args.selection})")

    # ── Compute the four DNA-CRAFT Table-2 metrics on the selected pool ──
    out = {
        "pool_dir": str(args.pool_dir),
        "pool_name": pool_name,
        "recipe": recipe_label,
        "seed": seed,
        "target_cell": args.cell,
        "selection": args.selection,
        "n_seqs": int(len(keep_idx)),
        "ref_n_seqs": int(ref["n_seqs"]),
        "ref_mingap_threshold": float(ref["mingap_threshold"]),
    }

    # 1. MinGap Eval (mean of per-seq MinGap)
    per_seq_mg = SEL.per_seq_mingap(sub_cell_scores, target_idx)
    out["mingap_eval_mean"] = float(per_seq_mg.mean())
    out["mingap_eval_std"] = float(per_seq_mg.std())
    out["mingap_eval_median"] = float(np.median(per_seq_mg))
    print(f"  [1/4] MinGap Eval mean = {out['mingap_eval_mean']:+.3f} (σ={out['mingap_eval_std']:.3f})")

    # 2. Motif Correlation (Spearman, paper §3.3)
    print("  [2/4] Motif Spearman ...")
    t1 = time.time()
    motifs, bg = _read_motifs()
    pool_motif_counts = _fimo_scan_pool(sub_seqs, motifs, bg)
    out["motif_corr_spearman"] = motif_spearman(pool_motif_counts, ref["motif_counts"])
    print(f"    motif_corr_spearman = {out['motif_corr_spearman']:.3f}  ({time.time()-t1:.1f}s)")

    # 3. 3-mer Pearson (pool-level)
    print("  [3/4] 3-mer Pearson ...")
    pool_kmer = _count_kmers_inline(sub_seqs, k=3)
    out["kmer3_corr_pearson"] = kmer3_pearson_pool(pool_kmer, ref["kmer3_counts"])
    print(f"    kmer3_corr_pearson = {out['kmer3_corr_pearson']:.3f}")

    # 4. Diversity (per-position Shannon entropy in bits)
    print("  [4/4] Diversity (bits) ...")
    out["diversity_bits"] = per_position_shannon_bits(sub_seqs)
    print(f"    diversity_bits = {out['diversity_bits']:.3f}")

    out["elapsed_seconds"] = float(time.time() - t0)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {args.output_json}  (total {out['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
