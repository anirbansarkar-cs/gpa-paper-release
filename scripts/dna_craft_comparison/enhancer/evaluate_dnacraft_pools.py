#!/usr/bin/env python3
"""Score GPA enhancer pools against the DNA-CRAFT paper-spec metrics.

Per plan §6:
  1. For each (cell, seed) GPA run, load `gpa_output_pool.h5`.
  2. Top-128 by MinGap under the Evaluation-Model.
  3. Compute (paper Table 2 columns):
       mingap_eval  — mean MinGap of the 128 selected sequences
       kmer_corr    — Pearson R of pool-mean 3-mer freq vs top-99.9%-real
       motif_corr   — Spearman of FIMO-hit motif freq vs top-99.9%-real
       diversity    — mean per-position Shannon entropy over the 128 seqs
  4. Report ΔMinGap = mingap_eval − real_top_mingap[cell] (anchor metric).
  5. Aggregate per cell → eval/aggregated.csv (mean ± std over seeds).

Outputs (under RESULTS_DIR/eval/):
  per_run_metrics.csv     — 9 rows × {cell,seed,mingap_eval,delta_mingap,kmer_corr,motif_corr,diversity}
  aggregated.csv          — 3 rows × {cell, mean±std for each metric}
  real_top_mingap.json    — anchor cache (per cell)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "dna_craft_comparison" / "enhancer"

# Make sibling modules (fimo_motif_corr, dual_oracle_adapter) importable from
# anywhere — without depending on `get_eval_pool` having already inserted
# PROJECT_ROOT/scripts. In Phase 3 the anchor + kmer-ref builders are cached,
# so `get_eval_pool` is never called, and `from scripts.dna_craft_comparison...`
# falls into ImportError -> motif_corr silently NaN. The previous run lost the
# motif_corr column for exactly this reason.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
ENFORMER_DIR = RESULTS_DIR / "enformer_oracles"
SEED_RANGE = range(3)
CELLS = ["hepg2", "k562", "sknsh"]
SELECT_TOP = 128
KMER_K = 3
ANCHOR_PERCENTILE = 99.9


def kmer_profile(indices: np.ndarray, k: int = KMER_K) -> np.ndarray:
    N, L = indices.shape
    powers = 4 ** np.arange(k)[::-1]
    ids = np.zeros((N, L - k + 1), dtype=np.int64)
    for i in range(k):
        ids += indices[:, i:L - k + 1 + i] * powers[i]
    freq = np.zeros((N, 4 ** k), dtype=np.float32)
    for n in range(N):
        c = np.bincount(ids[n], minlength=4 ** k)
        s = c.sum()
        if s > 0:
            freq[n] = c / s
    return freq


def per_position_shannon(indices: np.ndarray) -> float:
    N, L = indices.shape
    total = 0.0
    for p in range(L):
        counts = np.bincount(indices[:, p], minlength=4)
        probs = counts / counts.sum()
        nz = probs[probs > 0]
        total += -(nz * np.log2(nz)).sum()
    return float(total / L)


def load_best_eval(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    h5_path = run_dir / "gpa_output_pool.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"missing {h5_path}")
    with h5py.File(h5_path, "r") as f:
        idx = np.asarray(f["indices"]).astype(np.int64)
        mingap = np.asarray(f["oracle_preds_eval_mingap"]).astype(np.float64)
    return idx, mingap


def select_top_n(idx: np.ndarray, scores: np.ndarray,
                 n: int = SELECT_TOP) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)[:n]
    return idx[order], scores[order]


def get_eval_pool(cell: str, device: str):
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from scripts.dna_craft_comparison.enhancer.dual_oracle_adapter import (
        SplitModelOracle,
    )
    return SplitModelOracle(
        ckpt_dir=str(ENFORMER_DIR), half="eval",
        target_cell=cell, penalty_weight=0.0, device=device,
    )


def compute_real_top_anchors(real_csv: Path, device: str,
                             out_path: Path) -> dict[str, float]:
    """Mean MinGap (Eval-Model) on the top-99.9% of real Gosai seqs per cell.

    Cached: re-uses out_path if it exists.
    """
    if out_path.exists():
        return json.loads(out_path.read_text())
    print(f"[anchor] reading {real_csv}")
    df = pd.read_csv(real_csv)
    df = df[df["sequence"].str.len() == 200].reset_index(drop=True)
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    indices = np.array([[base_map[b] for b in seq] for seq in df["sequence"]],
                       dtype=np.int64)
    seq_t = torch.from_numpy(indices).long()
    anchors: dict[str, float] = {}
    for cell in CELLS:
        eval_pool = get_eval_pool(cell, device)
        mingap, _ = eval_pool.score_mingap_and_gc(seq_t)
        cutoff = np.percentile(mingap, ANCHOR_PERCENTILE)
        top = mingap[mingap >= cutoff]
        anchors[cell] = float(top.mean())
        print(f"  [anchor] {cell}: top-{ANCHOR_PERCENTILE}% n={len(top)}, "
              f"mean MinGap={anchors[cell]:.3f}")
        del eval_pool
        torch.cuda.empty_cache()
    out_path.write_text(json.dumps(anchors, indent=2))
    return anchors


def maybe_motif_corr(idx128: np.ndarray, cell: str,
                     cache_dir: Path) -> float:
    """Optional FIMO motif correlation. Returns NaN if FIMO env missing."""
    try:
        from fimo_motif_corr import (  # sibling import via _THIS_DIR
            list_jaspar_motifs, motif_corr_for_pool,
        )
    except ImportError as e:
        print(f"  [motif] import failed: {e}")
        return float("nan")
    ref_npy = cache_dir / f"motif_ref_{cell}.npy"
    if not ref_npy.exists():
        print(f"  [motif] reference for {cell} not built — skipping FIMO")
        return float("nan")
    ref = np.load(ref_npy)
    motifs = list_jaspar_motifs()
    try:
        return motif_corr_for_pool(idx128, ref, motifs)
    except Exception as e:  # noqa: BLE001
        print(f"  [motif] FIMO failed: {e}")
        return float("nan")


def evaluate_one_run(run_dir: Path, cell: str, seed: int,
                     anchor: float, kmer_ref: np.ndarray,
                     motif_cache_dir: Path) -> dict:
    idx, mingap = load_best_eval(run_dir)
    idx128, mingap128 = select_top_n(idx, mingap)

    pool_kmer = kmer_profile(idx128).mean(axis=0)
    r, _ = pearsonr(pool_kmer, kmer_ref)
    kmer_corr = float(r) if r is not None and not np.isnan(r) else 0.0
    shannon = per_position_shannon(idx128)
    motif = maybe_motif_corr(idx128, cell, motif_cache_dir)

    return {
        "cell": cell,
        "seed": seed,
        "n_pool": len(idx128),
        "mingap_eval": float(mingap128.mean()),
        "delta_mingap": float(mingap128.mean() - anchor),
        "kmer_corr": kmer_corr,
        "motif_corr": motif,
        "diversity": shannon,
    }


def build_reference_kmer(real_csv: Path,
                         eval_dir: Path,
                         device: str) -> dict[str, np.ndarray]:
    out_path = eval_dir / "kmer_refs.npz"
    if out_path.exists():
        npz = np.load(out_path)
        return {c: npz[c] for c in CELLS}
    print(f"[ref-kmer] building from {real_csv}")
    df = pd.read_csv(real_csv)
    df = df[df["sequence"].str.len() == 200].reset_index(drop=True)
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    indices = np.array([[base_map[b] for b in seq] for seq in df["sequence"]],
                       dtype=np.int64)
    seq_t = torch.from_numpy(indices).long()
    refs: dict[str, np.ndarray] = {}
    for cell in CELLS:
        eval_pool = get_eval_pool(cell, device)
        mingap, _ = eval_pool.score_mingap_and_gc(seq_t)
        cutoff = np.percentile(mingap, ANCHOR_PERCENTILE)
        keep = mingap >= cutoff
        kept_idx = indices[keep]
        prof = kmer_profile(kept_idx)
        refs[cell] = prof.mean(axis=0)
        # Dump the top-99.9% indices so fimo_motif_corr.py build_ref can
        # consume them without re-scoring all 798k real seqs.
        np.save(eval_dir / f"top_indices_{cell}.npy", kept_idx)
        print(f"  [ref-kmer] {cell}: n={int(keep.sum())} top-99.9% seqs "
              f"→ top_indices_{cell}.npy")
        del eval_pool
        torch.cuda.empty_cache()
    np.savez(out_path, **refs)
    return refs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root",
                   default=str(RESULTS_DIR / "gpa_runs"))
    p.add_argument("--real_csv",
                   default=str(RESULTS_DIR / "gosai_splits" / "half_B.csv"),
                   help="CSV of real Gosai sequences for anchors and "
                        "k-mer/motif references.")
    p.add_argument("--eval_dir", default=str(RESULTS_DIR / "eval"))
    p.add_argument("--cells", nargs="+", default=CELLS)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEED_RANGE))
    return p.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    anchors = compute_real_top_anchors(
        Path(args.real_csv), device, eval_dir / "real_top_mingap.json")
    kmer_refs = build_reference_kmer(Path(args.real_csv), eval_dir, device)

    rows: list[dict] = []
    runs_root = Path(args.runs_root)
    for cell in args.cells:
        for seed in args.seeds:
            run_dir = runs_root / cell / f"seed{seed}"
            if not run_dir.exists():
                print(f"  [skip] missing {run_dir}")
                continue
            try:
                row = evaluate_one_run(
                    run_dir, cell, seed, anchors[cell],
                    kmer_refs[cell], eval_dir)
            except FileNotFoundError as e:
                print(f"  [skip] {e}")
                continue
            rows.append(row)
            print(f"  [done] {cell} seed{seed}: "
                  f"mingap={row['mingap_eval']:+.3f} "
                  f"Δ={row['delta_mingap']:+.3f} "
                  f"kmer={row['kmer_corr']:.3f} "
                  f"motif={row['motif_corr']:.3f} "
                  f"shannon={row['diversity']:.3f}")
    if not rows:
        print("[evaluate] no runs scored")
        return
    df = pd.DataFrame(rows)
    df.to_csv(eval_dir / "per_run_metrics.csv", index=False)
    agg = df.groupby("cell").agg(["mean", "std"])
    agg.to_csv(eval_dir / "aggregated.csv")
    print(f"\n[evaluate] wrote {eval_dir / 'per_run_metrics.csv'}")
    print(f"[evaluate] wrote {eval_dir / 'aggregated.csv'}")


if __name__ == "__main__":
    main()
