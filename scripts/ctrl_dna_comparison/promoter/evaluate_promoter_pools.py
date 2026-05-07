#!/usr/bin/env python3
"""
6-metric evaluation of CtrlDNA / GPA promoter pools.

For each input pool directory, selects top-128 sequences by the target
cell's normalized reward (as written by `run_ctrldna_promoter.py`) and
rescores them with the frozen promoter Enformer oracles. Emits a CSV
table with one row per pool and the canonical six columns:
    N, jurkat, k562, thp1, composite, shannon, motif_corr
per `feedback_table_format.md`.

`motif_corr` is computed as the Pearson correlation of the pool's mean
3-mer profile against the target-cell reference 3-mer profile built from
the top-1000 high-activity training sequences for that cell. No FIMO
dependency — directly usable without external tools.

Pool auto-detection:
  - CtrlDNA: looks for `ctrldna_{TASK}_seed{SEED}.csv` (full run) or the
    `_partial.csv` fallback if the run is still in flight. Sequences are
    stored with columns sequence, reward_1, reward_2, reward_3 (cell
    order: JURKAT, K562, THP1).
  - GPA:     looks for `gpa_output_best.h5` / `gpa_output_pool.h5`
    / `gpa_output_final.h5` in that priority order and reads the
    `sequences` (int8, N×L) + `score_{CELL}` datasets.

Usage:
    python scripts/ctrl_dna_comparison/promoter/evaluate_promoter_pools.py \\
        --pools results/ctrl_dna_comparison/promoter/ctrldna_full_*_seed0* \\
        --output results/ctrl_dna_comparison/promoter/eval_summary.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from oracle_adapter import (  # noqa: E402
    PROMOTER_CELLS, PromoterOraclePool, PromoterCellAdapter,
    dna_to_indices,
)

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}
KMER_K = 3
KMER_DIM = 4 ** KMER_K


# ------------------------------------------------------------------
# k-mer reference profile (cached across pools)
# ------------------------------------------------------------------
def kmer_profile_from_indices(indices: np.ndarray, k: int = KMER_K) -> np.ndarray:
    """Per-sequence k-mer frequency vector (N, 4**k), L1-normalized."""
    N, L = indices.shape
    if L < k:
        raise ValueError(f"seq len {L} < k={k}")
    # Encode each k-mer as base-4 integer
    powers = 4 ** np.arange(k)[::-1]
    kmer_ids = np.zeros((N, L - k + 1), dtype=np.int64)
    for i in range(k):
        kmer_ids += indices[:, i:L - k + 1 + i] * powers[i]
    # Count frequency per sequence
    freq = np.zeros((N, 4 ** k), dtype=np.float32)
    for n in range(N):
        counts = np.bincount(kmer_ids[n], minlength=4 ** k)
        total = counts.sum()
        if total > 0:
            freq[n] = counts / total
    return freq


def build_reference_kmer_profiles(
    data_csv: str, top_k: int = 1000
) -> dict:
    """Return {cell: mean_kmer_profile (4**K,)} for top-k high-activity
    sequences per cell from training data."""
    df = pd.read_csv(data_csv)
    train = df.loc[df["is_train"].astype(bool)].reset_index(drop=True)
    ref = {}
    for cell in PROMOTER_CELLS:
        top = train.nlargest(min(top_k, len(train)), cell)
        indices = dna_to_indices(top["sequence"].tolist())
        prof = kmer_profile_from_indices(indices)
        ref[cell] = prof.mean(axis=0)
        print(f"  [ref-kmer] {cell}: {len(top)} seqs, "
              f"mean_activity={top[cell].mean():+.3f}")
    return ref


# ------------------------------------------------------------------
# Per-pool metric computation
# ------------------------------------------------------------------
def per_position_shannon(indices: np.ndarray) -> float:
    """Mean per-position Shannon entropy over 4 bases, in bits."""
    N, L = indices.shape
    total = 0.0
    for p in range(L):
        counts = np.bincount(indices[:, p], minlength=4)
        probs = counts / counts.sum()
        nz = probs[probs > 0]
        total += -(nz * np.log2(nz)).sum()
    return total / L


def detect_pool_and_target(pool_dir: Path) -> tuple:
    """Return (method, task, seed, suffix) parsed from directory name.

    Name convention:
      ctrldna_{mode}_{TASK}_seed{SEED}[_suffix]  (method='ctrldna')
      gpa_{mode}_{TASK}_{dps}_seed{SEED}[_suffix] (method='gpa')
    """
    name = pool_dir.name
    m = re.match(
        r"(ctrldna|gpa)_[^_]+_(JURKAT|K562|THP1)(?:_(?:dps|nodps))?_seed(\d+)(.*)$",
        name,
    )
    if m is None:
        raise ValueError(f"cannot parse pool name: {name}")
    method, task, seed, suffix = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    return method, task, seed, suffix


def load_ctrldna_top128(pool_dir: Path, task: str, seed: int) -> np.ndarray:
    """Return top-128 sequences (N=128, L=250) as int indices."""
    stem = pool_dir / f"ctrldna_{task}_seed{seed}"
    full_csv = stem.with_suffix(".csv")
    partial_csv = Path(str(stem) + "_partial.csv")
    csv_path = full_csv if full_csv.exists() else partial_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"no ctrldna CSV in {pool_dir} (tried {full_csv.name}, "
            f"{partial_csv.name})"
        )
    df = pd.read_csv(csv_path)
    target_idx = PROMOTER_CELLS.index(task)  # 0, 1, or 2
    reward_col = f"reward_{target_idx + 1}"
    if reward_col not in df.columns:
        raise KeyError(f"{csv_path} missing {reward_col}; columns={list(df.columns)}")
    top = df.nlargest(128, reward_col)
    return dna_to_indices(top["sequence"].tolist())


def load_gpa_best_pool(pool_dir: Path) -> Optional[np.ndarray]:
    for fname in ("gpa_output_pool.h5", "gpa_output_best.h5", "gpa_output_final.h5"):
        p = pool_dir / fname
        if p.exists():
            with h5py.File(p, "r") as f:
                seqs = f["sequences"][:].astype(np.int64)
            # GPA pool may be larger than 128 — keep all for fair comparison
            # (run-specific sizes differ; we'll also report top-128 by target).
            return seqs
    return None


def load_gpa_archive(pool_dir: Path) -> Optional[np.ndarray]:
    p = pool_dir / "gpa_output_filtered.h5"
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        return f["sequences"][:].astype(np.int64)


def score_pool(indices: np.ndarray, pool: PromoterOraclePool) -> dict:
    device = pool.device
    tensor = torch.from_numpy(indices).long().to(device)
    out = {}
    for cell in PROMOTER_CELLS:
        adapter = PromoterCellAdapter(pool, cell)
        scores = []
        for start in range(0, len(indices), 256):
            end = min(start + 256, len(indices))
            scores.append(adapter(tensor[start:end]).detach().cpu().numpy())
        out[cell] = np.concatenate(scores)
    return out


def _metrics_from_scores(indices: np.ndarray, cell_scores: dict,
                         target_cell: str, ref_kmer: dict) -> dict:
    off_cells = [c for c in PROMOTER_CELLS if c != target_cell]
    # Pool-mean legacy composite (t − 0.5·mean(off))
    composite = (
        cell_scores[target_cell].mean()
        - 0.5 * np.mean([cell_scores[c].mean() for c in off_cells])
    )
    # Ctrl-DNA paper specificity (per-seq): mean over seqs of [target_i − max(off_i)]
    off_stack = np.stack([cell_scores[c] for c in off_cells], axis=0)  # (2, N)
    specificity = float((cell_scores[target_cell] - off_stack.max(axis=0)).mean())
    pool_kmer = kmer_profile_from_indices(indices).mean(axis=0)
    if np.std(pool_kmer) == 0 or np.std(ref_kmer[target_cell]) == 0:
        motif_corr = float("nan")
    else:
        motif_corr = float(np.corrcoef(pool_kmer, ref_kmer[target_cell])[0, 1])
    return {
        "N": int(indices.shape[0]),
        "jurkat": float(cell_scores["JURKAT"].mean()),
        "k562": float(cell_scores["K562"].mean()),
        "thp1": float(cell_scores["THP1"].mean()),
        "composite": float(composite),
        "specificity": specificity,
        "shannon": float(per_position_shannon(indices)),
        "motif_corr": motif_corr,
    }


def compute_metrics(indices: np.ndarray, target_cell: str,
                    oracle_pool: PromoterOraclePool,
                    ref_kmer: dict) -> dict:
    cell_scores = score_pool(indices, oracle_pool)
    return _metrics_from_scores(indices, cell_scores, target_cell, ref_kmer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pools", nargs="+", required=True,
        help="Pool directories (glob expanded). Parsed names must match "
             "'(ctrldna|gpa)_*_{JURKAT,K562,THP1}*_seed{N}[_suffix]'.",
    )
    ap.add_argument(
        "--output",
        default="results/ctrl_dna_comparison/promoter/eval_summary.csv",
    )
    ap.add_argument(
        "--oracle_ckpt_dir",
        default="scripts/ctrl_dna_comparison/promoter/checkpoints",
    )
    ap.add_argument(
        "--data_csv",
        default="scripts/ctrl_dna_comparison/promoter/data/finetuning_data.csv",
    )
    ap.add_argument("--ref_top_k", type=int, default=1000,
                    help="Top-K high-activity train seqs per cell for ref k-mer profile.")
    args = ap.parse_args()

    # Expand globs (argparse doesn't expand *)
    pool_dirs = []
    for p in args.pools:
        if any(ch in p for ch in "*?["):
            pool_dirs.extend(sorted(glob.glob(p)))
        else:
            pool_dirs.append(p)
    pool_dirs = [Path(p) for p in pool_dirs if Path(p).is_dir()]
    if not pool_dirs:
        raise SystemExit("no pool directories matched")
    print(f"Found {len(pool_dirs)} pool dirs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading promoter Enformer pool on {device}")
    oracle_pool = PromoterOraclePool(args.oracle_ckpt_dir, device=device)

    print(f"\nBuilding reference k-mer profiles (top-{args.ref_top_k} per cell):")
    ref_kmer = build_reference_kmer_profiles(args.data_csv, top_k=args.ref_top_k)

    rows = []
    for pd_ in pool_dirs:
        try:
            method, task, seed, suffix = detect_pool_and_target(pd_)
        except ValueError as ex:
            print(f"  [skip] {pd_.name}: {ex}")
            continue

        try:
            if method == "ctrldna":
                indices = load_ctrldna_top128(pd_, task, seed)
                metrics = compute_metrics(indices, task, oracle_pool, ref_kmer)
                pool_rows = [("top128_by_reward", metrics)]
            else:
                indices = load_gpa_best_pool(pd_)
                if indices is None:
                    print(f"  [skip] {pd_.name}: no gpa_output_*.h5 file")
                    continue
                # Score full pool once, derive metrics for full and top-128-by-target
                cell_scores = score_pool(indices, oracle_pool)
                metrics_full = _metrics_from_scores(
                    indices, cell_scores, task, ref_kmer,
                )
                pool_rows = [(f"all_{indices.shape[0]}", metrics_full)]
                if indices.shape[0] > 128:
                    top_order = np.argsort(cell_scores[task])[::-1][:128].copy()
                    sub_indices = indices[top_order]
                    sub_scores = {c: cell_scores[c][top_order]
                                  for c in PROMOTER_CELLS}
                    metrics_top = _metrics_from_scores(
                        sub_indices, sub_scores, task, ref_kmer,
                    )
                    pool_rows.append(("top128_by_target", metrics_top))

                    # Strict-unique top-128: dedup by content first, then top-128 by target
                    _, uniq_idx = np.unique(indices, axis=0, return_index=True)
                    uniq_indices = indices[uniq_idx]
                    uniq_cell_scores = {c: cell_scores[c][uniq_idx]
                                        for c in PROMOTER_CELLS}
                    k_unique = min(128, len(uniq_indices))
                    u_order = np.argsort(uniq_cell_scores[task])[::-1][:k_unique].copy()
                    u_sub_indices = uniq_indices[u_order]
                    u_sub_scores = {c: uniq_cell_scores[c][u_order]
                                    for c in PROMOTER_CELLS}
                    metrics_uniq = _metrics_from_scores(
                        u_sub_indices, u_sub_scores, task, ref_kmer,
                    )
                    pool_rows.append(("top128_unique_by_target", metrics_uniq))

                # Archive selection modes (if gpa_output_filtered.h5 exists)
                arch_indices = load_gpa_archive(pd_)
                if arch_indices is not None and arch_indices.shape[0] > 0:
                    arch_cell_scores = score_pool(arch_indices, oracle_pool)
                    # Whole archive pool
                    metrics_arch_all = _metrics_from_scores(
                        arch_indices, arch_cell_scores, task, ref_kmer,
                    )
                    pool_rows.append((f"archive_{arch_indices.shape[0]}", metrics_arch_all))
                    # Top-128 archive by target (argmax)
                    if arch_indices.shape[0] > 128:
                        a_order = np.argsort(arch_cell_scores[task])[::-1][:128].copy()
                        a_sub = arch_indices[a_order]
                        a_scores = {c: arch_cell_scores[c][a_order]
                                    for c in PROMOTER_CELLS}
                        metrics_arch_top = _metrics_from_scores(
                            a_sub, a_scores, task, ref_kmer,
                        )
                        pool_rows.append(("top128_archive_by_target", metrics_arch_top))
                        # Top-128 unique archive by target
                        _, u_idx = np.unique(arch_indices, axis=0, return_index=True)
                        u_arch = arch_indices[u_idx]
                        u_arch_scores = {c: arch_cell_scores[c][u_idx]
                                          for c in PROMOTER_CELLS}
                        ku = min(128, len(u_arch))
                        uo = np.argsort(u_arch_scores[task])[::-1][:ku].copy()
                        u_arch_sub = u_arch[uo]
                        u_arch_sub_scores = {c: u_arch_scores[c][uo]
                                             for c in PROMOTER_CELLS}
                        metrics_arch_uniq = _metrics_from_scores(
                            u_arch_sub, u_arch_sub_scores, task, ref_kmer,
                        )
                        pool_rows.append(("top128_unique_archive_by_target", metrics_arch_uniq))
        except Exception as ex:
            print(f"  [skip] {pd_.name}: load error: {ex}")
            continue

        for selection, metrics in pool_rows:
            row = {
                "pool": pd_.name,
                "method": method,
                "task": task,
                "seed": seed,
                "suffix": suffix.lstrip("_"),
                "selection": selection,
                **metrics,
            }
            rows.append(row)
            print(f"  {pd_.name}  {selection}  "
                  f"{task}={metrics[task.lower()]:+.3f}  "
                  f"comp={metrics['composite']:+.3f}  "
                  f"H={metrics['shannon']:.3f}  "
                  f"motif_r={metrics['motif_corr']:+.3f}")

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}  rows={len(df)}")

    # Print pretty per-task comparison table
    if len(df) > 0:
        cols = ["pool", "selection", "N", "jurkat", "k562", "thp1", "composite", "specificity", "shannon", "motif_corr"]
        print("\n" + df[cols].to_string(index=False,
              formatters={
                  "jurkat": lambda v: f"{v:+.3f}",
                  "k562": lambda v: f"{v:+.3f}",
                  "thp1": lambda v: f"{v:+.3f}",
                  "composite": lambda v: f"{v:+.3f}",
                  "specificity": lambda v: f"{v:+.3f}",
                  "shannon": lambda v: f"{v:.3f}",
                  "motif_corr": lambda v: f"{v:+.3f}" if np.isfinite(v) else "  nan",
              }))


if __name__ == "__main__":
    main()
