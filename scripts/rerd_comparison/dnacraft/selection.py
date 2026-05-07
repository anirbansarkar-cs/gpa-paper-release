"""Top-128 selection rules for DNA-CRAFT-aligned scoring.

Two modes (per plan §3.5):
  - "by_mingap"   — paper-faithful: top-128 by per-seq MinGap
  - "by_composite" — alternative: top-128 by composite rank of
                     (per-seq motif Pearson, per-seq 3-mer Pearson, per-seq
                     leave-one-out diversity contribution).

Per-seq surrogate scores are computed against the same DNA-CRAFT reference
used for pool-level metric reporting, so the selection is consistent with
the metric being optimized. Both methods select 128 from a candidate pool.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def per_seq_mingap(cell_scores: np.ndarray, target_idx: int) -> np.ndarray:
    """cell_scores: (N, 3) [HepG2, K562, SKNSH] from Eval-Model.
    Returns (N,) of MinGap = target − max(off-targets)."""
    off = [i for i in range(cell_scores.shape[1]) if i != target_idx]
    return cell_scores[:, target_idx] - cell_scores[:, off].max(axis=1)


def per_seq_kmer_pearson(seq_kmer_freq: np.ndarray, ref_kmer_freq: np.ndarray) -> np.ndarray:
    """Per-seq Pearson correlation between each seq's 3-mer freq vector and
    the reference 3-mer freq vector.

    seq_kmer_freq: (N, 64) per-seq L1-normalized 3-mer frequency
    ref_kmer_freq: (64,) reference vector (L1-normalized)
    """
    # Center both
    seq_c = seq_kmer_freq - seq_kmer_freq.mean(axis=1, keepdims=True)
    ref_c = ref_kmer_freq - ref_kmer_freq.mean()
    # Pearson per seq
    num = (seq_c * ref_c).sum(axis=1)
    den = np.sqrt((seq_c ** 2).sum(axis=1) * (ref_c ** 2).sum())
    out = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return out


def per_seq_motif_pearson(seq_motif_counts: np.ndarray, ref_motif_freq: np.ndarray) -> np.ndarray:
    """Per-seq Pearson on motif count vectors vs ref motif freq.
    seq_motif_counts: (N, M) per-seq motif counts (raw or L1-normalized; only
                     correlation matters)
    ref_motif_freq: (M,) reference motif freq vector (aligned columns)
    """
    seq_c = seq_motif_counts - seq_motif_counts.mean(axis=1, keepdims=True)
    ref_c = ref_motif_freq - ref_motif_freq.mean()
    num = (seq_c * ref_c).sum(axis=1)
    den = np.sqrt((seq_c ** 2).sum(axis=1) * (ref_c ** 2).sum())
    out = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return out


def per_seq_diversity_contribution(seqs_int: np.ndarray) -> np.ndarray:
    """Leave-one-out diversity gain proxy: for each seq, compute how much
    average per-position uniqueness it adds vs the rest of the pool.
    Cheap proxy = mean over positions of indicator (this seq's base differs
    from the most-common base at that position in the rest of the pool).

    seqs_int: (N, L) int array {0,1,2,3}
    Returns (N,) array — higher = more diverse contribution.
    """
    N, L = seqs_int.shape
    # Per-position mode (most common base)
    counts = np.stack([(seqs_int == b).sum(axis=0) for b in range(4)], axis=0)  # (4, L)
    mode = counts.argmax(axis=0)  # (L,)
    # Each seq's fraction of positions where it differs from the population mode
    diffs = (seqs_int != mode[None, :]).mean(axis=1)
    return diffs


def select_top_n_by_mingap(cell_scores: np.ndarray, target_idx: int, n: int = 128) -> np.ndarray:
    """Returns indices of top-n seqs by per-seq MinGap (descending)."""
    mg = per_seq_mingap(cell_scores, target_idx)
    order = np.argsort(mg)[::-1].copy()
    return order[:n]


def select_top_n_by_composite(
    seq_kmer_freq: np.ndarray,
    seq_motif_counts: np.ndarray,
    ref_kmer_freq: np.ndarray,
    ref_motif_freq: np.ndarray,
    seqs_int: np.ndarray,
    n: int = 128,
) -> np.ndarray:
    """Returns indices of top-n seqs by composite rank of (per-seq motif Pearson,
    per-seq 3-mer Pearson, per-seq diversity contribution).

    All three components ranked separately, then averaged. Higher = better.
    """
    motif_r = per_seq_motif_pearson(seq_motif_counts, ref_motif_freq)
    kmer_r = per_seq_kmer_pearson(seq_kmer_freq, ref_kmer_freq)
    div = per_seq_diversity_contribution(seqs_int)

    motif_rank = rankdata(motif_r) / len(motif_r)
    kmer_rank = rankdata(kmer_r) / len(kmer_r)
    div_rank = rankdata(div) / len(div)
    composite = (motif_rank + kmer_rank + div_rank) / 3.0
    order = np.argsort(composite)[::-1].copy()
    return order[:n]


def select_top_n_by_motif(
    seq_motif_counts: np.ndarray,
    ref_motif_freq: np.ndarray,
    n: int = 128,
) -> np.ndarray:
    """Greedy max-Spearman(pool-sum motif vector, ref motif freq) selection.

    The reported metric is Spearman ρ between the pool-mean motif count
    vector and the reference. Picking top-128 by per-seq motif Pearson
    (the old proxy) does NOT optimize this set-level rank-correlation:
    it tends to over-saturate the pool on a few motifs that the ref
    also peaks on, which actually drops the broader-distribution Spearman.

    Greedy: maintain running pool-sum motif vector; at each step add the
    candidate whose insertion maximizes Spearman(new_pool_sum, ref).
    """
    N, M = seq_motif_counts.shape
    if N <= n:
        return np.arange(N, dtype=np.int64)

    ref_ranks = rankdata(ref_motif_freq)
    ref_ranks_centered = ref_ranks - ref_ranks.mean()
    ref_norm = float(np.sqrt((ref_ranks_centered ** 2).sum()))

    pool_sum = np.zeros(M, dtype=np.float64)
    selected = []
    avail = np.ones(N, dtype=bool)

    # Seed: best individual motif Pearson — gives non-zero starting Spearman
    # so the first greedy step has a real ranking to compare against.
    seed_scores = per_seq_motif_pearson(seq_motif_counts, ref_motif_freq)
    seed_idx = int(np.argmax(seed_scores))
    selected.append(seed_idx)
    avail[seed_idx] = False
    pool_sum += seq_motif_counts[seed_idx].astype(np.float64)

    for _ in range(n - 1):
        idx_avail = np.where(avail)[0]
        cands = seq_motif_counts[idx_avail].astype(np.float64)  # (Na, M)
        new_sums = pool_sum[None, :] + cands                     # (Na, M)
        # Vectorized rank with ties handled via scipy rankdata
        ranks = rankdata(new_sums, axis=1).astype(np.float64)    # (Na, M)
        rc = ranks - ranks.mean(axis=1, keepdims=True)           # (Na, M)
        num = (rc * ref_ranks_centered[None, :]).sum(axis=1)     # (Na,)
        rc_norm = np.sqrt((rc ** 2).sum(axis=1))                 # (Na,)
        with np.errstate(divide="ignore", invalid="ignore"):
            spearman = np.where(rc_norm * ref_norm > 0,
                                num / (rc_norm * ref_norm), 0.0)
        best_local = int(np.argmax(spearman))
        best_idx = int(idx_avail[best_local])
        selected.append(best_idx)
        avail[best_idx] = False
        pool_sum += seq_motif_counts[best_idx].astype(np.float64)

    return np.array(selected, dtype=np.int64)


def select_top_n_by_pool_composite(
    seq_motif_counts: np.ndarray,
    seq_kmer_freq: np.ndarray,
    seqs_int: np.ndarray,
    ref_motif_freq: np.ndarray,
    ref_kmer_freq: np.ndarray,
    per_seq_mingap: "np.ndarray | None" = None,
    n: int = 64,
    weights: tuple = (1.0, 1.0, 1.0, 1.0),
    mingap_floor_pct: float = 0.0,
) -> np.ndarray:
    """Greedy joint maximization of pool-aggregate metrics.

    At each step add the candidate that maximizes the rank-averaged blend of
        (i)   pool-Spearman(motif_sum, ref_motif_freq)
        (ii)  pool-Pearson(kmer_sum, ref_kmer_freq)
        (iii) pool-Shannon(per-position entropy)
        (iv)  pool-mean(MinGap)               [if per_seq_mingap given]

    Each metric is reranked across the available candidates at that step;
    the candidate with the highest rank-average wins. This is the "matched
    selection rule" for the pool-aggregate scoring used in the DNA-CRAFT
    paper protocol — directly maximizes the metric being reported.

    Parameters
    ----------
    weights : 4-tuple of floats (motif, kmer, shannon, mingap). Components
              with weight 0 are dropped from the rank blend. Default = all 1.
    mingap_floor_pct : if >0, exclude candidates whose per_seq_mingap is
              below this percentile (filters out grammar-rich-but-low-spec
              outliers). Requires per_seq_mingap.

    Complexity: O(n * Na * (M + K + L)) per step. Vectorised in numpy;
    typically a few seconds for pool_size=5000, n=64.
    """
    N, M = seq_motif_counts.shape
    Nk, K = seq_kmer_freq.shape
    Ns, L = seqs_int.shape
    assert N == Nk == Ns, "shape mismatch across seq_*"

    if N <= n:
        return np.arange(N, dtype=np.int64)

    # --- MinGap floor filter (optional) ---
    avail = np.ones(N, dtype=bool)
    if per_seq_mingap is not None and mingap_floor_pct > 0:
        floor = float(np.percentile(per_seq_mingap, mingap_floor_pct))
        avail &= (per_seq_mingap >= floor)
        if avail.sum() < n:
            # Floor too aggressive — relax to keep at least n candidates.
            order = np.argsort(-per_seq_mingap)
            avail = np.zeros(N, dtype=bool)
            avail[order[: max(n, int(N * 0.5))]] = True

    # --- Reference vectors for Spearman / Pearson ---
    ref_motif_ranks = rankdata(ref_motif_freq)
    rmr_centered = ref_motif_ranks - ref_motif_ranks.mean()
    rmr_norm = float(np.sqrt((rmr_centered ** 2).sum()))

    ref_kmer_centered = ref_kmer_freq - ref_kmer_freq.mean()
    rkc_norm = float(np.sqrt((ref_kmer_centered ** 2).sum()))

    seq_motif_f = seq_motif_counts.astype(np.float64)
    seq_kmer_f = seq_kmer_freq.astype(np.float64)
    pos = np.arange(L)

    # --- Running pool state ---
    pool_motif_sum = np.zeros(M, dtype=np.float64)
    pool_kmer_sum = np.zeros(K, dtype=np.float64)
    pool_pos_counts = np.zeros((L, 4), dtype=np.int64)
    pool_mingap_sum = 0.0
    selected: list[int] = []

    use_mingap = per_seq_mingap is not None and weights[3] > 0
    use_motif = weights[0] > 0
    use_kmer = weights[1] > 0
    use_shannon = weights[2] > 0

    for t in range(n):
        idx_avail = np.where(avail)[0]
        if idx_avail.size == 0:
            break
        Na = idx_avail.size

        comps = []  # rank-blended components per candidate

        # --- (i) pool-Spearman motif ---
        if use_motif:
            new_motif_sums = pool_motif_sum[None, :] + seq_motif_f[idx_avail]
            ranks = rankdata(new_motif_sums, axis=1).astype(np.float64)
            rc = ranks - ranks.mean(axis=1, keepdims=True)
            num = (rc * rmr_centered[None, :]).sum(axis=1)
            rc_norm = np.sqrt((rc ** 2).sum(axis=1))
            with np.errstate(divide="ignore", invalid="ignore"):
                spearman = np.where(rc_norm * rmr_norm > 0,
                                    num / (rc_norm * rmr_norm), 0.0)
            comps.append(weights[0] * rankdata(spearman))

        # --- (ii) pool-Pearson 3-mer ---
        if use_kmer:
            new_kmer_sums = pool_kmer_sum[None, :] + seq_kmer_f[idx_avail]
            kc = new_kmer_sums - new_kmer_sums.mean(axis=1, keepdims=True)
            knum = (kc * ref_kmer_centered[None, :]).sum(axis=1)
            kc_norm = np.sqrt((kc ** 2).sum(axis=1))
            with np.errstate(divide="ignore", invalid="ignore"):
                kmer_pearson = np.where(kc_norm * rkc_norm > 0,
                                        knum / (kc_norm * rkc_norm), 0.0)
            comps.append(weights[1] * rankdata(kmer_pearson))

        # --- (iii) pool-Shannon (entropy delta-table approach) ---
        if use_shannon and t > 0:
            T = t  # current selected size
            c = pool_pos_counts.astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                xlogx = np.where(c > 0, c * np.log2(c), 0.0)
                term_old = xlogx.sum(axis=1)
                xlogx_plus = (c + 1) * np.log2(c + 1)
                term_new_full = term_old[:, None] - xlogx + xlogx_plus
                H_old = np.log2(T) - term_old / T
                H_new = np.log2(T + 1) - term_new_full / (T + 1)
            delta = H_new - H_old[:, None]
            cand_letters = seqs_int[idx_avail]
            shannon_scores = delta[pos[None, :], cand_letters].sum(axis=1)
            comps.append(weights[2] * rankdata(shannon_scores))
        elif use_shannon and t == 0:
            # On first pick the pool is empty → Shannon undefined. Use
            # per-seq mode-distance as a sensible seed proxy.
            seed_proxy = per_seq_diversity_contribution(seqs_int[idx_avail])
            comps.append(weights[2] * rankdata(seed_proxy))

        # --- (iv) pool-mean MinGap ---
        if use_mingap:
            mingap_mean = (pool_mingap_sum + per_seq_mingap[idx_avail]) / (t + 1)
            comps.append(weights[3] * rankdata(mingap_mean))

        if not comps:
            raise ValueError("at least one of (motif, kmer, shannon, mingap) "
                             "must have positive weight")
        composite = np.mean(np.stack(comps, axis=0), axis=0)
        best_local = int(np.argmax(composite))
        best_idx = int(idx_avail[best_local])

        selected.append(best_idx)
        avail[best_idx] = False
        pool_motif_sum += seq_motif_f[best_idx]
        pool_kmer_sum += seq_kmer_f[best_idx]
        pool_pos_counts[pos, seqs_int[best_idx]] += 1
        if per_seq_mingap is not None:
            pool_mingap_sum += float(per_seq_mingap[best_idx])

    return np.array(selected, dtype=np.int64)


def select_top_n_by_kmer3(
    seq_kmer_freq: np.ndarray,
    ref_kmer_freq: np.ndarray,
    n: int = 128,
) -> np.ndarray:
    """Returns indices of top-n seqs by per-seq 3-mer Pearson against ref."""
    scores = per_seq_kmer_pearson(seq_kmer_freq, ref_kmer_freq)
    order = np.argsort(scores)[::-1].copy()
    return order[:n]


def select_top_n_by_diversity(
    seqs_int: np.ndarray,
    n: int = 128,
) -> np.ndarray:
    """Greedy max-Shannon-entropy selection: at each step add the sequence
    whose insertion gives the largest gain in per-position Shannon entropy
    (in bits) of the selected set's base distribution.

    Per-seq mode-distance (the old proxy) maximizes individual deviance from
    the population mode but does NOT maximize set-level entropy — picking
    128 maximally mode-deviant seqs can give a homogeneous "anti-mode"
    cluster with zero internal diversity. This greedy directly optimizes
    the metric being reported.

    seqs_int: (N, L) int array {0,1,2,3}.
    Returns sorted-by-selection-order indices (length n).
    """
    N, L = seqs_int.shape
    if N <= n:
        return np.arange(N, dtype=np.int64)
    counts = np.zeros((L, 4), dtype=np.int64)
    selected = []
    avail = np.ones(N, dtype=bool)

    # Seed: pick the per-seq mode-distance argmax to break the all-zero
    # entropy tie deterministically.
    seed_scores = per_seq_diversity_contribution(seqs_int)
    seed_idx = int(np.argmax(seed_scores))
    selected.append(seed_idx)
    avail[seed_idx] = False
    pos = np.arange(L)
    counts[pos, seqs_int[seed_idx]] += 1

    for _ in range(n - 1):
        T = len(selected)
        # Current per-position entropy term
        # H_old[j] = -sum_b p_b * log2(p_b) where p_b = counts[j, b] / T
        #   = log2(T) - (1/T) * sum_b counts[j, b] * log2(counts[j, b])
        # Adding a seq with base b at position j → counts[j, b] += 1, T → T+1
        # New H_new[j] = log2(T+1) - (1/(T+1)) * (sum_b counts[j, b] * log2(counts[j, b])
        #                                          - counts[j, b'] * log2(counts[j, b'])
        #                                          + (counts[j, b']+1) * log2(counts[j, b']+1))
        # where b' is the new seq's base at j.
        # Gain[j, b'] depends only on counts[j, b']. Precompute a (L, 4) delta table.
        c = counts.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            xlogx = np.where(c > 0, c * np.log2(c), 0.0)               # (L, 4)
            term_old = xlogx.sum(axis=1)                                # (L,)
            xlogx_plus = (c + 1) * np.log2(c + 1)                       # (L, 4)
            # New term for column b': sum_b c*log2(c) − c[b']*log2(c[b']) + (c[b']+1)*log2(c[b']+1)
            term_new_full = term_old[:, None] - xlogx + xlogx_plus      # (L, 4)
            H_old = np.log2(T) - term_old / T                           # (L,)
            H_new = np.log2(T + 1) - term_new_full / (T + 1)            # (L, 4)
        delta = H_new - H_old[:, None]  # (L, 4)

        idx_avail = np.where(avail)[0]
        cand = seqs_int[idx_avail]  # (M, L) base ints
        # gains[m] = sum over j of delta[j, cand[m, j]]
        gains = delta[np.arange(L)[None, :], cand].sum(axis=1)  # (M,)
        best_local = int(np.argmax(gains))
        best_idx = int(idx_avail[best_local])
        selected.append(best_idx)
        avail[best_idx] = False
        counts[pos, seqs_int[best_idx]] += 1

    return np.array(selected, dtype=np.int64)
