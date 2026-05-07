#!/usr/bin/env python3
"""
Enhanced bio-plausibility filters for generated DNA sequences.

Provides per-sequence hard filters (CpG, strand symmetry, homopolymer),
per-sequence Z-score for k-mer matching (tie-breaker ranking),
and pool-level k-mer diagnostics.

Usage:
    from bio_plausibility import is_bio_plausible, bio_plausibility_score
    from bio_plausibility import print_bio_plausibility_report

    # Quick filter
    mask, breakdown = is_bio_plausible(indices)

    # Full report
    ref = compute_reference_stats('model_zoo/lentimpra/lenti_MPRA_K562_data.h5')
    print_bio_plausibility_report(indices, ref_stats=ref, label='Round11')
"""

import numpy as np
import h5py
from pathlib import Path
from scipy.stats import pearsonr


# ============================================================================
# Reference statistics from real data
# ============================================================================

def compute_reference_stats(real_data_h5, split='test', cache_dir=None):
    """Compute reference bio-plausibility statistics from real data.

    Computes and optionally caches: CpG mean/std, 3-mer reference profile,
    per-sequence k-mer correlation stats, strand symmetry percentiles.

    Args:
        real_data_h5: Path to lenti_MPRA data h5 file
        split: Which split to use ('test', 'train', 'valid')
        cache_dir: Directory to cache results (None = no caching)

    Returns:
        dict with reference statistics
    """
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f'bio_plausibility_ref_{split}.npz'
        if cache_path.exists():
            data = dict(np.load(cache_path, allow_pickle=True))
            return data

    with h5py.File(real_data_h5, 'r') as f:
        onehot = f[f'onehot_{split}'][()]  # (N, 230, 4)
    onehot = np.transpose(onehot, (0, 2, 1))  # (N, 4, 230)
    indices = np.argmax(onehot, axis=1)  # (N, L)

    # CpG frequency
    cpg = cpg_frequency(indices)

    # Strand symmetry
    sym = strand_symmetry(indices)

    # 3-mer reference profile (global)
    kmer_3_ref = _kmer_pool_profile(indices, k=3)

    # Per-sequence k-mer correlation with reference
    per_seq_profiles = kmer_profile(indices[:5000], k=3)  # sample for speed
    per_seq_r = np.array([
        pearsonr(kmer_3_ref, per_seq_profiles[i])[0]
        for i in range(len(per_seq_profiles))
    ])

    # Homopolymer stats
    homo = max_homopolymer_run(indices[:5000])

    ref_stats = {
        'cpg_mean': cpg.mean(),
        'cpg_std': cpg.std(),
        'kmer_3_ref_profile': kmer_3_ref,
        'kmer_r_mean': per_seq_r.mean(),
        'kmer_r_std': per_seq_r.std(),
        'strand_sym_at_p95': np.percentile(sym[:, 0], 95),
        'strand_sym_gc_p95': np.percentile(sym[:, 1], 95),
        'homo_max_p99': np.percentile(homo, 99),
        'n_sequences': len(indices),
        'split': split,
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **ref_stats)

    return ref_stats


# ============================================================================
# Per-sequence metrics
# ============================================================================

def cpg_frequency(indices):
    """Per-sequence CpG dinucleotide frequency.

    CpG = C followed by G. In diffusion encoding: C=1, G=2.

    Args:
        indices: (N, L) array of base indices {0=A, 1=C, 2=G, 3=T}

    Returns:
        (N,) array of CpG frequencies (fraction of all dinucleotide positions)
    """
    N, L = indices.shape
    # Vectorized: check if position i is C (1) and position i+1 is G (2)
    is_c = indices[:, :-1] == 1
    is_g = indices[:, 1:] == 2
    cpg_count = (is_c & is_g).sum(axis=1)
    return cpg_count / (L - 1)


def strand_symmetry(indices):
    """Per-sequence strand symmetry (Chargaff's Second Parity Rule).

    Computes |Freq(A) - Freq(T)| and |Freq(G) - Freq(C)| per sequence.

    Args:
        indices: (N, L) array of base indices {0=A, 1=C, 2=G, 3=T}

    Returns:
        (N, 2) array: column 0 = |A-T| asymmetry, column 1 = |G-C| asymmetry
    """
    a_freq = np.mean(indices == 0, axis=1)
    t_freq = np.mean(indices == 3, axis=1)
    g_freq = np.mean(indices == 2, axis=1)
    c_freq = np.mean(indices == 1, axis=1)
    at_asym = np.abs(a_freq - t_freq)
    gc_asym = np.abs(g_freq - c_freq)
    return np.stack([at_asym, gc_asym], axis=1)


def max_homopolymer_run(indices):
    """Per-sequence maximum homopolymer run length for any base.

    Args:
        indices: (N, L) array of base indices {0,1,2,3}

    Returns:
        (N,) array of max homopolymer run lengths
    """
    N, L = indices.shape
    # Detect where the base changes: (N, L-1) boolean
    changes = indices[:, 1:] != indices[:, :-1]
    # Pad with True on both sides to mark boundaries
    boundaries = np.ones((N, L + 1), dtype=bool)
    boundaries[:, 1:-1] = changes
    # For each row, run lengths = diff of boundary positions
    max_runs = np.zeros(N, dtype=int)
    for i in range(N):
        positions = np.where(boundaries[i])[0]
        run_lengths = np.diff(positions)
        max_runs[i] = run_lengths.max() if len(run_lengths) > 0 else L
    return max_runs


def kmer_profile(indices, k=3):
    """Per-sequence k-mer frequency vector.

    Args:
        indices: (N, L) array of base indices {0,1,2,3}
        k: k-mer length (default 3)

    Returns:
        (N, 4^k) array of k-mer frequencies (normalized per sequence)
    """
    N, L = indices.shape
    n_kmers = 4 ** k

    # Vectorized k-mer index: base-4 encoding
    # kmer_idx = indices[:,0:L-k+1]*4^(k-1) + indices[:,1:L-k+2]*4^(k-2) + ...
    kmer_indices = np.zeros((N, L - k + 1), dtype=np.int64)
    for m in range(k):
        kmer_indices = kmer_indices * 4 + indices[:, m:L - k + 1 + m]

    # Count k-mers per sequence using bincount
    profiles = np.zeros((N, n_kmers), dtype=np.float32)
    for i in range(N):
        profiles[i] = np.bincount(kmer_indices[i], minlength=n_kmers)[:n_kmers]

    # Normalize each sequence
    row_sums = profiles.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    profiles /= row_sums

    return profiles


def kmer_zscore(indices, ref_mean=0.51, ref_std=0.14, ref_profile=None, k=3):
    """Per-sequence k-mer Z-score for tie-breaking.

    Z = (r_gen - ref_mean) / ref_std

    where r_gen is the Pearson correlation between the sequence's k-mer
    profile and the reference profile.

    Args:
        indices: (N, L) array of base indices
        ref_mean: Mean k-mer correlation of real sequences (default 0.51)
        ref_std: Std of k-mer correlation of real sequences (default 0.14)
        ref_profile: (4^k,) reference k-mer profile. If None, uses uniform.
        k: k-mer length

    Returns:
        (N,) array of Z-scores
    """
    if ref_profile is None:
        ref_profile = np.ones(4 ** k) / (4 ** k)

    profiles = kmer_profile(indices, k=k)  # (N, 4^k)

    # Vectorized Pearson correlation: r = cov(x,y) / (std(x) * std(y))
    ref = ref_profile.astype(np.float64)
    prof = profiles.astype(np.float64)  # (N, M)
    ref_centered = ref - ref.mean()
    prof_centered = prof - prof.mean(axis=1, keepdims=True)

    numer = prof_centered @ ref_centered  # (N,)
    ref_norm = np.sqrt((ref_centered ** 2).sum())
    prof_norms = np.sqrt((prof_centered ** 2).sum(axis=1))

    denom = ref_norm * prof_norms
    denom[denom == 0] = 1.0  # avoid division by zero
    r_values = numer / denom

    z_scores = (r_values - ref_mean) / ref_std
    return z_scores


# ============================================================================
# Pool-level diagnostics
# ============================================================================

def _kmer_pool_profile(indices, k=3):
    """Compute global (pool-level) k-mer frequency profile.

    Args:
        indices: (N, L) array
        k: k-mer length

    Returns:
        (4^k,) normalized frequency array
    """
    n_kmers = 4 ** k
    N, L = indices.shape

    # Vectorized k-mer index computation
    kmer_indices = np.zeros((N, L - k + 1), dtype=np.int64)
    for m in range(k):
        kmer_indices = kmer_indices * 4 + indices[:, m:L - k + 1 + m]

    # Global bincount over all k-mer indices
    counts = np.bincount(kmer_indices.ravel(), minlength=n_kmers)[:n_kmers].astype(np.float64)

    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def kmer_pool_diagnostics(indices, ref_profile, k=3):
    """Pool-level k-mer diagnostics.

    Args:
        indices: (N, L) array of generated sequences
        ref_profile: (4^k,) reference k-mer profile from real data
        k: k-mer length

    Returns:
        dict with: pool_pearson_r, kl_divergence, per_seq_r_mean,
                   per_seq_r_std, per_seq_r_variance
    """
    # Pool-level profile
    gen_profile = _kmer_pool_profile(indices, k=k)

    # Pool-level Pearson r
    pool_r, _ = pearsonr(ref_profile, gen_profile)

    # KL divergence: D_KL(gen || ref)
    eps = 1e-10
    kl_div = np.sum(gen_profile * np.log((gen_profile + eps) / (ref_profile + eps)))

    # Per-sequence correlation stats (sample for speed)
    sample_size = min(2000, len(indices))
    sample_idx = np.random.choice(len(indices), sample_size, replace=False)
    profiles = kmer_profile(indices[sample_idx], k=k)
    per_seq_r = np.array([
        pearsonr(ref_profile, profiles[i])[0]
        for i in range(sample_size)
    ])

    return {
        'pool_pearson_r': pool_r,
        'kl_divergence': kl_div,
        'per_seq_r_mean': per_seq_r.mean(),
        'per_seq_r_std': per_seq_r.std(),
        'per_seq_r_variance': per_seq_r.var(),
    }


# ============================================================================
# Hard filters
# ============================================================================

def is_bio_plausible(indices, gc_fractions=None, entropy=None,
                     gc_range=(0.40, 0.70), entropy_thresh=1.9,
                     cpg_max=0.072, strand_sym_max=0.15,
                     max_homo_run=12):
    """Enhanced per-sequence bio-plausibility check.

    Applies all hard filters:
    1. GC content within gc_range
    2. Shannon entropy above entropy_thresh
    3. CpG frequency below cpg_max
    4. Strand symmetry (|A-T| and |G-C|) below strand_sym_max
    5. Max homopolymer run below max_homo_run

    Args:
        indices: (N, L) array of base indices {0=A, 1=C, 2=G, 3=T}
        gc_fractions: (N,) precomputed GC fractions, or None to compute
        entropy: (N,) precomputed Shannon entropy, or None to compute
        gc_range: (low, high) GC fraction range
        entropy_thresh: Minimum Shannon entropy
        cpg_max: Maximum CpG dinucleotide frequency
        strand_sym_max: Maximum |A-T| or |G-C| asymmetry
        max_homo_run: Maximum homopolymer run length

    Returns:
        mask: (N,) boolean array
        breakdown: dict with per-filter pass counts
    """
    N = len(indices)

    # GC content
    if gc_fractions is None:
        gc_fractions = np.mean((indices == 1) | (indices == 2), axis=1)
    gc_pass = (gc_fractions >= gc_range[0]) & (gc_fractions <= gc_range[1])

    # Entropy
    if entropy is None:
        entropy = _compute_entropy(indices)
    ent_pass = entropy >= entropy_thresh

    # CpG frequency
    cpg = cpg_frequency(indices)
    cpg_pass = cpg <= cpg_max

    # Strand symmetry
    sym = strand_symmetry(indices)
    sym_pass = (sym[:, 0] < strand_sym_max) & (sym[:, 1] < strand_sym_max)

    # Homopolymer runs
    homo = max_homopolymer_run(indices)
    homo_pass = homo <= max_homo_run

    # Combined mask
    mask = gc_pass & ent_pass & cpg_pass & sym_pass & homo_pass

    breakdown = {
        'total': N,
        'gc_pass': gc_pass.sum(),
        'entropy_pass': ent_pass.sum(),
        'cpg_pass': cpg_pass.sum(),
        'strand_sym_pass': sym_pass.sum(),
        'homopolymer_pass': homo_pass.sum(),
        'all_pass': mask.sum(),
    }

    return mask, breakdown


def _compute_entropy(indices):
    """Compute per-sequence Shannon entropy.

    Args:
        indices: (N, L) array of base indices

    Returns:
        (N,) array of entropy values
    """
    L = indices.shape[1]
    # Count each base per sequence: (N, 4)
    counts = np.stack([(indices == b).sum(axis=1) for b in range(4)], axis=1).astype(np.float64)
    freqs = counts / L
    # entropy = -sum(f * log2(f)) with masking for f>0
    with np.errstate(divide='ignore', invalid='ignore'):
        log_freqs = np.log2(freqs)
    log_freqs[freqs == 0] = 0.0
    entropies = -np.sum(freqs * log_freqs, axis=1)
    return entropies


# ============================================================================
# Composite bio-plausibility score (for ranking, not filtering)
# ============================================================================

def bio_plausibility_score(indices, gc_fractions=None, entropy=None,
                           ref_stats=None, weights=None):
    """Composite bio-plausibility score per sequence.

    Each component is mapped to [0, 1] where 1 = perfectly matches real data.
    The composite score is a weighted sum normalized to [0, 1].

    Components:
    1. GC match: Gaussian around real GC mean (~0.52)
    2. Entropy match: Sigmoid around entropy threshold
    3. CpG match: Gaussian penalty for high CpG
    4. Symmetry match: Penalty for high strand asymmetry
    5. Homopolymer match: Penalty for long runs
    6. Z_kmer: Sigmoid-normalized k-mer Z-score

    Args:
        indices: (N, L) array of base indices
        gc_fractions: (N,) precomputed GC fractions
        entropy: (N,) precomputed Shannon entropy
        ref_stats: dict from compute_reference_stats()
        weights: dict of component weights (default: equal)

    Returns:
        (N,) array of composite scores in [0, 1]
    """
    N = len(indices)

    if gc_fractions is None:
        gc_fractions = np.mean((indices == 1) | (indices == 2), axis=1)
    if entropy is None:
        entropy = _compute_entropy(indices)

    # Default reference values
    gc_target = 0.52
    cpg_ref_mean = 0.036
    kmer_ref_mean = 0.51
    kmer_ref_std = 0.14
    kmer_ref_profile = None

    if ref_stats is not None:
        gc_target = 0.52  # real test data GC mean
        cpg_ref_mean = float(ref_stats.get('cpg_mean', 0.036))
        kmer_ref_mean = float(ref_stats.get('kmer_r_mean', 0.51))
        kmer_ref_std = float(ref_stats.get('kmer_r_std', 0.14))
        kmer_ref_profile = ref_stats.get('kmer_3_ref_profile', None)

    default_weights = {
        'gc': 1.0, 'entropy': 1.0, 'cpg': 1.0,
        'symmetry': 1.0, 'homopolymer': 1.0, 'kmer': 1.0
    }
    if weights is not None:
        default_weights.update(weights)
    w = default_weights

    # 1. GC match: Gaussian around target, sigma=0.08
    gc_score = np.exp(-0.5 * ((gc_fractions - gc_target) / 0.08) ** 2)

    # 2. Entropy match: sigmoid around 1.9
    ent_score = 1.0 / (1.0 + np.exp(-20 * (entropy - 1.9)))

    # 3. CpG match: Gaussian penalty for deviation from real mean
    cpg = cpg_frequency(indices)
    cpg_score = np.exp(-0.5 * ((cpg - cpg_ref_mean) / 0.03) ** 2)

    # 4. Symmetry match: penalty for high asymmetry
    sym = strand_symmetry(indices)
    sym_score = np.exp(-0.5 * (sym / 0.10) ** 2).mean(axis=1)

    # 5. Homopolymer match: penalty for long runs
    homo = max_homopolymer_run(indices)
    homo_score = np.where(homo <= 6, 1.0,
                 np.where(homo <= 12, 1.0 - (homo - 6) / 12.0, 0.0))

    # 6. k-mer Z-score (sigmoid-normalized)
    if kmer_ref_profile is not None:
        z = kmer_zscore(indices, ref_mean=kmer_ref_mean,
                        ref_std=kmer_ref_std, ref_profile=kmer_ref_profile, k=3)
    else:
        z = np.zeros(N)
    kmer_score = 1.0 / (1.0 + np.exp(-z))  # sigmoid

    # Weighted sum
    total_weight = sum(w.values())
    composite = (
        w['gc'] * gc_score +
        w['entropy'] * ent_score +
        w['cpg'] * cpg_score +
        w['symmetry'] * sym_score +
        w['homopolymer'] * homo_score +
        w['kmer'] * kmer_score
    ) / total_weight

    return composite


# ============================================================================
# Reporting
# ============================================================================

def print_bio_plausibility_report(indices, gc_fractions=None, entropy=None,
                                  ref_stats=None, label=''):
    """Print formatted bio-plausibility report.

    Args:
        indices: (N, L) array of base indices
        gc_fractions: (N,) precomputed GC fractions
        entropy: (N,) precomputed Shannon entropy
        ref_stats: dict from compute_reference_stats()
        label: Label for the sequence set
    """
    N = len(indices)
    header = f"BIO-PLAUSIBILITY REPORT: {label}" if label else "BIO-PLAUSIBILITY REPORT"
    print("=" * 72)
    print(header)
    print("=" * 72)
    print(f"Sequences: {N:,}")

    # Compute metrics
    if gc_fractions is None:
        gc_fractions = np.mean((indices == 1) | (indices == 2), axis=1)
    if entropy is None:
        entropy = _compute_entropy(indices)

    cpg = cpg_frequency(indices)
    sym = strand_symmetry(indices)
    homo = max_homopolymer_run(indices)

    # Hard filter results
    mask, breakdown = is_bio_plausible(indices, gc_fractions, entropy)

    print(f"\n--- Hard Filters ---")
    print(f"  {'Filter':<30} {'Pass':>8} {'Rate':>8}")
    print(f"  {'-'*48}")
    print(f"  {'GC [40-60%]':<30} {breakdown['gc_pass']:>8,} {breakdown['gc_pass']/N*100:>7.1f}%")
    print(f"  {'Entropy > 1.9':<30} {breakdown['entropy_pass']:>8,} {breakdown['entropy_pass']/N*100:>7.1f}%")
    print(f"  {'CpG <= 7.2%':<30} {breakdown['cpg_pass']:>8,} {breakdown['cpg_pass']/N*100:>7.1f}%")
    print(f"  {'Strand symmetry < 0.15':<30} {breakdown['strand_sym_pass']:>8,} {breakdown['strand_sym_pass']/N*100:>7.1f}%")
    print(f"  {'Homopolymer <= 12bp':<30} {breakdown['homopolymer_pass']:>8,} {breakdown['homopolymer_pass']/N*100:>7.1f}%")
    print(f"  {'ALL PASS':<30} {breakdown['all_pass']:>8,} {breakdown['all_pass']/N*100:>7.1f}%")

    print(f"\n--- Per-Sequence Metrics ---")
    print(f"  {'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*62}")
    print(f"  {'GC fraction':<30} {gc_fractions.mean():>10.4f} {gc_fractions.std():>10.4f} {gc_fractions.min():>10.4f} {gc_fractions.max():>10.4f}")
    print(f"  {'Entropy':<30} {entropy.mean():>10.4f} {entropy.std():>10.4f} {entropy.min():>10.4f} {entropy.max():>10.4f}")
    print(f"  {'CpG frequency':<30} {cpg.mean():>10.4f} {cpg.std():>10.4f} {cpg.min():>10.4f} {cpg.max():>10.4f}")
    print(f"  {'|A-T| asymmetry':<30} {sym[:,0].mean():>10.4f} {sym[:,0].std():>10.4f} {sym[:,0].min():>10.4f} {sym[:,0].max():>10.4f}")
    print(f"  {'|G-C| asymmetry':<30} {sym[:,1].mean():>10.4f} {sym[:,1].std():>10.4f} {sym[:,1].min():>10.4f} {sym[:,1].max():>10.4f}")
    print(f"  {'Max homopolymer run':<30} {homo.mean():>10.1f} {homo.std():>10.1f} {homo.min():>10.0f} {homo.max():>10.0f}")

    # Pool-level k-mer diagnostics
    if ref_stats is not None and 'kmer_3_ref_profile' in ref_stats:
        ref_profile = ref_stats['kmer_3_ref_profile']
        if isinstance(ref_profile, np.ndarray) and len(ref_profile) == 64:
            print(f"\n--- Pool-Level k-mer Diagnostics (k=3) ---")
            diag = kmer_pool_diagnostics(indices, ref_profile, k=3)
            print(f"  Pool Pearson r:    {diag['pool_pearson_r']:.4f}  (target: > 0.90)")
            print(f"  KL Divergence:     {diag['kl_divergence']:.6f}  (target: minimize)")
            print(f"  Per-seq r mean:    {diag['per_seq_r_mean']:.4f}  (real ref: {float(ref_stats.get('kmer_r_mean', 0.51)):.4f})")
            print(f"  Per-seq r std:     {diag['per_seq_r_std']:.4f}  (real ref: {float(ref_stats.get('kmer_r_std', 0.14)):.4f})")

    # Composite score
    scores = bio_plausibility_score(indices, gc_fractions, entropy, ref_stats)
    print(f"\n--- Composite Bio-Plausibility Score ---")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Std:    {scores.std():.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Max:    {scores.max():.4f}")
    print()

    return mask, breakdown, scores
