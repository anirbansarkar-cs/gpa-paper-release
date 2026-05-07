#!/usr/bin/env python3
"""Unified top-640 ranking — selection prefers high JAS + high 3mer + high ATAC.
ATAC is NOT a constraint, just enters the rank-sum with weight 0.5.

Caches per-seq mcorr/kcorr/oracle/kmer_counts on first run so subsequent
re-rank runs (different selection criteria) take <2 min instead of ~22 min.

Usage:
  python _unified_rank_motif_kmer_atac.py            # default selection
  ATAC_W=1.0 python _unified_rank_motif_kmer_atac.py # heavier ATAC weight
"""
import os, sys, glob
from pathlib import Path
import numpy as np, h5py, pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, rankdata
from itertools import product

REF_CACHE = '${GPA_REPO_ROOT}/results/rerd_comparison/drakes_protocol/_ref_cache'
PHASEC = '${GPA_REPO_ROOT}/results/rerd_comparison'
sys.path.insert(0, '${GPA_EXTERNAL_ROOT}/DRAKES/drakes_dna')
import dataloader_gosai
dataloader_gosai.base_path = '${GPA_EXTERNAL_ROOT}/DRAKES_data/data_and_model/'

# Lazy-load gReLU oracle on first ATAC cache miss
sys.path.insert(0, '${GPA_REPO_ROOT}/scripts/rerd_comparison')
from score_drakes_protocol import (
    load_atac_oracle, _grelu_batch_score, HEPG2_ATAC_IDX as _HEPG2_ATAC_IDX,
)
_ATAC_MODEL = None

ATAC_W = float(os.environ.get('ATAC_W', '0.5'))
HEPG2_IDX = 1

ref_motif = pd.read_csv(f'{REF_CACHE}/motif_count_top999.csv', index_col=0).iloc[:, 0]
ref_kmer = np.load(f'{REF_CACHE}/highexp_999.npz', allow_pickle=True)
ref_kmer_dict = ref_kmer['kmers'].item()
n_highexp = int(ref_kmer['n'])
kmer_keys = [''.join(p) for p in product('ACGT', repeat=3)]
ref_kvec = np.array([ref_kmer_dict.get(k, 0.0) for k in kmer_keys], dtype=np.float32)


def per_seq_kmer(seqs, k=3):
    L = len(seqs[0])
    arr = np.frombuffer(''.join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), L)
    d = np.full(arr.shape, -1, dtype=np.int32)
    for i, c in enumerate('ACGT'):
        d[arr == ord(c)] = i
    n_k = L - k + 1
    kidx = d[:, :n_k].astype(np.int64) * 16 + d[:, 1:1+n_k] * 4 + d[:, 2:2+n_k]
    valid = (d[:, :n_k] >= 0) & (d[:, 1:1+n_k] >= 0) & (d[:, 2:2+n_k] >= 0)
    flat = kidx + np.arange(arr.shape[0], dtype=np.int64)[:, None] * 64
    return np.bincount(flat[valid], minlength=arr.shape[0]*64).reshape(arr.shape[0], 64).astype(np.float32)


def get_or_build_perseq(pool):
    """Returns (csr, motifs_list, all_m, rv, mcorr, kcorr, kmer_mat, oracle, app, atac).
    Caches mcorr/kcorr/kmer_mat/oracle as npy files keyed by pool."""
    suffix, h5n = '', 'gpa_output_pool.h5'

    npz = f'{REF_CACHE}/pool_motif_perseq_{pool}{suffix}.npz'
    h5p = f'{PHASEC}/{pool}/{h5n}'
    app_p = f'{REF_CACHE}/pool_app_ll_perseq_{pool}{suffix}.npy'
    atc_p = f'{REF_CACHE}/pool_atac_perseq_{pool}{suffix}.npy'
    # ATAC cache is built lazily below — only require the upstream artifacts
    if not all(os.path.exists(p) for p in (npz, h5p, app_p)):
        return None

    csr = sparse.load_npz(npz).tocsr().astype(np.float32)
    motifs = np.load(npz.replace('.npz', '_motifs.npy'), allow_pickle=True).tolist()
    all_m = sorted(set(motifs) | set(ref_motif.index))
    rv = np.array([ref_motif.get(m, 0.0) for m in all_m], dtype=np.float32)
    pa_app = np.load(app_p)

    # ---- per-seq ATAC predictions (lazy build via gReLU) ----
    if os.path.exists(atc_p):
        pa_atac = np.load(atc_p)[:, HEPG2_IDX]
    else:
        global _ATAC_MODEL
        if _ATAC_MODEL is None:
            print(f'  [ATAC] loading gReLU oracle (first cache miss)...', flush=True)
            _ATAC_MODEL = load_atac_oracle()
        # Need raw seqs to score
        with h5py.File(h5p, 'r') as f:
            idx_for_atac = f['indices'][:].astype(np.int64)
        seqs_for_atac = dataloader_gosai.batch_dna_detokenize(idx_for_atac)
        atac_preds = _grelu_batch_score(seqs_for_atac, _ATAC_MODEL, batch_size=64)
        np.save(atc_p, atac_preds)
        print(f'  [ATAC] {pool}{suffix} {len(seqs_for_atac)} seqs, cached', flush=True)
        pa_atac = atac_preds[:, HEPG2_IDX]

    # ---- per-seq motif Pearson ----
    mcorr_p = f'{REF_CACHE}/pool_mcorr_perseq_{pool}{suffix}.npy'
    if os.path.exists(mcorr_p):
        mcorr = np.load(mcorr_p)
    else:
        rv_motifs = np.array([ref_motif.get(m, 0.0) for m in motifs], dtype=np.float32)
        row_sum = np.asarray(csr.sum(axis=1)).ravel()
        rv_mean = rv.mean()
        cov = csr @ rv_motifs - rv_mean * row_sum
        sq_sum = np.asarray(csr.multiply(csr).sum(axis=1)).ravel()
        pool_norm_sq = sq_sum - row_sum*row_sum / len(all_m)
        pool_norm = np.sqrt(np.clip(pool_norm_sq, 0, None)) + 1e-12
        rv_norm = np.linalg.norm(rv - rv_mean) + 1e-12
        mcorr = (cov / (pool_norm * rv_norm)).astype(np.float32)
        np.save(mcorr_p, mcorr)

    # ---- per-seq oracle ----
    oracle_p = f'{REF_CACHE}/pool_oracle_perseq_{pool}{suffix}.npy'
    if os.path.exists(oracle_p):
        oracle = np.load(oracle_p)
    else:
        with h5py.File(h5p, 'r') as f:
            for k in ('oracle_hepg2_eval', 'oracle_hepg2', 'oracle_preds'):
                if k in f:
                    oracle = f[k][:].astype(np.float32); break
        np.save(oracle_p, oracle)

    # ---- per-seq kmer counts ----
    km_p = f'{REF_CACHE}/pool_kmercnt_perseq_{pool}{suffix}.npy'
    if os.path.exists(km_p):
        km = np.load(km_p)
    else:
        with h5py.File(h5p, 'r') as f:
            idx = f['indices'][:].astype(np.int64)
        seqs = dataloader_gosai.batch_dna_detokenize(idx)
        km = per_seq_kmer(seqs)
        np.save(km_p, km)

    # ---- per-seq 3mer Pearson ----
    kcorr_p = f'{REF_CACHE}/pool_kcorr_perseq_{pool}{suffix}.npy'
    if os.path.exists(kcorr_p):
        kcorr = np.load(kcorr_p)
    else:
        rkc = ref_kvec - ref_kvec.mean(); rkn = np.linalg.norm(rkc) + 1e-12
        pkc = km - km.mean(axis=1, keepdims=True); pkn = np.linalg.norm(pkc, axis=1) + 1e-12
        kcorr = ((pkc @ rkc) / (pkn * rkn)).astype(np.float32)
        np.save(kcorr_p, kcorr)

    return dict(csr=csr, motifs=motifs, all_m=all_m, rv=rv,
                mcorr=mcorr, kcorr=kcorr, km=km, oracle=oracle,
                app=pa_app, atac=pa_atac)


def select_and_score(pool, ctx, atac_w=ATAC_W, app_min=-259.0):
    """Apply selection: app-LL >= app_min, then top by (motif rank + kmer rank + atac_w*atac rank).
    If too few valid, fill with highest-ranked from invalid set."""
    rmot = rankdata(ctx['mcorr'])
    rkm = rankdata(ctx['kcorr'])
    ratac = rankdata(ctx['atac'])
    composite = rmot + rkm + atac_w * ratac
    N = min(640, len(ctx['mcorr']))

    valid = np.where(ctx['app'] >= app_min)[0]
    if len(valid) >= N:
        order = valid[np.argsort(-composite[valid])[:N]]
    elif len(valid) > 0:
        invalid = np.where(ctx['app'] < app_min)[0]
        fill = N - len(valid)
        top_inv = invalid[np.argsort(-composite[invalid])[:fill]]
        order = np.concatenate([valid, top_inv])
    else:
        order = np.argsort(-composite)[:N]

    sub_m = np.asarray(ctx['csr'][order].sum(axis=0)).ravel()
    motif_idx_map = {m: i for i, m in enumerate(ctx['all_m'])}
    sub_m_full = np.zeros(len(ctx['all_m']), dtype=np.float32)
    for j, m in enumerate(ctx['motifs']):
        sub_m_full[motif_idx_map[m]] = sub_m[j]
    rv = ctx['rv']
    jas = float(pearsonr(sub_m_full, rv)[0]) if sub_m_full.std() > 0 and rv.std() > 0 else float('nan')

    sub_k = ctx['km'][order].sum(axis=0)
    sub_kd = {kmer_keys[i]: float(sub_k[i]) for i in range(64)}
    union = sorted(set(ref_kmer_dict.keys()) | set(sub_kd.keys()))
    cc = np.zeros((len(union), 2))
    for i, kk in enumerate(union):
        if kk in ref_kmer_dict: cc[i, 1] = ref_kmer_dict[kk] * len(sub_kd) / n_highexp
        if kk in sub_kd: cc[i, 0] = sub_kd[kk]
    kc = float(pearsonr(cc[:, 0], cc[:, 1])[0]) if cc[:, 0].std() > 0 and cc[:, 1].std() > 0 else float('nan')

    return dict(pool=pool, n_pool=len(ctx['mcorr']), n_sub=N,
                pa=float(np.median(ctx['oracle'][order])), jas=jas, kmer3=kc,
                app_ll=float(np.median(ctx['app'][order])),
                atac=float((ctx['atac'][order] > 0.5).mean() * 100))


def main():
    pools = sorted([Path(d).name for d in glob.glob(f'{PHASEC}/run_*')])
    print(f'{len(pools)} recipes; selection: app-LL >= -259 then top-640 by motif+kmer+{ATAC_W}*ATAC ranks')
    rows = []
    for i, p in enumerate(pools):
        ctx = get_or_build_perseq(p)
        if ctx is None: continue
        rows.append(select_and_score(p, ctx, atac_w=ATAC_W))
        if (i+1) % 10 == 0: print(f'  {i+1}/{len(pools)}', flush=True)

    df = pd.DataFrame(rows)
    out_path = f'{PHASEC}/drakes_protocol/top640_unified_motif_kmer_atac.csv'
    df.to_csv(out_path, index=False)
    print(f'\n{len(df)} rows')
    print(f'Saved to {out_path}')

    df['bio'] = df['jas'] + df['kmer3'] + ATAC_W * (df['atac'] / 100.0)

    print(f'\n=== Top-15 (rank by JAS + 3mer + {ATAC_W}*ATAC/100) ===')
    print(f'{"pool":<46} {"JAS":>7} {"3mer":>7} {"ATAC":>6} {"bio":>6} {"PA":>5} {"AppLL":>7}')
    for _, r in df.nlargest(15, 'bio').iterrows():
        print(f'{r.pool:<46} {r.jas:>+7.3f} {r.kmer3:>+7.3f} {r.atac:>5.1f}% {r.bio:>+6.3f} {r.pa:>5.2f} {r.app_ll:>7.1f}')


if __name__ == '__main__':
    main()
