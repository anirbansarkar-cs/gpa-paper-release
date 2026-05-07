#!/usr/bin/env python3
"""Unified app-filtered top-640 ranking on sampled pools.
All ATAC + motif + app-LL caches must already exist."""
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

ref_motif = pd.read_csv(f'{REF_CACHE}/motif_count_top999.csv', index_col=0).iloc[:, 0]
ref_kmer = np.load(f'{REF_CACHE}/highexp_999.npz', allow_pickle=True)
ref_kmer_dict = ref_kmer['kmers'].item()
n_highexp = int(ref_kmer['n'])
kmer_keys = [''.join(p) for p in product('ACGT', repeat=3)]
ref_kvec = np.array([ref_kmer_dict.get(k, 0.0) for k in kmer_keys], dtype=np.float32)
HEPG2_IDX = 1


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


def analyze(pool):
    suffix, h5n = '', 'gpa_output_pool.h5'
    npz = f'{REF_CACHE}/pool_motif_perseq_{pool}{suffix}.npz'
    h5p = f'{PHASEC}/{pool}/{h5n}'
    app = f'{REF_CACHE}/pool_app_ll_perseq_{pool}{suffix}.npy'
    atc = f'{REF_CACHE}/pool_atac_perseq_{pool}{suffix}.npy'
    if not all(os.path.exists(p) for p in (npz, h5p, app, atc)):
        return None

    csr = sparse.load_npz(npz).tocsr()
    motifs = np.load(npz.replace('.npz', '_motifs.npy'), allow_pickle=True).tolist()
    name_to_col = {m: i for i, m in enumerate(motifs)}
    # Build ref vec ONLY over the union of motif names (avoids huge dense)
    all_m = sorted(set(motifs) | set(ref_motif.index))
    rv = np.array([ref_motif.get(m, 0.0) for m in all_m], dtype=np.float32)

    # Per-seq motif Pearson against ref_motif aggregate via sparse matmul.
    # Equivalent dense: aligned[i,j] = csr[i, name_to_col[m]] if m in motifs else 0
    # Trick: construct ref-vector aligned to motifs only (cols of csr), pad zeros for missing.
    rv_motifs = np.array([ref_motif.get(m, 0.0) for m in motifs], dtype=np.float32)
    # Pearson numerator: (csr - csr.mean(1)) @ (rv_motifs - rv_motifs.mean())
    # Use full union mean for ref (since missing motifs in csr are zero):
    csr_arr = csr.astype(np.float32)
    n = csr_arr.shape[0]
    # Per-seq mean across ALL all_m columns: we need (sum of csr row + sum of zeros)/len(all_m) = csr_row.sum()/len(all_m)
    row_sum = np.asarray(csr_arr.sum(axis=1)).ravel()
    pool_mean = row_sum / len(all_m)
    # Pool centered dot ref_full_centered:
    # = sum_j (a_ij - pool_mean[i])*(rv[j] - rv.mean())
    # = sum_j a_ij * rv_full[j] - pool_mean[i]*sum_j rv_full[j] - rv.mean()*sum_j a_ij + pool_mean[i]*rv.mean()*len(all_m)
    # where rv_full has rv_motifs in csr cols and ref_motif.get(m,0) in non-csr cols.
    # Simplify: only csr cols contribute to a_ij*rv_full[j], so = csr @ rv_motifs.
    # And sum_j rv_full[j] = rv.sum().
    # pool_mean[i]*rv.mean()*len(all_m) = row_sum[i]*rv.mean() (the third term cancels with this when expanded)
    # Final: (csr @ rv_motifs) - rv.mean()*row_sum
    # (Verified: the mean-correction terms reduce because pool_mean*len(all_m)=row_sum.)
    rv_mean = rv.mean()
    rv_sum = rv.sum()
    cov = csr_arr @ rv_motifs - rv_mean * row_sum
    # Pool norm (centered): ||a - pool_mean||^2 = sum_j a_ij^2 - 2*pool_mean*row_sum + pool_mean^2 * len(all_m)
    # = ||csr_row||^2 + 0 (zeros contribute zero squared) - 2*pool_mean*row_sum + pool_mean^2*len(all_m)
    # = sq_sum - 2*row_sum^2/len + row_sum^2/len = sq_sum - row_sum^2/len(all_m)
    sq_sum = np.asarray(csr_arr.multiply(csr_arr).sum(axis=1)).ravel()
    pool_norm_sq = sq_sum - row_sum*row_sum/len(all_m)
    pool_norm = np.sqrt(np.clip(pool_norm_sq, 0, None)) + 1e-12
    rv_norm = np.linalg.norm(rv - rv_mean) + 1e-12
    mcorr = cov / (pool_norm * rv_norm)

    with h5py.File(h5p, 'r') as f:
        idx = f['indices'][:].astype(np.int64)
        for k in ('oracle_hepg2_eval', 'oracle_hepg2', 'oracle_preds'):
            if k in f:
                oracle = f[k][:]; break
    seqs = dataloader_gosai.batch_dna_detokenize(idx)
    km = per_seq_kmer(seqs)
    rkc = ref_kvec - ref_kvec.mean(); rkn = np.linalg.norm(rkc) + 1e-12
    pkc = km - km.mean(axis=1, keepdims=True); pkn = np.linalg.norm(pkc, axis=1) + 1e-12
    kcorr = (pkc @ rkc) / (pkn * rkn)

    pa_app = np.load(app); pa_atac = np.load(atc)[:, HEPG2_IDX]

    rmot = rankdata(mcorr); rkm = rankdata(kcorr)
    N = min(640, len(seqs))
    APP_MIN = -259.0
    valid = np.where(pa_app >= APP_MIN)[0]
    if len(valid) >= N:
        order = valid[np.argsort(-(rmot[valid] + rkm[valid]))[:N]]
    elif len(valid) > 0:
        invalid = np.where(pa_app < APP_MIN)[0]
        fill = N - len(valid)
        top_inv = invalid[np.argsort(-(rmot[invalid] + rkm[invalid]))[:fill]]
        order = np.concatenate([valid, top_inv])
    else:
        order = np.argsort(-(rmot + rkm))[:N]

    sub_m_in_motifs = csr_arr[order].sum(axis=0)
    sub_m_in_motifs = np.asarray(sub_m_in_motifs).ravel()
    # Build full sub_m vector aligned with rv (all_m):
    sub_m_full = np.zeros(len(all_m), dtype=np.float32)
    for j, m in enumerate(motifs):
        sub_m_full[all_m.index(m)] = sub_m_in_motifs[j]
    if sub_m_full.std() > 0 and rv.std() > 0:
        jas = float(pearsonr(sub_m_full, rv)[0])
    else:
        jas = float('nan')

    sub_k = km[order].sum(axis=0)
    sub_kd = {kmer_keys[i]: float(sub_k[i]) for i in range(64)}
    union = sorted(set(ref_kmer_dict.keys()) | set(sub_kd.keys()))
    cc = np.zeros((len(union), 2))
    for i, kk in enumerate(union):
        if kk in ref_kmer_dict: cc[i, 1] = ref_kmer_dict[kk] * len(sub_kd) / n_highexp
        if kk in sub_kd: cc[i, 0] = sub_kd[kk]
    kc = float(pearsonr(cc[:, 0], cc[:, 1])[0]) if cc[:, 0].std() > 0 and cc[:, 1].std() > 0 else float('nan')

    return dict(pool=pool, n_pool=len(seqs), n_sub=N,
                pa=float(np.median(oracle[order])), jas=jas, kmer3=kc,
                app_ll=float(np.median(pa_app[order])),
                atac=float((pa_atac[order] > 0.5).mean() * 100))


def main():
    pools = sorted([Path(d).name for d in glob.glob(f'{PHASEC}/run_*')])
    print(f'{len(pools)} recipes; selection: app-LL >= -259 then top-640 by motif+kmer composite ranks')
    rows = []
    for i, p in enumerate(pools):
        r = analyze(p)
        if r is not None: rows.append(r)
        if (i+1) % 10 == 0: print(f'  {i+1}/{len(pools)}', flush=True)

    df = pd.DataFrame(rows)
    out_path = f'{PHASEC}/drakes_protocol/top640_unified_app_filtered.csv'
    df.to_csv(out_path, index=False)
    print(f'\n{len(df)} rows')
    print(f'Saved to {out_path}')

    df['score'] = df['pa'] + 0.5*df['jas'] + 0.005*df['atac'] - 0.01*np.maximum(0, -260 - df['app_ll'])

    print('\n=== Top-15 by composite (PA + 0.5*JAS + 0.005*ATAC, app-penalty) ===')
    print(f'{"pool":<46} {"PA":>5} {"JAS":>6} {"3mer":>6} {"AppLL":>7} {"ATAC":>6} {"score":>6}')
    for _, r in df.nlargest(15, 'score').iterrows():
        print(f'{r.pool:<46} {r.pa:>5.2f} {r.jas:>+6.3f} {r.kmer3:>+6.3f} {r.app_ll:>7.1f} {r.atac:>5.1f}% {r.score:>5.2f}')


if __name__ == '__main__':
    main()
