#!/usr/bin/env python3
"""Scan a pool's sequences against JASPAR2024 and cache the per-seq sparse motif matrix.
Standalone version of score_drakes_protocol's matrix-saving step (no model loading).

Usage:
  python scan_and_cache_motifs.py --input_h5 path/to/gpa_output_pool.h5
"""
import argparse, sys, os, time
from pathlib import Path
import numpy as np
import h5py
import pandas as pd
from scipy import sparse

DRAKES_REPO = '${GPA_EXTERNAL_ROOT}/DRAKES/drakes_dna'
DRAKES_BASE = '${GPA_EXTERNAL_ROOT}/DRAKES_data/data_and_model/'
REF_CACHE = '${GPA_REPO_ROOT}/results/rerd_comparison/drakes_protocol/_ref_cache'

sys.path.insert(0, DRAKES_REPO)
import dataloader_gosai
dataloader_gosai.base_path = DRAKES_BASE

from grelu.interpret.motifs import scan_sequences
from grelu.io.motifs import get_jaspar


def _scan_chunk(args):
    seqs_chunk, motifs, start_idx = args
    df = scan_sequences(seqs_chunk, motifs)
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df['sequence'] = df['sequence'].astype(int) + start_idx
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_h5', required=True)
    ap.add_argument('--n_jobs', type=int, default=None)
    args = ap.parse_args()

    n_jobs = args.n_jobs or max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    pool_name = Path(args.input_h5).parent.name
    cache_suffix = ''
    out_npz = os.path.join(REF_CACHE, f'pool_motif_perseq_{pool_name}{cache_suffix}.npz')
    out_motifs = out_npz.replace('.npz', '_motifs.npy')

    print(f'[scan_and_cache] pool={pool_name}, n_jobs={n_jobs}')
    print(f'  input: {args.input_h5}')
    print(f'  output: {out_npz}')

    t0 = time.time()
    with h5py.File(args.input_h5, 'r') as f:
        idx = f['indices'][:].astype(np.int64)
    seqs = dataloader_gosai.batch_dna_detokenize(idx)
    print(f'  N={len(seqs)} L={len(seqs[0])}')

    motifs = get_jaspar(release='JASPAR2024', collection='CORE', tax_group='vertebrates')
    print(f'  motifs: {len(motifs)}')

    # Parallel scan
    t1 = time.time()
    if n_jobs > 1 and len(seqs) >= 200:
        from multiprocessing import Pool
        chunk_size = (len(seqs) + n_jobs - 1) // n_jobs
        chunks = [(seqs[i:i + chunk_size], motifs, i) for i in range(0, len(seqs), chunk_size)]
        print(f'  parallel scan: {len(chunks)} chunks of ~{chunk_size} seqs')
        with Pool(n_jobs) as pool:
            dfs = pool.map(_scan_chunk, chunks)
        dfs = [d for d in dfs if d is not None]
        if not dfs:
            print('  no hits'); return
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = scan_sequences(seqs, motifs)
        if df is None or len(df) == 0:
            print('  no hits'); return
    print(f'  scan: {len(df)} hits in {time.time()-t1:.0f}s')

    # Build sparse matrix
    motif_names = sorted(df['motif'].unique().tolist())
    motif_to_col = {m: i for i, m in enumerate(motif_names)}
    seq_idx_arr = df['sequence'].astype(np.int32).values
    motif_idx_arr = df['motif'].map(motif_to_col).astype(np.int32).values
    data = np.ones(len(df), dtype=np.int16)
    csr = sparse.coo_matrix((data, (seq_idx_arr, motif_idx_arr)),
                            shape=(len(seqs), len(motif_names)),
                            dtype=np.int16).tocsr()
    csr.sum_duplicates()
    sparse.save_npz(out_npz, csr)
    np.save(out_motifs, np.array(motif_names, dtype=object), allow_pickle=True)
    print(f'  saved: {csr.shape} sparse → {out_npz} ({time.time()-t0:.0f}s total)')


if __name__ == '__main__':
    main()
