#!/usr/bin/env python3
"""Compute and cache per-seq App-Log-Lik for a single pool.
Standalone — runs MDLM forward on the pool's h5.

Usage:
  python cache_app_ll_perseq.py --input_h5 path/to/gpa_output_pool.h5
"""
import argparse, sys, os, time
from pathlib import Path
import numpy as np
import h5py
import torch

DRAKES_REPO = '${GPA_EXTERNAL_ROOT}/DRAKES/drakes_dna'
DRAKES_BASE = '${GPA_EXTERNAL_ROOT}/DRAKES_data/data_and_model/'
REF_CACHE = '${GPA_REPO_ROOT}/results/rerd_comparison/drakes_protocol/_ref_cache'

sys.path.insert(0, DRAKES_REPO)
import dataloader_gosai
import diffusion_gosai_update
dataloader_gosai.base_path = DRAKES_BASE
import oracle as drakes_oracle
drakes_oracle.base_path = DRAKES_BASE


def load_pretrained_mdlm():
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    config_dir = os.path.join(DRAKES_REPO, 'configs_gosai')
    initialize_config_dir(config_dir=config_dir, job_name='cache_app_ll', version_base=None)
    cfg = compose(config_name='config_gosai.yaml')
    ckpt = os.path.join(DRAKES_BASE, 'mdlm/outputs_gosai/pretrained.ckpt')
    model = diffusion_gosai_update.Diffusion.load_from_checkpoint(ckpt, config=cfg)
    model.eval()
    return model


def compute_per_seq_app_ll(seqs, mdlm, n_time_samples=10, batch_size=64):
    device = next(mdlm.parameters()).device
    tokens = dataloader_gosai.batch_dna_tokenize(seqs)
    tokens = torch.tensor(tokens).long()
    N = tokens.shape[0]
    per_seq_nll = np.zeros(N, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            batch_tokens = tokens[s:e].to(device)
            losses = []
            for _ in range(n_time_samples):
                loss = mdlm._forward_pass_diffusion(batch_tokens)
                losses.append(loss.sum(-1).cpu().numpy())
            per_seq_nll[s:e] = np.stack(losses, axis=0).mean(axis=0)
    return -per_seq_nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_h5', required=True)
    ap.add_argument('--n_time_samples', type=int, default=10)
    args = ap.parse_args()

    pool_name = Path(args.input_h5).parent.name
    cache_suffix = ''
    out_path = os.path.join(REF_CACHE, f'pool_app_ll_perseq_{pool_name}{cache_suffix}.npy')
    if os.path.exists(out_path):
        existing = np.load(out_path)
        print(f'  exists: {out_path} (shape={existing.shape})')
        return

    t0 = time.time()
    print(f'[cache_app_ll] pool={pool_name}')
    with h5py.File(args.input_h5, 'r') as f:
        idx = f['indices'][:].astype(np.int64)
    seqs = dataloader_gosai.batch_dna_detokenize(idx)
    print(f'  N={len(seqs)} L={len(seqs[0])}')

    mdlm = load_pretrained_mdlm().cuda()
    print(f'  loaded MDLM ({time.time()-t0:.0f}s)')

    t1 = time.time()
    per_seq_app_ll = compute_per_seq_app_ll(seqs, mdlm, n_time_samples=args.n_time_samples)
    print(f'  computed App-LL per-seq, median={np.median(per_seq_app_ll):.1f} ({time.time()-t1:.0f}s)')

    np.save(out_path, per_seq_app_ll.astype(np.float32))
    print(f'  saved → {out_path} ({time.time()-t0:.0f}s total)')


if __name__ == '__main__':
    main()
