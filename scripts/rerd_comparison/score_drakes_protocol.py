#!/usr/bin/env python3
"""Score a GPA pool with the 5-metric DRAKES protocol used in
Pani-Ou-Li 2025 v4 Table 5 (https://arxiv.org/pdf/2505.22524v4).

Metrics:
  1. Pred-Activity (median)  -- DRAKES eval oracle, HepG2 column, median across pool
  2. ATAC-Acc (%)            -- binary_atac_cell_lines.ckpt, HepG2 col, fraction with prob > 0.5
  3. 3-mer Corr              -- pool kmer freq vs Gosai top-0.1% reference, Pearson r
  4. JASPAR Corr             -- pool motif counts vs Gosai top-0.1% reference, Pearson r
  5. App-Log-Lik (median)    -- MDLM pretrained, time-averaged denoising NLL summed over L,
                                negated, median across pool

Reuses DRAKES paper code via sys.path injection + base_path monkey-patch.

Usage:
  python score_drakes_protocol.py \
      --input_h5 results/rerd_comparison/run_paper_mean_random/gpa_output.h5 \
      --output_json results/rerd_comparison/drakes_protocol/run_paper_mean_random.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import h5py
import pandas as pd
import torch

# ── Inject DRAKES paths ────────────────────────────────────────────────
DRAKES_REPO = '${GPA_EXTERNAL_ROOT}/DRAKES/drakes_dna'
DRAKES_BASE = '${GPA_EXTERNAL_ROOT}/DRAKES_data/data_and_model/'
GPA_EVAL_CKPT = '${HOME}/SVDD/artifacts/DRAKES_oracles/reward_oracle_eval.ckpt'

sys.path.insert(0, DRAKES_REPO)

# Patch base_path BEFORE any DRAKES imports
import oracle as drakes_oracle  # noqa: E402
import dataloader_gosai  # noqa: E402
import diffusion_gosai_update  # noqa: E402

drakes_oracle.base_path = DRAKES_BASE
dataloader_gosai.base_path = DRAKES_BASE

from hydra import initialize, compose  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from grelu.lightning import LightningModel  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from grelu.interpret.motifs import scan_sequences
from grelu.io.motifs import get_jaspar


def _scan_chunk(args):
    """Worker for parallel scanning. Returns df with sequence indices offset to global."""
    seqs_chunk, motifs, start_idx = args
    df = scan_sequences(seqs_chunk, motifs)
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df['sequence'] = df['sequence'].astype(int) + start_idx
    return df


def _scan_count_vector(seqs, motifs, n_jobs=None):
    """Run scan_sequences (parallelized) and return a per-motif count vector
    (Series indexed by motif name)."""
    if n_jobs is None:
        import os as _os
        n_jobs = max(1, int(_os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    if n_jobs <= 1 or len(seqs) < 200:
        df = scan_sequences(seqs, motifs)
        if df is None or len(df) == 0:
            return pd.Series(dtype=float)
        return df.groupby('motif').size().astype(float)
    # parallel
    from multiprocessing import Pool
    chunk_size = (len(seqs) + n_jobs - 1) // n_jobs
    chunks = [(seqs[i:i + chunk_size], motifs, i) for i in range(0, len(seqs), chunk_size)]
    with Pool(n_jobs) as pool:
        dfs = pool.map(_scan_chunk, chunks)
    dfs = [d for d in dfs if d is not None]
    if not dfs:
        return pd.Series(dtype=float)
    df = pd.concat(dfs, ignore_index=True)
    return df.groupby('motif').size().astype(float)


def _scan_count_matrix(seqs, motifs, n_jobs=None):
    """Run scan_sequences (parallelized) and return BOTH:
    - per-motif aggregate Series (used for pool-level Pearson)
    - per-seq sparse CSR matrix (rows=seq_idx, cols=motif), int16 counts
    - motif column names list (matching CSR cols)
    """
    if n_jobs is None:
        import os as _os
        n_jobs = max(1, int(_os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    from multiprocessing import Pool
    if n_jobs > 1 and len(seqs) >= 200:
        chunk_size = (len(seqs) + n_jobs - 1) // n_jobs
        chunks = [(seqs[i:i + chunk_size], motifs, i) for i in range(0, len(seqs), chunk_size)]
        with Pool(n_jobs) as pool:
            dfs = pool.map(_scan_chunk, chunks)
        dfs = [d for d in dfs if d is not None]
        if not dfs:
            return pd.Series(dtype=float), None, []
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = scan_sequences(seqs, motifs)
        if df is None or len(df) == 0:
            return pd.Series(dtype=float), None, []
    # Per-motif aggregate
    per_motif = df.groupby('motif').size().astype(float)
    # Per-seq sparse matrix
    from scipy import sparse as _sparse
    motif_names = sorted(df['motif'].unique().tolist())
    motif_to_col = {m: i for i, m in enumerate(motif_names)}
    seq_idx_arr = df['sequence'].astype(np.int32).values
    motif_idx_arr = df['motif'].map(motif_to_col).astype(np.int32).values
    data = np.ones(len(df), dtype=np.int16)
    csr = _sparse.coo_matrix((data, (seq_idx_arr, motif_idx_arr)),
                             shape=(len(seqs), len(motif_names)),
                             dtype=np.int16).tocsr()
    csr.sum_duplicates()  # combine multiple hits into single count
    return per_motif, csr, motif_names

# Cell-type indices (per DRAKES eval.ipynb)
HEPG2_REWARD_IDX = 0   # cal_gosai_pred returns shape (n, 3) for [hepg2, k562, sknsh]
HEPG2_ATAC_IDX = 1     # cal_atac_pred returns shape (n, 7); idx 1 = HepG2

# ── Helpers ───────────────────────────────────────────────────────────
def detokenize(indices):
    """indices: (N, L) int [0..3]  →  list of N ACGT strings."""
    return dataloader_gosai.batch_dna_detokenize(indices.astype(np.int64))


def load_gpa_pool(h5_path):
    with h5py.File(h5_path, 'r') as f:
        idx = f['indices'][:]  # (N, L) int
    seqs = detokenize(idx)
    return idx, seqs


def load_eval_oracle():
    print(f'  Loading DRAKES eval oracle: {GPA_EVAL_CKPT}')
    m = LightningModel.load_from_checkpoint(GPA_EVAL_CKPT, map_location='cuda')
    m.train_params['logger'] = None
    return m


def load_atac_oracle():
    """ATAC ckpt has data_params nested under hyper_parameters; gReLU expects
    it at top level. Inject a top-level 'data_params' before loading."""
    p = os.path.join(DRAKES_BASE, 'mdlm/gosai_data/binary_atac_cell_lines.ckpt')
    print(f'  Loading ATAC oracle: {p}')
    p_patched = p.replace('.ckpt', '_patched.ckpt')
    # IMPORTANT: only write once. With many parallel SLURM jobs, concurrent
    # torch.save's to the same file produce a corrupted .ckpt. Atomic check.
    if not Path(p_patched).exists():
        raw = torch.load(p, map_location='cpu', weights_only=False)
        hp = raw.get('hyper_parameters', {})
        for key, default in [('data_params', {}), ('performance', {}),
                             ('train_params', {}), ('model_params', {})]:
            if key not in raw:
                raw[key] = hp.get(key, default)
        # Atomic write: write to a temp file then rename
        tmp_path = p_patched + '.tmp.' + str(os.getpid())
        torch.save(raw, tmp_path)
        try:
            os.rename(tmp_path, p_patched)  # atomic on same FS
        except OSError:
            # someone else won the race; remove our temp
            try: os.remove(tmp_path)
            except OSError: pass
    m = LightningModel.load_from_checkpoint(p_patched, map_location='cuda')
    m.train_params['logger'] = None
    return m


def load_pretrained_mdlm():
    """Load DRAKES pretrained MDLM via Hydra config.

    `initialize(config_path=...)` resolves relative to the *caller file* location
    (Hydra peculiarity), which fails when the script runs from a different dir.
    Use `initialize_config_dir` with an ABSOLUTE path instead.
    """
    from hydra import initialize_config_dir  # local import to avoid clutter
    GlobalHydra.instance().clear()
    config_dir = os.path.join(DRAKES_REPO, 'configs_gosai')
    initialize_config_dir(config_dir=config_dir, job_name='gpa_drakes_score', version_base=None)
    cfg = compose(config_name='config_gosai.yaml')
    old_path = os.path.join(DRAKES_BASE, 'mdlm/outputs_gosai/pretrained.ckpt')
    print(f'  Loading pretrained MDLM: {old_path}')
    model = diffusion_gosai_update.Diffusion.load_from_checkpoint(old_path, config=cfg)
    model.eval()
    return model


# ── Reference cache (top-0.1% Gosai by HepG2) ─────────────────────────
def build_highexp_reference(k=3):
    """Replicates DRAKES cal_highexp_kmers WITHOUT calling cal_gosai_pred_new
    (which OOMs on 5000+ seq batches). We only need seqs and kmer counts.

    Returns (highexp_seqs_999, highexp_kmers_999, n_highexp_999).
    """
    train_set = dataloader_gosai.get_datasets_gosai()
    exp_threshold_999 = np.quantile(train_set.clss[:, 0].numpy(), 0.999)
    highexp_indices = np.where(train_set.clss[:, 0].numpy() > exp_threshold_999)[0]
    highexp_seqs_arr = train_set.seqs[highexp_indices].numpy()
    highexp_seqs = dataloader_gosai.batch_dna_detokenize(highexp_seqs_arr)
    highexp_kmers = drakes_oracle.count_kmers(highexp_seqs, k=k)
    return highexp_seqs, highexp_kmers, int(len(highexp_indices))


# ── Batched gReLU scoring (avoids OOM) ────────────────────────────────
def _grelu_batch_score(seqs, model, batch_size=64):
    """Run gReLU model.forward over seqs in chunks. Returns (N, n_outputs)."""
    import torch.nn.functional as F
    model.eval()
    device = next(model.parameters()).device
    tokens = dataloader_gosai.batch_dna_tokenize(seqs)
    tokens_t = torch.tensor(tokens).long()
    out_chunks = []
    with torch.no_grad():
        for s in range(0, len(seqs), batch_size):
            e = min(s + batch_size, len(seqs))
            chunk = tokens_t[s:e].to(device)
            onehot = F.one_hot(chunk, num_classes=4).float().transpose(1, 2)
            pred = model(onehot).detach().cpu().numpy()
            if pred.ndim == 3 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            out_chunks.append(pred)
    return np.concatenate(out_chunks, axis=0)


# ── Metric 1: Pred-Activity median ────────────────────────────────────
def metric_pred_activity_median(seqs, eval_oracle):
    preds = _grelu_batch_score(seqs, eval_oracle, batch_size=64)  # (N, 3)
    return float(np.median(preds[:, HEPG2_REWARD_IDX]))


# ── Metric 2: ATAC-Acc % ──────────────────────────────────────────────
def metric_atac_acc(seqs, atac_model, threshold=0.5):
    preds = _grelu_batch_score(seqs, atac_model, batch_size=64)  # (N, 7)
    return float((preds[:, HEPG2_ATAC_IDX] > threshold).mean() * 100.0)


# ── Metric 3: 3-mer Pearson corr ──────────────────────────────────────
def metric_kmer_corr(seqs, highexp_kmers, n_highexp_kmers, k=3):
    gen_kmers = drakes_oracle.count_kmers(seqs, k=k)
    kmer_set = set(highexp_kmers.keys()) | set(gen_kmers.keys())
    counts = np.zeros((len(kmer_set), 2))
    for i, km in enumerate(kmer_set):
        if km in highexp_kmers:
            counts[i, 1] = highexp_kmers[km] * len(gen_kmers) / n_highexp_kmers
        if km in gen_kmers:
            counts[i, 0] = gen_kmers[km]
    r = pearsonr(counts[:, 0], counts[:, 1])[0]
    return float(r)


# ── Metric 4: JASPAR pool-level Pearson corr ──────────────────────────
def metric_jaspar_corr(seqs, ref_motif_counts, motifs):
    """
    seqs: list[str]; pool of generated sequences
    ref_motif_counts: pd.Series (index=motif, value=count) for high-exp Gosai ref
    motifs: dict[motif_name -> PPM] from get_jaspar()

    DRAKES protocol: scan the pool with FIMO/JASPAR, sum motif hits across
    all sequences → 1D motif-count vector; Pearson against ref count vector.
    Returns single scalar (pool-level Pearson r).
    """
    pool_counts = _scan_count_vector(seqs, motifs)
    if len(pool_counts) == 0:
        return float('nan')

    # Union of motifs; pad zeros for absent ones
    all_motifs = sorted(set(pool_counts.index) | set(ref_motif_counts.index))
    pool_vec = np.array([pool_counts.get(m, 0.0) for m in all_motifs], dtype=float)
    ref_vec = np.array([ref_motif_counts.get(m, 0.0) for m in all_motifs], dtype=float)

    if len(pool_vec) < 2 or pool_vec.std() == 0 or ref_vec.std() == 0:
        return float('nan')
    return float(pearsonr(pool_vec, ref_vec)[0])


# ── Metric 5: App-Log-Lik median ──────────────────────────────────────
def metric_app_log_lik_per_seq(seqs, mdlm_model, n_time_samples=10, batch_size=64):
    """Returns per-seq App-Log-Lik values (negated NLL averaged over n_time_samples).
    Higher = better."""
    mdlm_model.eval()
    device = next(mdlm_model.parameters()).device
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
                loss_per_token = mdlm_model._forward_pass_diffusion(batch_tokens)
                losses.append(loss_per_token.sum(-1).cpu().numpy())
            per_seq_nll[s:e] = np.stack(losses, axis=0).mean(axis=0)
    return -per_seq_nll  # higher better


def metric_app_log_lik_median(seqs, mdlm_model, n_time_samples=10, batch_size=64):
    """For each sequence, average the MDLM denoising NLL over n_time_samples
    random t draws; return median (negated → higher better).
    """
    mdlm_model.eval()
    device = next(mdlm_model.parameters()).device
    tokens = dataloader_gosai.batch_dna_tokenize(seqs)  # (N, L)
    tokens = torch.tensor(tokens).long()
    N = tokens.shape[0]
    per_seq_nll = np.zeros(N, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            batch_tokens = tokens[s:e].to(device)
            losses = []
            for _ in range(n_time_samples):
                # _forward_pass_diffusion samples t internally, returns per-token loss (B, L)
                loss_per_token = mdlm_model._forward_pass_diffusion(batch_tokens)
                losses.append(loss_per_token.sum(-1).cpu().numpy())
            per_seq_nll[s:e] = np.stack(losses, axis=0).mean(axis=0)
    # DRAKES reports likelihood as `-NLL`; we report median per-seq value
    # consistent with Pani v4 Table 5 sign convention (more negative = worse,
    # values around -260).
    return float(np.median(-per_seq_nll))


# ── Main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_h5', required=True, help='GPA pool h5 path')
    ap.add_argument('--output_json', required=True)
    ap.add_argument('--n_time_samples', type=int, default=10,
                    help='How many t draws per seq for App-Log-Lik')
    ap.add_argument('--ref_cache_dir', default='${GPA_REPO_ROOT}/results/rerd_comparison/drakes_protocol/_ref_cache')
    ap.add_argument('--skip_jaspar', action='store_true',
                    help='Skip JASPAR Pearson (slow; ~15-20min/pool)')
    ap.add_argument('--skip_app_log_lik', action='store_true',
                    help='Skip App-Log-Lik (~10-15min/pool)')
    args = ap.parse_args()

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ref_cache_dir).mkdir(parents=True, exist_ok=True)
    pool_name = Path(args.input_h5).parent.name
    cache_suffix = ''

    print(f'[score_drakes_protocol] pool={pool_name}')
    t0 = time.time()
    indices, seqs = load_gpa_pool(args.input_h5)
    print(f'  Loaded {len(seqs)} sequences (length {len(seqs[0])}). t={time.time()-t0:.1f}s')

    # ── Reference: top-0.1% Gosai by HepG2 (cached) ──────────────────
    ref_cache = Path(args.ref_cache_dir) / 'highexp_999.npz'
    motif_ref_cache = Path(args.ref_cache_dir) / 'motif_count_top999.parquet'
    if ref_cache.exists():
        print(f'  Using cached reference: {ref_cache}')
        cached = np.load(ref_cache, allow_pickle=True)
        highexp_kmers = cached['kmers'].item()
        n_highexp = int(cached['n'])
        highexp_seqs_999 = cached['seqs'].tolist()
    else:
        print('  Building Gosai top-0.1% reference (~20s)...')
        highexp_seqs_999, highexp_kmers, n_highexp = build_highexp_reference(k=3)
        np.savez(ref_cache,
                 kmers=np.array([highexp_kmers], dtype=object)[0],
                 n=n_highexp,
                 seqs=np.array(highexp_seqs_999, dtype=object))
        print(f'  Cached → {ref_cache}  (n_highexp_999={n_highexp})')

    # JASPAR reference (cache)
    motif_ref_csv = Path(args.ref_cache_dir) / 'motif_count_top999.csv'
    motifs = None
    ref_motif_counts = None
    if not args.skip_jaspar:
        print('  Fetching JASPAR2024 vertebrate motifs...')
        motifs = get_jaspar(release='JASPAR2024', collection='CORE', tax_group='vertebrates')
        print(f'    got {len(motifs)} motifs')
        if motif_ref_csv.exists():
            print(f'  Using cached JASPAR ref: {motif_ref_csv}')
            ref_motif_counts = pd.read_csv(motif_ref_csv, index_col=0).iloc[:, 0]
        else:
            print(f'  Computing JASPAR reference motif counts (~3min, n_ref={len(highexp_seqs_999)})...')
            ref_motif_counts = _scan_count_vector(highexp_seqs_999, motifs)
            ref_motif_counts.to_csv(motif_ref_csv)
            print(f'  Cached → {motif_ref_csv}  ({len(ref_motif_counts)} motifs)')

    out = {
        'pool_name': pool_name,
        'input_h5': args.input_h5,
        'n_seqs': len(seqs),
        'target_cell': 'hepg2',
    }

    # 1. Pred-Activity median
    print('  [1/5] Pred-Activity median...')
    eval_oracle = load_eval_oracle()
    t1 = time.time()
    out['pred_activity_median'] = metric_pred_activity_median(seqs, eval_oracle)
    print(f'    pred_activity_median = {out["pred_activity_median"]:.3f}  ({time.time()-t1:.1f}s)')
    del eval_oracle
    torch.cuda.empty_cache()

    # 2. ATAC-Acc %
    print('  [2/5] ATAC-Acc %...')
    atac_model = load_atac_oracle()
    t1 = time.time()
    out['atac_acc_pct'] = metric_atac_acc(seqs, atac_model)
    print(f'    atac_acc_pct = {out["atac_acc_pct"]:.2f}%  ({time.time()-t1:.1f}s)')
    del atac_model
    torch.cuda.empty_cache()

    # 3. 3-mer Pearson
    print('  [3/5] 3-mer Pearson corr...')
    t1 = time.time()
    out['kmer3_pearson'] = metric_kmer_corr(seqs, highexp_kmers, n_highexp, k=3)
    print(f'    kmer3_pearson = {out["kmer3_pearson"]:.4f}  ({time.time()-t1:.1f}s)')

    # 4. JASPAR Pearson (pool-level) + cache per-seq sparse matrix for future subset selection
    if not args.skip_jaspar:
        print('  [4/5] JASPAR pool-level Pearson + per-seq matrix...')
        t1 = time.time()
        per_motif, per_seq_csr, motif_names = _scan_count_matrix(seqs, motifs)
        # Pool-level Pearson (same as before)
        if len(per_motif) == 0:
            out['jaspar_pearson'] = float('nan')
        else:
            all_motifs = sorted(set(per_motif.index) | set(ref_motif_counts.index))
            pool_vec = np.array([per_motif.get(m, 0.0) for m in all_motifs], dtype=float)
            ref_vec = np.array([ref_motif_counts.get(m, 0.0) for m in all_motifs], dtype=float)
            if pool_vec.std() == 0 or ref_vec.std() == 0:
                out['jaspar_pearson'] = float('nan')
            else:
                out['jaspar_pearson'] = float(pearsonr(pool_vec, ref_vec)[0])
        # Save per-seq sparse matrix for cheap future subset selection
        if per_seq_csr is not None:
            from scipy import sparse as _sparse
            perseq_path = Path(args.ref_cache_dir) / f'pool_motif_perseq_{pool_name}{cache_suffix}.npz'
            _sparse.save_npz(str(perseq_path), per_seq_csr)
            # Save motif column names alongside
            np.save(str(perseq_path).replace('.npz', '_motifs.npy'),
                    np.array(motif_names, dtype=object), allow_pickle=True)
            print(f'    cached per-seq matrix: {per_seq_csr.shape} sparse → {perseq_path.name}')
        print(f'    jaspar_pearson = {out["jaspar_pearson"]:.4f}  ({time.time()-t1:.1f}s)')
    else:
        out['jaspar_pearson'] = None

    # 5. App-Log-Lik median + cache per-seq App-LL for future subset selection
    if not args.skip_app_log_lik:
        print(f'  [5/5] App-Log-Lik per-seq + median (n_time_samples={args.n_time_samples})...')
        mdlm = load_pretrained_mdlm().cuda()
        t1 = time.time()
        per_seq_app_ll = metric_app_log_lik_per_seq(
            seqs, mdlm, n_time_samples=args.n_time_samples)
        out['app_log_lik_median'] = float(np.median(per_seq_app_ll))
        # Save per-seq App-LL alongside per-seq motif matrix
        app_ll_path = Path(args.ref_cache_dir) / f'pool_app_ll_perseq_{pool_name}{cache_suffix}.npy'
        np.save(str(app_ll_path), per_seq_app_ll.astype(np.float32))
        print(f'    app_log_lik_median = {out["app_log_lik_median"]:.2f}  ({time.time()-t1:.1f}s)')
        print(f'    cached per-seq App-LL: {per_seq_app_ll.shape} → {app_ll_path.name}')
    else:
        out['app_log_lik_median'] = None

    out['total_seconds'] = time.time() - t0
    with open(args.output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[done] {out["pool_name"]}  → {args.output_json}  total={out["total_seconds"]:.0f}s')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
