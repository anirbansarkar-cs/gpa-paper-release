"""LEDIDI baseline: single-sequence gradient editor on K562 LentiMPRA.

Runs LEDIDI (Gumbel-softmax gradient editing) on the same seed sequences
used for the ISM comparison, targeting LegNet K562 oracle maximization.

Core hypothesis: single-sequence oracle exploiters (ISM, LEDIDI) achieve
high LegNet scores but fail AlphaGenome eval due to oracle exploitation,
while GPA's population-level SMC transfers.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.expandvars('${GPA_REPO_ROOT}'))
from scripts.k562_mdlm_gpa.k562_oracle import load_k562_oracle, K562Oracle

import h5py
from ledidi import Ledidi


# =============================================================================
# Config
# =============================================================================
ORACLE_CKPT = os.path.expandvars('${GPA_REPO_ROOT}/model_zoo/lentimpra/oracle_models/best_model-epoch=24-val_pearson=0.814.ckpt')
ISM_CSV = os.path.expandvars('${GPA_REPO_ROOT}/results/ism_high_oracle_scored.csv')
GPA_SEED_H5 = os.path.expandvars('${GPA_REPO_ROOT}/results/k562_mdlm_gpa/single_seed_299823.h5')
OUTPUT_DIR = os.path.expandvars('${GPA_REPO_ROOT}/results/ledidi_comparison')

BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
SEQ_LEN = 200
N_CATEGORIES = 4

# LEDIDI defaults (can override via CLI)
DEFAULT_TARGET = 15.0       # High target to maximize oracle
DEFAULT_L = 0.1             # L1 regularization weight
DEFAULT_TAU = 1.0           # Gumbel-softmax temperature
DEFAULT_LR = 1.0            # Learning rate
DEFAULT_MAX_ITER = 1000     # Max optimization steps
DEFAULT_BATCH_SIZE = 64     # Gumbel samples per iteration
DEFAULT_EARLY_STOP = 100    # Early stopping patience


# =============================================================================
# LegNet wrapper for LEDIDI (must be nn.Module)
# =============================================================================
class LedidiOracleWrapper(torch.nn.Module):
    """Wraps K562Oracle.dps_forward for LEDIDI.

    Routes LEDIDI's (B, 4, 200) soft-onehot input through the EXACT same
    differentiable scoring path GPA uses (K562Oracle.dps_forward). Adapters
    are injected as detached constants inside dps_forward, so gradients flow
    only through the 200 bp insert. No channel swap needed — GPA's path uses
    bio encoding throughout (per the comment in K562Oracle.dps_forward:
    "No channel swap needed — diffusion and LegNet use the same bio encoding").

    Bug found 2026-05-02: the previous wrapper called the raw nn.Module
    without adapter injection AND with a C<->G channel swap. LEDIDI exploited
    that misaligned objective, producing sequences with runtime LN ~9 that
    rescore to ~-0.4 via K562Oracle.score. Fix: just use K562Oracle.dps_forward.
    """

    def __init__(self, k562_oracle):
        super().__init__()
        self.k562_oracle = k562_oracle  # K562Oracle instance

    def forward(self, x):
        """x: (B, 4, L) bio encoding (A=0,C=1,G=2,T=3). L=200 (insert) or 230.
        Returns: (B, 1) per-sequence LegNet prediction.
        """
        preds = self.k562_oracle.dps_forward(x)            # (B,)
        return preds.view(-1, 1)


# =============================================================================
# Helpers
# =============================================================================
def seq_to_onehot(seq_str):
    """Convert sequence string to (1, 4, L) one-hot tensor (diffusion encoding)."""
    indices = [BASES[c] for c in seq_str]
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    onehot = torch.nn.functional.one_hot(idx_tensor, N_CATEGORIES).float()
    return onehot.T.unsqueeze(0)  # (1, 4, L)


def onehot_to_seq(onehot):
    """Convert (4, L) one-hot tensor to sequence string."""
    indices = onehot.argmax(dim=0).cpu().numpy()
    return ''.join(IDX_TO_BASE[i] for i in indices)


def indices_to_seq(indices):
    """Convert (L,) int array to sequence string."""
    return ''.join(IDX_TO_BASE[int(i)] for i in indices)


def generate_random_seeds(n, length=SEQ_LEN, rng_seed=42):
    """Generate n uniform random DNA sequences as a DataFrame."""
    rng = np.random.RandomState(rng_seed)
    rows = []
    for i in range(n):
        indices = rng.randint(0, 4, size=length)
        seq = indices_to_seq(indices)
        rows.append({'sequence': seq, 'seed_type': 'random_init'})
    return pd.DataFrame(rows)


def load_gpa_seed(h5_path, n_copies=1):
    """Load GPA seed from H5 and return as DataFrame.

    Args:
        h5_path: Path to single-seed H5 (with 'indices' dataset).
        n_copies: Number of copies (LEDIDI runs independently per copy,
                  so multiple copies test variance from Gumbel sampling).
    """
    with h5py.File(h5_path, 'r') as f:
        indices = f['indices'][:].astype(np.int64)
    # indices is (1, 200) or (N, 200)
    if indices.ndim == 1:
        indices = indices[np.newaxis, :]

    rows = []
    for i in range(min(len(indices), n_copies)):
        seq = indices_to_seq(indices[i % len(indices)])
        rows.append({'sequence': seq, 'seed_type': 'gpa_seed'})

    # If n_copies > len(indices), replicate
    if n_copies > len(indices):
        base_seq = indices_to_seq(indices[0])
        for i in range(len(indices), n_copies):
            rows.append({'sequence': base_seq, 'seed_type': 'gpa_seed'})

    return pd.DataFrame(rows)


def edit_distance(seq1, seq2):
    """Hamming distance between two sequence strings."""
    return sum(a != b for a, b in zip(seq1, seq2))


def load_pool_h5(h5_path: str, slice_spec: str | None = None) -> pd.DataFrame:
    """Load any seed-pool h5 (has /indices (N, L) int) and return DataFrame
    with columns ('sequence', 'seed_type', 'seed_idx_in_pool')."""
    with h5py.File(h5_path, 'r') as f:
        idx = f['indices'][:].astype(np.int64)
    if slice_spec:
        a, b = (int(x) for x in slice_spec.split(':'))
        idx = idx[a:b]
    rows = [{'sequence': indices_to_seq(idx[i]),
             'seed_type': 'pool_h5',
             'seed_idx_in_pool': i} for i in range(len(idx))]
    return pd.DataFrame(rows)


def run_ledidi_pool_batched(seeds_df: pd.DataFrame, oracle_wrapper, device, args):
    """Batched-across-seeds LEDIDI with iteration-snapshot capture.

    For each inner batch of M=cross_seed_batch_size distinct seeds:
      - Allocate weights tensor (M, 4, L) shared as one nn.Parameter
      - At each iter: sample 1 Gumbel-softmax draw per seed, score in one
        forward pass, compute mean loss, AdamW step
      - At each ``snapshot_iter`` and at the per-seed best-LN moment, capture
        the *deterministic* argmax(log(X+eps) + weights) → discrete sequence
        and score it with the oracle. This yields one row per (seed, snapshot)
        instead of one per seed, so a single ``--l`` per pool covers the
        edit-distance trajectory analogous to ISM's per-gen output.

    snapshot_iter=-1 marks the per-seed best-LN row (the prior 'final' output).

    Returns DataFrame with cols: seed_idx, seed_type, snapshot_iter,
    seed_sequence, edited_sequence, seed_legnet, edited_legnet, edit_distance,
    elapsed_s, status.
    """
    snapshot_iters = sorted({int(x) for x in args.snapshot_iters.split(',')
                             if x.strip()})
    snapshot_set = set(snapshot_iters)
    M_total = len(seeds_df)
    L = SEQ_LEN
    n_chan = N_CATEGORIES
    target_val = args.target
    eps = 1e-4
    M_chunk = args.cross_seed_batch_size
    results = []

    print(f"  Snapshot iters: {snapshot_iters} (+ per-seed best as iter=-1)",
          flush=True)
    print(f"  Scoring {M_total} seeds at baseline ...", flush=True)
    seed_onehots = []
    for _, row in seeds_df.iterrows():
        seed_onehots.append(seq_to_onehot(row['sequence']).squeeze(0))
    X_full = torch.stack(seed_onehots, dim=0).to(device)        # (M_total, 4, L)
    with torch.no_grad():
        seed_scores = oracle_wrapper(X_full).squeeze(-1).cpu().numpy()

    target_t = torch.full((1, 1), target_val, dtype=torch.float32, device=device)
    n_chunks = (M_total + M_chunk - 1) // M_chunk

    for ci in range(n_chunks):
        s0 = ci * M_chunk
        s1 = min(s0 + M_chunk, M_total)
        M = s1 - s0
        chunk_df = seeds_df.iloc[s0:s1].reset_index(drop=True)
        seed_seqs_chunk = chunk_df['sequence'].tolist()
        X = X_full[s0:s1].clone()                                # (M, 4, L)
        weights = torch.zeros(M, n_chan, L, device=device,
                              dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.AdamW([weights], lr=args.lr)

        # Per-seed best (over all iters): track max edited_legnet
        best_score = torch.full((M,), float('-inf'), device=device)
        best_seqs = [seed_seqs_chunk[m] for m in range(M)]
        best_edits = [0] * M
        no_improve = torch.zeros(M, dtype=torch.long, device=device)
        last_best_obj = torch.full((M,), float('inf'), device=device)

        log_X = torch.log(X + eps)                                # (M, 4, L)
        target_per_seed = target_t.expand(M, 1)
        chunk_rows = []
        t0 = time.time()

        def capture_snapshot(it_label):
            """Deterministic argmax snapshot at this iter; score + record."""
            with torch.no_grad():
                snap_idx = (log_X + weights).argmax(dim=1)        # (M, L)
                snap_onehot = (torch.nn.functional.one_hot(snap_idx, n_chan)
                               .permute(0, 2, 1).float())         # (M, 4, L)
                snap_scores = oracle_wrapper(snap_onehot).squeeze(-1).cpu().numpy()
                snap_idx_cpu = snap_idx.cpu().numpy()
            for m in range(M):
                seq_str = ''.join(IDX_TO_BASE[int(b)] for b in snap_idx_cpu[m])
                n_edits = edit_distance(seed_seqs_chunk[m], seq_str)
                chunk_rows.append({
                    'seed_idx': int(chunk_df.loc[m, 'seed_idx_in_pool']),
                    'seed_type': 'pool_h5',
                    'snapshot_iter': it_label,
                    'seed_sequence': seed_seqs_chunk[m],
                    'edited_sequence': seq_str,
                    'seed_legnet': float(seed_scores[s0 + m]),
                    'edited_legnet': float(snap_scores[m]),
                    'edit_distance': n_edits,
                    'elapsed_s': float(time.time() - t0),
                    'status': 'ok',
                })
                if snap_scores[m] > best_score[m].item():
                    best_score[m] = float(snap_scores[m])
                    best_seqs[m]  = seq_str
                    best_edits[m] = n_edits

        # iter=0 baseline (edit_distance=0) — covers the cap=0 corner
        capture_snapshot(0)

        last_it = 0
        for it in range(1, args.max_iter + 1):
            logits = log_X + weights
            X_hat = torch.nn.functional.gumbel_softmax(
                logits, tau=args.tau, hard=True, dim=1)
            y_hat = oracle_wrapper(X_hat)                          # (M, 1)
            output_loss_per = (y_hat - target_per_seed).pow(2).squeeze(-1)
            input_loss_per = (X_hat - X).abs().sum(dim=(1, 2)) / 2.0
            total_per = output_loss_per + args.l * input_loss_per
            loss = total_per.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                improved = output_loss_per < last_best_obj
                last_best_obj = torch.where(improved, output_loss_per, last_best_obj)
                no_improve = torch.where(improved, torch.zeros_like(no_improve),
                                         no_improve + 1)

            last_it = it
            if it in snapshot_set:
                capture_snapshot(it)

            if (it % 100 == 0) or (it == 1):
                print(f"    chunk {ci+1}/{n_chunks} iter {it}/{args.max_iter}  "
                      f"loss={loss.item():.3f}  best_obj_mean="
                      f"{last_best_obj.mean().item():.3f}", flush=True)

            if (no_improve >= args.early_stop).all():
                print(f"    chunk {ci+1}/{n_chunks} early stop at iter {it}",
                      flush=True)
                break

        # Always capture final iter if not already a snapshot
        if last_it not in snapshot_set and last_it > 0:
            capture_snapshot(last_it)

        # Per-seed best snapshot: a duplicate row tagged snapshot_iter=-1
        for m in range(M):
            chunk_rows.append({
                'seed_idx': int(chunk_df.loc[m, 'seed_idx_in_pool']),
                'seed_type': 'pool_h5',
                'snapshot_iter': -1,
                'seed_sequence': seed_seqs_chunk[m],
                'edited_sequence': best_seqs[m],
                'seed_legnet': float(seed_scores[s0 + m]),
                'edited_legnet': float(best_score[m].item()),
                'edit_distance': int(best_edits[m]),
                'elapsed_s': float(time.time() - t0),
                'status': 'ok',
            })

        elapsed = time.time() - t0
        results.extend(chunk_rows)
        n_per_seed = len(chunk_rows) // M
        print(f"  chunk {ci+1}/{n_chunks} done: M={M} wall={elapsed:.1f}s "
              f"({elapsed/M:.2f}s/seed) snapshots/seed={n_per_seed} "
              f"best_LN_mean={best_score.mean().item():.3f} "
              f"best_LN_max={best_score.max().item():.3f}",
              flush=True)

    return pd.DataFrame(results)


# =============================================================================
# Main LEDIDI optimization
# =============================================================================
def run_ledidi_on_seeds(seeds_df, oracle_wrapper, device, args):
    """Run LEDIDI on all seed sequences.

    Args:
        seeds_df: DataFrame with 'sequence' and 'seed_type' columns.
        oracle_wrapper: LedidiOracleWrapper (nn.Module).
        device: torch device.
        args: CLI arguments.

    Returns:
        DataFrame with results.
    """
    results = []

    for seq_num, (i, row) in enumerate(seeds_df.iterrows()):
        seed_seq = row['sequence']
        seed_type = row['seed_type']

        # Score seed with oracle
        seed_onehot = seq_to_onehot(seed_seq).to(device)

        with torch.no_grad():
            seed_score = oracle_wrapper(seed_onehot).squeeze().item()

        # Set target — (1, 1) shape to match model's (B, 1) output
        target = torch.tensor([[args.target]], dtype=torch.float32).to(device)

        # Create LEDIDI instance for this sequence
        ledidi = Ledidi(
            model=oracle_wrapper,
            shape=(N_CATEGORIES, SEQ_LEN),
            target=None,                          # Single-output model
            tau=args.tau,
            l=args.l,
            batch_size=args.ledidi_batch_size,
            max_iter=args.max_iter,
            early_stopping_iter=args.early_stop,
            lr=args.lr,
            verbose=False,
        )
        ledidi.to(device)

        # Run optimization
        t0 = time.time()
        try:
            edited_onehot = ledidi.fit_transform(seed_onehot, target)
        except Exception as e:
            print(f"  [{seq_num+1}] LEDIDI failed on {seed_type} seed: {e}")
            results.append({
                'seed_idx': i,
                'seed_type': seed_type,
                'seed_sequence': seed_seq,
                'edited_sequence': seed_seq,
                'seed_legnet': seed_score,
                'edited_legnet': seed_score,
                'edit_distance': 0,
                'elapsed_s': 0.0,
                'status': 'failed',
            })
            continue
        elapsed = time.time() - t0

        # fit_transform returns (batch_size, 4, L) — pick best from batch
        if isinstance(edited_onehot, torch.Tensor):
            edited_batch = edited_onehot.detach().to(device)
        else:
            edited_batch = torch.tensor(edited_onehot).to(device)

        # Score all samples in batch, pick highest
        with torch.no_grad():
            batch_scores = oracle_wrapper(edited_batch).squeeze(-1)  # (batch_size,)
        best_idx = batch_scores.argmax().item()
        edited_arr = edited_batch[best_idx]  # (4, L)
        edited_seq = onehot_to_seq(edited_arr)
        edited_score = batch_scores[best_idx].item()

        n_edits = edit_distance(seed_seq, edited_seq)

        if (seq_num + 1) % 10 == 0 or seq_num == 0:
            print(f"  [{seq_num+1}/{len(seeds_df)}] {seed_type}: "
                  f"LegNet {seed_score:.2f} -> {edited_score:.2f} "
                  f"({n_edits} edits, {elapsed:.1f}s)")

        results.append({
            'seed_idx': i,
            'seed_type': seed_type,
            'seed_sequence': seed_seq,
            'edited_sequence': edited_seq,
            'seed_legnet': seed_score,
            'edited_legnet': edited_score,
            'edit_distance': n_edits,
            'elapsed_s': elapsed,
            'status': 'ok',
        })

    return pd.DataFrame(results)


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description='LEDIDI baseline on K562 LentiMPRA')
    p.add_argument('--oracle_ckpt', default=ORACLE_CKPT)
    p.add_argument('--oracle_type', choices=['legnet', 'alphagenome_torch'],
                   default='legnet',
                   help='legnet = LegNet K562 (default); alphagenome_torch = '
                        'differentiable NEW torch stage2 AG (needs gpa_torchag env)')
    p.add_argument('--ism_csv', default=ISM_CSV)
    p.add_argument('--gpa_seed_h5', default=GPA_SEED_H5)
    p.add_argument('--output_dir', default=OUTPUT_DIR)
    p.add_argument('--seed_mode', choices=['ism', 'random_init', 'gpa_seed', 'pool_h5'],
                   default='ism',
                   help='Seed mode: ism (ISM CSV seeds), random_init (uniform '
                        'random 200bp), gpa_seed (single_seed_299823.h5), '
                        'pool_h5 (any seed-pool h5 with /indices (N,L) — uses '
                        'the batched-across-seeds inner loop)')
    p.add_argument('--seed_pool', default=None,
                   help='Path to seed-pool h5 (required if --seed_mode=pool_h5)')
    p.add_argument('--cross_seed_batch_size', type=int, default=64,
                   help='How many distinct seeds optimized in parallel per inner '
                        'batch (only used by --seed_mode=pool_h5). Each seed has '
                        'its own (4,L) weights tensor; one optimizer step is '
                        'shared across the inner batch.')
    p.add_argument('--snapshot_iters',
                   default='25,50,100,200,500,1000',
                   help='Comma-separated optimizer iters at which to capture a '
                        'deterministic argmax snapshot (per seed). Single LEDIDI '
                        'run thus walks across edit-distance values just like ISM '
                        'gens. Always also captures the per-seed best snapshot '
                        '(highest-LN over the run) at the end as snapshot_iter=-1.')
    p.add_argument('--slice', default=None,
                   help='Optional "a:b" subset for smoke testing (pool_h5 mode)')
    p.add_argument('--n_random', type=int, default=100,
                   help='Number of random seeds (for --seed_mode random_init)')
    p.add_argument('--n_gpa_copies', type=int, default=50,
                   help='Number of LEDIDI runs from GPA seed (for --seed_mode gpa_seed). '
                        'Each run uses same seed but different Gumbel samples.')
    p.add_argument('--target', type=float, default=DEFAULT_TARGET,
                   help='Target oracle score for LEDIDI optimization')
    p.add_argument('--l', type=float, default=DEFAULT_L,
                   help='L1 regularization weight (controls edit count)')
    p.add_argument('--tau', type=float, default=DEFAULT_TAU,
                   help='Gumbel-softmax temperature')
    p.add_argument('--lr', type=float, default=DEFAULT_LR,
                   help='Learning rate')
    p.add_argument('--max_iter', type=int, default=DEFAULT_MAX_ITER)
    p.add_argument('--ledidi_batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                   help='Gumbel samples per iteration')
    p.add_argument('--early_stop', type=int, default=DEFAULT_EARLY_STOP)
    p.add_argument('--seed_type', choices=['natural', 'random', 'both'],
                   default='both', help='Which ISM seed types to run (only for --seed_mode ism)')
    p.add_argument('--rng_seed', type=int, default=None,
                   help='RNG seed for Gumbel-softmax sampling (torch+numpy+cuda). '
                        'Set to different values for independent multi-seed runs; '
                        'None keeps torch default (legacy behavior).')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.rng_seed is not None:
        np.random.seed(args.rng_seed)
        torch.manual_seed(args.rng_seed)
        torch.cuda.manual_seed_all(args.rng_seed)
        print(f"  RNG seed: {args.rng_seed}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 72)
    print("LEDIDI Baseline — K562 LentiMPRA")
    print("=" * 72)
    print(f"  Seed mode: {args.seed_mode}")
    print(f"  Oracle: {args.oracle_ckpt}")
    print(f"  Target: {args.target}")
    print(f"  L1 weight (l): {args.l}")
    print(f"  Temperature (tau): {args.tau}")
    print(f"  LR: {args.lr}")
    print(f"  Max iter: {args.max_iter}")
    print(f"  Batch size: {args.ledidi_batch_size}")
    print(f"  Early stop: {args.early_stop}")
    print(f"  Device: {device}")

    # Load oracle
    if args.oracle_type == 'alphagenome_torch':
        from scripts.k562_mdlm_gpa.k562_oracle import TorchAGOracle
        print("\nLoading NEW torch stage2 AG oracle (alphagenome_torch)...")
        base_oracle = TorchAGOracle(device=str(device), cell="k562")
        # freeze the AG backbone so LEDIDI only optimizes the input edits
        for p in base_oracle.model.parameters():
            p.requires_grad_(False)
    else:
        print("\nLoading K562 LegNet oracle...")
        legnet_model = load_k562_oracle(args.oracle_ckpt, device=str(device))
        base_oracle = K562Oracle(legnet_model, device=str(device))
    # LedidiOracleWrapper calls base_oracle.dps_forward (present on both oracles)
    oracle_wrapper = LedidiOracleWrapper(base_oracle).to(device)
    oracle_wrapper.eval()
    for param in oracle_wrapper.parameters():
        param.requires_grad = False
    print("  Oracle loaded and frozen.")

    # Load seeds based on mode
    if args.seed_mode == 'ism':
        print(f"\nLoading ISM seeds from {args.ism_csv}...")
        df = pd.read_csv(args.ism_csv)
        df_dedup = df.drop_duplicates(subset='sequence').reset_index(drop=True)
        # Rename ISM seed_type column for consistency
        df_dedup = df_dedup.rename(columns={'ism_seed_type': 'seed_type'})
        print(f"  Total rows: {len(df)}, unique sequences: {len(df_dedup)}")
        print(f"  Seed types: {dict(df_dedup['seed_type'].value_counts())}")
        if args.seed_type != 'both':
            df_dedup = df_dedup[df_dedup['seed_type'] == args.seed_type].reset_index(drop=True)
            print(f"  Filtered to {args.seed_type}: {len(df_dedup)} sequences")

    elif args.seed_mode == 'random_init':
        print(f"\nGenerating {args.n_random} uniform random seeds...")
        df_dedup = generate_random_seeds(args.n_random)
        print(f"  Generated {len(df_dedup)} random 200bp sequences")

    elif args.seed_mode == 'gpa_seed':
        print(f"\nLoading GPA seed from {args.gpa_seed_h5}...")
        df_dedup = load_gpa_seed(args.gpa_seed_h5, n_copies=args.n_gpa_copies)
        print(f"  Loaded {len(df_dedup)} copies of GPA seed for independent LEDIDI runs")

    elif args.seed_mode == 'pool_h5':
        if not args.seed_pool:
            raise ValueError("--seed_mode=pool_h5 requires --seed_pool <h5>")
        print(f"\nLoading pool h5 from {args.seed_pool} (slice={args.slice})...")
        df_dedup = load_pool_h5(args.seed_pool, slice_spec=args.slice)
        print(f"  Loaded {len(df_dedup)} sequences from pool")

    # Run LEDIDI — dispatch to batched-across-seeds for pool_h5, per-seed otherwise
    if args.seed_mode == 'pool_h5':
        print(f"\nRunning batched-across-seeds LEDIDI on {len(df_dedup)} sequences "
              f"(M={args.cross_seed_batch_size} per inner batch)...")
        results_df = run_ledidi_pool_batched(df_dedup, oracle_wrapper, device, args)
    else:
        print(f"\nRunning LEDIDI (per-seed) on {len(df_dedup)} sequences...")
        results_df = run_ledidi_on_seeds(df_dedup, oracle_wrapper, device, args)

    # Split and save by seed type
    for stype in results_df['seed_type'].unique():
        subset = results_df[results_df['seed_type'] == stype]
        out_path = os.path.join(args.output_dir, f'ledidi_edited_{stype}.csv')
        subset.to_csv(out_path, index=False)
        print(f"\nSaved {len(subset)} {stype} results to {out_path}")

        # Summary
        ok = subset[subset['status'] == 'ok']
        if len(ok) > 0:
            print(f"  LegNet: seed {ok['seed_legnet'].mean():.2f} -> "
                  f"edited {ok['edited_legnet'].mean():.2f} "
                  f"(delta +{ok['edited_legnet'].mean() - ok['seed_legnet'].mean():.2f})")
            print(f"  Edit distance: {ok['edit_distance'].mean():.1f} +/- {ok['edit_distance'].std():.1f}")
            print(f"  Max edited LegNet: {ok['edited_legnet'].max():.2f}")
            print(f"  Runtime: {ok['elapsed_s'].mean():.1f}s/seq")

    # Also save combined
    combined_path = os.path.join(args.output_dir, 'ledidi_edited_all.csv')
    results_df.to_csv(combined_path, index=False)
    print(f"\nCombined results: {combined_path}")
    print("Done.")


if __name__ == '__main__':
    main()
