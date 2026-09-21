#!/usr/bin/env python3
"""
GPA + DPS with LentiMPRA MDLM backbone and K562 oracle.

Uses DiffusionPopulationAnnealer with LentiMPRA-trained MDLM (200bp core)
and K562 oracle (either LegNet or AlphaGenome).

Oracle types:
  - legnet: K562 lentiMPRA LegNet CNN (230bp = 15bp adapter + 200bp + 15bp adapter).
            Supports DPS (differentiable). Default.
  - alphagenome: AlphaGenome K562 fine-tuned oracle (JAX, via socket server).
                 NO DPS support (not differentiable). GPA-only.

Single cell type (K562 only) -- no specificity penalty. pw=0 everywhere.
Adapter injection happens inside the oracle wrapper at scoring time.

Usage:
    python scripts/k562_mdlm_gpa/run_k562_mdlm_gpa.py \
        --mdlm_checkpoint SGDD/.../best.ckpt \
        --oracle_checkpoint CFG-SDDD/model_zoo/lentimpra/oracle_models/best_model-epoch=24-val_pearson=0.814.ckpt \
        --from_random --population_size 5000 --max_beta 200.0 \
        --use_dps --dps_eta 3000 \
        --output_dir results/k562_mdlm_gpa/run_001

    # AlphaGenome as fitness oracle (no DPS)
    python scripts/k562_mdlm_gpa/run_k562_mdlm_gpa.py \
        --mdlm_checkpoint SGDD/.../best.ckpt \
        --oracle_type alphagenome \
        --from_random --population_size 5000 --max_beta 200.0 \
        --output_dir results/k562_mdlm_gpa/run_v3_ag_baseline
"""

import os
import sys
import argparse
import json
import random
import subprocess
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
import torch
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.gpa_sampling import DiffusionPopulationAnnealer
from scripts.k562_mdlm_gpa.mdlm_wrapper import (
    load_mdlm_lentimpra, MDLMMutator, MDLMMutatorDPS, MDLMMutatorKLConstrained,
)
from scripts.k562_mdlm_gpa.k562_oracle import load_k562_oracle, K562Oracle, CascadeOracle

SEQ_LENGTH = 200  # MDLM core sequence length


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPA + DPS with LentiMPRA MDLM + K562 LegNet oracle")

    # Model paths
    parser.add_argument("--mdlm_checkpoint", required=True,
                        help="LentiMPRA MDLM checkpoint (best.ckpt)")
    parser.add_argument("--oracle_checkpoint", default=None,
                        help="K562 LegNet oracle checkpoint (.ckpt) — required for oracle_type=legnet")
    parser.add_argument("--sgdd_dir", default=None,
                        help="Path to SGDD repo (default: ~/SGDD)")

    # Oracle type
    parser.add_argument("--oracle_type", default="legnet",
                        choices=["legnet", "alphagenome", "alphagenome_torch"],
                        help="Fitness oracle: legnet (LegNet CNN, supports DPS), "
                             "alphagenome (AG JAX socket, noDPS), or "
                             "alphagenome_torch (torch AG stage2, in-process, "
                             "DIFFERENTIABLE -> supports DPS)")
    parser.add_argument("--ag_socket_path", default=None,
                        help="Unix socket for AG oracle server (auto-generated if None)")
    parser.add_argument("--ag_server_batch_size", type=int, default=64,
                        help="Batch size for AG server inference")

    # Population initialization
    parser.add_argument("--seed_pool", default=None,
                        help="H5 seed pool (indices dataset, 200bp)")
    parser.add_argument("--from_random", action="store_true",
                        help="Initialize from random DNA sequences")
    parser.add_argument("--gc_balanced_init", action="store_true",
                        help="Generate equal particles per GC bin (with --from_random + --gc_bin_edges)")
    parser.add_argument("--population_size", type=int, default=5000)
    parser.add_argument("--top_k_init", action="store_true",
                        help="Use top-K seeds by oracle score")

    # GPA parameters
    parser.add_argument("--max_beta", type=float, default=200.0)
    parser.add_argument("--ess_threshold", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--min_delta_beta", type=float, default=1e-4)

    # Adaptive continuation
    parser.add_argument("--extend_on_delta_beta", type=float, default=0.5)
    parser.add_argument("--extend_steps", type=int, default=10)
    parser.add_argument("--hard_max_steps", type=int, default=200)
    parser.add_argument("--start_beta", type=float, default=0.0,
                        help="Starting beta for SMC continuation from a previous run")
    parser.add_argument("--max_delta_beta", type=float, default=0.0,
                        help="Cap delta-beta per step (0=disabled)")

    # Mutation parameters
    parser.add_argument("--noise_fraction", type=float, default=0.05,
                        help="Fraction of positions to mask per mutation")
    parser.add_argument("--mutation_batch_size", type=int, default=128)

    # DPS parameters
    parser.add_argument("--use_dps", action="store_true",
                        help="Use DPS oracle-guided mutation")
    parser.add_argument("--dps_eta", type=float, default=3000.0,
                        help="DPS gradient step size")
    parser.add_argument("--dps_tau", type=float, default=1.0,
                        help="Gumbel-softmax temperature for DPS")

    # DPS eta annealing
    parser.add_argument("--eta_start", type=float, default=0.0,
                        help="Starting DPS eta for annealing (0=disabled, use constant dps_eta)")
    parser.add_argument("--eta_end", type=float, default=0.0,
                        help="Ending DPS eta for annealing")

    # Diversity enforcement via duplicate re-mutation
    parser.add_argument("--dedup_threshold", type=int, default=0,
                        help="Hamming distance threshold for near-duplicate re-mutation (0=disabled)")
    parser.add_argument("--dedup_nf_mult", type=float, default=2.0,
                        help="Noise fraction multiplier for re-mutation of duplicates")

    # Grammar-preserving DPS modes
    parser.add_argument("--dps_mode", default="standard",
                        choices=["standard", "kl_constrained"],
                        help="DPS variant: standard or kl_constrained (KL-budget projection)")
    parser.add_argument("--kl_budget", type=float, default=1.0,
                        help="Per-position KL budget (nats) for kl_constrained mode")
    parser.add_argument("--kl_mode", default="per_position",
                        choices=["per_position", "global"],
                        help="KL constraint scope")

    # Per-step edit budget
    parser.add_argument("--max_edits_per_step", type=int, default=0,
                        help="Max new edits per GPA step (0=disabled)")

    # Hill-climb
    parser.add_argument("--hill_climb", action="store_true",
                        help="Hill-climb per position after K-branch selection")
    parser.add_argument("--hill_climb_budget", type=int, default=0,
                        help="Independent HC edit budget per GPA step (0=disabled)")
    parser.add_argument("--hill_climb_full_vocab", action="store_true",
                        help="Try all 3 alternative DNA tokens at all positions")
    parser.add_argument("--hill_climb_positions", type=int, default=0,
                        help="Number of positions to subsample per HC round (0=all)")

    # MCTS
    parser.add_argument("--mcts_depth", type=int, default=0,
                        help="MCTS max tree depth (0=disabled)")
    parser.add_argument("--mcts_iterations", type=int, default=12,
                        help="MCTS expansion budget per particle per GPA step")
    parser.add_argument("--mcts_c", type=float, default=1.41,
                        help="UCB1 exploration constant")

    # Elitism
    parser.add_argument("--elite_fraction", type=float, default=0.0,
                        help="Fraction of top particles preserved each step (0=disabled)")

    # GC fitness penalty
    parser.add_argument("--gc_fitness_weight", type=float, default=0.0,
                        help="GC penalty in fitness (0=disabled)")
    parser.add_argument("--gc_fitness_low", type=float, default=0.40)
    parser.add_argument("--gc_fitness_high", type=float, default=0.60)

    # Bio filter
    parser.add_argument("--bio_filter", action="store_true",
                        help="Reject non-bio-plausible mutations")
    parser.add_argument("--gc_low", type=float, default=0.40)
    parser.add_argument("--gc_high", type=float, default=0.70)

    # NF annealing
    parser.add_argument("--nf_start", type=float, default=0.0,
                        help="Starting noise fraction for annealing (0=disabled)")
    parser.add_argument("--nf_end", type=float, default=0.0,
                        help="Ending noise fraction for annealing")

    # GC-aware DPS
    parser.add_argument("--gc_dps_weight", type=float, default=0.0,
                        help="Weight for GC centering loss in DPS gradient (0=disabled)")
    parser.add_argument("--gc_dps_target", type=float, default=0.50,
                        help="Target GC fraction for GC-aware DPS")

    # Direct GC pull
    parser.add_argument("--gc_pull_weight", type=float, default=0.0,
                        help="Direct GC pull bias weight (0=disabled)")
    parser.add_argument("--gc_pull_target", type=float, default=0.50,
                        help="Target GC fraction for direct GC pull")

    # GC-stratified resampling
    parser.add_argument("--gc_bin_edges", default=None,
                        help="GC bin edges for stratified resampling, comma-separated")
    parser.add_argument("--gc_bin_min_quota", type=int, default=50,
                        help="Minimum particles per GC bin")
    parser.add_argument("--gc_bin_alloc_mode", default="raw",
                        choices=["raw", "minshift", "exp"],
                        help="GC bin allocation formula")
    parser.add_argument("--gc_bin_alloc_alpha", type=float, default=5.0,
                        help="Sharpening exponent for exp allocation mode")

    # Lagrangian multi-objective fitness (MOG-DFM-inspired)
    parser.add_argument("--use_lagrangian", action="store_true",
                        help="Use self-tuning Lagrangian fitness (rank-based, GC constraint)")
    parser.add_argument("--lagrangian_gc_target", type=float, default=0.50,
                        help="GC target for Lagrangian constraint")
    parser.add_argument("--lagrangian_gc_tolerance", type=float, default=0.05,
                        help="GC tolerance (lambda activates beyond this deviation)")
    parser.add_argument("--lagrangian_lr", type=float, default=0.1,
                        help="Lambda learning rate for Lagrangian dual ascent")

    # EDTS
    parser.add_argument("--branch_factor", type=int, default=1,
                        help="K-branch factor per particle per step (1=standard)")
    parser.add_argument("--max_edit_frac", type=float, default=1.0,
                        help="Max fraction of positions differing from original seed")
    parser.add_argument("--edit_select_mode", default="efficiency",
                        choices=["efficiency", "oracle"],
                        help="Branch selection mode")

    # v7 optimization flags
    parser.add_argument("--cache_mutation_scores", action="store_true",
                        help="Skip post-mutation re-score when K-branch already knows winner scores")
    parser.add_argument("--entropy_hc", action="store_true",
                        help="Entropy-weighted HC position sampling (high-entropy positions first)")
    parser.add_argument("--prefilter_topk", type=int, default=0,
                        help="MDLM pre-filter: score only top-J branches by logprob (0=off, score all K)")
    parser.add_argument("--gpa_fitness_mode", default="oracle",
                        choices=["oracle", "mdlm_pll", "hybrid"],
                        help="Fitness source: oracle (default), mdlm_pll (model logprob), "
                             "hybrid (PLL warmup then oracle)")
    parser.add_argument("--pll_warmup_steps", type=int, default=0,
                        help="Oracle-free PLL fitness for first N steps (hybrid mode)")
    parser.add_argument("--diversity_lambda", type=float, default=0.0,
                        help="Conformity penalty: fitness -= lambda * conformity (0=off)")
    parser.add_argument("--surrogate_prefilter", action="store_true",
                        help="Online k-mer surrogate for branch pre-filtering (re-fit every 5 steps)")
    parser.add_argument("--cascade_oracle", action="store_true",
                        help="LegNet/AG cascade: LegNet for branch selection, AG for final scoring")

    # AlphaGenome eval oracle (JAX subprocess)
    parser.add_argument("--eval_ag", action="store_true",
                        help="Score final population with AlphaGenome K562 oracle "
                             "(runs as subprocess in alphagenome conda env)")
    parser.add_argument("--ag_batch_size", type=int, default=64,
                        help="Batch size for AlphaGenome scoring")
    parser.add_argument("--eval_checkpoint_interval", type=int, default=5,
                        help="Score with AG eval oracle every N GPA steps (0=final only)")
    parser.add_argument("--eval_early_stop_patience", type=int, default=0,
                        help="Stop after N eval checkpoints with no AG improvement (0=disabled)")
    parser.add_argument("--archive_threshold", type=float, default=None,
                        help="AG score threshold: archive any sequence exceeding this at each "
                             "eval checkpoint step (requires --eval_ag; use with "
                             "--eval_checkpoint_interval 1 to catch transient sequences)")
    parser.add_argument("--topk_ag_size", type=int, default=0,
                        help="Capacity of the top-K-by-AG evicting pool (0=disabled). "
                             "Keeps the highest-AG sequences seen at any eval checkpoint, "
                             "deduped by sequence, with online eviction.")
    parser.add_argument("--nf_boost_delta", type=float, default=0.0,
                        help="Adaptive NF boost: increase NF by this amount on eval plateau (0=disabled)")
    parser.add_argument("--nf_boost_max", type=float, default=0.0,
                        help="Maximum NF when boosting (0=disabled)")


    # Diagnostics
    parser.add_argument("--diversity_subsample", type=int, default=500)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    # Output
    parser.add_argument("--output_dir", required=True)

    return parser.parse_args()


def initialize_population(args, oracle, device):
    """Initialize population from seed pool or random sequences."""
    N = args.population_size
    L = SEQ_LENGTH

    if args.seed_pool is not None:
        print(f"\n[Init] Loading seed pool: {args.seed_pool}")
        with h5py.File(args.seed_pool, 'r') as f:
            if 'indices' in f:
                indices = f['indices'][:]
            elif 'arr_0' in f:
                onehot = f['arr_0'][:]  # (N, 4, L)
                indices = np.argmax(onehot, axis=1)
            else:
                raise KeyError(f"No 'indices' or 'arr_0' in {args.seed_pool}")

        N_pool = len(indices)
        print(f"  Pool size: {N_pool:,}")

        if args.top_k_init:
            print(f"  Re-scoring {N_pool:,} sequences with oracle...")
            scores, _ = oracle.score(torch.from_numpy(indices).long().to(device))
            print(f"  Oracle: mean={scores.mean():.3f}, max={scores.max():.3f}")
            ranked = np.argsort(scores)[::-1].copy()
            chosen = ranked[:min(N, N_pool)]
            if len(chosen) < N:
                chosen = np.concatenate([
                    chosen, np.random.choice(chosen, N - len(chosen))
                ])
            print(f"  Top-K: selected top {N:,}")
        elif N_pool >= N:
            chosen = np.random.choice(N_pool, N, replace=False)
        else:
            chosen = np.random.choice(N_pool, N, replace=True)

        population = torch.from_numpy(indices[chosen]).long()
        print(f"  Initialized {N:,} particles")
        return population

    elif args.from_random:
        if args.gc_balanced_init and args.gc_bin_edges:
            bin_edges = np.array([float(x) for x in args.gc_bin_edges.split(",")])
            n_bins = len(bin_edges) - 1
            per_bin = N // n_bins
            remainder = N - per_bin * n_bins
            quotas = [per_bin + (1 if i < remainder else 0) for i in range(n_bins)]

            print(f"\n[Init] GC-balanced random: {N:,} x {L}bp, {n_bins} bins, {per_bin}/bin")
            all_seqs = []
            for b in range(n_bins):
                lo, hi = bin_edges[b], bin_edges[b + 1]
                target = quotas[b]
                collected = []
                batch = max(target * 4, 1000)
                attempts = 0
                while len(collected) < target and attempts < 100:
                    seqs = torch.randint(0, 4, (batch, L))
                    gc = ((seqs == 1) | (seqs == 2)).float().mean(dim=1)
                    mask = (gc >= lo) & (gc < hi)
                    collected.append(seqs[mask])
                    attempts += 1
                bin_seqs = torch.cat(collected, dim=0)[:target]
                if len(bin_seqs) < target:
                    pad = target - len(bin_seqs)
                    bin_seqs = torch.cat([bin_seqs, bin_seqs[:pad]], dim=0)
                all_seqs.append(bin_seqs)
                gc_actual = ((bin_seqs == 1) | (bin_seqs == 2)).float().mean(dim=1)
                print(f"  Bin [{lo:.2f}, {hi:.2f}): {len(bin_seqs)} seqs, "
                      f"GC={gc_actual.mean():.3f}")
            population = torch.cat(all_seqs, dim=0)
            perm = torch.randperm(population.shape[0])
            population = population[perm]
            return population
        else:
            print(f"\n[Init] Random DNA sequences: {N:,} x {L}bp")
            population = torch.randint(0, 4, (N, L))
            return population

    else:
        raise ValueError("Must specify --seed_pool or --from_random")


def start_ag_server(socket_path, batch_size=64):
    """Start AG oracle server as subprocess in alphagenome conda env.

    Returns (subprocess.Popen, socket_path).

    The subprocess inherits this GPA process's env by default — but the GPA
    process runs in the main `gpa` env (older ptxas, max PTX 7.8), and JAX needs
    PTX >= 8.0 for H100's sm_90a. Force-prepend alphagenome's cuda_nvcc 12.9
    onto PATH and set XLA_FLAGS / XLA_PYTHON_CLIENT_PREALLOCATE per
    reference_jax_alphagenome_cluster_setup.md.
    """
    # AG_ENV env var picks the conda env: 'gpa-alphagenome' (default, jaxlib 0.9 +
    # cuDNN 9, needs driver >=555) or 'alphagenome_oldcudnn' (jaxlib 0.4.34 +
    # cuDNN 8.9.7, works on driver >=520 — covers our GPU fleet).
    ag_env = os.environ.get("AG_ENV", "gpa-alphagenome")
    ag_python = os.path.expandvars(f"${CONDA_ENV_ROOT}/{ag_env}/bin/python")
    server_script = str(Path(__file__).parent / "ag_oracle_server.py")
    cuda_nvcc_dir = os.path.expandvars(f"${CONDA_ENV_ROOT}/{ag_env}/lib/"
                     f"python3.11/site-packages/nvidia/cuda_nvcc")

    sub_env = os.environ.copy()
    sub_env["PATH"] = f"{cuda_nvcc_dir}/bin:" + sub_env.get("PATH", "")
    sub_env["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_nvcc_dir}"
    sub_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    log_path = socket_path + ".log"
    print(f"\n[AG Server] Starting: {socket_path} (log: {log_path})", flush=True)
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [ag_python, server_script,
         "--socket", socket_path,
         "--batch_size", str(batch_size)],
        stdout=log_fh, stderr=log_fh, env=sub_env,
    )

    # Wait for ready signal (server writes .ready file)
    ready_path = socket_path + ".ready"
    timeout = 600  # AG model load takes ~60-180s; increase for slow filesystem
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(ready_path):
            print(f"[AG Server] Ready (PID={proc.pid}, {time.time()-t0:.1f}s)", flush=True)
            return proc, socket_path
        if proc.poll() is not None:
            log_fh.close()
            try:
                with open(log_path) as f:
                    log_contents = f.read()
                print(f"[AG Server] Log contents:\n{log_contents}", flush=True)
            except Exception:
                pass
            raise RuntimeError(f"AG server died with rc={proc.returncode}")
        time.sleep(0.5)

    proc.kill()
    raise TimeoutError(f"AG server failed to start within {timeout}s")


def stop_ag_server(proc, client):
    """Shut down AG oracle server gracefully."""
    if client is not None:
        try:
            client.shutdown_server()
            client.close()
        except Exception:
            pass

    if proc is not None and proc.poll() is None:
        proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate oracle_type constraints
    if args.oracle_type == "legnet" and args.oracle_checkpoint is None:
        raise ValueError("--oracle_checkpoint required for oracle_type=legnet")
    if args.oracle_type == "alphagenome" and args.use_dps:
        print("\n[WARNING] DPS not supported with AlphaGenome oracle -- forcing DPS off")
        args.use_dps = False

    # Reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Repo directories
    sgdd_dir = args.sgdd_dir or str(Path.home() / "SGDD")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # AG server state (cleaned up at exit)
    ag_proc = None
    ag_client = None
    ag_eval_proc = None
    ag_eval_client = None

    print("=" * 72)
    oracle_label = args.oracle_type.upper()
    print(f"GPA (K562 LentiMPRA MDLM + {oracle_label} oracle)")
    print("=" * 72)
    print(f"  Device:           {device}")
    print(f"  Oracle type:      {args.oracle_type}")
    print(f"  MDLM checkpoint:  {args.mdlm_checkpoint}")
    if args.oracle_type == "legnet":
        print(f"  Oracle checkpoint: {args.oracle_checkpoint}")
    print(f"  Population:       {args.population_size:,}")
    print(f"  Max beta:         {args.max_beta}")
    print(f"  Noise fraction:   {args.noise_fraction}")
    print(f"  DPS:              {args.use_dps} (eta={args.dps_eta})")
    if args.bio_filter:
        print(f"  Bio filter:       [{args.gc_low}, {args.gc_high}]")
    if args.gc_fitness_weight > 0:
        print(f"  GC fitness:       w={args.gc_fitness_weight}, [{args.gc_fitness_low}, {args.gc_fitness_high}]")
    if args.gc_dps_weight > 0:
        print(f"  GC DPS weight:    {args.gc_dps_weight} (target={args.gc_dps_target})")
    if args.gc_pull_weight > 0:
        print(f"  GC pull weight:   {args.gc_pull_weight} (target={args.gc_pull_target})")
    if args.seed is not None:
        print(f"  Random seed:      {args.seed}")
    # v7 optimization flags
    if args.cache_mutation_scores:
        print(f"  Cache scores:     ON (skip post-mutation re-score)")
    if args.entropy_hc:
        print(f"  Entropy HC:       ON")
    if args.prefilter_topk > 0:
        print(f"  Pre-filter:       top-{args.prefilter_topk} of K={args.branch_factor}")
    if args.gpa_fitness_mode != "oracle":
        print(f"  Fitness mode:     {args.gpa_fitness_mode}")
    if args.pll_warmup_steps > 0:
        print(f"  PLL warmup:       {args.pll_warmup_steps} steps")
    if args.diversity_lambda > 0:
        print(f"  Diversity λ:      {args.diversity_lambda}")
    if args.surrogate_prefilter:
        print(f"  Surrogate filter: ON")
    if args.cascade_oracle:
        print(f"  Cascade oracle:   ON")
    if args.eta_start > 0:
        print(f"  Eta anneal:       {args.eta_start} -> {args.eta_end}")
    if args.dedup_threshold > 0:
        print(f"  Dedup:            threshold={args.dedup_threshold}, nf_mult={args.dedup_nf_mult}")
    if args.eval_ag:
        print(f"  AG eval:          every {args.eval_checkpoint_interval} steps")
        if args.eval_early_stop_patience > 0:
            print(f"  Early stop:       patience={args.eval_early_stop_patience} eval checkpoints")
    print(f"  Output:           {args.output_dir}")
    print("=" * 72)

    # ---- Load LentiMPRA MDLM ----
    print("\n[Model] Loading LentiMPRA MDLM...")
    mdlm = load_mdlm_lentimpra(args.mdlm_checkpoint, sgdd_dir, device)
    print(f"  Loaded: {args.mdlm_checkpoint}")

    # ---- Start AG eval server early (overlaps with oracle load) ----
    if args.eval_ag and args.oracle_type == "legnet":
        from scripts.k562_mdlm_gpa.ag_oracle_client import AlphaGenomeK562Client

        ag_eval_socket = f"/tmp/ag_eval_{os.getpid()}.sock"
        ag_eval_proc, ag_eval_socket = start_ag_server(
            ag_eval_socket, args.ag_batch_size)
        ag_eval_client = AlphaGenomeK562Client(ag_eval_socket)
        if not ag_eval_client.ping():
            raise RuntimeError("AG eval server not responding to ping")
        print(f"[AG Eval] Connected to eval server at {ag_eval_socket}")

    # ---- Load oracle ----
    oracle = None  # Will be K562Oracle (legnet) or AlphaGenomeK562Client (ag)
    legnet_oracle = None  # LegNet for cross-scoring when using AG fitness

    if args.oracle_type == "legnet":
        print("\n[Oracle] Loading K562 LegNet oracle...")
        lit_oracle = load_k562_oracle(args.oracle_checkpoint, device)
        oracle = K562Oracle(lit_oracle, device=device)
        print(f"  Loaded: {args.oracle_checkpoint}")

    elif args.oracle_type == "alphagenome_torch":
        from scripts.k562_mdlm_gpa.k562_oracle import TorchAGOracle
        oracle = TorchAGOracle(device=device, cell="k562")
        # LegNet for cross-oracle scoring at the end (reward-hacking check)
        if args.oracle_checkpoint:
            print("\n[Oracle] Loading K562 LegNet for cross-scoring...")
            legnet_oracle = K562Oracle(load_k562_oracle(args.oracle_checkpoint, device),
                                       device=device)

    elif args.oracle_type == "alphagenome":
        from scripts.k562_mdlm_gpa.ag_oracle_client import AlphaGenomeK562Client

        # Start AG server
        socket_path = args.ag_socket_path or f"/tmp/ag_oracle_{os.getpid()}.sock"
        ag_proc, socket_path = start_ag_server(socket_path, args.ag_server_batch_size)

        # Create client
        ag_client = AlphaGenomeK562Client(socket_path)
        if not ag_client.ping():
            raise RuntimeError("AG oracle server not responding to ping")
        oracle = ag_client
        print(f"[AG Client] Connected to server at {socket_path}")

        # Also load LegNet for cross-oracle scoring at the end
        if args.oracle_checkpoint:
            print("\n[Oracle] Loading K562 LegNet for cross-scoring...")
            lit_oracle = load_k562_oracle(args.oracle_checkpoint, device)
            legnet_oracle = K562Oracle(lit_oracle, device=device)
            print(f"  Loaded: {args.oracle_checkpoint}")

            # Cascade oracle: LegNet for branch selection, AG for final scoring
            if args.cascade_oracle:
                oracle = CascadeOracle(fast_oracle=legnet_oracle, accurate_oracle=ag_client)
                print(f"[Cascade] LegNet for branch pre-filter, AG for scoring")

    # ---- Sanity check: score a few random sequences ----
    print("\n[Sanity] Scoring 100 random 200bp sequences...")
    test_seqs = torch.randint(0, 4, (100, SEQ_LENGTH), device=device)
    test_scores, test_gc = oracle.score(test_seqs)
    print(f"  Oracle ({args.oracle_type}): mean={test_scores.mean():.3f}, "
          f"min={test_scores.min():.3f}, max={test_scores.max():.3f}")
    print(f"  GC:     mean={test_gc.mean():.3f}")

    # ---- Build mutation function ----
    # DPS mutators need a differentiable oracle: LegNet or torch-AG (both expose
    # dps_forward). JAX "alphagenome" is non-differentiable -> DPS already off.
    dps_oracle = oracle if args.oracle_type in ("legnet", "alphagenome_torch") else None

    def _build_mutator(nf, use_dps, eta_override=None):
        if use_dps:
            eta_val = eta_override if eta_override is not None else args.dps_eta
            base_kwargs = dict(
                noise_fraction=nf, eta=eta_val, tau_start=args.dps_tau,
                gc_dps_weight=args.gc_dps_weight, gc_dps_target=args.gc_dps_target,
                gc_pull_weight=args.gc_pull_weight, gc_pull_target=args.gc_pull_target,
            )
            if args.dps_mode == "kl_constrained":
                return MDLMMutatorKLConstrained(
                    mdlm, dps_oracle, kl_budget=args.kl_budget,
                    kl_mode=args.kl_mode, **base_kwargs)
            else:
                return MDLMMutatorDPS(mdlm, dps_oracle, **base_kwargs)
        else:
            return MDLMMutator(mdlm, noise_fraction=nf)

    mutate_fn = _build_mutator(args.noise_fraction, args.use_dps)
    if args.use_dps:
        print(f"\n[Mutation] MDLM + DPS (eta={args.dps_eta}, nf={args.noise_fraction})")
        if args.dps_mode != "standard":
            print(f"  DPS mode: {args.dps_mode}")
            if args.dps_mode == "kl_constrained":
                print(f"  KL budget: {args.kl_budget} ({args.kl_mode})")
    else:
        print(f"\n[Mutation] MDLM blind (nf={args.noise_fraction})")
    if args.mcts_depth >= 2:
        print(f"  MCTS: depth={args.mcts_depth}, iter={args.mcts_iterations}, C={args.mcts_c}")
    if args.max_edits_per_step > 0:
        print(f"  Per-step cap: {args.max_edits_per_step} edits/step")
    if args.hill_climb:
        print(f"  Hill-climb: enabled")
    if args.hill_climb_budget > 0:
        mode = "full-vocab" if args.hill_climb_full_vocab else "branch-token"
        print(f"  HC budget: {args.hill_climb_budget} edits/step ({mode})")
        if args.hill_climb_positions > 0:
            print(f"  HC positions: {args.hill_climb_positions}/200")

    # ---- Build mutate_fn_factory for nf/eta annealing ----
    mutate_fn_factory = None
    need_factory = (args.nf_start > 0 or args.eta_start > 0
                    or args.dedup_threshold > 0 or args.nf_boost_delta > 0)
    if need_factory:
        def mutate_fn_factory(nf, guidance_weight=None, eta=None):
            return _build_mutator(nf, args.use_dps, eta_override=eta)
        if args.nf_start > 0:
            print(f"  [NF Factory] Will anneal nf: {args.nf_start} -> {args.nf_end}")
        if args.eta_start > 0:
            print(f"  [Eta Factory] Will anneal eta: {args.eta_start} -> {args.eta_end}")

    # ---- Oracle scoring function (for GPA fitness) ----
    def oracle_fn(sequences_tensor):
        if not isinstance(sequences_tensor, torch.Tensor):
            sequences_tensor = torch.from_numpy(sequences_tensor).long()
        return oracle.score(sequences_tensor)

    # ---- Bio filter ----
    bio_filter_fn = None
    if args.bio_filter:
        from bio_plausibility import is_bio_plausible
        def bio_filter_fn(indices_np):
            gc_mask = (indices_np == 1) | (indices_np == 2)
            gc_fracs = gc_mask.mean(axis=1)
            bio_mask, _ = is_bio_plausible(
                indices_np, gc_fractions=gc_fracs,
                gc_range=(args.gc_low, args.gc_high))
            return bio_mask

    # ---- Fitness function (GC penalty only, no specificity) ----
    fitness_fn = None
    lagrangian = None
    if args.use_lagrangian:
        from scripts.k562_mdlm_gpa.lagrangian_fitness import LagrangianFitness
        lagrangian = LagrangianFitness(
            gc_target=args.lagrangian_gc_target,
            gc_tolerance=args.lagrangian_gc_tolerance,
            lr=args.lagrangian_lr,
        )
        fitness_fn = lagrangian
        print(f"\n[Lagrangian] Enabled: gc_target={args.lagrangian_gc_target}, "
              f"tolerance={args.lagrangian_gc_tolerance}, lr={args.lagrangian_lr}")
    elif args.gc_fitness_weight > 0:
        gc_lo, gc_hi = args.gc_fitness_low, args.gc_fitness_high
        gc_fw = args.gc_fitness_weight
        def fitness_fn(oracle_scores, gc_fractions):
            fit = oracle_scores.copy()
            drift = np.maximum(0, gc_fractions - gc_hi) + np.maximum(0, gc_lo - gc_fractions)
            fit = fit - gc_fw * drift
            return fit

    # ---- Initialize population ----
    population = initialize_population(args, oracle, device)
    N = population.shape[0]
    labels = torch.zeros(N, 1, device=device)

    # ---- Parse GC bin edges for stratified resampling ----
    gc_bin_edges = None
    if args.gc_bin_edges:
        gc_bin_edges = np.array([float(x) for x in args.gc_bin_edges.split(",")])
        print(f"\n[GC Stratified] {len(gc_bin_edges)-1} bins, min_quota={args.gc_bin_min_quota}")

    # ---- Build eval oracle callback (AG eval) ----
    eval_oracle_fn = None
    if ag_eval_client is not None:
        def eval_oracle_fn(sequences_tensor):
            """Score population with AG eval oracle; return np.array of scores."""
            try:
                if isinstance(sequences_tensor, torch.Tensor):
                    seqs = sequences_tensor
                else:
                    seqs = torch.from_numpy(sequences_tensor).long()
                scores, _ = ag_eval_client.score(seqs)
                return scores
            except Exception as e:
                print(f"\n  [AG Eval] ERROR: {e}", flush=True)
                n = len(sequences_tensor)
                return np.full(n, -np.inf)
    elif args.eval_ag and args.oracle_type in ("alphagenome", "alphagenome_torch"):
        # AG is already the fitness oracle; reuse it as the eval oracle so the
        # eval-checkpoint harvest (best_eval + topk_ag cross-step pool) runs.
        # oracle is AlphaGenomeK562Client/CascadeOracle (jax) or TorchAGOracle.
        def eval_oracle_fn(sequences_tensor):
            try:
                seqs = (sequences_tensor if isinstance(sequences_tensor, torch.Tensor)
                        else torch.from_numpy(sequences_tensor).long())
                scores, _ = oracle.score(seqs)
                return np.asarray(scores, dtype=np.float64)
            except Exception as e:
                print(f"\n  [AG Eval/fitness] ERROR: {e}", flush=True)
                return np.full(len(sequences_tensor), -np.inf)

    # ---- Run GPA ----
    gpa = DiffusionPopulationAnnealer(
        mutate_fn, oracle_fn, model=None, device=device,
        fitness_fn=fitness_fn,
        mutate_fn_factory=mutate_fn_factory,
        default_nf=args.noise_fraction,
        branch_factor=args.branch_factor,
        max_edit_frac=args.max_edit_frac,
        edit_select_mode=args.edit_select_mode,
        mcts_depth=args.mcts_depth,
        mcts_iterations=args.mcts_iterations,
        mcts_c=args.mcts_c,
        max_edits_per_step=args.max_edits_per_step,
        hill_climb=args.hill_climb,
        hill_climb_budget=args.hill_climb_budget,
        hill_climb_full_vocab=args.hill_climb_full_vocab,
        hill_climb_positions=args.hill_climb_positions,
        oracle_fn_fast=oracle_fn,
        # v7 optimization flags
        cache_mutation_scores=args.cache_mutation_scores,
        entropy_hc=args.entropy_hc,
        prefilter_topk=args.prefilter_topk,
        gpa_fitness_mode=args.gpa_fitness_mode,
        pll_warmup_steps=args.pll_warmup_steps,
        diversity_lambda=args.diversity_lambda,
        surrogate_prefilter=args.surrogate_prefilter,
    )
    (population, oracle_scores, log_weights, history,
     best_population, best_oracle_scores,
     best_eval_population, best_eval_scores) = gpa.run(
        population, labels,
        max_beta=args.max_beta,
        ess_threshold=args.ess_threshold,
        max_steps=args.max_steps,
        min_delta_beta=args.min_delta_beta,
        bio_filter_fn=bio_filter_fn,
        mutation_batch_size=args.mutation_batch_size,
        diversity_subsample=args.diversity_subsample,
        extend_on_delta_beta=args.extend_on_delta_beta,
        extend_steps=args.extend_steps,
        hard_max_steps=args.hard_max_steps,
        start_beta=args.start_beta,
        elite_fraction=args.elite_fraction,
        nf_start=args.nf_start,
        nf_end=args.nf_end,
        gc_bin_edges=gc_bin_edges,
        gc_bin_min_quota=args.gc_bin_min_quota,
        gc_bin_alloc_mode=args.gc_bin_alloc_mode,
        gc_bin_alloc_alpha=args.gc_bin_alloc_alpha,
        max_delta_beta=args.max_delta_beta,
        eval_oracle_fn=eval_oracle_fn,
        eval_checkpoint_interval=args.eval_checkpoint_interval,
        archive_threshold=args.archive_threshold,
        topk_ag_size=args.topk_ag_size,
        eval_early_stop_patience=args.eval_early_stop_patience,
        nf_boost_delta=args.nf_boost_delta,
        nf_boost_max=args.nf_boost_max,
        eta_start=args.eta_start,
        eta_end=args.eta_end,
        dedup_threshold=args.dedup_threshold,
        dedup_nf_mult=args.dedup_nf_mult,
    )

    # ---- Save results ----
    out_h5 = output_dir / "gpa_output.h5"
    print(f"\n[Save] Writing {out_h5}")
    indices_np = population.numpy()
    onehot = np.eye(4, dtype=np.float32)[indices_np].transpose(0, 2, 1)
    gc_fracs = ((indices_np == 1) | (indices_np == 2)).mean(axis=1).astype(np.float32)

    final_beta = float(history.beta[-1]) if history.beta else 0.0
    with h5py.File(str(out_h5), "w") as f:
        f.create_dataset("arr_0", data=onehot, compression="gzip")
        f.create_dataset("oracle_preds", data=oracle_scores, compression="gzip")
        f.create_dataset("gc_fractions", data=gc_fracs, compression="gzip")
        f.create_dataset("log_weights", data=log_weights, compression="gzip")
        f.create_dataset("indices", data=indices_np, compression="gzip")
        f.attrs["final_beta"] = final_beta
        f.attrs["seq_length"] = SEQ_LENGTH

    # Best-seen population
    best_mean = float(best_oracle_scores.mean())
    final_mean = float(oracle_scores.mean())
    if best_mean > final_mean + 0.01:
        best_h5 = output_dir / "gpa_output_best.h5"
        print(f"[Save] Writing {best_h5} (best mean={best_mean:.3f} > final={final_mean:.3f})")
        best_indices_np = best_population.numpy()
        best_onehot = np.eye(4, dtype=np.float32)[best_indices_np].transpose(0, 2, 1)
        best_gc = ((best_indices_np == 1) | (best_indices_np == 2)).mean(axis=1).astype(np.float32)
        with h5py.File(str(best_h5), "w") as f:
            f.create_dataset("arr_0", data=best_onehot, compression="gzip")
            f.create_dataset("oracle_preds", data=best_oracle_scores, compression="gzip")
            f.create_dataset("gc_fractions", data=best_gc, compression="gzip")
            f.create_dataset("indices", data=best_indices_np, compression="gzip")

    # History JSON
    out_json = output_dir / "gpa_history.json"
    print(f"[Save] Writing {out_json}")
    history_dict = history.to_dict()
    history_dict["args"] = vars(args)
    history_dict["final_beta"] = final_beta
    history_dict["final_log_z"] = float(sum(history.log_z_increments))
    history_dict["final_oracle_mean"] = float(oracle_scores.mean())
    history_dict["final_oracle_max"] = float(oracle_scores.max())
    history_dict["best_oracle_mean"] = best_mean
    history_dict["best_oracle_max"] = float(best_oracle_scores.max())
    history_dict["n_particles"] = N
    history_dict["timestamp"] = datetime.now().isoformat()
    history_dict["seq_length"] = SEQ_LENGTH

    # Add eval checkpoint data to history
    if best_eval_population is not None and best_eval_scores is not None:
        history_dict["eval_checkpoint_means"] = history.eval_checkpoint_means
        history_dict["eval_checkpoint_steps"] = history.eval_checkpoint_steps
        history_dict["best_eval_ag_mean"] = float(best_eval_scores.mean())

    # Add Lagrangian trajectory if enabled
    if lagrangian is not None:
        history_dict["lagrangian_history"] = lagrangian.history
        print(f"[Lagrangian] Final: {lagrangian.get_summary()}")

    with open(str(out_json), "w") as f:
        json.dump(history_dict, f, indent=2, default=str)

    # Save Lagrangian history separately for easy analysis
    if lagrangian is not None:
        lag_json = output_dir / "lagrangian_history.json"
        with open(str(lag_json), "w") as f:
            json.dump(lagrangian.history, f, indent=2)
        print(f"[Save] Lagrangian trajectory: {lag_json}")

    # Save best-by-AG-eval population
    if best_eval_population is not None and best_eval_scores is not None:
        best_eval_h5 = output_dir / "gpa_output_best_eval.h5"
        best_eval_ag_mean = float(best_eval_scores.mean())
        print(f"[Save] Writing {best_eval_h5} (best AG eval mean={best_eval_ag_mean:.4f})")
        best_eval_indices = best_eval_population.numpy()
        best_eval_onehot = np.eye(4, dtype=np.float32)[best_eval_indices].transpose(0, 2, 1)
        best_eval_gc = ((best_eval_indices == 1) | (best_eval_indices == 2)).mean(axis=1).astype(np.float32)
        # Cross-score with LegNet
        legnet_scores_eval = None
        if args.oracle_type == "legnet":
            legnet_scores_eval, _ = oracle.score(
                torch.from_numpy(best_eval_indices).long().to(device))
            print(f"  LegNet on best-eval: mean={legnet_scores_eval.mean():.3f}, "
                  f"max={legnet_scores_eval.max():.3f}")
        with h5py.File(str(best_eval_h5), "w") as f:
            f.create_dataset("arr_0", data=best_eval_onehot, compression="gzip")
            f.create_dataset("ag_k562_scores", data=best_eval_scores, compression="gzip")
            f.create_dataset("gc_fractions", data=best_eval_gc, compression="gzip")
            f.create_dataset("indices", data=best_eval_indices, compression="gzip")
            if legnet_scores_eval is not None:
                f.create_dataset("oracle_preds", data=legnet_scores_eval, compression="gzip")

    # ---- Per-step archive (sequences above archive_threshold at any eval step) ----
    if history.archive_seqs:
        all_seqs   = np.concatenate(history.archive_seqs,   axis=0)  # (M, L)
        all_scores = np.concatenate(history.archive_scores,  axis=0)  # (M,)
        all_steps  = np.concatenate(history.archive_steps,   axis=0)  # (M,)
        # Deduplicate by exact sequence identity
        _, unique_idx = np.unique(all_seqs, axis=0, return_index=True)
        all_seqs   = all_seqs[unique_idx]
        all_scores = all_scores[unique_idx]
        all_steps  = all_steps[unique_idx]
        # Sort by score descending
        order = np.argsort(all_scores)[::-1]
        all_seqs, all_scores, all_steps = all_seqs[order], all_scores[order], all_steps[order]
        archive_h5 = output_dir / "gpa_output_archive.h5"
        with h5py.File(str(archive_h5), "w") as f:
            f.create_dataset("seqs",      data=all_seqs,   compression="gzip")
            f.create_dataset("ag_scores", data=all_scores, compression="gzip")
            f.create_dataset("step",      data=all_steps,  compression="gzip")
        print(f"\n[Archive] threshold={args.archive_threshold:.3f} | "
              f"{len(all_seqs)} unique seqs across {len(history.archive_steps)} steps | "
              f"mean={all_scores.mean():.4f} max={all_scores.max():.4f} "
              f"P99={np.percentile(all_scores, 99):.4f} -> {archive_h5}")

    # ---- Top-K-by-AG evicting pool (user's bounded best-AG-across-steps pool) ----
    if getattr(history, "topk_ag_seqs", None) is not None:
        tk_seqs = history.topk_ag_seqs            # (cap, L) int
        tk_scores = history.topk_ag_scores        # (cap,) float
        tk_onehot = np.eye(4, dtype=np.float32)[tk_seqs].transpose(0, 2, 1)
        tk_gc = ((tk_seqs == 1) | (tk_seqs == 2)).mean(axis=1).astype(np.float32)
        topk_h5 = output_dir / "gpa_output_topk_ag.h5"
        with h5py.File(str(topk_h5), "w") as f:
            f.create_dataset("arr_0", data=tk_onehot, compression="gzip")
            f.create_dataset("ag_k562_scores", data=tk_scores, compression="gzip")
            f.create_dataset("gc_fractions", data=tk_gc, compression="gzip")
            f.create_dataset("indices", data=tk_seqs, compression="gzip")
            f.create_dataset("step", data=history.topk_ag_steps, compression="gzip")
        print(f"[topk_ag] {len(tk_seqs)} seqs | "
              f"mean={tk_scores.mean():.4f} max={tk_scores.max():.4f} -> {topk_h5}")

    # ---- Cross-oracle scoring ----
    ag_mean, ag_max = None, None
    legnet_mean, legnet_max = None, None

    if args.oracle_type in ("alphagenome", "alphagenome_torch"):
        # oracle_scores ARE AG scores when using AG as fitness
        ag_mean = float(oracle_scores.mean())
        ag_max = float(oracle_scores.max())
        history_dict["ag_k562_mean"] = ag_mean
        history_dict["ag_k562_max"] = ag_max
        history_dict["ag_k562_median"] = float(np.median(oracle_scores))

        # Also save AG scores to H5
        with h5py.File(str(out_h5), "a") as f:
            if "ag_k562_scores" in f:
                del f["ag_k562_scores"]
            f.create_dataset("ag_k562_scores", data=oracle_scores, compression="gzip")

        # Cross-score with LegNet if available
        if legnet_oracle is not None:
            print(f"\n[Cross-Score] Scoring with LegNet for comparison...")
            legnet_scores, _ = legnet_oracle.score(
                torch.from_numpy(indices_np).long().to(device))
            legnet_mean = float(legnet_scores.mean())
            legnet_max = float(legnet_scores.max())
            print(f"  LegNet: mean={legnet_mean:.3f}, max={legnet_max:.3f}")

            corr = np.corrcoef(oracle_scores, legnet_scores)[0, 1]
            print(f"  AG-LegNet corr: {corr:.4f}")
            history_dict["legnet_mean"] = legnet_mean
            history_dict["legnet_max"] = legnet_max
            history_dict["ag_legnet_correlation"] = float(corr)

            # Save LegNet scores to H5
            with h5py.File(str(out_h5), "a") as f:
                if "legnet_scores" in f:
                    del f["legnet_scores"]
                f.create_dataset("legnet_scores", data=legnet_scores, compression="gzip")

    elif args.oracle_type == "legnet":
        legnet_mean = float(oracle_scores.mean())
        legnet_max = float(oracle_scores.max())

        # AlphaGenome eval scoring
        if args.eval_ag:
            print(f"\n[AG Eval] Scoring final population with AlphaGenome K562...")
            ag_scores_arr = None

            # Use running AG eval server if available (faster, already warm)
            if ag_eval_client is not None:
                try:
                    ag_scores_arr, _ = ag_eval_client.score(
                        torch.from_numpy(indices_np).long())
                    print(f"  (via eval server)")
                except Exception as e:
                    print(f"  AG eval server error: {e}, falling back to subprocess")

            # Fallback to subprocess
            if ag_scores_arr is None:
                ag_csv = output_dir / "ag_k562_scores.csv"
                ag_cmd = [
                    os.path.expandvars("${CONDA_ENV_ROOT}/alphagenome/bin/python"),
                    "scripts/alphagenome/score_sequences.py",
                    "--input", str(out_h5),
                    "--output", str(ag_csv),
                    "--cell_types", "k562",
                    "--append_h5",
                    "--batch_size", str(args.ag_batch_size),
                ]
                try:
                    result = subprocess.run(ag_cmd, capture_output=True, text=True, timeout=1800)
                    if result.returncode == 0:
                        ag_df = pd.read_csv(ag_csv)
                        ag_scores_arr = ag_df["k562_ag"].values
                    else:
                        print(f"  AG scoring FAILED (rc={result.returncode})")
                        print(f"  stderr: {result.stderr[:500]}")
                except subprocess.TimeoutExpired:
                    print("  AG scoring TIMEOUT (30 min)")
                except Exception as e:
                    print(f"  AG scoring ERROR: {e}")

            if ag_scores_arr is not None:
                ag_mean = float(ag_scores_arr.mean())
                ag_max = float(ag_scores_arr.max())
                print(f"  AG K562: mean={ag_mean:.4f}, max={ag_max:.4f}")

                history_dict["ag_k562_mean"] = ag_mean
                history_dict["ag_k562_max"] = ag_max
                history_dict["ag_k562_median"] = float(np.median(ag_scores_arr))

                corr = np.corrcoef(oracle_scores, ag_scores_arr)[0, 1]
                print(f"  LegNet-AG corr: {corr:.4f}")
                history_dict["legnet_ag_correlation"] = float(corr)

                # Save AG scores to main H5
                with h5py.File(str(out_h5), "a") as f:
                    if "ag_k562_scores" in f:
                        del f["ag_k562_scores"]
                    f.create_dataset("ag_k562_scores", data=ag_scores_arr, compression="gzip")

                # Save best-by-AG subset (top 500)
                ag_ranking = np.argsort(ag_scores_arr)[::-1].copy()
                top_k = min(500, len(ag_ranking))
                best_ag_idx = ag_ranking[:top_k]
                best_ag_h5 = output_dir / "gpa_output_best_ag.h5"
                best_ag_indices = indices_np[best_ag_idx]
                best_ag_onehot = np.eye(4, dtype=np.float32)[best_ag_indices].transpose(0, 2, 1)
                best_ag_gc = ((best_ag_indices == 1) | (best_ag_indices == 2)).mean(axis=1).astype(np.float32)
                with h5py.File(str(best_ag_h5), "w") as f:
                    f.create_dataset("arr_0", data=best_ag_onehot, compression="gzip")
                    f.create_dataset("oracle_preds", data=oracle_scores[best_ag_idx], compression="gzip")
                    f.create_dataset("ag_k562_scores", data=ag_scores_arr[best_ag_idx], compression="gzip")
                    f.create_dataset("gc_fractions", data=best_ag_gc, compression="gzip")
                    f.create_dataset("indices", data=best_ag_indices, compression="gzip")
                print(f"  Saved top-{top_k} by AG to {best_ag_h5}")
                print(f"  Top-{top_k} AG: mean={ag_scores_arr[best_ag_idx].mean():.4f}, "
                      f"LegNet mean={oracle_scores[best_ag_idx].mean():.3f}")

    # Re-save history with cross-scoring results
    with open(str(out_json), "w") as f:
        json.dump(history_dict, f, indent=2, default=str)

    # ---- Clean up AG servers ----
    if args.oracle_type == "alphagenome":
        print("\n[Cleanup] Shutting down AG oracle server...")
        stop_ag_server(ag_proc, ag_client)
    if ag_eval_proc is not None:
        print("\n[Cleanup] Shutting down AG eval server...")
        stop_ag_server(ag_eval_proc, ag_eval_client)

    # ---- Summary ----
    print(f"\n{'='*72}")
    print(f"GPA COMPLETE (K562 LentiMPRA MDLM + {oracle_label})")
    print(f"{'='*72}")
    print(f"  Steps:          {len(history.beta)}")
    print(f"  Final beta:     {final_beta:.4f}")
    if args.oracle_type == "alphagenome":
        print(f"  AG mean:        {ag_mean:.4f}")
        print(f"  AG max:         {ag_max:.4f}")
        print(f"  Best AG mean:   {best_mean:.4f}")
        if legnet_mean is not None:
            print(f"  LegNet mean:    {legnet_mean:.3f}")
            print(f"  LegNet max:     {legnet_max:.3f}")
    else:
        print(f"  LegNet mean:    {legnet_mean:.3f}")
        print(f"  LegNet max:     {legnet_max:.3f}")
        print(f"  Best LegNet:    {best_mean:.3f}")
        if ag_mean is not None:
            print(f"  AG K562 mean:   {ag_mean:.4f}")
            print(f"  AG K562 max:    {ag_max:.4f}")
    if best_eval_scores is not None:
        print(f"  Best AG eval:   {float(best_eval_scores.mean()):.4f}")
    print(f"  GC:             {gc_fracs.mean()*100:.1f}%")
    print(f"  Unique seqs:    {history.n_unique[-1]:,}")
    print(f"  Output:         {out_h5}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
