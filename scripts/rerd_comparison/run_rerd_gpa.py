#!/usr/bin/env python3
"""
GPA + DPS with SVDD's MDLM and Enformer oracle — RERD comparison.

Uses the same DiffusionPopulationAnnealer as our main pipeline, but swaps
in SVDD's pre-trained unconditional MDLM (masked diffusion, CNN, 200bp)
and Enformer oracle (3 cell types: hepg2/k562/sknsh).

No CFG is used — the MDLM is unconditional. Guidance comes from:
  - GPA: reweight particles by oracle fitness
  - DPS: oracle gradient during mutation (optional)

Usage:
    python scripts/rerd_comparison/run_rerd_gpa.py \
        --mdlm_checkpoint ~/SVDD/artifacts/DNA_Diffusion:v0/last.ckpt \
        --oracle_checkpoint ~/SVDD/artifacts/DNA_evaluation:v0/model.ckpt \
        --seed_pool results/rerd_comparison/gosai_seeds_hepg2.h5 \
        --target_cell hepg2 \
        --use_dps --dps_eta 3000 \
        --population_size 5000 --max_beta 50.0 \
        --output_dir results/rerd_comparison/run_gpa_dps_hepg2
"""

import os
import sys
import argparse
import json
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.gpa_sampling import DiffusionPopulationAnnealer
from scripts.rerd_comparison.mdlm_wrapper import (
    load_mdlm, MDLMMutator, MDLMMutatorDPS,
    MDLMMutatorConfGated, MDLMMutatorKLConstrained,
)
from scripts.rerd_comparison.enformer_oracle import load_enformer_oracle, EnformerOracle


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPA + DPS with SVDD MDLM + Enformer (RERD comparison)")

    # Model paths
    parser.add_argument("--mdlm_checkpoint", required=True,
                        help="MDLM diffusion model checkpoint (last.ckpt)")
    parser.add_argument("--oracle_checkpoint", required=True,
                        help="Enformer oracle checkpoint for guidance (model.ckpt)")
    parser.add_argument("--eval_oracle_checkpoint", default=None,
                        help="Separate Enformer oracle for evaluation only (split-oracle protocol)")
    parser.add_argument("--svdd_dir", default=None,
                        help="Path to SVDD repo (default: ~/SVDD)")
    parser.add_argument("--sgdd_dir", default=None,
                        help="Path to SGDD repo (default: ~/SGDD)")

    # Cell-type specificity
    parser.add_argument("--target_cell", default="hepg2",
                        choices=["hepg2", "k562", "sknsh"],
                        help="Cell type to maximize")
    parser.add_argument("--penalty_weight", type=float, default=0.5,
                        help="Weight for off-target cell type penalty")
    parser.add_argument("--fitness_mode", default="linear",
                        choices=["linear", "ratio", "log_ratio", "indicator"],
                        help="Fitness mode: linear (default), ratio, log_ratio, or indicator")
    parser.add_argument("--specificity_threshold", type=float, default=None,
                        help="Off-target threshold for indicator mode (required if fitness_mode=indicator)")

    # Population initialization
    parser.add_argument("--seed_pool", default=None,
                        help="H5 seed pool (indices dataset, 200bp)")
    parser.add_argument("--from_random", action="store_true",
                        help="Initialize from random DNA sequences")
    parser.add_argument("--gc_balanced_init", action="store_true",
                        help="When combined with --from_random and --gc_bin_edges, "
                             "generate equal particles per GC bin via rejection sampling")
    parser.add_argument("--population_size", type=int, default=5000)
    parser.add_argument("--top_k_init", action="store_true",
                        help="Use top-K seeds by oracle score")

    # GPA parameters
    parser.add_argument("--max_beta", type=float, default=50.0)
    parser.add_argument("--ess_threshold", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--min_delta_beta", type=float, default=1e-4)

    # Adaptive continuation
    parser.add_argument("--extend_on_delta_beta", type=float, default=0.5)
    parser.add_argument("--extend_steps", type=int, default=10)
    parser.add_argument("--hard_max_steps", type=int, default=200)
    parser.add_argument("--start_beta", type=float, default=0.0,
                        help="Starting beta for SMC continuation from a previous run")
    parser.add_argument("--mingap_archive_size", type=int, default=0,
                        help="DNA-CRAFT G*-style spec-bounded archive capacity. "
                             "0 = disabled. When >0, maintains a streaming top-N "
                             "archive by per-seq specificity (target − max off-target) "
                             "across eval checkpoints; saved to "
                             "gpa_output_mingap_filtered.h5. Requires "
                             "--eval_checkpoint_interval > 0 and split-oracle eval ckpt.")
    parser.add_argument("--perstep_archive_top_k", type=int, default=0,
                        help="Per-step top-K accumulated archive (design B): at "
                             "every eval checkpoint, take top-K of that step's "
                             "population by specificity, accumulate across all "
                             "checkpoints with no eviction. Final pool size ≈ "
                             "T_ckpt × K. Saved to gpa_output_perstep_filtered.h5. "
                             "0 = disabled. Same prerequisites as mingap_archive_size.")
    parser.add_argument("--archive_threshold", type=float, default=None,
                        help="Eval-oracle threshold; sequences exceeding this are "
                             "saved per step to gpa_output_filtered.h5 (None=disabled)")
    parser.add_argument("--eval_checkpoint_interval", type=int, default=0,
                        help="Score with eval oracle every N GPA steps (0=disabled)")
    parser.add_argument("--max_delta_beta", type=float, default=0.0,
                        help="Cap delta-beta per step (0=disabled). "
                             "Use for single-seed runs to prevent beta jumping in 1 step.")

    # Mutation parameters
    parser.add_argument("--noise_fraction", type=float, default=0.10,
                        help="Fraction of positions to mask per mutation")
    parser.add_argument("--mutation_batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of denoising steps per mutation (default: 1)")

    # DPS parameters
    parser.add_argument("--use_dps", action="store_true",
                        help="Use DPS oracle-guided mutation")
    parser.add_argument("--dps_eta", type=float, default=3000.0,
                        help="DPS gradient step size")
    parser.add_argument("--dps_tau", type=float, default=1.0,
                        help="Gumbel-softmax temperature for DPS")
    parser.add_argument("--dps_reward_mode", default="fitness",
                        choices=["fitness", "smooth_spec", "log_ratio_spec", "neg_offtarget", "sum_individual"],
                        help="DPS gradient objective: fitness (same as GPA), "
                             "smooth_spec (r_target - pw*mean(penalty)), "
                             "log_ratio_spec (log-space version), "
                             "neg_offtarget (-max(off-target), pure OT minimization), "
                             "sum_individual (r_target - pw_k*K - pw_s*S, per-cell)")
    parser.add_argument("--dps_penalty_weight", type=float, default=0.3,
                        help="Weight for off-target penalty in DPS reward")
    parser.add_argument("--dps_penalty_weight_k", type=float, default=None,
                        help="Per-cell DPS penalty weight for K562 (None=use dps_penalty_weight)")
    parser.add_argument("--dps_penalty_weight_s", type=float, default=None,
                        help="Per-cell DPS penalty weight for SKNSH (None=use dps_penalty_weight)")
    parser.add_argument("--penalty_weight_k", type=float, default=None,
                        help="Per-cell GPA penalty weight for K562 (None=use penalty_weight)")
    parser.add_argument("--penalty_weight_s", type=float, default=None,
                        help="Per-cell GPA penalty weight for SKNSH (None=use penalty_weight)")

    # DPS gating
    parser.add_argument("--guide_start_frac", type=float, default=0.0,
                        help="Fraction of denoising steps before DPS activates "
                             "(0.0=always, 0.95=last step only with steps=20)")

    # Grammar-preserving DPS modes
    parser.add_argument("--dps_mode", default="standard",
                        choices=["standard", "conf_gated", "kl_constrained"],
                        help="DPS variant: standard, conf_gated (entropy gating), "
                             "kl_constrained (KL-budget projection)")
    parser.add_argument("--gate_sharpness", type=float, default=1.0,
                        help="Exponent for entropy gate in conf_gated mode")
    parser.add_argument("--kl_budget", type=float, default=1.0,
                        help="Per-position KL budget (nats) for kl_constrained mode")
    parser.add_argument("--kl_mode", default="per_position",
                        choices=["per_position", "global"],
                        help="KL constraint scope")

    # Elitism
    parser.add_argument("--elite_fraction", type=float, default=0.0,
                        help="Fraction of top particles preserved each step (0=disabled)")

    # GC fitness penalty
    parser.add_argument("--gc_fitness_weight", type=float, default=0.0,
                        help="GC penalty in fitness (0=disabled)")
    parser.add_argument("--gc_fitness_low", type=float, default=0.40)
    parser.add_argument("--gc_fitness_high", type=float, default=0.60)
    parser.add_argument("--gc_center_weight", type=float, default=0.0,
                        help="Penalty for drift from band center (0=disabled). "
                             "Penalizes |gc - (gc_fitness_low+gc_fitness_high)/2|")

    # Bio filter
    parser.add_argument("--bio_filter", action="store_true",
                        help="Reject non-bio-plausible mutations")
    parser.add_argument("--gc_low", type=float, default=0.40)
    parser.add_argument("--gc_high", type=float, default=0.70)

    # NF annealing (Approach D: nf schedule)
    parser.add_argument("--nf_start", type=float, default=0.0,
                        help="Starting noise fraction for annealing (0=disabled)")
    parser.add_argument("--nf_end", type=float, default=0.0,
                        help="Ending noise fraction for annealing")

    # GC-aware DPS (Approach A: gradient-coupled)
    parser.add_argument("--gc_dps_weight", type=float, default=0.0,
                        help="Weight for GC centering loss in DPS gradient (0=disabled)")
    parser.add_argument("--gc_dps_target", type=float, default=0.50,
                        help="Target GC fraction for GC-aware DPS")

    # Direct GC pull (v13: decoupled from oracle gradient normalization)
    parser.add_argument("--gc_pull_weight", type=float, default=0.0,
                        help="Direct GC pull bias weight (0=disabled). Applied as "
                             "additive log-prob bias, decoupled from oracle gradient.")
    parser.add_argument("--gc_pull_target", type=float, default=0.50,
                        help="Target GC fraction for direct GC pull")

    # GC-stratified resampling (v16: oracle-proportional island allocation)
    parser.add_argument("--gc_bin_edges", default=None,
                        help="GC bin edges for stratified resampling, comma-separated "
                             "(e.g., '0.40,0.42,...,0.60'). None = global resampling.")
    parser.add_argument("--gc_bin_min_quota", type=int, default=50,
                        help="Minimum particles per GC bin (prevents bin extinction)")
    parser.add_argument("--gc_bin_alloc_mode", default="raw",
                        choices=["raw", "minshift", "exp"],
                        help="GC bin allocation formula: raw (v16b), minshift (v16), exp (sharpened)")
    parser.add_argument("--gc_bin_alloc_alpha", type=float, default=5.0,
                        help="Sharpening exponent for exp allocation mode")

    # EDTS (Edit-aware Diffusion Tree Search)
    parser.add_argument("--branch_factor", type=int, default=1,
                        help="K-branch factor per particle per step (1=standard, 3=tree search)")
    parser.add_argument("--max_edit_frac", type=float, default=1.0,
                        help="Max fraction of positions differing from original seed "
                             "(1.0=no cap, 0.10=20 edits on 200bp)")
    parser.add_argument("--edit_select_mode", default="efficiency",
                        choices=["efficiency", "oracle"],
                        help="Branch selection: efficiency=oracle/(1+hamming), oracle=highest score")

    # MCTS (Monte Carlo Tree Search mutation)
    parser.add_argument("--mcts_depth", type=int, default=0,
                        help="MCTS max tree depth (0=disabled/use EDTS, 2-3=multi-depth MCTS)")
    parser.add_argument("--mcts_iterations", type=int, default=12,
                        help="MCTS expansion budget per particle per GPA step")
    parser.add_argument("--mcts_c", type=float, default=1.41,
                        help="UCB1 exploration constant (sqrt(2)=1.41 standard)")
    parser.add_argument("--mcts_select_mode", choices=["ucb", "uniform"], default="ucb",
                        help="MCTS leaf selection: 'ucb' (heuristic, DNA-CRAFT-style) "
                             "or 'uniform' (BFS, target-preserved via Prop. K with K_eff=K^D)")

    # Per-step edit budget
    parser.add_argument("--max_edits_per_step", type=int, default=0,
                        help="Max new edits per GPA step (0=disabled). "
                             "Limits hamming(mutated, input) per step.")

    # Hill-climb (per-position greedy optimization after K-branch selection)
    parser.add_argument("--hill_climb", action="store_true",
                        help="After K-branch EDTS picks best branch, hill-climb "
                             "per position using tokens from other branches")
    parser.add_argument("--hill_climb_budget", type=int, default=0,
                        help="Independent HC edit budget per GPA step (0=disabled). "
                             "HC edits do NOT count toward max_edits_per_step.")
    parser.add_argument("--hill_climb_full_vocab", action="store_true",
                        help="Try all 3 alternative DNA tokens at all positions "
                             "(exhaustive single-position search, not just branch tokens)")
    parser.add_argument("--hill_climb_positions", type=int, default=0,
                        help="Number of positions to subsample per HC round "
                             "(0=all 200, 50-100 recommended for speed)")

    # Edit-efficiency
    parser.add_argument("--edit_discount", action="store_true",
                        help="Apply edit-distance discount to fitness for GPA reweighting")
    parser.add_argument("--edit_discount_mode", default="linear",
                        choices=["linear", "log"],
                        help="Edit discount mode: linear (fitness*(1-edits/L)) or "
                             "log (fitness - lambda*log(1+edits))")
    parser.add_argument("--edit_discount_lambda", type=float, default=1.0,
                        help="Lambda for log edit-discount mode")
    parser.add_argument("--pareto_edit_fitness", action="store_true",
                        help="Use 2D Pareto rank (score vs edits) for GPA fitness")

    # K-mer composition regularizer for HC
    parser.add_argument("--hc_kmer_weight", type=float, default=0.0,
                        help="K-mer composition penalty weight for hill-climb "
                             "(0=disabled, 1-10 typical range)")
    parser.add_argument("--gosai_csv", default=None,
                        help="Path to gosai_all.csv for k-mer reference profile "
                             "(required when hc_kmer_weight > 0)")
    parser.add_argument("--kmer_target_cell", default=None,
                        help="Cell type for k-mer ref profile (default: same as target_cell)")
    parser.add_argument("--kmer_top_frac", type=float, default=0.10,
                        help="Fraction of top-activity Gosai seqs for k-mer ref (default 0.10)")

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
    L = 200  # MDLM sequence length

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
            # Generate equal particles per GC bin via rejection sampling
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
                batch = max(target * 4, 1000)  # oversample for efficiency
                attempts = 0
                while len(collected) < target and attempts < 100:
                    seqs = torch.randint(0, 4, (batch, L))
                    gc = ((seqs == 1) | (seqs == 2)).float().mean(dim=1)
                    mask = (gc >= lo) & (gc < hi)
                    collected.append(seqs[mask])
                    attempts += 1
                bin_seqs = torch.cat(collected, dim=0)[:target]
                if len(bin_seqs) < target:
                    # Pad with duplicates if rejection sampling is slow
                    pad = target - len(bin_seqs)
                    bin_seqs = torch.cat([bin_seqs, bin_seqs[:pad]], dim=0)
                all_seqs.append(bin_seqs)
                gc_actual = ((bin_seqs == 1) | (bin_seqs == 2)).float().mean(dim=1)
                print(f"  Bin [{lo:.2f}, {hi:.2f}): {len(bin_seqs)} seqs, "
                      f"GC={gc_actual.mean():.3f}")
            population = torch.cat(all_seqs, dim=0)
            # Shuffle so bins aren't contiguous
            perm = torch.randperm(population.shape[0])
            population = population[perm]
            return population
        else:
            print(f"\n[Init] Random DNA sequences: {N:,} x {L}bp")
            population = torch.randint(0, 4, (N, L))
            return population

    else:
        raise ValueError("Must specify --seed_pool or --from_random")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"  Random seed:      {args.seed}")

    # Repo directories
    svdd_dir = args.svdd_dir or str(Path.home() / "SVDD")
    sgdd_dir = args.sgdd_dir or str(Path.home() / "SGDD")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GPA + DPS (RERD COMPARISON)")
    print("=" * 72)
    print(f"  Device:           {device}")
    print(f"  Backbone:         MDLM")
    print(f"  Target cell:      {args.target_cell}")
    print(f"  Penalty weight:   {args.penalty_weight}")
    print(f"  Population:       {args.population_size:,}")
    print(f"  Max beta:         {args.max_beta}")
    print(f"  Noise fraction:   {args.noise_fraction}")
    print(f"  Fitness mode:     {args.fitness_mode}")
    denoise_steps = args.steps if args.steps is not None else 1
    print(f"  Denoise steps:    {denoise_steps}")
    print(f"  DPS:              {args.use_dps} (eta={args.dps_eta})")
    print(f"  Guide start frac: {args.guide_start_frac}")
    print(f"  DPS reward mode:  {args.dps_reward_mode} (pw={args.dps_penalty_weight})")
    if args.dps_penalty_weight_k is not None or args.dps_penalty_weight_s is not None:
        print(f"  DPS pw per-cell:  K={args.dps_penalty_weight_k}, S={args.dps_penalty_weight_s}")
    if args.penalty_weight_k is not None or args.penalty_weight_s is not None:
        print(f"  GPA pw per-cell:  K={args.penalty_weight_k}, S={args.penalty_weight_s}")
    print(f"  Elite fraction:   {args.elite_fraction}")
    print(f"  GC center weight: {args.gc_center_weight}")
    if args.nf_start > 0:
        print(f"  NF anneal:        {args.nf_start} → {args.nf_end}")
    if args.gc_dps_weight > 0:
        print(f"  GC DPS weight:    {args.gc_dps_weight} (target={args.gc_dps_target})")
    if args.gc_pull_weight > 0:
        print(f"  GC pull weight:   {args.gc_pull_weight} (target={args.gc_pull_target})")
    if args.branch_factor > 1 or args.max_edit_frac < 1.0:
        print(f"  EDTS:             K={args.branch_factor}, max_edit={args.max_edit_frac}, "
              f"select={args.edit_select_mode}")
    if args.mcts_depth >= 2:
        print(f"  MCTS:             depth={args.mcts_depth}, iter={args.mcts_iterations}, C={args.mcts_c}")
    if args.max_edits_per_step > 0:
        print(f"  Per-step cap:     {args.max_edits_per_step} edits/step")
    if args.hill_climb:
        print(f"  Hill-climb:       enabled")
    if args.hill_climb_budget > 0:
        mode = "full-vocab" if args.hill_climb_full_vocab else "branch-token"
        print(f"  HC budget:        {args.hill_climb_budget} edits/step ({mode})")
        if args.hill_climb_positions > 0:
            print(f"  HC positions:     {args.hill_climb_positions}/200")
    if args.edit_discount:
        print(f"  Edit discount:    {args.edit_discount_mode} (lambda={args.edit_discount_lambda})")
    if args.pareto_edit_fitness:
        print(f"  Pareto fitness:   enabled (score vs edits)")
    print(f"  Bio filter:       {args.bio_filter}")
    if args.start_beta > 0:
        print(f"  Start beta:       {args.start_beta}")
    if args.eval_checkpoint_interval > 0:
        print(f"  Eval checkpoint:  every {args.eval_checkpoint_interval} steps")
    print(f"  Output:           {args.output_dir}")
    print("=" * 72)

    # ---- Load MDLM diffusion backbone ----
    print("\n[Model] Loading MDLM diffusion model...")
    mdlm = load_mdlm(args.mdlm_checkpoint, svdd_dir, device)
    print(f"  Loaded: {args.mdlm_checkpoint}")

    # ---- Load Enformer oracle(s) ----
    print("\n[Oracle] Loading Enformer oracle (guidance)...")
    lit_oracle = load_enformer_oracle(args.oracle_checkpoint, device)
    oracle = EnformerOracle(
        lit_oracle,
        target_cell=args.target_cell,
        penalty_weight=args.penalty_weight,
        fitness_mode=args.fitness_mode,
        specificity_threshold=args.specificity_threshold,
        dps_reward_mode=args.dps_reward_mode,
        dps_penalty_weight=args.dps_penalty_weight,
        penalty_weight_k=args.penalty_weight_k,
        penalty_weight_s=args.penalty_weight_s,
        dps_penalty_weight_k=args.dps_penalty_weight_k,
        dps_penalty_weight_s=args.dps_penalty_weight_s,
        device=device,
    )
    print(f"  Target: {args.target_cell}, penalty cells: "
          f"{[c for c in ['hepg2','k562','sknsh'] if c != args.target_cell]}")
    print(f"  Fitness mode:     {args.fitness_mode}")
    if args.specificity_threshold is not None:
        print(f"  Specificity thr:  {args.specificity_threshold}")

    eval_oracle = None
    if args.eval_oracle_checkpoint:
        print("\n[Oracle] Loading Enformer oracle (evaluation)...")
        eval_lit_oracle = load_enformer_oracle(args.eval_oracle_checkpoint, device)
        eval_oracle = EnformerOracle(
            eval_lit_oracle,
            target_cell=args.target_cell,
            penalty_weight=args.penalty_weight,
            fitness_mode=args.fitness_mode,
            specificity_threshold=args.specificity_threshold,
            device=device,
        )
        print(f"  Loaded: {args.eval_oracle_checkpoint}")

    # ---- Build mutation function ----
    def _build_mutator(nf, use_dps):
        """Build MDLM mutator for given noise_fraction and DPS setting."""
        if use_dps:
            base_kwargs = dict(
                noise_fraction=nf, eta=args.dps_eta, tau_start=args.dps_tau,
                gc_dps_weight=args.gc_dps_weight, gc_dps_target=args.gc_dps_target,
                gc_pull_weight=args.gc_pull_weight, gc_pull_target=args.gc_pull_target,
            )
            if args.dps_mode == "conf_gated":
                return MDLMMutatorConfGated(
                    mdlm, oracle, gate_sharpness=args.gate_sharpness, **base_kwargs)
            elif args.dps_mode == "kl_constrained":
                return MDLMMutatorKLConstrained(
                    mdlm, oracle, kl_budget=args.kl_budget, kl_mode=args.kl_mode, **base_kwargs)
            else:
                return MDLMMutatorDPS(mdlm, oracle, **base_kwargs)
        else:
            return MDLMMutator(mdlm, noise_fraction=nf)

    mutate_fn = _build_mutator(args.noise_fraction, args.use_dps)
    backbone_name = "MDLM"
    if args.use_dps:
        print(f"\n[Mutation] {backbone_name} + DPS (eta={args.dps_eta}, nf={args.noise_fraction})")
        if args.dps_mode != "standard":
            print(f"  DPS mode: {args.dps_mode}")
            if args.dps_mode == "conf_gated":
                print(f"  Gate sharpness: {args.gate_sharpness}")
            elif args.dps_mode == "kl_constrained":
                print(f"  KL budget: {args.kl_budget} ({args.kl_mode})")
    else:
        print(f"\n[Mutation] {backbone_name} blind (nf={args.noise_fraction})")

    # ---- Build mutate_fn_factory for nf annealing ----
    mutate_fn_factory = None
    if args.nf_start > 0:
        def mutate_fn_factory(nf, guidance_weight=None):
            return _build_mutator(nf, args.use_dps)
        print(f"  [NF Factory] Will anneal nf: {args.nf_start} → {args.nf_end}")

    # ---- Oracle scoring function (for GPA fitness) ----
    def oracle_fn(sequences_tensor):
        if not isinstance(sequences_tensor, torch.Tensor):
            sequences_tensor = torch.from_numpy(sequences_tensor).long()
        return oracle.score(sequences_tensor)

    # ---- Fast oracle for HC (target-only, no penalty extraction) ----
    def oracle_fn_fast(sequences_tensor):
        if not isinstance(sequences_tensor, torch.Tensor):
            sequences_tensor = torch.from_numpy(sequences_tensor).long()
        return oracle.score_target_only(sequences_tensor)

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

    # ---- Fitness function ----
    # oracle_fn now returns raw target-cell scores. fitness_fn applies:
    #   1. Cell-type specificity (indicator, penalty_weight, etc.) via oracle.compute_fitness()
    #   2. Optional GC fitness penalty
    needs_specificity_fitness = (args.fitness_mode != "linear" or args.penalty_weight > 0)
    gc_fitness = args.gc_fitness_weight > 0
    gc_center = args.gc_center_weight > 0

    fitness_fn = None
    if needs_specificity_fitness or gc_fitness or gc_center:
        gc_lo, gc_hi = args.gc_fitness_low, args.gc_fitness_high
        gc_fw = args.gc_fitness_weight
        gc_cw = args.gc_center_weight
        gc_band_center = (gc_lo + gc_hi) / 2
        def fitness_fn(oracle_scores, gc_fractions):
            # Apply cell-type specificity (indicator, pw, etc.)
            fit = oracle.compute_fitness(oracle_scores)
            # Apply GC drift penalty (outside bounds)
            if gc_fw > 0:
                drift = np.maximum(0, gc_fractions - gc_hi) + np.maximum(0, gc_lo - gc_fractions)
                fit = fit - gc_fw * drift
            # Apply GC centering penalty (drift from band center)
            if gc_cw > 0:
                drift_from_center = np.abs(gc_fractions - gc_band_center)
                fit = fit - gc_cw * drift_from_center
            return fit

    # ---- Initialize population ----
    population = initialize_population(args, oracle, device)
    N = population.shape[0]
    # Labels are unused (unconditional model) but GPA interface requires them
    labels = torch.zeros(N, 1, device=device)

    # ---- Parse GC bin edges for stratified resampling ----
    gc_bin_edges = None
    if args.gc_bin_edges:
        gc_bin_edges = np.array([float(x) for x in args.gc_bin_edges.split(",")])
        print(f"\n[GC Stratified] {len(gc_bin_edges)-1} bins, min_quota={args.gc_bin_min_quota}")

    # ---- Build eval oracle callback for checkpointing ----
    eval_oracle_fn = None
    eval_oracle_all_cells_fn = None
    need_spec_archives = args.mingap_archive_size > 0 or args.perstep_archive_top_k > 0
    if eval_oracle is not None and args.eval_checkpoint_interval > 0:
        def eval_oracle_fn(pop):
            scores, _ = eval_oracle.score(pop.to(device))
            return scores
        if need_spec_archives:
            def eval_oracle_all_cells_fn(pop):
                return eval_oracle.score_all_cells(pop.to(device))
    elif need_spec_archives:
        raise ValueError(
            "--mingap_archive_size or --perstep_archive_top_k > 0 requires a "
            "split-oracle eval checkpoint (--eval_oracle_checkpoint and "
            "--eval_checkpoint_interval > 0)")

    # ---- K-mer reference profile for HC composition penalty ----
    ref_kmer_profile = None
    if args.hc_kmer_weight > 0:
        if args.gosai_csv is None:
            gosai_csv = os.environ.get('GOSAI_CSV')
            if gosai_csv is None:
                raise ValueError("--gosai_csv or GOSAI_CSV env required when hc_kmer_weight > 0")
        else:
            gosai_csv = args.gosai_csv
        from scripts.gpa_sampling import _load_gosai_ref_kmer_profile
        kmer_cell = args.kmer_target_cell or args.target_cell
        ref_kmer_profile = _load_gosai_ref_kmer_profile(
            gosai_csv, target_cell=kmer_cell, top_frac=args.kmer_top_frac, k=3)

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
        mcts_select_mode=args.mcts_select_mode,
        max_edits_per_step=args.max_edits_per_step,
        hill_climb=args.hill_climb,
        hill_climb_budget=args.hill_climb_budget,
        hill_climb_full_vocab=args.hill_climb_full_vocab,
        hill_climb_positions=args.hill_climb_positions,
        oracle_fn_fast=oracle_fn_fast,
        edit_discount=args.edit_discount,
        edit_discount_mode=args.edit_discount_mode,
        edit_discount_lambda=args.edit_discount_lambda,
        pareto_edit_fitness=args.pareto_edit_fitness,
        hc_kmer_weight=args.hc_kmer_weight,
        ref_kmer_profile=ref_kmer_profile,
    )
    (population, oracle_scores, log_weights, history,
     best_population, best_oracle_scores,
     pool_population, best_eval_scores) = gpa.run(
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
        eval_oracle_fn=eval_oracle_fn,
        eval_checkpoint_interval=args.eval_checkpoint_interval,
        archive_threshold=args.archive_threshold,
        mingap_archive_size=args.mingap_archive_size,
        perstep_archive_top_k=args.perstep_archive_top_k,
        eval_oracle_all_cells_fn=eval_oracle_all_cells_fn,
        mingap_target_cell=args.target_cell,
        max_delta_beta=args.max_delta_beta,
    )

    # ---- Score all cell types for final population ----
    print("\n[Eval] Scoring final population with guidance oracle...")
    all_cell_scores_ft = oracle.score_all_cells(population.to(device))
    for cell, scores in all_cell_scores_ft.items():
        print(f"  {cell} (ft): mean={scores.mean():.3f}, max={scores.max():.3f}")

    if eval_oracle is not None:
        print("\n[Eval] Scoring final population with eval oracle...")
        all_cell_scores_eval = eval_oracle.score_all_cells(population.to(device))
        for cell, scores in all_cell_scores_eval.items():
            print(f"  {cell} (eval): mean={scores.mean():.3f}, max={scores.max():.3f}")
        # Use eval oracle as the "real" scores
        all_cell_scores = all_cell_scores_eval
    else:
        all_cell_scores = all_cell_scores_ft

    # ---- Score best-seen population (if better than final) ----
    best_mean_ft = float(best_oracle_scores.mean())
    final_mean_ft = float(oracle_scores.mean())
    best_cell_scores_ft = None
    best_cell_scores_eval = None
    if best_mean_ft > final_mean_ft + 0.01:
        print(f"\n[Eval] Best-seen population (ft mean={best_mean_ft:.3f} > final={final_mean_ft:.3f})")
        print("  Scoring best population with guidance oracle...")
        best_cell_scores_ft = oracle.score_all_cells(best_population.to(device))
        for cell, scores in best_cell_scores_ft.items():
            print(f"  {cell} (ft): mean={scores.mean():.3f}, max={scores.max():.3f}")
        if eval_oracle is not None:
            print("  Scoring best population with eval oracle...")
            best_cell_scores_eval = eval_oracle.score_all_cells(best_population.to(device))
            for cell, scores in best_cell_scores_eval.items():
                print(f"  {cell} (eval): mean={scores.mean():.3f}, max={scores.max():.3f}")

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
        # Eval oracle scores (or ft if no eval oracle)
        for cell, scores in all_cell_scores.items():
            f.create_dataset(f"oracle_{cell}", data=scores, compression="gzip")
        # FT oracle scores (always saved for reference)
        for cell, scores in all_cell_scores_ft.items():
            f.create_dataset(f"oracle_{cell}_ft", data=scores, compression="gzip")
        # Eval oracle scores explicitly (if split-oracle)
        if eval_oracle is not None:
            for cell, scores in all_cell_scores_eval.items():
                f.create_dataset(f"oracle_{cell}_eval", data=scores, compression="gzip")
        f.attrs["final_beta"] = final_beta
        f.attrs["target_cell"] = args.target_cell
        f.attrs["split_oracle"] = eval_oracle is not None

    # Best-seen population (saved separately when better than final)
    if best_mean_ft > final_mean_ft + 0.01:
        best_h5 = output_dir / "gpa_output_best.h5"
        print(f"[Save] Writing {best_h5} (best ft_mean={best_mean_ft:.3f} > final={final_mean_ft:.3f})")
        best_indices_np = best_population.numpy()
        best_onehot = np.eye(4, dtype=np.float32)[best_indices_np].transpose(0, 2, 1)
        best_gc = ((best_indices_np == 1) | (best_indices_np == 2)).mean(axis=1).astype(np.float32)
        with h5py.File(str(best_h5), "w") as f:
            f.create_dataset("arr_0", data=best_onehot, compression="gzip")
            f.create_dataset("oracle_preds", data=best_oracle_scores, compression="gzip")
            f.create_dataset("gc_fractions", data=best_gc, compression="gzip")
            f.create_dataset("indices", data=best_indices_np, compression="gzip")
            # Per-cell scores from both oracles
            if best_cell_scores_ft is not None:
                for cell, scores in best_cell_scores_ft.items():
                    f.create_dataset(f"oracle_{cell}_ft", data=scores, compression="gzip")
            if best_cell_scores_eval is not None:
                for cell, scores in best_cell_scores_eval.items():
                    f.create_dataset(f"oracle_{cell}_eval", data=scores, compression="gzip")

    # Best eval-oracle population (from eval checkpointing)
    if pool_population is not None:
        best_eval_h5 = output_dir / "gpa_output_pool.h5"
        print(f"[Save] Writing {best_eval_h5} (best eval checkpoint)")
        best_eval_indices_np = pool_population.numpy()
        best_eval_onehot = np.eye(4, dtype=np.float32)[best_eval_indices_np].transpose(0, 2, 1)
        best_eval_gc = ((best_eval_indices_np == 1) | (best_eval_indices_np == 2)).mean(axis=1).astype(np.float32)
        # Score best-eval population with both oracles
        print("  Scoring best-eval population...")
        best_eval_cell_scores_ft = oracle.score_all_cells(pool_population.to(device))
        best_eval_cell_scores_eval = eval_oracle.score_all_cells(pool_population.to(device))
        for cell, scores in best_eval_cell_scores_eval.items():
            print(f"  {cell} (eval): mean={scores.mean():.3f}, max={scores.max():.3f}")
        with h5py.File(str(best_eval_h5), "w") as f:
            f.create_dataset("arr_0", data=best_eval_onehot, compression="gzip")
            f.create_dataset("oracle_preds", data=best_eval_scores, compression="gzip")
            f.create_dataset("gc_fractions", data=best_eval_gc, compression="gzip")
            f.create_dataset("indices", data=best_eval_indices_np, compression="gzip")
            for cell, scores in best_eval_cell_scores_ft.items():
                f.create_dataset(f"oracle_{cell}_ft", data=scores, compression="gzip")
            for cell, scores in best_eval_cell_scores_eval.items():
                f.create_dataset(f"oracle_{cell}_eval", data=scores, compression="gzip")

    # Per-step top-K accumulated archive (design B): save first so the
    # mingap-archive (design A) printout still reads as the headline.
    if (args.perstep_archive_top_k > 0
            and history.perstep_archive_seqs is not None
            and len(history.perstep_archive_seqs) > 0):
        perstep_h5 = output_dir / "gpa_output_perstep_filtered.h5"
        b_seqs = history.perstep_archive_seqs
        b_specs = history.perstep_archive_specs
        b_cells = history.perstep_archive_cell_scores
        b_steps = history.perstep_archive_steps
        b_onehot = np.eye(4, dtype=np.float32)[b_seqs].transpose(0, 2, 1)
        b_gc = ((b_seqs == 1) | (b_seqs == 2)).mean(axis=1).astype(np.float32)
        b_target_eval = np.asarray(b_cells[args.target_cell],
                                   dtype=np.float32)
        n_unique_steps = len(np.unique(b_steps))
        print(f"\n[Save] Writing {perstep_h5} "
              f"({len(b_seqs)} seqs over {n_unique_steps} checkpoints, "
              f"top_k={args.perstep_archive_top_k}, "
              f"spec range=[{b_specs.min():.3f}, {b_specs.max():.3f}])")
        with h5py.File(str(perstep_h5), "w") as f:
            f.create_dataset("indices", data=b_seqs, compression="gzip")
            f.create_dataset("arr_0", data=b_onehot, compression="gzip")
            f.create_dataset("oracle_preds", data=b_target_eval,
                             compression="gzip")
            f.create_dataset("gc_fractions", data=b_gc, compression="gzip")
            f.create_dataset("mingap_spec",
                             data=b_specs.astype(np.float32),
                             compression="gzip")
            f.create_dataset("source_step", data=b_steps,
                             compression="gzip")
            for cell, scores in b_cells.items():
                f.create_dataset(f"oracle_{cell}_eval",
                                 data=np.asarray(scores, dtype=np.float32),
                                 compression="gzip")
            f.attrs["target_cell"] = args.target_cell
            f.attrs["per_step_top_k"] = args.perstep_archive_top_k
            f.attrs["actual_size"] = int(len(b_seqs))
            f.attrs["n_checkpoints"] = int(n_unique_steps)
            f.attrs["admission_rule"] = "per-step top-K by spec; accumulated"

    # Spec-bounded mingap archive (DNA-CRAFT G*-style, capacity-N by specificity)
    if (args.mingap_archive_size > 0
            and history.mingap_archive_seqs is not None
            and len(history.mingap_archive_seqs) > 0):
        mingap_h5 = output_dir / "gpa_output_mingap_filtered.h5"
        a_seqs = history.mingap_archive_seqs
        a_specs = history.mingap_archive_specs
        a_cells = history.mingap_archive_cell_scores
        a_steps = history.mingap_archive_steps
        a_onehot = np.eye(4, dtype=np.float32)[a_seqs].transpose(0, 2, 1)
        a_gc = ((a_seqs == 1) | (a_seqs == 2)).mean(axis=1).astype(np.float32)
        target_eval_scores = np.asarray(a_cells[args.target_cell],
                                        dtype=np.float32)
        print(f"\n[Save] Writing {mingap_h5} "
              f"({len(a_seqs)}/{args.mingap_archive_size} seqs, "
              f"spec range=[{a_specs.min():.3f}, {a_specs.max():.3f}])")
        with h5py.File(str(mingap_h5), "w") as f:
            f.create_dataset("indices", data=a_seqs, compression="gzip")
            f.create_dataset("arr_0", data=a_onehot, compression="gzip")
            f.create_dataset("oracle_preds", data=target_eval_scores,
                             compression="gzip")
            f.create_dataset("gc_fractions", data=a_gc, compression="gzip")
            f.create_dataset("mingap_spec",
                             data=a_specs.astype(np.float32),
                             compression="gzip")
            f.create_dataset("admission_step", data=a_steps,
                             compression="gzip")
            for cell, scores in a_cells.items():
                f.create_dataset(f"oracle_{cell}_eval",
                                 data=np.asarray(scores, dtype=np.float32),
                                 compression="gzip")
            f.attrs["target_cell"] = args.target_cell
            f.attrs["capacity"] = args.mingap_archive_size
            f.attrs["actual_size"] = int(len(a_seqs))
            f.attrs["admission_rule"] = "spec ≥ floor; capacity-bounded"

    # Per-step archive (sequences exceeding eval-oracle threshold during sampling)
    if args.archive_threshold is not None and len(history.archive_seqs) > 0:
        archive_h5 = output_dir / "gpa_output_filtered.h5"
        # Concatenate per-step archive arrays
        all_seqs = np.concatenate(history.archive_seqs, axis=0)        # (K, L)
        all_scores = np.concatenate(history.archive_scores, axis=0)    # (K,)
        all_steps = np.concatenate(history.archive_steps, axis=0)      # (K,)
        all_gc = ((all_seqs == 1) | (all_seqs == 2)).mean(axis=1).astype(np.float32)
        all_onehot = np.eye(4, dtype=np.float32)[all_seqs].transpose(0, 2, 1)
        print(f"\n[Save] Writing {archive_h5} ({len(all_seqs)} archived seqs across {len(history.archive_seqs)} checkpoints)")
        with h5py.File(str(archive_h5), "w") as f:
            f.create_dataset("indices", data=all_seqs, compression="gzip")
            f.create_dataset("arr_0", data=all_onehot, compression="gzip")
            f.create_dataset("oracle_preds", data=all_scores, compression="gzip")  # eval-oracle scores
            f.create_dataset("gc_fractions", data=all_gc, compression="gzip")
            f.create_dataset("step", data=all_steps.astype(np.int32), compression="gzip")
            f.attrs["archive_threshold"] = args.archive_threshold
            f.attrs["target_cell"] = args.target_cell
            f.attrs["n_checkpoints"] = len(history.archive_seqs)

    # History JSON
    out_json = output_dir / "gpa_history.json"
    print(f"[Save] Writing {out_json}")
    history_dict = history.to_dict()
    history_dict["args"] = vars(args)
    history_dict["final_beta"] = final_beta
    history_dict["final_log_z"] = float(sum(history.log_z_increments))
    history_dict["final_oracle_mean"] = float(oracle_scores.mean())
    history_dict["final_oracle_max"] = float(oracle_scores.max())
    history_dict["best_oracle_mean"] = best_mean_ft
    history_dict["best_oracle_max"] = float(best_oracle_scores.max())
    history_dict["n_particles"] = N
    history_dict["timestamp"] = datetime.now().isoformat()
    history_dict["cell_type_scores"] = {
        cell: {"mean": float(s.mean()), "max": float(s.max())}
        for cell, s in all_cell_scores.items()
    }
    history_dict["cell_type_scores_ft"] = {
        cell: {"mean": float(s.mean()), "max": float(s.max())}
        for cell, s in all_cell_scores_ft.items()
    }
    if best_cell_scores_ft is not None:
        history_dict["best_cell_type_scores_ft"] = {
            cell: {"mean": float(s.mean()), "max": float(s.max())}
            for cell, s in best_cell_scores_ft.items()
        }
    if best_cell_scores_eval is not None:
        history_dict["best_cell_type_scores"] = {
            cell: {"mean": float(s.mean()), "max": float(s.max())}
            for cell, s in best_cell_scores_eval.items()
        }
    history_dict["split_oracle"] = eval_oracle is not None
    if history.edit_dist_mean:
        history_dict["edit_dist_mean"] = history.edit_dist_mean
        history_dict["edit_dist_max"] = history.edit_dist_max
        history_dict["edit_efficiency"] = history.edit_efficiency
        history_dict["budget_reject_frac"] = history.budget_reject_frac
    if history.step_delta_mean:
        history_dict["step_delta_mean"] = history.step_delta_mean
        history_dict["step_delta_max"] = history.step_delta_max
    if history.eval_checkpoint_means:
        history_dict["eval_checkpoint_means"] = history.eval_checkpoint_means
        history_dict["eval_checkpoint_steps"] = history.eval_checkpoint_steps
        history_dict["best_eval_checkpoint_mean"] = max(history.eval_checkpoint_means)
        history_dict["best_eval_checkpoint_step"] = history.eval_checkpoint_steps[
            int(np.argmax(history.eval_checkpoint_means))]
    if pool_population is not None and best_eval_scores is not None:
        tc = args.target_cell
        history_dict["best_eval_oracle_mean"] = float(best_eval_scores.mean())

    with open(str(out_json), "w") as f:
        json.dump(history_dict, f, indent=2, default=str)

    # ---- Summary ----
    tc = args.target_cell
    final_ft_mean = all_cell_scores_ft[tc].mean()
    final_eval_mean = all_cell_scores[tc].mean() if eval_oracle else final_ft_mean

    print(f"\n{'='*72}")
    print("GPA COMPLETE (RERD COMPARISON)")
    print(f"{'='*72}")
    print(f"  Steps:          {len(history.beta)}")
    print(f"  Final beta:     {final_beta:.4f}")
    print(f"  Final:  ft_mean={final_ft_mean:.3f}, eval_mean={final_eval_mean:.3f} (step {len(history.beta)})")
    if best_mean_ft > final_mean_ft + 0.01:
        best_eval_mean = (best_cell_scores_eval[tc].mean()
                          if best_cell_scores_eval is not None
                          else best_cell_scores_ft[tc].mean() if best_cell_scores_ft is not None
                          else best_mean_ft)
        print(f"  Best:   ft_mean={best_mean_ft:.3f}, eval_mean={best_eval_mean:.3f} (saved separately)")
    specificity = all_cell_scores[tc].mean()
    for cell in all_cell_scores:
        if cell != tc:
            specificity -= all_cell_scores[cell].mean()
    print(f"  Specificity:    {specificity:.3f} "
          f"(target_mean - sum(off_target_means))")
    if history.eval_checkpoint_means:
        best_ec_mean = max(history.eval_checkpoint_means)
        best_ec_step = history.eval_checkpoint_steps[int(np.argmax(history.eval_checkpoint_means))]
        print(f"  Eval ckpt best: {best_ec_mean:.3f} (step {best_ec_step})")
    print(f"  GC:             {gc_fracs.mean()*100:.1f}%")
    print(f"  Unique seqs:    {history.n_unique[-1]:,}")
    print(f"  Output:         {out_h5}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
