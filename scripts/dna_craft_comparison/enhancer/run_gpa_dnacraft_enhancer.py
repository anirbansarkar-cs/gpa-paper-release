#!/usr/bin/env python3
"""GPA runner for the DNA-CRAFT (Gosai MPRA enhancer) benchmark.

One job = one (cell, replicate-seed) cell.

Layout differences vs scripts/ctrl_dna_comparison/run_gpa_hyenadna.py:
  - Oracle is a *dual* SplitModelOracle pair:
      design pool (half A) → GPA's internal reward + DPS gradient
      eval   pool (half B) → eval_oracle_fn (best_eval snapshots) + reporting
  - Internal reward is composite-mean (target − pw·mean_off) per the locked
    decision in the plan (smoother for SMC). Selection / reporting uses
    MinGap (target − max_off) which the eval pool exposes via
    `score_mingap_and_gc()`.
  - Backbone: HyenaDNA (default, ready-now) or DiMamba (when mamba_ssm
    builds against the cluster's CUDA toolchain). The backbone is purely
    a mutation kernel; the oracle stack is identical.

Recipe (locked per plan §5):
  --population_size 10000 --max_steps 60 --max_beta 100
  --noise_fraction 0.05 --branch_factor 8
  --penalty_weight 0.5 --no_bio_filter --use_dps  (DPS optional)

Output:
  output_dir/gpa_output.h5            — final population (NOT used for reporting)
  output_dir/gpa_output_best_eval.h5  — authoritative pool (best by eval-model MinGap)
  output_dir/gpa_history.json         — beta / oracle_mean / ESS curves
  output_dir/config.json              — args dump
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.gpa_sampling import DiffusionPopulationAnnealer  # noqa: E402
from scripts.dna_craft_comparison.enhancer.dual_oracle_adapter import (  # noqa: E402
    load_dual,
)

PROMPT_LABEL_MAP = {"hepg2": "100", "k562": "010", "sknsh": "001"}


def parse_args():
    p = argparse.ArgumentParser(__doc__)

    # Cell / seed
    p.add_argument("--cell", required=True, choices=["hepg2", "k562", "sknsh"])
    p.add_argument("--seed", type=int, default=0,
                   help="Replicate index (0/1/2) — also Python RNG seed.")
    p.add_argument("--seed_csv", default=None,
                   help="Path to seeds_{cell}_seed{seed}.csv. Defaults to "
                        "results/dna_craft_comparison/enhancer/seeds/.")
    p.add_argument("--from_random", action="store_true",
                   help="Initialize the population from uniform-random ACGT instead "
                        "of the natural seed pool (matches the DRAKES-protocol "
                        "random-init runs; for the rebuttal random-vs-seeded check).")

    # Backbone
    p.add_argument("--backbone", choices=["hyenadna", "dimamba"],
                   default="hyenadna")
    p.add_argument("--hyenadna_checkpoint",
                   default=str(PROJECT_ROOT / "scripts" /
                               "ctrl_dna_comparison" / "checkpoints" /
                               "hyenadna_gosai" / "best.ckpt"))
    p.add_argument("--dimamba_checkpoint", default=None,
                   help="Path to mdlm_dimamba/last.ckpt (only used if "
                        "--backbone dimamba).")

    # Oracle (split-model Enformer pair)
    p.add_argument("--enformer_dir",
                   default=str(PROJECT_ROOT / "results" /
                               "dna_craft_comparison" / "enhancer" /
                               "enformer_oracles"))

    # DPS
    p.add_argument("--use_dps", action="store_true")
    p.add_argument("--eta", type=float, default=3000.0)
    p.add_argument("--tau_start", type=float, default=1.0)
    p.add_argument("--tau_end", type=float, default=0.1)

    # GPA recipe (locked defaults)
    p.add_argument("--population_size", type=int, default=10000)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--max_beta", type=float, default=100.0)
    p.add_argument("--ess_threshold", type=float, default=0.5)
    p.add_argument("--noise_fraction", type=float, default=0.05)
    p.add_argument("--penalty_weight", type=float, default=0.5)
    p.add_argument("--branch_factor", type=int, default=8)
    p.add_argument("--hill_climb_budget", type=int, default=0)
    p.add_argument("--mutation_substeps", type=int, default=1)
    p.add_argument("--no_bio_filter", action="store_true", default=True,
                   help="Default ON per plan §5 (hard bio_filter off).")
    p.add_argument("--bio_filter", action="store_true",
                   help="Override: enable bio_filter (gc_low/gc_high).")
    p.add_argument("--gc_low", type=float, default=0.40)
    p.add_argument("--gc_high", type=float, default=0.60)

    # eval_oracle_fn cadence
    p.add_argument("--eval_checkpoint_interval", type=int, default=5,
                   help="Score with eval-pool every N GPA steps.")

    # Output
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mutation_batch_size", type=int, default=512)
    p.add_argument("--diversity_subsample", type=int, default=500)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_seeds(seed_csv: Path, n_target: int) -> torch.Tensor:
    df = pd.read_csv(seed_csv)
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    indices = np.array([[base_map[b] for b in seq] for seq in df["sequence"]])
    n_pool = len(indices)
    rng = np.random.default_rng(seed=42)
    chosen = rng.choice(n_pool, n_target,
                        replace=(n_pool < n_target))
    return torch.from_numpy(indices[chosen]).long()


def main():
    args = parse_args()
    if args.bio_filter:
        args.no_bio_filter = False
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed_csv is None:
        args.seed_csv = str(PROJECT_ROOT / "results" / "dna_craft_comparison" /
                            "enhancer" / "seeds" /
                            f"{args.cell}_seed{args.seed}.csv")

    print("=" * 72)
    print("GPA + DNA-CRAFT ENHANCER")
    print("=" * 72)
    print(f"  cell:           {args.cell}")
    print(f"  seed:           {args.seed}")
    print(f"  backbone:       {args.backbone}")
    print(f"  device:         {device}")
    print(f"  population:     {args.population_size:,}")
    print(f"  max_steps:      {args.max_steps}")
    print(f"  max_beta:       {args.max_beta}")
    print(f"  noise_fraction: {args.noise_fraction}")
    print(f"  branch_factor:  {args.branch_factor}")
    print(f"  penalty_weight: {args.penalty_weight}")
    print(f"  use_dps:        {args.use_dps}")
    print(f"  bio_filter:     {not args.no_bio_filter}")
    print(f"  output_dir:     {args.output_dir}")
    print("=" * 72)

    # ---- Oracle (dual split-model pool) ----
    print("\n[Oracle] Loading dual SplitModelOracle...")
    design, eval_pool = load_dual(
        ckpt_dir=args.enformer_dir,
        target_cell=args.cell,
        penalty_weight=args.penalty_weight,
        device=device,
    )

    def oracle_fn(seqs):
        if not isinstance(seqs, torch.Tensor):
            seqs = torch.from_numpy(seqs).long()
        return design.score(seqs)

    def fitness_fn(oracle_scores, gc_fracs):
        return design.compute_fitness(oracle_scores)

    def eval_oracle_fn(seqs):
        if not isinstance(seqs, torch.Tensor):
            seqs = torch.from_numpy(seqs).long()
        mingap, _ = eval_pool.score_mingap_and_gc(seqs)
        return mingap

    # ---- Backbone / mutator ----
    print(f"\n[Backbone] {args.backbone}")
    if args.backbone == "hyenadna":
        from scripts.ctrl_dna_comparison.hyenadna_wrapper import (
            load_hyenadna, HyenaDNAMutator, HyenaDNAMutatorDPS)
        backbone = load_hyenadna(args.hyenadna_checkpoint, device)
        prefix = PROMPT_LABEL_MAP[args.cell]
        if args.use_dps:
            mutator = HyenaDNAMutatorDPS(
                backbone, oracle_model=design,
                noise_fraction=args.noise_fraction, label=prefix,
                mutation_substeps=args.mutation_substeps,
                eta=args.eta, tau_start=args.tau_start, tau_end=args.tau_end)
        else:
            mutator = HyenaDNAMutator(
                backbone, noise_fraction=args.noise_fraction, label=prefix,
                mutation_substeps=args.mutation_substeps)

        def mutate_factory(nf, **kw):
            if args.use_dps:
                return HyenaDNAMutatorDPS(
                    backbone, oracle_model=design, noise_fraction=nf,
                    label=prefix, mutation_substeps=args.mutation_substeps,
                    eta=args.eta, tau_start=args.tau_start,
                    tau_end=args.tau_end)
            return HyenaDNAMutator(backbone, noise_fraction=nf, label=prefix,
                                    mutation_substeps=args.mutation_substeps)

    elif args.backbone == "dimamba":
        from scripts.dna_craft_comparison.enhancer.mdlm_wrapper import (
            load_dimamba, DiMambaMutator, DiMambaMutatorDPS)
        if args.dimamba_checkpoint is None:
            raise ValueError("--dimamba_checkpoint required when --backbone dimamba")
        backbone = load_dimamba(args.dimamba_checkpoint, device)
        if args.use_dps:
            mutator = DiMambaMutatorDPS(
                backbone, oracle_model=design,
                noise_fraction=args.noise_fraction,
                mutation_substeps=args.mutation_substeps,
                eta=args.eta, tau_start=args.tau_start, tau_end=args.tau_end)
        else:
            mutator = DiMambaMutator(
                backbone, noise_fraction=args.noise_fraction,
                mutation_substeps=args.mutation_substeps)

        def mutate_factory(nf, **kw):
            if args.use_dps:
                return DiMambaMutatorDPS(
                    backbone, oracle_model=design, noise_fraction=nf,
                    mutation_substeps=args.mutation_substeps,
                    eta=args.eta, tau_start=args.tau_start, tau_end=args.tau_end)
            return DiMambaMutator(backbone, noise_fraction=nf,
                                  mutation_substeps=args.mutation_substeps)
    else:
        raise ValueError(f"unknown backbone {args.backbone!r}")

    # ---- Bio filter ----
    bio_filter_fn = None
    if not args.no_bio_filter:
        from bio_plausibility import is_bio_plausible

        def bio_filter_fn(idx_np):
            gc = ((idx_np == 1) | (idx_np == 2)).mean(axis=1)
            mask, _ = is_bio_plausible(idx_np, gc_fractions=gc,
                                       gc_range=(args.gc_low, args.gc_high))
            return mask

    # ---- Init population: random ACGT or natural seed pool ----
    if args.from_random:
        L = 200
        rng_pop = np.random.default_rng(seed=args.seed)
        population = torch.from_numpy(
            rng_pop.integers(0, 4, size=(args.population_size, L))).long()
        print(f"\n[Init] Random DNA sequences: {args.population_size:,} x {L}bp "
              f"(seed={args.seed})")
    else:
        print(f"\n[Init] seeds <- {args.seed_csv}")
        population = load_seeds(Path(args.seed_csv), args.population_size)
    print(f"  population: {tuple(population.shape)}")
    labels = torch.zeros(population.shape[0], 1, device=device)

    # ---- Run GPA ----
    gpa = DiffusionPopulationAnnealer(
        mutator, oracle_fn, model=None, device=device,
        fitness_fn=fitness_fn,
        mutate_fn_factory=mutate_factory,
        default_nf=args.noise_fraction,
        branch_factor=args.branch_factor,
        hill_climb_budget=args.hill_climb_budget,
        oracle_fn_fast=None,
    )
    t0 = time.time()
    (population, oracle_scores, log_weights, history,
     best_population, best_oracle_scores,
     best_eval_population, best_eval_scores) = gpa.run(
        population, labels,
        max_beta=args.max_beta,
        ess_threshold=args.ess_threshold,
        max_steps=args.max_steps,
        bio_filter_fn=bio_filter_fn,
        mutation_batch_size=args.mutation_batch_size,
        diversity_subsample=args.diversity_subsample,
        eval_oracle_fn=eval_oracle_fn,
        eval_checkpoint_interval=args.eval_checkpoint_interval,
    )
    elapsed = time.time() - t0
    print(f"\n[GPA] completed in {elapsed:.1f}s")

    # ---- Final population ----
    indices = population.cpu().numpy().astype(np.int8)
    onehot = np.eye(4, dtype=np.float32)[indices].transpose(0, 2, 1)
    gc_fracs = ((indices == 1) | (indices == 2)).mean(axis=1).astype(np.float32)
    all_cells = eval_pool.score_all_cells(population.to(device))

    final_h5 = out_dir / "gpa_output.h5"
    with h5py.File(final_h5, "w") as f:
        f.create_dataset("indices", data=indices, compression="gzip")
        f.create_dataset("arr_0", data=onehot, compression="gzip")
        f.create_dataset("oracle_preds", data=oracle_scores, compression="gzip")
        f.create_dataset("gc_fractions", data=gc_fracs, compression="gzip")
        f.create_dataset("log_weights", data=log_weights, compression="gzip")
        for c, sc in all_cells.items():
            f.create_dataset(f"oracle_eval_{c}", data=sc, compression="gzip")
        f.attrs["target_cell"] = args.cell
        f.attrs["seed"] = args.seed
        f.attrs["backbone"] = args.backbone
        f.attrs["elapsed_seconds"] = elapsed

    # ---- Best-eval population (authoritative for reporting) ----
    if best_eval_population is not None:
        be_indices = best_eval_population.cpu().numpy().astype(np.int8)
        be_onehot = np.eye(4, dtype=np.float32)[be_indices].transpose(0, 2, 1)
        be_gc = ((be_indices == 1) | (be_indices == 2)).mean(axis=1).astype(np.float32)
        be_all = eval_pool.score_all_cells(best_eval_population.to(device))
        be_path = out_dir / "gpa_output_best_eval.h5"
        with h5py.File(be_path, "w") as f:
            f.create_dataset("indices", data=be_indices, compression="gzip")
            f.create_dataset("arr_0", data=be_onehot, compression="gzip")
            f.create_dataset("oracle_preds_eval_mingap", data=best_eval_scores,
                              compression="gzip")
            f.create_dataset("gc_fractions", data=be_gc, compression="gzip")
            for c, sc in be_all.items():
                f.create_dataset(f"oracle_eval_{c}", data=sc, compression="gzip")
            f.attrs["target_cell"] = args.cell
            f.attrs["seed"] = args.seed
            f.attrs["selection_metric"] = "eval_mingap"
        print(f"  best_eval written → {be_path} ({len(be_indices):,} rows)")
    else:
        print("  WARNING: best_eval_population is None — no eval checkpoints fired.")

    # ---- History + config ----
    with open(out_dir / "gpa_history.json", "w") as f:
        json.dump({
            "beta": [float(b) for b in history.beta],
            "oracle_mean": [float(m) for m in history.oracle_mean],
            "oracle_max": [float(m) for m in history.oracle_max],
            "ess_fraction": [float(e) for e in history.ess_fraction],
            "gc_mean": [float(g) for g in history.gc_mean],
        }, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nDone: {out_dir}")


if __name__ == "__main__":
    main()
