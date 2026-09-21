#!/usr/bin/env python3
"""
GPA with HyenaDNA backbone for the Reddy 2024 promoter MPRA.

Shared backbone with run_ctrldna_promoter.py — both load the same
fine-tuned HyenaDNA (hyenadna_promoter_full/best.ckpt). GPA freezes it
and drives sampling via DiffusionPopulationAnnealer + HyenaDNAMutator
(optional DPS gradient from PromoterMultiAdapter.dps_forward).

Minimal feature set (no islands / step-archive / k-mer / parallel tempering —
those live in the enhancer run_gpa_hyenadna.py and aren't part of the
head-to-head scope).

Usage:
    python scripts/ctrl_dna_comparison/promoter/run_gpa_hyenadna_promoter.py \
        --hyenadna_checkpoint scripts/ctrl_dna_comparison/promoter/checkpoints/hyenadna_promoter_full/best.ckpt \
        --target_cell JURKAT \
        --seed_csv scripts/ctrl_dna_comparison/promoter/data/seeds_JURKAT.csv \
        --output_dir results/ctrl_dna_comparison/promoter/gpa_jurkat_seed0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from oracle_adapter import (  # noqa: E402
    PROMOTER_CELLS, PromoterOraclePool, PromoterMultiAdapter,
)

_PROJECT_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.gpa_sampling import DiffusionPopulationAnnealer  # noqa: E402
from scripts.ctrl_dna_comparison.hyenadna_wrapper import (  # noqa: E402
    load_hyenadna, HyenaDNAMutator, HyenaDNAMutatorDPS,
)
from bio_plausibility import is_bio_plausible  # noqa: E402

DEFAULT_PREFIX_LABEL = {"JURKAT": "100", "K562": "010", "THP1": "001"}


def load_seeds_from_csv(csv_path, population_size, seq_len):
    df = pd.read_csv(csv_path)
    seqs = df["sequence"].values
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    indices = np.array([[base_map[b] for b in seq] for seq in seqs])
    assert indices.shape[1] == seq_len, (
        f"seed CSV has {indices.shape[1]}bp sequences but --seq_len={seq_len}"
    )
    N_pool = len(indices)
    if N_pool >= population_size:
        chosen = np.random.choice(N_pool, population_size, replace=False)
    else:
        chosen = np.random.choice(N_pool, population_size, replace=True)
    return torch.from_numpy(indices[chosen]).long()


def parse_args():
    p = argparse.ArgumentParser(description="GPA + HyenaDNA on Reddy promoter")
    p.add_argument("--hyenadna_checkpoint", required=True)
    p.add_argument(
        "--oracle_ckpt_dir",
        default="scripts/ctrl_dna_comparison/promoter/checkpoints",
    )
    p.add_argument("--target_cell", required=True, choices=list(PROMOTER_CELLS))
    p.add_argument("--prefix_label", default=None,
                   help="Defaults to '100'/'010'/'001' per target_cell.")
    p.add_argument("--seed_csv", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--seq_len", type=int, default=250)
    p.add_argument("--population_size", type=int, default=2000)
    p.add_argument("--max_beta", type=float, default=50.0)
    p.add_argument("--ess_threshold", type=float, default=0.5)
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--noise_fraction", type=float, default=0.10)
    p.add_argument("--mutation_substeps", type=int, default=1)
    p.add_argument("--mutation_batch_size", type=int, default=256)

    p.add_argument("--use_dps", action="store_true")
    p.add_argument("--eta", type=float, default=3000.0)
    p.add_argument("--tau_start", type=float, default=1.0)
    p.add_argument("--penalty_weight", type=float, default=0.0,
                   help="GPA & DPS off-target penalty (0 = target-only).")

    p.add_argument("--bio_filter", action="store_true", default=True)
    p.add_argument("--no_bio_filter", action="store_true")
    p.add_argument("--gc_low", type=float, default=0.45)
    p.add_argument("--gc_high", type=float, default=0.55)

    p.add_argument("--from_random", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--branch_factor", type=int, default=1)
    p.add_argument("--branch_factor_schedule", type=str, default=None,
                   help="K-annealing schedule. Overrides static --branch_factor per "
                        "step if set. Forms: 'linear_8_1', 'step_8_4_2_1', "
                        "'late_8_1_80' (K=8 until 80%% of steps, then K=1), "
                        "'early_8_1_20'. See _branch_factor_at_step().")
    p.add_argument("--hill_climb_budget", type=int, default=0)
    p.add_argument("--branch_selection_tau", type=float, default=0.0,
                   help="K-branch selection temperature. 0=argmax (default), "
                        ">0=softmax sampling (ε-greedy).")
    p.add_argument("--nf_start", type=float, default=0.0,
                   help="NF anneal start (high, exploratory). 0 disables.")
    p.add_argument("--nf_end", type=float, default=0.0,
                   help="NF anneal end (low, fine-tuning). Used only when "
                        "--nf_start > 0.")

    # Diversity knobs (Phase 2 port from enhancer run_gpa_hyenadna.py)
    p.add_argument("--max_copies", type=int, default=0,
                   help="Cap duplicates post-resample (0=off).")
    p.add_argument("--diversity_lambda", type=float, default=0.0,
                   help="Fitness penalty λ·conformity (0=off).")
    p.add_argument("--dedup_threshold", type=int, default=0,
                   help="Re-mutate near-duplicates within Hamming<threshold (0=off).")
    p.add_argument("--rejuvenation_fraction", type=float, default=0.0,
                   help="Fraction of lowest-fitness particles replaced each step (0=off).")

    # SMC-soundness corrections (S5, S7)
    p.add_argument("--apf_correction", action="store_true",
                   help="S5: APF reweight for K-branch τ-softmax selection "
                        "(Pitt–Shephard 1999). Requires --branch_selection_tau > 0.")
    p.add_argument("--dps_iw_correct", action="store_true",
                   help="S7: importance-weight correction for DPS gradient-warped "
                        "proposal (Del Moral–Doucet–Jasra 2006 §3.1). "
                        "Active only when --use_dps is also set.")

    # Archive: collect per-step seqs exceeding target threshold, save as gpa_output_archive.h5
    p.add_argument("--archive_threshold", type=float, default=None,
                   help="Per-step archive threshold on target score. None=off.")

    return p.parse_args()


def main():
    args = parse_args()
    if args.no_bio_filter:
        args.bio_filter = False
    if args.prefix_label is None:
        args.prefix_label = DEFAULT_PREFIX_LABEL[args.target_cell]
    assert all(c in "01" for c in args.prefix_label), (
        f"prefix_label must be binary (0/1 digits) per hyenadna_wrapper.LABEL_STOI — "
        f"got {args.prefix_label!r}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"GPA + HYENADNA PROMOTER  target={args.target_cell}  label={args.prefix_label!r}")
    print(f"  seq_len={args.seq_len}  pop={args.population_size:,}  max_beta={args.max_beta}")
    print(f"  DPS={args.use_dps}  eta={args.eta}  pw={args.penalty_weight}  nf={args.noise_fraction}")
    print(f"  bio_filter={args.bio_filter}  GC=[{args.gc_low}, {args.gc_high}]")
    print(f"  output: {args.output_dir}")
    print("=" * 72)

    print("\n[Model] Loading fine-tuned HyenaDNA...")
    hyenadna = load_hyenadna(args.hyenadna_checkpoint, device)

    print(f"\n[Oracle] Loading promoter EnformerModel pool from {args.oracle_ckpt_dir}")
    pool = PromoterOraclePool(args.oracle_ckpt_dir, device=device)
    oracle = PromoterMultiAdapter(
        pool, target_cell=args.target_cell, penalty_weight=args.penalty_weight,
    )

    # Mutation kernel
    if args.use_dps:
        print(f"\n[DPS] eta={args.eta} tau={args.tau_start}  iw_correct={args.dps_iw_correct}")
        mutator = HyenaDNAMutatorDPS(
            hyenadna, oracle_model=oracle,
            noise_fraction=args.noise_fraction, label=args.prefix_label,
            mutation_substeps=args.mutation_substeps,
            eta=args.eta, tau_start=args.tau_start, tau_end=args.tau_start,
            dps_iw_correct=args.dps_iw_correct,
        )
    else:
        mutator = HyenaDNAMutator(
            hyenadna, noise_fraction=args.noise_fraction, label=args.prefix_label,
            mutation_substeps=args.mutation_substeps,
        )

    def mutate_fn_factory(nf, guidance_weight=None, **kwargs):
        if args.use_dps:
            return HyenaDNAMutatorDPS(
                hyenadna, oracle_model=oracle,
                noise_fraction=nf, label=args.prefix_label,
                mutation_substeps=args.mutation_substeps,
                eta=args.eta, tau_start=args.tau_start, tau_end=args.tau_start,
                dps_iw_correct=args.dps_iw_correct,
            )
        return HyenaDNAMutator(
            hyenadna, noise_fraction=nf, label=args.prefix_label,
            mutation_substeps=args.mutation_substeps,
        )

    def oracle_fn(sequences_tensor):
        if not isinstance(sequences_tensor, torch.Tensor):
            sequences_tensor = torch.from_numpy(np.asarray(sequences_tensor)).long()
        return oracle.score(sequences_tensor)

    fitness_fn = None
    if args.penalty_weight > 0:
        def fitness_fn(oracle_scores, gc_fractions):
            return oracle.compute_fitness(oracle_scores)

    bio_filter_fn = None
    if args.bio_filter:
        def bio_filter_fn(indices_np):
            gc_mask = (indices_np == 1) | (indices_np == 2)
            gc_fracs = gc_mask.mean(axis=1)
            ok, _ = is_bio_plausible(
                indices_np, gc_fractions=gc_fracs,
                gc_range=(args.gc_low, args.gc_high),
            )
            return ok

    # Population init
    if args.from_random:
        print(f"\n[Init] Random DNA: {args.population_size:,} x {args.seq_len}bp")
        population = torch.randint(0, 4, (args.population_size, args.seq_len))
    else:
        print(f"\n[Init] Loading seeds: {args.seed_csv}")
        population = load_seeds_from_csv(
            args.seed_csv, args.population_size, args.seq_len
        )
    labels = torch.zeros(population.shape[0], 1, device=device)
    print(f"  Population shape: {tuple(population.shape)}")

    # GPA run
    annealer = DiffusionPopulationAnnealer(
        mutate_fn=mutator,
        oracle_fn=oracle_fn,
        model=hyenadna,
        device=device,
        fitness_fn=fitness_fn,
        mutate_fn_factory=mutate_fn_factory,
        branch_factor=args.branch_factor,
        hill_climb_budget=args.hill_climb_budget,
        branch_selection_tau=args.branch_selection_tau,
        diversity_lambda=args.diversity_lambda,
        apf_correction=args.apf_correction,
        dps_iw_correct=args.dps_iw_correct,
    )

    print(f"  [diversity] max_copies={args.max_copies}  diversity_lambda={args.diversity_lambda}  "
          f"dedup_threshold={args.dedup_threshold}  rejuvenation_fraction={args.rejuvenation_fraction}")

    # Archive wiring: eval_oracle_fn returns just scores (target-cell);
    # eval_checkpoint_interval=1 triggers archive at every step.
    eval_oracle_fn = None
    eval_ckpt_interval = 0
    if args.archive_threshold is not None:
        def eval_oracle_fn(sequences_tensor):
            if not isinstance(sequences_tensor, torch.Tensor):
                sequences_tensor = torch.from_numpy(np.asarray(sequences_tensor)).long()
            scores, _ = oracle.score(sequences_tensor)
            return scores
        eval_ckpt_interval = 1
        print(f"  [archive] threshold={args.archive_threshold}  every_step")

    t0 = time.time()
    result = annealer.run(
        population=population,
        labels=labels,
        max_beta=args.max_beta,
        max_steps=args.max_steps,
        ess_threshold=args.ess_threshold,
        bio_filter_fn=bio_filter_fn,
        mutation_batch_size=args.mutation_batch_size,
        max_copies=args.max_copies,
        dedup_threshold=args.dedup_threshold,
        rejuvenation_fraction=args.rejuvenation_fraction,
        eval_oracle_fn=eval_oracle_fn,
        eval_checkpoint_interval=eval_ckpt_interval,
        archive_threshold=args.archive_threshold,
        nf_start=args.nf_start,
        nf_end=args.nf_end,
        branch_factor_schedule=args.branch_factor_schedule,
    )
    elapsed = time.time() - t0
    print(f"\nGPA run finished in {elapsed:.0f}s")

    # run() returns an 8-tuple (see scripts/gpa_sampling.py:972).
    (final_pop, final_scores, log_weights, history,
     best_population, best_oracle_scores,
     best_eval_population, best_eval_scores) = result

    def _save_pool(tag, pop, target_scores):
        if pop is None:
            return None
        pop_np = pop.cpu().numpy() if isinstance(pop, torch.Tensor) else np.asarray(pop)
        scores_np = (target_scores.cpu().numpy() if isinstance(target_scores, torch.Tensor)
                     else np.asarray(target_scores)) if target_scores is not None else None
        _, gc_np = oracle.score(torch.from_numpy(pop_np).long())
        all_preds = oracle._last_all_preds
        if scores_np is None:
            scores_np = all_preds[args.target_cell]
        h5_path = out_dir / f"gpa_output_{tag}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("sequences", data=pop_np.astype(np.int8))
            f.create_dataset("target_scores", data=scores_np.astype(np.float32))
            f.create_dataset("gc_fractions", data=gc_np.astype(np.float32))
            for cell in PROMOTER_CELLS:
                f.create_dataset(f"score_{cell}", data=all_preds[cell].astype(np.float32))
            f.attrs["target_cell"] = args.target_cell
            f.attrs["prefix_label"] = args.prefix_label
            f.attrs["penalty_weight"] = args.penalty_weight
            f.attrs["use_dps"] = int(args.use_dps)
            f.attrs["seq_len"] = args.seq_len
            f.attrs["seed"] = args.seed
            f.attrs["pool_tag"] = tag
        print(f"Saved {h5_path}  N={pop_np.shape[0]}  "
              f"target_mean={scores_np.mean():+.3f}  max={scores_np.max():+.3f}")
        return {"N": int(pop_np.shape[0]),
                "mean": float(scores_np.mean()), "max": float(scores_np.max()),
                "p95": float(np.percentile(scores_np, 95)),
                "mean_all_cells": {c: float(all_preds[c].mean()) for c in PROMOTER_CELLS}}

    stats = {}
    stats["final"] = _save_pool("final", final_pop, final_scores)
    stats["best"] = _save_pool("best", best_population, best_oracle_scores)
    # per rule: authoritative reported pool is best_eval
    stats["best_eval"] = _save_pool("best_eval", best_eval_population, best_eval_scores)

    # Archive save: concat all per-step archive shards and persist to h5
    if args.archive_threshold is not None and len(history.archive_seqs) > 0:
        arch_seqs = np.concatenate(history.archive_seqs, axis=0)
        arch_scores = np.concatenate(history.archive_scores, axis=0)
        arch_steps = np.concatenate(history.archive_steps, axis=0)
        arch_tensor = torch.from_numpy(arch_seqs).long()
        # Re-score all 3 cells on the archive (so eval doesn't have to)
        _, arch_gc = oracle.score(arch_tensor)
        all_preds = oracle._last_all_preds
        h5_path = out_dir / "gpa_output_archive.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("sequences", data=arch_seqs.astype(np.int8))
            f.create_dataset("target_scores", data=arch_scores.astype(np.float32))
            f.create_dataset("archive_steps", data=arch_steps.astype(np.int32))
            f.create_dataset("gc_fractions", data=arch_gc.astype(np.float32))
            for cell in PROMOTER_CELLS:
                f.create_dataset(f"score_{cell}", data=all_preds[cell].astype(np.float32))
            f.attrs["target_cell"] = args.target_cell
            f.attrs["archive_threshold"] = args.archive_threshold
            f.attrs["seed"] = args.seed
            f.attrs["pool_tag"] = "archive"
        print(f"Saved {h5_path}  N={arch_seqs.shape[0]}  "
              f"target_mean={arch_scores.mean():+.3f}  max={arch_scores.max():+.3f}  "
              f"steps_spanned={arch_steps.min()}..{arch_steps.max()}")
        stats["archive"] = {"N": int(arch_seqs.shape[0]),
                             "mean": float(arch_scores.mean()),
                             "max": float(arch_scores.max()),
                             "p95": float(np.percentile(arch_scores, 95)),
                             "mean_all_cells": {c: float(all_preds[c].mean()) for c in PROMOTER_CELLS}}
    elif args.archive_threshold is not None:
        print(f"  [archive] threshold={args.archive_threshold} yielded 0 seqs — not saving.")

    meta = {
        "target_cell": args.target_cell,
        "prefix_label": args.prefix_label,
        "penalty_weight": args.penalty_weight,
        "use_dps": args.use_dps,
        "eta": args.eta,
        "seq_len": args.seq_len,
        "max_beta": args.max_beta,
        "max_steps": args.max_steps,
        "noise_fraction": args.noise_fraction,
        "elapsed_seconds": elapsed,
        "pool_stats": stats,
    }
    with open(out_dir / "gpa_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
