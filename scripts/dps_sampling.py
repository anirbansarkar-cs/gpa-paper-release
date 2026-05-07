#!/usr/bin/env python3
"""
Predictor Gradient Guidance (DPS-style) Sampling for the discrete-diffusion DNA model.

At each denoising step, after computing transition probabilities, uses oracle
gradients (via Gumbel-Softmax relaxation) to bias probabilities toward higher
oracle scores before sampling.

Usage:
    python scripts/dps_sampling.py \
        --checkpoint $CKPT --oracle_checkpoint $ORACLE_CKPT \
        --output_dir results/dps_YYYYMMDD
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from scripts.cfg_sampling import get_cfg_score_fn
from scripts.sampling import get_predictor, Denoiser
from model_zoo.lentimpra.cfg_models import load_trained_cfg_model
from model_zoo.lentimpra.cfg_sweep import (
    load_oracle, score_with_oracle, indices_to_onehot,
    save_h5, build_archives, load_warm_start_sequences,
    compute_diversity, print_results_table,
)
from utils.catsample import sample_categorical


def dps_sample(model, graph, noise, oracle_model, num_samples, seq_len, steps,
               activity, guidance_weight, gc_penalty, gc_target_low, gc_target_high, device,
               eta=1.0, tau_start=1.0, tau_end=0.1,
               guide_start_step=0, sampling_batch_size=256):
    """Run DPS-guided sampling.

    Args:
        model: Diffusion model.
        graph: Discrete diffusion graph.
        noise: Noise schedule.
        oracle_model: Loaded oracle for gradient guidance.
        num_samples: Number of sequences to generate.
        seq_len: Sequence length.
        steps: Number of denoising steps.
        activity: Conditioning activity label.
        guidance_weight: CFG guidance weight.
        gc_penalty: GC composition penalty.
        gc_target_low: Lower GC fraction guardrail.
        gc_target_high: Upper GC fraction guardrail.
        device: CUDA device.
        eta: Gradient guidance strength.
        tau_start: Gumbel-Softmax start temperature.
        tau_end: Gumbel-Softmax end temperature.
        guide_start_step: Step index to begin gradient guidance.
        sampling_batch_size: GPU batch size.

    Returns:
        sequences: Index tensor (N, L) on CPU.
    """
    if sampling_batch_size >= num_samples:
        return _dps_sample_batch(
            model, graph, noise, oracle_model, num_samples, seq_len, steps,
            activity, guidance_weight, gc_penalty, gc_target_low, gc_target_high, device,
            eta, tau_start, tau_end, guide_start_step)

    all_seqs = []
    for start in range(0, num_samples, sampling_batch_size):
        bs = min(sampling_batch_size, num_samples - start)
        batch_seqs = _dps_sample_batch(
            model, graph, noise, oracle_model, bs, seq_len, steps,
            activity, guidance_weight, gc_penalty, gc_target_low, gc_target_high, device,
            eta, tau_start, tau_end, guide_start_step)
        all_seqs.append(batch_seqs)
        torch.cuda.empty_cache()

    return torch.cat(all_seqs, dim=0)


def _dps_sample_batch(model, graph, noise, oracle_model, batch_size, seq_len,
                      steps, activity, guidance_weight, gc_penalty, gc_target_low,
                      gc_target_high, device, eta, tau_start, tau_end, guide_start_step):
    """DPS sampling for a single batch."""
    sigma_max = noise.sigmas[1].item()
    labels = torch.full((batch_size, 1), activity, device=device)
    eps = 1e-5

    # Build CFG score function (returns probs when sampling=True)
    sampling_score_fn = get_cfg_score_fn(
        model, guidance_weight, train=False, sampling=True,
        gc_penalty=gc_penalty, gc_target_low=gc_target_low, gc_target_high=gc_target_high, sigma_max=sigma_max)

    # Initialize from uniform noise
    x = graph.sample_limit(batch_size, seq_len).to(device)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    for i in range(steps):
        t = timesteps[i] * torch.ones(batch_size, 1, device=device)
        curr_sigma = noise(t)[0]
        next_sigma = noise(t - (1 - eps) / steps)[0]
        dsigma = curr_sigma - next_sigma

        # 1. Standard diffusion step (no_grad)
        with torch.no_grad():
            score = sampling_score_fn(x, curr_sigma, labels)  # (B, L, 4)
            stag_score = graph.staggered_score(score, dsigma)
            probs = stag_score * graph.transp_transition(x, dsigma)  # (B, L, 4)

        # 2. Gradient guidance (only after guide_start_step)
        if i >= guide_start_step and eta > 0:
            log_probs = probs.clamp(min=1e-10).log()
            log_probs_grad = log_probs.detach().requires_grad_(True)

            # Gumbel-Softmax temperature schedule
            tau = max(tau_end, tau_start * (1.0 - i / steps))
            soft_onehot = F.gumbel_softmax(log_probs_grad, tau=tau, hard=False)  # (B, L, 4)

            # Convert to oracle format: (B, 4, L) with channel swap C↔G
            soft_oracle = soft_onehot.permute(0, 2, 1).clone()  # (B, 4, L)
            ch1 = soft_oracle[:, 1, :].clone()
            ch2 = soft_oracle[:, 2, :].clone()
            soft_oracle[:, 1, :] = ch2  # oracle G ← diffusion G
            soft_oracle[:, 2, :] = ch1  # oracle C ← diffusion C

            oracle_scores = oracle_model.model(soft_oracle)  # (B, 1) or (B,)
            oracle_scores = oracle_scores.squeeze(-1) if oracle_scores.dim() > 1 else oracle_scores
            oracle_scores.sum().backward()

            grad = log_probs_grad.grad  # (B, L, 4)
            # Per-sample gradient normalization to prevent mode collapse
            grad_norm = grad.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
            guided_log_probs = log_probs + eta * (grad / grad_norm)
            guided_probs = F.softmax(guided_log_probs, dim=-1)

            x = sample_categorical(guided_probs)
        else:
            x = sample_categorical(probs)

    # Denoising step
    with torch.no_grad():
        t = timesteps[-1] * torch.ones(batch_size, 1, device=device)
        sigma = noise(t)[0]
        score = sampling_score_fn(x, sigma, labels)
        stag_score = graph.staggered_score(score, sigma)
        probs = stag_score * graph.transp_transition(x, sigma)
        if graph.absorb:
            probs = probs[..., :-1]
        x = sample_categorical(probs)

    return x.cpu()


def parse_args():
    parser = argparse.ArgumentParser(
        description='DPS (Predictor Gradient Guidance) Sampling')

    # Required
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--oracle_checkpoint', type=str, required=True)

    # DPS-specific
    parser.add_argument('--etas', type=float, nargs='+',
                        default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                        help='Gradient guidance strengths to sweep')
    parser.add_argument('--tau_start', type=float, default=1.0,
                        help='Gumbel-Softmax start temperature')
    parser.add_argument('--tau_end', type=float, default=0.1,
                        help='Gumbel-Softmax end temperature')
    parser.add_argument('--guide_start_steps', type=str, default='0,quarter,half',
                        help='Comma-separated: step indices or "quarter","half" fractions')

    # CFG / sampling
    parser.add_argument('--guidance_weights', type=float, nargs='+',
                        default=[5, 10, 15])
    parser.add_argument('--gc_penalties', type=float, nargs='+',
                        default=[0.0, 3.0, 5.0])
    parser.add_argument('--gc_target_low', type=float, default=0.40)
    parser.add_argument('--gc_target_high', type=float, default=0.60)
    parser.add_argument('--activities', type=float, nargs='+', default=[3.0])
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--sampling_batch_size', type=int, default=256)

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--oracle_threshold', type=float, default=4.0)
    parser.add_argument('--gc_threshold', type=float, default=0.45)

    return parser.parse_args()


def _parse_guide_start(spec, steps):
    """Parse guide_start_steps spec into list of integers."""
    result = []
    for s in spec.split(','):
        s = s.strip()
        if s == 'quarter':
            result.append(steps // 4)
        elif s == 'half':
            result.append(steps // 2)
        else:
            result.append(int(s))
    return sorted(set(result))


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_start = time.time()

    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"results/dps_sampling_{timestamp}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if args.config is None:
        config_path = project_root / 'model_zoo' / 'lentimpra' / 'configs' / 'cfg_transformer.yaml'
    else:
        config_path = Path(args.config)
    config = OmegaConf.load(str(config_path))

    print("=" * 80)
    print("DPS (Predictor Gradient Guidance) Sampling")
    print("=" * 80)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Oracle:         {args.oracle_checkpoint}")
    print(f"Device:         {device}")
    print(f"Etas:           {args.etas}")
    print(f"Tau:            {args.tau_start} → {args.tau_end}")
    print(f"Guide starts:   {args.guide_start_steps}")
    print(f"Guidance wts:   {args.guidance_weights}")
    print(f"GC penalties:   {args.gc_penalties}")
    print(f"Samples/cond:   {args.num_samples}")
    print(f"Steps:          {args.steps}")
    print(f"Output:         {output_dir}")

    # Load models
    print("\n--- Loading diffusion model ---")
    model, graph, noise = load_trained_cfg_model(
        args.checkpoint, config, 'cfg_transformer', device)
    model.eval()
    seq_len = config.dataset.sequence_length

    print("\n--- Loading oracle model ---")
    oracle_model = load_oracle(args.oracle_checkpoint, device)

    # Build condition grid
    guide_starts = _parse_guide_start(args.guide_start_steps, args.steps)
    conditions = [
        (eta, gw, gcp, gs, act)
        for eta in args.etas
        for gw in args.guidance_weights
        for gcp in args.gc_penalties
        for gs in guide_starts
        for act in args.activities
    ]
    print(f"\nTotal conditions: {len(conditions)}")
    print(f"Total samples:   {len(conditions) * args.num_samples:,}")

    # Sweep
    results = []
    all_sequences = {}

    for ci, (eta, gw, gcp, gs, act) in enumerate(conditions):
        tag = f"eta{eta}_w{gw}_gc{gcp}_gs{gs}_act{act}"
        print(f"\n[{ci+1}/{len(conditions)}] {tag}")
        t_cond = time.time()

        sequences = dps_sample(
            model, graph, noise, oracle_model,
            args.num_samples, seq_len, args.steps,
            act, gw, gcp, args.gc_target_low, args.gc_target_high, device,
            eta=eta, tau_start=args.tau_start, tau_end=args.tau_end,
            guide_start_step=gs,
            sampling_batch_size=args.sampling_batch_size)

        preds, gc_fracs = score_with_oracle(oracle_model, sequences)
        entropy, pairwise_id, n_unique = compute_diversity(sequences)
        elapsed = time.time() - t_cond

        result = {
            'tag': tag, 'eta': eta, 'guidance_weight': gw,
            'gc_penalty': gcp, 'guide_start_step': gs, 'activity': act,
            'mean_oracle': float(np.mean(preds)),
            'max_oracle': float(np.max(preds)),
            'median_oracle': float(np.median(preds)),
            'std_oracle': float(np.std(preds)),
            'gc_pct': float(100 * np.mean(gc_fracs)),
            'pct_above_3': float(100 * np.mean(preds > 3.0)),
            'pct_above_4': float(100 * np.mean(preds > 4.0)),
            'pct_above_5': float(100 * np.mean(preds > 5.0)),
            'entropy': entropy, 'pairwise_id': pairwise_id,
            'n_unique': n_unique, 'n_samples': len(preds),
            'elapsed_s': elapsed,
        }
        results.append(result)
        all_sequences[tag] = (sequences, preds, gc_fracs)
        save_h5(sequences, preds, gc_fracs, output_dir / f"samples_{tag}.h5")

        print(f"  oracle: mean={result['mean_oracle']:.3f}, "
              f"max={result['max_oracle']:.3f}, "
              f"GC={result['gc_pct']:.1f}%, "
              f"entropy={entropy:.3f}, unique={n_unique}, ({elapsed:.1f}s)")

    # Results
    print_results_table(results)

    print("\n--- Building archives ---")
    build_archives(all_sequences, output_dir,
                   oracle_threshold=args.oracle_threshold,
                   gc_threshold=args.gc_threshold)

    # Save summary
    summary = {
        'method': 'dps_gradient_guidance',
        'checkpoint': args.checkpoint,
        'oracle_checkpoint': args.oracle_checkpoint,
        'tau_start': args.tau_start, 'tau_end': args.tau_end,
        'num_samples_per_condition': args.num_samples,
        'steps': args.steps,
        'total_conditions': len(conditions),
        'total_samples': sum(r['n_samples'] for r in results),
        'total_time_s': time.time() - t_start,
        'results': results,
    }
    summary_path = output_dir / 'sweep_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Total time: {(time.time() - t_start)/60:.1f} minutes")
    print(f"Results in: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
