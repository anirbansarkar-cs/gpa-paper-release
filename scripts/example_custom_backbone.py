#!/usr/bin/env python3
"""Minimal end-to-end GPA run with a custom generator and a custom oracle.

This is a template, not an experiment. It shows the only two things GPA needs
from you — a mutation proposal and a reward — using stand-ins small enough to
read in one sitting:

  * ToyGenerator  : masks a fraction of positions and refills them uniformly.
                    Replace with partial-corruption + re-denoise under your
                    diffusion model, or partial-context resampling under your
                    autoregressive model.
  * ToyOracle     : counts matches of a target motif. Replace with your
                    sequence-to-function predictor.

Run it directly; it needs only torch and numpy:

    python scripts/example_custom_backbone.py

Real implementations to copy from:
    scripts/rerd_comparison/mdlm_wrapper.py            masked diffusion (MDLM)
    scripts/dna_craft_comparison/enhancer/mdlm_wrapper.py   DiMamba
    scripts/ctrl_dna_comparison/hyenadna_wrapper.py    autoregressive (HyenaDNA)
    scripts/rerd_comparison/enformer_oracle.py         oracle, incl. dps_forward
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gpa_sampling import DiffusionPopulationAnnealer  # noqa: E402

BASES = "ACGT"                      # A=0, C=1, G=2, T=3 throughout
MOTIF = "TGACTCA"                   # the toy objective: maximise AP-1 hits
SEQ_LEN = 100
POP_SIZE = 512


class ToyGenerator:
    """Stand-in for a pretrained generator.

    The contract: given clean sequences, return sequences that differ in a
    limited, model-defined way. Everything GPA does on top — annealing,
    resampling, branching — assumes only that this proposal stays close to the
    prior. Here "the prior" is uniform, so the toy makes no biological sense;
    that is the part you replace.
    """

    def __init__(self, noise_fraction: float = 0.05):
        self.noise_fraction = noise_fraction

    def __call__(self, model_unused, x_clean, labels, branch_factor=1, **kwargs):
        """(B, L) -> (B, L), or (K, B, L) when branch_factor > 1."""
        B, L = x_clean.shape
        device = x_clean.device
        K = max(1, branch_factor)

        # One independent proposal per branch, all from the same parents.
        parents = x_clean.unsqueeze(0).expand(K, B, L)
        mask = torch.rand(K, B, L, device=device) < self.noise_fraction
        redrawn = torch.randint(0, 4, (K, B, L), device=device, dtype=x_clean.dtype)
        out = torch.where(mask, redrawn, parents)
        return out[0] if branch_factor == 1 else out


class ToyOracle:
    """Stand-in for a sequence-to-function model.

    The contract: score an (B, L) index tensor and return (rewards, gc), both
    float arrays of length B. GPA only ever ranks and exponentiates `rewards`,
    so any monotone scale works. `gc` is used for reporting and the optional GC
    controls; return zeros if it is meaningless for your alphabet.
    """

    def __init__(self, motif: str = MOTIF):
        self.motif = np.array([BASES.index(c) for c in motif], dtype=np.int64)

    def score(self, sequences_tensor):
        x = sequences_tensor.detach().cpu().numpy()
        B, L = x.shape
        m = len(self.motif)
        # Count motif occurrences by sliding comparison.
        windows = np.lib.stride_tricks.sliding_window_view(x, m, axis=1)
        hits = (windows == self.motif).all(axis=2).sum(axis=1).astype(np.float64)
        gc = np.isin(x, [1, 2]).mean(axis=1).astype(np.float64)
        return hits, gc

    def __call__(self, sequences_tensor):
        return self.score(sequences_tensor)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # Start from random sequences. In a real run, initialise from samples of
    # your generator (or from natural seeds via --seed_pool in the drivers).
    population = torch.from_numpy(
        rng.integers(0, 4, size=(POP_SIZE, SEQ_LEN))).long().to(device)
    labels = torch.zeros(POP_SIZE, 1, device=device)   # unconditional

    generator = ToyGenerator(noise_fraction=0.05)
    oracle = ToyOracle()

    before, _ = oracle.score(population)
    print(f"[init]  mean motif hits {before.mean():.3f}   max {before.max():.0f}")

    gpa = DiffusionPopulationAnnealer(
        mutate_fn=generator,
        oracle_fn=oracle,
        model=None,              # ToyGenerator holds no model
        device=device,
        branch_factor=4,         # K: propose 4 children per parent, keep the best
    )

    (population, scores, log_weights, history,
     best_population, best_scores, _, _) = gpa.run(
        population=population,
        labels=labels,
        max_beta=10.0,           # target tilt; the single activity/fidelity knob
        ess_threshold=0.5,       # alpha: resample when ESS drops below 0.5 N
        max_steps=30,
        mutation_batch_size=256,
    )

    after, _ = oracle.score(best_population)
    print(f"[final] mean motif hits {after.mean():.3f}   max {after.max():.0f}")
    print(f"[final] beta reached {history.beta[-1]:.2f} in {len(history.beta)} steps")
    print(f"[final] unique sequences {len(np.unique(best_population.cpu().numpy(), axis=0))}"
          f" / {POP_SIZE}")


if __name__ == "__main__":
    main()
