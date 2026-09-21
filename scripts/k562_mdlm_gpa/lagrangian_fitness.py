"""Self-tuning multi-objective fitness via Lagrangian multipliers.

Inspired by MOG-DFM (arxiv 2505.07086): instead of choosing fixed omega weights
for multi-objective resampling (which requires sweeps), use Lagrangian multipliers
that self-tune based on constraint violations at each GPA step.

Primary objective (oracle) is maximized freely via rank-based fitness.
Constraints (GC balance, etc.) get self-tuning lambda multipliers that activate
only when the population violates the specified tolerance.

Usage:
    lagrangian = LagrangianFitness(gc_target=0.50, gc_tolerance=0.05, lr=0.1)
    # Pass as fitness_fn to DiffusionPopulationAnnealer
    gpa = DiffusionPopulationAnnealer(..., fitness_fn=lagrangian, ...)
"""

import numpy as np
from scipy.stats import rankdata


class LagrangianFitness:
    """Self-tuning multi-objective fitness via Lagrangian dual ascent.

    At each GPA step:
    1. Compute GC constraint violation: |gc_mean - target| - tolerance
    2. Update lambda_gc via dual ascent: lambda += lr * violation (clamped >= 0)
    3. Rank-normalize oracle scores and GC deviations (scale-independent)
    4. Return combined fitness: rank(oracle) - lambda_gc * rank(gc_deviation)

    When GC is within tolerance, lambda decreases toward 0 and oracle dominates.
    When GC drifts outside tolerance, lambda increases and GC correction activates.
    At convergence, lambda settles to exactly the weight needed to maintain the
    constraint — no manual omega tuning required.

    Args:
        gc_target: Target GC fraction (default: 0.50).
        gc_tolerance: Acceptable deviation from target (default: 0.05, i.e. ±5%).
        lr: Learning rate for lambda dual update (default: 0.1).
            Affects convergence speed, not final solution.
    """

    def __init__(self, gc_target=0.50, gc_tolerance=0.05, lr=0.1):
        self.gc_target = gc_target
        self.gc_tolerance = gc_tolerance
        self.lr = lr

        # Lagrange multiplier (self-tuning)
        self.lambda_gc = 0.0

        # Tracking
        self.step = 0
        self.history = []

    def __call__(self, oracle_scores, gc_fractions):
        """Compute Lagrangian fitness for the population.

        Args:
            oracle_scores: (N,) numpy array of oracle predictions.
            gc_fractions: (N,) numpy array of GC content per sequence.

        Returns:
            (N,) numpy array of fitness values (rank-based, higher is better).
        """
        N = len(oracle_scores)

        # 1. Compute constraint violation (population-level)
        gc_mean = float(gc_fractions.mean())
        gc_violation = abs(gc_mean - self.gc_target) - self.gc_tolerance

        # 2. Dual ascent: update lambda
        self.lambda_gc = max(0.0, self.lambda_gc + self.lr * gc_violation)

        # 3. Rank-normalize objectives (scale-independent, in (0, 1])
        oracle_ranks = rankdata(oracle_scores) / N  # higher oracle = higher rank
        gc_dev = np.abs(gc_fractions - self.gc_target)
        gc_dev_ranks = rankdata(gc_dev) / N  # higher deviation = higher rank (bad)

        # 4. Combined fitness: maximize oracle, penalize GC deviation
        fitness = oracle_ranks - self.lambda_gc * gc_dev_ranks

        # 5. Log for analysis
        self.history.append({
            'step': self.step,
            'lambda_gc': round(self.lambda_gc, 6),
            'gc_mean': round(gc_mean, 4),
            'gc_violation': round(gc_violation, 4),
            'oracle_mean': round(float(oracle_scores.mean()), 4),
            'oracle_max': round(float(oracle_scores.max()), 4),
            'fitness_mean': round(float(fitness.mean()), 4),
        })
        self.step += 1

        return fitness

    def get_summary(self):
        """Return summary string for logging."""
        if not self.history:
            return "LagrangianFitness: no steps yet"
        last = self.history[-1]
        return (f"LagrangianFitness: step={last['step']}, "
                f"lambda_gc={last['lambda_gc']:.4f}, "
                f"gc_mean={last['gc_mean']:.3f}, "
                f"violation={last['gc_violation']:.4f}")
