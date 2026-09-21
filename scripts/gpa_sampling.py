#!/usr/bin/env python3
"""
Diffusion Population Annealing (GPA) — Population Annealing SMC with
CFG-guided discrete diffusion mutations.

Targets the reward-tilted distribution:
    π_β(x) = p_θ(x) · exp(β · r(x)) / Z(β)

Algorithm:
    Maintain N weighted particles. Gradually increase β from 0 to β_max.
    At each step:
        1. REWEIGHT:  log w_i += Δβ · oracle(x_i)
        2. RESAMPLE:  systematic resampling (always, standard PA)
        3. MUTATE:    noise + CFG denoise each particle (warm-start sampler)

The mutation kernel uses existing get_cfg_warm_start_sampler() from
scripts/cfg_sampling.py — CFG provides directional mutations toward
high-activity regions, making IS much more efficient than undirected
denoising.

Free energy estimate (byproduct):
    log Z(β_max) = Σ_k log(mean(w_k))
"""

import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as scipy_stats
import torch


def _fast_kmer_corr(indices_np, ref_profile, k=3):
    """Per-sequence Pearson r vs reference k-mer profile.

    Args:
        indices_np: (N, L) array of base indices {0,1,2,3}
        ref_profile: (4^k,) reference k-mer frequency profile
        k: k-mer length

    Returns:
        (N,) array of Pearson correlations
    """
    from bio_plausibility import kmer_profile
    profiles = kmer_profile(indices_np, k=k)  # (N, 4^k)
    # Vectorized Pearson correlation
    ref = ref_profile - ref_profile.mean()
    prof = profiles - profiles.mean(axis=1, keepdims=True)
    numer = prof @ ref
    denom = np.sqrt((ref ** 2).sum()) * np.sqrt((prof ** 2).sum(axis=1))
    denom[denom == 0] = 1.0
    return numer / denom


def _load_gosai_ref_kmer_profile(gosai_csv, target_cell='hepg2', top_frac=0.1, k=3):
    """Load Gosai high-activity sequences and compute pool-level k-mer profile.

    Args:
        gosai_csv: Path to gosai_all.csv
        target_cell: Column to rank by ('hepg2', 'k562', 'sknsh')
        top_frac: Fraction of top-activity sequences to use (default 10%)
        k: k-mer length

    Returns:
        (4^k,) mean k-mer frequency profile of high-activity real sequences
    """
    import pandas as pd
    from bio_plausibility import kmer_profile

    df = pd.read_csv(gosai_csv)
    # Select top fraction by activity
    n_top = max(1, int(len(df) * top_frac))
    top_df = df.nlargest(n_top, target_cell)
    # Convert sequences to indices
    dna_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    indices = np.array([[dna_map[c] for c in s] for s in top_df['seq']], dtype=np.int64)
    profiles = kmer_profile(indices, k=k)  # (n_top, 4^k)
    ref = profiles.mean(axis=0)  # (4^k,)
    print(f"  [K-mer Ref] Loaded {len(indices)} Gosai top-{top_frac*100:.0f}% "
          f"{target_cell} sequences, {4**k} {k}-mers")
    return ref


@dataclass
class GPAHistory:
    """Tracks diagnostics across GPA steps."""
    beta: List[float] = field(default_factory=list)
    delta_beta: List[float] = field(default_factory=list)
    ess: List[float] = field(default_factory=list)
    ess_fraction: List[float] = field(default_factory=list)
    resampled: List[bool] = field(default_factory=list)
    log_z_increments: List[float] = field(default_factory=list)
    oracle_mean: List[float] = field(default_factory=list)
    oracle_max: List[float] = field(default_factory=list)
    oracle_std: List[float] = field(default_factory=list)
    oracle_p99: List[float] = field(default_factory=list)
    pw_identity: List[float] = field(default_factory=list)
    gc_mean: List[float] = field(default_factory=list)
    n_unique: List[int] = field(default_factory=list)
    wall_time: List[float] = field(default_factory=list)
    eval_checkpoint_means: List[float] = field(default_factory=list)
    eval_checkpoint_steps: List[int] = field(default_factory=list)
    edit_dist_mean: List[float] = field(default_factory=list)
    edit_dist_max: List[float] = field(default_factory=list)
    edit_efficiency: List[float] = field(default_factory=list)
    budget_reject_frac: List[float] = field(default_factory=list)
    step_delta_mean: List[float] = field(default_factory=list)
    step_delta_max: List[float] = field(default_factory=list)
    kmer_r_mean: List[float] = field(default_factory=list)
    # Per-step archive: sequences exceeding a high eval-oracle threshold
    archive_seqs: List[np.ndarray] = field(default_factory=list)    # list of (K, L) int arrays
    archive_scores: List[np.ndarray] = field(default_factory=list)   # list of (K,) float64 arrays
    archive_steps: List[np.ndarray] = field(default_factory=list)    # list of (K,) int32 arrays
    # DNA-CRAFT G*-style spec-bounded archive (design A): top-N by per-seq
    # specificity (target − max off-target) streamed across eval checkpoints
    # with eviction. Final size = capacity (or fewer if total < capacity).
    mingap_archive_seqs: Optional[np.ndarray] = None        # (cap, L) int
    mingap_archive_specs: Optional[np.ndarray] = None       # (cap,) float
    mingap_archive_cell_scores: Optional[Dict[str, np.ndarray]] = None  # {cell: (cap,)}
    mingap_archive_steps: Optional[np.ndarray] = None       # (cap,) int — step admitted
    # Per-step top-K-by-spec accumulated archive (design B): at each eval
    # checkpoint take the top-K of that step's population by spec, accumulate
    # across all checkpoints with no eviction. Final size = T_ckpt × K (less
    # if any step's pop was smaller than K).
    perstep_archive_seqs: Optional[np.ndarray] = None       # (M, L) int
    perstep_archive_specs: Optional[np.ndarray] = None      # (M,) float
    perstep_archive_cell_scores: Optional[Dict[str, np.ndarray]] = None  # {cell: (M,)}
    perstep_archive_steps: Optional[np.ndarray] = None      # (M,) int — source step
    # Budget-capped streaming archive (Option B/C): top-N by composite rank
    # across all eval checkpoints with online eviction. Final size = capacity.
    # Option B signal: rank(eval) + rank(3mer_corr)
    # Option C signal: rank(eval) + rank(3mer_corr) + rank(app_ll)
    budget_archive_seqs: Optional[np.ndarray] = None        # (cap, L) int
    budget_archive_eval: Optional[np.ndarray] = None        # (cap,) float — eval_score at admit
    budget_archive_kmer_corr: Optional[np.ndarray] = None   # (cap,) float — 3mer corr with ref
    budget_archive_app_ll: Optional[np.ndarray] = None      # (cap,) float — app_ll at admit (Option C)
    budget_archive_steps: Optional[np.ndarray] = None       # (cap,) int — step admitted
    # Top-K-by-AG evicting pool: capacity-bounded pool holding the highest
    # eval-oracle (AG) sequences seen at ANY eval checkpoint, deduped by exact
    # sequence (keep max AG), with online eviction of the weakest. Captures
    # transient high-AG sequences the end-of-run best_eval snapshot drops.
    # Final size = min(capacity, n_unique_seqs_evaluated).
    topk_ag_seqs: Optional[np.ndarray] = None               # (cap, L) int
    topk_ag_scores: Optional[np.ndarray] = None             # (cap,) float — eval (AG) score
    topk_ag_steps: Optional[np.ndarray] = None              # (cap,) int — step of max AG

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class DiffusionPopulationAnnealer:
    """Population Annealing SMC with CFG-guided discrete diffusion mutations.

    Args:
        mutate_fn: callable(model, x_clean, labels) -> x_new
            The existing warm-start CFG sampler. Takes (model, x_clean, labels)
            and returns mutated sequences (same shape).
        oracle_fn: callable(sequences_tensor) -> (np.array of scores, np.array of gc)
            Wraps score_with_oracle(). Takes index tensor, returns (scores, gc).
        model: The diffusion model (passed to mutate_fn).
        device: torch device.
    """

    def __init__(self, mutate_fn: Callable, oracle_fn: Callable,
                 model: torch.nn.Module, device: torch.device,
                 fitness_fn: Optional[Callable] = None,
                 mutate_fn_factory: Optional[Callable] = None,
                 default_nf: float = 0.05,
                 branch_factor: int = 1,
                 max_edit_frac: float = 1.0,
                 edit_select_mode: str = "efficiency",
                 branch_selection_tau: float = 0.0,
                 mcts_depth: int = 0,
                 mcts_iterations: int = 12,
                 mcts_c: float = 1.41,
                 mcts_select_mode: str = "ucb",
                 max_edits_per_step: int = 0,
                 hill_climb: bool = False,
                 hill_climb_budget: int = 0,
                 hill_climb_full_vocab: bool = False,
                 hill_climb_positions: int = 0,
                 oracle_fn_fast: Optional[Callable] = None,
                 edit_discount: bool = False,
                 edit_discount_mode: str = "linear",
                 edit_discount_lambda: float = 1.0,
                 pareto_edit_fitness: bool = False,
                 hc_kmer_weight: float = 0.0,
                 ref_kmer_profile: Optional[np.ndarray] = None,
                 # v7 optimization flags
                 cache_mutation_scores: bool = False,
                 entropy_hc: bool = False,
                 prefilter_topk: int = 0,
                 gpa_fitness_mode: str = "oracle",
                 pll_warmup_steps: int = 0,
                 diversity_lambda: float = 0.0,
                 surrogate_prefilter: bool = False,
                 apf_correction: bool = False,
                 dps_iw_correct: bool = False):
        self.mutate_fn = mutate_fn
        self.oracle_fn = oracle_fn
        self.model = model
        self.device = device
        self.fitness_fn = fitness_fn  # (oracle_scores, gc_fractions[, population_tensor]) -> fitness_scores
        # Check if fitness_fn accepts population_tensor kwarg
        self._fitness_fn_wants_population = False
        if fitness_fn is not None:
            import inspect
            sig = inspect.signature(fitness_fn)
            self._fitness_fn_wants_population = 'population_tensor' in sig.parameters
        self.mutate_fn_factory = mutate_fn_factory  # callable(nf, guidance_weight=None) -> mutate_fn
        self._default_nf = default_nf  # fallback nf when factory is used for w-only annealing
        self.branch_factor = branch_factor      # K=1 disables branching (may be overridden per-step when schedule is set)
        self._branch_factor_init = branch_factor   # static K, restored when no schedule active
        self.branch_factor_schedule = None      # set by run() if provided; None = static K
        self.max_edit_frac = max_edit_frac      # 1.0 = no cap
        self.edit_select_mode = edit_select_mode # "efficiency" or "oracle"
        self.branch_selection_tau = branch_selection_tau  # 0 = argmax, >0 = softmax sampling
        self.mcts_depth = mcts_depth            # 0=disabled, 2-3=MCTS
        self.mcts_iterations = mcts_iterations  # expansion budget per particle
        self.mcts_c = mcts_c                    # UCB exploration constant
        self.mcts_select_mode = mcts_select_mode  # "ucb" or "uniform" (BFS, target-preserved)
        self.max_edits_per_step = max_edits_per_step  # 0=disabled, per-step Hamming cap
        self.hill_climb = hill_climb            # per-position hill-climbing after K-branch
        self.hill_climb_budget = hill_climb_budget  # independent HC edit budget (0=disabled)
        self.hill_climb_full_vocab = hill_climb_full_vocab  # try all tokens at all positions
        self.hill_climb_positions = hill_climb_positions  # 0=all, >0=subsample N positions
        self.oracle_fn_fast = oracle_fn_fast    # lightweight oracle for HC (skips penalty)
        self.edit_discount = edit_discount      # edit-discount fitness modifier
        self.edit_discount_mode = edit_discount_mode  # "linear" or "log"
        self.edit_discount_lambda = edit_discount_lambda  # lambda for log mode
        self.pareto_edit_fitness = pareto_edit_fitness  # Pareto rank fitness (score vs edits)
        self.hc_kmer_weight = hc_kmer_weight          # k-mer composition penalty weight for HC
        self.ref_kmer_profile = ref_kmer_profile      # (4^k,) reference k-mer profile
        # v7 optimization flags
        self.cache_mutation_scores = cache_mutation_scores  # skip post-mutation re-score when K-branch knows winner
        self.entropy_hc = entropy_hc                        # entropy-weighted HC position sampling
        self.prefilter_topk = prefilter_topk                # MDLM pre-filter: score only top-J branches (0=off)
        self.gpa_fitness_mode = gpa_fitness_mode            # "oracle", "mdlm_pll", or "hybrid"
        self.pll_warmup_steps = pll_warmup_steps            # oracle-free for first N steps (hybrid mode)
        self.diversity_lambda = diversity_lambda            # conformity penalty (0=off)
        self.surrogate_prefilter = surrogate_prefilter      # online k-mer surrogate for branch pre-filtering
        self._surrogate_model = None                        # fitted Ridge regression
        self._surrogate_data_X = []                         # accumulated kmer profiles
        self._surrogate_data_y = []                         # accumulated oracle scores
        self._surrogate_fit_step = -1                       # last step surrogate was re-fit
        # SMC-soundness corrections (S5, S7)
        self.apf_correction = apf_correction                # APF reweight for K-branch τ-softmax (S5)
        self.dps_iw_correct = dps_iw_correct                # IW correction for DPS gradient-warped proposal (S7)
        self._last_apf_log_correction = None                # (N,) written by _mutate_population, consumed at reweight
        self._last_dps_iw_log_correction = None             # (N,) written by mutator when DPS active
        if self.apf_correction and self.branch_selection_tau <= 0:
            raise ValueError(
                "apf_correction=True requires branch_selection_tau > 0 — argmax (τ=0) "
                "is not APF-correctable by simple reweight (selection is deterministic, "
                "p_τ(k*) → 1, correction = 0). Use τ>0 for SMC-sound K-branch."
            )

    def _branch_factor_at_step(self, step: int, total_steps: int) -> int:
        """Resolve K(step) from self.branch_factor_schedule.

        schedule forms:
          None                                  → static K (self._branch_factor_init)
          list/tuple [K0, K1, ...]              → K[min(step, len-1)]
          "linear_KSTART_KEND"                  → round(K_start + (K_end-K_start)*t/(T-1))
          "step_K0_K1_..._Kn"                   → equal-width tiers over [0, T)
          "late_KSTART_KEND[_FRAC]"             → K_start until FRAC*T (default 0.8) then K_end
          "early_KSTART_KEND[_FRAC]"            → K_start for first FRAC*T (default 0.2) then K_end
        """
        sched = self.branch_factor_schedule
        if sched is None:
            return self._branch_factor_init
        if isinstance(sched, (list, tuple)):
            return int(sched[min(step, len(sched) - 1)])
        if not isinstance(sched, str):
            raise TypeError(f"branch_factor_schedule must be None, list, or str; got {type(sched).__name__}")
        parts = sched.split("_")
        kw = parts[0]
        T = max(total_steps, 1)
        progress = step / max(T - 1, 1)
        if kw == "linear":
            if len(parts) != 3:
                raise ValueError(f"linear schedule needs 'linear_KSTART_KEND', got {sched!r}")
            K_start, K_end = int(parts[1]), int(parts[2])
            return max(1, round(K_start * (1 - progress) + K_end * progress))
        if kw == "step":
            tiers = [int(p) for p in parts[1:]]
            if not tiers:
                raise ValueError(f"step schedule needs at least one K, got {sched!r}")
            i = min(int(progress * len(tiers)), len(tiers) - 1)
            return max(1, tiers[i])
        if kw in ("late", "early"):
            if len(parts) not in (3, 4):
                raise ValueError(f"{kw} schedule: 'late_KSTART_KEND[_FRAC_PERCENT]', got {sched!r}")
            K_start, K_end = int(parts[1]), int(parts[2])
            frac = float(parts[3]) / 100.0 if len(parts) == 4 else (0.8 if kw == "late" else 0.2)
            if kw == "late":
                return max(1, K_start if progress < frac else K_end)
            return max(1, K_start if progress < frac else K_end)
        raise ValueError(f"unknown branch_factor_schedule keyword {kw!r} in {sched!r}")

    def run(
        self,
        population: torch.Tensor,
        labels: torch.Tensor,
        max_beta: float = 5.0,
        ess_threshold: float = 0.5,
        max_steps: int = 50,
        min_delta_beta: float = 1e-4,
        bio_filter_fn: Optional[Callable] = None,
        mutation_batch_size: int = 512,
        diversity_subsample: int = 500,
        patience: int = 0,
        min_improvement: float = 0.01,
        extend_on_delta_beta: float = 0.0,
        extend_steps: int = 10,
        hard_max_steps: int = 200,
        start_beta: float = 0.0,
        nf_start: float = 0.0,
        nf_end: float = 0.0,
        activity_start: float = 0.0,
        activity_end: float = 0.0,
        activity_noise: float = 0.0,
        w_start: float = 0.0,
        w_end: float = 0.0,
        elite_fraction: float = 0.0,
        gc_bin_edges: Optional[np.ndarray] = None,
        gc_bin_min_quota: int = 50,
        gc_bin_alloc_mode: str = "raw",
        gc_bin_alloc_alpha: float = 5.0,
        eval_oracle_fn: Optional[Callable] = None,
        eval_checkpoint_interval: int = 0,
        archive_threshold: Optional[float] = None,
        topk_ag_size: int = 0,
        mingap_archive_size: int = 0,
        perstep_archive_top_k: int = 0,
        eval_oracle_all_cells_fn: Optional[Callable] = None,
        mingap_target_cell: Optional[str] = None,
        max_delta_beta: float = 0.0,
        eta_start: float = 0.0,
        eta_end: float = 0.0,
        dedup_threshold: int = 0,
        dedup_nf_mult: float = 2.0,
        eval_early_stop_patience: int = 0,
        nf_boost_delta: float = 0.0,
        nf_boost_max: float = 0.0,
        step_callback: Optional[Callable] = None,
        resample_temperature: float = 1.0,
        max_copies: int = 0,
        island_configs: Optional[list] = None,
        mixture_nf: Optional[list] = None,
        rejuvenation_fraction: float = 0.0,
        rejuvenation_pool: Optional[torch.Tensor] = None,
        rejuvenation_nf: float = 0.0,
        rejuvenation_diverse: bool = False,
        pt_temperatures: Optional[list] = None,
        pt_swap_interval: int = 5,
        branch_factor_schedule: Optional[Union[list, str]] = None,
        budget_archive_size: int = 0,
        budget_archive_ref_kmer: Optional[np.ndarray] = None,
        budget_archive_use_appll: bool = False,
        budget_archive_mdlm_model: Optional[torch.nn.Module] = None,
        budget_archive_appll_samples: int = 3,
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, GPAHistory]:
        """Run the full Population Annealing loop.

        Args:
            population: (N, L) index tensor of initial particles.
            labels: (N, 1) activity labels for CFG conditioning.
            max_beta: Target inverse temperature.
            ess_threshold: Resample when ESS/N drops below this.
            max_steps: Maximum number of annealing steps.
            min_delta_beta: Stop if Δβ falls below this (distribution converged).
            bio_filter_fn: Optional callable(indices_np) -> bool mask.
                If provided, reject non-plausible mutations (keep parent).
            mutation_batch_size: Batch size for mutation step.
            diversity_subsample: Subsample size for pairwise identity (speed).
            patience: If >0, enable oracle plateau detection. Continue past
                max_steps as long as oracle mean improved by at least
                min_improvement within the last `patience` steps.
                Set to 0 to disable (default).
            min_improvement: Minimum oracle mean improvement over the patience
                window to justify continuing. Only used when patience > 0.
            extend_on_delta_beta: If >0, enable Δβ-based extension. When
                max_steps is reached but Δβ is still above this threshold,
                extend by extend_steps more steps. Set to 0 to disable (default).
            extend_steps: Number of steps to extend when Δβ threshold is met.
            hard_max_steps: Absolute upper bound on total steps (safety cap).
            start_beta: Starting β for SMC continuation. Set to the final β
                of a previous GPA run when resuming from its output. Default 0.
            nf_start: If >0, anneal noise_fraction from this value (high,
                exploratory) down to nf_end (low, fine-tuning) as β progresses.
                Requires mutate_fn_factory to be set. Default 0 (disabled).
            nf_end: Final noise_fraction when annealing. Only used when
                nf_start > 0.
            activity_start: If >0, anneal activity label from this value to
                activity_end as β progresses. Default 0 (disabled, use fixed labels).
            activity_end: Final activity label when annealing.
            activity_noise: If >0, add uniform noise ±this value to activity
                labels per particle before each mutation. Creates diverse mutation
                directions; resampling selects winners. Default 0 (disabled).
            w_start: If >0, anneal guidance weight from this value to w_end as
                β progresses. Requires mutate_fn_factory. Default 0 (disabled).
            w_end: Final guidance weight when annealing.
            elite_fraction: Fraction of top particles to preserve unchanged
                each mutation step. 0.0 = no elitism (default). E.g. 0.05
                preserves the top 5% by oracle score.
            gc_bin_edges: If provided, use oracle-proportional stratified
                resampling by GC bins instead of global resampling. Array of
                bin edges, e.g., [0.40, 0.42, ..., 0.60] for 10 bins of 2%.
                None = standard global resampling (default).
            gc_bin_min_quota: Minimum particles per GC bin (prevents bin
                extinction). Only used when gc_bin_edges is set. Default 50.
            eval_oracle_fn: Optional callback scoring population with eval
                oracle. Signature: (population_tensor) -> np.array of scores.
            eval_checkpoint_interval: If >0, call eval_oracle_fn every N GPA
                steps to track eval-oracle performance. Default 0 (disabled).
            max_delta_beta: If >0, cap Δβ per step to this value. Prevents
                beta from jumping to max_beta in one step when population has
                low diversity (e.g., single-seed init). Default 0 (disabled).
            nf_boost_delta: If >0, enable adaptive NF boosting. When eval oracle
                shows no improvement for eval_early_stop_patience checkpoints,
                boost NF by this amount instead of early-stopping. When NF
                reaches nf_boost_max, the next plateau triggers early stop.
                Default 0 (disabled).
            nf_boost_max: Maximum NF allowed when boosting. Default 0.

        Returns:
            population: (N, L) final particle population.
            oracle_scores: (N,) final oracle scores.
            log_weights: (N,) final log importance weights.
            history: GPAHistory with per-step diagnostics.
            best_population: (N, L) best-seen population by oracle mean.
            best_oracle_scores: (N,) oracle scores for best-seen population.
            best_eval_population: (N, L) best-seen population by eval oracle
                mean (None if eval checkpointing disabled).
            best_eval_scores: (N,) eval scores for best eval population
                (None if eval checkpointing disabled).
        """
        N = population.shape[0]
        history = GPAHistory()
        beta = start_beta
        log_weights = np.zeros(N, dtype=np.float64)
        log_z = 0.0  # cumulative free energy estimate
        # SMC-soundness correction buffers (S5 APF, S7 DPS IW). Size N.
        # Written at mutation time (by _mutate_population / mutator), consumed
        # and zeroed at the next reweight so they don't double-apply.
        self._last_apf_log_correction = np.zeros(N, dtype=np.float64) if self.apf_correction else None
        self._last_dps_iw_log_correction = np.zeros(N, dtype=np.float64) if self.dps_iw_correct else None

        # Adaptive continuation state
        current_step_limit = max_steps
        n_extensions = 0

        # K-annealing schedule (K(step) may differ from static branch_factor)
        self.branch_factor_schedule = branch_factor_schedule
        if branch_factor_schedule is not None:
            # Preview resolved schedule for operator visibility
            preview = [self._branch_factor_at_step(s, max_steps)
                       for s in range(min(max_steps, 20))]
            print(f"[GPA] branch_factor_schedule={branch_factor_schedule!r} "
                  f"→ K(t) preview (first {len(preview)} steps): {preview}")

        # Noise fraction annealing state
        nf_annealing = nf_start > 0 and self.mutate_fn_factory is not None

        # Activity annealing state
        act_annealing = activity_start > 0

        # Guidance weight annealing state
        w_annealing = w_start > 0 and self.mutate_fn_factory is not None

        # Eta annealing state
        eta_annealing = eta_start > 0 and self.mutate_fn_factory is not None

        # Adaptive NF boosting state
        nf_boost_active = (nf_boost_delta > 0 and nf_boost_max > 0
                           and self.mutate_fn_factory is not None)
        nf_boost_current = self._default_nf if nf_boost_active else 0.0

        # Need factory for any annealing (nf, w, or eta) or adaptive NF boost
        use_factory = nf_annealing or w_annealing or eta_annealing or nf_boost_active

        # Base labels for activity noise (before any annealing)
        base_activity = labels[0, 0].item() if labels is not None else 0.0

        # --- Island model setup ---
        use_islands = island_configs is not None and len(island_configs) > 0
        if use_islands:
            K_islands = len(island_configs)
            island_size = N // K_islands
            island_partition = [np.arange(k * island_size, min((k + 1) * island_size, N))
                                for k in range(K_islands)]
            print(f"[GPA] Island model: {K_islands} islands × {island_size} particles")
            for k, cfg in enumerate(island_configs):
                print(f"  Island {k}: {cfg.get('label', f'island_{k}')}")

        t_start_wall = time.time()

        # --- Helper: compute fitness (global or per-island) ---
        def _compute_island_fitness(pop, os, gc, island_cfgs=None, island_part=None):
            """Compute fitness per-island or globally."""
            if island_cfgs is not None and island_part is not None:
                fit = np.empty_like(os)
                for k, cfg in enumerate(island_cfgs):
                    idx = island_part[k]
                    isl_fn = cfg.get('fitness_fn', self.fitness_fn)
                    if isl_fn is not None:
                        import inspect
                        sig = inspect.signature(isl_fn)
                        if 'population_tensor' in sig.parameters:
                            fit[idx] = isl_fn(os[idx], gc[idx], population_tensor=pop[idx])
                        else:
                            fit[idx] = isl_fn(os[idx], gc[idx])
                    else:
                        fit[idx] = os[idx]
                return fit
            else:
                if self.fitness_fn is not None:
                    if self._fitness_fn_wants_population:
                        return self.fitness_fn(os, gc, population_tensor=pop)
                    else:
                        return self.fitness_fn(os, gc)
                return os

        # Initial oracle scoring
        print(f"\n[GPA] Scoring initial population (N={N:,})...")
        oracle_scores, gc_fractions = self.oracle_fn(population)
        fitness = _compute_island_fitness(
            population, oracle_scores, gc_fractions,
            island_configs if use_islands else None,
            island_partition if use_islands else None)
        if self.fitness_fn is not None or use_islands:
            print(f"  Oracle: mean={oracle_scores.mean():.3f}, "
                  f"max={oracle_scores.max():.3f}, std={oracle_scores.std():.3f}")
            print(f"  Fitness: mean={fitness.mean():.3f}, "
                  f"max={fitness.max():.3f}, std={fitness.std():.3f}")
        else:
            print(f"  Oracle: mean={oracle_scores.mean():.3f}, "
                  f"max={oracle_scores.max():.3f}, std={oracle_scores.std():.3f}")

        if start_beta > 0:
            print(f"  [Resume] Starting from β={start_beta:.4f} (SMC continuation)")
        if nf_annealing:
            print(f"  [NF Anneal] noise_fraction: {nf_start:.3f} → {nf_end:.3f} over β=[{start_beta:.1f}, {max_beta:.1f}]")
        if act_annealing:
            print(f"  [Act Anneal] activity: {activity_start:.2f} → {activity_end:.2f} over β=[{start_beta:.1f}, {max_beta:.1f}]")
        if activity_noise > 0:
            print(f"  [Act Noise] ±{activity_noise:.2f} per particle per step")
        if w_annealing:
            print(f"  [W Anneal] guidance_weight: {w_start:.2f} → {w_end:.2f} over β=[{start_beta:.1f}, {max_beta:.1f}]")
        if patience > 0:
            print(f"  [Adaptive] Oracle plateau detection: patience={patience}, "
                  f"min_improvement={min_improvement}")
        if elite_fraction > 0:
            n_elite = int(elite_fraction * N)
            print(f"  [Elitism] Preserving top {elite_fraction:.1%} = {n_elite} particles per step")
        if gc_bin_edges is not None:
            print(f"  [GC Stratified] {len(gc_bin_edges)-1} bins, edges={gc_bin_edges}, "
                  f"min_quota={gc_bin_min_quota}")
        if eval_oracle_fn is not None and eval_checkpoint_interval > 0:
            print(f"  [Eval Checkpoint] Every {eval_checkpoint_interval} steps")
        if eta_annealing:
            print(f"  [Eta Anneal] DPS eta: {eta_start:.0f} → {eta_end:.0f} over β=[{start_beta:.1f}, {max_beta:.1f}]")
        if dedup_threshold > 0:
            print(f"  [Dedup] Re-mutate near-duplicates: hamming < {dedup_threshold}, "
                  f"nf_mult={dedup_nf_mult:.1f}, max 2 rounds/step")
        if extend_on_delta_beta > 0:
            print(f"  [Adaptive] Δβ extension: threshold={extend_on_delta_beta}, "
                  f"extend_steps={extend_steps}, hard_max={hard_max_steps}")
        if max_delta_beta > 0:
            min_steps = int(np.ceil((max_beta - start_beta) / max_delta_beta))
            print(f"  [Beta Cap] max_delta_beta={max_delta_beta:.2f} "
                  f"(≥{min_steps} steps to reach max_beta)")
        if nf_boost_active:
            print(f"  [NF Boost] Adaptive: +{nf_boost_delta:.3f} on AG plateau "
                  f"(patience={eval_early_stop_patience}), max={nf_boost_max:.3f}, "
                  f"start={nf_boost_current:.3f}")
        if self.cache_mutation_scores:
            print(f"  [v7] Cache mutation scores: skip post-mutation re-score for K-branch")
        if self.entropy_hc:
            print(f"  [v7] Entropy-weighted HC position sampling")
        if self.prefilter_topk > 0:
            print(f"  [v7] MDLM pre-filter: score only top-{self.prefilter_topk} of K={self.branch_factor} branches")
        if self.gpa_fitness_mode != "oracle":
            print(f"  [v7] Fitness mode: {self.gpa_fitness_mode}")
            if self.gpa_fitness_mode == "hybrid":
                print(f"       Oracle-free for first {self.pll_warmup_steps} steps")
        if self.diversity_lambda > 0:
            print(f"  [v7] Diversity regularization: λ={self.diversity_lambda}")
        if self.surrogate_prefilter:
            print(f"  [v7] Online k-mer surrogate pre-filter (re-fit every 5 steps)")

        # EDTS (Edit-aware Diffusion Tree Search) state
        L = population.shape[1]
        original_population = population.clone()  # (N, L) — never mutated, tracks ancestry
        max_edits = int(self.max_edit_frac * L)
        if self.branch_factor > 1:
            print(f"  [EDTS] K={self.branch_factor}, max_edit_frac={self.max_edit_frac} "
                  f"({max_edits}/{L}bp), select={self.edit_select_mode}")
        elif self.max_edit_frac < 1.0:
            print(f"  [EDTS] edit cap: max_edit_frac={self.max_edit_frac} ({max_edits}/{L}bp)")
        if self.mcts_depth >= 2:
            print(f"  [MCTS] depth={self.mcts_depth}, iterations={self.mcts_iterations}, "
                  f"C={self.mcts_c}, K={self.branch_factor}")
        if self.max_edits_per_step > 0:
            print(f"  [Per-step cap] max_edits_per_step={self.max_edits_per_step}")
        if self.hill_climb and self.branch_factor > 1:
            print(f"  [Hill-Climb] Per-position greedy optimization from K={self.branch_factor} branches")
        if self.hill_climb_budget > 0:
            mode = "full-vocab" if self.hill_climb_full_vocab else "branch-token"
            print(f"  [Hill-Climb] budget={self.hill_climb_budget} edits/step ({mode}), independent of step_cap")
            if self.hill_climb_positions > 0:
                print(f"  [Hill-Climb] position subsample: {self.hill_climb_positions}/{L}")
            if self.oracle_fn_fast is not None:
                print(f"  [Hill-Climb] using fast oracle (target-only)")
        if self.edit_discount:
            if self.edit_discount_mode == "log":
                print(f"  [Edit-Discount] log mode, lambda={self.edit_discount_lambda}")
            else:
                print(f"  [Edit-Discount] linear mode: fitness *= (1 - edits/L)")
        if self.pareto_edit_fitness:
            print(f"  [Pareto] 2D Pareto rank fitness (score vs edits)")

        # Record initial state
        self._record_step(history, beta, 0.0, N, N, False, 0.0,
                          oracle_scores, gc_fractions, population,
                          diversity_subsample, time.time() - t_start_wall)

        # Track best-seen population (by oracle mean)
        best_oracle_mean = float(oracle_scores.mean())
        best_population = population.clone()
        best_oracle_scores = oracle_scores.copy()
        best_step = 0

        # Track best-seen population by eval oracle (if checkpointing enabled)
        best_eval_mean = -float('inf')
        best_eval_population = None
        best_eval_scores = None
        best_eval_step = 0
        eval_no_improve_count = 0

        # Spec-bounded mingap archive (DNA-CRAFT G*-style, design A). Tuple of
        #   (seqs (M, L) int8, specs (M,) f64, cells {cell: (M,) f64}, steps (M,) i32)
        # or None until first checkpoint. M ≤ mingap_archive_size at all times.
        mingap_archive_state = None
        # Per-step top-K accumulated archive (design B): list-of-chunks per
        # field, concatenated at run end. Lazy-init on first checkpoint.
        perstep_chunks = None
        need_spec_archives = mingap_archive_size > 0 or perstep_archive_top_k > 0
        if need_spec_archives:
            if eval_oracle_all_cells_fn is None:
                raise ValueError(
                    "spec-archive flags require eval_oracle_all_cells_fn")
            if mingap_target_cell is None:
                raise ValueError(
                    "spec-archive flags require mingap_target_cell")

        # Budget-capped streaming archive (Option B / C). Tuple of
        #   (seqs (M, L) int8, eval (M,) f64, kmer_corr (M,) f64,
        #    app_ll (M,) f64 or None, steps (M,) i32)
        # or None until first checkpoint. M ≤ budget_archive_size at all times.
        budget_archive_state = None
        # Top-K-by-AG evicting pool state. Lazy-init on first checkpoint;
        # tuple (seqs[M,L] int8, scores[M] float64, steps[M] int32), M ≤ topk_ag_size.
        topk_ag_state = None
        if budget_archive_size > 0:
            if budget_archive_ref_kmer is None:
                raise ValueError(
                    "budget_archive_size > 0 requires budget_archive_ref_kmer "
                    "(64-element reference 3mer profile)")
            if eval_oracle_fn is None:
                raise ValueError(
                    "budget_archive_size > 0 requires eval_oracle_fn for the "
                    "eval-checkpoint ranking signal")
            if budget_archive_use_appll and budget_archive_mdlm_model is None:
                raise ValueError(
                    "budget_archive_use_appll=True requires budget_archive_mdlm_model "
                    "(MDLM Diffusion instance with _forward_pass_diffusion)")
            # Validate reference: must be 64-vec (4^3 3-mer space)
            _bref = np.asarray(budget_archive_ref_kmer, dtype=np.float64)
            if _bref.shape != (64,):
                raise ValueError(
                    f"budget_archive_ref_kmer must be shape (64,); got {_bref.shape}")
            _bref_centered = _bref - _bref.mean()
            _bref_norm = np.sqrt((_bref_centered ** 2).sum())

        step = 0
        while step < current_step_limit:
            step_t0 = time.time()

            # --- K-annealing: resolve K(step) and override self.branch_factor ---
            # Everything downstream (_mutate_population, _mcts_mutate_batch, logging)
            # reads self.branch_factor; setting it here propagates without other changes.
            if self.branch_factor_schedule is not None:
                self.branch_factor = self._branch_factor_at_step(step, current_step_limit)

            # --- ADAPT β (uses fitness, not raw oracle) ---
            remaining = max_beta - beta
            if max_delta_beta > 0:
                remaining = min(remaining, max_delta_beta)
            delta_beta = self._adapt_beta(
                fitness, log_weights, ess_threshold, remaining)

            if delta_beta < min_delta_beta:
                print(f"\n[GPA] Step {step+1}: Δβ={delta_beta:.6f} < {min_delta_beta} — converged.")
                break

            beta += delta_beta

            # --- REWEIGHT (uses fitness, not raw oracle) ---
            # SMC-soundness corrections: carried over from the previous step's
            # mutation. S5 APF: -log p_τ(k*) for K-branch τ-softmax selection.
            # S7 DPS IW: log q_base - log q_DPS for gradient-warped proposal.
            if use_islands:
                for k_isl in range(K_islands):
                    idx = island_partition[k_isl]
                    log_weights[idx] += delta_beta * fitness[idx]
                    if self._last_apf_log_correction is not None:
                        log_weights[idx] -= self._last_apf_log_correction[idx]
                    if self._last_dps_iw_log_correction is not None:
                        log_weights[idx] += self._last_dps_iw_log_correction[idx]
                    log_weights[idx] -= log_weights[idx].max()
            else:
                log_weights += delta_beta * fitness
                if self._last_apf_log_correction is not None:
                    log_weights -= self._last_apf_log_correction
                if self._last_dps_iw_log_correction is not None:
                    log_weights += self._last_dps_iw_log_correction
                log_weights -= log_weights.max()
            # Zero the correction buffers now that they've been consumed.
            if self._last_apf_log_correction is not None:
                self._last_apf_log_correction.fill(0.0)
            if self._last_dps_iw_log_correction is not None:
                self._last_dps_iw_log_correction.fill(0.0)

            ess = self._compute_ess(log_weights)
            ess_frac = ess / N

            # Free energy increment: log(mean(w)) before normalization
            log_mean_w = self._log_mean_exp(delta_beta * fitness)
            log_z += log_mean_w

            # --- RESAMPLE (always — standard PA) ---
            if use_islands:
                # Per-island resampling: each island resamples independently
                indices = np.arange(N)
                for k_isl in range(K_islands):
                    idx = island_partition[k_isl]
                    n_k = len(idx)
                    # Per-island temperature for parallel tempering
                    isl_temp = pt_temperatures[k_isl] if pt_temperatures is not None else resample_temperature
                    lw_k = log_weights[idx] / isl_temp if isl_temp > 1.0 else log_weights[idx]
                    local_idx = self._systematic_resample(lw_k, n_k)
                    indices[idx] = idx[local_idx]
            else:
                # Tempered resampling: divide log_weights by T>1 to soften selection
                resample_lw = log_weights / resample_temperature if resample_temperature > 1.0 else log_weights
                if gc_bin_edges is not None:
                    indices = self._stratified_resample(
                        resample_lw, N, gc_fractions, gc_bin_edges,
                        oracle_scores, min_quota=gc_bin_min_quota,
                        alloc_mode=gc_bin_alloc_mode, alloc_alpha=gc_bin_alloc_alpha)
                else:
                    indices = self._systematic_resample(resample_lw, N)
            # Copy cap: limit max duplicates per particle
            if max_copies > 0:
                unique_idx, counts = np.unique(indices, return_counts=True)
                excess = (counts - max_copies).clip(min=0).sum()
                if excess > 0:
                    capped_indices = []
                    for u, c in zip(unique_idx, counts):
                        capped_indices.extend([u] * min(c, max_copies))
                    deficit = N - len(capped_indices)
                    if deficit > 0:
                        # Fill from low-copy particles proportional to weight
                        low_mask = counts < max_copies
                        if low_mask.any():
                            fill_pool = unique_idx[low_mask]
                            fw = np.exp(log_weights[fill_pool] - log_weights[fill_pool].max())
                            fw /= fw.sum()
                            fill = np.random.choice(fill_pool, deficit, replace=True, p=fw)
                            capped_indices.extend(fill.tolist())
                        else:
                            # All at cap — fill randomly from existing
                            fill = np.random.choice(unique_idx, deficit, replace=True)
                            capped_indices.extend(fill.tolist())
                    indices = np.array(capped_indices[:N])
            population = population[indices]
            original_population = original_population[indices]  # ancestry tracks through resampling
            oracle_scores = oracle_scores[indices]
            gc_fractions = gc_fractions[indices]
            fitness = fitness[indices]
            log_weights = np.zeros(N, dtype=np.float64)
            did_resample = True

            # --- REJUVENATION: replace lowest-fitness particles ---
            n_rejuv_actual = 0
            if rejuvenation_fraction > 0 and step > 0:
                n_rejuv = max(1, int(rejuvenation_fraction * N))
                replace_idx = np.argsort(fitness)[:n_rejuv]

                # Choose source for fresh particles
                if rejuvenation_diverse:
                    # Adaptive SMC: source from most diverse current particles
                    pop_arr = population.numpy()
                    n_sub = min(500, N)
                    subsample = np.random.choice(N, n_sub, replace=False)
                    sub_arr = pop_arr[subsample]
                    mean_dist = np.array([
                        np.mean(np.sum(pop_arr[i] != sub_arr, axis=1))
                        for i in range(N)
                    ])
                    source_idx = np.argsort(mean_dist)[::-1][:n_rejuv].copy()
                    fresh = population[source_idx].clone()
                elif rejuvenation_pool is not None:
                    # Prior refreshing from seed pool
                    pool_idx = np.random.choice(len(rejuvenation_pool), n_rejuv, replace=True)
                    fresh = rejuvenation_pool[pool_idx].clone()
                else:
                    # Prior refreshing from uniform random
                    fresh = torch.randint(0, 4, (n_rejuv, population.shape[1]))

                population[replace_idx] = fresh
                original_population[replace_idx] = fresh

                # Optional: boost fresh particles with high-NF mutation
                if rejuvenation_nf > 0 and self.mutate_fn_factory is not None:
                    boost_fn = self.mutate_fn_factory(rejuvenation_nf)
                    fresh_batch = population[replace_idx].to(self.device)
                    labels_batch = torch.zeros(n_rejuv, 1, device=self.device)
                    boosted = boost_fn(self.model, fresh_batch, labels_batch)
                    if isinstance(boosted, tuple):
                        boosted = boosted[0]
                    population[replace_idx] = boosted.cpu()

                n_rejuv_actual = n_rejuv

            # --- PARALLEL TEMPERING: swap particles between adjacent replicas ---
            n_pt_swaps = 0
            if pt_temperatures is not None and use_islands and step % pt_swap_interval == 0:
                for k in range(K_islands - 1):
                    idx_cold = island_partition[k]
                    idx_hot = island_partition[k + 1]
                    i_cold = np.random.choice(idx_cold)
                    i_hot = np.random.choice(idx_hot)
                    T_cold = pt_temperatures[k]
                    T_hot = pt_temperatures[k + 1]
                    delta_f = fitness[i_hot] - fitness[i_cold]
                    # Acceptance: exp((beta/T_cold - beta/T_hot) * delta_f)
                    log_alpha = (beta / T_cold - beta / T_hot) * delta_f
                    if np.log(np.random.random() + 1e-30) < log_alpha:
                        population[[i_cold, i_hot]] = population[[i_hot, i_cold]]
                        oracle_scores[[i_cold, i_hot]] = oracle_scores[[i_hot, i_cold]]
                        gc_fractions[[i_cold, i_hot]] = gc_fractions[[i_hot, i_cold]]
                        fitness[[i_cold, i_hot]] = fitness[[i_hot, i_cold]]
                        n_pt_swaps += 1

            # Print step header BEFORE mutation (so user sees progress while waiting)
            extra = ""
            if n_rejuv_actual > 0:
                extra += f" rejuv={n_rejuv_actual}"
            if n_pt_swaps > 0:
                extra += f" swaps={n_pt_swaps}"
            k_log = f" K={self.branch_factor}" if self.branch_factor_schedule is not None else ""
            print(f"[GPA] Step {step+1:>3}: β={beta:.4f} Δβ={delta_beta:.4f} "
                  f"ESS={ess_frac:.3f}{k_log}{extra}", end='', flush=True)
            if gc_bin_edges is not None:
                bin_asn = np.digitize(gc_fractions, gc_bin_edges) - 1
                bin_asn = np.clip(bin_asn, 0, len(gc_bin_edges) - 2)
                n_bins = len(gc_bin_edges) - 1
                counts = [str((bin_asn == b).sum()) for b in range(n_bins)]
                print(f"  bins:[{'|'.join(counts)}]", end='', flush=True)
            print(f"  mutating...", end='', flush=True)

            # --- ACTIVITY ANNEALING (update labels before mutation) ---
            beta_range = max_beta - start_beta
            progress = (beta - start_beta) / beta_range if beta_range > 0 else 1.0
            current_act = None
            if act_annealing and labels is not None:
                current_act = activity_start + (activity_end - activity_start) * progress
                labels = torch.full_like(labels, current_act)

            # Per-particle activity noise
            if activity_noise > 0 and labels is not None:
                act_base = current_act if current_act is not None else base_activity
                noise_val = torch.empty_like(labels).uniform_(-activity_noise, activity_noise)
                mutation_labels = torch.full_like(labels, act_base) + noise_val
            else:
                mutation_labels = labels

            # --- MUTATE (with optional nf/w/eta annealing via factory) ---
            population_before = population.clone() if (self.max_edits_per_step > 0 or self.hill_climb_budget > 0) else None
            current_nf = None
            current_w = None
            current_eta = None
            if use_factory:
                current_nf = (nf_start + (nf_end - nf_start) * progress) if nf_annealing else None
                # Adaptive NF boost overrides default NF when not annealing
                if nf_boost_active and current_nf is None:
                    current_nf = nf_boost_current
                current_w = (w_start + (w_end - w_start) * progress) if w_annealing else None
                current_eta = (eta_start + (eta_end - eta_start) * progress) if eta_annealing else None
                factory_nf = current_nf if current_nf is not None else self._default_nf
                current_mutate_fn = self.mutate_fn_factory(factory_nf, guidance_weight=current_w, eta=current_eta)

                # --- Mixture proposal kernel (Option C) ---
                if mixture_nf is not None and self.mutate_fn_factory is not None:
                    K_mix = len(mixture_nf)
                    assignments = np.random.randint(0, K_mix, N)
                    mutated_pop = population.clone()
                    cached_scores_mix = None
                    for k_mix in range(K_mix):
                        mask = assignments == k_mix
                        n_mix = int(mask.sum())
                        if n_mix == 0:
                            continue
                        mix_fn = self.mutate_fn_factory(mixture_nf[k_mix], guidance_weight=current_w, eta=current_eta)
                        mix_result = self._mutate_population(
                            population[mask], mutation_labels[:n_mix], mutation_batch_size,
                            bio_filter_fn, oracle_scores[mask.nonzero()[0]],
                            mutate_fn_override=mix_fn, elite_fraction=elite_fraction,
                            original_population=original_population[mask],
                            max_edits=max_edits, max_edits_per_step=self.max_edits_per_step)
                        if isinstance(mix_result, tuple):
                            mutated_pop[mask] = mix_result[0]
                        else:
                            mutated_pop[mask] = mix_result
                    mutate_result = mutated_pop
                    cached_scores = None  # invalidate cache for mixture
                else:
                    mutate_result = self._mutate_population(
                        population, mutation_labels, mutation_batch_size, bio_filter_fn,
                        oracle_scores, mutate_fn_override=current_mutate_fn,
                        elite_fraction=elite_fraction,
                        original_population=original_population,
                        max_edits=max_edits,
                        max_edits_per_step=self.max_edits_per_step)
            else:
                mutate_result = self._mutate_population(
                    population, mutation_labels, mutation_batch_size, bio_filter_fn,
                    oracle_scores, elite_fraction=elite_fraction,
                    original_population=original_population,
                    max_edits=max_edits,
                    max_edits_per_step=self.max_edits_per_step)

            # Unpack cached scores if returned (1A: cache_mutation_scores)
            if isinstance(mutate_result, tuple):
                population, cached_scores, cached_gc = mutate_result
            else:
                population = mutate_result
                cached_scores = None

            # --- DIVERSITY: Re-mutate near-duplicates ---
            if dedup_threshold > 0:
                n_remutated = self._detect_and_remutate_duplicates(
                    population, mutation_labels, mutation_batch_size,
                    bio_filter_fn, dedup_threshold, dedup_nf_mult,
                    current_nf or self._default_nf,
                    mutate_fn_override=(current_mutate_fn if use_factory else None),
                )
                if n_remutated > 0:
                    # Invalidate cached scores since some particles changed
                    cached_scores = None

            # --- RE-SCORE (skip if cache_mutation_scores and scores are cached) ---
            use_pll_fitness = (self.gpa_fitness_mode == "mdlm_pll" or
                               (self.gpa_fitness_mode == "hybrid" and step < self.pll_warmup_steps))
            if cached_scores is not None and not use_pll_fitness:
                oracle_scores = cached_scores
                gc_fractions = cached_gc
                fitness = _compute_island_fitness(
                    population, oracle_scores, gc_fractions,
                    island_configs if use_islands else None,
                    island_partition if use_islands else None)
            elif use_pll_fitness:
                # 3A: Use MDLM PLL as fitness (oracle-free)
                oracle_scores, gc_fractions = self.oracle_fn(population)
                pll_scores = self._compute_pll_fitness(population)
                fitness = pll_scores
            else:
                oracle_scores, gc_fractions = self.oracle_fn(population)
                fitness = _compute_island_fitness(
                    population, oracle_scores, gc_fractions,
                    island_configs if use_islands else None,
                    island_partition if use_islands else None)

            # --- DIVERSITY REGULARIZATION (3B) ---
            if self.diversity_lambda > 0:
                if use_islands:
                    for k_isl in range(K_islands):
                        idx = island_partition[k_isl]
                        conformity_k = self._compute_conformity(population[idx])
                        fitness[idx] = fitness[idx] - self.diversity_lambda * conformity_k
                else:
                    conformity = self._compute_conformity(population)
                    fitness = fitness - self.diversity_lambda * conformity

            # --- SURROGATE DATA COLLECTION (3C) ---
            if self.surrogate_prefilter:
                self._accumulate_surrogate_data(population, oracle_scores, step)

            # --- EDIT-DISCOUNT / PARETO FITNESS MODIFIER ---
            edit_dists_int = (population != original_population).sum(dim=1)
            if self.pareto_edit_fitness:
                fitness = self._pareto_rank_fitness(fitness, edit_dists_int.float().numpy())
            elif self.edit_discount:
                ed = edit_dists_int.float().numpy()
                if self.edit_discount_mode == "log":
                    fitness = fitness - self.edit_discount_lambda * np.log1p(ed)
                else:  # linear
                    fitness = fitness * (1.0 - ed / L)

            # --- EDIT DISTANCE DIAGNOSTICS ---
            edit_dists = edit_dists_int.float()
            history.edit_dist_mean.append(float(edit_dists.mean()))
            history.edit_dist_max.append(float(edit_dists.max()))
            history.edit_efficiency.append(
                float(oracle_scores.mean()) / (1.0 + float(edit_dists.mean())))

            # --- STEP DELTA DIAGNOSTICS ---
            if population_before is not None:
                step_deltas = (population != population_before).sum(dim=1).float()
                history.step_delta_mean.append(float(step_deltas.mean()))
                history.step_delta_max.append(float(step_deltas.max()))
            else:
                step_deltas = None

            # --- TRACK BEST ---
            current_mean = float(oracle_scores.mean())
            if current_mean > best_oracle_mean:
                best_oracle_mean = current_mean
                best_population = population.clone()
                best_oracle_scores = oracle_scores.copy()
                best_step = step + 1

            # --- DIAGNOSTICS ---
            elapsed = time.time() - t_start_wall
            self._record_step(history, beta, delta_beta,
                              self._compute_ess(log_weights), N,
                              did_resample, log_mean_w,
                              oracle_scores, gc_fractions, population,
                              diversity_subsample, elapsed)

            # --- EVAL CHECKPOINT ---
            if (eval_oracle_fn is not None and eval_checkpoint_interval > 0
                    and (step + 1) % eval_checkpoint_interval == 0):
                eval_scores = eval_oracle_fn(population)
                eval_mean = float(eval_scores.mean())
                history.eval_checkpoint_means.append(eval_mean)
                history.eval_checkpoint_steps.append(step + 1)
                if eval_mean > best_eval_mean:
                    best_eval_mean = eval_mean
                    best_eval_population = population.clone()
                    best_eval_scores = eval_scores.copy()
                    best_eval_step = step + 1
                    eval_no_improve_count = 0
                else:
                    eval_no_improve_count += 1
                # Per-step archive: collect sequences exceeding high threshold
                if archive_threshold is not None:
                    mask = eval_scores > archive_threshold
                    if mask.any():
                        history.archive_seqs.append(population[mask].cpu().numpy())
                        history.archive_scores.append(eval_scores[mask].copy())
                        history.archive_steps.append(
                            np.full(int(mask.sum()), step + 1, dtype=np.int32))
                # Top-K-by-AG evicting pool: merge this checkpoint's full
                # population (no threshold floor), dedup by exact sequence keeping
                # the max AG seen, then keep the top `topk_ag_size` by AG.
                if topk_ag_size > 0:
                    pop_now = population.cpu().numpy().astype(np.int8)
                    eval_now = np.asarray(eval_scores, dtype=np.float64)
                    steps_now = np.full(len(pop_now), step + 1, dtype=np.int32)
                    if topk_ag_state is None:
                        t_seqs, t_scores, t_steps = pop_now, eval_now, steps_now
                    else:
                        p_seqs, p_scores, p_steps = topk_ag_state
                        t_seqs = np.concatenate([p_seqs, pop_now], axis=0)
                        t_scores = np.concatenate([p_scores, eval_now])
                        t_steps = np.concatenate([p_steps, steps_now])
                    # Dedup by exact sequence, keeping the row with max AG.
                    uniq, inv = np.unique(t_seqs, axis=0, return_inverse=True)
                    inv = inv.ravel()
                    best = np.full(len(uniq), -np.inf)
                    np.maximum.at(best, inv, t_scores)
                    # Recover the step of each unique seq's max-AG occurrence.
                    order_by_score = np.argsort(t_scores)  # ascending
                    u_steps = np.empty(len(uniq), dtype=np.int32)
                    u_steps[inv[order_by_score]] = t_steps[order_by_score]
                    t_seqs, t_scores, t_steps = uniq, best, u_steps
                    if len(t_seqs) > topk_ag_size:
                        keep = np.argpartition(t_scores, -topk_ag_size)[-topk_ag_size:]
                        keep = keep[np.argsort(-t_scores[keep])].copy()
                        t_seqs, t_scores, t_steps = t_seqs[keep], t_scores[keep], t_steps[keep]
                    topk_ag_state = (t_seqs, t_scores, t_steps)
                # Spec-based archives (designs A and B share per-checkpoint
                # spec/cell-score computation; do it once, then dispatch).
                if need_spec_archives:
                    cells_now = eval_oracle_all_cells_fn(population)  # {cell: (N,)}
                    cells_now_np = {
                        k: np.asarray(v, dtype=np.float64)
                        for k, v in cells_now.items()
                    }
                    target_now = cells_now_np[mingap_target_cell]
                    off_now = np.stack(
                        [v for k, v in cells_now_np.items()
                         if k != mingap_target_cell],
                        axis=1)  # (N, |C|-1)
                    spec_now = target_now - off_now.max(axis=1)
                    seqs_now = population.cpu().numpy().astype(np.int8)
                    steps_now = np.full(seqs_now.shape[0], step + 1,
                                        dtype=np.int32)
                # Design A — DNA-CRAFT G* style: merge current population with
                # running archive, keep global top-N by spec with eviction.
                if mingap_archive_size > 0:
                    if mingap_archive_state is None:
                        c_seqs, c_specs = seqs_now, spec_now
                        c_steps, c_cells = steps_now, cells_now_np
                    else:
                        a_seqs, a_specs, a_cells, a_steps = mingap_archive_state
                        c_seqs = np.concatenate([a_seqs, seqs_now], axis=0)
                        c_specs = np.concatenate([a_specs, spec_now], axis=0)
                        c_steps = np.concatenate([a_steps, steps_now], axis=0)
                        c_cells = {
                            k: np.concatenate([a_cells[k], cells_now_np[k]],
                                              axis=0)
                            for k in cells_now_np
                        }
                    if c_specs.shape[0] > mingap_archive_size:
                        keep = np.argpartition(
                            c_specs, -mingap_archive_size
                        )[-mingap_archive_size:]
                        # Sort within for descending readability; .copy() keeps
                        # the array contiguous after the negative-stride sort.
                        keep = keep[np.argsort(-c_specs[keep])].copy()
                        c_seqs = c_seqs[keep]
                        c_specs = c_specs[keep]
                        c_steps = c_steps[keep]
                        c_cells = {k: c_cells[k][keep] for k in c_cells}
                    mingap_archive_state = (c_seqs, c_specs, c_cells, c_steps)
                # Design B — per-step top-K by spec, accumulated, no eviction.
                if perstep_archive_top_k > 0:
                    K = perstep_archive_top_k
                    if spec_now.shape[0] > K:
                        keep_b = np.argpartition(spec_now, -K)[-K:]
                    else:
                        keep_b = np.arange(spec_now.shape[0])
                    # Order each step's chunk by descending spec for
                    # readability; .copy() avoids a negative-stride view.
                    keep_b = keep_b[np.argsort(-spec_now[keep_b])].copy()
                    if perstep_chunks is None:
                        perstep_chunks = {
                            'seqs': [], 'specs': [], 'steps': [],
                            'cells': {k: [] for k in cells_now_np},
                        }
                    perstep_chunks['seqs'].append(seqs_now[keep_b])
                    perstep_chunks['specs'].append(spec_now[keep_b])
                    perstep_chunks['steps'].append(
                        np.full(len(keep_b), step + 1, dtype=np.int32))
                    for k in cells_now_np:
                        perstep_chunks['cells'][k].append(
                            cells_now_np[k][keep_b])
                # Budget-capped streaming archive (Option B / C): top-N by
                # composite rank across all eval checkpoints with eviction.
                # Option B (use_appll=False): rank(eval) + rank(3mer_corr_with_ref)
                # Option C (use_appll=True):  rank(eval) + rank(3mer_corr) + rank(app_ll)
                if budget_archive_size > 0:
                    pop_np = population.cpu().numpy().astype(np.int8)
                    Npop, Lpop = pop_np.shape
                    # Per-seq 3mer counts: (N, 64), using base codes A/C/G/T = 0..3
                    flat = (pop_np[:, :Lpop-2].astype(np.int64) * 16
                            + pop_np[:, 1:Lpop-1].astype(np.int64) * 4
                            + pop_np[:, 2:Lpop].astype(np.int64))
                    valid = ((pop_np[:, :Lpop-2] >= 0)
                             & (pop_np[:, 1:Lpop-1] >= 0)
                             & (pop_np[:, 2:Lpop] >= 0))
                    rows = np.repeat(np.arange(Npop, dtype=np.int64), Lpop - 2)
                    flat_r = flat.reshape(-1)
                    valid_r = valid.reshape(-1)
                    kmer_now = np.bincount(
                        rows[valid_r] * 64 + flat_r[valid_r],
                        minlength=Npop * 64
                    ).reshape(Npop, 64).astype(np.float64)
                    x = kmer_now - kmer_now.mean(axis=1, keepdims=True)
                    x_norm = np.sqrt((x ** 2).sum(axis=1)) + 1e-12
                    kmer_corr_now = (x * _bref_centered).sum(axis=1) / (
                        x_norm * (_bref_norm + 1e-12))
                    # Optional Option C: in-loop App-LL via MDLM forward
                    appll_now = None
                    if budget_archive_use_appll:
                        _t_appll = time.time()
                        mdlm = budget_archive_mdlm_model
                        mdlm.eval()
                        mdev = next(mdlm.parameters()).device
                        # token codes A/C/G/T match population's {0,1,2,3}
                        tokens = population.to(mdev).long()
                        nll = np.zeros(Npop, dtype=np.float64)
                        bs = 64
                        with torch.no_grad():
                            for s_b in range(0, Npop, bs):
                                e_b = min(s_b + bs, Npop)
                                batch = tokens[s_b:e_b]
                                losses = []
                                for _ in range(budget_archive_appll_samples):
                                    loss_pt = mdlm._forward_pass_diffusion(batch)
                                    losses.append(loss_pt.sum(-1).cpu().numpy())
                                nll[s_b:e_b] = np.stack(losses, axis=0).mean(axis=0)
                        appll_now = -nll  # higher = more likely under prior
                        print(f"  [budget_archive App-LL] {time.time()-_t_appll:.1f}s "
                              f"for {Npop} seqs x {budget_archive_appll_samples} t-samples",
                              end='', flush=True)
                    # Merge new with running archive
                    steps_now_arr = np.full(Npop, step + 1, dtype=np.int32)
                    eval_now_arr = np.asarray(eval_scores, dtype=np.float64)
                    if budget_archive_state is None:
                        c_seqs = pop_np
                        c_eval = eval_now_arr
                        c_kmer = kmer_corr_now
                        c_appll = appll_now if appll_now is not None else None
                        c_steps_bd = steps_now_arr
                    else:
                        b_seqs, b_eval, b_kmer, b_appll, b_steps_bd = budget_archive_state
                        c_seqs = np.concatenate([b_seqs, pop_np], axis=0)
                        c_eval = np.concatenate([b_eval, eval_now_arr])
                        c_kmer = np.concatenate([b_kmer, kmer_corr_now])
                        c_steps_bd = np.concatenate([b_steps_bd, steps_now_arr])
                        if budget_archive_use_appll:
                            c_appll = np.concatenate([b_appll, appll_now])
                        else:
                            c_appll = None
                    # Compose rank-sum signal
                    r_e = scipy_stats.rankdata(c_eval)
                    r_k = scipy_stats.rankdata(c_kmer)
                    signal = r_e + r_k
                    if budget_archive_use_appll:
                        r_a = scipy_stats.rankdata(c_appll)
                        signal = signal + r_a
                    if c_seqs.shape[0] > budget_archive_size:
                        keep = np.argpartition(
                            signal, -budget_archive_size
                        )[-budget_archive_size:]
                        keep = keep[np.argsort(-signal[keep])].copy()
                        c_seqs = c_seqs[keep]
                        c_eval = c_eval[keep]
                        c_kmer = c_kmer[keep]
                        c_steps_bd = c_steps_bd[keep]
                        if c_appll is not None:
                            c_appll = c_appll[keep]
                    budget_archive_state = (c_seqs, c_eval, c_kmer, c_appll, c_steps_bd)
                print(f"  [eval ckpt] eval_mean={eval_mean:.3f} "
                      f"(best={best_eval_mean:.3f} @ step {best_eval_step})",
                      end='', flush=True)
                # Adaptive NF boost OR early stop when AG eval plateaus
                if (eval_early_stop_patience > 0 and
                        eval_no_improve_count >= eval_early_stop_patience):
                    if (nf_boost_active and nf_boost_current < nf_boost_max):
                        # Boost NF and give fresh patience
                        old_nf = nf_boost_current
                        nf_boost_current = min(nf_boost_current + nf_boost_delta,
                                               nf_boost_max)
                        eval_no_improve_count = 0
                        print(f"  [NF Boost] {old_nf:.3f} -> {nf_boost_current:.3f} "
                              f"(AG plateau for {eval_early_stop_patience} ckpts)",
                              end='', flush=True)
                    else:
                        # NF maxed out (or boost not active) — early stop
                        print(f"\n[Early Stop] AG eval no improvement for "
                              f"{eval_no_improve_count} checkpoints "
                              f"(best={best_eval_mean:.3f} @ step {best_eval_step})")
                        break

            step_dt = time.time() - step_t0
            nf_str = f" nf={current_nf:.3f}" if current_nf is not None else ""
            act_str = f" act={current_act:.2f}" if current_act is not None else ""
            w_str = f" w={current_w:.2f}" if current_w is not None else ""
            eta_str = f" eta={current_eta:.0f}" if current_eta is not None else ""
            edit_str = (f" edits={edit_dists.mean():.1f}/{L}"
                        f"({edit_dists.mean()/L*100:.1f}%)")
            if step_deltas is not None and self.max_edits_per_step > 0:
                edit_str += f" Δ/step={step_deltas.mean():.1f}/{self.max_edits_per_step}"
            print(f"  oracle={oracle_scores.mean():.3f}±{oracle_scores.std():.3f} "
                  f"max={oracle_scores.max():.3f} "
                  f"GC={gc_fractions.mean()*100:.1f}% "
                  f"logZ={log_z:.3f}{nf_str}{act_str}{w_str}{eta_str}{edit_str} "
                  f"({step_dt:.1f}s)")

            if step_callback is not None:
                step_callback(step + 1, population, oracle_scores, gc_fractions, history)

            if beta >= max_beta - 1e-8:
                print(f"\n[GPA] Reached max_beta={max_beta:.4f} after {step+1} steps.")
                break

            step += 1

            # --- ADAPTIVE CONTINUATION CHECK (at step limit boundary) ---
            if step >= current_step_limit and step < hard_max_steps:
                should_extend = False
                reason = ""

                # Check 1: Δβ-based extension
                if extend_on_delta_beta > 0 and delta_beta >= extend_on_delta_beta:
                    should_extend = True
                    reason = (f"Δβ={delta_beta:.4f} >= {extend_on_delta_beta} "
                              f"(still making progress in β)")

                # Check 2: Oracle plateau detection
                if patience > 0 and len(history.oracle_mean) > patience:
                    recent_improvement = (history.oracle_mean[-1]
                                          - history.oracle_mean[-patience - 1])
                    if recent_improvement >= min_improvement:
                        should_extend = True
                        reason = (f"oracle improved {recent_improvement:.4f} "
                                  f">= {min_improvement} over last {patience} steps")

                if should_extend:
                    new_limit = min(current_step_limit + extend_steps,
                                   hard_max_steps)
                    n_extensions += 1
                    print(f"\n[GPA] Adaptive extension #{n_extensions}: "
                          f"{current_step_limit} → {new_limit} steps ({reason})")
                    current_step_limit = new_limit
                else:
                    if step >= current_step_limit:
                        print(f"\n[GPA] Step limit {current_step_limit} reached, "
                              f"no extension criteria met — stopping.")

        total_time = time.time() - t_start_wall
        ext_msg = f" ({n_extensions} extensions)" if n_extensions > 0 else ""
        final_mean = float(oracle_scores.mean())
        best_msg = ""
        if best_oracle_mean > final_mean + 0.01:
            best_msg = f" (best mean={best_oracle_mean:.3f} at step {best_step})"
        eval_msg = ""
        if best_eval_population is not None:
            eval_msg = f" (best eval={best_eval_mean:.3f} at step {best_eval_step})"
        print(f"\n[GPA] Complete: {len(history.beta)} steps{ext_msg}, "
              f"{total_time:.1f}s total, log Z(β={beta:.3f}) = {log_z:.3f}{best_msg}{eval_msg}")

        # Persist spec-bounded mingap archive (design A).
        if mingap_archive_state is not None:
            a_seqs, a_specs, a_cells, a_steps = mingap_archive_state
            history.mingap_archive_seqs = a_seqs.astype(np.int64)
            history.mingap_archive_specs = a_specs
            history.mingap_archive_cell_scores = a_cells
            history.mingap_archive_steps = a_steps
            print(f"[mingap_archive] capacity={mingap_archive_size} "
                  f"final_size={len(a_seqs)} "
                  f"spec range=[{a_specs.min():.3f}, {a_specs.max():.3f}]")
        # Persist per-step accumulated archive (design B).
        if perstep_chunks is not None:
            ps_seqs = np.concatenate(perstep_chunks['seqs'], axis=0)
            ps_specs = np.concatenate(perstep_chunks['specs'], axis=0)
            ps_steps = np.concatenate(perstep_chunks['steps'], axis=0)
            ps_cells = {
                k: np.concatenate(perstep_chunks['cells'][k], axis=0)
                for k in perstep_chunks['cells']
            }
            history.perstep_archive_seqs = ps_seqs.astype(np.int64)
            history.perstep_archive_specs = ps_specs
            history.perstep_archive_cell_scores = ps_cells
            history.perstep_archive_steps = ps_steps
            print(f"[perstep_archive] top_k={perstep_archive_top_k} "
                  f"n_checkpoints={len(perstep_chunks['seqs'])} "
                  f"final_size={len(ps_seqs)} "
                  f"spec range=[{ps_specs.min():.3f}, {ps_specs.max():.3f}]")
        # Persist budget-capped streaming archive (Option B / C).
        if budget_archive_state is not None:
            b_seqs, b_eval, b_kmer, b_appll, b_steps_bd = budget_archive_state
            history.budget_archive_seqs = b_seqs.astype(np.int64)
            history.budget_archive_eval = b_eval
            history.budget_archive_kmer_corr = b_kmer
            history.budget_archive_app_ll = b_appll  # None for Option B
            history.budget_archive_steps = b_steps_bd
            appll_msg = (f" appll range=[{b_appll.min():.2f}, {b_appll.max():.2f}]"
                         if b_appll is not None else "")
            print(f"[budget_archive] capacity={budget_archive_size} "
                  f"final_size={len(b_seqs)} "
                  f"eval range=[{b_eval.min():.3f}, {b_eval.max():.3f}] "
                  f"kmer_corr range=[{b_kmer.min():.3f}, {b_kmer.max():.3f}]"
                  f"{appll_msg} "
                  f"steps covered={len(np.unique(b_steps_bd))}")

        # Persist top-K-by-AG evicting pool.
        if topk_ag_state is not None:
            t_seqs, t_scores, t_steps = topk_ag_state
            history.topk_ag_seqs = t_seqs.astype(np.int64)
            history.topk_ag_scores = t_scores
            history.topk_ag_steps = t_steps
            print(f"[topk_ag] capacity={topk_ag_size} final_size={len(t_seqs)} "
                  f"AG range=[{t_scores.min():.3f}, {t_scores.max():.3f}] "
                  f"mean={t_scores.mean():.3f} "
                  f"steps covered={len(np.unique(t_steps))}")

        return (population, oracle_scores, log_weights, history,
                best_population, best_oracle_scores,
                best_eval_population, best_eval_scores)

    def _adapt_beta(self, oracle_scores: np.ndarray, log_weights: np.ndarray,
                    ess_threshold: float, max_delta: float,
                    n_search: int = 50) -> float:
        """Binary search for largest Δβ that maintains ESS ≥ N·threshold.

        Args:
            oracle_scores: (N,) current oracle scores.
            log_weights: (N,) current log importance weights.
            ess_threshold: Target ESS fraction.
            max_delta: Maximum allowed Δβ (remaining budget).
            n_search: Number of binary search iterations.

        Returns:
            Optimal Δβ.
        """
        N = len(oracle_scores)
        target_ess = ess_threshold * N

        # Check if max_delta is feasible
        trial_lw = log_weights + max_delta * oracle_scores
        trial_lw -= trial_lw.max()
        if self._compute_ess(trial_lw) >= target_ess:
            return max_delta

        lo, hi = 0.0, max_delta
        for _ in range(n_search):
            mid = (lo + hi) / 2
            trial_lw = log_weights + mid * oracle_scores
            trial_lw -= trial_lw.max()
            ess = self._compute_ess(trial_lw)
            if ess >= target_ess:
                lo = mid
            else:
                hi = mid

        return lo

    @staticmethod
    def _compute_ess(log_weights: np.ndarray) -> float:
        """Compute Effective Sample Size from log weights."""
        # Normalize
        lw = log_weights - log_weights.max()
        w = np.exp(lw)
        w_sum = w.sum()
        if w_sum < 1e-300:
            return 0.0
        w_norm = w / w_sum
        return 1.0 / np.sum(w_norm ** 2)

    @staticmethod
    def _log_mean_exp(values: np.ndarray) -> float:
        """Compute log(mean(exp(values))) in a numerically stable way."""
        v_max = values.max()
        return v_max + np.log(np.mean(np.exp(values - v_max)))

    @staticmethod
    def _systematic_resample(log_weights: np.ndarray, N: int) -> np.ndarray:
        """Low-variance systematic resampling.

        Args:
            log_weights: (N,) unnormalized log weights.
            N: Number of particles to resample.

        Returns:
            indices: (N,) resampled particle indices.
        """
        lw = log_weights - log_weights.max()
        weights = np.exp(lw)
        weights /= weights.sum()

        # Cumulative sum
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0  # ensure exact

        # Systematic resampling: single random offset
        u0 = np.random.uniform(0, 1.0 / N)
        u = u0 + np.arange(N) / N

        indices = np.searchsorted(cumsum, u)
        indices = np.clip(indices, 0, N - 1)
        return indices

    @staticmethod
    def _stratified_resample(log_weights, N, gc_fractions, gc_bin_edges,
                             oracle_scores, min_quota=50,
                             alloc_mode="raw", alloc_alpha=5.0):
        """Oracle-proportional stratified resampling by GC bins.

        Allocates resampling quotas proportional to per-bin mean oracle
        score, ensuring no bin dies (min_quota guarantee). Within each
        bin, standard systematic resampling using local fitness weights.

        Args:
            log_weights: (N,) unnormalized log weights
            N: total particles to resample
            gc_fractions: (N,) GC content per particle
            gc_bin_edges: array of bin edges (e.g., [0.40, 0.42, ..., 0.60])
            oracle_scores: (N,) raw oracle scores for allocation weighting
            min_quota: minimum particles per bin (no bin dies)
            alloc_mode: allocation formula — "raw" (v16b: weights=max(means,0)),
                "minshift" (v16: means-min), "exp" (exponential sharpening)
            alloc_alpha: sharpening exponent for "exp" mode (default=5.0)

        Returns:
            indices: (N,) resampled particle indices
        """
        n_bins = len(gc_bin_edges) - 1
        bin_assignments = np.digitize(gc_fractions, gc_bin_edges) - 1
        bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

        # Per-bin mean oracle score (for allocation)
        bin_oracle_means = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            mask = bin_assignments == b
            bin_counts[b] = mask.sum()
            if bin_counts[b] > 0:
                bin_oracle_means[b] = oracle_scores[mask].mean()

        # Allocation formula
        if alloc_mode == "minshift":
            # v16 original: captures relative advantage between bins
            shifted = bin_oracle_means - bin_oracle_means[bin_counts > 0].min()
            shifted[bin_counts == 0] = 0.0
        elif alloc_mode == "exp":
            # Exponential sharpening: exp(alpha * (mean - global_mean))
            active = bin_counts > 0
            global_mean = bin_oracle_means[active].mean() if active.any() else 0.0
            shifted = np.zeros(n_bins)
            shifted[active] = np.exp(alloc_alpha * (bin_oracle_means[active] - global_mean))
            shifted[~active] = 0.0
        else:
            # "raw" (v16b default): proportional to raw oracle means
            shifted = np.maximum(bin_oracle_means, 0.0)
            shifted[bin_counts == 0] = 0.0
        total_score = shifted.sum()

        if total_score > 0:
            alloc_weights = shifted / total_score
        else:
            alloc_weights = np.ones(n_bins) / n_bins

        # Floor + proportional allocation
        floor_total = min_quota * n_bins
        if floor_total >= N:
            quotas = np.full(n_bins, N // n_bins, dtype=int)
        else:
            remaining = N - floor_total
            quotas = np.full(n_bins, min_quota, dtype=int)
            extra = np.round(alloc_weights * remaining).astype(int)
            quotas += extra
            # Adjust to hit exact N
            diff = N - quotas.sum()
            if diff != 0:
                idx = int(np.argmax(alloc_weights))
                quotas[idx] += diff

        # Diagnostic: show per-bin oracle means and quotas
        bin_info = ", ".join(f"b{b}={bin_oracle_means[b]:.2f}({bin_counts[b]})->{quotas[b]}" for b in range(n_bins))
        print(f"[stratified_resample] {bin_info}")

        # Per-bin systematic resampling
        all_indices = []
        for b in range(n_bins):
            bin_idx = np.where(bin_assignments == b)[0]
            quota = quotas[b]

            if len(bin_idx) == 0 or quota == 0:
                continue

            # Local fitness weights within bin
            local_lw = log_weights[bin_idx]
            local_lw = local_lw - local_lw.max()
            local_w = np.exp(local_lw)
            local_w_sum = local_w.sum()
            if local_w_sum < 1e-300:
                local_w = np.ones(len(bin_idx)) / len(bin_idx)
            else:
                local_w = local_w / local_w_sum

            cumsum = np.cumsum(local_w)
            cumsum[-1] = 1.0
            u0 = np.random.uniform(0, 1.0 / quota)
            u = u0 + np.arange(quota) / quota
            local_indices = np.searchsorted(cumsum, u)
            local_indices = np.clip(local_indices, 0, len(bin_idx) - 1)

            all_indices.append(bin_idx[local_indices])

        indices = np.concatenate(all_indices) if all_indices else np.arange(N)

        # If short (due to empty bins), fill from global resampling
        if len(indices) < N:
            shortfall = N - len(indices)
            global_idx = DiffusionPopulationAnnealer._systematic_resample(
                log_weights, shortfall)
            indices = np.concatenate([indices, global_idx])

        return indices[:N]

    def _mutate_population(
        self,
        population: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        bio_filter_fn: Optional[Callable],
        oracle_scores_before: np.ndarray,
        mutate_fn_override: Optional[Callable] = None,
        elite_fraction: float = 0.0,
        original_population: Optional[torch.Tensor] = None,
        max_edits: Optional[int] = None,
        max_edits_per_step: int = 0,
    ) -> "torch.Tensor | Tuple[torch.Tensor, np.ndarray, np.ndarray]":
        """Mutate all particles using CFG warm-start sampler.

        Args:
            population: (N, L) current particles.
            labels: (N, 1) activity labels.
            batch_size: Process in batches of this size.
            bio_filter_fn: If provided, reject non-plausible mutations.
            oracle_scores_before: Pre-mutation scores (unused here but
                available for future MH-style acceptance).
            mutate_fn_override: If provided, use this instead of self.mutate_fn
                (for noise fraction annealing).
            elite_fraction: Fraction of top particles (by oracle_scores_before)
                to preserve unchanged. 0.0 = mutate all (default).
            original_population: (N, L) original seed sequences for edit tracking.
            max_edits: Maximum Hamming distance from original allowed.
            max_edits_per_step: Max new edits per step (0=disabled).
                Limits hamming(mutated, input) per step.

        Returns:
            mutated: (N, L) mutated particles.
            If cache_mutation_scores is True AND K-branch scored winners:
                (mutated, cached_scores, cached_gc) — scores/gc from K-branch selection.
        """
        fn = mutate_fn_override if mutate_fn_override is not None else self.mutate_fn
        N, L = population.shape

        # Elitism: identify top particles to skip mutation
        n_elite = int(elite_fraction * N) if elite_fraction > 0 else 0
        if n_elite > 0:
            elite_idx = np.argsort(oracle_scores_before)[::-1][:n_elite].copy()
            elite_set = set(elite_idx)
            mutate_mask = np.array([i not in elite_set for i in range(N)])
            mutate_indices = np.where(mutate_mask)[0]
        else:
            mutate_indices = np.arange(N)

        # Build output — start with copy of population
        result = population.clone()

        # Cached scores tracking (for cache_mutation_scores)
        cached_scores = np.full(N, np.nan) if self.cache_mutation_scores else None
        cached_gc = np.full(N, np.nan) if self.cache_mutation_scores else None
        # Elite particles keep their pre-mutation scores
        if cached_scores is not None and n_elite > 0:
            cached_scores[elite_idx] = oracle_scores_before[elite_idx]
            gc_np = ((population.numpy() == 1) | (population.numpy() == 2)).mean(axis=1)
            cached_gc[elite_idx] = gc_np[elite_idx]
        has_cached = False  # track whether K-branch provided scores

        # Only mutate non-elite particles
        n_mutate = len(mutate_indices)
        if n_mutate == 0:
            if self.cache_mutation_scores:
                return result, cached_scores, cached_gc
            return result

        pop_to_mutate = population[mutate_indices]
        labels_to_mutate = labels[mutate_indices] if labels is not None else None

        K = self.branch_factor
        n_batches = (n_mutate + batch_size - 1) // batch_size
        mutated_parts = []
        total_budget_rejects = 0
        total_particles = 0

        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_mutate)
            x_batch = pop_to_mutate[start:end]
            B_batch = end - start
            labels_batch = labels_to_mutate[start:end] if labels_to_mutate is not None else None

            if self.mcts_depth >= 2 and K > 1:
                # === MCTS PATH ===
                batch_idx = mutate_indices[start:end]
                originals = original_population[batch_idx] if original_population is not None else None
                x_new = self._mcts_mutate_batch(
                    fn, x_batch, labels_batch, originals, bio_filter_fn, B_batch, L)
                mutated_parts.append(x_new)
            elif K <= 1:
                # === STANDARD PATH ===
                x_new = fn(self.model, x_batch.to(self.device), labels_batch)
                x_new = x_new.cpu()

                # Bio-plausibility gating: reject non-plausible, keep parent
                if bio_filter_fn is not None:
                    bio_mask = bio_filter_fn(x_new.numpy())
                    n_rejected = (~bio_mask).sum()
                    if n_rejected > 0:
                        x_new[~bio_mask] = x_batch[~bio_mask]

                # Edit cap (simple rejection for K=1)
                if max_edits is not None and max_edits < L and original_population is not None:
                    batch_idx = mutate_indices[start:end]
                    originals = original_population[batch_idx]
                    hamming = (x_new != originals).sum(dim=1)
                    too_far = hamming > max_edits
                    if too_far.any():
                        x_new[too_far] = x_batch[too_far]
                        total_budget_rejects += int(too_far.sum())
                    total_particles += B_batch

                # Per-step delta cap (K=1: reject if too many new edits)
                if max_edits_per_step > 0:
                    step_delta = (x_new != x_batch).sum(dim=1)
                    too_many = step_delta > max_edits_per_step
                    if too_many.any():
                        x_new[too_many] = x_batch[too_many]

                # Full-vocab hill-climb for K=1 path
                if self.hill_climb_full_vocab and self.hill_climb_budget > 0:
                    scores_before, _ = self.oracle_fn(x_new)
                    orig_batch = None
                    if original_population is not None:
                        k1_batch_idx = mutate_indices[start:end]
                        orig_batch = original_population[k1_batch_idx]
                    x_new, n_swaps = self._hill_climb_full_vocab(
                        x_new, x_batch, scores_before, bio_filter_fn,
                        self.hill_climb_budget,
                        original_seqs=orig_batch, max_edits=max_edits)
                    total_swaps = int(n_swaps.sum())
                    if total_swaps > 0:
                        mean_swaps = float(n_swaps.float().mean())
                        print(f" HC:{mean_swaps:.1f}", end='', flush=True)

                mutated_parts.append(x_new)
            else:
                # === K-BRANCH TREE SEARCH ===
                # 1. Get K branches from ONE forward pass
                result_or_tuple = fn(self.model, x_batch.to(self.device), labels_batch,
                                     branch_factor=K, return_metadata=(self.prefilter_topk > 0 or self.entropy_hc))
                if isinstance(result_or_tuple, tuple):
                    branches, mutation_meta = result_or_tuple
                else:
                    branches = result_or_tuple
                    mutation_meta = None
                branches = branches.cpu()

                # 2. Bio-filter each branch independently
                for k in range(K):
                    if bio_filter_fn is not None:
                        bio_mask = bio_filter_fn(branches[k].numpy())
                        if (~bio_mask).sum() > 0:
                            branches[k][~bio_mask] = x_batch[~bio_mask]

                # 3. Score branches with oracle (with optional MDLM pre-filter)
                J = self.prefilter_topk
                if J > 0 and J < K and mutation_meta is not None and "branch_logprobs" in mutation_meta:
                    # Pre-filter: rank by MDLM branch logprob, score only top-J
                    blp = mutation_meta["branch_logprobs"].cpu().numpy()  # (K, B)
                    top_j_per_particle = np.argsort(-blp, axis=0)[:J]  # (J, B) — top-J branch indices
                    # Build flat tensor of only top-J branches per particle
                    flat_parts = []
                    for j_rank in range(J):
                        branch_indices = top_j_per_particle[j_rank]  # (B,)
                        selected = branches[branch_indices, torch.arange(B_batch)]  # (B, L)
                        flat_parts.append(selected)
                    flat_topj = torch.cat(flat_parts, dim=0)  # (J*B, L)
                    flat_scores_topj, _ = self.oracle_fn(flat_topj)
                    # Fill scores_kb: only top-J have real scores, rest get -inf
                    scores_kb = np.full((K, B_batch), -np.inf)
                    for j_rank in range(J):
                        branch_indices = top_j_per_particle[j_rank]
                        for b in range(B_batch):
                            scores_kb[branch_indices[b], b] = flat_scores_topj[j_rank * B_batch + b]
                elif self.surrogate_prefilter and self._surrogate_model is not None and K > 1:
                    # Surrogate pre-filter: rank by kmer surrogate, score only top-J
                    J_sur = max(2, K // 2)  # score top half
                    flat_np = branches.reshape(K * B_batch, -1).numpy()
                    from bio_plausibility import kmer_profile
                    flat_kmers = kmer_profile(flat_np, k=3)
                    sur_preds = self._surrogate_model.predict(flat_kmers).reshape(K, B_batch)
                    top_j = np.argsort(-sur_preds, axis=0)[:J_sur]  # (J_sur, B)
                    flat_parts = []
                    for j_rank in range(J_sur):
                        branch_indices = top_j[j_rank]
                        selected = branches[branch_indices, torch.arange(B_batch)]
                        flat_parts.append(selected)
                    flat_topj = torch.cat(flat_parts, dim=0)
                    flat_scores_topj, _ = self.oracle_fn(flat_topj)
                    scores_kb = np.full((K, B_batch), -np.inf)
                    for j_rank in range(J_sur):
                        branch_indices = top_j[j_rank]
                        for b in range(B_batch):
                            scores_kb[branch_indices[b], b] = flat_scores_topj[j_rank * B_batch + b]
                else:
                    # Standard: score all K*B candidates
                    flat = branches.reshape(K * B_batch, -1)
                    flat_scores, _ = self.oracle_fn(flat)
                    scores_kb = flat_scores.reshape(K, B_batch)

                # 4. Compute TOTAL Hamming from original seeds
                if original_population is not None:
                    batch_idx = mutate_indices[start:end]
                    originals = original_population[batch_idx]  # (B, L)
                    hamming_kb = (branches != originals.unsqueeze(0)).sum(dim=-1).float()  # (K, B)
                else:
                    hamming_kb = torch.zeros(K, B_batch)

                # 5. Edit-aware selection
                if max_edits is not None and max_edits < L:
                    valid = hamming_kb <= max_edits  # (K, B)
                else:
                    valid = torch.ones(K, B_batch, dtype=torch.bool)

                # Per-step delta cap: hamming from current particle (not original)
                if max_edits_per_step > 0:
                    step_delta_kb = (branches != x_batch.unsqueeze(0)).sum(dim=-1).float()  # (K, B)
                    step_valid = step_delta_kb <= max_edits_per_step  # (K, B)
                    valid = valid & step_valid

                if self.edit_select_mode == "efficiency":
                    criterion = scores_kb / (1.0 + hamming_kb.numpy())
                else:  # "oracle"
                    criterion = scores_kb.copy() if isinstance(scores_kb, np.ndarray) else scores_kb.clone()

                # Convert to numpy for consistent handling
                if isinstance(criterion, torch.Tensor):
                    criterion = criterion.numpy()
                if isinstance(scores_kb, torch.Tensor):
                    scores_kb = scores_kb.numpy()

                criterion_masked = criterion.copy()
                criterion_masked[~valid.numpy()] = -np.inf
                if self.branch_selection_tau > 0:
                    # ε-greedy: sample from softmax(criterion/τ) over K axis.
                    # Replace -inf (invalid entries) with a large negative number
                    # before normalization so softmax arithmetic stays finite.
                    # Columns with all-invalid are handled by the all_invalid
                    # fallback below (best_k=0 → reverted to parent).
                    valid_np = valid.numpy()
                    safe_crit = np.where(valid_np, criterion, -1e9)
                    logits = safe_crit / self.branch_selection_tau  # (K, B)
                    logits = logits - logits.max(axis=0, keepdims=True)
                    probs = np.exp(logits)
                    probs = probs / probs.sum(axis=0, keepdims=True)
                    cum = probs.cumsum(axis=0)
                    u = np.random.random(size=(cum.shape[1],))
                    best_k = (cum > u[None, :]).argmax(axis=0)  # (B,)
                else:
                    best_k = criterion_masked.argmax(axis=0)  # (B,) — best branch per particle

                # 6. Handle all-invalid: fall back to parent (frozen)
                all_invalid = ~valid.numpy().any(axis=0)  # (B,)
                best_k[all_invalid] = 0
                total_budget_rejects += int(all_invalid.sum())
                total_particles += B_batch

                # APF correction (S5): log p_τ(k*) per selected particle.
                # Only applies when τ>0 (stochastic selection). For all-invalid
                # particles (reverted to parent, no selection happened), set 0.
                if (self.apf_correction
                        and self._last_apf_log_correction is not None
                        and self.branch_selection_tau > 0):
                    selected_probs = probs[best_k, np.arange(B_batch)]
                    selected_logp = np.log(np.maximum(selected_probs, 1e-30))
                    selected_logp = np.where(all_invalid, 0.0, selected_logp)
                    batch_particle_ids = mutate_indices[start:end]
                    self._last_apf_log_correction[batch_particle_ids] = selected_logp

                # S7 DPS IW correction: mutator-side per-branch correction is
                # (K, B). Pick the winner row with best_k.
                if (self.dps_iw_correct
                        and self._last_dps_iw_log_correction is not None):
                    corr_kb = getattr(fn, "_last_dps_iw_correction_kb", None)
                    if corr_kb is not None and corr_kb.shape[1] == B_batch:
                        winner_corr = corr_kb[best_k, np.arange(B_batch)]  # (B_batch,)
                        winner_corr = np.where(all_invalid, 0.0, winner_corr)
                        batch_particle_ids = mutate_indices[start:end]
                        self._last_dps_iw_log_correction[batch_particle_ids] = winner_corr

                x_new = branches[best_k, torch.arange(B_batch)]  # (B, L)
                x_new[all_invalid] = x_batch[all_invalid]  # revert to parent

                # Cache winning scores from K-branch selection
                if cached_scores is not None:
                    winner_scores = scores_kb[best_k, np.arange(B_batch)]
                    # For all-invalid particles, keep pre-mutation scores
                    winner_scores[all_invalid] = oracle_scores_before[mutate_indices[start:end]][all_invalid]
                    cached_scores[mutate_indices[start:end]] = winner_scores
                    # Compute GC for winners
                    gc_w = ((x_new.numpy() == 1) | (x_new.numpy() == 2)).mean(axis=1)
                    cached_gc[mutate_indices[start:end]] = gc_w
                    has_cached = True

                # --- HILL-CLIMB: per-position greedy optimization ---
                do_hc = self.hill_climb or self.hill_climb_budget > 0
                if do_hc:
                    best_scores = scores_kb[best_k, np.arange(B_batch)]  # (B,)
                    orig_batch = None
                    if original_population is not None:
                        hc_batch_idx = mutate_indices[start:end]
                        orig_batch = original_population[hc_batch_idx]
                    if self.hill_climb_full_vocab:
                        entropy_weights = None
                        if self.entropy_hc and mutation_meta is not None and "entropy" in mutation_meta:
                            entropy_weights = mutation_meta["entropy"].cpu()  # (B_full, L)
                            # Slice to this batch's active particles
                            entropy_weights = entropy_weights[:B_batch]
                        x_new, n_swaps = self._hill_climb_full_vocab(
                            x_new, x_batch, best_scores, bio_filter_fn,
                            self.hill_climb_budget,
                            original_seqs=orig_batch, max_edits=max_edits,
                            entropy_weights=entropy_weights)
                    else:
                        x_new, n_swaps = self._hill_climb_positions(
                            x_new, branches, x_batch, best_scores,
                            bio_filter_fn, max_edits_per_step,
                            original_seqs=orig_batch, max_edits=max_edits)
                    total_swaps = int(n_swaps.sum())
                    if total_swaps > 0:
                        mean_swaps = float(n_swaps.float().mean())
                        print(f" HC:{mean_swaps:.1f}", end='', flush=True)
                        # HC changed sequences — invalidate cached scores
                        if cached_scores is not None:
                            has_cached = False

                mutated_parts.append(x_new)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result[mutate_indices] = torch.cat(mutated_parts, dim=0)

        # For K=1 without HC, cached_scores won't have been populated — mark as not cached
        if self.cache_mutation_scores and has_cached and not np.any(np.isnan(cached_scores)):
            return result, cached_scores, cached_gc
        return result

    # ======================================================================
    # Hill-Climb: per-position greedy optimization from K branches
    # ======================================================================

    def _hill_climb_positions(self, x_best, branches, x_parent, best_scores,
                              bio_filter_fn, max_edits_per_step,
                              oracle_batch_size=128,
                              original_seqs=None, max_edits=None):
        """Hill-climb on best branch by testing per-position swaps from other branches.

        For each disagreement position (where branches differ from x_best),
        try swapping to alternative tokens proposed by other branches.
        Accept if oracle improves (greedy).

        Batched across particles: for each position, collect all (particle, token)
        trials into a single oracle batch call.

        Args:
            x_best: (B, L) best whole branch per particle
            branches: (K, B, L) all K branches
            x_parent: (B, L) parent sequences (before mutation)
            best_scores: (B,) oracle scores for x_best
            bio_filter_fn: optional bio-plausibility check
            max_edits_per_step: cap on total edits from parent (0=disabled)
            oracle_batch_size: max batch size for oracle calls

        Returns:
            x_improved: (B, L) hill-climbed sequences
            n_swaps: (B,) number of accepted swaps per particle
        """
        K, B, L = branches.shape
        x_current = x_best.clone()
        current_scores = best_scores.copy()  # numpy (B,)
        n_swaps = torch.zeros(B, dtype=torch.long)

        # Find disagreement positions: where any branch differs from x_best
        disagree = (branches != x_best.unsqueeze(0)).any(dim=0)  # (B, L)

        # Compute token entropy per position to prioritize high-disagreement positions
        # Count distinct tokens across K branches at each position
        token_diversity = torch.zeros(B, L)
        for pos in range(L):
            tokens_at_pos = branches[:, :, pos]  # (K, B)
            for b_idx in range(B):
                token_diversity[b_idx, pos] = len(tokens_at_pos[:, b_idx].unique())

        # Process positions in order of decreasing token diversity
        # Get all disagreement positions and sort by diversity
        pos_order = token_diversity.argsort(dim=1, descending=True)  # (B, L)

        # Track which particles still have edit budget
        if max_edits_per_step > 0:
            current_edits = (x_current != x_parent).sum(dim=1)  # (B,)
        else:
            current_edits = None

        # Iterate over positions (by rank, not absolute position)
        max_disagree_positions = int(disagree.float().sum(dim=1).max().item())
        for rank in range(max_disagree_positions):
            if rank >= L:
                break

            # Each particle's next position to try (sorted by diversity)
            positions = pos_order[:, rank]  # (B,) — position index per particle

            # Which particles have a disagreement at this position?
            active = disagree[torch.arange(B), positions]  # (B,)

            # Check edit budget
            if current_edits is not None:
                budget_ok = current_edits < max_edits_per_step
                active = active & budget_ok

            if not active.any():
                continue

            active_idx = active.nonzero(as_tuple=True)[0]
            B_active = len(active_idx)

            # Collect unique alternative tokens per active particle
            # For each active particle, gather tokens from all K branches at its position
            trial_particles = []  # list of (particle_idx, position, alt_token)
            for i in range(B_active):
                b = active_idx[i].item()
                pos = positions[b].item()
                original_token = x_current[b, pos].item()
                seen = {original_token}
                for k in range(K):
                    t = branches[k, b, pos].item()
                    if t not in seen:
                        seen.add(t)
                        trial_particles.append((b, pos, t))

            if not trial_particles:
                continue

            # Batch oracle evaluation: create trial sequences
            n_trials = len(trial_particles)
            trial_seqs = torch.zeros(n_trials, L, dtype=torch.long)
            trial_particle_idx = []
            for j, (b, pos, alt_t) in enumerate(trial_particles):
                trial_seqs[j] = x_current[b]
                trial_seqs[j, pos] = alt_t
                trial_particle_idx.append(b)

            # Bio filter on trials
            if bio_filter_fn is not None:
                bio_ok = bio_filter_fn(trial_seqs.numpy())
            else:
                bio_ok = np.ones(n_trials, dtype=bool)

            # Score valid trials in batches
            trial_scores = np.full(n_trials, -np.inf)
            valid_mask = bio_ok
            valid_indices = np.where(valid_mask)[0]

            for chunk_start in range(0, len(valid_indices), oracle_batch_size):
                chunk_end = min(chunk_start + oracle_batch_size, len(valid_indices))
                chunk_idx = valid_indices[chunk_start:chunk_end]
                chunk_scores, _ = self.oracle_fn(trial_seqs[chunk_idx])
                trial_scores[chunk_idx] = chunk_scores

            # Greedy accept: for each particle, pick best improving swap
            # Group trials by particle
            particle_trials = {}  # b -> list of (trial_idx, pos, alt_t)
            for j, (b, pos, alt_t) in enumerate(trial_particles):
                if b not in particle_trials:
                    particle_trials[b] = []
                particle_trials[b].append((j, pos, alt_t))

            for b, trials in particle_trials.items():
                best_score = current_scores[b]
                best_token = None
                best_pos = None
                best_trial_score = best_score
                for j, pos, alt_t in trials:
                    if trial_scores[j] > best_trial_score:
                        best_trial_score = trial_scores[j]
                        best_token = alt_t
                        best_pos = pos

                if best_token is not None:
                    if max_edits is not None and original_seqs is not None:
                        trial = x_current[b].clone()
                        trial[best_pos] = best_token
                        if (trial != original_seqs[b]).sum() > max_edits:
                            continue  # would exceed total edit budget
                    x_current[b, best_pos] = best_token
                    current_scores[b] = best_trial_score
                    n_swaps[b] += 1
                    if current_edits is not None:
                        current_edits[b] = (x_current[b] != x_parent[b]).sum()

        return x_current, n_swaps

    # ======================================================================
    # Full-Vocab Hill-Climb: exhaustive single-position greedy ascent
    # ======================================================================

    def _hill_climb_full_vocab(self, x_best, x_parent, best_scores,
                               bio_filter_fn, budget,
                               original_seqs=None, max_edits=None,
                               entropy_weights=None):
        """Hill-climb by trying all alternative tokens at all positions.

        Per-token batching: iterates over 4 token values (not 200×4 pairs),
        building all trial sequences for each token in one batch. This reduces
        ~600 serial oracle calls to ~4 large batched calls.

        Optional position subsampling (hill_climb_positions > 0) further
        reduces candidates per round.

        Args:
            x_best: (B, L) current best sequences
            x_parent: (B, L) parent sequences (before mutation this step)
            best_scores: (B,) oracle scores for x_best (numpy)
            bio_filter_fn: optional bio-plausibility check
            budget: number of HC rounds (edits to attempt)
            entropy_weights: (B, L) optional per-position entropy for weighted
                position sampling. High-entropy positions are tried first.
                None = uniform/random subsample (default).

        Returns:
            x_improved: (B, L) hill-climbed sequences
            n_swaps: (B,) number of accepted swaps per particle
        """
        B, L = x_best.shape
        VOCAB = 4  # DNA: A=0, C=1, G=2, T=3
        x_current = x_best.clone()
        current_scores = best_scores.copy()  # numpy (B,)
        n_swaps = torch.zeros(B, dtype=torch.long)

        score_fn = self.oracle_fn_fast or self.oracle_fn

        for hc_round in range(budget):
            best_gain = np.zeros(B)
            best_swap_pos = np.full(B, -1, dtype=np.int64)
            best_swap_token = np.full(B, -1, dtype=np.int64)

            x_current_np = x_current.numpy()

            # Position subsampling (Option 2), optionally entropy-weighted (1B)
            if self.hill_climb_positions > 0 and self.hill_climb_positions < L:
                if entropy_weights is not None:
                    # Entropy-weighted sampling: high-entropy positions first
                    ent_mean = entropy_weights.mean(dim=0).numpy()  # (L,) mean entropy across batch
                    ent_prob = ent_mean / (ent_mean.sum() + 1e-10)
                    chosen = np.random.choice(L, self.hill_climb_positions, replace=False, p=ent_prob)
                else:
                    chosen = np.random.choice(L, self.hill_climb_positions, replace=False)
                col_mask = np.zeros(L, dtype=bool)
                col_mask[chosen] = True
            else:
                col_mask = None

            for alt_t in range(VOCAB):
                # Find all (particle, position) pairs where current != alt_t
                differs_mask = (x_current_np != alt_t)  # (B, L)
                if col_mask is not None:
                    differs_mask = differs_mask & col_mask[None, :]

                particle_idx, pos_idx = np.where(differs_mask)
                if len(particle_idx) == 0:
                    continue

                # Build trial sequences: copy parent, swap one position
                trial_seqs = x_current[particle_idx].clone()  # (N_trials, L)
                trial_seqs[torch.arange(len(particle_idx)), pos_idx] = alt_t

                # Bio filter
                if bio_filter_fn is not None:
                    bio_ok = bio_filter_fn(trial_seqs.numpy())
                    if not bio_ok.any():
                        continue
                    valid_mask = bio_ok
                    particle_idx = particle_idx[valid_mask]
                    pos_idx = pos_idx[valid_mask]
                    trial_seqs = trial_seqs[valid_mask]

                if len(particle_idx) == 0:
                    continue

                # Score all trials in one batched call
                trial_scores, _ = score_fn(trial_seqs)

                # K-mer composition penalty: adjust scores by composition delta
                if self.hc_kmer_weight > 0 and self.ref_kmer_profile is not None:
                    trial_kmer_r = _fast_kmer_corr(
                        trial_seqs.numpy(), self.ref_kmer_profile, k=3)
                    parent_kmer_r = _fast_kmer_corr(
                        x_current_np[particle_idx], self.ref_kmer_profile, k=3)
                    kmer_delta = trial_kmer_r - parent_kmer_r  # positive = improving
                    trial_scores = trial_scores + self.hc_kmer_weight * kmer_delta

                # Compute gains and scatter-max per particle
                gains = trial_scores - current_scores[particle_idx]
                improved = gains > 0
                if not improved.any():
                    continue

                for idx in np.where(improved)[0]:
                    b = particle_idx[idx]
                    if gains[idx] > best_gain[b]:
                        best_gain[b] = gains[idx]
                        best_swap_pos[b] = pos_idx[idx]
                        best_swap_token[b] = alt_t

            # Apply best swaps for this round
            has_swap = best_swap_pos >= 0
            if not has_swap.any():
                break  # No improvements found, stop early

            for b in np.where(has_swap)[0]:
                if max_edits is not None and original_seqs is not None:
                    trial = x_current[b].clone()
                    trial[best_swap_pos[b]] = best_swap_token[b]
                    if (trial != original_seqs[b]).sum() > max_edits:
                        continue  # would exceed total edit budget
                x_current[b, best_swap_pos[b]] = best_swap_token[b]
                current_scores[b] += best_gain[b]
                n_swaps[b] += 1

        return x_current, n_swaps

    def _pareto_rank_fitness(self, scores, edit_dists):
        """Assign fitness based on Pareto rank in (score, -edits) space.

        Non-dominated particles (rank 0) get highest fitness.
        Within each rank, NSGA-II crowding distance preserves diversity
        along the Pareto front.

        Args:
            scores: (N,) np.array of fitness/oracle scores
            edit_dists: (N,) np.array of edit distances from original

        Returns:
            (N,) np.array of Pareto-rank-based fitness values
        """
        N = len(scores)
        # Objectives: maximize score, minimize edits (= maximize -edits)
        obj = np.column_stack([scores, -edit_dists])
        ranks = np.full(N, -1, dtype=int)
        remaining = np.arange(N)
        rank = 0

        while len(remaining) > 0:
            # Find non-dominated in remaining
            n_rem = len(remaining)
            is_dominated = np.zeros(n_rem, dtype=bool)
            obj_rem = obj[remaining]

            for i in range(n_rem):
                if is_dominated[i]:
                    continue
                # i dominates j if all objectives >= and at least one >
                geq = obj_rem >= obj_rem[i]  # (n_rem, 2)
                gt = obj_rem > obj_rem[i]
                dominates_i = geq.all(axis=1) & gt.any(axis=1)  # who dominates i?
                if dominates_i.any():
                    is_dominated[i] = True

            non_dom_local = np.where(~is_dominated)[0]
            non_dom_global = remaining[non_dom_local]
            ranks[non_dom_global] = rank
            remaining = np.setdiff1d(remaining, non_dom_global)
            rank += 1

        # Crowding distance within each rank
        crowding = np.zeros(N)
        for r in range(rank):
            front = np.where(ranks == r)[0]
            if len(front) <= 2:
                crowding[front] = np.inf
                continue
            for m in range(2):  # 2 objectives
                sorted_idx = front[np.argsort(obj[front, m])]
                crowding[sorted_idx[0]] = np.inf
                crowding[sorted_idx[-1]] = np.inf
                obj_range = obj[sorted_idx[-1], m] - obj[sorted_idx[0], m]
                if obj_range > 0:
                    for i in range(1, len(sorted_idx) - 1):
                        crowding[sorted_idx[i]] += (
                            (obj[sorted_idx[i+1], m] - obj[sorted_idx[i-1], m])
                            / obj_range
                        )

        # Convert to fitness: lower rank = higher fitness, crowding as tiebreaker
        max_rank = rank
        # Normalize crowding to [0, 1) to use as tiebreaker within rank
        finite_crowd = crowding[np.isfinite(crowding)]
        crowd_max = finite_crowd.max() if len(finite_crowd) > 0 else 1.0
        crowd_norm = np.where(np.isfinite(crowding),
                              crowding / (crowd_max + 1e-8),
                              1.0)
        fitness = (max_rank - ranks).astype(float) + 0.5 * crowd_norm
        return fitness

    # ======================================================================
    # MCTS (Monte Carlo Tree Search) mutation operator
    # ======================================================================

    def _mcts_mutate_batch(self, fn, x_batch, labels_batch,
                           original_batch, bio_filter_fn, B_batch, L):
        """Run MCTS for one mutation batch of B particles.

        Each particle gets an independent tree. UCB1 selects which leaf to
        expand, MDLM+DPS generates K children, oracle scores them, and
        scores backpropagate up the tree.

        Returns:
            x_new: (B, L) tensor of selected mutations (CPU).
        """
        K = self.branch_factor
        I = self.mcts_iterations
        D = self.mcts_depth
        C = self.mcts_c
        M = 1 + I * K + 2  # max nodes: root + I expansions of K children + buffer

        # Dense tensor tree storage
        sequences = torch.zeros(B_batch, M, L, dtype=torch.long)
        scores = torch.full((B_batch, M), -float('inf'), dtype=torch.float32)
        visit_count = torch.zeros(B_batch, M, dtype=torch.long)
        value_sum = torch.zeros(B_batch, M, dtype=torch.float64)
        parent = torch.full((B_batch, M), -1, dtype=torch.long)
        children = torch.full((B_batch, M, K), -1, dtype=torch.long)
        depth = torch.zeros(B_batch, M, dtype=torch.long)
        n_children = torch.zeros(B_batch, M, dtype=torch.long)
        active = torch.zeros(B_batch, M, dtype=torch.bool)
        next_free = torch.ones(B_batch, dtype=torch.long)  # 0 = root

        # Initialize root = current particle
        sequences[:, 0] = x_batch.cpu()
        active[:, 0] = True
        root_scores, _ = self.oracle_fn(x_batch.cpu())
        root_scores_t = torch.from_numpy(root_scores).float()
        scores[:, 0] = root_scores_t
        visit_count[:, 0] = 1
        value_sum[:, 0] = root_scores_t.double()

        B_idx = torch.arange(B_batch)

        for iteration in range(I):
            # 1. SELECT: UCB1 traversal (default), or BFS over unexpanded
            #    nodes (uniform mode — target-preserved under Prop. K with
            #    K_eff = K^D, since all leaves are iid samples from K_β^D).
            if self.mcts_select_mode == "uniform":
                selected = self._uniform_select(
                    B_batch, n_children, active, depth, D)
            else:
                selected = self._ucb_select(
                    B_batch, K, D, C, n_children, children, visit_count, value_sum)

            # 2. Check depth limit and expansion eligibility
            sel_depth = depth[B_idx, selected]
            can_expand = (sel_depth < D) & (n_children[B_idx, selected] == 0)

            if can_expand.any():
                exp_idx = can_expand.nonzero(as_tuple=True)[0]
                B_exp = len(exp_idx)
                exp_seqs = sequences[exp_idx, selected[exp_idx]].to(self.device)
                exp_labels = labels_batch[exp_idx] if labels_batch is not None else None

                # 3. EXPAND: K children via MDLM+DPS
                branches = fn(self.model, exp_seqs, exp_labels,
                              branch_factor=K)  # (K, B_exp, L)
                branches = branches.cpu()

                # 4. EVALUATE: score K*B_exp candidates
                flat = branches.reshape(K * B_exp, -1)
                flat_scores, _ = self.oracle_fn(flat)
                child_scores = flat_scores.reshape(K, B_exp)  # numpy

                # 5. INSERT children into tree
                for k in range(K):
                    slot = next_free[exp_idx].clone()
                    sequences[exp_idx, slot] = branches[k]
                    scores[exp_idx, slot] = torch.from_numpy(child_scores[k]).float()
                    visit_count[exp_idx, slot] = 1
                    value_sum[exp_idx, slot] = torch.from_numpy(
                        child_scores[k].astype(np.float64))
                    parent[exp_idx, slot] = selected[exp_idx]
                    depth[exp_idx, slot] = sel_depth[exp_idx] + 1
                    children[exp_idx, selected[exp_idx], k] = slot
                    active[exp_idx, slot] = True
                    next_free[exp_idx] += 1
                n_children[exp_idx, selected[exp_idx]] = K

                # 6. BACKPROPAGATE: best child score up ancestor chain
                best_child_val = child_scores.max(axis=0)  # (B_exp,)
                self._backpropagate(
                    visit_count, value_sum, parent, D,
                    exp_idx, selected[exp_idx], best_child_val)

            # Re-visit max-depth leaves (update visit counts → UCB shifts away)
            at_limit = ~can_expand
            if at_limit.any():
                lim_idx = at_limit.nonzero(as_tuple=True)[0]
                lim_nodes = selected[lim_idx]
                lim_scores = scores[lim_idx, lim_nodes].numpy()
                self._backpropagate(
                    visit_count, value_sum, parent, D,
                    lim_idx, lim_nodes, lim_scores)

        # 7. FINAL SELECTION
        return self._mcts_select_final(
            sequences, scores, active, B_batch, L, M,
            x_batch, original_batch, bio_filter_fn)

    def _uniform_select(self, B, n_children, active, depth, max_depth):
        """BFS-order selection: first active unexpanded node with depth<D.

        Returns leaves at depth D last (they hit at_limit downstream and only
        bump visit counts). With dense BFS-order insertion, picking the lowest
        active slot with n_children=0 fills the tree breadth-first.
        """
        is_unexpanded = active & (n_children == 0) & (depth < max_depth)
        has_any = is_unexpanded.any(dim=1)
        # argmax on bool returns first True index → BFS order
        selected = is_unexpanded.int().argmax(dim=1)
        # Fallback: if every node is at depth=D (tree fully filled), pick any
        # depth-D leaf (will hit at_limit, no expansion). Use root as safe
        # fallback when tree is in a weird state.
        if not has_any.all():
            any_active_leaf = active & (n_children == 0)
            fallback = any_active_leaf.int().argmax(dim=1)
            selected = torch.where(has_any, selected, fallback)
        return selected

    def _ucb_select(self, B, K, max_depth, C,
                    n_children, children, visit_count, value_sum):
        """UCB1 traversal from root to leaf for all B particles."""
        current = torch.zeros(B, dtype=torch.long)  # start at root
        B_idx = torch.arange(B)

        for d in range(max_depth):
            is_internal = n_children[B_idx, current] > 0
            if not is_internal.any():
                break

            int_idx = is_internal.nonzero(as_tuple=True)[0]
            B_int = len(int_idx)
            children_ids = children[int_idx, current[int_idx]]  # (B_int, K)
            valid = children_ids >= 0

            safe_ids = children_ids.clamp(min=0)
            idx_expand = int_idx.unsqueeze(1).expand(-1, K)

            child_visits = visit_count[idx_expand, safe_ids]  # (B_int, K)
            child_vsum = value_sum[idx_expand, safe_ids]
            parent_visits = visit_count[int_idx, current[int_idx]]

            Q = child_vsum / child_visits.clamp(min=1).double()
            explore = C * torch.sqrt(
                torch.log(parent_visits.unsqueeze(1).double().clamp(min=1))
                / child_visits.clamp(min=1).double()
            )
            ucb = Q + explore
            ucb[child_visits == 0] = float('inf')  # explore unvisited first
            ucb[~valid] = float('-inf')

            best_local = ucb.argmax(dim=1)
            best_global = children_ids[torch.arange(B_int), best_local]
            current[int_idx] = best_global

        return current

    def _backpropagate(self, visit_count, value_sum, parent, max_depth,
                       particle_idx, node_idx, values):
        """Update visit counts and value sums from node to root."""
        values_t = torch.from_numpy(values.astype(np.float64))
        current = node_idx.clone()

        for _ in range(max_depth + 1):
            visit_count[particle_idx, current] += 1
            value_sum[particle_idx, current] += values_t
            par = parent[particle_idx, current]
            at_root = par < 0
            if at_root.all():
                break
            current = torch.where(at_root, current, par)

    def _mcts_select_final(self, sequences, scores, active, B, L, M,
                           x_batch, original_batch, bio_filter_fn):
        """Pick best leaf per particle from MCTS tree."""
        # All non-root active nodes are candidates
        valid = active.clone()
        valid[:, 0] = False  # exclude root

        # Edit cap (vectorized)
        if self.max_edit_frac < 1.0 and original_batch is not None:
            max_edits = int(self.max_edit_frac * L)
            hamming = (sequences != original_batch.unsqueeze(1)).sum(dim=-1)  # (B, M)
            valid &= (hamming <= max_edits)

        # Per-step delta cap: hamming from root (current particle before MCTS)
        if self.max_edits_per_step > 0:
            step_hamming = (sequences != x_batch.cpu().unsqueeze(1)).sum(dim=-1)  # (B, M)
            valid &= (step_hamming <= self.max_edits_per_step)

        # Bio-filter (batch all valid nodes)
        if bio_filter_fn is not None:
            valid_idx = valid.nonzero()  # (num_valid, 2)
            if len(valid_idx) > 0:
                valid_seqs = sequences[valid_idx[:, 0], valid_idx[:, 1]]
                bio_ok = bio_filter_fn(valid_seqs.numpy())
                bio_ok_t = torch.from_numpy(bio_ok)
                reject_mask = ~bio_ok_t
                if reject_mask.any():
                    reject_idx = valid_idx[reject_mask]
                    valid[reject_idx[:, 0], reject_idx[:, 1]] = False

        # Selection criterion
        if self.edit_select_mode == "efficiency" and original_batch is not None:
            hamming = (sequences != original_batch.unsqueeze(1)).sum(dim=-1).float()
            criterion = scores / (1.0 + hamming)
        else:
            criterion = scores.clone()

        criterion[~valid] = -float('inf')
        best_node = criterion.argmax(dim=1)  # (B,)
        x_new = sequences[torch.arange(B), best_node]

        # Fallback: no valid node → keep parent
        no_valid = ~valid.any(dim=1)
        x_new[no_valid] = x_batch[no_valid].cpu()

        return x_new

    # ======================================================================
    # v7 helper methods
    # ======================================================================

    def _compute_pll_fitness(self, population: torch.Tensor) -> np.ndarray:
        """Compute pseudo-log-likelihood as fitness (3A: oracle-free).

        For each particle, mask each position, predict with MDLM, and
        accumulate log P(true_token). Uses a single forward pass by
        randomly masking noise_fraction positions (approximation).

        Returns:
            (N,) np.array of PLL scores.
        """
        N, L = population.shape
        device = self.device
        MASK_TOKEN = 4

        # Approximate PLL: mask noise_fraction positions and score those
        nf = self._default_nf
        x = population.to(device)
        mask = torch.rand(N, L, device=device) < nf
        x_noisy = x.clone()
        x_noisy[mask] = MASK_TOKEN

        sigma_val = -torch.log(torch.tensor(1.0 - nf + 1e-8, device=device))
        sigma = sigma_val.expand(N)

        with torch.no_grad():
            log_probs = self.mutate_fn.mdlm.forward(x_noisy, sigma)  # (N, L, 5)

        # Log prob of true token at masked positions
        probs = log_probs[:, :, :4].exp()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        log_p = probs.log().clamp(min=-20)  # (N, L, 4)

        # Gather true token log-probs
        true_tokens = x.unsqueeze(-1)  # (N, L, 1)
        true_logp = log_p.gather(2, true_tokens).squeeze(-1)  # (N, L)

        # Sum only at masked positions
        pll = (true_logp * mask.float()).sum(dim=1)  # (N,)
        return pll.cpu().numpy()

    def _detect_and_remutate_duplicates(
        self,
        population: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        bio_filter_fn: Optional[Callable],
        threshold: int,
        nf_mult: float,
        base_nf: float,
        mutate_fn_override: Optional[Callable] = None,
        max_rounds: int = 2,
    ) -> int:
        """Re-mutate near-duplicate particles for diversity enforcement.

        After mutation, detect particles that are too similar (hamming < threshold)
        and re-mutate them with higher noise fraction. This is a rejection sampling
        variant on the mutation kernel conditioned on novelty.

        Args:
            population: (N, L) current particles (mutated in-place).
            labels: (N, 1) activity labels.
            batch_size: Mutation batch size.
            bio_filter_fn: Bio filter callback.
            threshold: Hamming distance threshold for near-duplicate detection.
            nf_mult: Multiply noise fraction by this for re-mutation.
            base_nf: Current noise fraction.
            mutate_fn_override: Current mutate function (from factory).
            max_rounds: Maximum re-mutation attempts per step.

        Returns:
            Total number of particles re-mutated across all rounds.
        """
        N = population.shape[0]
        total_remutated = 0

        for round_idx in range(max_rounds):
            # Compute pairwise hamming on a subsample for speed
            # Use sorted-hash trick: hash each sequence, find collisions
            pop_np = population.numpy() if not population.is_cuda else population.cpu().numpy()

            # Efficient: compare each particle to its nearest neighbor
            # Subsample for O(n*s) instead of O(n^2)
            L = population.shape[1]
            max_elements = 100_000_000
            subsample = min(max(10, max_elements // (N * L)), N)
            sample_idx = np.random.choice(N, subsample, replace=False)
            sample = pop_np[sample_idx]  # (s, L)

            # For each particle, compute hamming to all subsample members
            # (N, 1, L) != (1, s, L) -> (N, s, L) -> sum -> (N, s)
            hamming = (pop_np[:, None, :] != sample[None, :, :]).sum(axis=2)  # (N, s)
            # Set self-distances to large value
            for i, si in enumerate(sample_idx):
                hamming[si, i] = 999

            min_hamming = hamming.min(axis=1)  # (N,)
            dup_mask = min_hamming < threshold
            n_dups = dup_mask.sum()

            if n_dups == 0:
                break

            # Re-mutate duplicates with higher noise
            dup_indices = np.where(dup_mask)[0]
            remut_nf = base_nf * nf_mult
            if self.mutate_fn_factory is not None:
                remut_fn = self.mutate_fn_factory(remut_nf, guidance_weight=None, eta=0.0)
            else:
                remut_fn = self.mutate_fn

            dup_pop = population[dup_indices]
            dup_labels = labels[dup_indices] if labels is not None else None

            # Mutate in batches
            new_seqs = []
            for start in range(0, len(dup_indices), batch_size):
                end = min(start + batch_size, len(dup_indices))
                batch = dup_pop[start:end].to(self.device)
                batch_labels = dup_labels[start:end].to(self.device) if dup_labels is not None else None
                mutated = remut_fn(self.model, batch, batch_labels)
                new_seqs.append(mutated.cpu())
            new_seqs = torch.cat(new_seqs, dim=0)

            # Bio filter: keep parent if mutation fails filter
            if bio_filter_fn is not None:
                bio_ok = bio_filter_fn(new_seqs.numpy())
                new_seqs[~bio_ok] = dup_pop[~bio_ok]

            population[dup_indices] = new_seqs
            total_remutated += n_dups
            # print(f"  [dedup r{round_idx+1}] {n_dups} near-duplicates re-mutated (nf={remut_nf:.3f})", flush=True)

        return total_remutated

    @staticmethod
    def _compute_conformity(population: torch.Tensor) -> np.ndarray:
        """Compute per-particle conformity (3B: diversity regularization).

        Conformity = fraction of positions matching the population mode
        (most common token at each position). Higher = more generic.

        Returns:
            (N,) np.array of conformity scores in [0, 1].
        """
        pop_np = population.numpy()
        N, L = pop_np.shape
        # Mode at each position
        mode = np.zeros(L, dtype=pop_np.dtype)
        for pos in range(L):
            counts = np.bincount(pop_np[:, pos], minlength=4)
            mode[pos] = counts.argmax()
        # Fraction matching mode per particle
        conformity = (pop_np == mode[None, :]).mean(axis=1)
        return conformity

    def _accumulate_surrogate_data(self, population: torch.Tensor,
                                    oracle_scores: np.ndarray, step: int):
        """Accumulate data and periodically re-fit k-mer surrogate (3C)."""
        from bio_plausibility import kmer_profile

        pop_np = population.numpy()
        kmers = kmer_profile(pop_np, k=3)  # (N, 64)
        self._surrogate_data_X.append(kmers)
        self._surrogate_data_y.append(oracle_scores)

        # Re-fit every 5 steps
        if step - self._surrogate_fit_step >= 5 and step >= 5:
            X = np.concatenate(self._surrogate_data_X, axis=0)
            y = np.concatenate(self._surrogate_data_y, axis=0)
            # Keep last 50K points
            if len(y) > 50000:
                X = X[-50000:]
                y = y[-50000:]
            from sklearn.linear_model import Ridge
            self._surrogate_model = Ridge(alpha=1.0)
            self._surrogate_model.fit(X, y)
            self._surrogate_fit_step = step
            r2 = self._surrogate_model.score(X, y)
            print(f" [surrogate r²={r2:.3f} n={len(y)}]", end='', flush=True)

    def _record_step(self, history: GPAHistory, beta: float, delta_beta: float,
                     ess: float, N: int, resampled: bool, log_z_inc: float,
                     oracle_scores: np.ndarray, gc_fractions: np.ndarray,
                     population: torch.Tensor, diversity_subsample: int,
                     wall_time: float):
        """Record diagnostics for one step."""
        history.beta.append(beta)
        history.delta_beta.append(delta_beta)
        history.ess.append(ess)
        history.ess_fraction.append(ess / N)
        history.resampled.append(resampled)
        history.log_z_increments.append(log_z_inc)
        history.oracle_mean.append(float(oracle_scores.mean()))
        history.oracle_max.append(float(oracle_scores.max()))
        history.oracle_std.append(float(oracle_scores.std()))
        history.oracle_p99.append(float(np.percentile(oracle_scores, 99)))
        history.gc_mean.append(float(gc_fractions.mean()))
        history.wall_time.append(wall_time)

        # Diversity: pairwise identity on subsample
        seq_np = population.numpy()
        N_pop = seq_np.shape[0]
        n_unique = len(np.unique(seq_np, axis=0))
        history.n_unique.append(n_unique)

        if N_pop > 1:
            sub_n = min(diversity_subsample, N_pop)
            idx = np.random.choice(N_pop, sub_n, replace=False)
            sub = seq_np[idx]
            # Pairwise identity: mean fraction of matching positions
            # Use vectorized upper-triangle comparison
            n_pairs = sub_n * (sub_n - 1) // 2
            L = sub.shape[1]
            if n_pairs > 0 and n_pairs * L <= 5_000_000:
                # Direct computation for moderate sizes
                matches = 0.0
                L = sub.shape[1]
                for j in range(sub_n):
                    match_counts = np.sum(sub[j+1:] == sub[j], axis=1)
                    matches += match_counts.sum()
                pw_id = matches / (n_pairs * L)
            else:
                # Sample pairs for large populations
                n_sample_pairs = min(5000, n_pairs)
                idx_a = np.random.randint(0, sub_n, n_sample_pairs)
                idx_b = np.random.randint(0, sub_n, n_sample_pairs)
                # Avoid self-pairs
                mask = idx_a != idx_b
                idx_a, idx_b = idx_a[mask], idx_b[mask]
                if len(idx_a) > 0:
                    pw_id = float(np.mean(sub[idx_a] == sub[idx_b]))
                else:
                    pw_id = 0.0
            history.pw_identity.append(pw_id)
        else:
            history.pw_identity.append(0.0)

        # K-mer composition tracking
        if self.hc_kmer_weight > 0 and self.ref_kmer_profile is not None:
            kmer_r = _fast_kmer_corr(seq_np, self.ref_kmer_profile, k=3)
            history.kmer_r_mean.append(float(kmer_r.mean()))
        else:
            history.kmer_r_mean.append(0.0)
