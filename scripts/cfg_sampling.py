"""
CFG Sampling Module for D3-DNA Discrete Diffusion

This module provides Classifier-Free Guidance (CFG) score functions and samplers.
It implements the dual forward pass approach: one conditional, one unconditional,
combined via logit-level interpolation.

Supports multiple CFG strategies:
- logit: Standard logit-space interpolation
- prob_geometric: Geometric interpolation in probability space (rescaled CFG)
- prob_linear: Linear interpolation in probability space

Additional features:
- Guidance annealing: ramp guidance strength based on noise level
- GC composition penalty: counteract GC bias at high guidance weights

Key functions:
- get_cfg_score_fn: Creates a CFG-aware score function
- get_cfg_pc_sampler: Creates a PC sampler that uses CFG score function
"""

import math
import torch
import torch.nn.functional as F
from utils.catsample import sample_categorical
from scripts.sampling import get_predictor, Predictor, Denoiser


def _compute_effective_w(guidance_weight, sigma, anneal_strategy='none', anneal_start=0.5, sigma_max=20.0, w_min=1.0):
    """Compute effective guidance weight based on annealing schedule.

    Args:
        guidance_weight: Base guidance weight.
        sigma: Current noise level (scalar or 1D tensor).
        anneal_strategy: 'none', 'linear', or 'cosine'.
        anneal_start: Fraction of sigma_max below which guidance ramps up.
            At sigma >= anneal_start * sigma_max, w_eff = w_min (floor guidance).
            At sigma = 0, w_eff = guidance_weight (full guidance).
        sigma_max: Maximum sigma value from noise schedule.
        w_min: Minimum guidance weight (floor) during annealing. Default 1.0.

    Returns:
        Effective guidance weight (scalar).
    """
    if anneal_strategy == 'none':
        return guidance_weight

    # Use max sigma value from the batch
    sigma_val = sigma.max().item() if sigma.numel() > 1 else sigma.item()
    threshold = anneal_start * sigma_max

    if sigma_val >= threshold:
        return w_min

    # progress: 0 at threshold, 1 at sigma=0
    progress = 1.0 - sigma_val / threshold

    if anneal_strategy == 'linear':
        ramp = progress
    elif anneal_strategy == 'cosine':
        ramp = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        return guidance_weight

    return w_min + (guidance_weight - w_min) * ramp


def _apply_gc_penalty(guided, x, gc_penalty, gc_target_low=0.40, gc_target_high=0.60):
    """Adaptive GC governor with dead-zone band.

    No penalty when GC is within [gc_target_low, gc_target_high].
    Outside the band, penalty scales proportionally to the drift magnitude:
    - GC < gc_target_low → penalizes A/T logits (push GC up)
    - GC > gc_target_high → penalizes C/G logits (push GC down)
    Token encoding: A=0, C=1, G=2, T=3.

    Args:
        guided: Guided logits, shape (batch, seq_len, num_classes).
        x: Current token indices, shape (batch, seq_len).
        gc_penalty: Penalty gain k_p (stiffness of the spring).
        gc_target_low: Lower GC guardrail (default 0.40).
        gc_target_high: Upper GC guardrail (default 0.60).

    Returns:
        Adjusted guided logits.
    """
    if gc_penalty == 0.0:
        return guided

    # Compute current GC fraction per sequence
    gc_mask = (x == 1) | (x == 2)  # C=1, G=2
    current_gc = gc_mask.float().mean(dim=1, keepdim=True)  # (batch, 1)

    # High GC drift: penalize C (1) and G (2) tokens
    high_drift = (current_gc - gc_target_high).clamp(min=0.0)  # (batch, 1)
    if high_drift.any():
        penalty = gc_penalty * high_drift  # (batch, 1) broadcasts to (batch, seq_len)
        guided[:, :, 1] = guided[:, :, 1] - penalty  # C
        guided[:, :, 2] = guided[:, :, 2] - penalty  # G

    # Low GC drift: penalize A (0) and T (3) tokens
    low_drift = (gc_target_low - current_gc).clamp(min=0.0)  # (batch, 1)
    if low_drift.any():
        penalty = gc_penalty * low_drift  # (batch, 1)
        guided[:, :, 0] = guided[:, :, 0] - penalty  # A
        guided[:, :, 3] = guided[:, :, 3] - penalty  # T

    return guided


def _apply_gc_penalty_logit_norm(guided, x, gamma, gc_target_low=0.40, gc_target_high=0.60):
    """GC penalty scaled by per-position logit standard deviation.

    Adapts penalty magnitude to the model's current logit scale, which varies
    across noise levels (small at high noise, large near t=0).

    Args:
        guided: Post-CFG logits, shape (batch, seq_len, num_classes).
        x: Current token indices, shape (batch, seq_len).
        gamma: Scaling gain (replaces gc_penalty as tunable hyperparameter).
        gc_target_low: Lower GC guardrail.
        gc_target_high: Upper GC guardrail.

    Returns:
        Adjusted guided logits.
    """
    if gamma == 0.0:
        return guided

    # Per-position logit scale
    logit_scale = guided.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)

    # Current GC fraction per sequence
    gc_mask = (x == 1) | (x == 2)
    current_gc = gc_mask.float().mean(dim=1, keepdim=True)  # (batch, 1)

    # High GC drift: penalize C and G
    high_drift = (current_gc - gc_target_high).clamp(min=0.0)  # (batch, 1)
    if high_drift.any():
        penalty = gamma * logit_scale[:, :, 0] * high_drift  # (batch, seq_len)
        guided[:, :, 1] = guided[:, :, 1] - penalty  # C
        guided[:, :, 2] = guided[:, :, 2] - penalty  # G

    # Low GC drift: penalize A and T
    low_drift = (gc_target_low - current_gc).clamp(min=0.0)  # (batch, 1)
    if low_drift.any():
        penalty = gamma * logit_scale[:, :, 0] * low_drift  # (batch, seq_len)
        guided[:, :, 0] = guided[:, :, 0] - penalty  # A
        guided[:, :, 3] = guided[:, :, 3] - penalty  # T

    return guided


def _apply_gc_penalty_pi(guided, x, gc_penalty, gc_target_low=0.40, gc_target_high=0.60,
                          integral_state=None):
    """GC penalty with proportional-integral (PI) controller.

    Adds an integral term that accumulates error over denoising steps,
    allowing the controller to correct persistent GC drift that the
    proportional-only controller cannot eliminate.

    Args:
        guided: Post-CFG logits, shape (batch, seq_len, num_classes).
        x: Current token indices, shape (batch, seq_len).
        gc_penalty: Proportional gain k_p.
        gc_target_low: Lower GC guardrail.
        gc_target_high: Upper GC guardrail.
        integral_state: Mutable dict with 'error_sum' tensor, persists across steps.

    Returns:
        Adjusted guided logits.
    """
    if gc_penalty == 0.0:
        return guided

    ki = 0.1  # Integral gain (ratio relative to gc_penalty)

    gc_mask = (x == 1) | (x == 2)
    current_gc = gc_mask.float().mean(dim=1, keepdim=True)  # (batch, 1)

    # Compute signed drift (positive = too high, negative = too low)
    high_drift = (current_gc - gc_target_high).clamp(min=0.0)
    low_drift = (gc_target_low - current_gc).clamp(min=0.0)

    # Update integral state
    if integral_state is not None:
        if integral_state['error_sum'] is None:
            integral_state['error_sum'] = torch.zeros_like(current_gc)
        # Accumulate: positive for high GC, negative for low GC
        integral_state['error_sum'] = integral_state['error_sum'] + high_drift - low_drift
        integral_state['error_sum'].clamp_(-5.0, 5.0)  # Anti-windup

        integral_high = integral_state['error_sum'].clamp(min=0.0)
        integral_low = (-integral_state['error_sum']).clamp(min=0.0)
    else:
        integral_high = torch.zeros_like(high_drift)
        integral_low = torch.zeros_like(low_drift)

    # High GC: penalize C and G
    if high_drift.any() or integral_high.any():
        penalty = gc_penalty * (high_drift + ki * integral_high)
        guided[:, :, 1] = guided[:, :, 1] - penalty
        guided[:, :, 2] = guided[:, :, 2] - penalty

    # Low GC: penalize A and T
    if low_drift.any() or integral_low.any():
        penalty = gc_penalty * (low_drift + ki * integral_low)
        guided[:, :, 0] = guided[:, :, 0] - penalty
        guided[:, :, 3] = guided[:, :, 3] - penalty

    return guided


def _apply_gc_penalty_entropy(guided, x, gc_penalty, gc_target_low=0.40, gc_target_high=0.60):
    """Entropy-aware GC penalty that protects confident motif positions.

    Scales penalty by per-position model uncertainty (entropy of softmax).
    High-entropy positions (uncertain) get full penalty; low-entropy positions
    (confident motifs) are protected from GC correction.

    Args:
        guided: Post-CFG logits, shape (batch, seq_len, num_classes).
        x: Current token indices, shape (batch, seq_len).
        gc_penalty: Base penalty gain.
        gc_target_low: Lower GC guardrail.
        gc_target_high: Upper GC guardrail.

    Returns:
        Adjusted guided logits.
    """
    if gc_penalty == 0.0:
        return guided

    # Per-position uncertainty from logit distribution
    probs = F.softmax(guided, dim=-1)  # (batch, seq_len, 4)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)  # (batch, seq_len)
    uncertainty = entropy / math.log(4)  # Normalize to [0, 1]
    uncertainty = uncertainty.unsqueeze(-1)  # (batch, seq_len, 1)

    gc_mask = (x == 1) | (x == 2)
    current_gc = gc_mask.float().mean(dim=1, keepdim=True)  # (batch, 1)

    # High GC drift: penalize C and G, weighted by uncertainty
    high_drift = (current_gc - gc_target_high).clamp(min=0.0)  # (batch, 1)
    if high_drift.any():
        penalty = gc_penalty * uncertainty[:, :, 0] * high_drift  # (batch, seq_len)
        guided[:, :, 1] = guided[:, :, 1] - penalty
        guided[:, :, 2] = guided[:, :, 2] - penalty

    # Low GC drift: penalize A and T, weighted by uncertainty
    low_drift = (gc_target_low - current_gc).clamp(min=0.0)
    if low_drift.any():
        penalty = gc_penalty * uncertainty[:, :, 0] * low_drift
        guided[:, :, 0] = guided[:, :, 0] - penalty
        guided[:, :, 3] = guided[:, :, 3] - penalty

    return guided


def _apply_entropy_penalty(guided, x, entropy_penalty, max_base_frac=0.30):
    """Penalize dominant bases to prevent entropy collapse.

    At each step, compute per-sequence base fractions from current x,
    penalize any base that exceeds max_base_frac frequency.

    Args:
        guided: Guided logits, shape (batch, seq_len, num_classes).
        x: Current token indices, shape (batch, seq_len).
        entropy_penalty: Penalty strength (0.0 = off).
        max_base_frac: Maximum allowed fraction per base (default 0.30).

    Returns:
        Adjusted guided logits.
    """
    if entropy_penalty == 0.0:
        return guided
    N, L = x.shape
    for b in range(4):
        frac = (x == b).float().mean(dim=1, keepdim=True)  # (N, 1)
        excess = (frac - max_base_frac).clamp(min=0.0)     # (N, 1)
        if excess.any():
            guided[:, :, b] = guided[:, :, b] - entropy_penalty * excess
    return guided


def _apply_gc_penalty_dispatch(guided, x, gc_penalty, gc_target_low=0.40, gc_target_high=0.60,
                                gc_method='fixed', integral_state=None):
    """Dispatch to the appropriate GC penalty method.

    Args:
        guided: Post-CFG logits.
        x: Current token indices.
        gc_penalty: Penalty gain (gamma for logit_norm).
        gc_target_low: Lower GC guardrail.
        gc_target_high: Upper GC guardrail.
        gc_method: One of 'fixed', 'logit_norm', 'pi', 'entropy'.
        integral_state: Mutable dict for PI controller state.

    Returns:
        Adjusted guided logits.
    """
    if gc_penalty == 0.0:
        return guided
    if gc_method == 'fixed':
        return _apply_gc_penalty(guided, x, gc_penalty, gc_target_low, gc_target_high)
    elif gc_method == 'logit_norm':
        return _apply_gc_penalty_logit_norm(guided, x, gc_penalty, gc_target_low, gc_target_high)
    elif gc_method == 'pi':
        return _apply_gc_penalty_pi(guided, x, gc_penalty, gc_target_low, gc_target_high, integral_state)
    elif gc_method == 'entropy':
        return _apply_gc_penalty_entropy(guided, x, gc_penalty, gc_target_low, gc_target_high)
    else:
        raise ValueError(f"Unknown gc_method: {gc_method}")


def get_cfg_score_fn(model, guidance_weight: float, train: bool = False, sampling: bool = False,
                     cfg_method: str = 'logit', anneal_strategy: str = 'none',
                     anneal_start: float = 0.5, sigma_max: float = 20.0,
                     gc_penalty: float = 0.0, gc_target_low: float = 0.40,
                     gc_target_high: float = 0.60, w_min: float = 1.0,
                     gc_method: str = 'fixed', gc_penalty_gate: dict = None,
                     entropy_penalty: float = 0.0):
    """
    Create a CFG-aware score function.

    For guidance_weight=1.0 or when labels are None: single forward pass (standard).
    For guidance_weight>1.0: two forward passes per step with CFG.

    Args:
        model: The CFGTransformerModel.
        guidance_weight: CFG guidance weight (w). w=1.0 means no guidance.
        train: Training mode flag.
        sampling: If True, returns exp(guided) for sampling.
        cfg_method: CFG interpolation method.
            'logit': Standard logit-space interpolation (default).
            'prob_geometric': Geometric interpolation in probability space.
            'prob_linear': Linear interpolation in probability space.
        anneal_strategy: Guidance annealing schedule.
            'none': Constant guidance (default).
            'linear': Linear ramp from no guidance at high noise to full at low noise.
            'cosine': Cosine ramp (smoother onset).
        anneal_start: Fraction of sigma_max where annealing begins (default 0.5).
        sigma_max: Maximum sigma from noise schedule (default 20.0).
        gc_penalty: Penalty gain k_p for adaptive GC governor (default 0.0 = off).
        gc_target_low: Lower GC guardrail (default 0.40).
        gc_target_high: Upper GC guardrail (default 0.60).
        w_min: Minimum guidance weight (floor) during annealing (default 1.0).
        gc_method: GC penalty strategy. One of 'fixed', 'logit_norm', 'pi', 'entropy'.

    Returns:
        A score function with signature score_fn(x, sigma, labels=None).
    """
    # PI controller state: persists across denoising steps within one trajectory
    integral_state = {'error_sum': None}

    def score_fn(x, sigma, labels=None):
        if train:
            model.train()
        else:
            model.eval()

        with torch.amp.autocast('cuda', dtype=torch.float16):
            sigma = sigma.reshape(-1)

            if guidance_weight == 1.0 or labels is None:
                # Standard single forward pass (no guidance)
                model_output = model(x, labels, train, sigma)
                if isinstance(model_output, tuple):
                    score, _ = model_output
                else:
                    score = model_output

                if sampling:
                    return score.exp()
                return score

            # CFG: dual forward passes
            batch_size = x.shape[0]

            # Pass 1: conditional (normal forward)
            output_cond = model(x, labels, False, sigma)
            if isinstance(output_cond, tuple):
                logits_cond, _ = output_cond
            else:
                logits_cond = output_cond

            # Pass 2: unconditional (force all conditions dropped)
            force_drop = torch.ones(batch_size, device=x.device, dtype=torch.bool)
            output_uncond = model(x, labels, False, sigma, force_drop_ids=force_drop)
            if isinstance(output_uncond, tuple):
                logits_uncond, _ = output_uncond
            else:
                logits_uncond = output_uncond

            # Compute effective guidance weight (may be annealed)
            w_eff = _compute_effective_w(guidance_weight, sigma, anneal_strategy, anneal_start, sigma_max, w_min)

            # CFG interpolation — dispatch on method
            if cfg_method == 'logit':
                # Standard logit-space interpolation
                guided = logits_uncond + w_eff * (logits_cond - logits_uncond)
                guided = guided.clamp(min=-50.0, max=50.0)

            elif cfg_method == 'prob_geometric':
                # Geometric interpolation in probability space (rescaled CFG)
                log_p_cond = F.log_softmax(logits_cond.float(), dim=-1)
                log_p_uncond = F.log_softmax(logits_uncond.float(), dim=-1)
                log_guided = (1.0 - w_eff) * log_p_uncond + w_eff * log_p_cond
                p_guided = F.softmax(log_guided, dim=-1)
                # Convert to score format: score = p * dim (number of classes)
                num_classes = logits_cond.shape[-1]
                score = p_guided * num_classes
                # Apply GC penalty in log-space before returning
                gc_active = gc_penalty > 0.0 and (gc_penalty_gate is None or gc_penalty_gate.get('active', True))
                if gc_active:
                    log_score = score.clamp(min=1e-10).log()
                    log_score = _apply_gc_penalty_dispatch(log_score, x, gc_penalty, gc_target_low, gc_target_high, gc_method=gc_method, integral_state=integral_state)
                    if entropy_penalty > 0.0:
                        log_score = _apply_entropy_penalty(log_score, x, entropy_penalty)
                    if sampling:
                        return log_score.exp()
                    return log_score
                log_score = score.clamp(min=1e-10).log()
                if entropy_penalty > 0.0:
                    log_score = _apply_entropy_penalty(log_score, x, entropy_penalty)
                if sampling:
                    return log_score.exp()
                return log_score

            elif cfg_method == 'prob_linear':
                # Linear interpolation in probability space
                p_cond = F.softmax(logits_cond.float(), dim=-1)
                p_uncond = F.softmax(logits_uncond.float(), dim=-1)
                p_guided = (1.0 - w_eff) * p_uncond + w_eff * p_cond
                p_guided = p_guided.clamp(min=1e-10)
                p_guided = p_guided / p_guided.sum(dim=-1, keepdim=True)
                num_classes = logits_cond.shape[-1]
                score = p_guided * num_classes
                gc_active = gc_penalty > 0.0 and (gc_penalty_gate is None or gc_penalty_gate.get('active', True))
                if gc_active:
                    log_score = score.clamp(min=1e-10).log()
                    log_score = _apply_gc_penalty_dispatch(log_score, x, gc_penalty, gc_target_low, gc_target_high, gc_method=gc_method, integral_state=integral_state)
                    if entropy_penalty > 0.0:
                        log_score = _apply_entropy_penalty(log_score, x, entropy_penalty)
                    if sampling:
                        return log_score.exp()
                    return log_score
                log_score = score.clamp(min=1e-10).log()
                if entropy_penalty > 0.0:
                    log_score = _apply_entropy_penalty(log_score, x, entropy_penalty)
                if sampling:
                    return log_score.exp()
                return log_score

            else:
                raise ValueError(f"Unknown cfg_method: {cfg_method}")

            # Apply GC penalty (logit method path)
            gc_active = gc_penalty > 0.0 and (gc_penalty_gate is None or gc_penalty_gate.get('active', True))
            if gc_active:
                guided = _apply_gc_penalty_dispatch(guided, x, gc_penalty, gc_target_low, gc_target_high, gc_method=gc_method, integral_state=integral_state)

            # Apply entropy penalty (after GC penalty)
            if entropy_penalty > 0.0:
                guided = _apply_entropy_penalty(guided, x, entropy_penalty)

            if sampling:
                return guided.exp()
            return guided

    return score_fn


def get_cfg_pc_sampler(graph, noise, batch_dims, predictor, steps,
                       guidance_weight: float = 1.5,
                       denoise=True, eps=1e-5, device=torch.device('cpu'),
                       proj_fun=lambda x: x, save_elements_list=None,
                       cfg_method: str = 'logit', anneal_strategy: str = 'none',
                       anneal_start: float = 0.5, sigma_max: float = 20.0,
                       gc_penalty: float = 0.0, gc_target_low: float = 0.40,
                       gc_target_high: float = 0.60, w_min: float = 1.0,
                       gc_method: str = 'fixed',
                       gc_penalty_start_frac: float = 0.0):
    """
    Create a PC sampler with CFG support.

    Identical to the existing get_pc_sampler but uses get_cfg_score_fn
    instead of get_score_fn.

    Args:
        graph: Discrete diffusion graph.
        noise: Noise schedule.
        batch_dims: Tuple of (num_samples, sequence_length).
        predictor: Predictor name ('analytic', 'euler', etc.).
        steps: Number of sampling steps.
        guidance_weight: CFG guidance weight.
        denoise: Whether to apply denoising step.
        eps: Small epsilon for numerical stability.
        device: Device for sampling.
        proj_fun: Projection function (identity by default).
        save_elements_list: List of elements to save during sampling.
        cfg_method: CFG interpolation method ('logit', 'prob_geometric', 'prob_linear').
        anneal_strategy: Guidance annealing ('none', 'linear', 'cosine').
        anneal_start: Fraction of sigma_max where annealing begins.
        sigma_max: Maximum sigma from noise schedule.
        gc_penalty: Penalty gain k_p for adaptive GC governor.
        gc_target_low: Lower GC guardrail (default 0.40).
        gc_target_high: Upper GC guardrail (default 0.60).
        w_min: Minimum guidance weight (floor) during annealing.

    Returns:
        A sampler function with signature sampler(model, labels).
    """
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model, labels):
        # GC penalty gate: shared mutable dict between loop and score function
        gc_gate = {'active': gc_penalty_start_frac <= 0.0}

        # Use CFG score function instead of standard score function
        sampling_score_fn = get_cfg_score_fn(
            model, guidance_weight, train=False, sampling=True,
            cfg_method=cfg_method, anneal_strategy=anneal_strategy,
            anneal_start=anneal_start, sigma_max=sigma_max,
            gc_penalty=gc_penalty, gc_target_low=gc_target_low,
            gc_target_high=gc_target_high, w_min=w_min,
            gc_method=gc_method,
            gc_penalty_gate=gc_gate,
        )

        # Sample from limiting distribution
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        # Initialize element storage if requested
        saved_elements = {}
        if save_elements_list:
            for elem in save_elements_list:
                saved_elements[elem] = []

        # Save initial state
        if saved_elements and 'sequence' in saved_elements:
            saved_elements['sequence'].append(x.clone())

        for i in range(steps):
            gc_gate['active'] = (i / steps) >= gc_penalty_start_frac
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)

            # Create save_elements dict excluding 'sequence'
            predictor_save_elements = {}
            if saved_elements:
                for key in saved_elements:
                    if key != 'sequence':
                        predictor_save_elements[key] = saved_elements[key]

            x = predictor.update_fn(sampling_score_fn, x, labels, t, dt,
                                    save_elements=predictor_save_elements if predictor_save_elements else None)

            # Save sequence state after predictor update
            if saved_elements and 'sequence' in saved_elements:
                saved_elements['sequence'].append(x.clone())

        if denoise:
            gc_gate['active'] = True  # Always apply GC penalty on final denoise
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)

            denoiser_save_elements = {}
            if saved_elements:
                for key in saved_elements:
                    if key != 'sequence':
                        denoiser_save_elements[key] = saved_elements[key]

            x = denoiser.update_fn(sampling_score_fn, x, labels, t,
                                   save_elements=denoiser_save_elements if denoiser_save_elements else None)

            if saved_elements and 'sequence' in saved_elements:
                saved_elements['sequence'].append(x.clone())

        if saved_elements:
            return x, saved_elements
        return x

    return pc_sampler


def get_cfg_warm_start_sampler(graph, noise, predictor, steps,
                               guidance_weight: float = 1.5,
                               noise_fraction: float = 0.3,
                               denoise=True, eps=1e-5, device=torch.device('cpu'),
                               proj_fun=lambda x: x,
                               cfg_method: str = 'logit', anneal_strategy: str = 'none',
                               anneal_start: float = 0.5, sigma_max: float = 20.0,
                               gc_penalty: float = 0.0, gc_target_low: float = 0.40,
                               gc_target_high: float = 0.60, w_min: float = 1.0,
                               gc_method: str = 'fixed',
                               gc_penalty_start_frac: float = 0.0,
                               entropy_penalty: float = 0.0):
    """
    Create a warm-start re-diffusion sampler with CFG support.

    Instead of starting from pure noise, this sampler:
    1. Takes clean input sequences (x_clean)
    2. Forward-noises them to a partial noise level (controlled by noise_fraction)
    3. Runs the reverse diffusion from that intermediate point with CFG guidance

    This allows using high guidance weights without collapse, since the
    sequences are anchored to a reasonable starting point.

    Args:
        graph: Discrete diffusion graph.
        noise: Noise schedule (GeometricNoise).
        predictor: Predictor name ('analytic', 'euler', etc.).
        steps: Number of sampling steps (for the partial trajectory).
        guidance_weight: CFG guidance weight.
        noise_fraction: Fraction of corruption to apply (0=clean, 1=fully noised).
            Controls how much of the forward process to apply.
        denoise: Whether to apply denoising step.
        eps: Small epsilon for numerical stability.
        device: Device for sampling.
        proj_fun: Projection function (identity by default).
        cfg_method: CFG interpolation method.
        anneal_strategy: Guidance annealing schedule.
        anneal_start: Fraction of sigma_max where annealing begins.
        sigma_max: Maximum sigma from noise schedule.
        gc_penalty: Penalty gain k_p for adaptive GC governor.
        gc_target_low: Lower GC guardrail (default 0.40).
        gc_target_high: Upper GC guardrail (default 0.60).
        w_min: Minimum guidance weight (floor) during annealing.

    Returns:
        A sampler function with signature sampler(model, x_clean, labels).
    """
    pred = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    # Get sigma bounds from noise schedule
    sigma_min = noise.sigmas[0].item()
    sigma_max_val = noise.sigmas[1].item()

    _printed = [False]

    @torch.no_grad()
    def warm_start_sampler(model, x_clean, labels):
        # Compute forward noise level from noise_fraction
        # sigma = sigma_min^(1-t) * sigma_max^t, so for a given noise_fraction
        # we want move_chance = noise_fraction, where move_chance = 1 - exp(-sigma)
        # So sigma_fwd = -log(1 - noise_fraction)
        noise_frac_clamped = min(max(noise_fraction, 1e-6), 1.0 - 1e-6)
        sigma_fwd = -math.log(1.0 - noise_frac_clamped)

        # Invert the geometric schedule: sigma = sigma_min^(1-t) * sigma_max^t
        # log(sigma) = (1-t)*log(sigma_min) + t*log(sigma_max)
        # t = (log(sigma) - log(sigma_min)) / (log(sigma_max) - log(sigma_min))
        log_ratio = math.log(sigma_max_val) - math.log(sigma_min)
        if log_ratio > 0:
            t_start = (math.log(sigma_fwd) - math.log(sigma_min)) / log_ratio
            t_start = min(max(t_start, eps), 1.0)
        else:
            t_start = 1.0

        if not _printed[0]:
            print(f"Warm-start: noise_fraction={noise_fraction:.3f}, sigma_fwd={sigma_fwd:.4f}, t_start={t_start:.4f}")
            _printed[0] = True

        # Forward-noise clean sequences
        x_clean = x_clean.to(device)
        # sigma needs shape (batch, 1) for broadcasting with (batch, seq_len) in sample_transition
        sigma_fwd_tensor = torch.full((x_clean.shape[0], 1), sigma_fwd, device=device)
        x = graph.sample_transition(x_clean, sigma_fwd_tensor)

        # GC penalty gate: shared mutable dict between loop and score function
        gc_gate = {'active': gc_penalty_start_frac <= 0.0}

        # Build CFG score function
        sampling_score_fn = get_cfg_score_fn(
            model, guidance_weight, train=False, sampling=True,
            cfg_method=cfg_method, anneal_strategy=anneal_strategy,
            anneal_start=anneal_start, sigma_max=sigma_max,
            gc_penalty=gc_penalty, gc_target_low=gc_target_low,
            gc_target_high=gc_target_high, w_min=w_min,
            gc_method=gc_method,
            gc_penalty_gate=gc_gate,
            entropy_penalty=entropy_penalty,
        )

        # Build timesteps from t_start down to eps
        timesteps = torch.linspace(t_start, eps, steps + 1, device=device)
        dt = (t_start - eps) / steps

        for i in range(steps):
            gc_gate['active'] = (i / steps) >= gc_penalty_start_frac
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = pred.update_fn(sampling_score_fn, x, labels, t, dt)

        if denoise:
            gc_gate['active'] = True  # Always apply GC penalty on final denoise
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, labels, t)

        return x

    return warm_start_sampler


def get_cfg_warm_start_sampler_factory(graph, noise, predictor, steps, **kwargs):
    """Factory that returns a warm-start sampler with noise_fraction set at call time.

    Unlike get_cfg_warm_start_sampler which bakes noise_fraction into the closure,
    this returns a callable(nf, guidance_weight=None) -> sampler, enabling noise
    fraction and/or guidance weight annealing in GPA.

    Args:
        graph, noise, predictor, steps: Same as get_cfg_warm_start_sampler.
        **kwargs: All other kwargs passed to get_cfg_warm_start_sampler,
            EXCEPT noise_fraction (which is provided at call time).

    Returns:
        make_sampler: callable(noise_fraction, guidance_weight=None) -> sampler_fn
    """
    # Remove noise_fraction from kwargs if present (it's provided at call time)
    kwargs.pop('noise_fraction', None)
    default_w = kwargs.pop('guidance_weight', None)

    def make_sampler(noise_fraction, guidance_weight=None):
        kw = dict(**kwargs)
        w = guidance_weight if guidance_weight is not None else default_w
        if w is not None:
            kw['guidance_weight'] = w
        return get_cfg_warm_start_sampler(
            graph, noise, predictor, steps,
            noise_fraction=noise_fraction, **kw)

    return make_sampler


def get_cfg_warm_start_dps_sampler_factory(graph, noise, predictor, steps, **kwargs):
    """Factory for DPS warm-start sampler with noise_fraction set at call time.

    Returns a callable(noise_fraction, guidance_weight=None) -> dps_sampler_fn,
    enabling noise fraction and/or guidance weight annealing in GPA with DPS.

    Args:
        graph, noise, predictor, steps: Same as get_cfg_warm_start_dps_sampler.
        **kwargs: All other kwargs passed to get_cfg_warm_start_dps_sampler,
            EXCEPT noise_fraction (provided at call time).

    Returns:
        make_sampler: callable(noise_fraction, guidance_weight=None) -> sampler_fn
    """
    kwargs.pop('noise_fraction', None)
    default_w = kwargs.pop('guidance_weight', None)
    _printed_once = [False]  # shared across all samplers from this factory

    def make_sampler(noise_fraction, guidance_weight=None):
        kw = dict(**kwargs)
        w = guidance_weight if guidance_weight is not None else default_w
        if w is not None:
            kw['guidance_weight'] = w
        kw['_printed_once'] = _printed_once
        return get_cfg_warm_start_dps_sampler(
            graph, noise, predictor, steps,
            noise_fraction=noise_fraction, **kw)

    return make_sampler


def get_cfg_warm_start_dps_sampler(
    graph, noise, predictor, steps,
    # Warm-start / CFG params (same as existing warm-start sampler)
    guidance_weight=1.5, noise_fraction=0.3, denoise=True, eps=1e-5,
    device=torch.device('cpu'), proj_fun=lambda x: x,
    cfg_method='logit', anneal_strategy='none', anneal_start=0.5,
    sigma_max=20.0, gc_penalty=0.0, gc_target_low=0.40,
    gc_target_high=0.60, w_min=1.0, gc_method='fixed',
    gc_penalty_start_frac=0.0,
    # DPS params
    oracle_model=None, eta=1.0, tau_start=1.0, tau_end=0.1,
    guide_start_frac=0.0, gc_grad_weight=0.0, gc_grad_target=0.50,
    entropy_aware=False, eta_schedule='constant',
    # Internal: shared print-once flag from factory
    _printed_once=None,
):
    """
    Warm-start re-diffusion sampler with DPS oracle gradient guidance.

    Combines warm-start initialization (forward-noise clean sequences, then
    reverse-diffuse with CFG) with DPS-style gradient guidance from an oracle
    model. Supports composite gradient (oracle + GC) and entropy-aware scaling.

    Unlike the standard warm-start sampler, this version does NOT use
    @torch.no_grad() globally — gradients are needed for oracle backprop.

    Args:
        (warm-start/CFG args: same as get_cfg_warm_start_sampler)
        oracle_model: Loaded oracle model (must have .model attribute).
        eta: DPS gradient strength.
        tau_start: Gumbel-Softmax initial temperature.
        tau_end: Gumbel-Softmax final temperature.
        guide_start_frac: Fraction of steps before DPS activates (0.0=always).
        gc_grad_weight: Weight for GC gradient in composite loss (0=oracle only).
        gc_grad_target: Target GC fraction for gradient (default 0.50).
        entropy_aware: If True, scale gradient by positional entropy.
        eta_schedule: 'constant', 'linear', or 'sqrt'.

    Returns:
        A sampler function with signature sampler(model, x_clean, labels).
    """
    pred = get_predictor(predictor)(graph, noise)
    denoiser = Denoiser(graph, noise)

    sigma_min = noise.sigmas[0].item()
    sigma_max_val = noise.sigmas[1].item()
    if _printed_once is None:
        _printed_once = [False]  # mutable closure flag

    def warm_start_dps_sampler(model, x_clean, labels):
        # --- 1. Forward-noise x_clean (same as existing warm-start) ---
        noise_frac_clamped = min(max(noise_fraction, 1e-6), 1.0 - 1e-6)
        sigma_fwd = -math.log(1.0 - noise_frac_clamped)

        log_ratio = math.log(sigma_max_val) - math.log(sigma_min)
        if log_ratio > 0:
            t_start = (math.log(sigma_fwd) - math.log(sigma_min)) / log_ratio
            t_start = min(max(t_start, eps), 1.0)
        else:
            t_start = 1.0

        if not _printed_once[0]:
            print(f"Warm-start DPS: noise_fraction={noise_fraction:.3f}, "
                  f"sigma_fwd={sigma_fwd:.4f}, t_start={t_start:.4f}, "
                  f"eta={eta}, gc_grad_weight={gc_grad_weight}, "
                  f"entropy_aware={entropy_aware}, eta_schedule={eta_schedule}")
            _printed_once[0] = True

        x_clean = x_clean.to(device)
        sigma_fwd_tensor = torch.full((x_clean.shape[0], 1), sigma_fwd, device=device)
        x = graph.sample_transition(x_clean, sigma_fwd_tensor)

        # --- 2. Build GC gate + CFG score function ---
        gc_gate = {'active': gc_penalty_start_frac <= 0.0}

        sampling_score_fn = get_cfg_score_fn(
            model, guidance_weight, train=False, sampling=True,
            cfg_method=cfg_method, anneal_strategy=anneal_strategy,
            anneal_start=anneal_start, sigma_max=sigma_max,
            gc_penalty=gc_penalty, gc_target_low=gc_target_low,
            gc_target_high=gc_target_high, w_min=w_min,
            gc_method=gc_method,
            gc_penalty_gate=gc_gate,
        )

        # --- 3. Compute timesteps ---
        timesteps = torch.linspace(t_start, eps, steps + 1, device=device)
        dt = (t_start - eps) / steps

        # --- 4. Denoising loop with inlined predictor + DPS ---
        for i in range(steps):
            gc_gate['active'] = (i / steps) >= gc_penalty_start_frac

            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            curr_sigma = noise(t)[0]
            next_t = t - dt
            next_sigma = noise(next_t)[0]
            dsigma = curr_sigma - next_sigma

            # Standard diffusion step (no grad needed)
            with torch.no_grad():
                score = sampling_score_fn(x, curr_sigma, labels)
                stag_score = graph.staggered_score(score, dsigma)
                probs = stag_score * graph.transp_transition(x, dsigma)

            # DPS gradient guidance
            frac = i / steps
            if frac >= guide_start_frac and eta > 0 and oracle_model is not None:
                log_probs = probs.clamp(min=1e-10).log()
                log_probs_grad = log_probs.detach().requires_grad_(True)

                # Gumbel-Softmax relaxation
                tau = max(tau_end, tau_start * (1.0 - frac))
                soft_onehot = F.gumbel_softmax(log_probs_grad, tau=tau, hard=False)

                # Channel swap for oracle encoding (C↔G)
                soft_oracle = soft_onehot.permute(0, 2, 1).clone()
                ch1 = soft_oracle[:, 1, :].clone()
                ch2 = soft_oracle[:, 2, :].clone()
                soft_oracle[:, 1, :] = ch2
                soft_oracle[:, 2, :] = ch1

                # Oracle forward
                oracle_scores = oracle_model.model(soft_oracle)
                oracle_scores = oracle_scores.squeeze(-1) if oracle_scores.dim() > 1 else oracle_scores

                # Composite objective
                objective = oracle_scores.sum()
                if gc_grad_weight > 0:
                    soft_gc = (soft_onehot[:, :, 1] + soft_onehot[:, :, 2]).mean(dim=1)
                    gc_loss = (soft_gc - gc_grad_target).pow(2)
                    objective = objective - gc_grad_weight * gc_loss.sum()

                objective.backward()
                grad = log_probs_grad.grad

                # Entropy-aware gradient scaling
                if entropy_aware:
                    ent = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1)  # (B, L)
                    max_ent = math.log(4.0)
                    ent_scale = (ent / max_ent).unsqueeze(-1)  # (B, L, 1)
                    grad = grad * ent_scale

                # Adaptive eta schedule
                if eta_schedule == 'constant':
                    eta_t = eta
                elif eta_schedule == 'linear':
                    eta_t = eta * (1.0 - frac)
                elif eta_schedule == 'sqrt':
                    eta_t = eta * math.sqrt(1.0 - frac)
                else:
                    eta_t = eta

                # Per-sample gradient normalization
                grad_norm = grad.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
                guided_log_probs = log_probs + eta_t * (grad / grad_norm)
                guided_probs = F.softmax(guided_log_probs, dim=-1)

                x = sample_categorical(guided_probs)
            else:
                x = sample_categorical(probs)

        # --- 5. Final denoise step ---
        if denoise:
            gc_gate['active'] = True
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            sigma = noise(t)[0]

            with torch.no_grad():
                score = sampling_score_fn(x, sigma, labels)
                stag_score = graph.staggered_score(score, sigma)
                probs = stag_score * graph.transp_transition(x, sigma)

            # Apply DPS on final step too
            if eta > 0 and oracle_model is not None:
                log_probs = probs.clamp(min=1e-10).log()
                log_probs_grad = log_probs.detach().requires_grad_(True)

                tau = tau_end
                soft_onehot = F.gumbel_softmax(log_probs_grad, tau=tau, hard=False)

                soft_oracle = soft_onehot.permute(0, 2, 1).clone()
                ch1 = soft_oracle[:, 1, :].clone()
                ch2 = soft_oracle[:, 2, :].clone()
                soft_oracle[:, 1, :] = ch2
                soft_oracle[:, 2, :] = ch1

                oracle_scores = oracle_model.model(soft_oracle)
                oracle_scores = oracle_scores.squeeze(-1) if oracle_scores.dim() > 1 else oracle_scores

                objective = oracle_scores.sum()
                if gc_grad_weight > 0:
                    soft_gc = (soft_onehot[:, :, 1] + soft_onehot[:, :, 2]).mean(dim=1)
                    gc_loss = (soft_gc - gc_grad_target).pow(2)
                    objective = objective - gc_grad_weight * gc_loss.sum()

                objective.backward()
                grad = log_probs_grad.grad

                if entropy_aware:
                    ent = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1)
                    max_ent = math.log(4.0)
                    ent_scale = (ent / max_ent).unsqueeze(-1)
                    grad = grad * ent_scale

                # Use minimum eta for final step
                if eta_schedule == 'constant':
                    eta_t = eta
                else:
                    eta_t = eta * 0.1  # Very small at end

                grad_norm = grad.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
                guided_log_probs = log_probs + eta_t * (grad / grad_norm)
                guided_probs = F.softmax(guided_log_probs, dim=-1)

                x = sample_categorical(guided_probs)
            else:
                x = sample_categorical(probs)

        return x

    return warm_start_dps_sampler
