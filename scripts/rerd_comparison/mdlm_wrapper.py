"""
MDLM (Masked Discrete Diffusion) wrapper for GPA mutation operator.

Wraps the SVDD pre-trained MDLM diffusion model to provide the
mutate_fn(model, x_clean, labels) -> x_new interface expected by
DiffusionPopulationAnnealer.

The MDLM model uses absorbing-state (mask token = 4) diffusion with
a CNN backbone. Mutation works by:
  1. Masking a fraction of positions
  2. Predicting log-probs at masked positions via the CNN
  3. (Optional) Adding DPS oracle gradient to guide predictions
  4. Sampling new tokens at masked positions
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np


def load_mdlm(checkpoint_path, svdd_dir, device="cuda"):
    """Load pre-trained MDLM diffusion model from SVDD checkpoint.

    Args:
        checkpoint_path: Path to last.ckpt (DNA_Diffusion:v0).
        svdd_dir: Path to cloned SVDD repo (needed for imports).
        device: Device to load model on.

    Returns:
        Diffusion model instance (eval mode, on device).
    """
    if svdd_dir not in sys.path:
        sys.path.insert(0, svdd_dir)

    try:
        import diffusion_gosai
    except ImportError:
        # The SVDD clone is no longer on disk (its ~/SVDD tree was removed during a
        # home cleanup). SVDD vendored this module from DRAKES, so fall back to the
        # DRAKES copy, which is byte-compatible for load_from_checkpoint: the class
        # path recorded in the ckpt is resolved by us, not by pickle.
        import os as _os
        drakes_dna = _os.environ.get(
            "DRAKES_DNA_DIR",
            _os.path.expanduser("~/external/DRAKES/drakes_dna"))
        if not _os.path.isdir(drakes_dna):
            raise ImportError(
                "Neither `diffusion_gosai` (SVDD) nor the DRAKES fallback is "
                f"available. Looked for DRAKES at {drakes_dna}; set DRAKES_DNA_DIR.")
        if drakes_dna not in sys.path:
            sys.path.insert(0, drakes_dna)
        import diffusion_gosai_update as diffusion_gosai
        print(f"  [mdlm_wrapper] SVDD not found; using DRAKES module at {drakes_dna}")
    from omegaconf import OmegaConf

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
    model = diffusion_gosai.Diffusion.load_from_checkpoint(
        checkpoint_path, config=config, map_location=device
    )
    model.eval()
    model.to(device)
    return model


class MDLMMutator:
    """Wraps SVDD MDLM for use as GPA mutation operator (no DPS)."""

    MASK_TOKEN = 4

    def __init__(self, mdlm_model, noise_fraction=0.10):
        """
        Args:
            mdlm_model: Loaded SVDD Diffusion instance.
            noise_fraction: Fraction of positions to mask and re-predict.
        """
        self.mdlm = mdlm_model
        self.noise_fraction = noise_fraction

    def __call__(self, model_unused, x_clean, labels, branch_factor=1, **kwargs):
        """GPA mutation interface.

        Args:
            model_unused: Ignored (MDLM is stored internally).
            x_clean: (B, L) long tensor, tokens {0..3}, on device.
            labels: (B, 1) float tensor, ignored (unconditional model).
            branch_factor: If >1, return (K, B, L) with K independent samples
                from the same forward pass. If 1, return (B, L) as usual.

        Returns:
            (B, L) long tensor with mutated sequences (branch_factor=1), or
            (K, B, L) tensor with K branches (branch_factor>1).
        """
        B, L = x_clean.shape
        device = x_clean.device

        # 1. Randomly mask positions
        mask = torch.rand(B, L, device=device) < self.noise_fraction
        x_noisy = x_clean.clone()
        x_noisy[mask] = self.MASK_TOKEN

        # 2. Compute sigma for this masking level
        # LogLinear schedule: move_chance = 1 - exp(-sigma)
        # => sigma = -log(1 - move_chance)
        sigma_val = -torch.log(torch.tensor(
            1.0 - self.noise_fraction + 1e-8, device=device
        ))
        sigma = sigma_val.expand(B)

        # 3. Get log-probs from MDLM
        with torch.no_grad():
            log_probs = self.mdlm.forward(x_noisy, sigma)  # (B, L, 5)

        # Extract ACGT probs (drop mask channel), renormalize
        probs = log_probs[:, :, :4].exp()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # 4. Sample at masked positions only
        if branch_factor <= 1:
            x_new = x_clean.clone()
            if mask.any():
                flat_probs = probs[mask]  # (num_masked, 4)
                new_tokens = torch.multinomial(flat_probs, 1).squeeze(-1)
                x_new[mask] = new_tokens
            return x_new
        else:
            # K-branch: sample K times from SAME forward pass
            branches = []
            for k in range(branch_factor):
                x_k = x_clean.clone()
                if mask.any():
                    new_tokens = torch.multinomial(probs[mask], 1).squeeze(-1)
                    x_k[mask] = new_tokens
                branches.append(x_k)
            return torch.stack(branches, dim=0)  # (K, B, L)


class MDLMMutatorDPS(MDLMMutator):
    """MDLM mutation with DPS (Diffusion Posterior Sampling) oracle guidance."""

    def __init__(self, mdlm_model, oracle_model, noise_fraction=0.10,
                 eta=3000.0, tau_start=1.0, tau_end=0.1,
                 gc_dps_weight=0.0, gc_dps_target=0.50,
                 gc_pull_weight=0.0, gc_pull_target=0.50):
        """
        Args:
            mdlm_model: Loaded SVDD Diffusion instance.
            oracle_model: Oracle with .dps_forward(soft_onehot) -> (B,) reward.
            noise_fraction: Fraction of positions to mask.
            eta: DPS gradient step size.
            tau_start: Gumbel-softmax temperature (not annealed in single-step).
            tau_end: Not used in single-step, kept for interface compatibility.
            gc_dps_weight: Weight for GC centering loss in DPS gradient (0=disabled).
            gc_dps_target: Target GC fraction for GC centering (default 0.50).
            gc_pull_weight: Direct GC pull bias weight (0=disabled). Applied as
                additive log-prob bias decoupled from oracle gradient normalization.
            gc_pull_target: Target GC fraction for direct pull (default 0.50).
        """
        super().__init__(mdlm_model, noise_fraction)
        self.oracle = oracle_model
        self.eta = eta
        self.tau = tau_start
        self.gc_dps_weight = gc_dps_weight
        self.gc_dps_target = gc_dps_target
        self.gc_pull_weight = gc_pull_weight
        self.gc_pull_target = gc_pull_target

    def __call__(self, model_unused, x_clean, labels, branch_factor=1, **kwargs):
        """GPA mutation with DPS guidance.

        Same interface as MDLMMutator but adds oracle gradient to log-probs
        before sampling. If branch_factor>1, returns (K, B, L) with K
        independent samples from the same guided distribution.
        """
        B, L = x_clean.shape
        device = x_clean.device

        # 1. Mask positions
        mask = torch.rand(B, L, device=device) < self.noise_fraction
        x_noisy = x_clean.clone()
        x_noisy[mask] = self.MASK_TOKEN

        # 2. Compute sigma
        sigma_val = -torch.log(torch.tensor(
            1.0 - self.noise_fraction + 1e-8, device=device
        ))
        sigma = sigma_val.expand(B)

        # 3. Get log-probs from MDLM (no grad needed for model)
        with torch.no_grad():
            log_probs_raw = self.mdlm.forward(x_noisy, sigma)  # (B, L, 5)

        # 4. DPS gradient via differentiable path
        log_probs_acgt = log_probs_raw[:, :, :4].detach().clone()
        log_probs_acgt.requires_grad_(True)

        # Gumbel-softmax relaxation for differentiable sampling
        soft_onehot = F.gumbel_softmax(log_probs_acgt, tau=self.tau, hard=False)
        # (B, L, 4) -> (B, 4, L) for oracle
        soft_input = soft_onehot.permute(0, 2, 1)

        # Oracle forward (differentiable)
        reward = self.oracle.dps_forward(soft_input)  # (B,)
        total = reward.sum()

        # GC centering loss: penalize deviation from target GC
        if self.gc_dps_weight > 0:
            # soft_onehot is (B, L, 4): A=0, C=1, G=2, T=3
            gc_frac = (soft_onehot[:, :, 1] + soft_onehot[:, :, 2]).mean(dim=1)  # (B,)
            gc_loss = self.gc_dps_weight * ((gc_frac - self.gc_dps_target) ** 2).sum()
            total = total - gc_loss  # maximize reward, minimize GC drift

        total.backward()

        grad = log_probs_acgt.grad  # (B, L, 4)

        # Normalize gradient
        grad_norm = grad.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        normalized_grad = grad / grad_norm

        # 5. Apply guidance (overridable by subclasses)
        guided_log_probs = self._apply_guidance(
            log_probs_acgt, normalized_grad, device, x_clean)

        guided_probs = F.softmax(guided_log_probs, dim=-1)

        # 6. Sample at masked positions only
        if branch_factor <= 1:
            x_new = x_clean.clone()
            if mask.any():
                flat_probs = guided_probs[mask]
                new_tokens = torch.multinomial(flat_probs, 1).squeeze(-1)
                x_new[mask] = new_tokens
            return x_new
        else:
            # K-branch: sample K times from SAME guided distribution
            branches = []
            for k in range(branch_factor):
                x_k = x_clean.clone()
                if mask.any():
                    new_tokens = torch.multinomial(guided_probs[mask], 1).squeeze(-1)
                    x_k[mask] = new_tokens
                branches.append(x_k)
            return torch.stack(branches, dim=0)  # (K, B, L)

    def _apply_guidance(self, log_probs_acgt, normalized_grad, device, x_clean):
        """Apply DPS gradient to MDLM log-probs. Override in subclasses."""
        B, L, _ = log_probs_acgt.shape
        guided_log_probs = log_probs_acgt.detach() + self.eta * normalized_grad

        # Direct GC pull bias (decoupled from oracle gradient normalization)
        if self.gc_pull_weight > 0:
            gc_frac = ((x_clean == 1) | (x_clean == 2)).float().mean(dim=1)  # (B,)
            gc_pull = self.gc_pull_weight * (self.gc_pull_target - gc_frac)   # (B,)
            gc_bias = torch.zeros(B, L, 4, device=device)
            gc_bias[:, :, 0] = -gc_pull.unsqueeze(1)  # A: suppress when GC low
            gc_bias[:, :, 1] =  gc_pull.unsqueeze(1)   # C: boost when GC low
            gc_bias[:, :, 2] =  gc_pull.unsqueeze(1)   # G: boost when GC low
            gc_bias[:, :, 3] = -gc_pull.unsqueeze(1)  # T: suppress when GC low
            guided_log_probs = guided_log_probs + gc_bias
        return guided_log_probs


class MDLMMutatorConfGated(MDLMMutatorDPS):
    """DPS with MDLM entropy gating — grammar positions get less oracle gradient."""

    def __init__(self, mdlm_model, oracle_model, gate_sharpness=1.0, **kwargs):
        super().__init__(mdlm_model, oracle_model, **kwargs)
        self.gate_sharpness = gate_sharpness
        self._last_gate_mean = 0.0
        self._last_effective_eta = 0.0

    def _apply_guidance(self, log_probs_acgt, normalized_grad, device, x_clean):
        B, L, _ = log_probs_acgt.shape

        # Per-position MDLM entropy
        probs_mdlm = F.softmax(log_probs_acgt.detach(), dim=-1)  # (B, L, 4)
        log_p = probs_mdlm.log().clamp(min=-20)
        entropy = -(probs_mdlm * log_p).sum(dim=-1)  # (B, L)
        max_entropy = torch.log(torch.tensor(4.0, device=device))  # ln(4) ≈ 1.386
        gate = (entropy / max_entropy).clamp(0, 1)  # (B, L) in [0, 1]
        if self.gate_sharpness != 1.0:
            gate = gate.pow(self.gate_sharpness)

        # Diagnostics
        self._last_gate_mean = float(gate.mean())
        self._last_effective_eta = float((self.eta * gate).mean())

        # Position-gated DPS: grammar positions (low entropy) get near-zero DPS
        guided_log_probs = log_probs_acgt.detach() + self.eta * gate.unsqueeze(-1) * normalized_grad

        # GC pull (same as parent)
        if self.gc_pull_weight > 0:
            gc_frac = ((x_clean == 1) | (x_clean == 2)).float().mean(dim=1)
            gc_pull = self.gc_pull_weight * (self.gc_pull_target - gc_frac)
            gc_bias = torch.zeros(B, L, 4, device=device)
            gc_bias[:, :, 0] = -gc_pull.unsqueeze(1)
            gc_bias[:, :, 1] =  gc_pull.unsqueeze(1)
            gc_bias[:, :, 2] =  gc_pull.unsqueeze(1)
            gc_bias[:, :, 3] = -gc_pull.unsqueeze(1)
            guided_log_probs = guided_log_probs + gc_bias
        return guided_log_probs


class MDLMMutatorKLConstrained(MDLMMutatorDPS):
    """DPS with per-position KL budget — projects back toward MDLM when KL exceeds epsilon."""

    def __init__(self, mdlm_model, oracle_model, kl_budget=1.0, kl_mode="per_position", **kwargs):
        super().__init__(mdlm_model, oracle_model, **kwargs)
        self.kl_budget = kl_budget
        self.kl_mode = kl_mode
        self._last_kl_mean = 0.0
        self._last_kl_max = 0.0
        self._last_alpha_mean = 0.0
        self._last_frac_clamped = 0.0

    def _apply_guidance(self, log_probs_acgt, normalized_grad, device, x_clean):
        B, L, _ = log_probs_acgt.shape

        # Unconstrained guided log-probs (standard DPS)
        guided_raw = log_probs_acgt.detach() + self.eta * normalized_grad

        # KL(p_guided || p_mdlm) per position
        p_mdlm = F.softmax(log_probs_acgt.detach(), dim=-1)       # (B, L, 4)
        p_guided = F.softmax(guided_raw, dim=-1)                   # (B, L, 4)
        log_p_mdlm = p_mdlm.log().clamp(min=-20)
        log_p_guided = p_guided.log().clamp(min=-20)
        kl_per_pos = (p_guided * (log_p_guided - log_p_mdlm)).sum(dim=-1)  # (B, L)

        # Constraint satisfaction ratio
        if self.kl_mode == "per_position":
            alpha = (self.kl_budget / kl_per_pos.clamp(min=1e-8)).clamp(max=1.0)  # (B, L)
        else:  # "global"
            mean_kl = kl_per_pos.mean(dim=1, keepdim=True)  # (B, 1)
            alpha = (self.kl_budget / mean_kl.clamp(min=1e-8)).clamp(max=1.0).expand_as(kl_per_pos)

        # Diagnostics
        self._last_kl_mean = float(kl_per_pos.mean())
        self._last_kl_max = float(kl_per_pos.max())
        self._last_alpha_mean = float(alpha.mean())
        self._last_frac_clamped = float((kl_per_pos > self.kl_budget).float().mean())

        # Interpolate: alpha=1 keeps guided, alpha<1 pulls back toward MDLM
        guided_log_probs = alpha.unsqueeze(-1) * guided_raw + (1 - alpha.unsqueeze(-1)) * log_probs_acgt.detach()

        # GC pull (same as parent)
        if self.gc_pull_weight > 0:
            gc_frac = ((x_clean == 1) | (x_clean == 2)).float().mean(dim=1)
            gc_pull = self.gc_pull_weight * (self.gc_pull_target - gc_frac)
            gc_bias = torch.zeros(B, L, 4, device=device)
            gc_bias[:, :, 0] = -gc_pull.unsqueeze(1)
            gc_bias[:, :, 1] =  gc_pull.unsqueeze(1)
            gc_bias[:, :, 2] =  gc_pull.unsqueeze(1)
            gc_bias[:, :, 3] = -gc_pull.unsqueeze(1)
            guided_log_probs = guided_log_probs + gc_bias
        return guided_log_probs
