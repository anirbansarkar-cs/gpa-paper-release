"""
MDLM wrapper for K562 LentiMPRA GPA pipeline.

Provides:
  - load_mdlm_lentimpra(): Load LentiMPRA-trained MDLM from SGDD checkpoint
  - MDLMMutator: Blind mutation (no DPS)
  - MDLMMutatorDPS: Oracle-guided mutation with DPS
  - MDLMMutatorKLConstrained: KL-budget-constrained DPS (grammar-preserving)

The mutator classes are copied from scripts/rerd_comparison/mdlm_wrapper.py
(they work with any MDLM that has .forward(x_noisy, sigma)).
"""

import os
import sys
import datetime
import random
import string
import itertools

import torch
import torch.nn.functional as F
import numpy as np


def _register_resolvers():
    """Register Hydra/OmegaConf resolvers needed by SGDD configs."""
    from omegaconf import OmegaConf

    def _safe_register(name, fn, **kwargs):
        try:
            OmegaConf.register_new_resolver(name, fn, **kwargs)
        except ValueError:
            pass  # already registered

    _safe_register("uuid", lambda: ''.join(
        random.choice(string.ascii_letters) for _ in range(10)
    ) + '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")), use_cache=False)
    _safe_register('cwd', os.getcwd)
    _safe_register('device_count', lambda: max(1, torch.cuda.device_count()))
    _safe_register('eval', eval)
    _safe_register('div_up', lambda x, y: (x + y - 1) // y)


def load_mdlm_lentimpra(checkpoint_path, sgdd_dir=None, device="cuda"):
    """Load LentiMPRA-trained MDLM.

    Args:
        checkpoint_path: Path to best.ckpt (LentiMPRA MDLM).
        sgdd_dir: Deprecated/ignored. The original SGDD pipeline was deleted in
            the 2026-06 home cleanup; the diffusion_lentimpra module now lives in
            the in-repo package scripts/k562_mdlm_gpa/mdlm_lentimpra/ (faithful
            reconstruction from DRAKES diffusion_gosai). Kept in the signature so
            existing callers passing SGDD_DIR still work.
        device: Device to load model on.

    Returns:
        Diffusion model instance (eval mode, on device).
    """
    # In-repo reconstructed package (replaces deleted SGDD applications/drakes_dna).
    pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "mdlm_lentimpra")
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    from omegaconf import OmegaConf
    import diffusion_lentimpra

    # Register Hydra resolvers needed by config
    _register_resolvers()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
    model = diffusion_lentimpra.Diffusion(config)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # Apply EMA weights (used during validation)
    if "ema" in ckpt and model.ema is not None:
        model.ema.load_state_dict(ckpt["ema"])
        model.ema.copy_to(itertools.chain(
            model.backbone.parameters(), model.noise.parameters()))

    model.eval()
    model.to(device)
    return model


class MDLMMutator:
    """Wraps MDLM for use as GPA mutation operator (no DPS)."""

    MASK_TOKEN = 4

    def __init__(self, mdlm_model, noise_fraction=0.10):
        self.mdlm = mdlm_model
        self.noise_fraction = noise_fraction

    def __call__(self, model_unused, x_clean, labels, branch_factor=1,
                 return_metadata=False):
        """GPA mutation interface.

        Args:
            model_unused: Ignored (MDLM is stored internally).
            x_clean: (B, L) long tensor, tokens {0..3}, on device.
            labels: (B, 1) float tensor, ignored (unconditional model).
            branch_factor: If >1, return (K, B, L) with K independent samples.
            return_metadata: If True, return (x_new, metadata_dict) with
                entropy (B, L) and branch_logprobs (K, B).

        Returns:
            (B, L) or (K, B, L) long tensor with mutated sequences.
            If return_metadata=True, returns (x_new, metadata_dict).
        """
        B, L = x_clean.shape
        device = x_clean.device

        # 1. Randomly mask positions
        mask = torch.rand(B, L, device=device) < self.noise_fraction
        x_noisy = x_clean.clone()
        x_noisy[mask] = self.MASK_TOKEN

        # 2. Compute sigma for this masking level
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

        # Per-position entropy (B, L) — free from existing probs
        entropy = -(probs * probs.log().clamp(min=-20)).sum(dim=-1)

        # 4. Sample at masked positions only
        if branch_factor <= 1:
            x_new = x_clean.clone()
            branch_logprobs = torch.zeros(1, B, device=device)
            if mask.any():
                flat_probs = probs[mask]  # (num_masked, 4)
                new_tokens = torch.multinomial(flat_probs, 1).squeeze(-1)
                x_new[mask] = new_tokens
                # Log-prob of selected tokens at masked positions
                selected_logprobs = flat_probs[torch.arange(len(new_tokens), device=device), new_tokens].log().clamp(min=-20)
                # Sum per particle: scatter masked positions back to (B,)
                particle_idx = mask.nonzero(as_tuple=True)[0]  # which particle each masked pos belongs to
                branch_logprobs[0].scatter_add_(0, particle_idx, selected_logprobs)
            if not return_metadata:
                return x_new
            return x_new, {"entropy": entropy, "branch_logprobs": branch_logprobs}
        else:
            branches = []
            branch_logprobs = torch.zeros(branch_factor, B, device=device)
            for k in range(branch_factor):
                x_k = x_clean.clone()
                if mask.any():
                    flat_probs = probs[mask]  # (num_masked, 4)
                    new_tokens = torch.multinomial(flat_probs, 1).squeeze(-1)
                    x_k[mask] = new_tokens
                    selected_logprobs = flat_probs[torch.arange(len(new_tokens), device=device), new_tokens].log().clamp(min=-20)
                    particle_idx = mask.nonzero(as_tuple=True)[0]
                    branch_logprobs[k].scatter_add_(0, particle_idx, selected_logprobs)
                branches.append(x_k)
            x_new = torch.stack(branches, dim=0)  # (K, B, L)
            if not return_metadata:
                return x_new
            return x_new, {"entropy": entropy, "branch_logprobs": branch_logprobs}


class MDLMMutatorDPS(MDLMMutator):
    """MDLM mutation with DPS (Diffusion Posterior Sampling) oracle guidance."""

    def __init__(self, mdlm_model, oracle_model, noise_fraction=0.10,
                 eta=3000.0, tau_start=1.0, tau_end=0.1,
                 gc_dps_weight=0.0, gc_dps_target=0.50,
                 gc_pull_weight=0.0, gc_pull_target=0.50):
        super().__init__(mdlm_model, noise_fraction)
        self.oracle = oracle_model
        self.eta = eta
        self.tau = tau_start
        self.gc_dps_weight = gc_dps_weight
        self.gc_dps_target = gc_dps_target
        self.gc_pull_weight = gc_pull_weight
        self.gc_pull_target = gc_pull_target

    def __call__(self, model_unused, x_clean, labels, branch_factor=1,
                 return_metadata=False):
        """GPA mutation with DPS guidance."""
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

        # GC centering loss
        if self.gc_dps_weight > 0:
            gc_frac = (soft_onehot[:, :, 1] + soft_onehot[:, :, 2]).mean(dim=1)
            gc_loss = self.gc_dps_weight * ((gc_frac - self.gc_dps_target) ** 2).sum()
            total = total - gc_loss

        total.backward()

        grad = log_probs_acgt.grad  # (B, L, 4)

        # Normalize gradient
        grad_norm = grad.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        normalized_grad = grad / grad_norm

        # 5. Apply guidance
        guided_log_probs = self._apply_guidance(
            log_probs_acgt, normalized_grad, device, x_clean)

        guided_probs = F.softmax(guided_log_probs, dim=-1)

        # Per-position entropy from guided (post-guidance) probs (B, L)
        entropy = -(guided_probs * guided_probs.log().clamp(min=-20)).sum(dim=-1)

        # 6. Sample at masked positions only
        if branch_factor <= 1:
            x_new = x_clean.clone()
            branch_logprobs = torch.zeros(1, B, device=device)
            if mask.any():
                flat_probs = guided_probs[mask]
                new_tokens = torch.multinomial(flat_probs, 1).squeeze(-1)
                x_new[mask] = new_tokens
                selected_logprobs = flat_probs[torch.arange(len(new_tokens), device=device), new_tokens].log().clamp(min=-20)
                particle_idx = mask.nonzero(as_tuple=True)[0]
                branch_logprobs[0].scatter_add_(0, particle_idx, selected_logprobs)
            if not return_metadata:
                return x_new
            return x_new, {"entropy": entropy, "branch_logprobs": branch_logprobs}
        else:
            branches = []
            branch_logprobs = torch.zeros(branch_factor, B, device=device)
            for k in range(branch_factor):
                x_k = x_clean.clone()
                if mask.any():
                    flat_probs = guided_probs[mask]
                    new_tokens = torch.multinomial(flat_probs, 1).squeeze(-1)
                    x_k[mask] = new_tokens
                    selected_logprobs = flat_probs[torch.arange(len(new_tokens), device=device), new_tokens].log().clamp(min=-20)
                    particle_idx = mask.nonzero(as_tuple=True)[0]
                    branch_logprobs[k].scatter_add_(0, particle_idx, selected_logprobs)
                branches.append(x_k)
            x_new = torch.stack(branches, dim=0)  # (K, B, L)
            if not return_metadata:
                return x_new
            return x_new, {"entropy": entropy, "branch_logprobs": branch_logprobs}

    def _apply_guidance(self, log_probs_acgt, normalized_grad, device, x_clean):
        """Apply DPS gradient to MDLM log-probs."""
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
