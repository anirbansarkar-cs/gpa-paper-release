"""DiMamba (MDLM bidirectional) mutation kernel for GPA — DNA-CRAFT enhancer track.

Mirrors the HyenaDNAMutator interface in
`scripts/ctrl_dna_comparison/hyenadna_wrapper.py` but operates on the
MDLM substitution-parameterized Diffusion model:

  - Input: x_clean (B, L) tokens in GPA space {A=0, C=1, G=2, T=3}.
  - Token map is identity for ACGT; MDLM additionally has PAD=4 / MASK=5.
  - Mutation = mask `nf*L` random positions to MASK=5, forward MDLM,
    sample at masked positions from the ACGT logits.
  - DPS variant: Gumbel-softmax over the masked-position ACGT logits ->
    oracle.dps_forward(soft_onehot) -> backward to logits, normalised
    gradient -> guided sampling. Backbone runs under no_grad, exactly
    like HyenaDNAMutatorDPS.

NOTE: Track B (mamba_ssm install) is currently blocked by a CUDA toolkit
mismatch on the cluster (see task #2). This file is implementation-ready
and will succeed once `import mamba_ssm, causal_conv1d` works inside
the main `gpa` env. The GPA runner switches to this wrapper via
`--backbone dimamba`.
"""
from __future__ import annotations

from pathlib import Path
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

MDLM_ROOT = Path(os.path.expandvars("${HOME}/mdlm"))
if str(MDLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MDLM_ROOT))

# Apply the mamba/torch-2.0 compatibility shims (flash_attn stub,
# mamba_ssm.layernorm alias, torch.no_grad compat) BEFORE mdlm/ is imported.
# Lives in `enhancer/mdlm/_compat_shims.py`; we add that dir to sys.path so
# this works regardless of cwd. Idempotent — also imported by train_dimamba.
_OUR_MDLM_DIR = Path(__file__).resolve().parent / "mdlm"
if str(_OUR_MDLM_DIR) not in sys.path:
    sys.path.insert(0, str(_OUR_MDLM_DIR))
import _compat_shims as _mdlm_shims  # noqa: E402, F401

# Token IDs (per plan §3 tokenisation: ACGT={0,1,2,3}, PAD=4, MASK=5)
ACGT_TOKENS = (0, 1, 2, 3)
MASK_TOKEN = 5

# Inference sigma. MDLM was trained with cosine schedule sigma in [0,1];
# during mask-and-predict inference the model is robust to a fixed
# moderate sigma. 0.5 keeps the score distribution well-conditioned.
SIGMA_INFER = 0.5


def load_dimamba(checkpoint_path: str | Path,
                 device: str = "cuda"):
    """Load a Diffusion-LightningModule from MDLM checkpoint into eval mode."""
    import diffusion as mdlm_diffusion  # type: ignore  # noqa: E402
    # The DNA tokenizer is the same one used at pretrain time. Diffusion's
    # __init__ reads `tokenizer.vocab_size`, `mask_token_id`, and `pad_token_id`
    # to size the score / logits heads — passing tokenizer=None blows up.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "mdlm"))
    from dna_dataloader import DNATokenizer  # type: ignore  # noqa: E402

    ckpt = torch.load(str(checkpoint_path), map_location="cpu",
                      weights_only=False)
    if "hyper_parameters" in ckpt:
        cfg = ckpt["hyper_parameters"].get("config")
    else:
        cfg = None
    if cfg is None:
        raise RuntimeError(
            f"checkpoint {checkpoint_path} missing hydra 'config' in "
            "hyper_parameters; pretraining must save it via Lightning.")
    model = mdlm_diffusion.Diffusion(cfg, tokenizer=DNATokenizer())
    state = ckpt.get("state_dict", ckpt)
    cleaned = {(k.replace("model.", "", 1) if k.startswith("model.") else k): v
               for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [load_dimamba] missing keys: {len(missing)}")
    if unexpected:
        print(f"  [load_dimamba] unexpected keys: {len(unexpected)}")
    model.eval()
    model.to(device)
    return model


def _select_mask(B: int, L: int, n_mask: int, device) -> torch.Tensor:
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for b in range(B):
        pos = torch.randperm(L, device=device)[:n_mask]
        mask[b, pos] = True
    return mask


def _forward_logits_acgt(model, x_noisy: torch.Tensor,
                         sigma_value: float = SIGMA_INFER) -> torch.Tensor:
    """Run the MDLM diffusion forward and return ACGT logits at all positions.

    Returns (B, L, 4) logits ready for sampling at masked positions.
    """
    B = x_noisy.shape[0]
    sigma = torch.full((B,), sigma_value, device=x_noisy.device,
                       dtype=torch.float32)
    log_p = model.forward(x_noisy, sigma)         # (B, L, V)
    return log_p[..., :4]


class DiMambaMutator:
    """MDLM-bidirectional mask-and-predict mutator (no DPS gradient)."""

    def __init__(self, diffusion_model, noise_fraction: float = 0.05,
                 mutation_substeps: int = 1):
        self.model = diffusion_model
        self.noise_fraction = noise_fraction
        self.mutation_substeps = mutation_substeps

    def _single_step(self, x_clean: torch.Tensor, branch_factor: int = 1):
        B, L = x_clean.shape
        device = x_clean.device
        n_mask = max(1, int(self.noise_fraction * L))
        mask = _select_mask(B, L, n_mask, device)

        x_noisy = x_clean.clone()
        x_noisy[mask] = MASK_TOKEN
        with torch.no_grad():
            logits = _forward_logits_acgt(self.model, x_noisy)   # (B, L, 4)
            probs = F.softmax(logits, dim=-1)                    # (B, L, 4)

        if branch_factor <= 1:
            x_new = x_clean.clone()
            if mask.any():
                flat = probs[mask]                               # (M, 4)
                flat = flat / flat.sum(-1, keepdim=True).clamp(min=1e-10)
                tokens = torch.multinomial(flat, 1).squeeze(-1)
                x_new[mask] = tokens
            return x_new

        branches = []
        for _ in range(branch_factor):
            x_k = x_clean.clone()
            if mask.any():
                flat = probs[mask]
                flat = flat / flat.sum(-1, keepdim=True).clamp(min=1e-10)
                tokens = torch.multinomial(flat, 1).squeeze(-1)
                x_k[mask] = tokens
            branches.append(x_k)
        return torch.stack(branches, dim=0)

    def __call__(self, _model_unused, x_clean: torch.Tensor,
                 labels: Optional[torch.Tensor] = None,
                 branch_factor: int = 1, **_kw):
        if self.mutation_substeps <= 1:
            return self._single_step(x_clean, branch_factor)
        cur = x_clean.clone()
        for _ in range(self.mutation_substeps):
            cur = self._single_step(cur, branch_factor=1)
        if branch_factor > 1:
            return self._single_step(cur, branch_factor)
        return cur


class DiMambaMutatorDPS(DiMambaMutator):
    """MDLM-bidirectional mutator with DPS gradient guidance.

    Same gradient mechanics as `HyenaDNAMutatorDPS`: backbone forward under
    no_grad, then a differentiable Gumbel-softmax over masked-position
    logits is fed to `oracle.dps_forward(soft_onehot)`. The reward gradient
    w.r.t. the logits is normalised and added to them with weight `eta`.
    """

    def __init__(self, diffusion_model, oracle_model,
                 noise_fraction: float = 0.05, mutation_substeps: int = 1,
                 eta: float = 3000.0, tau_start: float = 1.0,
                 tau_end: float = 0.1, dps_iw_correct: bool = False):
        super().__init__(diffusion_model, noise_fraction, mutation_substeps)
        self.oracle = oracle_model
        self.eta = eta
        self.tau = tau_start
        self.tau_end = tau_end
        self.dps_iw_correct = dps_iw_correct
        self._last_dps_iw_correction_kb = None

    def _single_step(self, x_clean: torch.Tensor, branch_factor: int = 1):
        B, L = x_clean.shape
        device = x_clean.device
        n_mask = max(1, int(self.noise_fraction * L))
        mask = _select_mask(B, L, n_mask, device)

        x_noisy = x_clean.clone()
        x_noisy[mask] = MASK_TOKEN
        with torch.no_grad():
            logits = _forward_logits_acgt(self.model, x_noisy)   # (B, L, 4)

        log_probs = logits.detach().clone()
        log_probs.requires_grad_(True)
        soft = F.gumbel_softmax(log_probs, tau=self.tau, hard=False)
        soft_input = soft.permute(0, 2, 1).contiguous()           # (B, 4, L)
        reward = self.oracle.dps_forward(soft_input)              # (B,)
        reward.sum().backward()
        grad = log_probs.grad
        grad_norm = grad.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        guided = log_probs.detach() + self.eta * (grad / grad_norm)
        guided_probs = F.softmax(guided, dim=-1)

        if branch_factor <= 1:
            x_new = x_clean.clone()
            if mask.any():
                flat = guided_probs[mask]
                flat = flat / flat.sum(-1, keepdim=True).clamp(min=1e-10)
                tokens = torch.multinomial(flat, 1).squeeze(-1)
                x_new[mask] = tokens
            return x_new

        branches = []
        for _ in range(branch_factor):
            x_k = x_clean.clone()
            if mask.any():
                flat = guided_probs[mask]
                flat = flat / flat.sum(-1, keepdim=True).clamp(min=1e-10)
                tokens = torch.multinomial(flat, 1).squeeze(-1)
                x_k[mask] = tokens
            branches.append(x_k)
        return torch.stack(branches, dim=0)
