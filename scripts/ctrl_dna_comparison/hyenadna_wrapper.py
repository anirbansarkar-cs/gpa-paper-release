"""
HyenaDNA mutation kernel for GPA.

Wraps a fine-tuned HyenaDNA (causal LM) to provide the
mutate_fn(model, x_clean, labels) -> x_new interface expected by
DiffusionPopulationAnnealer.

Mutation works by causal random-position resampling:
  1. Convert GPA tokens {A=0,C=1,G=2,T=3} → HyenaDNA tokens {A=7,C=8,G=9,T=10}
  2. Prepend label prefix: [START=0, label_tok_1, ..., label_tok_k]
  3. Forward pass → logits (B, 16, L_full)
  4. Mask nf * L random positions in the sequence portion
  5. At each masked position: extract ACGT logits, sample
  6. Convert back to GPA token space

Limitation: HyenaDNA is causal (left-to-right) — position p can only attend
to 0..p. This is weaker than MDLM (bidirectional) but exactly the fair
comparison we want with Ctrl-DNA's same backbone.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np


# Token mappings
GPA_TO_HYENA = {0: 7, 1: 8, 2: 9, 3: 10}  # A,C,G,T → HyenaDNA IDs
HYENA_TO_GPA = {7: 0, 8: 1, 9: 2, 10: 3}
HYENA_ACGT = [7, 8, 9, 10]  # HyenaDNA token IDs for A, C, G, T
LABEL_STOI = {"0": 2, "1": 3}  # Binary label token encoding


def load_hyenadna(checkpoint_path, device="cuda"):
    """Load fine-tuned HyenaDNA from checkpoint.

    Supports both our custom checkpoint format (from train_hyenadna.py)
    and Lightning checkpoints.

    Args:
        checkpoint_path: Path to .ckpt file.
        device: Device to load model on.

    Returns:
        AutoModelForCausalLM instance (eval mode, on device).
    """
    from transformers import AutoModelForCausalLM

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "LongSafari/hyenadna-medium-160k-seqlen-hf",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load fine-tuned weights
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        # Strip "model." prefix if present (our format)
        cleaned = {}
        for k, v in state_dict.items():
            key = k.replace("model.", "", 1) if k.startswith("model.") else k
            cleaned[key] = v
        model.load_state_dict(cleaned)

    model.eval()
    model.to(device)
    return model


class HyenaDNAMutator:
    """Wraps fine-tuned HyenaDNA for use as GPA mutation operator (no DPS)."""

    def __init__(self, hyenadna_model, noise_fraction=0.10, label="100",
                 mutation_substeps=1):
        """
        Args:
            hyenadna_model: Fine-tuned AutoModelForCausalLM instance.
            noise_fraction: Fraction of positions to mask and re-predict per sub-step.
            label: Conditioning label string (e.g., "100" for HepG2).
            mutation_substeps: Number of sequential mask-resample sub-steps.
                1 = original behavior. >1 = hierarchical multi-step mutation
                (each sub-step masks nf positions with a fresh forward pass).
        """
        self.model = hyenadna_model
        self.noise_fraction = noise_fraction
        self.label = label
        self.mutation_substeps = mutation_substeps

        # Pre-encode label prefix: [START=0] + label tokens
        self.label_tokens = torch.LongTensor(
            [0] + [LABEL_STOI[c] for c in label]
        )  # (1 + label_len,)
        self.prefix_len = len(self.label_tokens)

    def _gpa_to_hyena(self, x):
        """Convert GPA tokens {0..3} → HyenaDNA tokens {7..10}."""
        return x + 7  # Simple offset: 0→7, 1→8, 2→9, 3→10

    def _hyena_to_gpa(self, x):
        """Convert HyenaDNA tokens {7..10} → GPA tokens {0..3}."""
        return x - 7

    def _single_step(self, x_clean, branch_factor=1):
        """One mutation step: forward pass, mask nf positions, resample.

        Args:
            x_clean: (B, L) long tensor, tokens {0..3}, on device.
            branch_factor: If >1, return (K, B, L) with K independent samples.

        Returns:
            (B, L) or (K, B, L) long tensor with mutated sequences.
        """
        B, L = x_clean.shape
        device = x_clean.device

        # 1. Convert to HyenaDNA token space
        hyena_seq = self._gpa_to_hyena(x_clean)  # (B, L), tokens {7..10}

        # 2. Prepend label prefix: [START, label_1, ..., label_k, seq_1, ..., seq_L]
        prefix = self.label_tokens.to(device).unsqueeze(0).expand(B, -1)  # (B, prefix_len)
        full_input = torch.cat([prefix, hyena_seq], dim=1)  # (B, prefix_len + L)

        # 3. Forward pass → logits (B, L_full, 16) from HF model
        # Position j logits predict token j+1 given tokens 0..j
        with torch.no_grad():
            out = self.model(full_input)
            logits = out.logits  # (B, prefix_len + L, 16)
            logits = logits.permute(0, 2, 1)  # (B, 16, prefix_len + L)

        # Mask to valid bases only (A=7, C=8, G=9, T=10)
        masked_logits = torch.full_like(logits, fill_value=-1e8)
        masked_logits[:, HYENA_ACGT, :] = logits[:, HYENA_ACGT, :]
        logits = masked_logits

        # 4. Extract ACGT logits for sequence positions
        seq_logits = logits[:, :, self.prefix_len - 1: self.prefix_len + L - 1]  # (B, 16, L)

        # Extract only ACGT channels
        acgt_logits = seq_logits[:, HYENA_ACGT, :]  # (B, 4, L)
        acgt_probs = F.softmax(acgt_logits, dim=1)  # (B, 4, L)

        # 5. Select random positions to mutate
        n_mask = max(1, int(self.noise_fraction * L))
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            pos = torch.randperm(L, device=device)[:n_mask]
            mask[b, pos] = True

        # 6. Sample at masked positions
        if branch_factor <= 1:
            x_new = x_clean.clone()
            if mask.any():
                flat_probs = acgt_probs.permute(0, 2, 1)[mask]  # (num_masked, 4)
                flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                new_tokens = torch.multinomial(flat_probs, 1).squeeze(-1)  # GPA space {0..3}
                x_new[mask] = new_tokens
            return x_new
        else:
            # K-branch: sample K times from SAME forward pass
            branches = []
            for k in range(branch_factor):
                x_k = x_clean.clone()
                if mask.any():
                    flat_probs = acgt_probs.permute(0, 2, 1)[mask]
                    flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                    new_tokens = torch.multinomial(flat_probs, 1).squeeze(-1)
                    x_k[mask] = new_tokens
                branches.append(x_k)
            return torch.stack(branches, dim=0)  # (K, B, L)

    def __call__(self, model_unused, x_clean, labels, branch_factor=1, **kwargs):
        """GPA mutation interface.

        Args:
            model_unused: Ignored (HyenaDNA stored internally).
            x_clean: (B, L) long tensor, tokens {0..3}, on device.
            labels: (B, 1) float tensor, ignored (label stored internally).
            branch_factor: If >1, return (K, B, L) with K independent samples.

        Returns:
            (B, L) long tensor with mutated sequences (branch_factor=1), or
            (K, B, L) tensor with K branches (branch_factor>1).
        """
        if self.mutation_substeps <= 1:
            return self._single_step(x_clean, branch_factor)

        # Multi-step: iterate K sub-steps, each with a fresh forward pass.
        # Each sub-step masks nf positions and resamples from updated context.
        current = x_clean.clone()
        for _k in range(self.mutation_substeps):
            current = self._single_step(current, branch_factor=1)

        # For branching: run one final step with branch_factor from the
        # multi-step result to get K diverse branches.
        if branch_factor > 1:
            return self._single_step(current, branch_factor)
        return current


class HyenaDNAMutatorDPS(HyenaDNAMutator):
    """HyenaDNA mutation with DPS (Diffusion Posterior Sampling) guidance.

    Uses oracle gradients to shift HyenaDNA's logits toward higher-reward
    sequences before sampling. The gradient flows through:
        logits → Gumbel-softmax → oracle.dps_forward() → backward
    and NEVER through the HyenaDNA backbone (which runs under no_grad).

    This is the same mechanism as MDLMMutatorDPS (scripts/rerd_comparison/
    mdlm_wrapper.py) — the backbone just provides the prior distribution.
    """

    def __init__(self, hyenadna_model, oracle_model, noise_fraction=0.10,
                 label="100", mutation_substeps=1,
                 eta=3000.0, tau_start=1.0, tau_end=0.1,
                 dps_iw_correct=False):
        """
        Args:
            hyenadna_model: Fine-tuned AutoModelForCausalLM instance.
            oracle_model: Oracle with .dps_forward(soft_onehot) -> (B,) reward.
            noise_fraction: Fraction of positions to mask and re-predict.
            label: Conditioning label string (e.g., "100" for HepG2).
            mutation_substeps: Number of sequential mask-resample sub-steps.
            eta: DPS gradient step size.
            tau_start: Gumbel-Softmax temperature.
            tau_end: Not used in single-step; kept for interface compat.
            dps_iw_correct: If True, compute per-branch log q_base − log q_DPS
                at masked positions (S7 — importance-weight correction for
                gradient-warped proposal, Del Moral–Doucet–Jasra 2006 §3.1).
                Stored on ``self._last_dps_iw_correction_kb`` as shape (K, B).
        """
        super().__init__(hyenadna_model, noise_fraction, label, mutation_substeps)
        self.oracle = oracle_model
        self.eta = eta
        self.tau = tau_start
        self.dps_iw_correct = dps_iw_correct
        self._last_dps_iw_correction_kb = None  # (K, B) or (1, B) for K=1

    def _single_step(self, x_clean, branch_factor=1):
        """One DPS-guided mutation step.

        1. HyenaDNA forward (no_grad) → ACGT logits
        2. Detach logits, enable grad
        3. Gumbel-softmax → soft_onehot → oracle → reward
        4. Backward → gradient w.r.t. logits
        5. guided_logits = logits + eta * normalized_grad
        6. Sample at masked positions from guided distribution
        """
        B, L = x_clean.shape
        device = x_clean.device

        # --- 1. Get HyenaDNA logits (no gradient through backbone) ---
        hyena_seq = self._gpa_to_hyena(x_clean)
        prefix = self.label_tokens.to(device).unsqueeze(0).expand(B, -1)
        full_input = torch.cat([prefix, hyena_seq], dim=1)

        with torch.no_grad():
            out = self.model(full_input)
            logits = out.logits  # (B, prefix_len + L, 16)
            logits = logits.permute(0, 2, 1)  # (B, 16, prefix_len + L)

        # Extract ACGT logits for sequence positions
        seq_logits = logits[:, :, self.prefix_len - 1: self.prefix_len + L - 1]
        acgt_logits = seq_logits[:, HYENA_ACGT, :]  # (B, 4, L)

        # --- 2. Select mask positions ---
        n_mask = max(1, int(self.noise_fraction * L))
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            pos = torch.randperm(L, device=device)[:n_mask]
            mask[b, pos] = True

        # --- 3. DPS gradient via differentiable path ---
        # Permute to (B, L, 4) for Gumbel-softmax
        log_probs = acgt_logits.detach().permute(0, 2, 1).clone()  # (B, L, 4)
        log_probs.requires_grad_(True)

        # Gumbel-softmax relaxation
        soft_onehot = F.gumbel_softmax(log_probs, tau=self.tau, hard=False)  # (B, L, 4)
        # Oracle expects (B, 4, L)
        soft_input = soft_onehot.permute(0, 2, 1).contiguous()

        # Oracle forward (differentiable)
        reward = self.oracle.dps_forward(soft_input)  # (B,)
        reward.sum().backward()

        grad = log_probs.grad  # (B, L, 4)

        # Normalize gradient (same as MDLMMutatorDPS)
        grad_norm = grad.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        normalized_grad = grad / grad_norm

        # --- 4. Guided logits ---
        guided_log_probs = log_probs.detach() + self.eta * normalized_grad  # (B, L, 4)
        guided_probs = F.softmax(guided_log_probs, dim=-1)  # (B, L, 4)

        # S7 IW correction: base (un-DPSed) proposal density per position.
        # log_correction[k, b] = Σ_{i∈mask_b} [log p_base(token_i) − log p_DPS(token_i)].
        base_log_probs = None
        if self.dps_iw_correct:
            base_log_probs = F.log_softmax(log_probs.detach(), dim=-1)   # (B, L, 4)
            guided_log_probs_norm = F.log_softmax(guided_log_probs, dim=-1)  # (B, L, 4)

        # --- 5. Sample at masked positions ---
        if branch_factor <= 1:
            x_new = x_clean.clone()
            corr_1b = torch.zeros(B, device=device)
            if mask.any():
                flat_probs = guided_probs[mask]  # (num_masked, 4)
                flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                new_tokens = torch.multinomial(flat_probs, 1).squeeze(-1)
                x_new[mask] = new_tokens
                if self.dps_iw_correct and base_log_probs is not None:
                    flat_base = base_log_probs[mask].gather(-1, new_tokens.unsqueeze(-1)).squeeze(-1)  # (num_masked,)
                    flat_guided = guided_log_probs_norm[mask].gather(-1, new_tokens.unsqueeze(-1)).squeeze(-1)
                    per_position = flat_base - flat_guided  # (num_masked,)
                    # Aggregate per batch index using the batch dim of mask
                    batch_ids = mask.nonzero(as_tuple=False)[:, 0]  # (num_masked,)
                    corr_1b = corr_1b.scatter_add(0, batch_ids, per_position)
            if self.dps_iw_correct:
                self._last_dps_iw_correction_kb = corr_1b.detach().cpu().numpy()[None, :]  # (1, B)
            return x_new
        else:
            branches = []
            corr_kb = torch.zeros(branch_factor, B, device=device) if self.dps_iw_correct else None
            for k in range(branch_factor):
                x_k = x_clean.clone()
                if mask.any():
                    flat_probs = guided_probs[mask]
                    flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                    new_tokens = torch.multinomial(flat_probs, 1).squeeze(-1)
                    x_k[mask] = new_tokens
                    if self.dps_iw_correct and base_log_probs is not None:
                        flat_base = base_log_probs[mask].gather(-1, new_tokens.unsqueeze(-1)).squeeze(-1)
                        flat_guided = guided_log_probs_norm[mask].gather(-1, new_tokens.unsqueeze(-1)).squeeze(-1)
                        per_position = flat_base - flat_guided
                        batch_ids = mask.nonzero(as_tuple=False)[:, 0]
                        corr_kb[k] = corr_kb[k].scatter_add(0, batch_ids, per_position)
                branches.append(x_k)
            if self.dps_iw_correct:
                self._last_dps_iw_correction_kb = corr_kb.detach().cpu().numpy()  # (K, B)
            return torch.stack(branches, dim=0)
