"""
K562 LegNet oracle wrapper for GPA pipeline with adapter injection.

Wraps the K562 lentiMPRA LegNet CNN oracle to provide:
  - oracle_fn(sequences_tensor) -> (scores_np, gc_np) for GPA fitness scoring
  - dps_forward(soft_onehot) -> (B,) reward for DPS gradient computation

Handles 200bp MDLM sequences by injecting 15bp adapters at oracle scoring time
to produce the 230bp input the oracle expects.

No channel swap needed: both diffusion and the LegNet checkpoint (trained on
H5 data) use bio encoding A=0,C=1,G=2,T=3.

Adapters (in bio encoding):
  5' = AGGACCGGATCAACT (15bp)
  3' = CATTGCGTGAACCGA (15bp)
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


# Adapter sequences (in DNA letters)
ADAPTER_5P = "AGGACCGGATCAACT"  # 15bp
ADAPTER_3P = "CATTGCGTGAACCGA"  # 15bp

# Bio encoding (same as diffusion and the LegNet H5 training data): A=0, C=1, G=2, T=3
BIO_NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def _seq_to_onehot(seq_str):
    """Convert DNA string to (1, 4, L) one-hot in bio encoding."""
    indices = [BIO_NUC_MAP[c] for c in seq_str]
    onehot = np.eye(4, dtype=np.float32)[indices]  # (L, 4)
    return torch.from_numpy(onehot.T).unsqueeze(0)  # (1, 4, L)


def load_k562_oracle(checkpoint_path, device="cuda"):
    """Load K562 LegNet oracle from checkpoint.

    Args:
        checkpoint_path: Path to LegNet .ckpt file.
        device: Device to load model on.

    Returns:
        LegNet model instance (eval mode, on device).
    """
    sys.path.insert(0, os.path.join(os.environ.get("HOME", ""), "d3_evaluation_pipeline"))
    from mpralegnet import load_model

    model, config = load_model(checkpoint_path)
    model.eval()
    model.to(device)
    return model


class K562Oracle:
    """Wraps K562 LegNet for GPA pipeline with adapter injection (200bp -> 230bp)."""

    def __init__(self, model, device="cuda"):
        """
        Args:
            model: Loaded LegNet model (from load_k562_oracle).
            device: Device for computation.
        """
        self.model = model
        self.device = device

        # Precompute adapter one-hot constants in bio encoding
        self.adapter_5p_onehot = _seq_to_onehot(ADAPTER_5P).to(device)  # (1, 4, 15)
        self.adapter_3p_onehot = _seq_to_onehot(ADAPTER_3P).to(device)  # (1, 4, 15)

    def _indices_to_onehot(self, sequences_tensor):
        """Convert token indices to one-hot. (N, L) -> (N, 4, L)."""
        onehot = F.one_hot(sequences_tensor.long(), 4).float()
        return onehot.permute(0, 2, 1)  # (N, 4, L)

    def _swap_channels(self, onehot):
        """Identity — no channel swap needed.

        Both diffusion and the LegNet checkpoint trained on H5 data use the
        same bio encoding: A=0, C=1, G=2, T=3.

        The CODES dict in mpralegnet.py (A=0, G=1, C=2, T=3) is only used by
        the Seq2Tensor string→tensor path; the HDF5Dataset path (which the
        checkpoint was trained on) transposes the H5 one-hot directly, inheriting
        the bio channel order.  Verified: no-swap gives Pearson r=0.815 on the
        K562 test set vs r=0.136 with swap.
        """
        return onehot

    def _inject_adapters(self, core_onehot):
        """Concatenate 15bp adapters around 200bp core -> 230bp.

        Args:
            core_onehot: (N, 4, 200) in oracle encoding.

        Returns:
            (N, 4, 230) in oracle encoding.
        """
        N = core_onehot.shape[0]
        adapter_5p = self.adapter_5p_onehot.expand(N, -1, -1)  # (N, 4, 15)
        adapter_3p = self.adapter_3p_onehot.expand(N, -1, -1)  # (N, 4, 15)
        return torch.cat([adapter_5p, core_onehot, adapter_3p], dim=2)  # (N, 4, 230)

    def score(self, sequences_tensor, batch_size=256):
        """Score sequences -- returns oracle predictions and GC fractions.

        Handles both 200bp (MDLM) and 230bp (backward compat) inputs.

        Args:
            sequences_tensor: (N, L) int tensor {0..3}, L=200 or 230.

        Returns:
            (np.array scores, np.array gc_fractions) -- both (N,) on CPU.
        """
        sequences_tensor = sequences_tensor.to(self.device)
        L = sequences_tensor.shape[1]

        # GC fraction: C=1, G=2 in diffusion encoding
        gc = ((sequences_tensor == 1) | (sequences_tensor == 2)).float()
        gc_fracs = gc.mean(dim=1).cpu().numpy()

        # Convert to one-hot and swap encoding
        onehot = self._indices_to_onehot(sequences_tensor)
        onehot_oracle = self._swap_channels(onehot)

        # Inject adapters if 200bp input
        if L == 200:
            onehot_oracle = self._inject_adapters(onehot_oracle)

        # LitModel.predict() handles batching internally
        preds = self.model.predict(onehot_oracle, batch_size=batch_size, keepgrad=False)
        scores = preds.cpu().numpy().flatten()

        return scores, gc_fracs

    def dps_forward(self, soft_onehot):
        """Differentiable forward for DPS gradient computation.

        Args:
            soft_onehot: (B, 4, L) float tensor in bio encoding (A=0,C=1,G=2,T=3)
                (from Gumbel-softmax relaxation). Must have gradient.
                L=200 (MDLM) or L=230 (backward compat).

        Returns:
            (B,) reward tensor with gradient.
        """
        L = soft_onehot.shape[2]

        # No channel swap needed — diffusion and LegNet use the same bio encoding.

        # Inject adapters if 200bp input
        if L == 200:
            B = soft_onehot.shape[0]
            # Adapters are detached constants -- gradients only flow through core
            adapter_5p = self.adapter_5p_onehot.expand(B, -1, -1)  # (B, 4, 15)
            adapter_3p = self.adapter_3p_onehot.expand(B, -1, -1)  # (B, 4, 15)
            soft_onehot = torch.cat([adapter_5p, soft_onehot, adapter_3p], dim=2)

        # Forward through oracle
        preds = self.model.model(soft_onehot)  # LegNet .model is the raw nn.Module
        return preds.squeeze(-1).squeeze(-1)  # (B,) scalar per sequence


class CascadeOracle:
    """Two-stage cascade: LegNet for cheap branch selection, AG for final scoring.

    Used when oracle_type=alphagenome with --cascade_oracle flag.
    Branch selection uses LegNet (fast, differentiable); AG confirms only
    the winning branch per particle, reducing AG calls from K*N to ~N per step.

    Both oracles must implement .score(sequences_tensor) -> (scores, gc).
    """

    def __init__(self, fast_oracle, accurate_oracle):
        """
        Args:
            fast_oracle: K562Oracle (LegNet) for branch pre-filtering.
            accurate_oracle: AlphaGenomeK562Client for final scoring.
        """
        self.fast = fast_oracle
        self.accurate = accurate_oracle

    def score(self, sequences_tensor, batch_size=256):
        """Score using the accurate (AG) oracle.

        This is the main scoring path called by GPA for fitness evaluation.
        Uses the accurate oracle directly (no cascade needed for non-branch scoring).
        """
        return self.accurate.score(sequences_tensor, batch_size=batch_size)

    def score_fast(self, sequences_tensor, batch_size=256):
        """Score using the fast (LegNet) oracle for branch pre-filtering."""
        return self.fast.score(sequences_tensor, batch_size=batch_size)

    def dps_forward(self, soft_onehot):
        """DPS through the fast oracle (LegNet is differentiable, AG is not)."""
        return self.fast.dps_forward(soft_onehot)


# LentiMPRA 281bp training construct (per torch ckpt construct_config):
#   A5(15) + insert_core(200) + A3(15) + promoter(36) + barcode(15) = 281
_AG_PROMOTER = "TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG"
_AG_BARCODE = "AGAGACTGAGGCCAC"
_AG_TORCH_CKPTS = {
    "k562":  os.path.expandvars("${GPA_SHARED_ROOT}/models/alphagenome_encoder/torch/mpra_K562/finetuned_encoder.pt"),
    "hepg2": os.path.expandvars("${GPA_SHARED_ROOT}/models/alphagenome_encoder/torch/mpra_HepG2/finetuned_encoder.pt"),
    "wtc11": os.path.expandvars("${GPA_SHARED_ROOT}/models/alphagenome_encoder/torch/mpra_WTC11/finetuned_encoder.pt"),
}


class TorchAGOracle:
    """Differentiable torch AG (stage2 finetuned) oracle, K562Oracle-compatible:
      - score(seq_tensor (N,200) int) -> (scores_np, gc_np)
      - dps_forward(soft_onehot (B,4,200) float, grad) -> (B,) reward (grad)
    Assembles the 281bp construct channels-LAST in ACGT order and calls
    EncoderMPRAModel.forward (verified to match predict_sequences exactly, and
    gradients flow to the editable core). EncoderMPRAModel is imported lazily so
    this module still imports in envs without alphagenome_encoder_ft (e.g. the main `gpa` env).
    Channel order A=0,C=1,G=2,T=3 matches both the diffusion bio encoding and the
    AG one-hot, so no channel swap is needed.
    """

    _BASES = "ACGT"

    def __init__(self, device="cuda", cell="k562"):
        from alphagenome_encoder_ft import EncoderMPRAModel  # lazy
        self.device = device
        ckpt = _AG_TORCH_CKPTS[cell]
        print(f"[Oracle] Loading torch AG stage2 ({cell}): {ckpt}")
        self.model = EncoderMPRAModel.from_checkpoint(ckpt, device=device)
        self.model.eval()

        def _oh(s):
            t = torch.zeros(1, len(s), 4, dtype=torch.float32)
            for j, ch in enumerate(s):
                t[0, j, self._BASES.index(ch)] = 1.0
            return t.to(device)
        self._a5 = _oh(ADAPTER_5P)
        self._a3 = _oh(ADAPTER_3P)
        self._prom = _oh(_AG_PROMOTER)
        self._bar = _oh(_AG_BARCODE)

    def _assemble(self, core_bl):
        """core_bl: (B,200,4) channels-last -> (B,281,4) full construct."""
        B = core_bl.shape[0]
        return torch.cat([
            self._a5.expand(B, -1, -1), core_bl, self._a3.expand(B, -1, -1),
            self._prom.expand(B, -1, -1), self._bar.expand(B, -1, -1)], dim=1)

    def score(self, sequences_tensor, batch_size=256):
        seq = sequences_tensor.to(self.device).long()
        gc = ((seq == 1) | (seq == 2)).float().mean(dim=1).cpu().numpy()
        oh = F.one_hot(seq, 4).float()              # (N,200,4) ACGT channels-last
        out = []
        for s in range(0, oh.shape[0], batch_size):
            with torch.no_grad():
                p = self.model.forward(self._assemble(oh[s:s + batch_size]))
            out.append(p.reshape(-1).detach().cpu().numpy())
        return np.concatenate(out), gc

    def dps_forward(self, soft_onehot):
        """soft_onehot: (B,4,200) channels-first float (Gumbel relaxation), grad.
        Flanks are detached constants so grad flows only through the core."""
        core_bl = soft_onehot.permute(0, 2, 1)      # (B,4,200) -> (B,200,4)
        preds = self.model.forward(self._assemble(core_bl))
        return preds.reshape(-1)
