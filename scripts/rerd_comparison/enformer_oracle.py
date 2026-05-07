"""
Enformer oracle wrapper for GPA pipeline.

Wraps the gReLU Enformer-based oracle from the SVDD/RERD codebase to provide:
  - oracle_fn(sequences_tensor) -> (scores_np, gc_np) for GPA fitness scoring
  - dps_forward(soft_onehot) -> (B,) reward for DPS gradient computation

The oracle outputs predictions for 3 cell types: [hepg2, k562, sknsh].
Cell-type specificity is achieved via configurable fitness modes:
  linear:    r_target - penalty_weight * mean(r_penalty)  (default)
  ratio:     r_target / (1 + clamp(max(r_penalty), min=0))
  log_ratio: log(1 + clamp(r_target, min=0)) - pw * log(1 + clamp(max(r_penalty), min=0))
  indicator: r_target * I(all off-targets < threshold)  (RERD paper formula)

DPS reward modes include sum_individual: r_target - pw_k * K - pw_s * S (per-cell weights).
"""

import torch
import torch.nn.functional as F
import numpy as np


def load_enformer_oracle(checkpoint_path, device="cuda"):
    """Load gReLU Enformer oracle from checkpoint.

    The checkpoint needs data_params and performance keys at top level
    for gReLU compatibility. If missing, we patch them in.

    Args:
        checkpoint_path: Path to model.ckpt or model_fixed.ckpt.
        device: Device to load model on.

    Returns:
        gReLU LightningModel instance (eval mode, on device).
    """
    # Check if checkpoint needs fixing
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    needs_fix = "data_params" not in ckpt or "performance" not in ckpt

    if needs_fix:
        if "data_params" not in ckpt and "hyper_parameters" in ckpt:
            ckpt["data_params"] = ckpt["hyper_parameters"]["data_params"]
        if "performance" not in ckpt:
            ckpt["performance"] = {}
        fixed_path = checkpoint_path.replace(".ckpt", "_fixed.ckpt")
        torch.save(ckpt, fixed_path)
        checkpoint_path = fixed_path

    from grelu.lightning import LightningModel
    oracle = LightningModel.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    oracle.eval()
    oracle.to(device)
    return oracle


# Cell type indices in oracle output
CELL_TYPES = {"hepg2": 0, "k562": 1, "sknsh": 2}


class EnformerOracle:
    """Wraps gReLU Enformer for GPA pipeline with cell-type specific reward."""

    VALID_MODES = ("linear", "ratio", "log_ratio", "indicator")
    VALID_DPS_REWARD_MODES = ("fitness", "smooth_spec", "log_ratio_spec", "neg_offtarget", "sum_individual")

    def __init__(self, lit_model, target_cell="hepg2",
                 penalty_cells=None, penalty_weight=0.5,
                 fitness_mode="linear", specificity_threshold=None,
                 dps_reward_mode="fitness", dps_penalty_weight=0.3,
                 penalty_weight_k=None, penalty_weight_s=None,
                 dps_penalty_weight_k=None, dps_penalty_weight_s=None,
                 device="cuda"):
        """
        Args:
            lit_model: gReLU LightningModel loaded from checkpoint.
            target_cell: Cell type to maximize ("hepg2", "k562", or "sknsh").
            penalty_cells: List of cell types to penalize. Default: all others.
            penalty_weight: Weight for penalty term (used by linear and log_ratio).
            fitness_mode: "linear", "ratio", "log_ratio", or "indicator".
            specificity_threshold: For indicator mode, max allowed off-target value.
            dps_reward_mode: DPS gradient objective — "fitness" (same as GPA),
                "smooth_spec" (r_target - pw * mean(r_penalty)), or
                "log_ratio_spec" (log-space version).
            dps_penalty_weight: Weight for off-target penalty in DPS reward.
            device: Device for computation.
        """
        if fitness_mode not in self.VALID_MODES:
            raise ValueError(f"fitness_mode must be one of {self.VALID_MODES}, got '{fitness_mode}'")
        if fitness_mode == "indicator" and specificity_threshold is None:
            raise ValueError("specificity_threshold required for indicator mode")
        if dps_reward_mode not in self.VALID_DPS_REWARD_MODES:
            raise ValueError(f"dps_reward_mode must be one of {self.VALID_DPS_REWARD_MODES}, got '{dps_reward_mode}'")

        self.lit_model = lit_model
        self.model = lit_model.model  # raw nn.Module for DPS
        self.device = device
        self.fitness_mode = fitness_mode

        self.target_idx = CELL_TYPES[target_cell]
        if penalty_cells is None:
            penalty_cells = [c for c in CELL_TYPES if c != target_cell]
        self.penalty_idxs = [CELL_TYPES[c] for c in penalty_cells]
        self.penalty_weight = penalty_weight
        self.specificity_threshold = specificity_threshold
        self.dps_reward_mode = dps_reward_mode
        self.dps_penalty_weight = dps_penalty_weight
        self.penalty_weight_k = penalty_weight_k
        self.penalty_weight_s = penalty_weight_s
        self.dps_penalty_weight_k = dps_penalty_weight_k
        self.dps_penalty_weight_s = dps_penalty_weight_s

        # Cache last penalty predictions for fitness_fn use
        self._last_penalty_max = None
        self._last_penalty_mean = None
        self._last_penalty_per_cell = None  # list of per-cell numpy arrays

    def _get_per_cell_pw_array(self):
        """Return per-cell penalty weights in penalty_idxs order.

        Maps penalty_weight_k/penalty_weight_s to the correct positions
        based on which cells are in self.penalty_idxs.
        """
        cell_pw = {
            CELL_TYPES["k562"]: self.penalty_weight_k or 0.0,
            CELL_TYPES["sknsh"]: self.penalty_weight_s or 0.0,
            CELL_TYPES["hepg2"]: self.penalty_weight or 0.0,  # fallback if hepg2 is a penalty cell
        }
        return [cell_pw.get(idx, self.penalty_weight) for idx in self.penalty_idxs]

    def _get_per_cell_dps_pw_array(self):
        """Return per-cell DPS penalty weights in penalty_idxs order."""
        cell_pw = {
            CELL_TYPES["k562"]: self.dps_penalty_weight_k or 0.0,
            CELL_TYPES["sknsh"]: self.dps_penalty_weight_s or 0.0,
            CELL_TYPES["hepg2"]: self.dps_penalty_weight or 0.0,
        }
        return [cell_pw.get(idx, self.dps_penalty_weight) for idx in self.penalty_idxs]

    def _indices_to_onehot(self, sequences_tensor):
        """Convert token indices to one-hot. (N, L) -> (N, 4, L)."""
        onehot = F.one_hot(sequences_tensor.long(), 4).float()
        return onehot.permute(0, 2, 1)  # (N, 4, L)

    def _compute_reward(self, preds):
        """Compute cell-type specific reward from oracle predictions.

        Args:
            preds: (N, 3, 1) oracle output.

        Returns:
            (N,) reward tensor.
        """
        preds = preds.squeeze(-1)  # (N, 3)
        r_target = preds[:, self.target_idx]
        penalty_preds = torch.stack([preds[:, idx] for idx in self.penalty_idxs], dim=1)  # (N, P)

        if self.fitness_mode == "linear":
            # r_target - pw * mean(r_penalty)
            reward = r_target - self.penalty_weight * penalty_preds.mean(dim=1)
        elif self.fitness_mode == "ratio":
            # r_target / (1 + max(0, max(r_penalty)))
            max_penalty = penalty_preds.max(dim=1).values.clamp(min=0)
            reward = r_target / (1.0 + max_penalty)
        elif self.fitness_mode == "log_ratio":
            # log(1 + clamp(r_target)) - pw * log(1 + clamp(max(r_penalty)))
            max_penalty = penalty_preds.max(dim=1).values.clamp(min=0)
            reward = torch.log1p(r_target.clamp(min=0)) - \
                     self.penalty_weight * torch.log1p(max_penalty)
        elif self.fitness_mode == "indicator":
            # r_target * I(all off-targets < threshold)
            max_penalty = penalty_preds.max(dim=1).values
            mask = max_penalty < self.specificity_threshold
            reward = torch.where(mask, r_target, torch.full_like(r_target, -1e6))

        return reward

    def score(self, sequences_tensor, batch_size=256):
        """Score sequences — returns RAW target-cell scores (no fitness modification).

        The fitness transformation (penalty_weight, indicator, etc.) is applied
        separately via fitness_fn in the GPA pipeline, using cached penalty info.

        Args:
            sequences_tensor: (N, L) int tensor {0..3}, on any device.

        Returns:
            (np.array target_scores, np.array gc_fractions) — both (N,) on CPU.
        """
        sequences_tensor = sequences_tensor.to(self.device)
        N = sequences_tensor.shape[0]
        all_target = []
        all_penalty_max = []
        all_penalty_mean = []
        # Per-cell penalty collectors: one list per penalty cell
        all_penalty_per_cell = [[] for _ in self.penalty_idxs]

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = sequences_tensor[start:end]
            onehot = self._indices_to_onehot(batch)

            with torch.no_grad():
                preds = self.model(onehot)  # (B, 3, 1)

            preds_sq = preds.squeeze(-1)  # (B, 3)
            r_target = preds_sq[:, self.target_idx]
            penalty_preds = torch.stack(
                [preds_sq[:, idx] for idx in self.penalty_idxs], dim=1)  # (B, P)

            all_target.append(r_target.cpu())
            all_penalty_max.append(penalty_preds.max(dim=1).values.cpu())
            all_penalty_mean.append(penalty_preds.mean(dim=1).cpu())
            for i in range(len(self.penalty_idxs)):
                all_penalty_per_cell[i].append(penalty_preds[:, i].cpu())

        target_scores = torch.cat(all_target, dim=0).numpy()
        self._last_penalty_max = torch.cat(all_penalty_max, dim=0).numpy()
        self._last_penalty_mean = torch.cat(all_penalty_mean, dim=0).numpy()
        self._last_penalty_per_cell = [
            torch.cat(cell_list, dim=0).numpy()
            for cell_list in all_penalty_per_cell
        ]

        # GC fraction
        gc = ((sequences_tensor == 1) | (sequences_tensor == 2)).float()
        gc_fracs = gc.mean(dim=1).cpu().numpy()

        return target_scores, gc_fracs

    def compute_fitness(self, raw_scores):
        """Apply fitness transformation to raw target scores.

        Uses cached penalty predictions from the last score() call.

        Args:
            raw_scores: (N,) np.array of raw target-cell scores.

        Returns:
            (N,) np.array of fitness values.
        """
        if self._last_penalty_max is None:
            return raw_scores

        penalty_max = self._last_penalty_max

        if self.fitness_mode == "linear":
            # Per-cell GPA weights: H - pw_k * K - pw_s * S
            if self.penalty_weight_k is not None and self.penalty_weight_s is not None:
                # penalty_idxs order matches penalty_cells order (k562=idx0, sknsh=idx1 for target=hepg2)
                pw_per_cell = self._get_per_cell_pw_array()
                fit = raw_scores.copy()
                for i, pw_i in enumerate(pw_per_cell):
                    fit = fit - pw_i * self._last_penalty_per_cell[i]
                return fit
            if self.penalty_weight == 0.0:
                return raw_scores
            return raw_scores - self.penalty_weight * self._last_penalty_mean
        elif self.fitness_mode == "ratio":
            max_pen = np.maximum(0, penalty_max)
            return raw_scores / (1.0 + max_pen)
        elif self.fitness_mode == "log_ratio":
            max_pen = np.maximum(0, penalty_max)
            return np.log1p(np.maximum(0, raw_scores)) - \
                   self.penalty_weight * np.log1p(max_pen)
        elif self.fitness_mode == "indicator":
            mask = penalty_max < self.specificity_threshold
            return np.where(mask, raw_scores, -1e6)
        else:
            return raw_scores

    def score_target_only(self, sequences_tensor, batch_size=256):
        """Score sequences — returns RAW target-cell scores only (no penalty).

        Lightweight version of score() that skips penalty extraction.
        Use for hill-climb candidate scoring where only relative target
        scores are compared.

        Args:
            sequences_tensor: (N, L) int tensor {0..3}, on any device.

        Returns:
            (np.array target_scores, np.array gc_fractions) — both (N,) on CPU.
        """
        sequences_tensor = sequences_tensor.to(self.device)
        N = sequences_tensor.shape[0]
        all_target = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = sequences_tensor[start:end]
            onehot = self._indices_to_onehot(batch)

            with torch.no_grad():
                preds = self.model(onehot)  # (B, 3, 1)

            r_target = preds.squeeze(-1)[:, self.target_idx]
            all_target.append(r_target.cpu())

        target_scores = torch.cat(all_target, dim=0).numpy()

        # GC fraction
        gc = ((sequences_tensor == 1) | (sequences_tensor == 2)).float()
        gc_fracs = gc.mean(dim=1).cpu().numpy()

        return target_scores, gc_fracs

    def score_all_cells(self, sequences_tensor, batch_size=256):
        """Score sequences for all 3 cell types separately.

        Returns:
            dict with keys 'hepg2', 'k562', 'sknsh', each (N,) np.array.
        """
        sequences_tensor = sequences_tensor.to(self.device)
        N = sequences_tensor.shape[0]
        all_preds = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = sequences_tensor[start:end]
            onehot = self._indices_to_onehot(batch)

            with torch.no_grad():
                preds = self.model(onehot).squeeze(-1)  # (B, 3)
            all_preds.append(preds.cpu())

        preds = torch.cat(all_preds, dim=0).numpy()
        return {name: preds[:, idx] for name, idx in CELL_TYPES.items()}

    def _compute_dps_reward(self, preds):
        """Compute DPS gradient objective (may differ from GPA fitness).

        When dps_reward_mode="fitness", uses the same objective as GPA.
        Other modes provide smooth gradients even for indicator-gated particles.

        Args:
            preds: (N, 3, 1) oracle output.

        Returns:
            (N,) reward tensor with gradient.
        """
        if self.dps_reward_mode == "fitness":
            return self._compute_reward(preds)

        preds = preds.squeeze(-1)  # (N, 3)
        r_target = preds[:, self.target_idx]
        penalty_preds = torch.stack(
            [preds[:, idx] for idx in self.penalty_idxs], dim=1)  # (N, P)

        if self.dps_reward_mode == "smooth_spec":
            # r_target - dps_pw * mean(r_penalty)
            return r_target - self.dps_penalty_weight * penalty_preds.mean(dim=1)
        elif self.dps_reward_mode == "log_ratio_spec":
            # log(1 + r_target) - dps_pw * log(1 + max(r_penalty))
            max_pen = penalty_preds.max(dim=1).values.clamp(min=0)
            return torch.log1p(r_target.clamp(min=0)) - \
                   self.dps_penalty_weight * torch.log1p(max_pen)
        elif self.dps_reward_mode == "neg_offtarget":
            # -max(off-target): DPS purely minimizes off-targets
            return -penalty_preds.max(dim=1).values
        elif self.dps_reward_mode == "sum_individual":
            # r_target - dps_pw_k * K - dps_pw_s * S (per-cell weights)
            dps_pws = self._get_per_cell_dps_pw_array()
            reward = r_target
            for i, pw_i in enumerate(dps_pws):
                reward = reward - pw_i * penalty_preds[:, i]
            return reward

    def dps_forward(self, soft_onehot):
        """Differentiable forward for DPS gradient computation.

        Args:
            soft_onehot: (B, 4, L) float tensor (differentiable).

        Returns:
            (B,) reward tensor with gradient.
        """
        preds = self.model(soft_onehot)  # (B, 3, 1)
        return self._compute_dps_reward(preds)
