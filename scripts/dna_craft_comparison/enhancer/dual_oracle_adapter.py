"""Dual-oracle adapter for the GPA-vs-DNA-CRAFT enhancer benchmark.

Wraps the DNA-CRAFT-protocol 50/50 split-model Enformer pair:
  - Design-Model  (half A) — used by GPA during optimization
  - Evaluation-Model (half B) — used for MinGap reporting

Adapted from scripts/ctrl_dna_comparison/mse_oracle_adapter.py. Differences:
  - Per-half directory layout: {ckpt_dir}/enformer_{cell}_{half}/best.ckpt
  - Adds `score_mingap_and_gc()` for MinGap-based best_eval snapshots and
    post-hoc top-128 selection (paper convention: target - max_off).
  - The existing `score()` + "linear" fitness_mode with `penalty_weight=0.5`
    implements GPA's internal "composite_mean" reward (target - 0.5 * mean_off).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CTRL_DNA_SRC = Path("${HOME}/Ctrl-DNA/ctrl_dna/src")
if str(CTRL_DNA_SRC) not in sys.path:
    sys.path.insert(0, str(CTRL_DNA_SRC))
from reglm.regression import EnformerModel, SeqDataset  # noqa: E402

EnformerModel.validation_epoch_end = None  # PL 2.x compat

IDX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}
CELL_TYPES = {"hepg2": 0, "k562": 1, "sknsh": 2}
CELL_NAMES = ["hepg2", "k562", "sknsh"]


def _indices_to_dna(sequences_tensor: torch.Tensor):
    arr = sequences_tensor.cpu().numpy()
    return ["".join(IDX_TO_BASE[int(b)] for b in row) for row in arr]


class SplitModelOracle:
    """Three per-cell MSE EnformerModels trained on one half of Gosai MPRA.

    Drop-in for GPA's `oracle_fn` / `fitness_fn` / `eval_oracle_fn` contract.
    """

    def __init__(
        self,
        ckpt_dir,
        half: str,                       # "design" or "eval"
        target_cell: str = "hepg2",
        penalty_cells=None,
        penalty_weight: float = 0.5,     # composite_mean default
        fitness_mode: str = "linear",    # internal GPA reward
        device="cuda",
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.half = half
        assert half in {"design", "eval"}
        self.device = device if isinstance(device, str) else f"cuda:{device}"
        self.fitness_mode = fitness_mode
        self.penalty_weight = penalty_weight

        self.target_cell = target_cell
        self.target_idx = CELL_TYPES[target_cell]
        if penalty_cells is None:
            penalty_cells = [c for c in CELL_NAMES if c != target_cell]
        self.penalty_cells = penalty_cells
        self.penalty_idxs = [CELL_TYPES[c] for c in penalty_cells]

        ckpt_name = os.environ.get("ENFORMER_CKPT_NAME", "best.ckpt")
        self.models = {}
        for cell in CELL_NAMES:
            ckpt_path = self.ckpt_dir / f"enformer_{cell}_{half}" / ckpt_name
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing {ckpt_path}")
            model = EnformerModel.load_from_checkpoint(str(ckpt_path))
            model.eval()
            self.models[cell] = model
        print(f"  SplitModelOracle[{half}]: loaded 3 {ckpt_name} models from {self.ckpt_dir}")

        # Caches populated by score()
        self._last_penalty_max = None
        self._last_penalty_mean = None
        self._last_penalty_per_cell = None
        self._last_all_scores = None

    # ------------------------------ Internal helpers ------------------------
    def _predict_cell(self, sequences, cell, batch_size=512):
        dummy_labels = np.zeros(len(sequences))
        df = pd.DataFrame({"sequence": sequences, "label": dummy_labels})
        dataset = SeqDataset(df, seq_len=len(sequences[0]))
        if isinstance(self.device, str) and self.device.startswith("cuda:"):
            dev_idx = int(self.device.split(":")[1])
        else:
            dev_idx = 0
        preds = self.models[cell].predict_on_dataset(
            dataset, device=dev_idx, batch_size=batch_size)
        return preds.flatten()

    def _predict_all(self, sequences, batch_size=512):
        return {c: self._predict_cell(sequences, c, batch_size)
                for c in CELL_NAMES}

    # ------------------------------ GPA contract ---------------------------
    def score(self, sequences_tensor, batch_size=512):
        """GPA's `oracle_fn` interface. Returns (target_scores, gc_fracs).

        Caches per-cell predictions for compute_fitness() and for MinGap eval.
        """
        sequences_tensor = sequences_tensor.to("cpu")
        dna_strings = _indices_to_dna(sequences_tensor)
        all_scores = self._predict_all(dna_strings, batch_size)
        self._last_all_scores = all_scores

        target_scores = all_scores[self.target_cell]

        penalty_arrays = [all_scores[CELL_NAMES[i]] for i in self.penalty_idxs]
        if penalty_arrays:
            stack = np.stack(penalty_arrays, axis=1)  # (N, P)
            self._last_penalty_max = stack.max(axis=1)
            self._last_penalty_mean = stack.mean(axis=1)
            self._last_penalty_per_cell = penalty_arrays
        else:
            self._last_penalty_max = np.zeros(len(dna_strings))
            self._last_penalty_mean = np.zeros(len(dna_strings))
            self._last_penalty_per_cell = []

        gc = ((sequences_tensor == 1) | (sequences_tensor == 2)).float()
        gc_fracs = gc.mean(dim=1).numpy()
        return target_scores, gc_fracs

    def compute_fitness(self, raw_scores):
        """Apply GPA's internal reward. Default = linear (target - pw*mean_off)."""
        if self._last_penalty_max is None or self.penalty_weight == 0.0:
            return raw_scores
        if self.fitness_mode == "linear":
            return raw_scores - self.penalty_weight * self._last_penalty_mean
        raise ValueError(f"fitness_mode={self.fitness_mode!r} not supported")

    # ------------------------------ MinGap (paper metric) -------------------
    def score_mingap_and_gc(self, sequences_tensor, batch_size=512):
        """For `eval_oracle_fn` and post-hoc selection.

        Returns (mingap_scores, gc_fracs) where
            mingap = target - max(off_target_cells).
        Also caches all 3 per-cell scores under self._last_all_scores.
        """
        sequences_tensor = sequences_tensor.to("cpu")
        dna_strings = _indices_to_dna(sequences_tensor)
        all_scores = self._predict_all(dna_strings, batch_size)
        self._last_all_scores = all_scores

        target_scores = all_scores[self.target_cell]
        pen_arrays = [all_scores[CELL_NAMES[i]] for i in self.penalty_idxs]
        pen_stack = np.stack(pen_arrays, axis=1)           # (N, P)
        mingap = target_scores - pen_stack.max(axis=1)     # (N,)

        gc = ((sequences_tensor == 1) | (sequences_tensor == 2)).float()
        gc_fracs = gc.mean(dim=1).numpy()
        return mingap, gc_fracs

    def score_all_cells(self, sequences_tensor, batch_size=512):
        """Return per-cell scores as dict. Used by evaluate_dnacraft_pools."""
        sequences_tensor = sequences_tensor.to("cpu")
        dna_strings = _indices_to_dna(sequences_tensor)
        return self._predict_all(dna_strings, batch_size)

    # ------------------------------ DPS gradient ---------------------------
    def dps_forward(self, soft_onehot):
        """Differentiable composite-mean reward for DPS gradient."""
        device = soft_onehot.device
        x = soft_onehot.permute(0, 2, 1)                    # (B, L, 4)
        self.models[self.target_cell].to(device)
        r_target = self.models[self.target_cell](x, return_logits=True).squeeze(-1)
        if self.penalty_weight == 0.0 or not self.penalty_cells:
            return r_target
        pen_scores = []
        for cell in self.penalty_cells:
            self.models[cell].to(device)
            pen_scores.append(self.models[cell](x, return_logits=True).squeeze(-1))
        r_penalty = torch.stack(pen_scores, dim=1).mean(dim=1)
        return r_target - self.penalty_weight * r_penalty


def load_dual(ckpt_dir, target_cell, penalty_weight=0.5, device="cuda"):
    """Convenience: build both Design-Model and Evaluation-Model pools."""
    design = SplitModelOracle(
        ckpt_dir, half="design", target_cell=target_cell,
        penalty_weight=penalty_weight, device=device)
    eval_ = SplitModelOracle(
        ckpt_dir, half="eval", target_cell=target_cell,
        penalty_weight=0.0,  # reporting-only; no composite applied
        device=device)
    return design, eval_
