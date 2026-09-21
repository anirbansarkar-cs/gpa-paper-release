"""
Promoter oracle adapter — bridges three regLM EnformerModel per-cell
checkpoints to the interface expected by run_ctrldna_promoter.py and
run_gpa_hyenadna_promoter.py.

Unlike the enhancer side (single 3-task gReLU model, wrapped by
scripts/ctrl_dna_comparison/oracle_adapter.py:GrELUOracleAdapter), the
promoter oracle is three separate regLM EnformerModel checkpoints:
  - human_paired_jurkat.ckpt
  - human_paired_k562.ckpt
  - human_paired_THP1.ckpt
Each outputs a scalar activity prediction for ONE cell, in linear space
(loss="mse"). Input is one-hot (N, L, 4) channels-LAST (enformer-pytorch
convention), in contrast to the gReLU oracle which takes (N, 4, L).

Cell-index convention here is (JURKAT=0, K562=1, THP1=2) to match
finetuning_data.csv column order and the decile-label prefix order
built by build_hyenadna_labels.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F

_CTRLDNA_SRC = os.path.expandvars("${HOME}/Ctrl-DNA/ctrl_dna")
if _CTRLDNA_SRC not in sys.path:
    sys.path.insert(0, _CTRLDNA_SRC)

from src.reglm.regression import EnformerModel  # noqa: E402

PROMOTER_CELLS = ("JURKAT", "K562", "THP1")
CELL_TAG = {"JURKAT": "jurkat", "K562": "k562", "THP1": "THP1"}
CELL_INDEX = {c: i for i, c in enumerate(PROMOTER_CELLS)}

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX_TO_BASE = {v: k for k, v in BASE_TO_IDX.items()}


def dna_to_indices(dna_strings: List[str]) -> np.ndarray:
    return np.array([[BASE_TO_IDX[b] for b in seq] for seq in dna_strings])


def indices_to_dna(indices: np.ndarray) -> List[str]:
    return ["".join(IDX_TO_BASE[int(b)] for b in row) for row in indices]


def _ckpt_path(ckpt_dir: Union[str, Path], cell: str) -> Path:
    return Path(ckpt_dir) / f"human_paired_{CELL_TAG[cell]}.ckpt"


class PromoterOraclePool:
    """Loads three per-cell EnformerModel checkpoints once; shared across adapters.

    EnformerModel.forward takes (N, L, 4) channels-last one-hot and returns
    (N, 1) scalar activity (linear, loss="mse"). All parameters are frozen.
    """

    def __init__(
        self,
        ckpt_dir: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
        cells=PROMOTER_CELLS,
    ):
        self.cells = list(cells)
        self.device = torch.device(device)
        self.models = {}
        for cell in self.cells:
            ckpt = _ckpt_path(ckpt_dir, cell)
            if not ckpt.exists():
                raise FileNotFoundError(f"missing oracle ckpt for {cell}: {ckpt}")
            # Force pretrained=False on instantiation: some nodes have a
            # transformers version with a CVE-2025-32434 check that refuses
            # `Enformer.from_pretrained` under torch<2.6. The state_dict
            # loaded next line overwrites the init weights anyway, so the
            # pretrained download is wasted — skip it entirely.
            model = EnformerModel.load_from_checkpoint(
                str(ckpt), map_location="cpu", pretrained=False
            )
            model = model.to(self.device).eval()
            for p in model.parameters():
                p.requires_grad = False
            self.models[cell] = model


class PromoterCellAdapter:
    """Per-cell scorer, signature matches GrELUOracleAdapter.__call__.

    Accepts a list of DNA strings, a numpy (N, L) index array, or a torch
    (N, L) LongTensor. Returns a (N,) tensor on the pool's device.
    """

    def __init__(self, pool: PromoterOraclePool, cell: str):
        if cell not in pool.models:
            raise KeyError(f"cell {cell} not in pool.cells={pool.cells}")
        self.model = pool.models[cell]
        self.device = pool.device
        self.cell = cell

    @torch.no_grad()
    def __call__(self, dna_input):
        if isinstance(dna_input, list) and len(dna_input) > 0 and isinstance(
            dna_input[0], str
        ):
            indices = dna_to_indices(dna_input)
            tensor = torch.from_numpy(indices).long()
        elif isinstance(dna_input, np.ndarray):
            tensor = torch.from_numpy(dna_input).long()
        else:
            tensor = dna_input.long()

        tensor = tensor.to(self.device)
        onehot = F.one_hot(tensor, num_classes=4).float()  # (N, L, 4) channels-last
        preds = self.model(onehot)  # (N, 1)
        return preds.squeeze(-1)


class PromoterMultiAdapter:
    """DPS-capable multi-cell adapter for GPA.

    dps_forward takes a (B, 4, L) channels-first *soft* one-hot (gradient
    flows back through the oracle) and returns (B,) reward:
        reward = target_pred - penalty_weight * mean(off_target_preds)
    Matches GrELUOracleAdapter.dps_forward() signature.
    """

    def __init__(
        self,
        pool: PromoterOraclePool,
        target_cell: str,
        penalty_weight: float = 0.0,
        penalty_cells=None,
    ):
        if target_cell not in pool.cells:
            raise KeyError(f"target_cell {target_cell} not in {pool.cells}")
        self.pool = pool
        self.target_cell = target_cell
        self.target_idx = pool.cells.index(target_cell)
        self.penalty_weight = float(penalty_weight)
        if penalty_cells is None:
            self.penalty_cells = [c for c in pool.cells if c != target_cell]
        else:
            for c in penalty_cells:
                if c not in pool.cells:
                    raise KeyError(f"penalty cell {c} not in {pool.cells}")
            self.penalty_cells = list(penalty_cells)

    def dps_forward(self, soft_onehot: torch.Tensor) -> torch.Tensor:
        # (B, 4, L) channels-first -> (B, L, 4) channels-last for EnformerModel
        x = soft_onehot.permute(0, 2, 1)
        target_pred = self.pool.models[self.target_cell](x).squeeze(-1)
        if self.penalty_weight == 0.0 or not self.penalty_cells:
            return target_pred
        off_preds = torch.stack(
            [self.pool.models[c](x).squeeze(-1) for c in self.penalty_cells],
            dim=-1,
        )
        return target_pred - self.penalty_weight * off_preds.mean(dim=-1)

    @torch.no_grad()
    def score_all(self, dna_input) -> torch.Tensor:
        """All-cell scoring, returns (N, len(cells)) tensor."""
        adapters = [PromoterCellAdapter(self.pool, c) for c in self.pool.cells]
        cols = [a(dna_input) for a in adapters]
        return torch.stack(cols, dim=-1)

    @torch.no_grad()
    def score(self, sequences_tensor, batch_size: int = 256):
        """Matches scripts/rerd_comparison/enformer_oracle.py:EnformerOracle.score.

        Returns (target_scores (N,) np.ndarray, gc_fractions (N,) np.ndarray).
        Caches per-cell raw predictions in self._last_all_preds for later
        fitness computation without a second forward pass.
        """
        tensor = sequences_tensor.to(self.pool.device) if isinstance(
            sequences_tensor, torch.Tensor
        ) else torch.as_tensor(sequences_tensor).to(self.pool.device)
        tensor = tensor.long()
        N = tensor.shape[0]
        per_cell = {c: [] for c in self.pool.cells}
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            indices = tensor[start:end]
            onehot = F.one_hot(indices, num_classes=4).float()  # (B, L, 4)
            for cell in self.pool.cells:
                preds = self.pool.models[cell](onehot).squeeze(-1)  # (B,)
                per_cell[cell].append(preds.cpu())
        all_preds = {c: torch.cat(per_cell[c], dim=0).numpy() for c in self.pool.cells}
        self._last_all_preds = all_preds
        target_scores = all_preds[self.target_cell]
        gc = ((tensor == 1) | (tensor == 2)).float().mean(dim=1).cpu().numpy()
        return target_scores, gc

    def compute_fitness(self, target_scores):
        """target - penalty_weight * mean(off-target cached preds)."""
        if self.penalty_weight == 0.0 or not self.penalty_cells:
            return target_scores
        if not hasattr(self, "_last_all_preds"):
            raise RuntimeError("call score() before compute_fitness()")
        off = np.stack([self._last_all_preds[c] for c in self.penalty_cells], axis=0)
        return target_scores - self.penalty_weight * off.mean(axis=0)


def load_promoter_oracle_ranges(path: Union[str, Path]):
    """Read {cell: {min, max}} from promoter oracle_ranges.json."""
    with open(path) as f:
        ranges = json.load(f)
    for cell in PROMOTER_CELLS:
        if cell not in ranges:
            raise KeyError(f"{path} missing cell {cell}")
    return ranges


def get_fitness_info(cell: str, ranges_path: Union[str, Path], seq_len: int = 250):
    """Matches scripts/ctrl_dna_comparison/oracle_adapter.py:get_fitness_info_from_oracle
    signature so the enhancer-style (length, min, max) tuple is returned."""
    ranges = load_promoter_oracle_ranges(ranges_path)
    return (seq_len, float(ranges[cell]["min"]), float(ranges[cell]["max"]))
