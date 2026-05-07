"""Three-head LentiMPRA LegNet wrapper (K562 + HepG2 + WTC11).

Reuses the K562Oracle's adapter-injection / one-hot plumbing. Loads each
cell-type checkpoint independently (LegNet is per-cell, not multi-task) and
exposes `score_all(indices)` returning a dict of three score arrays plus GC.

Checkpoint paths (verified):
    K562  : CFG-SDDD/model_zoo/lentimpra/oracle_models/best_model-epoch=24-val_pearson=0.814.ckpt
    HepG2 : D3-DNA-Discrete-Diffusion/model_zoo/lentimpra/oracle_models/best_model-epoch=15-val_pearson=0.741.ckpt
    WTC11 : D3-DNA-Discrete-Diffusion/model_zoo/lentimpra/oracle_models/best_model_wtc11-epoch=28-val_pearson=0.712.ckpt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts.k562_mdlm_gpa.k562_oracle import K562Oracle, load_k562_oracle


CKPT_K562 = ("${GPA_REPO_ROOT}/model_zoo/lentimpra/"
             "oracle_models/best_model-epoch=24-val_pearson=0.814.ckpt")
CKPT_HEPG2 = ("${HOME}/D3-DNA-Discrete-Diffusion/model_zoo/"
              "lentimpra/oracle_models/best_model-epoch=15-val_pearson=0.741.ckpt")
CKPT_WTC11 = ("${HOME}/D3-DNA-Discrete-Diffusion/model_zoo/"
              "lentimpra/oracle_models/best_model_wtc11-epoch=28-val_pearson=0.712.ckpt")

CELL_CKPTS = {"k562": CKPT_K562, "hepg2": CKPT_HEPG2, "wtc11": CKPT_WTC11}


class MultiCellOracle:
    """Score sequences through K562/HepG2/WTC11 LegNets with shared plumbing."""

    def __init__(self, device: str = "cuda", cells=("k562", "hepg2", "wtc11")):
        self.device = device
        self.cells = tuple(cells)
        self.oracles = {}
        for cell in self.cells:
            ckpt = CELL_CKPTS[cell]
            assert Path(ckpt).exists(), f"Missing checkpoint: {ckpt}"
            lit = load_k562_oracle(ckpt, device=device)
            self.oracles[cell] = K562Oracle(lit, device=device)
            print(f"[MultiCellOracle] loaded {cell:5s} <- {Path(ckpt).name}",
                  flush=True)

    @torch.no_grad()
    def score_all(self, sequences_tensor, batch_size: int = 1024):
        """Return dict {cell: np.array scores} and gc_fracs (shared)."""
        sequences_tensor = sequences_tensor.to(self.device)
        out = {}
        gc = None
        for cell in self.cells:
            scores, gc_fracs = self.oracles[cell].score(
                sequences_tensor, batch_size=batch_size)
            out[cell] = scores
            if gc is None:
                gc = gc_fracs
        return out, gc

    @torch.no_grad()
    def specificity(self, sequences_tensor, target: str = "k562",
                    batch_size: int = 1024):
        """K562 − mean(other cells) — higher = more specific."""
        scores, _ = self.score_all(sequences_tensor, batch_size=batch_size)
        off = np.mean([scores[c] for c in self.cells if c != target], axis=0)
        return scores[target] - off, scores
