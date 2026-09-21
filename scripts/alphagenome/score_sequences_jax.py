#!/usr/bin/env python3
"""Score DNA sequences with the JAX fine-tuned AlphaGenome encoder.

Replacement for the (broken) ``alphagenome_ft_mpra.oracle.load_oracle`` flow on
the new restored ckpts at ``${GPA_SHARED_ROOT}/models/alphagenome_encoder/jax/``.
Those ckpts ship with ``"use_encoder_output": false`` in ``config.json`` even
though the head requires ``encoder_output``; hence ``EncoderMPRAHead requires
'encoder_output' in embeddings object``. The training script
(``alphagenome_FT_MPRA/scripts/finetune_mpra.py:657``) sets
``use_encoder_output=True`` explicitly, so the saved flag is wrong.

This module materialises a per-cell tmp checkpoint dir whose
``config.json`` has the flag flipped to ``true`` and whose ``checkpoint/``
sub-tree is symlinked back to the read-only shared dir (~180 MB per cell, no
copy). Then it calls ``load_oracle`` against the tmp dir.

Picks PyT-style scales (raw expression for K562, log-like for HepG2/WTC11) out
of the equation entirely — JAX outputs a single ~0..3 log range across all
three cells, so cross-cell specificity is meaningful again.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

JAX_CHECKPOINTS = {
    "k562":  Path(os.path.expandvars("${GPA_SHARED_ROOT}/models/alphagenome_encoder/jax/mpra-K562-optimal/stage1")),
    "hepg2": Path(os.path.expandvars("${GPA_SHARED_ROOT}/models/alphagenome_encoder/jax/mpra-HepG2-optimal/stage1")),
    "wtc11": Path(os.path.expandvars("${GPA_SHARED_ROOT}/models/alphagenome_encoder/jax/mpra-WTC11-optimal/stage1")),
}
JAX_CHECKPOINTS_STAGE2 = {
    "k562":  Path(os.path.expandvars("${GPA_SHARED_ROOT}/models/alphagenome_encoder/jax/mpra-K562-optimal/stage2")),
    "hepg2": Path(os.path.expandvars("${GPA_SHARED_ROOT}/models/alphagenome_encoder/jax/mpra-HepG2-optimal/stage2")),
    "wtc11": Path(os.path.expandvars("${GPA_SHARED_ROOT}/models/alphagenome_encoder/jax/mpra-WTC11-optimal/stage2")),
}

ADAPTER_5 = "AGGACCGGATCAACT"
ADAPTER_3 = "CATTGCGTGAACCGA"

IDX_TO_BASE = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(IDX_TO_BASE)}


def indices_to_onehot(indices: np.ndarray) -> np.ndarray:
    N, L = indices.shape
    oh = np.zeros((N, L, 4), dtype=np.float32)
    for i in range(4):
        oh[:, :, i] = (indices == i).astype(np.float32)
    return oh


def sequences_to_onehot(seqs: list[str]) -> np.ndarray:
    indices = np.array(
        [[BASE_TO_IDX.get(c.upper(), 0) for c in s] for s in seqs],
        dtype=np.int64,
    )
    return indices_to_onehot(indices)


@contextmanager
def patched_checkpoint(src_dir: Path):
    """Yield a tmp dir whose config.json forces use_encoder_output=True and
    whose checkpoint/ sub-tree is symlinked to the shared read-only ckpt."""
    src_dir = Path(src_dir).resolve()
    cfg_path = src_dir / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["use_encoder_output"] = True

    with tempfile.TemporaryDirectory(prefix="ag_jax_ckpt_") as td:
        td = Path(td)
        with open(td / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        # Symlink every entry under the shared ckpt dir except config.json.
        for entry in src_dir.iterdir():
            if entry.name == "config.json":
                continue
            os.symlink(entry, td / entry.name)
        yield td


def load_jax_oracle(cell: str, init_seq_len: int = 281, stage: str = "stage1"):
    """Load a JAX MPRA oracle for one cell, working around two bugs in
    ``load_oracle``: (1) it reads ``use_encoder_output: false`` from the
    saved config but EncoderMPRAHead requires it True, and (2) it does not
    forward ``init_seq_len`` to ``load_checkpoint`` so the template defaults
    to 2**14, yielding 128 encoder positions and a (196608, 2048) shape that
    does not match the saved (4608, 2048) flatten layer.

    init_seq_len=281 → ceil(281/128)=3 encoder positions × 1536 hidden = 4608,
    matching the matched-seed construct length: left(15)+payload(200)+right(15)
    +promoter(35)+barcode(15)=280bp."""
    from alphagenome.models import dna_output
    from alphagenome_ft import (
        load_checkpoint,
        register_custom_head,
        HeadConfig,
        HeadType,
    )
    from alphagenome_ft_mpra.mpra_heads import EncoderMPRAHead
    from alphagenome_ft_mpra.oracle import MPRAOracle

    if stage == "stage2":
        src = JAX_CHECKPOINTS_STAGE2[cell]
    elif stage == "stage1":
        src = JAX_CHECKPOINTS[cell]
    else:
        raise ValueError(f"Unknown stage: {stage}")
    with open(src / "config.json") as f:
        cfg = json.load(f)
    head_name = cfg["custom_heads"][0]
    head_cfg = cfg["head_configs"][head_name]
    head_metadata = head_cfg.get("metadata", {}) or {}
    num_tracks = int(head_cfg.get("num_tracks", 1))

    register_custom_head(
        head_name,
        EncoderMPRAHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name=head_name,
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=num_tracks,
            metadata=head_metadata,
        ),
    )

    with patched_checkpoint(src) as td:
        model = load_checkpoint(str(td), init_seq_len=init_seq_len)

    return MPRAOracle(
        model,
        head_name=head_name,
        pooling_type=head_metadata.get("pooling_type", "sum"),
        center_bp=int(head_metadata.get("center_bp", 256)),
        left_adapter=ADAPTER_5,
        right_adapter=ADAPTER_3,
    )


def score(oracle, onehot: np.ndarray, batch_size: int = 64) -> np.ndarray:
    out = []
    N = onehot.shape[0]
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        preds = oracle.predict(onehot[s:e], mode="core", batch_size=batch_size)
        out.append(np.asarray(preds).reshape(-1))
    return np.concatenate(out).astype(np.float64)


if __name__ == "__main__":
    # Smoke test: load each cell, score 32 seqs from one GPA pool, print scales.
    import h5py
    h5p = os.path.expandvars("${GPA_REPO_ROOT}/results/k562_mdlm_gpa/run_v13_nodps_bf10_b1k_cap10_ism50178/gpa_output_best_eval.h5")
    with h5py.File(h5p, "r") as f:
        idx = f["indices"][:32]
    oh = indices_to_onehot(idx)
    print(f"smoke test: N={oh.shape[0]} L={oh.shape[1]}", flush=True)
    for cell in ("k562", "hepg2", "wtc11"):
        print(f"\n[{cell}]", flush=True)
        oracle = load_jax_oracle(cell)
        s = score(oracle, oh, batch_size=32)
        print(f"[{cell}] N={len(s)} min={s.min():.4f} mean={s.mean():.4f} "
              f"max={s.max():.4f} std={s.std():.4f}", flush=True)
        del oracle
