#!/usr/bin/env python3
"""QC sampler for the trained DiMamba backbone.

Samples N unconditional 350 bp sequences and reports per-position Shannon
entropy + mean 3-mer frequency Pearson R against the held-out cCRE test
split (paper Table 1 target: R > 0.98; hard gate 0.95).

Run with:
    python sample_dimamba.py \
        eval.checkpoint_path=<path>/last.ckpt \
        sampling.num_sample_batches=4 sampling.steps=128
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import omegaconf
import pyarrow.parquet as pq
import torch
from scipy.stats import pearsonr

MDLM_ROOT = Path("${HOME}/mdlm")
if str(MDLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MDLM_ROOT))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Apply d3_mamba/torch-2.0 compatibility shims BEFORE importing mdlm.
import _compat_shims as _mdlm_shims  # noqa: E402, F401

import dataloader as mdlm_dataloader  # noqa: E402
import dna_dataloader  # noqa: E402
mdlm_dataloader.get_tokenizer = dna_dataloader.get_tokenizer
mdlm_dataloader.get_dataloaders = dna_dataloader.get_dataloaders

import diffusion as mdlm_diffusion  # noqa: E402

omegaconf.OmegaConf.register_new_resolver("cwd", lambda: ".", replace=True)
omegaconf.OmegaConf.register_new_resolver(
    "device_count", torch.cuda.device_count, replace=True)
omegaconf.OmegaConf.register_new_resolver("eval", eval, replace=True)
omegaconf.OmegaConf.register_new_resolver(
    "div_up", lambda x, y: (x + y - 1) // y, replace=True)

CONFIG_DIR = THIS_DIR / "configs"


def kmer_profile(indices: np.ndarray, k: int = 3) -> np.ndarray:
    N, L = indices.shape
    powers = 4 ** np.arange(k)[::-1]
    ids = np.zeros((N, L - k + 1), dtype=np.int64)
    for i in range(k):
        ids += indices[:, i:L - k + 1 + i] * powers[i]
    freq = np.zeros((N, 4 ** k), dtype=np.float32)
    for n in range(N):
        c = np.bincount(ids[n], minlength=4 ** k)
        s = c.sum()
        if s > 0:
            freq[n] = c / s
    return freq


def per_position_shannon(indices: np.ndarray) -> float:
    L = indices.shape[1]
    out = 0.0
    for p in range(L):
        c = np.bincount(indices[:, p], minlength=4)
        probs = c / c.sum()
        nz = probs[probs > 0]
        out += -(nz * np.log2(nz)).sum()
    return float(out / L)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR),
            config_name="dna_config")
def main(config):
    L.seed_everything(config.seed)
    tokenizer = dna_dataloader.get_tokenizer(config)
    if not config.eval.checkpoint_path:
        raise ValueError("set eval.checkpoint_path=<path>/last.ckpt")
    model = mdlm_diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config,
    ).to("cuda").eval()
    samples = []
    with torch.no_grad():
        for _ in range(config.sampling.num_sample_batches):
            x = model.restore_model_and_sample(num_steps=config.sampling.steps)
            samples.append(x.cpu().numpy())
    indices = np.concatenate(samples, axis=0)[:, :config.model.length]
    indices = np.clip(indices, 0, 3)              # drop PAD/MASK if any
    pool_kmer = kmer_profile(indices).mean(axis=0)

    test_path = Path(config.data.parquet_dir) / "test.parquet"
    table = pq.read_table(str(test_path), columns=["tokens"])
    real_indices = np.asarray(table.column("tokens").to_pylist(),
                              dtype=np.int64)
    rng = np.random.default_rng(seed=42)
    n_ref = min(10000, len(real_indices))
    chosen = rng.choice(len(real_indices), n_ref, replace=False)
    real_indices = real_indices[chosen]
    real_kmer = kmer_profile(real_indices).mean(axis=0)

    r, _ = pearsonr(pool_kmer, real_kmer)
    shannon = per_position_shannon(indices)
    print(f"[QC] n_samples={len(indices):,} "
          f"3mer_pearson_R={r:.4f} per_position_shannon={shannon:.3f}")
    gate_hard = 0.95
    gate_target = 0.98
    if r >= gate_target:
        print(f"[QC] PASS — R > {gate_target}")
    elif r >= gate_hard:
        print(f"[QC] BELOW TARGET ({gate_target}) but ABOVE GATE ({gate_hard})")
    else:
        print(f"[QC] FAIL — R < {gate_hard} (continue training or "
              "investigate)")


if __name__ == "__main__":
    main()
