#!/usr/bin/env python3
"""Train DiMamba on ENCODE V4 cCRE under MDLM's diffusion framework.

This script reuses MDLM's core trainer (`mdlm/main.py::_train`) but swaps the
text-oriented `dataloader.get_tokenizer` and `dataloader.get_dataloaders` for
our DNA-specific versions in `dna_dataloader.py`. The Hydra config is loaded
from `configs/dna_config.yaml` in *this* directory (so we don't pollute the
upstream MDLM repo).

Run with:
    python train_dimamba.py [hydra-overrides ...]

By default it expects the parquet shards at
`results/dna_craft_comparison/enhancer/ccre/{train,val,test}.parquet` produced
by Track A (`data/curate_encode_ccre.py`).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

# Per-rank triton cache dir. Triton 2.0.0's filelock race on a shared
# autotune cache causes `FileNotFoundError: [Errno 2]` when 4 DDP ranks
# concurrently compile the same kernel — `os.fstat(fd).st_nlink == 0` after
# another rank unlinks the file. Pinning each rank to its own cache dir
# eliminates the contention and is harmless (cache is ~10 KB per kernel).
_local_rank = os.environ.get("LOCAL_RANK", "0")
_job = os.environ.get("SLURM_JOB_ID", "local")
os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_{_job}_rank{_local_rank}"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

# Stub flash_attn before importing mdlm: mdlm/models/__init__.py pulls in dit
# unconditionally, and dit.py top-level-imports flash_attn even when we use the
# dimamba backbone. The flash_attn calls live inside method bodies (not class
# definitions), so a namespace stub is sufficient — they're never reached.
if "flash_attn" not in sys.modules:
    _fa = types.ModuleType("flash_attn")
    _fa.layers = types.ModuleType("flash_attn.layers")
    _fa.layers.rotary = types.ModuleType("flash_attn.layers.rotary")
    _fa.flash_attn_interface = types.ModuleType("flash_attn.flash_attn_interface")
    sys.modules["flash_attn"] = _fa
    sys.modules["flash_attn.layers"] = _fa.layers
    sys.modules["flash_attn.layers.rotary"] = _fa.layers.rotary
    sys.modules["flash_attn.flash_attn_interface"] = _fa.flash_attn_interface

# mamba_ssm 2.2.2 renamed `mamba_ssm.ops.triton.layernorm` (no underscore) to
# `mamba_ssm.ops.triton.layer_norm` (underscore). mdlm/models/dimamba.py:28
# uses the old name. Alias the module so the old import resolves to the new
# implementation; the public symbols (RMSNorm, layer_norm_fn, rms_norm_fn) are
# unchanged between versions.
if "mamba_ssm.ops.triton.layernorm" not in sys.modules:
    import importlib
    sys.modules["mamba_ssm.ops.triton.layernorm"] = importlib.import_module(
        "mamba_ssm.ops.triton.layer_norm")

import hydra
import lightning as L
import omegaconf
import torch

# torch 2.0.1's `torch.no_grad` class does not accept a positional `orig_func`
# argument, so `@torch.no_grad` (no parens) raises TypeError. mdlm/diffusion.py
# has one such site (sample_subs_guidance, line 957). Replace with a callable
# that supports both `@torch.no_grad`, `@torch.no_grad()`, and the context
# manager form.
import functools as _functools
_orig_no_grad = torch.no_grad


def _no_grad_compat(orig_func=None):
    if callable(orig_func):
        @_functools.wraps(orig_func)
        def _wrapper(*args, **kwargs):
            with _orig_no_grad():
                return orig_func(*args, **kwargs)
        return _wrapper
    return _orig_no_grad()


torch.no_grad = _no_grad_compat

# 1. Make MDLM source importable.
MDLM_ROOT = Path("${HOME}/mdlm")
if str(MDLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MDLM_ROOT))

# 2. Make our DNA dataloader importable and patch MDLM's dataloader module.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import dataloader as mdlm_dataloader  # noqa: E402  (mdlm/dataloader.py)
import dna_dataloader  # noqa: E402

mdlm_dataloader.get_tokenizer = dna_dataloader.get_tokenizer
mdlm_dataloader.get_dataloaders = dna_dataloader.get_dataloaders

# 3. Pull MDLM's training entry points. main.py registers the four hydra
# resolvers ('cwd', 'device_count', 'eval', 'div_up') at module-level, so we
# don't need to pre-register them ourselves — doing so triggers a duplicate-
# registration error inside main's own register call (which doesn't pass
# replace=True).
import main as mdlm_main  # noqa: E402

# NOTE: an earlier iteration of this file forced fused_add_norm=False to dodge
# triton's JIT linker failure (`ld: cannot find -lcuda`). That broke adaln
# time-conditioning (mdlm/models/dimamba.py raises `NotImplementedError: adaln
# only implemented for fused_add_norm` on the non-fused path). The triton
# link issue is now solved properly via CUDA-stub LIBRARY_PATH in the sbatch,
# so we leave fused_add_norm at its default (True) and let triton compile the
# fused kernels.

CONFIG_DIR = THIS_DIR / "configs"


@hydra.main(version_base=None, config_path=str(CONFIG_DIR),
            config_name="dna_config")
def main(config):
    L.seed_everything(config.seed)
    mdlm_main._print_config(config, resolve=True, save_cfg=True)
    logger = __import__("utils").get_logger("train_dimamba")
    if config.mode == "train":
        mdlm_main._train(config, logger, dna_dataloader.get_tokenizer(config))
    elif config.mode == "ppl_eval":
        mdlm_main._ppl_eval(config, logger,
                            dna_dataloader.get_tokenizer(config))
    else:
        raise ValueError(f"unsupported mode {config.mode!r}")


if __name__ == "__main__":
    main()
