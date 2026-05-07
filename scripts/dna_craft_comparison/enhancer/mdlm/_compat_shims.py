"""Compatibility shims that must run BEFORE any `mdlm/` module is imported.

`mdlm` was developed against torch ≥ 2.1, mamba_ssm < 2.2, and flash_attn,
none of which match our pinned d3_mamba env (torch 2.0.1+cu118, mamba_ssm
2.2.2, no flash_attn). The shims below patch sys.modules and torch APIs so
the unmodified upstream `mdlm/` code imports and runs cleanly.

Import this module FIRST from any entry point that subsequently imports
mdlm — e.g. `train_dimamba.py` (DiMamba pretrain runner) and
`mdlm_wrapper.py` (GPA's mutation kernel that loads a Diffusion checkpoint).

The shims are idempotent — re-importing this module is a no-op.
"""
from __future__ import annotations

import functools
import importlib
import sys
import types

# 1. flash_attn namespace stub. mdlm/models/__init__.py unconditionally
# imports `dit`, and dit.py top-level-imports flash_attn even when we use
# the dimamba backbone. The flash_attn calls live inside method bodies
# (not class definitions), so a namespace stub is sufficient.
if "flash_attn" not in sys.modules:
    _fa = types.ModuleType("flash_attn")
    _fa.layers = types.ModuleType("flash_attn.layers")
    _fa.layers.rotary = types.ModuleType("flash_attn.layers.rotary")
    _fa.flash_attn_interface = types.ModuleType("flash_attn.flash_attn_interface")
    sys.modules["flash_attn"] = _fa
    sys.modules["flash_attn.layers"] = _fa.layers
    sys.modules["flash_attn.layers.rotary"] = _fa.layers.rotary
    sys.modules["flash_attn.flash_attn_interface"] = _fa.flash_attn_interface

# 2. `mamba_ssm.ops.triton.layernorm` was renamed to `.layer_norm` (with
# underscore) in mamba_ssm 2.2.2. Public symbols are unchanged. Alias the
# old import path so mdlm/models/dimamba.py:28 resolves.
if "mamba_ssm.ops.triton.layernorm" not in sys.modules:
    sys.modules["mamba_ssm.ops.triton.layernorm"] = importlib.import_module(
        "mamba_ssm.ops.triton.layer_norm")

# 3. torch 2.0.1's `torch.no_grad.__init__` doesn't accept a positional
# `orig_func`, so `@torch.no_grad` (no parens) at mdlm/diffusion.py:957
# raises `TypeError: no_grad.__init__() takes 1 positional argument but 2
# were given`. The class is callable in torch ≥ 2.1. Wrap it with a
# callable that handles both `@torch.no_grad` and `@torch.no_grad()` forms.
import torch  # noqa: E402

if not getattr(torch.no_grad, "_d3_mamba_compat_wrapped", False):
    _orig_no_grad = torch.no_grad

    def _no_grad_compat(orig_func=None):  # noqa: D401
        if callable(orig_func):
            @functools.wraps(orig_func)
            def _wrapper(*args, **kwargs):
                with _orig_no_grad():
                    return orig_func(*args, **kwargs)
            return _wrapper
        return _orig_no_grad()

    _no_grad_compat._d3_mamba_compat_wrapped = True  # type: ignore[attr-defined]
    torch.no_grad = _no_grad_compat  # type: ignore[assignment]
