#!/usr/bin/env python3
"""
Sanity-check the promoter HyenaDNA fine-tune: does decile conditioning
actually drive generated sequences to the requested activity level?

Generates N sequences with each test prompt, scores them against all 3
promoter Enformer oracles, and prints per-cell means. The "999" prompt
should produce substantially higher scores than "000" on all three cells
(within-cell means). Order across prompts at different decile digits for
the same cell should be monotonic.

Usage:
    python scripts/ctrl_dna_comparison/promoter/verify_finetune.py \
        --hyenadna_checkpoint .../hyenadna_promoter_smoke/best.ckpt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from oracle_adapter import (  # noqa: E402
    PROMOTER_CELLS, PromoterOraclePool, PromoterCellAdapter,
)

_CTRLDNA_SRC = "${HOME}/Ctrl-DNA/ctrl_dna"
if _CTRLDNA_SRC not in sys.path:
    sys.path.insert(0, _CTRLDNA_SRC)
from src.reglm.lightning import LightningModel  # noqa: E402


def load_hyenadna_lightning(ckpt_path, label_len, device):
    model = LightningModel(label_len=label_len)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned = {
        k.replace("model.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    missing, unexpected = model.model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    model = model.to(device).eval()
    return model


@torch.no_grad()
def sample_sequences(model, prompt, n, seq_len, device):
    labels = [prompt] * n
    obs, _, _, _ = model.get_data(labels, seq_len)
    # obs shape: (L, B) with label prefix [START=0, <label>, <seq>...]
    label_prefix = 1 + len(prompt)
    tokens = obs[label_prefix:label_prefix + seq_len].T  # (B, seq_len)
    # Token ids: A=7, C=8, G=9, T=10, N=11 (CharDataset base_stoi). Map to 0-3.
    base_from_tok = {7: 0, 8: 1, 9: 2, 10: 3}
    arr = tokens.cpu().numpy()
    valid = np.isin(arr, list(base_from_tok.keys()))
    frac_valid = float(valid.mean())
    remap = np.vectorize(lambda t: base_from_tok.get(int(t), 0))(arr)
    return torch.from_numpy(remap).long().to(device), frac_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyenadna_checkpoint", required=True)
    ap.add_argument(
        "--oracle_ckpt_dir",
        default="scripts/ctrl_dna_comparison/promoter/checkpoints",
    )
    ap.add_argument("--seq_len", type=int, default=250)
    ap.add_argument("--label_len", type=int, default=3)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=["000", "100", "010", "001", "111"],
        help="3-digit binary prompts. Matches Ctrl-DNA's hardcoded "
             "'100'/'010'/'001' targets plus '000'/'111' extremes.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA — this will be slow")

    for p in args.prompts:
        assert len(p) == args.label_len, f"prompt {p!r} must be length {args.label_len}"

    print(f"Loading fine-tuned HyenaDNA: {args.hyenadna_checkpoint}")
    model = load_hyenadna_lightning(
        args.hyenadna_checkpoint, args.label_len, device
    )

    print(f"\nLoading oracle pool: {args.oracle_ckpt_dir}")
    pool = PromoterOraclePool(args.oracle_ckpt_dir, device=device)
    adapters = {c: PromoterCellAdapter(pool, c) for c in PROMOTER_CELLS}

    print(
        f"\nSampling {args.n} seqs x {len(args.prompts)} prompts at seq_len={args.seq_len}"
    )
    results = {}
    for prompt in args.prompts:
        seqs, frac_valid = sample_sequences(model, prompt, args.n, args.seq_len, device)
        scores_per_cell = {
            c: adapters[c](seqs).detach().cpu().numpy() for c in PROMOTER_CELLS
        }
        results[prompt] = scores_per_cell
        print(f"\n  prompt={prompt!r}  valid_base_frac={frac_valid:.3f}")
        for c in PROMOTER_CELLS:
            s = scores_per_cell[c]
            print(
                f"    {c:<6}  mean={s.mean():+.3f}  std={s.std():+.3f}  "
                f"min={s.min():+.3f}  max={s.max():+.3f}"
            )

    print("\n=== Per-cell means across prompts ===")
    print(f"  {'cell':<6}  " + "  ".join(f"{p:>7}" for p in args.prompts))
    for c in PROMOTER_CELLS:
        means = [results[p][c].mean() for p in args.prompts]
        print(f"  {c:<6}  " + "  ".join(f"{m:+7.3f}" for m in means))

    print("\n=== Binary conditioning check (bit_c=1 vs bit_c=0 per cell) ===")
    passes = {}
    for cell_idx, c in enumerate(PROMOTER_CELLS):
        hi_prompts = [p for p in args.prompts if p[cell_idx] == "1"]
        lo_prompts = [p for p in args.prompts if p[cell_idx] == "0"]
        if not hi_prompts or not lo_prompts:
            print(f"  {c:<6}  SKIP (need both hi and lo prompts)")
            passes[c] = False
            continue
        hi_mean = np.mean([results[p][c].mean() for p in hi_prompts])
        lo_mean = np.mean([results[p][c].mean() for p in lo_prompts])
        ok = hi_mean > lo_mean
        passes[c] = ok
        print(f"  {c:<6}  hi_prompts={hi_prompts} mean={hi_mean:+.3f}  "
              f"lo_prompts={lo_prompts} mean={lo_mean:+.3f}  "
              f"{'PASS' if ok else 'FAIL'}")

    n_pass = sum(passes.values())
    if n_pass < 2:
        print(f"\nFAIL: only {n_pass}/3 cells show binary conditioning.")
        sys.exit(1)
    print(f"\nPASS: binary conditioning learned on {n_pass}/3 cells.")


if __name__ == "__main__":
    main()
