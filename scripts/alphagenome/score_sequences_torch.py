#!/usr/bin/env python3
"""Score DNA sequences with the PyTorch fine-tuned AlphaGenome encoder.

Replaces the old JAX-based scripts/alphagenome/score_sequences.py, which
loaded `mpra-K562-optimal/stage1` from ${GPA_SHARED_ROOT}/alphagenome_encoder/
(now removed). This version uses EncoderMPRAModel.from_checkpoint with the
new PyTorch ckpts at ${GPA_SHARED_ROOT}/models/alphagenome_encoder/torch/.

Reads CSV with a sequence column, scores each cell type, writes <input>_ag.csv
with one column per cell type: "<cell>_ag".
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch

from alphagenome_encoder_ft import EncoderMPRAModel

CHECKPOINTS = {
    "k562":  "${GPA_SHARED_ROOT}/models/alphagenome_encoder/torch/mpra_K562/finetuned_encoder.pt",
    "hepg2": "${GPA_SHARED_ROOT}/models/alphagenome_encoder/torch/mpra_HepG2/finetuned_encoder.pt",
    "wtc11": "${GPA_SHARED_ROOT}/models/alphagenome_encoder/torch/mpra_WTC11/finetuned_encoder.pt",
}

# LentiMPRA library adapters — GPA / ISM / LEDIDI sequences are 200 bp cores.
# The encoder expects 230 bp inserts (15 bp adapter + 200 bp + 15 bp adapter).
ADAPTER_5 = "AGGACCGGATCAACT"   # 15 bp
ADAPTER_3 = "CATTGCGTGAACCGA"   # 15 bp


def pad_to_230(seq: str) -> str:
    """If a 200 bp core comes in, pad with library adapters to make 230 bp."""
    if len(seq) == 230:
        return seq
    if len(seq) == 200:
        return ADAPTER_5 + seq + ADAPTER_3
    raise ValueError(f"Unexpected seq length {len(seq)}; expected 200 or 230 bp")


def score(model, seqs: list[str], batch_size: int, device: torch.device):
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i : i + batch_size]
        with torch.no_grad():
            preds = model.predict_sequences(
                chunk,
                construct_mode="promoter_barcode",
            )
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        out.extend(list(preds.reshape(-1)))
    return out


IDX_TO_BASE = "ACGT"


def load_seqs_from_h5(path: Path) -> list[str]:
    with h5py.File(path, "r") as f:
        if "indices" in f:
            indices = f["indices"][:]
            return ["".join(IDX_TO_BASE[i] for i in row.astype(int)) for row in indices]
        if "sequences" in f:
            raw = f["sequences"][:]
            return [s.decode() if isinstance(s, bytes) else s for s in raw]
    raise KeyError(f"No /indices or /sequences in {path}")


def append_h5(path: Path, cell_to_preds: dict, replace_old: bool):
    with h5py.File(path, "a") as f:
        for cell, preds in cell_to_preds.items():
            key_new = f"ag_{cell}_scores_v2"
            if key_new in f:
                del f[key_new]
            f.create_dataset(key_new, data=np.asarray(preds, dtype=np.float32))
            if replace_old and cell == "k562" and "ag_k562_scores" in f:
                # Keep the original under a `_jax` suffix; use v2 as primary.
                if "ag_k562_scores_jax" in f:
                    del f["ag_k562_scores_jax"]
                f["ag_k562_scores_jax"] = f["ag_k562_scores"][...]
                del f["ag_k562_scores"]
                f.create_dataset("ag_k562_scores", data=np.asarray(preds, dtype=np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV (sequence column) or H5 (indices/sequences).")
    p.add_argument("--output", default=None, help="CSV to write. Required for CSV inputs.")
    p.add_argument("--append_h5", action="store_true",
                   help="For H5 inputs: write ag_<cell>_scores_v2 datasets back into the input H5.")
    p.add_argument("--replace_h5_k562", action="store_true",
                   help="Also overwrite the H5's ag_k562_scores with the new scores; "
                        "old values stashed at ag_k562_scores_jax.")
    p.add_argument("--seq_column", default="sequence")
    p.add_argument("--cell_types", nargs="+", default=["k562"], choices=list(CHECKPOINTS.keys()))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    inp = Path(args.input)
    is_h5 = inp.suffix in (".h5", ".hdf5")

    if is_h5:
        seqs = load_seqs_from_h5(inp)
        df = None
    else:
        df = pd.read_csv(inp)
        if args.seq_column not in df.columns:
            for cand in ("sequence", "edited_sequence"):
                if cand in df.columns:
                    args.seq_column = cand
                    break
            else:
                raise SystemExit(f"No usable sequence column in {inp}: {list(df.columns)}")
        seqs = df[args.seq_column].astype(str).tolist()

    print(f"[score] {inp}  N={len(seqs)}", flush=True)
    if seqs:
        print(f"[score] first seq len={len(seqs[0])} (will be padded to 230 if 200)", flush=True)
    seqs = [pad_to_230(s) for s in seqs]

    device = torch.device(args.device)
    cell_preds = {}
    for cell in args.cell_types:
        ckpt = CHECKPOINTS[cell]
        print(f"[score] loading {cell} from {ckpt}", flush=True)
        model = EncoderMPRAModel.from_checkpoint(ckpt, device=device)
        model.eval()
        preds = score(model, seqs, args.batch_size, device)
        cell_preds[cell] = preds
        if df is not None:
            df[f"{cell}_ag"] = preds
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[score] done {cell}: mean={pd.Series(preds).mean():.3f}  "
              f"max={pd.Series(preds).max():.3f}", flush=True)

    if is_h5 and args.append_h5:
        append_h5(inp, cell_preds, replace_old=args.replace_h5_k562)
        print(f"[score] appended ag_<cell>_scores_v2 to {inp}", flush=True)

    if df is not None and args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"[score] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
