#!/usr/bin/env python3
"""Quality metrics for the 6 split-model Enformer oracles.

For each (cell, half) pair, computes:
  * inner_val Pearson R / Spearman / MSE — held-out 10% of the OWN half
    (matches train_enformer_split.py's random_state=42, test_size=0.1)
  * cross_half Pearson R — predictions on the OTHER half (full),
    measuring how well the design oracle generalises to eval-half data
    and vice-versa.

The R≥0.85 threshold previously used here was a self-imposed engineering
target with no external basis (DNA-CRAFT paper does not publish their
oracle's R). We now just report the metrics; downstream code consumes the
JSON directly.

Output:
  results/dna_craft_comparison/enhancer/enformer_oracles/eval_gate.json
  results/dna_craft_comparison/enhancer/enformer_oracles/eval_gate.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

CTRL_DNA_SRC = Path("${HOME}/Ctrl-DNA/ctrl_dna/src")
sys.path.insert(0, str(CTRL_DNA_SRC))
from reglm.regression import EnformerModel, SeqDataset  # noqa: E402

EnformerModel.validation_epoch_end = None  # PL 2.x compat

PROJECT_DIR = Path("${GPA_REPO_ROOT}")
SPLIT_DIR = PROJECT_DIR / "results/dna_craft_comparison/enhancer/gosai_splits"
OUT_DIR = PROJECT_DIR / "results/dna_craft_comparison/enhancer/enformer_oracles"

CELL_COLUMNS = {
    "hepg2": "HepG2_log2FC",
    "k562":  "K562_log2FC",
    "sknsh": "SKNSH_log2FC",
}
HALF_FILES = {"design": "half_A.csv", "eval": "half_B.csv"}
OPP = {"design": "eval", "eval": "design"}
SEED = 42
INNER_VAL_FRAC = 0.1
BATCH = 512
DEVICE = 0


def _load_half(cell: str, half: str) -> pd.DataFrame:
    col = CELL_COLUMNS[cell]
    df = pd.read_csv(SPLIT_DIR / HALF_FILES[half], usecols=["sequence", col])
    df = df.rename(columns={col: "label"})
    return df[["sequence", "label"]]


def _inner_val(cell: str, half: str) -> pd.DataFrame:
    df = _load_half(cell, half)
    _, val_df = train_test_split(df, test_size=INNER_VAL_FRAC, random_state=SEED)
    return val_df.reset_index(drop=True)


def _score(model: EnformerModel, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ds = SeqDataset(df, seq_len=200)
    preds = model.predict_on_dataset(ds, device=DEVICE, num_workers=4, batch_size=BATCH)
    preds = preds.reshape(-1)
    labels = df["label"].to_numpy(dtype=np.float32)
    return preds, labels


def _stats(pred: np.ndarray, true: np.ndarray) -> dict:
    r, _ = pearsonr(pred, true)
    rho, _ = spearmanr(pred, true)
    mse = float(np.mean((pred - true) ** 2))
    return {"pearson_r": float(r), "spearman_rho": float(rho), "mse": mse, "n": int(len(pred))}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"seed": SEED, "inner_val_frac": INNER_VAL_FRAC, "cells": {}}

    for cell in CELL_COLUMNS:
        results["cells"][cell] = {}
        for half in HALF_FILES:
            ckpt = OUT_DIR / f"enformer_{cell}_{half}" / "best.ckpt"
            if not ckpt.exists():
                print(f"[skip] {cell}/{half}: no checkpoint at {ckpt}", flush=True)
                continue

            t0 = time.time()
            print(f"\n=== {cell}/{half} ===", flush=True)
            print(f"[load] {ckpt}", flush=True)
            model = EnformerModel.load_from_checkpoint(str(ckpt))
            model.eval()

            val_df = _inner_val(cell, half)
            print(f"[inner_val] n={len(val_df)}", flush=True)
            pred, true = _score(model, val_df)
            inner = _stats(pred, true)
            inner["dt"] = round(time.time() - t0, 1)
            print(f"[inner_val] R={inner['pearson_r']:.4f}  rho={inner['spearman_rho']:.4f}  "
                  f"MSE={inner['mse']:.4f}  n={inner['n']}  ({inner['dt']}s)", flush=True)

            t1 = time.time()
            other = _load_half(cell, OPP[half])
            print(f"[cross_half=>{OPP[half]}] n={len(other)}", flush=True)
            pred_x, true_x = _score(model, other)
            cross = _stats(pred_x, true_x)
            cross["dt"] = round(time.time() - t1, 1)
            print(f"[cross_half] R={cross['pearson_r']:.4f}  rho={cross['spearman_rho']:.4f}  "
                  f"MSE={cross['mse']:.4f}  n={cross['n']}  ({cross['dt']}s)", flush=True)

            results["cells"][cell][half] = {
                "inner_val": inner,
                f"cross_half_{OPP[half]}": cross,
                "ckpt": str(ckpt),
            }

            del model
            import torch
            torch.cuda.empty_cache()

    out_json = OUT_DIR / "eval_gate.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[json] {out_json}", flush=True)

    rows = ["| Cell | Half | inner_val R | inner_val rho | inner_val MSE | n | cross_half R |",
            "|---|---|---|---|---|---|---|"]
    for cell in CELL_COLUMNS:
        for half in HALF_FILES:
            d = results["cells"].get(cell, {}).get(half)
            if d is None:
                rows.append(f"| {cell} | {half} | — | — | — | — | — |")
                continue
            iv = d["inner_val"]
            cr = d[f"cross_half_{OPP[half]}"]["pearson_r"]
            rows.append(f"| {cell} | {half} | {iv['pearson_r']:.4f} | {iv['spearman_rho']:.4f} "
                        f"| {iv['mse']:.4f} | {iv['n']} | {cr:.4f} |")

    md = (OUT_DIR / "eval_gate.md")
    md.write_text("# Enformer 50/50-split oracle quality metrics\n\n"
                  + "\n".join(rows) + "\n")
    print(f"[md]   {md}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
