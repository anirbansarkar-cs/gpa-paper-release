"""Evaluate trained promoter oracles on val + test splits.

Prints Pearson r and Spearman rho per cell. Uses the copied `human_paired_*`
checkpoints that will be consumed by Ctrl-DNA / GPA.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

CTRLDNA_SRC = "${HOME}/Ctrl-DNA/ctrl_dna"
if CTRLDNA_SRC not in sys.path:
    sys.path.insert(0, CTRLDNA_SRC)

from src.reglm.regression import EnformerModel, SeqDataset  # noqa: E402

CELL_TAG = {"JURKAT": "jurkat", "K562": "k562", "THP1": "THP1"}


def eval_one(ckpt_path, df_split, cell, batch_size=64, device=0):
    model = EnformerModel.load_from_checkpoint(ckpt_path, map_location="cpu")
    model = model.to(f"cuda:{device}").eval()
    sub = df_split[["sequence", cell]].reset_index(drop=True)
    ds = SeqDataset(sub, seq_len=250)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    preds, trues = [], []
    with torch.no_grad():
        for seq, y in dl:
            seq = seq.to(f"cuda:{device}")
            out = model(seq).detach().cpu().numpy().ravel()
            preds.append(out)
            trues.append(y.numpy().ravel())
    p = np.concatenate(preds)
    t = np.concatenate(trues)
    return pearsonr(p, t)[0], spearmanr(p, t)[0], np.mean((p - t) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    val_df = df.loc[df["is_val"]].reset_index(drop=True)
    test_df = df.loc[df["is_test"]].reset_index(drop=True)
    print(f"val={len(val_df)} test={len(test_df)}")

    rows = []
    for cell, tag in CELL_TAG.items():
        ckpt = Path(args.ckpt_dir) / f"human_paired_{tag}.ckpt"
        for split_name, sdf in [("val", val_df), ("test", test_df)]:
            r, rho, mse = eval_one(str(ckpt), sdf, cell)
            rows.append(
                {"cell": cell, "split": split_name, "pearson": r,
                 "spearman": rho, "mse": mse, "n": len(sdf)}
            )
            print(f"  {cell:<6} {split_name:<4} r={r:.4f} rho={rho:.4f} "
                  f"mse={mse:.4f} n={len(sdf)}", flush=True)

    out = Path(args.ckpt_dir) / "eval_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
