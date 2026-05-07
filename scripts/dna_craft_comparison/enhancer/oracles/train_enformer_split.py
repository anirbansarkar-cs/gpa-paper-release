#!/usr/bin/env python3
"""Train one Enformer (per cell, per half) on Gosai MPRA under the DNA-CRAFT
50/50 split-model protocol.

Half A -> Design-Model (used by GPA during optimization)
Half B -> Evaluation-Model (used post-hoc to score MinGap for reporting)

Adapted from scripts/ctrl_dna_comparison/train_ctrldna_oracle.py. Differences:
- Reads pre-materialized half_A.csv / half_B.csv (from gosai_50_50_split.py)
- 90/10 inner train/val split within the chosen half for early stopping
- Output path layout mirrors the plan:
    checkpoints/enformer_{cell}_{half}/best.ckpt
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

CTRL_DNA_SRC = Path("${HOME}/Ctrl-DNA/ctrl_dna/src")
sys.path.insert(0, str(CTRL_DNA_SRC))
from reglm.regression import EnformerModel, SeqDataset  # noqa: E402

EnformerModel.validation_epoch_end = None  # PL 2.x compat

PROJECT_DIR = Path("${GPA_REPO_ROOT}")
SPLIT_DIR = PROJECT_DIR / "results/dna_craft_comparison/enhancer/gosai_splits"
DEFAULT_OUT = PROJECT_DIR / "results/dna_craft_comparison/enhancer/enformer_oracles"

CELL_COLUMNS = {
    "hepg2": "HepG2_log2FC",
    "k562":  "K562_log2FC",
    "sknsh": "SKNSH_log2FC",
}
HALF_FILES = {"design": "half_A.csv", "eval": "half_B.csv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=str, required=True, choices=list(CELL_COLUMNS))
    ap.add_argument("--half", type=str, required=True, choices=list(HALF_FILES))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--loss", type=str, default="mse", choices=["poisson", "mse"])
    ap.add_argument("--inner_val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--resume_from", type=str, default=None,
                    help="Path to a Lightning .ckpt to resume training from. "
                         "When set, optimizer/lr-scheduler state is restored "
                         "and training continues until --max_epochs.")
    args = ap.parse_args()

    col = CELL_COLUMNS[args.cell]
    half_csv = SPLIT_DIR / HALF_FILES[args.half]
    if args.save_dir is None:
        args.save_dir = str(DEFAULT_OUT / f"enformer_{args.cell}_{args.half}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] cell={args.cell} half={args.half} col={col}")
    print(f"[config] save_dir={save_dir}")
    print(f"[config] csv={half_csv}")

    df = pd.read_csv(half_csv, usecols=["sequence", col])
    print(f"[data] half rows: {len(df)}")
    df = df.rename(columns={col: "label"})
    df = df[["sequence", "label"]]

    train_df, val_df = train_test_split(
        df, test_size=args.inner_val_frac, random_state=args.seed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"[data] inner train: {len(train_df)}  inner val: {len(val_df)}")
    print(f"[data] label range: [{df.label.min():.3f}, {df.label.max():.3f}] "
          f"mean={df.label.mean():.3f} std={df.label.std():.3f}")

    train_ds = SeqDataset(train_df, seq_len=200)
    val_ds = SeqDataset(val_df, seq_len=200)

    print(f"[model] EnformerModel(dim=1536, depth=11, n_downsamples=7, "
          f"lr={args.lr}, loss={args.loss}, pretrained=False)")
    model = EnformerModel(
        lr=args.lr, loss=args.loss, pretrained=False,
        dim=1536, depth=11, n_downsamples=7,
    )

    print(f"[train] {args.max_epochs} epochs, batch={args.batch_size}")
    if args.resume_from:
        torch.set_float32_matmul_precision("medium")
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu",
            devices=[args.device],
            logger=CSVLogger(str(save_dir)),
            callbacks=[ModelCheckpoint(monitor="val_loss", mode="min",
                                       save_last=True)],
        )
        train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
        print(f"[resume] from {args.resume_from}")
        trainer.fit(model=model, train_dataloaders=train_dl,
                    val_dataloaders=val_dl, ckpt_path=args.resume_from)
    else:
        trainer = model.train_on_dataset(
            train_dataset=train_ds, val_dataset=val_ds,
            device=args.device, batch_size=args.batch_size, num_workers=4,
            save_dir=str(save_dir), max_epochs=args.max_epochs,
        )

    best_path = trainer.checkpoint_callback.best_model_path
    best_score = trainer.checkpoint_callback.best_model_score
    print(f"[done] best val_loss={best_score:.4f}  path={best_path}")

    final_path = save_dir / "best.ckpt"
    shutil.copy2(best_path, final_path)
    print(f"[done] copied to {final_path}")


if __name__ == "__main__":
    main()
