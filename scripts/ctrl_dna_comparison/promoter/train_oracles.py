"""Train one regLM EnformerModel per cell on Reddy finetuning_data.csv.

Produces checkpoints named `human_paired_<cell_tag>.ckpt` to match Ctrl-DNA's
`base_optimizer.load_target_model` path convention:
    JURKAT -> human_paired_jurkat.ckpt
    K562   -> human_paired_k562.ckpt
    THP1   -> human_paired_THP1.ckpt
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

CTRLDNA_SRC = "${HOME}/Ctrl-DNA/ctrl_dna"
if CTRLDNA_SRC not in sys.path:
    sys.path.insert(0, CTRLDNA_SRC)

from src.reglm.regression import EnformerModel, SeqDataset  # noqa: E402

CELL_TAG = {"JURKAT": "jurkat", "K562": "k562", "THP1": "THP1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELL_TAG))
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument(
        "--pretrained",
        action="store_true",
        help="Init Enformer trunk from EleutherAI pretrained weights.",
    )
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_csv)
    need_cols = {"sequence", args.cell, "is_train", "is_val"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    train_df = df.loc[df["is_train"], ["sequence", args.cell]].reset_index(drop=True)
    val_df = df.loc[df["is_val"], ["sequence", args.cell]].reset_index(drop=True)
    print(f"[{args.cell}] train={len(train_df)} val={len(val_df)}", flush=True)

    train_ds = SeqDataset(train_df, seq_len=args.seq_len)
    val_ds = SeqDataset(val_df, seq_len=args.seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = EnformerModel(lr=args.lr, loss="mse", pretrained=args.pretrained)
    # regLM's EnformerModel implements the removed-in-Lightning-2.0
    # `validation_epoch_end` hook (just prints val_loss). Strip it so the
    # trainer doesn't reject the module.
    if hasattr(type(model), "validation_epoch_end"):
        delattr(type(model), "validation_epoch_end")

    logger = CSVLogger(save_dir=str(out_dir), name=f"{args.cell}_log")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / f"{args.cell}_ckpts"),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[args.device],
        logger=logger,
        callbacks=[ckpt_cb],
        precision="16-mixed",
        gradient_clip_val=1.0,
    )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_src = ckpt_cb.best_model_path
    if not best_src or not os.path.exists(best_src):
        raise RuntimeError(f"[{args.cell}] no best checkpoint written")

    dest = out_dir / f"human_paired_{CELL_TAG[args.cell]}.ckpt"
    shutil.copyfile(best_src, dest)
    print(
        f"[{args.cell}] best val_loss={ckpt_cb.best_model_score.item():.4f} "
        f"-> {dest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
