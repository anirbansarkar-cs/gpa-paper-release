#!/usr/bin/env python3
"""
Fine-tune HyenaDNA on Gosai dataset as conditional LM.

Uses Ctrl-DNA's CharDataset for tokenization but handles training directly
with PyTorch (avoids Lightning version incompatibilities).

Checkpoint is shared by both GPA and Ctrl-DNA methods.

Usage:
    python scripts/ctrl_dna_comparison/train_hyenadna.py \
        --labeled_csv scripts/ctrl_dna_comparison/data/gosai_labeled.csv \
        --output_dir scripts/ctrl_dna_comparison/checkpoints/hyenadna_gosai \
        --batch_size 128 --max_epochs 3 --lr 1e-4
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add Ctrl-DNA source to path
CTRL_DNA_DIR = os.path.join(
    os.environ.get("CTRL_DNA_HOME", os.path.expanduser("~/Ctrl-DNA")), "ctrl_dna")
if CTRL_DNA_DIR not in sys.path:
    sys.path.insert(0, CTRL_DNA_DIR)

from src.reglm.dataset import CharDataset
from transformers import AutoModelForCausalLM


class HyenaDNATrainer:
    """Pure-PyTorch trainer for HyenaDNA conditional LM (no Lightning)."""

    def __init__(self, label_len, lr=1e-4, device="cuda"):
        self.device = torch.device(device)
        self.label_len = label_len
        self.lr = lr

        # Load pretrained HyenaDNA
        print("  Loading pretrained HyenaDNA-medium-160k...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "LongSafari/hyenadna-medium-160k-seqlen-hf",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {n_params / 1e6:.2f}M")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        # ignore_index=0 skips padding, ignore END token (1) via target masking
        self.loss_fn = lambda logits, y: F.cross_entropy(
            logits.float(), y, ignore_index=0)

    def forward(self, x):
        """Forward pass with ACGT masking (matches LightningModel.forward)."""
        logits = self.model(x).logits.swapaxes(1, 2)  # (B, 16, L)
        # Drop label positions
        logits = logits[:, :, self.label_len:]  # (B, 16, seq+end+trailing)
        # Mask to valid bases only (A=7, C=8, G=9, T=10)
        mask = torch.full_like(logits, fill_value=-1e8)
        mask[:, [7, 8, 9, 10]] = logits[:, [7, 8, 9, 10]]
        return mask

    def _mask_end_token(self, y):
        """Replace END token (1) with 0 (ignore_index) so it doesn't
        contribute to loss — the ACGT-only logit mask makes END unpredictable."""
        y = y.clone()
        y[y == 1] = 0
        return y

    def train_epoch(self, dataloader, epoch, log_interval=100):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for i, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = self._mask_end_token(y.to(self.device))

            logits = self.forward(x)  # (B, 16, seq+end+trailing)
            loss = self.loss_fn(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (i + 1) % log_interval == 0:
                avg = total_loss / n_batches
                elapsed = time.time() - t0
                print(f"  Epoch {epoch} step {i+1}/{len(dataloader)}: "
                      f"loss={avg:.4f}, {elapsed:.0f}s")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        for x, y in dataloader:
            x = x.to(self.device)
            y = self._mask_end_token(y.to(self.device))

            logits = self.forward(x)
            loss = self.loss_fn(logits, y)
            total_loss += loss.item()
            n_batches += 1

            # Accuracy (ignoring padding=0 and masked END)
            preds = logits.argmax(dim=1)  # (B, L)
            mask = y != 0
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def save_checkpoint(self, path, epoch, val_loss, val_acc):
        """Save in Lightning-compatible format for load_from_checkpoint."""
        torch.save({
            "state_dict": {f"model.{k}": v for k, v in self.model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hyper_parameters": {
                "lr": self.lr,
                "label_len": self.label_len,
            },
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, path)

    def load_checkpoint(self, path):
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        cleaned = {k.replace("model.", "", 1): v for k, v in state_dict.items()
                    if k.startswith("model.")}
        self.model.load_state_dict(cleaned)
        self.model.to(self.device)
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune HyenaDNA on Gosai")
    parser.add_argument("--labeled_csv", required=True,
                        help="Path to gosai_labeled.csv (from prepare_data.py)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--resume_from", default=None,
                        help="Resume training from this checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_float32_matmul_precision("medium")

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    print(f"Loading labeled data: {args.labeled_csv}")
    df = pd.read_csv(args.labeled_csv, dtype={"label": str})
    print(f"  {len(df):,} sequences")

    seqs = df["sequence"].tolist()
    labels = df["label"].tolist()
    label_len = len(labels[0])
    print(f"  Label length: {label_len} (e.g., '{labels[0]}')")

    # Train/val split
    n = len(seqs)
    n_val = int(n * args.val_fraction)
    indices = np.random.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_seqs = [seqs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_seqs = [seqs[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print(f"  Train: {len(train_seqs):,}, Val: {len(val_seqs):,}")

    # Create datasets and dataloaders
    seq_len = len(seqs[0])  # 200
    train_dataset = CharDataset(train_seqs, train_labels, seq_len=seq_len)
    val_dataset = CharDataset(val_seqs, val_labels, seq_len=seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # Initialize model
    print("\nInitializing HyenaDNA...")
    trainer = HyenaDNATrainer(label_len=label_len, lr=args.lr)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        print(f"\nResuming from: {args.resume_from}")
        start_epoch, prev_val_loss = trainer.load_checkpoint(args.resume_from)
        print(f"  Resumed at epoch {start_epoch}, val_loss={prev_val_loss:.4f}")

    # Initial validation
    val_loss, val_acc = trainer.validate(val_loader)
    print(f"\nPre-training: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    best_val_loss = val_loss
    best_ckpt_path = None

    # Training loop
    print(f"\nTraining: epochs {start_epoch+1}-{args.max_epochs}, "
          f"batch_size={args.batch_size}, lr={args.lr}")
    for epoch in range(start_epoch + 1, args.max_epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(train_loader, epoch, args.log_interval)
        val_loss, val_acc = trainer.validate(val_loader)
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch}/{args.max_epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_acc={val_acc:.4f}, time={elapsed:.0f}s")

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"epoch{epoch}.ckpt")
        trainer.save_checkpoint(ckpt_path, epoch, val_loss, val_acc)
        print(f"  Saved: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = ckpt_path
            print(f"  New best! val_loss={val_loss:.4f}")

    # Symlink best checkpoint
    if best_ckpt_path is None:
        best_ckpt_path = ckpt_path  # Use last if none improved

    best_link = os.path.join(args.output_dir, "best.ckpt")
    if os.path.exists(best_link) or os.path.islink(best_link):
        os.remove(best_link)
    os.symlink(os.path.abspath(best_ckpt_path), best_link)
    print(f"\nBest checkpoint: {best_link} -> {best_ckpt_path}")
    print(f"  val_loss={best_val_loss:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
