#!/usr/bin/env python3
"""LEDIDI run matched to the ISM slide seeds (uid 50178 + uid 181692).

For each seed, runs `--n_copies` independent LEDIDI optimizations (same seed,
different Gumbel noise) over a grid of L1-regularization `l` values and
target values. Writes one CSV per (uid, l, target) triple.

Fork of `run_ledidi_baseline.py` — the LedidiOracleWrapper and oracle loading
stay identical; this version:
  - accepts an explicit `--seed_h5` (the ISM single-seed HDF5 already on disk)
  - records `uid` in the output
  - iterates the `(l, target)` grid in one process so we only pay LEDIDI /
    oracle load once
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "${GPA_REPO_ROOT}")

from scripts.ledidi_comparison.run_ledidi_baseline import (
    LedidiOracleWrapper,
    seq_to_onehot,
    onehot_to_seq,
    edit_distance,
)
from scripts.k562_mdlm_gpa.k562_oracle import load_k562_oracle

from ledidi import Ledidi

ORACLE_CKPT = ("${GPA_REPO_ROOT}/model_zoo/lentimpra/"
               "oracle_models/best_model-epoch=24-val_pearson=0.814.ckpt")


def load_seed_sequence(h5_path: str) -> str:
    from scripts.ledidi_comparison.run_ledidi_baseline import indices_to_seq
    with h5py.File(h5_path, "r") as f:
        indices = f["indices"][0].astype(np.int64)
    return indices_to_seq(indices)


def run_one_copy(oracle_wrapper, seed_seq: str, target_val: float, l_val: float,
                 tau: float, lr: float, max_iter: int, batch_size: int,
                 early_stop: int, device):
    seed_onehot = seq_to_onehot(seed_seq).to(device)
    target = torch.tensor([[target_val]], dtype=torch.float32).to(device)
    ledidi = Ledidi(
        model=oracle_wrapper,
        shape=(4, len(seed_seq)),
        target=None,
        tau=tau,
        l=l_val,
        batch_size=batch_size,
        max_iter=max_iter,
        early_stopping_iter=early_stop,
        lr=lr,
        verbose=False,
    )
    ledidi.to(device)

    t0 = time.perf_counter()
    edited_onehot = ledidi.fit_transform(seed_onehot, target)
    elapsed = time.perf_counter() - t0

    if isinstance(edited_onehot, torch.Tensor):
        batch = edited_onehot.detach().to(device)
    else:
        batch = torch.tensor(edited_onehot).to(device)

    with torch.no_grad():
        scores = oracle_wrapper(batch).squeeze(-1)
    best = int(scores.argmax().item())
    edited = batch[best]
    edited_seq = onehot_to_seq(edited)
    edited_score = float(scores[best].item())

    return dict(
        edited_sequence=edited_seq,
        edited_legnet=edited_score,
        edit_distance=edit_distance(seed_seq, edited_seq),
        elapsed_s=elapsed,
        all_batch_max=float(scores.max().item()),
        all_batch_mean=float(scores.mean().item()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_h5", required=True)
    p.add_argument("--uid", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_copies", type=int, default=50)
    p.add_argument("--l_grid", nargs="+", type=float,
                   default=[0.02, 0.05, 0.15, 0.4, 1.0])
    p.add_argument("--target_grid", nargs="+", type=float, default=[10.0, 15.0])
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--ledidi_batch_size", type=int, default=64)
    p.add_argument("--early_stop", type=int, default=100)
    p.add_argument("--oracle_ckpt", default=ORACLE_CKPT)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LEDIDI uid={args.uid}] device={device}  "
          f"l_grid={args.l_grid}  target_grid={args.target_grid}  "
          f"n_copies={args.n_copies}", flush=True)

    # Load oracle once
    print(f"[LEDIDI uid={args.uid}] loading LegNet K562", flush=True)
    lit = load_k562_oracle(args.oracle_ckpt, device=str(device))
    wrapper = LedidiOracleWrapper(lit).to(device)
    wrapper.eval()
    for param in wrapper.parameters():
        param.requires_grad = False

    seed_seq = load_seed_sequence(args.seed_h5)
    with torch.no_grad():
        seed_score = wrapper(seq_to_onehot(seed_seq).to(device)).squeeze().item()
    print(f"[LEDIDI uid={args.uid}] seed LegNet={seed_score:.3f}", flush=True)

    seed_type = "natural" if args.uid == 181692 else "random"

    wall_start = time.perf_counter()
    for target_val, l_val in itertools.product(args.target_grid, args.l_grid):
        cfg_tag = f"t{int(target_val)}_l{l_val:.3f}".rstrip("0").rstrip(".")
        rows = []
        cfg_t0 = time.perf_counter()
        print(f"[uid={args.uid}] === target={target_val} l={l_val} ===",
              flush=True)
        for copy_i in range(args.n_copies):
            try:
                r = run_one_copy(
                    wrapper, seed_seq, target_val, l_val,
                    tau=args.tau, lr=args.lr, max_iter=args.max_iter,
                    batch_size=args.ledidi_batch_size,
                    early_stop=args.early_stop, device=device,
                )
                r.update(
                    copy_idx=copy_i,
                    seed_sequence=seed_seq,
                    seed_legnet=seed_score,
                    uid=args.uid,
                    seed_type=seed_type,
                    target=target_val,
                    l=l_val,
                    status="ok",
                )
            except Exception as exc:  # noqa: BLE001
                r = dict(
                    copy_idx=copy_i, seed_sequence=seed_seq,
                    edited_sequence=seed_seq,
                    seed_legnet=seed_score, edited_legnet=seed_score,
                    edit_distance=0, elapsed_s=0.0,
                    uid=args.uid, seed_type=seed_type,
                    target=target_val, l=l_val, status=f"err:{type(exc).__name__}",
                )
            rows.append(r)
            if (copy_i + 1) % 10 == 0 or copy_i == 0:
                print(f"  [{copy_i+1}/{args.n_copies}] "
                      f"LN={r.get('edited_legnet', 'nan'):.2f} "
                      f"edits={r.get('edit_distance', -1)} "
                      f"t={r.get('elapsed_s', 0):.1f}s",
                      flush=True)
        df = pd.DataFrame(rows)
        out_csv = out_dir / f"ledidi_uid{args.uid}_{cfg_tag}.csv"
        df.to_csv(out_csv, index=False)
        ok = df[df["status"] == "ok"]
        print(f"  -> {out_csv}  n_ok={len(ok)}/{len(df)}  "
              f"mean_edits={ok['edit_distance'].mean():.1f}  "
              f"mean_LN={ok['edited_legnet'].mean():.2f}  "
              f"cfg_wall={time.perf_counter()-cfg_t0:.1f}s",
              flush=True)

    print(f"[LEDIDI uid={args.uid}] total wall "
          f"{time.perf_counter() - wall_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
