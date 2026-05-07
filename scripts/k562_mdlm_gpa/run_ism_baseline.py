#!/usr/bin/env python3
"""Greedy ISM (in-silico mutagenesis) baseline against K562 LegNet oracle.

For each input seed, runs 115 steps of greedy single-nt hill-climbing:
  at each step, evaluate all 200 positions x 3 alternative bases (batched),
  apply the single flip that maximises LegNet, repeat.

Logs per-step wall time, AG checkpoint scores at generations {12, 23, 57, 115}
(5%, 10%, 25%, 50% edits matching the reference ISM dataset), and writes a CSV
compatible with create_ism_comparison_slides_with_ln.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "${GPA_REPO_ROOT}")
from scripts.k562_mdlm_gpa.k562_oracle import K562Oracle, load_k562_oracle

SEQ_LEN = 200
BASES = "ACGT"  # bio encoding A=0 C=1 G=2 T=3
GEN_CHECKPOINTS = (12, 23, 57, 115)  # match shared ISM CSV

ORACLE_K562 = (
    "${GPA_REPO_ROOT}/model_zoo/lentimpra/oracle_models/"
    "best_model-epoch=24-val_pearson=0.814.ckpt"
)


def indices_to_seq(idx: np.ndarray) -> str:
    return "".join(BASES[int(i)] for i in idx)


@torch.no_grad()
def score_batch(oracle: K562Oracle, seqs_idx: torch.Tensor,
                batch_size: int = 1024) -> np.ndarray:
    """Score (N,200) int tensor with the LegNet K562 oracle."""
    preds, _ = oracle.score(seqs_idx, batch_size=batch_size)
    return np.asarray(preds, dtype=np.float64)


def ism_trajectory(seed_idx: np.ndarray, oracle: K562Oracle,
                   total_steps: int = 115, device: str = "cuda",
                   batch_size: int = 1024):
    """Run greedy ISM for total_steps on a single seed."""
    L = seed_idx.shape[0]
    current = torch.from_numpy(seed_idx.astype(np.int64)).clone()
    cur_score = float(score_batch(oracle,
                                   current.unsqueeze(0).to(device),
                                   batch_size=batch_size)[0])

    history = []
    # gen=0 snapshot
    history.append(dict(ism_generation=0, oracle_pred=cur_score,
                        sequence=indices_to_seq(current.cpu().numpy()),
                        step_elapsed_s=0.0, cum_elapsed_s=0.0,
                        edit_position=-1, edit_from=-1, edit_to=-1))
    cum_t = 0.0
    t0_all = time.perf_counter()

    # precompute flat base grid (3*L candidate flips)
    for step in range(1, total_steps + 1):
        t0 = time.perf_counter()

        # Build all L*3 candidate single-base substitutions.
        cand = current.unsqueeze(0).expand(L * 3, -1).clone().contiguous()
        alt_bases = torch.zeros(L * 3, dtype=torch.long)
        positions = torch.zeros(L * 3, dtype=torch.long)
        row = 0
        cur_np = current.numpy()
        for pos in range(L):
            for b in range(4):
                if b == cur_np[pos]:
                    continue
                cand[row, pos] = b
                positions[row] = pos
                alt_bases[row] = b
                row += 1

        scores = score_batch(oracle, cand.to(device), batch_size=batch_size)
        best = int(np.argmax(scores))
        gain = float(scores[best]) - cur_score
        if gain <= 0:
            # local maximum — still apply the best flip so edit-count grows monotonically
            pass
        pos = int(positions[best])
        b_from = int(cur_np[pos])
        b_to = int(alt_bases[best])
        current[pos] = b_to
        cur_score = float(scores[best])
        dt = time.perf_counter() - t0
        cum_t += dt
        history.append(dict(ism_generation=step, oracle_pred=cur_score,
                            sequence=indices_to_seq(current.cpu().numpy()),
                            step_elapsed_s=dt, cum_elapsed_s=cum_t,
                            edit_position=pos, edit_from=b_from, edit_to=b_to))

    total_wall = time.perf_counter() - t0_all
    return history, total_wall


def score_ag(seq_indices: np.ndarray, socket_path: str):
    """Post-hoc AG K562 scoring via server-client (optional)."""
    from scripts.k562_mdlm_gpa.ag_oracle_client import AlphaGenomeK562Client
    client = AlphaGenomeK562Client(socket_path)
    try:
        scores, _ = client.score(torch.from_numpy(seq_indices.astype(np.int64)))
    finally:
        client.close()
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_h5", required=True,
                   help="HDF5 with /indices[0] = 200-int seed array")
    p.add_argument("--uid", type=int, required=True)
    p.add_argument("--total_steps", type=int, default=115)
    p.add_argument("--oracle_ckpt", default=ORACLE_K562)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--ag_socket", default=None,
                   help="Unix-socket path to AG server; if omitted AG left blank.")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ISM uid={args.uid}] device={device} steps={args.total_steps}",
          flush=True)
    print(f"[ISM uid={args.uid}] loading oracle {args.oracle_ckpt}", flush=True)
    lit = load_k562_oracle(args.oracle_ckpt, device=device)
    oracle = K562Oracle(lit, device=device)

    with h5py.File(args.seed_h5, "r") as hf:
        seed_idx = hf["indices"][0].astype(np.int64)
    print(f"[ISM uid={args.uid}] seed: {indices_to_seq(seed_idx)[:30]}...",
          flush=True)

    history, total_wall = ism_trajectory(
        seed_idx, oracle, total_steps=args.total_steps,
        device=device, batch_size=args.batch_size,
    )
    print(f"[ISM uid={args.uid}] wall-clock {total_wall:.2f}s  "
          f"({total_wall / args.total_steps:.3f}s/step)", flush=True)

    df = pd.DataFrame(history)
    df["uid"] = args.uid
    df["ism_seed_type"] = "natural" if args.uid == 181692 else "random"

    # Optional AG scoring over all generations
    if args.ag_socket is not None and os.path.exists(args.ag_socket):
        print(f"[ISM uid={args.uid}] AG-scoring {len(df)} sequences via "
              f"{args.ag_socket}", flush=True)
        t0 = time.perf_counter()
        arr = np.stack([np.array([BASES.index(c) for c in s], dtype=np.int64)
                        for s in df["sequence"].values])
        ag_scores = score_ag(arr, args.ag_socket)
        df["y_pred_alphagenome"] = ag_scores
        print(f"[ISM uid={args.uid}] AG scoring done in {time.perf_counter()-t0:.1f}s",
              flush=True)
    else:
        df["y_pred_alphagenome"] = np.nan

    traj_csv = out_dir / f"ism_trajectory_uid{args.uid}.csv"
    df.to_csv(traj_csv, index=False)
    summary = {
        "uid": args.uid,
        "total_steps": args.total_steps,
        "total_wall_s": total_wall,
        "per_step_s_mean": float(df["step_elapsed_s"][1:].mean()),
        "per_step_s_median": float(df["step_elapsed_s"][1:].median()),
        "final_legnet": float(df["oracle_pred"].iloc[-1]),
    }
    for g in GEN_CHECKPOINTS:
        row = df[df["ism_generation"] == g]
        if len(row):
            summary[f"legnet_gen{g}"] = float(row["oracle_pred"].iloc[0])
            ag = row["y_pred_alphagenome"].iloc[0]
            if pd.notna(ag):
                summary[f"ag_gen{g}"] = float(ag)
    with open(out_dir / f"ism_summary_uid{args.uid}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[ISM uid={args.uid}] wrote {traj_csv}", flush=True)


if __name__ == "__main__":
    main()
