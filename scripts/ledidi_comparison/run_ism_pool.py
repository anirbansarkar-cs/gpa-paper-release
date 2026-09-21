#!/usr/bin/env python3
"""Batched greedy ISM trajectory runner over a 5K-seed pool, K562 LegNet oracle.

This is a pool-aware, batched-across-seeds replacement for the per-seed
``scripts/k562_mdlm_gpa/run_ism_baseline.py``. For each of the N seeds in the
input pool, runs ``--total_steps`` generations of greedy single-base
mutagenesis: at each generation, evaluate all 200x3 single-base substitutions
per seed in one batched LegNet forward pass, apply the best flip per seed.

Per-seed candidate batches (size 600) are concatenated across
``--seeds_per_chunk`` seeds to amortize launch overhead while keeping memory
bounded.

Output: ``trajectories.csv`` with one row per (seed_id, ism_generation):
  seed_id, ism_generation, edit_distance, oracle_legnet, sequence,
  step_elapsed_s, cum_elapsed_s, edit_position, edit_from, edit_to
plus a ``trajectories_meta.json`` summarising total wall and per-step pace.
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

sys.path.insert(0, os.path.expandvars("${GPA_REPO_ROOT}"))
from scripts.k562_mdlm_gpa.k562_oracle import K562Oracle, load_k562_oracle

SEQ_LEN = 200
BASES = "ACGT"
DEFAULT_ORACLE = os.path.expandvars(
    "${GPA_REPO_ROOT}/model_zoo/lentimpra/oracle_models/"
    "best_model-epoch=24-val_pearson=0.814.ckpt"
)


def indices_to_seq(idx: np.ndarray) -> str:
    return "".join(BASES[int(i)] for i in idx)


def build_candidate_tensor(seeds_chunk: torch.Tensor) -> tuple[torch.Tensor,
                                                               torch.Tensor,
                                                               torch.Tensor]:
    """For each of M seeds (each length L), build all 3*L single-base
    substitutions (skipping the identity base at each position).

    Returns:
        cand: (M*3L, L) int64
        positions: (M*3L,) int64 — column flipped
        alt_bases: (M*3L,) int64 — new base
    """
    M, L = seeds_chunk.shape
    n_per_seed = 3 * L
    # Each seed contributes 3L candidates: at every position p, 3 alternate bases.
    # cand[m, k] = seed m with k-th alteration applied.
    cand = seeds_chunk.unsqueeze(1).expand(M, n_per_seed, L).clone().contiguous()

    # Position k -> (pos, j) where j ∈ {0,1,2} indexes the alternate base
    pos_grid = torch.arange(L).repeat_interleave(3)        # (3L,)
    alt_grid = torch.arange(4).expand(L, 4)                # (L, 4) bases 0..3
    # We want, per position p, the 3 bases NOT equal to seeds_chunk[m, p].
    # Build (M, L, 3) alt_bases by masking out the current base.
    bases = torch.arange(4).view(1, 1, 4).expand(M, L, 4)  # (M, L, 4)
    cur = seeds_chunk.unsqueeze(-1)                        # (M, L, 1)
    not_cur = bases != cur                                 # (M, L, 4) bool
    # For each (m,p), the 3 surviving bases. argsort puts True first; take first 3.
    alt_per_pos = bases.masked_select(not_cur).view(M, L, 3)  # (M, L, 3)
    alt_flat = alt_per_pos.reshape(M, n_per_seed)             # (M, 3L)

    # Apply substitutions: for each k in 0..3L-1, set cand[m, k, pos_grid[k]] = alt_flat[m, k]
    m_idx = torch.arange(M).view(M, 1).expand(M, n_per_seed)
    k_idx = torch.arange(n_per_seed).view(1, n_per_seed).expand(M, n_per_seed)
    cand[m_idx, k_idx, pos_grid.view(1, n_per_seed).expand(M, n_per_seed)] = alt_flat

    cand_flat = cand.view(M * n_per_seed, L)
    pos_full = pos_grid.repeat(M)              # (M*3L,)
    alt_full = alt_flat.reshape(M * n_per_seed)
    return cand_flat, pos_full, alt_full


@torch.no_grad()
def score_chunk(oracle: K562Oracle, cand: torch.Tensor, device: str,
                inner_batch: int) -> np.ndarray:
    preds, _ = oracle.score(cand.to(device), batch_size=inner_batch)
    return np.asarray(preds, dtype=np.float64)


def ism_pool_trajectory(seed_pool: np.ndarray, oracle: K562Oracle,
                        total_steps: int, device: str,
                        seeds_per_chunk: int, inner_batch: int,
                        log_every: int = 5):
    """Run greedy ISM on every seed in seed_pool for ``total_steps`` gens.

    seed_pool: (N, L) int64 in diffusion encoding.
    Returns: list[dict] history records, plus total wall-clock seconds.
    """
    N, L = seed_pool.shape
    n_per_seed = 3 * L
    current = torch.from_numpy(seed_pool.astype(np.int64))    # (N, L) on CPU
    seeds_orig = current.clone()

    # Score gen-0 once for all seeds
    print(f"[gen 0/{total_steps}] scoring {N} seeds at gen 0", flush=True)
    cur_score = np.empty(N, dtype=np.float64)
    for s0 in range(0, N, seeds_per_chunk * 4):  # gen-0 needs no expansion; bigger chunks ok
        s1 = min(s0 + seeds_per_chunk * 4, N)
        cur_score[s0:s1] = score_chunk(oracle, current[s0:s1], device, inner_batch)

    history = []
    for s in range(N):
        history.append(dict(
            seed_id=s, ism_generation=0, edit_distance=0,
            oracle_legnet=float(cur_score[s]),
            sequence=indices_to_seq(current[s].numpy()),
            step_elapsed_s=0.0, cum_elapsed_s=0.0,
            edit_position=-1, edit_from=-1, edit_to=-1,
        ))

    cum_t = 0.0
    t0_all = time.perf_counter()

    for gen in range(1, total_steps + 1):
        t0 = time.perf_counter()
        new_score = np.empty(N, dtype=np.float64)
        new_pos   = np.empty(N, dtype=np.int64)
        new_from  = np.empty(N, dtype=np.int64)
        new_to    = np.empty(N, dtype=np.int64)

        for s0 in range(0, N, seeds_per_chunk):
            s1 = min(s0 + seeds_per_chunk, N)
            chunk = current[s0:s1]                  # (M, L)
            cand, pos_full, alt_full = build_candidate_tensor(chunk)
            scores = score_chunk(oracle, cand, device, inner_batch)
            # Reshape (M*3L,) -> (M, 3L)
            scores_2d = scores.reshape(s1 - s0, n_per_seed)
            best_k = scores_2d.argmax(axis=1)        # (M,)
            row_idx = np.arange(s1 - s0)
            best_score = scores_2d[row_idx, best_k]
            # pos/alt for each chosen substitution
            pos_per_seed = pos_full.view(s1 - s0, n_per_seed)[row_idx, best_k].numpy()
            alt_per_seed = alt_full.view(s1 - s0, n_per_seed)[row_idx, best_k].numpy()

            from_per_seed = chunk.numpy()[row_idx, pos_per_seed]
            new_score[s0:s1] = best_score
            new_pos[s0:s1]   = pos_per_seed
            new_from[s0:s1]  = from_per_seed
            new_to[s0:s1]    = alt_per_seed
            # Apply substitution into ``current``
            current[s0 + row_idx, pos_per_seed] = torch.from_numpy(alt_per_seed)

        dt = time.perf_counter() - t0
        cum_t += dt
        edits = (current != seeds_orig).sum(dim=1).numpy()
        for s in range(N):
            history.append(dict(
                seed_id=s, ism_generation=gen,
                edit_distance=int(edits[s]),
                oracle_legnet=float(new_score[s]),
                sequence=indices_to_seq(current[s].numpy()),
                step_elapsed_s=(dt / N),  # amortized
                cum_elapsed_s=cum_t,
                edit_position=int(new_pos[s]),
                edit_from=int(new_from[s]),
                edit_to=int(new_to[s]),
            ))
        cur_score = new_score
        if gen % log_every == 0 or gen == 1:
            print(f"[gen {gen}/{total_steps}] {dt:.1f}s | "
                  f"mean LN {new_score.mean():.3f} | "
                  f"max LN {new_score.max():.3f} | "
                  f"mean edits {edits.mean():.1f}", flush=True)

    total_wall = time.perf_counter() - t0_all
    return history, total_wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True,
                    help="Seed pool h5 with /indices (N, 200) int64")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to write trajectories.csv + meta.json")
    ap.add_argument("--total_steps", type=int, default=115)
    ap.add_argument("--oracle_ckpt", default=DEFAULT_ORACLE)
    ap.add_argument("--oracle_type", choices=["legnet", "alphagenome_torch"],
                    default="legnet",
                    help="legnet = LegNet K562 (default); alphagenome_torch = "
                         "NEW torch stage2 AG (needs gpa_torchag env)")
    ap.add_argument("--seeds_per_chunk", type=int, default=128,
                    help="Number of seeds processed per inner GPU chunk; "
                         "each chunk evaluates seeds_per_chunk * 600 candidates.")
    ap.add_argument("--inner_batch", type=int, default=2048,
                    help="LegNet forward-pass batch size inside score().")
    ap.add_argument("--slice", default=None,
                    help="Optional 'a:b' to subset seeds for smoke testing.")
    ap.add_argument("--log_every", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[ISM-pool] device={device}  total_steps={args.total_steps}  "
          f"seeds_per_chunk={args.seeds_per_chunk}  inner_batch={args.inner_batch}",
          flush=True)
    if args.oracle_type == "alphagenome_torch":
        from scripts.k562_mdlm_gpa.k562_oracle import TorchAGOracle
        print("[ISM-pool] loading NEW torch stage2 AG oracle (alphagenome_torch)", flush=True)
        oracle = TorchAGOracle(device=device, cell="k562")
    else:
        print(f"[ISM-pool] loading oracle {args.oracle_ckpt}", flush=True)
        lit = load_k562_oracle(args.oracle_ckpt, device=device)
        oracle = K562Oracle(lit, device=device)

    with h5py.File(args.pool, "r") as f:
        seed_pool = f["indices"][:].astype(np.int64)
    if args.slice:
        a, b = (int(x) for x in args.slice.split(":"))
        seed_pool = seed_pool[a:b]
    print(f"[ISM-pool] loaded {len(seed_pool)} seeds (length {seed_pool.shape[1]}) "
          f"from {args.pool}", flush=True)

    history, total_wall = ism_pool_trajectory(
        seed_pool, oracle, total_steps=args.total_steps, device=device,
        seeds_per_chunk=args.seeds_per_chunk, inner_batch=args.inner_batch,
        log_every=args.log_every,
    )
    df = pd.DataFrame(history)
    csv_path = out_dir / "trajectories.csv"
    df.to_csv(csv_path, index=False)

    meta = {
        "pool": args.pool,
        "n_seeds": int(len(seed_pool)),
        "total_steps": int(args.total_steps),
        "seeds_per_chunk": int(args.seeds_per_chunk),
        "inner_batch": int(args.inner_batch),
        "total_wall_s": float(total_wall),
        "wall_per_seed_per_gen_s": float(total_wall / (len(seed_pool) * args.total_steps)),
        "wall_per_seed_total_s": float(total_wall / len(seed_pool)),
    }
    final_gen = df[df["ism_generation"] == args.total_steps]
    if len(final_gen):
        meta["final_legnet_mean"] = float(final_gen["oracle_legnet"].mean())
        meta["final_legnet_max"]  = float(final_gen["oracle_legnet"].max())
        meta["final_edit_distance_mean"] = float(final_gen["edit_distance"].mean())
    with open(out_dir / "trajectories_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[ISM-pool] wrote {csv_path}  ({total_wall:.1f}s wall)", flush=True)


if __name__ == "__main__":
    main()
