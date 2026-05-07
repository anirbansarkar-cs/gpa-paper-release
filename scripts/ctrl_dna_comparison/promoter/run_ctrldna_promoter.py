#!/usr/bin/env python3
"""
Ctrl-DNA pipeline for the Reddy 2024 promoter MPRA (250 bp, 3 cells).

Mirrors scripts/ctrl_dna_comparison/run_ctrldna.py but:
  - Uses 3 separate regLM EnformerModel oracles (PromoterCellAdapter)
    instead of one 3-task gReLU Enformer.
  - Seq length 250 bp (parameterised via --seq_len).
  - Decile label prompts (e.g. "900" for JURKAT-specific) instead of
    binary "100"/"010"/"001".
  - Seed CSVs carry only {sequence, <target_cell>}; missing cells are
    filled in by scoring with the oracle pool at startup.

Usage:
    python scripts/ctrl_dna_comparison/promoter/run_ctrldna_promoter.py \
        --hyenadna_checkpoint scripts/ctrl_dna_comparison/promoter/checkpoints/hyenadna_promoter_full/best.ckpt \
        --oracle_ckpt_dir   scripts/ctrl_dna_comparison/promoter/checkpoints \
        --oracle_ranges     scripts/ctrl_dna_comparison/promoter/data/oracle_ranges.json \
        --seed_csv          scripts/ctrl_dna_comparison/promoter/data/seeds_JURKAT.csv \
        --task JURKAT \
        --max_iter 100 --epoch 5 --batch_size 128 \
        --out_dir results/ctrl_dna_comparison/promoter/ctrldna_jurkat_seed0
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from oracle_adapter import (  # noqa: E402
    PROMOTER_CELLS, PromoterOraclePool, PromoterCellAdapter,
    load_promoter_oracle_ranges,
)

_PROJECT_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_CTRL_DNA_DIR = "${HOME}/Ctrl-DNA/ctrl_dna"
if _CTRL_DNA_DIR not in sys.path:
    sys.path.insert(0, _CTRL_DNA_DIR)

from src.reglm.lightning import LightningModel  # noqa: E402
from dna_optimizers_multi.experience import Experience  # noqa: E402
from dna_optimizers_multi.base_optimizer import evaluate  # noqa: E402

CELL_IDX = {c: i for i, c in enumerate(PROMOTER_CELLS)}
# Matches Ctrl-DNA/ctrl_dna/reinforce_multi_lagrange.py:get_prefix_label().
DEFAULT_TARGET_LABEL = {"JURKAT": "100", "K562": "010", "THP1": "001"}


def load_hyenadna_lightning(ckpt_path, label_len=3, device="cuda"):
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
    model = model.to(device)
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Ctrl-DNA on Reddy promoter MPRA")

    p.add_argument("--hyenadna_checkpoint", required=True)
    p.add_argument(
        "--oracle_ckpt_dir",
        default="scripts/ctrl_dna_comparison/promoter/checkpoints",
    )
    p.add_argument(
        "--oracle_ranges",
        default="scripts/ctrl_dna_comparison/promoter/data/oracle_ranges.json",
    )
    p.add_argument(
        "--finetuning_csv",
        default="scripts/ctrl_dna_comparison/promoter/data/finetuning_data.csv",
        help="Used to backfill all 3 per-cell activities on seeds that carry "
             "only the target cell column.",
    )

    p.add_argument("--task", required=True, choices=list(PROMOTER_CELLS))
    p.add_argument("--seed_csv", required=True)
    p.add_argument("--target_label", default=None,
                   help="Decile prompt; default '900'/'090'/'009' for the task.")

    p.add_argument("--seq_len", type=int, default=250)
    p.add_argument("--label_len", type=int, default=3)

    p.add_argument("--max_iter", type=int, default=100)
    p.add_argument("--epoch", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--lambda_lr", type=float, default=3e-1)
    p.add_argument("--lambda_value", nargs="+", type=float, default=[0.5, 0.5])
    p.add_argument("--constraint", nargs="+", type=float, default=[0.5, 0.5, -0.5])

    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_interval", type=int, default=50)
    p.add_argument("--resume_from", default=None)

    return p.parse_args()


class CtrlDNAPromoter:
    """Promoter-adapted Ctrl-DNA optimizer (PPO + Lagrangian)."""

    def __init__(self, args, agent, oracle_pool):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = int(args.seq_len)

        self.agent = agent.to(self.device)
        self.agent.train()

        self.prefix_label = args.target_label
        self.label = self.prefix_label
        assert len(self.prefix_label) == args.label_len, (
            f"target_label {self.prefix_label!r} must have length {args.label_len}"
        )

        # Frozen pretrained-base reference for KL penalty (mirrors enhancer)
        self.ref_model = LightningModel(label_len=args.label_len).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            (p for p in self.agent.parameters() if p.requires_grad),
            lr=args.lr,
        )

        self.pool = oracle_pool
        self.targets = {c: PromoterCellAdapter(oracle_pool, c) for c in PROMOTER_CELLS}

        self.ranges = load_promoter_oracle_ranges(args.oracle_ranges)

        self.task_idx = CELL_IDX[args.task]
        self.constraint_indices = [i for i in range(len(PROMOTER_CELLS)) if i != self.task_idx]

        self.beta = args.beta
        self.epsilon = args.epsilon
        self.constraint = list(args.constraint)

        self.lagrangian_multipliers = [
            torch.nn.Parameter(torch.tensor(args.lambda_value[0]), requires_grad=True),
            torch.nn.Parameter(torch.tensor(args.lambda_value[1]), requires_grad=True),
        ]
        self.lambda_optimizers = [
            torch.optim.Adam([self.lagrangian_multipliers[i]], lr=args.lambda_lr, eps=1e-4)
            for i in range(2)
        ]

        self.experience = Experience(max_size=100, priority=True)
        self.dna_buffer = {}
        self.total_oracle_calls = 0

    def _normalize(self, raw, cell):
        r = self.ranges[cell]
        return (raw - r["min"]) / (r["max"] - r["min"])

    @torch.no_grad()
    def score_enformer(self, dna):
        if dna in self.dna_buffer:
            return self.dna_buffer[dna][0]
        self.total_oracle_calls += 1
        scores = []
        for cell in PROMOTER_CELLS:
            raw = self.targets[cell]([dna]).item()
            scores.append(self._normalize(raw, cell))
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        reward = scores[self.task_idx] - 0.5 * (
            scores[self.constraint_indices[0]] + scores[self.constraint_indices[1]]
        )
        self.dna_buffer[dna] = [scores_tensor, reward, self.total_oracle_calls, 1]
        return scores_tensor

    def predict_enformer(self, dna_list):
        return [self.score_enformer(dna) for dna in dna_list]

    def update_lambda(self, avg_epcost):
        for i, epcost in enumerate(avg_epcost):
            loss = -self.lagrangian_multipliers[i] * (epcost - self.constraint[i])
            self.lambda_optimizers[i].zero_grad()
            loss.backward()
            self.lambda_optimizers[i].step()

    def _get_advantages(self, scores):
        mean_r = scores.view(-1, scores.shape[-1]).mean(dim=-1, keepdim=True)
        std_r = scores.view(-1, scores.shape[-1]).std(dim=-1, keepdim=True)
        return (scores - mean_r) / (std_r + 1e-4)

    def update(self, obs, old_logprobs, rewards, nonterms, episode_lens, cfg, iteration, epoch):
        self.agent.train()
        cost_batch = rewards[-1, self.constraint_indices, :].mean(-1).unsqueeze(-1)
        self.update_lambda(cost_batch)
        for i in range(2):
            self.lagrangian_multipliers[i].data.clamp_(min=0, max=1.0)

        scores = rewards[-1, self.task_idx]
        advantages = self._get_advantages(scores)
        adv_2 = self._get_advantages(rewards[-1, self.constraint_indices[0]])
        adv_3 = self._get_advantages(rewards[-1, self.constraint_indices[1]])

        total_lambda = self.lagrangian_multipliers[0] + self.lagrangian_multipliers[1]
        boost = max(1, 2 - total_lambda)
        advantages = (boost * advantages
                      - self.lagrangian_multipliers[0] * adv_2
                      - self.lagrangian_multipliers[1] * adv_3)

        logprobs = self.agent.sequences_log_probs(obs, nonterms)
        old_per_token_logps = old_logprobs.detach().to(logprobs.device)
        ratio = torch.exp(logprobs - old_per_token_logps)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

        per_token_loss = -torch.min(
            ratio * advantages.unsqueeze(0),
            clipped_ratio * advantages.unsqueeze(0),
        )

        if self.beta != 0.0:
            ref_logps = self.ref_model.sequences_log_probs(obs, nonterms)
            kl = torch.exp(ref_logps - logprobs) - (ref_logps - logprobs) - 1
            per_token_loss = per_token_loss + self.beta * kl

        loss = per_token_loss.sum() / nonterms[:-1].sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

    def _save_checkpoint(self, path, it, df, summary_df, oracle_calls_per_iter, t0):
        torch.save({
            "round": it,
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lagrangian_multipliers": [lm.data.clone() for lm in self.lagrangian_multipliers],
            "lambda_optimizer_states": [lo.state_dict() for lo in self.lambda_optimizers],
            "best_top": getattr(self, "_best_top", None),
            "best_round": getattr(self, "_best_round", None),
            "total_oracle_calls": self.total_oracle_calls,
            "dna_buffer": self.dna_buffer,
            "experience_memory": self.experience.memory,
            "df": df.to_dict("records"),
            "summary_df": summary_df.to_dict("records"),
            "oracle_calls_per_iter": oracle_calls_per_iter,
            "elapsed_before_resume": time.time() - t0,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
        }, path)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.agent.load_state_dict(ckpt["agent_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for i, lm_data in enumerate(ckpt["lagrangian_multipliers"]):
            self.lagrangian_multipliers[i].data.copy_(lm_data)
        for i, lo_state in enumerate(ckpt["lambda_optimizer_states"]):
            self.lambda_optimizers[i].load_state_dict(lo_state)
        self._best_top = ckpt["best_top"]
        self._best_round = ckpt["best_round"]
        self.total_oracle_calls = ckpt["total_oracle_calls"]
        self.dna_buffer = ckpt.get("dna_buffer", {})
        if "experience_memory" in ckpt:
            self.experience.memory = ckpt["experience_memory"]
        torch.set_rng_state(ckpt["torch_rng_state"])
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        df = pd.DataFrame(ckpt["df"])
        summary_df = pd.DataFrame(ckpt["summary_df"])
        oracle_calls_per_iter = ckpt.get("oracle_calls_per_iter", [])
        elapsed_offset = ckpt.get("elapsed_before_resume", 0)
        start_round = ckpt["round"] + 1
        print(f"  [resume] round={ckpt['round']} oracle_calls={self.total_oracle_calls} "
              f"experience={len(self.experience)} df_rows={len(df)}")
        return start_round, df, summary_df, oracle_calls_per_iter, elapsed_offset

    def _decode_population(self, obs):
        label_prefix = 1 + self.args.label_len
        dna_list = []
        for dna in obs.cpu().numpy().T:
            dna_seq = self.agent.decode(dna, ignore_num=label_prefix)[0]
            assert len(dna_seq) == self.seq_len, (
                f"Expected {self.seq_len}bp, got {len(dna_seq)}"
            )
            dna_list.append(dna_seq)
        return dna_list

    def optimize(self, starting_sequences):
        cfg = self.args
        os.makedirs(cfg.out_dir, exist_ok=True)

        if cfg.resume_from is not None:
            print(f"\n[Resume] Loading checkpoint: {cfg.resume_from}")
            start_round, df, summary_df, oracle_calls_per_iter, elapsed_offset = \
                self._load_checkpoint(cfg.resume_from)
            t0 = time.time() - elapsed_offset
        else:
            df = pd.DataFrame(columns=["round", "sequence", "true_score"])
            summary_df = pd.DataFrame(columns=["round", "top", "fitness", "diversity", "novelty"])

            init_seqs = starting_sequences["sequence"].tolist()
            init_tokens = self.agent.encode(
                init_seqs, [self.prefix_label] * len(init_seqs), add_start=True,
            ).T.to(self.device)

            num_rewards = len(PROMOTER_CELLS)
            init_rewards_list = starting_sequences["rewards"].tolist()
            init_rewards = torch.tensor(init_rewards_list).to(self.device)
            init_rewards = init_rewards.T.unsqueeze(0).repeat(
                cfg.batch_size + len(self.prefix_label), 1, 1,
            )[:init_tokens.shape[0], :, :]

            init_nonterms = (
                [False] * (len(self.prefix_label) + 1)
                + [True] * self.seq_len
            )[:init_tokens.shape[0]]
            init_nonterms = torch.tensor(init_nonterms, dtype=torch.bool)
            init_nonterms = init_nonterms.unsqueeze(-1).expand(
                -1, len(init_seqs)
            ).to(self.device)
            init_lens = torch.tensor(
                [self.seq_len] * len(init_seqs), dtype=torch.long
            )

            init_logprobs = self.agent.sequences_log_probs(
                init_tokens, init_nonterms
            ).detach().cpu()

            for i in range(cfg.epoch):
                self.update(init_tokens, init_logprobs, init_rewards,
                            init_nonterms, init_lens, cfg, 0, i)

            start_round = 1
            t0 = time.time()
            oracle_calls_per_iter = []

        def sigterm_handler(signum, frame):
            print(f"\n  [SIGTERM] Caught signal, saving checkpoint...")
            ckpt_path = f"{cfg.out_dir}/checkpoint_sigterm.pt"
            self._save_checkpoint(ckpt_path, _current_round, df, summary_df,
                                  oracle_calls_per_iter, t0)
            df.to_csv(f"{cfg.out_dir}/ctrldna_{cfg.task}_seed{cfg.seed}.csv", index=False)
            summary_df.to_csv(f"{cfg.out_dir}/ctrldna_{cfg.task}_seed{cfg.seed}_summary.csv",
                              index=False)
            print(f"  [SIGTERM] Saved {ckpt_path} at round {_current_round}")
            sys.exit(0)

        signal.signal(signal.SIGTERM, sigterm_handler)
        _current_round = start_round - 1

        num_rewards = len(PROMOTER_CELLS)
        print(f"\nStarting Ctrl-DNA (rounds {start_round}-{cfg.max_iter}, "
              f"seq_len={self.seq_len}, target_label={self.prefix_label!r})")

        for it in range(start_round, cfg.max_iter + 1):
            _current_round = it
            calls_before = self.total_oracle_calls

            with torch.no_grad():
                labels = [self.prefix_label] * cfg.batch_size
                obs, rewards, nonterms, episode_lens = self.agent.get_data(labels, self.seq_len)
                rewards = rewards.unsqueeze(-1).repeat(1, 1, num_rewards)
                old_logprobs = self.agent.sequences_log_probs(obs, nonterms).detach().cpu()

            dna_list = self._decode_population(obs)
            scores = np.array(self.predict_enformer(dna_list))
            scores_multi = torch.tensor(scores, dtype=torch.float32, device=self.device)

            combined = (
                scores_multi[:, self.task_idx]
                - (scores_multi[:, self.constraint_indices[0]] - self.constraint[0])
                + scores_multi[:, self.task_idx]
                - (scores_multi[:, self.constraint_indices[1]] - self.constraint[1])
            )
            combined_np = combined.detach().cpu().numpy()

            rewards[-1, :] += scores_multi
            rewards = rewards.transpose(-2, -1)

            if len(self.experience) > 24:
                e_obs, e_logprobs, e_scores, e_rewards, e_nonterms, e_lens = \
                    self.experience.sample(24, self.device)
                e_L, e_B = e_obs.shape
                L, B = obs.shape
                f_L = max(e_L, L)

                f_obs = torch.zeros((f_L, B + e_B), dtype=torch.long, device=self.device)
                f_nonterms = torch.zeros((f_L, B + e_B), dtype=torch.bool, device=self.device)
                f_obs[:L, :B] = obs
                f_obs[:e_L, B:] = e_obs
                f_nonterms[:L, :B] = nonterms
                f_nonterms[:e_L, B:] = e_nonterms
                f_rewards = torch.cat([rewards, e_rewards], dim=-1)
                f_lens = torch.cat([episode_lens, e_lens])
                f_logprobs = torch.cat([old_logprobs, e_logprobs], dim=-1)

                for i in range(cfg.epoch):
                    self.update(f_obs, f_logprobs, f_rewards, f_nonterms, f_lens, cfg, it, i)
            else:
                for i in range(cfg.epoch):
                    self.update(obs, old_logprobs, rewards, nonterms, episode_lens, cfg, it, i)

            self.experience.add_experience(
                dna_list, obs, old_logprobs, combined_np, rewards, nonterms, episode_lens
            )

            round_df = pd.DataFrame({
                "round": [it] * len(dna_list),
                "sequence": dna_list,
                "true_score": combined_np[:len(dna_list)],
            })
            for i in range(num_rewards):
                round_df[f"reward_{i+1}"] = np.array(scores_multi[:len(dna_list), i].cpu())
            df = pd.concat([df, round_df], ignore_index=True)

            round_results = evaluate(round_df, starting_sequences)
            round_results["round"] = it
            summary_df = pd.concat(
                [summary_df, pd.DataFrame([round_results])], ignore_index=True
            )

            calls_this_iter = self.total_oracle_calls - calls_before
            oracle_calls_per_iter.append(calls_this_iter)

            if it % 10 == 0:
                df.to_csv(f"{cfg.out_dir}/ctrldna_{cfg.task}_seed{cfg.seed}_partial.csv",
                          index=False)
                summary_df.to_csv(
                    f"{cfg.out_dir}/ctrldna_{cfg.task}_seed{cfg.seed}_summary_partial.csv",
                    index=False,
                )

            if it % 10 == 0 or it == cfg.max_iter:
                elapsed = time.time() - t0
                print(f"  Round {it}/{cfg.max_iter}: "
                      f"top={round_results['top']:.3f} "
                      f"fitness={round_results['fitness']:.3f} "
                      f"diversity={round_results['diversity']:.0f} "
                      f"novelty={round_results['novelty']:.0f} "
                      f"oracle_calls={self.total_oracle_calls} "
                      f"elapsed={elapsed:.0f}s")

            if cfg.checkpoint_interval > 0 and it % cfg.checkpoint_interval == 0:
                ckpt_path = f"{cfg.out_dir}/checkpoint_round{it:04d}.pt"
                self._save_checkpoint(ckpt_path, it, df, summary_df,
                                      oracle_calls_per_iter, t0)
                print(f"  [ckpt] Saved: {ckpt_path}")

            round_top = round_results["top"]
            if not hasattr(self, "_best_top") or round_top > self._best_top:
                self._best_top = round_top
                self._best_round = it
                torch.save(self.agent.state_dict(), f"{cfg.out_dir}/agent_best.pt")

        save_name = f"{cfg.out_dir}/ctrldna_{cfg.task}_seed{cfg.seed}.csv"
        summary_name = f"{cfg.out_dir}/ctrldna_{cfg.task}_seed{cfg.seed}_summary.csv"
        df.to_csv(save_name, index=False)
        summary_df.to_csv(summary_name, index=False)

        meta = {
            "total_oracle_calls": self.total_oracle_calls,
            "unique_sequences": len(self.dna_buffer),
            "oracle_calls_per_iter": oracle_calls_per_iter,
            "elapsed_seconds": time.time() - t0,
            "best_round": getattr(self, "_best_round", cfg.max_iter),
            "best_top": getattr(self, "_best_top", None),
            "target_label": self.prefix_label,
            "task": cfg.task,
            "seq_len": self.seq_len,
        }
        with open(f"{cfg.out_dir}/ctrldna_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nDone. Results: {save_name}")
        print(f"  Total oracle calls: {self.total_oracle_calls}")
        print(f"  Unique sequences: {len(self.dna_buffer)}")
        return df, summary_df


def prepare_seeds(args, pool):
    """Load seeds_{CELL}.csv and fill in all 3 cells' normalized activities
    either from finetuning_data.csv (preferred: matches training data exactly)
    or by scoring with the oracle pool as a fallback."""
    seeds = pd.read_csv(args.seed_csv)
    assert "sequence" in seeds.columns, f"{args.seed_csv} missing 'sequence' column"

    missing = [c for c in PROMOTER_CELLS if c not in seeds.columns]
    if missing:
        ft = pd.read_csv(args.finetuning_csv)[["sequence"] + list(PROMOTER_CELLS)]
        merged = seeds[["sequence"]].merge(ft, on="sequence", how="left")
        if merged[list(PROMOTER_CELLS)].isna().any().any():
            n_missing = merged[list(PROMOTER_CELLS)].isna().any(axis=1).sum()
            print(f"  [seeds] {n_missing} seed rows not found in finetuning_data; "
                  "scoring those with oracle")
            mask = merged[list(PROMOTER_CELLS)].isna().any(axis=1)
            to_score = merged.loc[mask, "sequence"].tolist()
            with torch.no_grad():
                for cell in PROMOTER_CELLS:
                    preds = PromoterCellAdapter(pool, cell)(to_score).detach().cpu().numpy()
                    merged.loc[mask, cell] = preds
        seeds = merged

    ranges = load_promoter_oracle_ranges(args.oracle_ranges)
    for cell in PROMOTER_CELLS:
        col_in = cell
        col_out = f"{cell}_mean"
        lo, hi = ranges[cell]["min"], ranges[cell]["max"]
        seeds[col_out] = (seeds[col_in] - lo) / (hi - lo)

    score_col = f"{args.task}_mean"
    other_cols = [f"{c}_mean" for c in PROMOTER_CELLS if c != args.task]
    seeds["target"] = (
        seeds[score_col]
        - (seeds[other_cols[0]] - args.constraint[0])
        + seeds[score_col]
        - (seeds[other_cols[1]] - args.constraint[1])
    )
    reward_cols = [f"{c}_mean" for c in PROMOTER_CELLS]
    seeds["rewards"] = seeds[reward_cols].values.tolist()
    seeds = seeds.sort_values("target", ascending=False).head(128).reset_index(drop=True)
    print(f"  Selected top {len(seeds)} seeds (by combined target score)")
    return seeds


def main():
    args = parse_args()
    if args.target_label is None:
        args.target_label = DEFAULT_TARGET_LABEL[args.task]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 72)
    print(f"CTRL-DNA PROMOTER  task={args.task}  target_label={args.target_label!r}")
    print(f"  seq_len:    {args.seq_len}")
    print(f"  max_iter:   {args.max_iter}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs:     {args.epoch}")
    print(f"  beta(KL):   {args.beta}")
    print(f"  lambda_lr:  {args.lambda_lr}")
    print(f"  lambda0:    {args.lambda_value}")
    print(f"  constraint: {args.constraint}")
    print(f"  seed:       {args.seed}")
    print("=" * 72)

    print(f"\n[Oracle] Loading promoter Enformer pool from {args.oracle_ckpt_dir}")
    pool = PromoterOraclePool(args.oracle_ckpt_dir, device=device)

    print(f"\n[Model] Loading fine-tuned HyenaDNA: {args.hyenadna_checkpoint}")
    agent = load_hyenadna_lightning(
        args.hyenadna_checkpoint, label_len=args.label_len, device=device
    )

    print(f"\n[Data] Loading seeds: {args.seed_csv}")
    seeds = prepare_seeds(args, pool)

    opt = CtrlDNAPromoter(args, agent, pool)
    opt.optimize(seeds)


if __name__ == "__main__":
    main()
