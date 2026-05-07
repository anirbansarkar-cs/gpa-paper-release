#!/usr/bin/env python3
"""Run new selections (by_composite, by_motif, by_kmer3, by_diversity) on
the 9 cell-targeted GPA pools.

  HepG2 ← run_dps_pw030_b25_gct060_nobio_r{1,2,3}
  K562  ← run_dps_pw030_b25_gct060_nobio_k562_r{1,2,3}
  SKNSH ← run_dps_pw030_b25_gct060_nobio_sknsh_r{1,2,3}

Each pool only scored against its target cell (saves 2/3 of FIMO work).
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

PROJECT = Path("${GPA_REPO_ROOT}")
sys.path.insert(0, str(PROJECT / "scripts/rerd_comparison/dnacraft"))

import run_batch_score as RB  # noqa: E402

POOLS = [
    ("run_dps_pw030_b25_gct060_nobio", "HepG2"),
    ("run_dps_pw030_b25_gct060_nobio_k562", "K562"),
    ("run_dps_pw030_b25_gct060_nobio_sknsh", "SKNSH"),
]
SEEDS = ["r1", "r2", "r3"]
SELECTIONS = ["by_composite", "by_motif", "by_kmer3", "by_diversity"]
SOURCE = "pool"
OUT_DIR = PROJECT / "results/rerd_comparison/dnacraft/per_pool_jsons"


def task(spec):
    base, cell, seed, sel = spec
    pool_dir = PROJECT / "results/rerd_comparison" / f"{base}_{seed}"
    out_path = OUT_DIR / f"{pool_dir.name}__{cell}__{sel}.json"
    if out_path.exists():
        return ("skip", str(out_path))
    try:
        class A: pass
        a = A()
        a.pool_dir = pool_dir
        a.cell = cell
        a.selection = sel
        a.n = 128
        a.ref_dir = PROJECT / "results/rerd_comparison/dnacraft/_ref_cache"
        a.output_json = out_path
        a.source = SOURCE
        RB._score_one(a)
        return ("ok", f"{pool_dir.name}__{cell}__{sel}")
    except Exception as ex:
        import traceback
        return ("err", f"{spec}: {ex}\n{traceback.format_exc()[-800:]}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for (base, cell) in POOLS:
        for seed in SEEDS:
            for sel in SELECTIONS:
                tasks.append((base, cell, seed, sel))
    n_workers = int(os.environ.get("N_WORKERS", "8"))
    print(f"Tasks: {len(tasks)}  workers: {n_workers}")
    t0 = time.time()
    if n_workers > 1:
        with Pool(n_workers, initializer=RB._worker_init) as p:
            for i, (status, msg) in enumerate(p.imap_unordered(task, tasks), 1):
                print(f"  [{i}/{len(tasks)}] [{status}] {msg}", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            status, msg = task(t)
            print(f"  [{i}/{len(tasks)}] [{status}] {msg}", flush=True)
    print(f"\nElapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
