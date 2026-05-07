#!/usr/bin/env python3
"""Parallel driver that scores every (GPA pool × cell × selection) combination
under the DNA-CRAFT Table-2 protocol. Uses multiprocessing.Pool with N_WORKERS.

Discovers pools by globbing `run_*` and `run_paper_*` and `run_v17_*` dirs
under results/rerd_comparison/. For each pool, runs both selections × 3 cells.

Skips combinations whose output JSON already exists (idempotent).
"""
import argparse
import json
import os
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

PROJECT = Path("${GPA_REPO_ROOT}")
sys.path.insert(0, str(PROJECT / "scripts/rerd_comparison/dnacraft"))

# Lazy import — module-heavy (loads DRAKES paths)
import score_dnacraft_protocol as SC  # noqa: E402

POOLS_DIR = PROJECT / "results/rerd_comparison"
# OUT_DIR is set in main() based on --source so different sources
# Write to a single output folder per --source.
DEFAULT_OUT_DIR = POOLS_DIR / "dnacraft/per_pool_jsons"
CELLS = ["HepG2", "K562", "SKNSH"]
SELECTIONS_ALL = ["by_mingap", "by_composite", "by_motif", "by_kmer3",
                  "by_diversity", "by_pool_composite"]
SELECTIONS = ["by_mingap", "by_composite"]  # default for legacy callers
SOURCE = "final"   # set in main(); read by task_fn / _score_one workers
N_SELECT = 128     # set in main() via --n; controls top-N selection size
OUT_DIR = DEFAULT_OUT_DIR  # set in main()


def discover_pools(patterns: list = None, source: str = "final") -> list:
    """Find all GPA pool dirs that have the required h5 file for `source`."""
    if patterns is None:
        patterns = ["run_*", "run_paper_*", "run_v17_*", "run_v25_*"]
    out = []
    for pat in patterns:
        for d in sorted(POOLS_DIR.glob(pat)):
            if not d.is_dir():
                continue
            try:
                _ = SC.find_input_h5(d, source=source)
                out.append(d)
            except FileNotFoundError:
                continue
    return out


def _worker_init():
    """Initialize each worker process. Force inner FIMO single-threaded so
    outer parallelism (Pool workers) is the only multiprocessing layer.
    Without this: 16 outer × 16 inner = 256 procs on 16 cores → thrashing."""
    os.environ["SLURM_CPUS_PER_TASK"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def task_fn(spec: tuple) -> tuple:
    """Run one (pool, cell, selection) scoring. Returns (status, msg)."""
    pool_dir, cell, selection = spec
    out_path = OUT_DIR / f"{pool_dir.name}__{cell}__{selection}.json"
    if out_path.exists():
        return ("skip", str(out_path))
    try:
        # Mock argparse
        class Args:
            pass
        args = Args()
        args.pool_dir = pool_dir
        args.cell = cell
        args.selection = selection
        args.n = N_SELECT
        args.ref_dir = PROJECT / "results/rerd_comparison/dnacraft/_ref_cache"
        args.output_json = out_path
        args.source = SOURCE
        # Reuse the main scoring logic by calling a slimmed-down version
        _score_one(args)
        return ("ok", str(out_path))
    except Exception as ex:
        import traceback
        return ("err", f"{pool_dir.name} | {cell} | {selection} : {ex}\n{traceback.format_exc()[-500:]}")


def _score_one(args):
    """Inline equivalent of score_dnacraft_protocol.main() — uses pure pymemesuite
    (no DRAKES/grelu/lightning imports — those take 20+ min on slow NFS)."""
    import h5py
    import numpy as np
    import pandas as pd
    import build_dnacraft_reference as BR
    import selection as SEL

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    pool_name = args.pool_dir.name
    recipe_label, seed = SC.parse_recipe_seed(pool_name)
    t0 = time.time()
    h5_path = SC.find_input_h5(args.pool_dir, source=getattr(args, "source", "final"))
    indices, seqs, cell_scores = SC.load_pool_with_cell_scores(h5_path)

    ref = BR.load_ref(args.cell, ref_dir=args.ref_dir)
    ref_kmer_vec = SC.reference_kmer3_freq(ref["kmer3_counts"])
    target_idx = SC.CELL_TO_IDX[args.cell]

    if args.selection == "by_mingap":
        keep_idx = SEL.select_top_n_by_mingap(cell_scores, target_idx, n=args.n)
    elif args.selection == "by_diversity":
        # Pure index-based, no FIMO needed
        keep_idx = SEL.select_top_n_by_diversity(indices, n=args.n)
    elif args.selection == "by_kmer3":
        seq_kmer = SC.per_seq_kmer3_freq(seqs)
        keep_idx = SEL.select_top_n_by_kmer3(seq_kmer, ref_kmer_vec, n=args.n)
    else:
        # by_composite / by_motif — both need per-seq FIMO scan.
        # Cache key includes source: different pool variants have
        # different seqs even though sizes may match — using the wrong cache
        # silently selects wrong indices.
        cache_dir = args.pool_dir.parent / "dnacraft" / "_pool_fimo_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        src = getattr(args, "source", "final")
        cache_path = cache_dir / f"{pool_name}__{src}_fimo_csr.npz"
        if cache_path.exists():
            from scipy.sparse import csr_matrix
            z = np.load(cache_path, allow_pickle=True)
            seq_motif_counts = csr_matrix((z["data"], z["indices"], z["indptr"]),
                                          shape=tuple(z["shape"])).toarray()
            motif_names = z["motif_names"].tolist()
        else:
            motifs, bg = SC._read_motifs()
            csr, motif_names = SC._fimo_scan_per_seq(seqs, motifs, bg)
            seq_motif_counts = csr.toarray()
            np.savez(cache_path,
                     data=csr.data, indices=csr.indices, indptr=csr.indptr,
                     shape=np.array(csr.shape, dtype=np.int64),
                     motif_names=np.array(motif_names, dtype=object))
        ref_motif_freq = np.array([ref["motif_counts"].get(m, 0.0) for m in motif_names], dtype=float)
        if args.selection == "by_composite":
            seq_kmer = SC.per_seq_kmer3_freq(seqs)
            keep_idx = SEL.select_top_n_by_composite(
                seq_kmer, seq_motif_counts, ref_kmer_vec, ref_motif_freq, indices, n=args.n
            )
        elif args.selection == "by_motif":
            keep_idx = SEL.select_top_n_by_motif(seq_motif_counts, ref_motif_freq, n=args.n)
        elif args.selection == "by_pool_composite":
            seq_kmer = SC.per_seq_kmer3_freq(seqs)
            target_idx = SC.CELL_TO_IDX[args.cell]
            per_seq_mg = SEL.per_seq_mingap(cell_scores, target_idx)
            keep_idx = SEL.select_top_n_by_pool_composite(
                seq_motif_counts=seq_motif_counts,
                seq_kmer_freq=seq_kmer,
                seqs_int=indices,
                ref_motif_freq=ref_motif_freq,
                ref_kmer_freq=ref_kmer_vec,
                per_seq_mingap=per_seq_mg,
                n=args.n,
                weights=(1.0, 1.0, 1.0, 1.0),
                mingap_floor_pct=0.0,
            )
        else:
            raise ValueError(f"unknown selection: {args.selection}")

    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    sub_seqs = [seqs[i] for i in keep_idx]
    sub_cell_scores = cell_scores[keep_idx]

    out = {
        "pool_dir": str(args.pool_dir),
        "pool_name": pool_name,
        "recipe": recipe_label,
        "seed": seed,
        "target_cell": args.cell,
        "selection": args.selection,
        "n_seqs": int(len(keep_idx)),
        "ref_n_seqs": int(ref["n_seqs"]),
        "ref_mingap_threshold": float(ref["mingap_threshold"]),
    }
    per_seq_mg = SEL.per_seq_mingap(sub_cell_scores, target_idx)
    out["mingap_eval_mean"] = float(per_seq_mg.mean())
    out["mingap_eval_std"] = float(per_seq_mg.std())
    out["mingap_eval_median"] = float(np.median(per_seq_mg))

    motifs, bg = SC._read_motifs()
    pool_motif_counts = SC._fimo_scan_pool(sub_seqs, motifs, bg)
    out["motif_corr_spearman"] = SC.motif_spearman(pool_motif_counts, ref["motif_counts"])

    pool_kmer = SC._count_kmers_inline(sub_seqs, k=3)
    out["kmer3_corr_pearson"] = SC.kmer3_pearson_pool(pool_kmer, ref["kmer3_counts"])
    out["diversity_bits"] = SC.per_position_shannon_bits(sub_seqs)
    out["elapsed_seconds"] = float(time.time() - t0)
    args.output_json.write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", nargs="+", default=None,
                    help="Pool dir glob patterns (default: run_*, run_paper_*, run_v17_*, run_v25_*)")
    ap.add_argument("--cells", nargs="+", default=CELLS, choices=CELLS)
    ap.add_argument("--selections", nargs="+", default=SELECTIONS, choices=SELECTIONS_ALL)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("N_WORKERS", "1")))
    ap.add_argument("--source", default="final",
                    choices=["final", "pool", "filtered", "mingap_filtered",
                             "perstep_filtered"],
                    help="Which candidate pool h5 "
                         "(final|pool|filtered|mingap_filtered|perstep_filtered)")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="JSON output dir; defaults to per_pool_jsons_<source>")
    ap.add_argument("--n", type=int, default=128,
                    help="Top-N selection size (paper headline = 128; "
                         "DNA-CRAFT G* matched comparison = 64)")
    args = ap.parse_args()

    global SOURCE, OUT_DIR, N_SELECT
    SOURCE = args.source
    N_SELECT = args.n
    if args.out_dir is None:
        suffix = "" if args.n == 128 else f"_n{args.n}"
        if args.source == "final":
            OUT_DIR = (DEFAULT_OUT_DIR if not suffix
                       else POOLS_DIR / f"dnacraft/per_pool_jsons{suffix}")
        else:
            OUT_DIR = POOLS_DIR / f"dnacraft/per_pool_jsons_{args.source}{suffix}"
    else:
        OUT_DIR = args.out_dir
    print(f"Source: {args.source}  OUT_DIR={OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool_dirs = discover_pools(args.pattern, source=args.source)
    print(f"Found {len(pool_dirs)} pool dirs")

    tasks = list(product(pool_dirs, args.cells, args.selections))
    # Filter already-done
    pending = [t for t in tasks
               if not (OUT_DIR / f"{t[0].name}__{t[1]}__{t[2]}.json").exists()]
    print(f"Total tasks: {len(tasks)}  pending: {len(pending)}  workers: {args.workers}")

    if len(pending) == 0:
        print("Nothing to do — all outputs already exist.")
        return

    if args.workers <= 1:
        _worker_init()
        for t in pending:
            status, msg = task_fn(t)
            print(f"  [{status}] {msg}")
    else:
        with Pool(args.workers, initializer=_worker_init) as p:
            for i, (status, msg) in enumerate(p.imap_unordered(task_fn, pending), 1):
                print(f"  [{i}/{len(pending)}] [{status}] {msg}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
