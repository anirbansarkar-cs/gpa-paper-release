#!/usr/bin/env python3
"""Build the two 5K seed pools for the GPA vs ISM vs LEDIDI v2 K562 comparison.

  Pool A: 5,000 uniform-random 200 bp DNA sequences.
  Pool B: 5,000 randomly-sampled K562 lentimpra test sequences (200 bp insert,
          adapters stripped).

H5 schema (matches scripts/rerd_comparison/prepare_seeds.py):
  arr_0          (N, 4, L) float32   one-hot, diffusion encoding
  indices        (N, L)    int64     diffusion encoding A=0 C=1 G=2 T=3
  oracle_preds   (N,)      float32   for Pool A: zeros; Pool B: y_test value
  gc_fractions   (N,)      float32
  attrs: target_cell="k562", n_seqs, seq_len, source

Usage:
  python scripts/ledidi_comparison/build_seed_pools.py
  python scripts/ledidi_comparison/build_seed_pools.py --n_seqs 100 --suffix _smoke
  python scripts/ledidi_comparison/build_seed_pools.py --force
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

LENTIMPRA_H5 = "${HOME}/Deep-SEDD/data/lenti_MPRA_K562_data.h5"
ADAPTER_FWD = "AGGACCGGATCAACT"  # 15 bp 5' adapter
ADAPTER_RC  = "CATTGCGTGAACCGA"  # 15 bp 3' adapter
ADAPTER_LEN = 15
INSERT_LEN  = 200
TOTAL_LEN   = 230
DEFAULT_OUT = "${GPA_REPO_ROOT}/results/ledidi_comparison/gpa_vs_ism_ledidi_v2/seed_pools"


def write_h5(path: Path, indices: np.ndarray, oracle_preds: np.ndarray,
             source: str) -> None:
    """Write the standard seed-pool h5 schema. indices is (N, L) int64."""
    n, L = indices.shape
    onehot = np.eye(4, dtype=np.float32)[indices].transpose(0, 2, 1)  # (N, 4, L)
    gc = ((indices == 1) | (indices == 2)).mean(axis=1).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("arr_0", data=onehot, compression="gzip")
        f.create_dataset("indices", data=indices, compression="gzip")
        f.create_dataset("oracle_preds", data=oracle_preds.astype(np.float32),
                         compression="gzip")
        f.create_dataset("gc_fractions", data=gc, compression="gzip")
        f.attrs["target_cell"] = "k562"
        f.attrs["n_seqs"] = n
        f.attrs["seq_len"] = L
        f.attrs["source"] = source
    print(f"  Wrote {path}: shape={indices.shape} GC mean={gc.mean()*100:.1f}%")


def build_pool_a(n_seqs: int, rng: np.random.RandomState) -> np.ndarray:
    """Pool A: uniform random A/C/G/T at every position."""
    return rng.randint(0, 4, size=(n_seqs, INSERT_LEN), dtype=np.int64)


def build_pool_b(n_seqs: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Pool B: 5K random sample from the full K562 lentimpra test split.

    The lentimpra h5 stores 230 bp as (N, 230, 4) float onehot in diffusion
    encoding (A=0 C=1 G=2 T=3 — verified against seq_test strings). About half
    of the test rows are stored in reverse-complement orientation (the 5'-end
    starts with revcomp(ADAPTER_RC) instead of ADAPTER_FWD). For those rows we
    reverse-complement the entire 230 bp back to canonical fwd orientation
    BEFORE stripping adapters, so every sampled insert sits in fwd-adapter
    context — which is what the K562 oracle expects when it injects its own
    15-bp adapters during ``score()``. This recovers all 39,340 test rows
    (vs. only 19,670 if we filtered to natively fwd).

    RC-of-indices uses the (A,C,G,T)=(0,1,2,3) encoding's nice property:
    complement = 3 - x  (0↔3, 1↔2). The ``[::-1]`` slice gets a ``.copy()`` per
    the project numpy-pitfalls rule.
    """
    print(f"  Reading {LENTIMPRA_H5}")
    with h5py.File(LENTIMPRA_H5, "r") as f:
        all_seqs = f["seq_test"][:]
        all_onehot = f["onehot_test"]
        all_y = f["y_test"][:].astype(np.float32).ravel()
        n_total = len(all_seqs)
        prefixes = np.array([
            (s.decode() if isinstance(s, bytes) else s)[:ADAPTER_LEN]
            for s in all_seqs
        ])
        fwd_mask = prefixes == ADAPTER_FWD
        rc_mask  = prefixes == _revcomp_str(ADAPTER_RC)
        unknown  = ~(fwd_mask | rc_mask)
        n_fwd, n_rc, n_unk = int(fwd_mask.sum()), int(rc_mask.sum()), int(unknown.sum())
        print(f"  Test split: {n_total:,} total — fwd-oriented {n_fwd:,} "
              f"({100 * n_fwd / n_total:.1f}%), RC-oriented {n_rc:,} "
              f"({100 * n_rc / n_total:.1f}%), unknown {n_unk:,}")
        if n_unk:
            raise ValueError(f"{n_unk} test rows have neither fwd nor RC adapter "
                             f"prefix — refusing to sample (would corrupt insert)")
        if n_seqs > n_total:
            raise ValueError(f"Requested {n_seqs} > test split size {n_total}")
        sel = np.sort(rng.choice(n_total, size=n_seqs, replace=False))
        onehot_full = all_onehot[sel]                       # (N, 230, 4)
        seqs_bytes  = all_seqs[sel]
        y           = all_y[sel]
        sel_is_rc   = rc_mask[sel]                          # (N,) bool
    indices_full = onehot_full.argmax(axis=-1).astype(np.int64)  # (N, 230)

    # Reverse-complement RC-oriented rows back to canonical fwd orientation
    if sel_is_rc.any():
        rc_rows = np.where(sel_is_rc)[0]
        # complement: 3 - x  (in A=0 C=1 G=2 T=3 encoding); reverse the position axis
        rc_flipped = (3 - indices_full[rc_rows])[:, ::-1].copy()
        indices_full[rc_rows] = rc_flipped
        print(f"  Reverse-complemented {len(rc_rows)} RC-oriented rows to fwd")

    indices = indices_full[:, ADAPTER_LEN:ADAPTER_LEN + INSERT_LEN]  # (N, 200)

    # Sanity check: every (now-canonical) row should match the fwd adapter pattern
    bases = np.array(list("ACGT"))
    fwd_idx_pattern = np.array([{"A": 0, "C": 1, "G": 2, "T": 3}[c] for c in ADAPTER_FWD])
    rc_idx_pattern  = np.array([{"A": 0, "C": 1, "G": 2, "T": 3}[c] for c in ADAPTER_RC])
    for i in (0, n_seqs // 2, n_seqs - 1):
        head = indices_full[i, :ADAPTER_LEN]
        tail = indices_full[i, -ADAPTER_LEN:]
        assert np.array_equal(head, fwd_idx_pattern), (
            f"row {i}: head not fwd adapter after RC normalization "
            f"(orig stored prefix={'rc' if sel_is_rc[i] else 'fwd'})")
        assert np.array_equal(tail, rc_idx_pattern), (
            f"row {i}: tail not rc adapter after RC normalization")
    print(f"  Adapter sanity OK on 3 samples (after RC normalization).")
    return indices, y


def _revcomp_str(s: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp[c] for c in s[::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--n_seqs", type=int, default=5000)
    ap.add_argument("--suffix", default="",
                    help="Filename suffix, e.g. '_smoke' for a 100-seq variant")
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="Re-build even if output h5 already exists")
    args = ap.parse_args()

    rng_a = np.random.RandomState(args.rng_seed)
    rng_b = np.random.RandomState(args.rng_seed + 1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pool A: random ----
    name_a = f"pool_A_random_{args.n_seqs // 1000}k{args.suffix}.h5" \
        if args.n_seqs >= 1000 and args.n_seqs % 1000 == 0 \
        else f"pool_A_random_n{args.n_seqs}{args.suffix}.h5"
    out_a = out_dir / name_a
    if out_a.exists() and not args.force:
        print(f"[skip] {out_a} already exists (use --force to rebuild)")
    else:
        print(f"\nBuilding Pool A — uniform random {args.n_seqs} × {INSERT_LEN} bp")
        idx_a = build_pool_a(args.n_seqs, rng_a)
        oracle_a = np.zeros(args.n_seqs, dtype=np.float32)
        write_h5(out_a, idx_a, oracle_a,
                 source=f"uniform random (RandomState={args.rng_seed})")

    # --- Pool B: lentimpra test sample ----
    name_b = f"pool_B_lentimpra_test_{args.n_seqs // 1000}k{args.suffix}.h5" \
        if args.n_seqs >= 1000 and args.n_seqs % 1000 == 0 \
        else f"pool_B_lentimpra_test_n{args.n_seqs}{args.suffix}.h5"
    out_b = out_dir / name_b
    if out_b.exists() and not args.force:
        print(f"[skip] {out_b} already exists (use --force to rebuild)")
    else:
        print(f"\nBuilding Pool B — {args.n_seqs} random K562 lentimpra test sequences")
        idx_b, y_b = build_pool_b(args.n_seqs, rng_b)
        write_h5(out_b, idx_b, y_b,
                 source=f"K562 lentimpra test split, RandomState={args.rng_seed + 1}")

    print("\nDone.")


if __name__ == "__main__":
    main()
