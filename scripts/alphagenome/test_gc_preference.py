#!/usr/bin/env python3
"""Test AG oracle GC content preference with controlled random sequences."""

import numpy as np
from alphagenome_ft_mpra.oracle import load_oracle

STAGE1 = "${GPA_SHARED_ROOT}/alphagenome_encoder/mpra-K562-optimal/stage1"
LEFT_ADAPTER = "AGGACCGGATCAACT"
RIGHT_ADAPTER = "CATTGCGTGAACCGA"

N_SEQS = 50  # per GC level
SEQ_LEN = 200
GC_LEVELS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def generate_gc_controlled(gc_frac, n=50, seq_len=200, seed=42):
    """Generate random sequences with exact GC content."""
    rng = np.random.default_rng(seed)
    n_gc = int(round(gc_frac * seq_len))
    n_at = seq_len - n_gc
    seqs = []
    for i in range(n):
        # Build sequence with exact GC count
        bases = []
        # GC bases: randomly split between G and C
        n_g = rng.binomial(n_gc, 0.5)
        n_c = n_gc - n_g
        # AT bases: randomly split between A and T
        n_a = rng.binomial(n_at, 0.5)
        n_t = n_at - n_a
        bases = ["G"] * n_g + ["C"] * n_c + ["A"] * n_a + ["T"] * n_t
        rng.shuffle(bases)
        seqs.append("".join(bases))
    return seqs


def main():
    print("Loading AG K562 stage1 oracle...")
    oracle = load_oracle(STAGE1, left_adapter=LEFT_ADAPTER, right_adapter=RIGHT_ADAPTER)

    print(f"\nGC preference test: {N_SEQS} random sequences per GC level, {SEQ_LEN}bp")
    print(f"{'GC%':>6}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'Median':>8}")
    print("-" * 56)

    all_results = []
    for gc in GC_LEVELS:
        seqs = generate_gc_controlled(gc, n=N_SEQS, seq_len=SEQ_LEN)
        scores = np.asarray(oracle.predict_sequences(seqs, mode="core"), dtype=np.float64)
        # Verify actual GC
        actual_gc = np.mean([(s.count("G") + s.count("C")) / len(s) for s in seqs])
        print(f"{actual_gc:>5.1%}  {scores.mean():>8.4f}  {scores.std():>8.4f}  "
              f"{scores.min():>8.4f}  {scores.max():>8.4f}  {np.median(scores):>8.4f}")
        all_results.append((gc, scores.mean(), scores.std()))

    # Also test some special sequences
    print(f"\n{'Special sequences':}")
    print(f"{'Sequence':>20}  {'GC%':>6}  {'Score':>8}")
    print("-" * 40)
    specials = {
        "ACGT*50": "ACGT" * 50,
        "poly-A": "A" * 200,
        "poly-T": "T" * 200,
        "poly-G": "G" * 200,
        "poly-C": "C" * 200,
        "AT*100": "AT" * 100,
        "GC*100": "GC" * 100,
        "AATT*50": "AATT" * 50,
        "GGCC*50": "GGCC" * 50,
        "ACGT*50 RC": "ACGT" * 50,  # same as ACGT*50, will compute RC below
    }
    # Replace RC with actual reverse complement
    rc_map = str.maketrans("ACGT", "TGCA")
    specials["ACGT*50 RC"] = ("ACGT" * 50)[::-1].translate(rc_map)

    for name, seq in specials.items():
        gc = (seq.count("G") + seq.count("C")) / len(seq)
        score = np.asarray(oracle.predict_sequences([seq], mode="core"), dtype=np.float64)[0]
        print(f"{name:>20}  {gc:>5.1%}  {score:>8.4f}")


if __name__ == "__main__":
    main()
