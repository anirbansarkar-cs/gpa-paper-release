#!/usr/bin/env python3
"""Check if stage2 also scores ACGT*50 repeat high."""
import numpy as np
from alphagenome_ft_mpra.oracle import load_oracle

STAGE1 = "${GPA_SHARED_ROOT}/alphagenome_encoder/mpra-K562-optimal/stage1"
STAGE2 = "${GPA_SHARED_ROOT}/alphagenome_encoder/mpra-K562-optimal/stage2"
LEFT_ADAPTER = "AGGACCGGATCAACT"
RIGHT_ADAPTER = "CATTGCGTGAACCGA"

repeat_seq = "ACGT" * 50

for name, ckpt in [("stage1", STAGE1), ("stage2", STAGE2)]:
    oracle = load_oracle(ckpt, left_adapter=LEFT_ADAPTER, right_adapter=RIGHT_ADAPTER)
    score = np.asarray(oracle.predict_sequences([repeat_seq], mode="core"), dtype=np.float64)
    print(f"{name}: ACGT*50 = {score[0]:.4f}")
