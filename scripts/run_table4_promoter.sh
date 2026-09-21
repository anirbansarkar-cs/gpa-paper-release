#!/bin/bash
# Table 4 — Ctrl-DNA promoter comparison on the Reddy MPRA benchmark
# (JURKAT / K562 / THP1, 250 bp, frozen HyenaDNA backbone).
#
# One universal inference-time recipe across all three cells (Appendix Table 6):
#   N = 5000, alpha = 0.5, beta* = 100, T_max = 60, nf = 0.05,
#   K = 8 (argmax branch selection), pw = 0.35, lambda = 1.0, no DPS,
#   no bio filter.
#
# 3 cells x 5 seeds = 15 runs, one GPU each, ~2.2 h/seed on an H100.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GPA_DATA_ROOT:?set GPA_DATA_ROOT}"
: "${GPA_SHARED_ROOT:?set GPA_SHARED_ROOT}"
PROMOTER_DIR="scripts/ctrl_dna_comparison/promoter"
OUT_ROOT="${OUT_ROOT:-results/ctrl_dna_comparison/promoter}"

CKPT="${CKPT:-${GPA_SHARED_ROOT}/hyenadna/promoter_stage1_e3.ckpt}"
ORACLE_DIR="${ORACLE_DIR:-${GPA_DATA_ROOT}/reddy_promoter_oracles}"

POP="${POP:-5000}"                 # Appendix Table 6
CELLS=(${CELLS:-JURKAT K562 THP1})
SEEDS=(${SEEDS:-0 1 2 3 4})        # seeds_arbitrary_set{0..4}.csv

for CELL in "${CELLS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    OUT="${OUT_ROOT}/gpa_universal_${CELL}_seed${SEED}"
    mkdir -p "${OUT}"
    python "${PROMOTER_DIR}/run_gpa_hyenadna_promoter.py" \
        --hyenadna_checkpoint "${CKPT}" \
        --oracle_ckpt_dir "${ORACLE_DIR}" \
        --seed_csv "${PROMOTER_DIR}/data/seeds_arbitrary_set${SEED}.csv" \
        --target_cell "${CELL}" \
        --population_size "${POP}" \
        --max_beta 100 \
        --max_steps 60 \
        --ess_threshold 0.5 \
        --noise_fraction 0.05 \
        --branch_factor 8 \
        --penalty_weight 0.35 \
        --diversity_lambda 1.0 \
        --eta 3000 \
        --no_bio_filter \
        --seed "${SEED}" \
        --output_dir "${OUT}"
  done
done
