#!/bin/bash
# Table 4 — Ctrl-DNA promoter comparison on the Reddy MPRA benchmark
# (JURKAT / K562 / THP1, 250 bp, frozen HyenaDNA backbone).
#
# One universal inference-time recipe across all three cells (Appendix Table 6):
#   N = 10000, alpha = 0.5, beta* = 100, T_max = 60, nf = 0.05,
#   K = 8 (argmax branch selection), pw = 0.35, lambda = 1.0, no DPS,
#   no bio filter.
#
# 3 cells x 5 seeds = 15 runs, one GPU each, ~2.2 h/seed on an H100.
set -euo pipefail
cd "$(dirname "$0")/.."

PROMOTER_DIR="scripts/ctrl_dna_comparison/promoter"
OUT_ROOT="${OUT_ROOT:-results/ctrl_dna_comparison/promoter}"

# Defaults chain from run_promoter_stage1.sh and run_promoter_oracles.sh.
CKPT="${CKPT:-${PROMOTER_DIR}/checkpoints/hyenadna_promoter_e3/best.ckpt}"
ORACLE_DIR="${ORACLE_DIR:-${PROMOTER_DIR}/checkpoints}"

POP="${POP:-10000}"                # what the reported runs used
CELLS=(${CELLS:-JURKAT K562 THP1})
SEEDS=(${SEEDS:-0 1 2 3 4})        # seeds_arbitrary_set{0..4}.csv

SEED_CSV_DIR="${SEED_CSV_DIR:-${PROMOTER_DIR}/data}"
if [ ! -f "${SEED_CSV_DIR}/seeds_arbitrary_set0.csv" ]; then
  echo "error: promoter seed pools not found in ${SEED_CSV_DIR}." >&2
  echo "Build them: python ${PROMOTER_DIR}/build_arbitrary_seeds.py" >&2
  echo "See 'Seed selection' in the README." >&2
  exit 1
fi

for CELL in "${CELLS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    OUT="${OUT_ROOT}/gpa_universal_${CELL}_seed${SEED}"
    mkdir -p "${OUT}"
    python "${PROMOTER_DIR}/run_gpa_hyenadna_promoter.py" \
        --hyenadna_checkpoint "${CKPT}" \
        --oracle_ckpt_dir "${ORACLE_DIR}" \
        --seed_csv "${SEED_CSV_DIR}/seeds_arbitrary_set${SEED}.csv" \
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
