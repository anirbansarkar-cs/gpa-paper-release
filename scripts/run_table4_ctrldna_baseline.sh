#!/bin/bash
# Ctrl-DNA baseline for the promoter comparison (Table 4).
#
# Constrained-RL fine-tuning of a HyenaDNA policy, one policy per target cell,
# against the same three oracle checkpoints GPA designs against. Run
# scripts/run_promoter_oracles.sh first.
#
# "R200" means --max_iter 200. Budget roughly two days per seed on a single
# H100. Every setting used for our runs is spelled out below rather than left
# to a default.
#
# Requires the Ctrl-DNA release on disk; the wrapper imports
# `src.reglm` and `dna_optimizers_multi` from it. Set CTRL_DNA_HOME if it is not
# at ~/Ctrl-DNA.
set -euo pipefail
cd "$(dirname "$0")/.."

PROMOTER_DIR="scripts/ctrl_dna_comparison/promoter"
OUT_ROOT="${OUT_ROOT:-results/ctrl_dna_comparison/promoter}"

# Stage-1 HyenaDNA policy, the shared starting point for both methods.
# Defaults chain from run_promoter_stage1.sh and run_promoter_oracles.sh.
CKPT="${CKPT:-${PROMOTER_DIR}/checkpoints/hyenadna_promoter_e3/best.ckpt}"
ORACLE_DIR="${ORACLE_DIR:-${PROMOTER_DIR}/checkpoints}"

CELLS=(${CELLS:-JURKAT K562 THP1})
SEEDS=(${SEEDS:-0 1 2 3 4})

SEED_CSV_DIR="${SEED_CSV_DIR:-${PROMOTER_DIR}/data}"
if [ ! -f "${SEED_CSV_DIR}/seeds_JURKAT.csv" ]; then
  echo "error: no per-cell seed CSVs in ${SEED_CSV_DIR}." >&2
  echo "Ctrl-DNA needs one CSV per cell, named seeds_<CELL>.csv, with columns" >&2
  echo "  sequence,<CELL>   (250 bp, measured activity)" >&2
  echo "Choosing how to select them is left to you; see their release for the" >&2
  echo "rule their own pipeline uses. Override the directory with SEED_CSV_DIR." >&2
  exit 1
fi

for CELL in "${CELLS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    OUT="${OUT_ROOT}/ctrldna_r200_${CELL}_seed${SEED}"
    mkdir -p "${OUT}"
    python "${PROMOTER_DIR}/run_ctrldna_promoter.py" \
        --hyenadna_checkpoint "${CKPT}" \
        --oracle_ckpt_dir "${ORACLE_DIR}" \
        --oracle_ranges "${PROMOTER_DIR}/data/oracle_ranges.json" \
        --seed_csv "${SEED_CSV_DIR}/seeds_${CELL}.csv" \
        --task "${CELL}" \
        --max_iter 200 \
        --epoch 5 \
        --batch_size 128 \
        --beta 0.01 \
        --lambda_lr 3e-1 \
        --lambda_value 0.5 0.5 \
        --checkpoint_interval 50 \
        --seed "${SEED}" \
        --out_dir "${OUT}"
  done
done
