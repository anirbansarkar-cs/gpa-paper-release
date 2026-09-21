#!/bin/bash
# Table 2 — Gosai HepG2 activity/naturalness trade-off, beta* in {5, 10, 15}.
#
# Recipe (Appendix Table 5): N = 5000, alpha = 0.5, T_max = 30, nf = 0.10,
# K = 8, lambda = 1.0, DPS on (eta = 3000), pw = 0 (single-cell task),
# no GC-centering, random initialization.
#
# 3 beta values x 3 reps = 9 runs, one GPU each. Wrap the `python` line in
# whatever scheduler you use.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GPA_EXTERNAL_ROOT:?set GPA_EXTERNAL_ROOT}"
OUT_ROOT="${OUT_ROOT:-results/rerd_comparison}"

MDLM_CKPT="${MDLM_CKPT:-${GPA_EXTERNAL_ROOT}/DRAKES_data/data_and_model/mdlm/outputs_gosai/pretrained.ckpt}"
ORACLE_CKPT="${ORACLE_CKPT:-${GPA_EXTERNAL_ROOT}/DRAKES_data/data_and_model/reward_oracle_ft.ckpt}"
EVAL_CKPT="${EVAL_CKPT:-${GPA_EXTERNAL_ROOT}/DRAKES_data/data_and_model/reward_oracle_eval.ckpt}"

# K = 8, lambda = 1.0, DPS on, no off-target penalty.
RECIPE=(--branch_factor 8 --diversity_lambda 1.0
        --use_dps --dps_eta 3000 --dps_penalty_weight 0.00 --penalty_weight 0.00)

BETAS=(${BETAS:-5 10 15})
REPS=(${REPS:-1 2 3})

for B in "${BETAS[@]}"; do
  for REP in "${REPS[@]}"; do
    OUT="${OUT_ROOT}/K8lam1_dps_nobio_b${B}_rand_r${REP}"
    mkdir -p "${OUT}"
    python scripts/rerd_comparison/run_rerd_gpa.py \
        --mdlm_checkpoint "${MDLM_CKPT}" \
        --oracle_checkpoint "${ORACLE_CKPT}" \
        --eval_oracle_checkpoint "${EVAL_CKPT}" \
        --eval_checkpoint_interval 1 \
        --archive_threshold 5.0 \
        --target_cell hepg2 \
        --population_size 5000 \
        --noise_fraction 0.10 \
        --max_beta "${B}" \
        --from_random \
        "${RECIPE[@]}" \
        --seed $(( REP * 1000 + 42 )) \
        --output_dir "${OUT}"
  done
done
