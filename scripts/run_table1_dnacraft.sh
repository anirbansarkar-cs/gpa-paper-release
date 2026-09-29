#!/bin/bash
# Table 1 — DNA-CRAFT enhancer benchmark (HepG2 / K562 / SK-N-SH).
#
# Recipe `dps_pw030_b25_gct060_nobio` (Appendix Table 7, headline column):
#   K = 8, beta* = 25, pw = 0.30, DPS on (eta = 3000), GC-centering target 0.60.
# Drop --gc_dps_target/--gc_dps_weight for the no-GC-centering ablation column.
#
# One GPU per invocation, ~3.6 min/seed on an H100. Wrap the `python` line in
# whatever scheduler you use; nothing here assumes a particular cluster.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GPA_DATA_ROOT:?set GPA_DATA_ROOT}"
: "${GPA_EXTERNAL_ROOT:?set GPA_EXTERNAL_ROOT}"
OUT_ROOT="${OUT_ROOT:-results/dna_craft_comparison}"

# Same Gosai MDLM prior as run_table2_gosai.sh, from the DRAKES release.
MDLM_CKPT="${MDLM_CKPT:-${GPA_EXTERNAL_ROOT}/DRAKES_data/data_and_model/mdlm/outputs_gosai/pretrained.ckpt}"
# Same DRAKES multi-cell oracles as run_table2_gosai.sh; --target_cell selects
# the head. Seed pools are per cell, GC-filtered to [0.45, 0.55].
ORACLE_CKPT="${ORACLE_CKPT:-${GPA_EXTERNAL_ROOT}/DRAKES_data/data_and_model/reward_oracle_ft.ckpt}"
EVAL_CKPT="${EVAL_CKPT:-${GPA_EXTERNAL_ROOT}/DRAKES_data/data_and_model/reward_oracle_eval.ckpt}"
SEED_DIR="${SEED_DIR:-${GPA_DATA_ROOT}/gosai_seed_pools}"

POP="${POP:-5000}"                 # Appendix Table 7; sweep 128..20000 for Appendix N-ablation
CELLS=(${CELLS:-hepg2 k562 sknsh})
SEEDS=(${SEEDS:-42 1042 2042})     # 3 runs per cell

for CELL in "${CELLS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    OUT="${OUT_ROOT}/dnacraft_n${POP}_${CELL}_s${SEED}"
    SEED_FILE="${SEED_DIR}/gosai_seeds_${CELL}_gc45_55.h5"
    mkdir -p "${OUT}"
    python scripts/rerd_comparison/run_rerd_gpa.py \
        --mdlm_checkpoint "${MDLM_CKPT}" \
        --oracle_checkpoint "${ORACLE_CKPT}" \
        --eval_oracle_checkpoint "${EVAL_CKPT}" \
        --eval_checkpoint_interval 1 \
        --mingap_archive_size 64 \
        --perstep_archive_top_k 64 \
        --target_cell "${CELL}" \
        --population_size "${POP}" \
        --max_steps 30 \
        --noise_fraction 0.10 \
        --branch_factor 8 \
        --max_beta 25 \
        --use_dps --dps_eta 3000 \
        --dps_penalty_weight 0.30 --penalty_weight 0.30 \
        --gc_dps_target 0.60 --gc_dps_weight 10 \
        --seed_pool "${SEED_FILE}" --top_k_init \
        --seed "${SEED}" \
        --output_dir "${OUT}"
  done
done
