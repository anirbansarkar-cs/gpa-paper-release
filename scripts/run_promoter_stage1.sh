#!/bin/bash
# Stage 1 for the promoter benchmark: fine-tune the HyenaDNA backbone.
#
# This checkpoint is the shared starting point for BOTH methods — GPA samples
# from it frozen, Ctrl-DNA RL-fine-tunes it. Run this before
# run_table4_promoter.sh and run_table4_ctrldna_baseline.sh.
#
# Conditioning follows Ctrl-DNA's published promoter scheme: a 3-digit prefix
# "<JURKAT><K562><THP1>", each digit 1 if that cell's activity is at or above
# the cell's median computed on TRAIN rows only. Their
# reinforce_multi_lagrange.py:get_prefix_label() prompts with "100"/"010"/"001",
# which are the three "this cell only high" cases under that scheme.
#
# One GPU. Requires the Ctrl-DNA release for its CharDataset tokenizer; set
# CTRL_DNA_HOME if it is not at ~/Ctrl-DNA.
set -euo pipefail
cd "$(dirname "$0")/.."

COMPARISON_DIR="scripts/ctrl_dna_comparison"
PROMOTER_DIR="${COMPARISON_DIR}/promoter"
DATA_CSV="${DATA_CSV:-${PROMOTER_DIR}/data/finetuning_data.csv}"
LABELED_CSV="${LABELED_CSV:-${COMPARISON_DIR}/data/promoter_labeled.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROMOTER_DIR}/checkpoints/hyenadna_promoter_e3}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"

if [ ! -f "${DATA_CSV}" ]; then
  echo "error: ${DATA_CSV} not found. See 'The promoter comparison (Table 4)'" >&2
  echo "in the README for the schema this file must have." >&2
  exit 1
fi

mkdir -p "$(dirname "${LABELED_CSV}")" "${OUTPUT_DIR}"

# 1. Median-threshold the activities into Ctrl-DNA's 3-digit prefix labels.
python "${PROMOTER_DIR}/build_hyenadna_labels.py" \
    --input_csv "${DATA_CSV}" \
    --output_csv "${LABELED_CSV}"

# 2. Fine-tune HyenaDNA as a conditional LM on those labels.
python "${COMPARISON_DIR}/train_hyenadna.py" \
    --labeled_csv "${LABELED_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 128 \
    --max_epochs "${MAX_EPOCHS}" \
    --lr 1e-4 \
    --val_fraction 0.10 \
    --num_workers 8 \
    --seed 42
