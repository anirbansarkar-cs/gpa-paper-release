#!/bin/bash
# Train the three promoter activity oracles (JURKAT / K562 / THP1).
#
# These are a prerequisite for both scripts/run_table4_promoter.sh (GPA) and
# scripts/run_table4_ctrldna_baseline.sh (Ctrl-DNA). Both methods design against
# and are scored by this same set of three checkpoints.
#
# The model is Ctrl-DNA's own regLM `EnformerModel` (imported from their release,
# not reimplemented here), trained per cell with an MSE head on the promoter MPRA
# activity values.
#
# Input CSV schema (see "Training the promoter oracles" in the README):
#   sequence   250 bp string
#   JURKAT     float, measured activity
#   K562       float, measured activity
#   THP1       float, measured activity
#   is_train   bool, split flag
#   is_val     bool, split flag
#   is_test    bool, split flag
# The splits are read from those flags; the script does not resplit.
#
# One GPU per cell. Writes <out_dir>/human_paired_<tag>.ckpt, which is the
# filename Ctrl-DNA's base_optimizer.load_target_model expects.
set -euo pipefail
cd "$(dirname "$0")/.."

PROMOTER_DIR="scripts/ctrl_dna_comparison/promoter"
DATA_CSV="${DATA_CSV:-${PROMOTER_DIR}/data/finetuning_data.csv}"
OUT_DIR="${OUT_DIR:-${PROMOTER_DIR}/checkpoints}"

if [ ! -f "${DATA_CSV}" ]; then
  echo "error: ${DATA_CSV} not found. See the README section 'Training the" >&2
  echo "promoter oracles' for the schema this file must have." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

for CELL in JURKAT K562 THP1; do
  python "${PROMOTER_DIR}/train_oracles.py" \
      --cell "${CELL}" \
      --data_csv "${DATA_CSV}" \
      --out_dir "${OUT_DIR}" \
      --seq_len 250 \
      --epochs 20 \
      --batch_size 128 \
      --lr 1e-4 \
      --num_workers 4 \
      --pretrained
done

# Held-out Pearson / Spearman / MSE per cell on the val and test splits.
python "${PROMOTER_DIR}/eval_oracles.py" \
    --data_csv "${DATA_CSV}" \
    --ckpt_dir "${OUT_DIR}"

# Per-cell activity ranges, used to normalise the composite objective.
python "${PROMOTER_DIR}/build_oracle_ranges.py" \
    --data_csv "${DATA_CSV}" \
    --ckpt_dir "${OUT_DIR}" \
    --output "${PROMOTER_DIR}/data/oracle_ranges.json"
