#!/bin/bash
#SBATCH --job-name=ag_score
#SBATCH --output=${HOME}/sbatch_out/alphagenome/ag_score_%j_stdout.out
#SBATCH --error=${HOME}/sbatch_out/alphagenome/ag_score_%j_stderr.out
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anonymous@example.com

# =============================================================================
# Score GPA outputs with AlphaGenome MPRA oracles (K562 + HepG2)
#
# Usage:
#   # Score a single H5
#   INPUT_H5=results/rerd_comparison/run_v9_pw0_smooth01_hard4555/gpa_output.h5 \
#   OUTPUT_CSV=results/alphagenome/v9_mean_hard4555.csv \
#   sbatch scripts/alphagenome/score_gpa.sh
#
#   # Score with append-back to H5
#   INPUT_H5=... OUTPUT_CSV=... APPEND_H5=1 sbatch scripts/alphagenome/score_gpa.sh
#
#   # Score CSV input
#   INPUT_H5=results/ism_high_oracle_scored.csv \
#   SEQ_COLUMN=sequence \
#   OUTPUT_CSV=results/alphagenome/ism_scores.csv \
#   sbatch scripts/alphagenome/score_gpa.sh
# =============================================================================

set -euo pipefail

# ---- Conda environment setup ------------------------------------------------
export PATH="${HOME}/.conda/envs/alphagenome/bin:$PATH"

# ---- Paths -------------------------------------------------------------------
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"

INPUT="${INPUT_H5:?Set INPUT_H5 to the H5/CSV file to score}"
OUTPUT="${OUTPUT_CSV:?Set OUTPUT_CSV for the output scores CSV}"
CELL_TYPES="${CELL_TYPES:-k562 hepg2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODE="${MODE:-core}"
SEQ_COLUMN="${SEQ_COLUMN:-sequence}"

# Build command
CMD=(
    python scripts/alphagenome/score_sequences.py
    --input "$INPUT"
    --output "$OUTPUT"
    --cell_types $CELL_TYPES
    --batch_size "$BATCH_SIZE"
    --mode "$MODE"
    --seq_column "$SEQ_COLUMN"
)

if [[ "${APPEND_H5:-0}" == "1" ]]; then
    CMD+=(--append_h5)
fi

echo "============================================================"
echo "AlphaGenome MPRA Oracle Scoring"
echo "============================================================"
echo "Input:      $INPUT"
echo "Output:     $OUTPUT"
echo "Cell types: $CELL_TYPES"
echo "Batch size: $BATCH_SIZE"
echo "Mode:       $MODE"
echo "Append H5:  ${APPEND_H5:-0}"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "============================================================"

"${CMD[@]}"

echo ""
echo "Done. Output: $OUTPUT"
