#!/bin/bash
#SBATCH --job-name=gpa_dnacraft
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-8%8
#SBATCH --output=${GPA_REPO_ROOT}/sbatch_out/dna_craft_comparison/gpa_enhancer_%A_%a.out
#SBATCH --error=${GPA_REPO_ROOT}/sbatch_out/dna_craft_comparison/gpa_enhancer_%A_%a.err
#SBATCH --exclude=bamgpu29

# 9-task array: 3 cells × 3 replicate seeds.
# Index map: cell = cells[i // 3], seed = i % 3.
# bio_ai MaxJobsPU=8 → array throttled to 8 concurrent tasks; one job
# joins after the first finishes.

set -euo pipefail

REPO="${GPA_REPO_ROOT}"
RUNNER="${REPO}/scripts/dna_craft_comparison/enhancer/run_gpa_dnacraft_enhancer.py"

mkdir -p "${REPO}/sbatch_out/dna_craft_comparison"

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118

CELLS=("hepg2" "k562" "sknsh")
IDX="${SLURM_ARRAY_TASK_ID:-0}"
CELL="${CELLS[$((IDX / 3))]}"
SEED="$((IDX % 3))"

OUTPUT_DIR="${REPO}/results/dna_craft_comparison/enhancer/gpa_runs/${CELL}/seed${SEED}"
mkdir -p "${OUTPUT_DIR}"

BACKBONE="${BACKBONE:-hyenadna}"
USE_DPS_FLAG="${USE_DPS:-}"   # set USE_DPS=--use_dps to enable

echo "[gpa_dnacraft] start: $(date)"
echo "[gpa_dnacraft] node:  $(hostname)"
echo "[gpa_dnacraft] cell=${CELL} seed=${SEED} backbone=${BACKBONE}"
echo "[gpa_dnacraft] output_dir=${OUTPUT_DIR}"

python "${RUNNER}" \
    --cell "${CELL}" \
    --seed "${SEED}" \
    --backbone "${BACKBONE}" \
    --output_dir "${OUTPUT_DIR}" \
    --population_size 10000 \
    --max_steps 60 \
    --max_beta 100 \
    --noise_fraction 0.05 \
    --branch_factor 8 \
    --penalty_weight 0.5 \
    --eval_checkpoint_interval 5 \
    ${USE_DPS_FLAG}

echo "[gpa_dnacraft] done:  $(date)"
