#!/bin/bash
#SBATCH --job-name=ledidi_k562
#SBATCH --output=${GPA_REPO_ROOT}/sbatch_out/ledidi_comparison/ledidi_%j_stdout.out
#SBATCH --error=${GPA_REPO_ROOT}/sbatch_out/ledidi_comparison/ledidi_%j_stderr.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --qos=fast
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anonymous@example.com

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118

set -euo pipefail

# ---- Arguments ----
# SEED_MODE: ism, random_init, gpa_seed
SEED_MODE="${SEED_MODE:-ism}"
OUTPUT_DIR="${OUTPUT_DIR:-${GPA_REPO_ROOT}/results/ledidi_comparison/${SEED_MODE}}"

SCRIPT=${GPA_REPO_ROOT}/scripts/ledidi_comparison/run_ledidi_baseline.py

echo "=== LEDIDI baseline: seed_mode=${SEED_MODE} ==="
echo "=== Output: ${OUTPUT_DIR} ==="

python ${SCRIPT} \
    --seed_mode ${SEED_MODE} \
    --output_dir ${OUTPUT_DIR} \
    --target 15.0 \
    --l 0.1 \
    --tau 1.0 \
    --lr 1.0 \
    --max_iter 1000 \
    --ledidi_batch_size 64 \
    --early_stop 100 \
    --n_random 100 \
    --n_gpa_copies 50

echo ""
echo "=== Done: ${SEED_MODE} ==="
