#!/bin/bash
#SBATCH --job-name=enformer_gate
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=${GPA_REPO_ROOT}/sbatch_out/dna_craft_comparison/enformer_gate_%j.out
#SBATCH --error=${GPA_REPO_ROOT}/sbatch_out/dna_craft_comparison/enformer_gate_%j.err
#SBATCH --exclude=gpunode29

set -euo pipefail

REPO="${GPA_REPO_ROOT}"
SCRIPT="${REPO}/scripts/dna_craft_comparison/enhancer/oracles/eval_enformer_split.py"

mkdir -p "${REPO}/sbatch_out/dna_craft_comparison"

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118

echo "[gate] start: $(date)"
echo "[gate] node:  $(hostname)"

python "${SCRIPT}"
RC=$?

echo "[gate] done:  $(date)  rc=${RC}"
exit ${RC}
