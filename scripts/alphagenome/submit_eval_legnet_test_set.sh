#!/bin/bash
#SBATCH --job-name=ln_test_eval
#SBATCH --output=${HOME}/sbatch_out/alphagenome/ln_test_eval_%j_stdout.out
#SBATCH --error=${HOME}/sbatch_out/alphagenome/ln_test_eval_%j_stderr.out
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anonymous@example.com

set -eo pipefail

source ~/.bashrc 2>/dev/null || true
set -u
conda activate d3_cuda118

cd ${GPA_REPO_ROOT}

echo "============================================================"
echo "Evaluate LegNet on K562 lentiMPRA test set"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU:    ${CUDA_VISIBLE_DEVICES:-none}"
echo "============================================================"

python scripts/alphagenome/eval_legnet_test_set.py

echo "Done."
