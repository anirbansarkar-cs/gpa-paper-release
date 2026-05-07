#!/bin/bash
#SBATCH --job-name=ag_test_eval
#SBATCH --output=${HOME}/sbatch_out/alphagenome/ag_test_eval_%j_stdout.out
#SBATCH --error=${HOME}/sbatch_out/alphagenome/ag_test_eval_%j_stderr.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anonymous@example.com

set -euo pipefail

export PATH="${HOME}/.conda/envs/alphagenome/bin:$PATH"

cd ${GPA_REPO_ROOT}

echo "============================================================"
echo "Evaluate AG + LegNet on K562 lentiMPRA test set"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU:    ${CUDA_VISIBLE_DEVICES:-none}"
echo "============================================================"

python scripts/alphagenome/eval_ag_test_set.py

echo "Done."
