#!/bin/bash
#SBATCH --job-name=ag_compare
#SBATCH --output=${HOME}/sbatch_out/alphagenome/ag_compare_%j_stdout.out
#SBATCH --error=${HOME}/sbatch_out/alphagenome/ag_compare_%j_stderr.out
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anonymous@example.com

set -euo pipefail

export PATH="${HOME}/.conda/envs/alphagenome/bin:$PATH"
cd ${GPA_REPO_ROOT}

python scripts/alphagenome/test_gc_preference.py
