#!/bin/bash
#SBATCH --job-name=ag_all
#SBATCH --output=${HOME}/sbatch_out/alphagenome/ag_all_%j_stdout.out
#SBATCH --error=${HOME}/sbatch_out/alphagenome/ag_all_%j_stderr.out
#SBATCH --time=12:00:00
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

python scripts/alphagenome/score_all_sequential.py \
    --results_dir results/rerd_comparison \
    --output results/alphagenome/all_runs_hepg2_summary.csv \
    --cell_types hepg2 \
    --batch_size 64
