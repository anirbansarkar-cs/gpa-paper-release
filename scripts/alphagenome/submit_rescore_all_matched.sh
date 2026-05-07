#!/bin/bash
#SBATCH --job-name=ag_rescore_all
#SBATCH --output=sbatch_out/k562_mdlm_gpa/%j_agrescore_stdout.out
#SBATCH --error=sbatch_out/k562_mdlm_gpa/%j_agrescore_stderr.out
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=fast
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --exclude=bamgpu29
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anonymous@example.com

# Full PyTorch AG rescore: 20 GPA pools (~100K seqs) + 2 ISM trajectories
# (~232 seqs) + 20 LEDIDI matched CSVs (~1000 seqs), all on K562/HepG2/WTC11.
# Replaces the dead JAX-based ag_k562_scores in GPA H5s with new PyTorch
# scores (old stashed at ag_k562_scores_jax). Adds k562_ag/hepg2_ag/wtc11_ag
# columns to ISM/LEDIDI _ag.csv outputs.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p sbatch_out/k562_mdlm_gpa

# Use absolute python path: ~/.bashrc prepends d3_cuda118/bin to PATH, so
# `conda activate alphagenome` ends up shadowed by the d3_cuda118 python.
~/.conda/envs/alphagenome/bin/python scripts/alphagenome/rescore_all_matched.py
