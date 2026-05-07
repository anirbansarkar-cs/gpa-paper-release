#!/bin/bash
#SBATCH --job-name=ag_rescore_jax_smoke
#SBATCH --output=sbatch_out/k562_mdlm_gpa/%j_agrescore_jax_smoke_stdout.out
#SBATCH --error=sbatch_out/k562_mdlm_gpa/%j_agrescore_jax_smoke_stderr.out
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=fast
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
# Driver split (probed 2026-04-25): NEW (580.126) on gpunode{15,16,17,18,19,22,23};
# OLD (545.23.8) on gpunode{20,21,24,25,27,28}. cuDNN 9.x in jaxlib 0.9.1 needs
# driver >=555, so force onto the new-driver H100s. (gpunode26 = V100, also stale.)
#SBATCH --exclude=gpunode20,gpunode21,gpunode24,gpunode25,gpunode26,gpunode27,gpunode28,gpunode29
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anonymous@example.com

# Smoke: 3 inputs (1 GPA H5 + 1 ISM CSV + 1 LEDIDI CSV) × 3 cells. Dry-run
# (no writes) to confirm GPU JAX works end-to-end before the full job.
# No --constraint=h100 — JAX broke on H100 with PTX 7.8 vs sm_90a; let SLURM
# place this on V100/A100 where jaxlib's CUDA build works.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p sbatch_out/k562_mdlm_gpa

CUDA_NVCC_DIR=~/.conda/envs/alphagenome/lib/python3.11/site-packages/nvidia/cuda_nvcc
# ~/.bashrc prepends d3_cuda118/bin (CUDA 11.8 ptxas, PTX 7.8 max) — too old
# for sm_90a. Force PATH so XLA's ptxas-finder hits cuda_nvcc 12.9 first.
export PATH="${CUDA_NVCC_DIR}/bin:${PATH}"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_NVCC_DIR}" \
~/.conda/envs/alphagenome/bin/python scripts/alphagenome/rescore_all_matched_jax.py \
    --limit 3 --dry_run
