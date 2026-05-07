#!/bin/bash
#SBATCH --job-name=ag_rescore_jax
#SBATCH --output=sbatch_out/k562_mdlm_gpa/%j_agrescore_jax_stdout.out
#SBATCH --error=sbatch_out/k562_mdlm_gpa/%j_agrescore_jax_stderr.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --qos=fast
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
# Driver split (probed 2026-04-25): NEW (580.126) on gpunode{15,16,17,18,19,22,23};
# OLD (545.23.8) on gpunode{20,21,24,25,27,28}. cuDNN 9.x in jaxlib 0.9.1 needs
# driver >=555. V100s also failed (sm_70 too old for jaxlib 0.9.1's cuDNN convs).
#SBATCH --exclude=gpunode20,gpunode21,gpunode24,gpunode25,gpunode26,gpunode27,gpunode28,gpunode29
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anonymous@example.com

# JAX AG rescore: 20 GPA pools (~100K seqs) + 2 ISM trajectories (~232 seqs)
# + 20 LEDIDI matched CSVs (~1000 seqs), all on K562/HepG2/WTC11. Writes
# ag_<cell>_scores_jax_v2 datasets to GPA H5s and appends <cell>_ag_jax columns
# to the existing PyT-scored ISM/LEDIDI _ag.csv outputs. Does NOT touch
# the PyT v2 keys or the original ag_k562_scores_jax (old JAX) keys.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p sbatch_out/k562_mdlm_gpa

# Use absolute python path: ~/.bashrc prepends d3_cuda118/bin to PATH, so
# `conda activate alphagenome` ends up shadowed by the d3_cuda118 python.
# Disable pre-allocation so JAX shares the GPU politely; the model is small.
CUDA_NVCC_DIR=~/.conda/envs/alphagenome/lib/python3.11/site-packages/nvidia/cuda_nvcc
# ~/.bashrc prepends d3_cuda118/bin (CUDA 11.8 ptxas, PTX 7.8 max) — too old
# for sm_90a on H100. Force PATH so XLA's ptxas-finder hits cuda_nvcc 12.9 first.
export PATH="${CUDA_NVCC_DIR}/bin:${PATH}"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_NVCC_DIR}" \
~/.conda/envs/alphagenome/bin/python scripts/alphagenome/rescore_all_matched_jax.py
