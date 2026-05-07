#!/bin/bash
#SBATCH --job-name=ag_v2_rescore
#SBATCH --output=sbatch_out/gpa_vs_ism_ledidi_v2/ag_rescore_%j.out
#SBATCH --error=sbatch_out/gpa_vs_ism_ledidi_v2/ag_rescore_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=fast
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
# JAX driver compat: cuDNN 9.x in jaxlib 0.9.1 needs driver >=555 (per
# reference_jax_alphagenome_cluster_setup.md). Old-driver H100s and V100s
# get excluded here. This is a real driver constraint, NOT a stale policy
# (so feedback_no_gpu_exclusions.md doesn't apply to this job).
#SBATCH --exclude=gpunode20,gpunode21,gpunode24,gpunode25,gpunode26,gpunode27,gpunode28
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anonymous@example.com

# ===========================================================================
# AG K562 rescore for v2 outputs (GPA + ISM + LEDIDI bundled)
# ===========================================================================
# Iterates every GPA h5 (pool), ISM trajectories.csv, and
# LEDIDI ledidi_edited_*.csv under the v2 root and appends K562 JAX AG
# scores in-place (dataset for h5; sibling _ag.csv for csv).
#
# Usage:
#   sbatch scripts/ledidi_comparison/score_ag_pool.sh
#   FILTER=pool_A sbatch scripts/ledidi_comparison/score_ag_pool.sh
#   bash scripts/ledidi_comparison/score_ag_pool.sh DRY  # interactive dry-run
# ===========================================================================
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p sbatch_out/gpa_vs_ism_ledidi_v2

V2_ROOT="${V2_ROOT:-results/ledidi_comparison/gpa_vs_ism_ledidi_v2}"

# JAX cluster setup (per reference_jax_alphagenome_cluster_setup.md):
# - Use absolute python: ~/.bashrc prepends d3_cuda118/bin to PATH which
#   would otherwise shadow the alphagenome env's python.
# - Disable JAX preallocation so it shares the GPU politely.
# - Force PATH to put cuda_nvcc 12.9 ahead of the d3_cuda118 ptxas (PTX 7.8,
#   too old for sm_90a on H100).
CUDA_NVCC_DIR=~/.conda/envs/alphagenome/lib/python3.11/site-packages/nvidia/cuda_nvcc
export PATH="${CUDA_NVCC_DIR}/bin:${PATH}"

CMD=(~/.conda/envs/alphagenome/bin/python
     scripts/ledidi_comparison/score_ag_pool.py
     --v2_root "$V2_ROOT"
     --batch_size "${BATCH_SIZE:-64}")

if [[ -n "${FILTER:-}" ]]; then
    CMD+=(--filter "$FILTER")
fi

if [[ "${1:-}" == "DRY" ]]; then
    CMD+=(--dry_run)
    echo "Dry-run discovery (no scoring):"
    "${CMD[@]}"
    exit 0
fi

echo "========================================================================"
echo "AG v2 rescore"
echo "  V2_ROOT: $V2_ROOT"
echo "  Filter:  ${FILTER:-<none>}"
echo "========================================================================"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_NVCC_DIR}" \
    "${CMD[@]}"
