#!/bin/bash
#SBATCH --job-name=ag_v2_rescore_stage2
#SBATCH --output=sbatch_out/gpa_vs_ism_ledidi_v2/ag_rescore_stage2_%A_%a.out
#SBATCH --error=sbatch_out/gpa_vs_ism_ledidi_v2/ag_rescore_stage2_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --array=0-3
#SBATCH --exclude=gpunode20,gpunode21,gpunode24,gpunode25,gpunode26,gpunode27,gpunode28,gpunode29
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anonymous@example.com

# ===========================================================================
# AG K562 STAGE2 rescore for v2 outputs (GPA + ISM + LEDIDI bundled)
# ===========================================================================
# Same pipeline as score_ag_pool.sh but:
#   - Uses STAGE2 (full fine-tune) JAX checkpoint
#   - Runs as a 4-task SLURM array (set --array=0-N to split into N+1 shards)
#   - Each task scores 1/N_SHARDS of the discovered inputs (mod-N split)
#   - Writes to *_stage2{,_rc} dataset keys / column names; stage1 keys
#     remain untouched.
#
# Override shard count:
#   NUM_SHARDS=8 sbatch --array=0-7 scripts/ledidi_comparison/score_ag_pool_stage2.sh
# ===========================================================================
set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p sbatch_out/gpa_vs_ism_ledidi_v2

V2_ROOT="${V2_ROOT:-results/ledidi_comparison/gpa_vs_ism_ledidi_v2}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

CUDA_NVCC_DIR=~/.conda/envs/alphagenome/lib/python3.11/site-packages/nvidia/cuda_nvcc
export PATH="${CUDA_NVCC_DIR}/bin:${PATH}"

CMD=(~/.conda/envs/alphagenome/bin/python
     scripts/ledidi_comparison/score_ag_pool.py
     --v2_root "$V2_ROOT"
     --stage stage2
     --shard "$SHARD"
     --num_shards "$NUM_SHARDS"
     --batch_size "${BATCH_SIZE:-64}")

if [[ -n "${FILTER:-}" ]]; then
    CMD+=(--filter "$FILTER")
fi

echo "========================================================================"
echo "AG v2 STAGE2 rescore  (shard $SHARD / $NUM_SHARDS)"
echo "  V2_ROOT: $V2_ROOT"
echo "  Filter:  ${FILTER:-<none>}"
echo "  Stage:   stage2 (full fine-tune)"
echo "========================================================================"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_NVCC_DIR}" \
    "${CMD[@]}"
