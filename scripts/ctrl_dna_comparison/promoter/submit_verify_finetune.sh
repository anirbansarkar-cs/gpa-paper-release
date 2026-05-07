#!/bin/bash
# Verify promoter HyenaDNA fine-tune + compute oracle ranges in one short
# SLURM job. Reads the smoke (or full) fine-tune ckpt, samples from three
# decile prompts, scores with Enformer oracles, and reports whether
# conditioning is learned.
#
# Usage:
#   bash scripts/ctrl_dna_comparison/promoter/submit_verify_finetune.sh [smoke|full]
set -euo pipefail

MODE="${1:-smoke}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
    echo "Usage: $0 [smoke|full]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

PROMOTER_DIR="scripts/ctrl_dna_comparison/promoter"
CKPT="${PROMOTER_DIR}/checkpoints/hyenadna_promoter_${MODE}/best.ckpt"
if [[ ! -e "${CKPT}" ]]; then
    echo "ERROR: ${CKPT} not found" >&2
    exit 1
fi

mkdir -p sbatch_out/ctrl_dna_comparison

JOB_NAME="verify_finetune_${MODE}"
JID=$(sbatch --parsable <<JOB_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=sbatch_out/ctrl_dna_comparison/${JOB_NAME}_%j.out
#SBATCH --error=sbatch_out/ctrl_dna_comparison/${JOB_NAME}_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_long
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=anonymous@example.com

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118
set -euo pipefail
cd ${PROJECT_DIR}

echo "=== Building oracle_ranges.json ==="
python ${PROMOTER_DIR}/build_oracle_ranges.py

echo
echo "=== Verifying fine-tune conditioning ==="
python ${PROMOTER_DIR}/verify_finetune.py \
    --hyenadna_checkpoint ${CKPT} \
    --oracle_ckpt_dir ${PROMOTER_DIR}/checkpoints \
    --n 32 \
    --prompts 000 100 010 001 111
JOB_EOF
)
echo "${JOB_NAME}: job ${JID}"
