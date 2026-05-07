#!/bin/bash
# =============================================================================
# Fine-tune HyenaDNA on Reddy 2024 promoter MPRA (Stage 1, shared backbone).
# Output is used by BOTH Ctrl-DNA (PPO starting point) and GPA (frozen AR
# proposal with DPS guidance).
#
# Modes produce epoch-tagged output directories so multiple Stage 1 variants
# can coexist (e.g. e3 for paper-match vs e15 for a denser-conditioning arm):
#   smoke -> checkpoints/hyenadna_promoter_smoke/   (3 epochs, 2H wall, qos_long)
#   e3    -> checkpoints/hyenadna_promoter_e3/      (3 epochs, 2H wall, qos_long)
#   e15   -> checkpoints/hyenadna_promoter_e15/     (15 epochs, 4H wall, qos_long)
# For a custom epoch count, export EPOCHS=<N> and pass mode=`custom`.
#
# Usage: bash submit_train_hyenadna.sh [smoke|e3|e15|custom]
# =============================================================================
set -euo pipefail

MODE="${1:-smoke}"
case "${MODE}" in
    smoke)  MAX_EPOCHS=3;  WALL="02:00:00" ;;
    e3)     MAX_EPOCHS=3;  WALL="02:00:00" ;;
    e15)    MAX_EPOCHS=15; WALL="04:00:00" ;;
    custom) MAX_EPOCHS="${EPOCHS:?set EPOCHS env var for custom mode}"; WALL="${WALL:-04:00:00}" ;;
    *) echo "Usage: $0 [smoke|e3|e15|custom]" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

COMPARISON_DIR="scripts/ctrl_dna_comparison"
LABELED_CSV="${COMPARISON_DIR}/data/promoter_labeled.csv"
OUTPUT_DIR="${COMPARISON_DIR}/promoter/checkpoints/hyenadna_promoter_${MODE}"

if [[ ! -f "${LABELED_CSV}" ]]; then
    echo "ERROR: labeled CSV not found at ${LABELED_CSV}" >&2
    echo "Run: python ${COMPARISON_DIR}/promoter/build_hyenadna_labels.py" >&2
    exit 1
fi

mkdir -p sbatch_out/ctrl_dna_comparison "${OUTPUT_DIR}"

JOB_NAME="hyenadna_promoter_${MODE}"

JID=$(sbatch --parsable <<JOB_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=sbatch_out/ctrl_dna_comparison/${JOB_NAME}_%j.out
#SBATCH --error=sbatch_out/ctrl_dna_comparison/${JOB_NAME}_%j.err
#SBATCH --time=${WALL}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
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

python ${COMPARISON_DIR}/train_hyenadna.py \
    --labeled_csv ${LABELED_CSV} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --max_epochs ${MAX_EPOCHS} \
    --lr 1e-4 \
    --val_fraction 0.10 \
    --num_workers 8 \
    --seed 42
JOB_EOF
)

echo "${JOB_NAME}: job ${JID} (mode=${MODE}, epochs=${MAX_EPOCHS}, wall=${WALL})"
echo "Output: ${OUTPUT_DIR}"
