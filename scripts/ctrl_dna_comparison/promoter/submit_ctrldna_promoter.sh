#!/bin/bash
# Submit Ctrl-DNA promoter run (smoke, full, or resume) for one target cell / seed.
# Usage:
#   bash submit_ctrldna_promoter.sh [smoke|full|resume] <task> [seed]
# Examples:
#   bash submit_ctrldna_promoter.sh smoke JURKAT 0
#   bash submit_ctrldna_promoter.sh full K562 3
#   bash submit_ctrldna_promoter.sh resume K562 3    # picks up checkpoint_sigterm.pt
#
# Time budget note: enhancer R200 took ~48H. Promoter is 250bp (vs 200bp)
# so per-iter cost is ~25% higher — R200 may TIMEOUT on the 48H wall.
# If that happens, the SIGTERM handler writes checkpoint_sigterm.pt; rerun
# with the `resume` mode to continue from that checkpoint.
set -euo pipefail

MODE="${1:-smoke}"
TASK="${2:-JURKAT}"
SEED="${3:-0}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" && "${MODE}" != "resume" ]]; then
    echo "Usage: $0 [smoke|full|resume] <task> [seed]" >&2
    exit 1
fi
if [[ "${TASK}" != "JURKAT" && "${TASK}" != "K562" && "${TASK}" != "THP1" ]]; then
    echo "Task must be JURKAT, K562, or THP1" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

PROMOTER_DIR="scripts/ctrl_dna_comparison/promoter"
# Canonical Stage 1 = 3 epochs (matches Ctrl-DNA paper). Override with
# env var STAGE1_CKPT to use the archived 15-epoch variant or any other.
CKPT="${STAGE1_CKPT:-${PROMOTER_DIR}/checkpoints/hyenadna_promoter_e3/best.ckpt}"
if [[ ! -e "${CKPT}" ]]; then
    echo "ERROR: ${CKPT} not found — run submit_train_hyenadna.sh e3 first" >&2
    exit 1
fi

# Optional suffix for ablation arms (e.g. TAG=_e15 for the 15-epoch Stage 1 arm).
SUFFIX="${TAG:-}"
if [[ "${MODE}" == "resume" ]]; then
    OUT_DIR="results/ctrl_dna_comparison/promoter/ctrldna_full_${TASK}_seed${SEED}${SUFFIX}"
    RESUME_CKPT="${OUT_DIR}/checkpoint_sigterm.pt"
    if [[ ! -e "${RESUME_CKPT}" ]]; then
        RESUME_CKPT=$(ls -t "${OUT_DIR}"/checkpoint_round*.pt 2>/dev/null | head -1)
    fi
    if [[ -z "${RESUME_CKPT}" || ! -e "${RESUME_CKPT}" ]]; then
        echo "ERROR: no resume ckpt found in ${OUT_DIR}" >&2
        exit 1
    fi
    echo "Resuming from: ${RESUME_CKPT}"
    RESUME_FLAG="--resume_from ${RESUME_CKPT}"
    MAX_ITER=200; EPOCH=5; BATCH=128; WALL="48:00:00"; CKPT_INT=50
    QOS="bio_ai"
    JOB_NAME="ctrldna_promoter_resume_${TASK}_s${SEED}${SUFFIX}"
else
    OUT_DIR="results/ctrl_dna_comparison/promoter/ctrldna_${MODE}_${TASK}_seed${SEED}${SUFFIX}"
    RESUME_FLAG=""
    if [[ "${MODE}" == "smoke" ]]; then
        MAX_ITER=10; EPOCH=3; BATCH=32; WALL="02:00:00"; CKPT_INT=0
        QOS="qos_long"
    else
        MAX_ITER=200; EPOCH=5; BATCH=128; WALL="48:00:00"; CKPT_INT=50
        QOS="bio_ai"
    fi
    JOB_NAME="ctrldna_promoter_${MODE}_${TASK}_s${SEED}${SUFFIX}"
fi
mkdir -p sbatch_out/ctrl_dna_comparison "${OUT_DIR}"
JID=$(sbatch --parsable <<JOB_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=sbatch_out/ctrl_dna_comparison/${JOB_NAME}_%j.out
#SBATCH --error=sbatch_out/ctrl_dna_comparison/${JOB_NAME}_%j.err
#SBATCH --time=${WALL}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --qos=${QOS}
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=anonymous@example.com

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118
set -euo pipefail
cd ${PROJECT_DIR}

python ${PROMOTER_DIR}/run_ctrldna_promoter.py \
    --hyenadna_checkpoint ${CKPT} \
    --oracle_ckpt_dir ${PROMOTER_DIR}/checkpoints \
    --oracle_ranges ${PROMOTER_DIR}/data/oracle_ranges.json \
    --seed_csv ${PROMOTER_DIR}/data/seeds_${TASK}.csv \
    --task ${TASK} \
    --max_iter ${MAX_ITER} \
    --epoch ${EPOCH} \
    --batch_size ${BATCH} \
    --checkpoint_interval ${CKPT_INT} \
    --seed ${SEED} \
    --out_dir ${OUT_DIR} ${RESUME_FLAG}
JOB_EOF
)
echo "${JOB_NAME}: job ${JID}  (out: ${OUT_DIR})"
