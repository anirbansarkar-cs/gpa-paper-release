#!/bin/bash
# Timing-only run: full-run hyperparameters (epoch=5, batch=128) but
# max_iter=20 so we can measure per-iter wall and extrapolate to R200.
# Submits a single (cell, seed) pair.
#   bash submit_ctrldna_promoter_timing.sh [task] [seed] [max_iter]
# Defaults: K562, seed 0, max_iter=20.
set -euo pipefail

TASK="${1:-K562}"
SEED="${2:-0}"
MAX_ITER="${3:-20}"
if [[ "${TASK}" != "JURKAT" && "${TASK}" != "K562" && "${TASK}" != "THP1" ]]; then
    echo "Task must be JURKAT, K562, or THP1" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

PROMOTER_DIR="scripts/ctrl_dna_comparison/promoter"
# Match the e15 Stage 1 ckpt — same as the 15 completed full runs we are sanity-checking.
CKPT="${STAGE1_CKPT:-${PROMOTER_DIR}/checkpoints/hyenadna_promoter_e15/best.ckpt}"
if [[ ! -e "${CKPT}" ]]; then
    echo "ERROR: ${CKPT} not found" >&2
    exit 1
fi

OUT_DIR="results/ctrl_dna_comparison/promoter/ctrldna_timing_${TASK}_seed${SEED}_e15"
JOB_NAME="ctrldna_promoter_timing_${TASK}_s${SEED}"
mkdir -p sbatch_out/ctrl_dna_comparison "${OUT_DIR}"

JID=$(sbatch --parsable <<JOB_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=sbatch_out/ctrl_dna_comparison/${JOB_NAME}_%j.out
#SBATCH --error=sbatch_out/ctrl_dna_comparison/${JOB_NAME}_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --qos=fast
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --exclude=gpunode29
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=anonymous@example.com

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118
set -euo pipefail
cd ${PROJECT_DIR}

# Iteration-level timestamps for per-iter timing.
export PYTHONUNBUFFERED=1
START=\$(date +%s)
echo "TIMING_START \${START}"

python ${PROMOTER_DIR}/run_ctrldna_promoter.py --hyenadna_checkpoint ${CKPT} --oracle_ckpt_dir ${PROMOTER_DIR}/checkpoints --oracle_ranges ${PROMOTER_DIR}/data/oracle_ranges.json --seed_csv ${PROMOTER_DIR}/data/seeds_${TASK}.csv --task ${TASK} --max_iter ${MAX_ITER} --epoch 5 --batch_size 128 --checkpoint_interval 0 --seed ${SEED} --out_dir ${OUT_DIR}

END=\$(date +%s)
echo "TIMING_END \${END}"
echo "TIMING_TOTAL_SEC \$((END-START))"
echo "TIMING_PER_ITER_SEC \$(python -c \"print((\${END}-\${START})/${MAX_ITER})\")"
echo "TIMING_EXTRAPOLATED_R200_HOURS \$(python -c \"print((\${END}-\${START})/${MAX_ITER}*200/3600)\")"
JOB_EOF
)
echo "${JOB_NAME}: job ${JID}  (out: ${OUT_DIR})"
