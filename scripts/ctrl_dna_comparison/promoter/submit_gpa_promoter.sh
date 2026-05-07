#!/bin/bash
# Submit GPA + HyenaDNA promoter run (smoke or full) for one target cell / seed.
# Usage:
#   bash submit_gpa_promoter.sh [smoke|full] <task> [seed] [--dps]
# Examples:
#   bash submit_gpa_promoter.sh smoke JURKAT 0
#   bash submit_gpa_promoter.sh full K562 0 --dps
set -euo pipefail

MODE="${1:-smoke}"
TASK="${2:-JURKAT}"
SEED="${3:-0}"
DPS_FLAG=""
for arg in "$@"; do
    [[ "${arg}" == "--dps" ]] && DPS_FLAG="--use_dps"
done
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
    echo "Usage: $0 [smoke|full] <task> [seed] [--dps]" >&2
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
# Canonical Stage 1 = 3 epochs. Override via STAGE1_CKPT to use the e15
# variant as an ablation arm (GPA may benefit from a denser prior).
CKPT="${STAGE1_CKPT:-${PROMOTER_DIR}/checkpoints/hyenadna_promoter_e3/best.ckpt}"
if [[ ! -e "${CKPT}" ]]; then
    echo "ERROR: ${CKPT} not found — run submit_train_hyenadna.sh e3 first" >&2
    exit 1
fi

DPS_TAG="nodps"
[[ -n "${DPS_FLAG}" ]] && DPS_TAG="dps"
# Optional suffix for recipe ablations (e.g. TAG=_pw035, TAG=_pw150).
SUFFIX="${TAG:-}"
OUT_DIR="results/ctrl_dna_comparison/promoter/gpa_${MODE}_${TASK}_${DPS_TAG}_seed${SEED}${SUFFIX}"
mkdir -p sbatch_out/ctrl_dna_comparison "${OUT_DIR}"

if [[ "${MODE}" == "smoke" ]]; then
    POP=500; MAX_BETA=10; MAX_STEPS=8; WALL="02:00:00"; MUT_BS=128
    QOS="qos_long"
    PW="${PW:-0.0}"
    ETA="${ETA:-3000}"
else
    # GPA full: ~30 mutation/resample steps. Enhancer GPA-HyenaDNA runs
    # completed in well under 6H. Use `default` QOS (12H MaxWall) so bio_ai
    # stays free for the 48H Ctrl-DNA runs.
    # Winning enhancer recipe (ctrldna_vs_gpa_final_tables.md): DPS + pw=0.35
    # + eta=3000. Override with PW/ETA env vars if needed.
    POP="${POP:-5000}"; MAX_BETA="${MAX_BETA:-50}"; MAX_STEPS="${MAX_STEPS:-30}"
    WALL="${WALL:-06:00:00}"; MUT_BS=256
    QOS="${QOS_OVERRIDE:-default}"
    PW="${PW:-0.35}"
    ETA="${ETA:-3000}"
fi

# Diversity knobs (Phase 2 — env-var overrides)
NOISE_FRACTION="${NF:-0.10}"
ESS_THRESHOLD="${ESS:-0.5}"
MAX_COPIES="${MAX_COPIES:-0}"
DIV_LAMBDA="${DIV_LAMBDA:-0.0}"
DEDUP_THRESHOLD="${DEDUP_THRESHOLD:-0}"
REJUV_FRAC="${REJUV_FRAC:-0.0}"
MUT_SUBSTEPS="${MUT_SUBSTEPS:-1}"
TAU_START="${TAU_START:-1.0}"
BRANCH_FACTOR="${BRANCH_FACTOR:-1}"
BRANCH_FACTOR_SCHEDULE="${BRANCH_FACTOR_SCHEDULE:-}"   # e.g. linear_8_1, late_8_1_80, step_8_4_2_1
HILL_CLIMB_BUDGET="${HILL_CLIMB_BUDGET:-0}"
BRANCH_TAU="${BRANCH_TAU:-0.0}"
NF_START="${NF_START:-0.0}"
NF_END="${NF_END:-0.0}"
# SMC-soundness corrections: APF (S5) for K-branch τ-softmax selection,
# DPS IW (S7) for gradient-warped proposal. 0=off, 1=on (pass bare --flag).
APF_CORRECTION="${APF_CORRECTION:-0}"
DPS_IW_CORRECT="${DPS_IW_CORRECT:-0}"
APF_FLAG=""; [[ "${APF_CORRECTION}" == "1" ]] && APF_FLAG="--apf_correction"
DPS_IW_FLAG=""; [[ "${DPS_IW_CORRECT}" == "1" ]] && DPS_IW_FLAG="--dps_iw_correct"
BFS_FLAG=""; [[ -n "${BRANCH_FACTOR_SCHEDULE}" ]] && BFS_FLAG="--branch_factor_schedule ${BRANCH_FACTOR_SCHEDULE}"
# Bio-filter override. BIO_FILTER_OFF=1 -> pass --no_bio_filter (mutation-stage
# GC filter disabled; archive capture is already target-threshold-only).
BIO_FILTER_OFF="${BIO_FILTER_OFF:-0}"
GC_LOW="${GC_LOW:-0.45}"
GC_HIGH="${GC_HIGH:-0.55}"
BIO_FLAG="--bio_filter --gc_low ${GC_LOW} --gc_high ${GC_HIGH}"
[[ "${BIO_FILTER_OFF}" == "1" ]] && BIO_FLAG="--no_bio_filter"
# Archive: per-step collection of seqs exceeding target threshold.
# Cell-specific defaults when ARCHIVE_THRESHOLD unset and ARCHIVE_ENABLED=1.
# JURKAT: high range (~7 max), K562: mid range (~6), THP1: low range (~4).
if [[ "${ARCHIVE_ENABLED:-0}" == "1" && -z "${ARCHIVE_THRESHOLD:-}" ]]; then
    case "${TASK}" in
        JURKAT) ARCHIVE_THRESHOLD=5.0 ;;
        K562)   ARCHIVE_THRESHOLD=4.0 ;;
        THP1)   ARCHIVE_THRESHOLD=2.5 ;;
    esac
fi
ARCHIVE_FLAG=""
[[ -n "${ARCHIVE_THRESHOLD:-}" ]] && ARCHIVE_FLAG="--archive_threshold ${ARCHIVE_THRESHOLD}"

JOB_NAME="gpa_promoter_${MODE}_${TASK}_${DPS_TAG}_s${SEED}${SUFFIX}"
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

python ${PROMOTER_DIR}/run_gpa_hyenadna_promoter.py \
    --hyenadna_checkpoint ${CKPT} \
    --oracle_ckpt_dir ${PROMOTER_DIR}/checkpoints \
    --seed_csv ${PROMOTER_DIR}/data/seeds_arbitrary_set${SEED}.csv \
    --target_cell ${TASK} \
    --population_size ${POP} \
    --max_beta ${MAX_BETA} \
    --max_steps ${MAX_STEPS} \
    --ess_threshold ${ESS_THRESHOLD} \
    --noise_fraction ${NOISE_FRACTION} \
    --mutation_substeps ${MUT_SUBSTEPS} \
    --tau_start ${TAU_START} \
    --mutation_batch_size ${MUT_BS} \
    --penalty_weight ${PW} \
    --eta ${ETA} \
    --max_copies ${MAX_COPIES} \
    --diversity_lambda ${DIV_LAMBDA} \
    --dedup_threshold ${DEDUP_THRESHOLD} \
    --rejuvenation_fraction ${REJUV_FRAC} \
    --branch_factor ${BRANCH_FACTOR} \
    ${BFS_FLAG} \
    --hill_climb_budget ${HILL_CLIMB_BUDGET} \
    --branch_selection_tau ${BRANCH_TAU} \
    --nf_start ${NF_START} \
    --nf_end ${NF_END} \
    ${APF_FLAG} \
    ${DPS_IW_FLAG} \
    ${BIO_FLAG} \
    --seed ${SEED} \
    ${ARCHIVE_FLAG} \
    --output_dir ${OUT_DIR} ${DPS_FLAG}
JOB_EOF
)
echo "${JOB_NAME}: job ${JID}  (out: ${OUT_DIR})"
