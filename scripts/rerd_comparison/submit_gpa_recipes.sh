#!/bin/bash
# Phase C v2 — GPA recipe sweep targeting JASPAR/3-mer recovery (post-Phase-A re-plan).
# 28 unique recipes × 3 reps = 84 jobs.
# Routing: {default, qos_long, fast} only — bio_ai/qos_long excluded per user 2026-04-29.
# Wall-time fit:
#   - bf=1, max_beta ≤ 50           → fast (4h wall) eligible
#   - bf=1, max_beta ≥ 75 OR bf > 1 → default (12h) or qos_long (2d)
set -euo pipefail

PROJECT_DIR="${GPA_REPO_ROOT}"
cd "$PROJECT_DIR"

mkdir -p sbatch_out/rerd_comparison results/rerd_comparison
TMPDIR_SCRIPTS=$(mktemp -d)

# ── Common paths ──────────────────────────────────────────────────────
SVDD_DIR="${HOME}/SVDD"
MDLM_CKPT="${SVDD_DIR}/artifacts/DNA_Diffusion:v0/last.ckpt"
ORACLE_CKPT="${SVDD_DIR}/artifacts/DRAKES_oracles/reward_oracle_ft.ckpt"
EVAL_CKPT="${SVDD_DIR}/artifacts/DRAKES_oracles/reward_oracle_eval.ckpt"
SEED_POOL_NATURAL="results/rerd_comparison/gosai_seeds_hepg2_gc45_55.h5"

# ── QOS rotation ──────────────────────────────────────────────────────
# Tracks slot index per QOS; routing picks the first QOS in preference list
# that the job's wall-time can fit into.
declare -A SLOT_COUNT=( [default]=0 [qos_long]=0 [fast]=0 )
declare -A QOS_TIME=( [default]="05:00:00" [qos_long]="07:00:00" [fast]="04:00:00" )

# Pick QOS based on whether the job fits in fast (≤3h50m wall).
# Round-robin across eligible QOSes.
choose_qos() {
    local FITS_FAST="$1"   # "yes" or "no"
    if [ "${FITS_FAST}" = "yes" ]; then
        # cycle across all 3
        local TOTAL=$((SLOT_COUNT[default] + SLOT_COUNT[qos_long] + SLOT_COUNT[fast]))
        local idx=$((TOTAL % 3))
        case $idx in
            0) echo "default" ;;
            1) echo "qos_long" ;;
            2) echo "fast" ;;
        esac
    else
        # cycle across default and qos_long only
        local TOTAL=$((SLOT_COUNT[default] + SLOT_COUNT[qos_long]))
        local idx=$((TOTAL % 2))
        case $idx in
            0) echo "default" ;;
            1) echo "qos_long" ;;
        esac
    fi
}

# ── Submit one job ───────────────────────────────────────────────────
submit_job() {
    local NAME="$1"          # human-readable recipe name (e.g. dps_pw030_b50)
    local REP="$2"           # rep id
    local FITS_FAST="$3"     # "yes" or "no"
    shift 3
    local EXTRA_ARGS="$@"

    local TAG="run_${NAME}_r${REP}"
    local OUTDIR="results/rerd_comparison/${TAG}"

    # Skip if already complete
    if [ -f "${OUTDIR}/gpa_output.h5" ]; then
        echo "  [SKIP] ${TAG} (output exists)"
        return
    fi

    local QOS=$(choose_qos "${FITS_FAST}")
    local TIME="${QOS_TIME[$QOS]}"
    SLOT_COUNT[$QOS]=$((SLOT_COUNT[$QOS] + 1))

    local SCRIPT="${TMPDIR_SCRIPTS}/${TAG}.sh"

    cat > "${SCRIPT}" <<'SCRIPT_EOF'
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=gpunode29
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=anonymous@example.com

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118
set -euo pipefail
SCRIPT_EOF

    cat >> "${SCRIPT}" << EOF
cd ${PROJECT_DIR}
mkdir -p ${OUTDIR}

python scripts/rerd_comparison/run_rerd_gpa.py \\
    --backbone mdlm \\
    --mdlm_checkpoint ${MDLM_CKPT} \\
    --oracle_checkpoint ${ORACLE_CKPT} \\
    --eval_oracle_checkpoint ${EVAL_CKPT} \\
    --eval_checkpoint_interval 1 \\
    --archive_threshold 7.0 \\
    --target_cell hepg2 \\
    --population_size 5000 \\
    --max_steps 30 \\
    --noise_fraction 0.10 \\
    --seed \$(( ${REP} * 1000 + 42 )) \\
    --output_dir ${OUTDIR} \\
    ${EXTRA_ARGS}
EOF

    JID=$(sbatch --parsable \
        --partition=gpuq \
        --qos="${QOS}" \
        --time="${TIME}" \
        --job-name="${TAG}" \
        --output="sbatch_out/rerd_comparison/${TAG}_%j.out" \
        --error="sbatch_out/rerd_comparison/${TAG}_%j.err" \
        "${SCRIPT}")
    printf '  [%-13s] %-50s rep=%s : %s\n' "${QOS}" "${NAME}" "${REP}" "${JID}"
}

# ── Recipe matrix ────────────────────────────────────────────────────
# Paper Table 2: SMC family on Gosai-HepG2, GPA β∈{25, 50}.
# Each call: NAME REP FITS_FAST EXTRA_ARGS

REPS=(1 2 3)

echo "============================================================================="
echo "Phase C — GPA recipes for paper Table 2 (β∈{25,50}, 3 reps)"
echo "============================================================================="

# DPS Mean + pw=0.30 + max_beta ∈ {25, 50} + bio on
for B in 25 50; do
    for R in "${REPS[@]}"; do
        submit_job "dps_pw030_b${B}" "$R" "yes" \
            "--use_dps --dps_eta 3000 --dps_penalty_weight 0.30 --penalty_weight 0.30 --max_beta ${B} --bio_filter --gc_low 0.45 --gc_high 0.55 --seed_pool ${SEED_POOL_NATURAL} --top_k_init"
    done
done
# noDPS Mean + pw=0.30 + max_beta ∈ {25, 50} + bio on
for B in 25 50; do
    for R in "${REPS[@]}"; do
        submit_job "nodps_pw030_b${B}" "$R" "yes" \
            "--penalty_weight 0.30 --max_beta ${B} --bio_filter --gc_low 0.45 --gc_high 0.55 --seed_pool ${SEED_POOL_NATURAL} --top_k_init"
    done
done
# DPS Mean + pw=0.00 + max_beta ∈ {25, 50} + bio on
for B in 25 50; do
    for R in "${REPS[@]}"; do
        submit_job "dps_pw000_b${B}" "$R" "yes" \
            "--use_dps --dps_eta 3000 --dps_penalty_weight 0.00 --penalty_weight 0.00 --max_beta ${B} --bio_filter --gc_low 0.45 --gc_high 0.55 --seed_pool ${SEED_POOL_NATURAL} --top_k_init"
    done
done

rm -rf "${TMPDIR_SCRIPTS}"
echo
echo "============================================================================="
echo "Phase C submission complete."
echo "Slot counts: default=${SLOT_COUNT[default]}  qos_long=${SLOT_COUNT[qos_long]}  fast=${SLOT_COUNT[fast]}"
echo "Outputs → results/rerd_comparison/run_*/gpa_output*.h5"
