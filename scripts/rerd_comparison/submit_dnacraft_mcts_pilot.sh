#!/bin/bash
# DNA-CRAFT MCTS pilot — 3 arms × 3 cells × 5 seeds = 45 jobs.
#
# Arm A (mcts_d2_ucb_stacked):     standard GPA recipe + UCB-MCTS d=2 i=12 K=8.
#                                  Heuristic proposal improvement matched to DNA-CRAFT.
# Arm B (mcts_d2_uniform_stacked): standard GPA recipe + uniform-MCTS d=2 i=9 K=8.
#                                  Target-preserved (Prop. K, K_eff = K^D).
# Arm C (mcts_d2_uniform_dialed):  matched-aggressiveness — max_beta=10, no DPS,
#                                  max_steps=12, K=8 + uniform-MCTS d=2 i=9.
#
# Resource sizing (tight, justified):
#   - Baseline standard recipe: ~3 min wall, ~7 GB MaxRSS (sacct 1907112-1907124).
#   - MCTS d=2 overhead: 3-4× per-step (empirical from K562 March MCTS runs).
#   - Stacked arms (max_steps=30): ~12 min expected → 30 min wall.
#   - Dialed arm (max_steps=12, no DPS): ~5 min expected → 20 min wall.
#   - Memory: 4 CPU × 3 GB = 12 GB (~1.5× peak).
#
# QOS routing: round-robin {fast, default, qos_long, bio_ai} — wall fits
# all four. Spreads load when fast hits MaxJobsPU.

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

declare -A SEED_POOL=(
    [hepg2]="results/rerd_comparison/gosai_seeds_hepg2_gc45_55.h5"
    [k562]="results/rerd_comparison/gosai_seeds_k562_gc45_55.h5"
    [sknsh]="results/rerd_comparison/gosai_seeds_sknsh_gc45_55.h5"
)

# ── QOS routing (round-robin across 4) ────────────────────────────────
QOS_LIST=(fast default bio_ai)         # qos_long excluded (full)
QOS_FULL=(0 0 0)                        # 1 = QOS hit submit limit, skip
QOS_IDX=0
SELECTED_QOS=""

# Round-robin over QOS_LIST, skipping any that have been marked full.
# Sets SELECTED_QOS; updates QOS_IDX; returns 1 if all QOS are full.
next_qos() {
    local n=${#QOS_LIST[@]}
    for _ in $(seq 1 $n); do
        local i=$((QOS_IDX % n))
        QOS_IDX=$((QOS_IDX + 1))
        if [ "${QOS_FULL[$i]}" = "0" ]; then
            SELECTED_QOS="${QOS_LIST[$i]}"
            return 0
        fi
    done
    SELECTED_QOS=""
    return 1
}

# Mark current QOS as full so future calls skip it.
mark_qos_full() {
    local q="$1"
    local n=${#QOS_LIST[@]}
    for i in $(seq 0 $((n - 1))); do
        if [ "${QOS_LIST[$i]}" = "$q" ]; then
            QOS_FULL[$i]=1
            echo "  [INFO] marked QOS $q as full; rotating to remaining"
            return
        fi
    done
}

# ── Submit one job ───────────────────────────────────────────────────
# Args: ARM_NAME CELL SEED EXTRA_RECIPE_ARGS WALL MCTS_ENV...
submit_job() {
    local ARM="$1"
    local CELL="$2"
    local SEED="$3"
    local WALL="$4"
    local RECIPE_ARGS="$5"
    local MCTS_DEPTH_V="$6"
    local MCTS_ITER_V="$7"
    local MCTS_MODE_V="$8"
    local MAX_STEPS_V="$9"

    local TAG="dnacraft_pilot_${ARM}_${CELL}_s${SEED}"
    local OUTDIR="results/rerd_comparison/${TAG}"

    if [ -f "${OUTDIR}/gpa_output.h5" ]; then
        echo "  [SKIP] ${TAG} (output exists)"
        return
    fi

    local SEED_FILE="${SEED_POOL[$CELL]}"

    if [ ! -f "${SEED_FILE}" ]; then
        echo "  [ERROR] missing seed pool: ${SEED_FILE}"
        return
    fi

    local SCRIPT="${TMPDIR_SCRIPTS}/${TAG}.sh"

    cat > "${SCRIPT}" <<'SCRIPT_EOF'
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
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
    --target_cell ${CELL} \\
    --population_size 5000 \\
    --max_steps ${MAX_STEPS_V} \\
    --noise_fraction 0.10 \\
    --seed ${SEED} \\
    --seed_pool ${SEED_FILE} \\
    --top_k_init \\
    --branch_factor 8 \\
    --mcts_depth ${MCTS_DEPTH_V} \\
    --mcts_iterations ${MCTS_ITER_V} \\
    --mcts_select_mode ${MCTS_MODE_V} \\
    --output_dir ${OUTDIR} \\
    ${RECIPE_ARGS}
EOF

    # Retry submit on QOSMaxSubmitJobPerUserLimit by rotating QOS.
    local JID="" QOS="" SBATCH_ERR
    while next_qos; do
        QOS="$SELECTED_QOS"
        SBATCH_ERR=$(sbatch --parsable \
            --partition=gpuq \
            --qos="${QOS}" \
            --time="${WALL}" \
            --job-name="${TAG}" \
            --output="sbatch_out/rerd_comparison/${TAG}_%j.out" \
            --error="sbatch_out/rerd_comparison/${TAG}_%j.err" \
            "${SCRIPT}" 2>&1) || true
        if [[ "$SBATCH_ERR" =~ QOSMaxSubmitJobPerUserLimit ]]; then
            mark_qos_full "${QOS}"
            continue
        fi
        if [[ "$SBATCH_ERR" =~ ^[0-9]+$ ]]; then
            JID="$SBATCH_ERR"
            break
        fi
        echo "  [ERROR] sbatch failed for ${TAG} on ${QOS}: ${SBATCH_ERR}"
        return 1
    done
    if [ -z "$JID" ]; then
        echo "  [ERROR] all QOS exhausted; cannot submit ${TAG}"
        return 1
    fi
    printf '  [%-13s] %-50s : %s\n' "${QOS}" "${TAG}" "${JID}"
}

# ── Recipe args ──────────────────────────────────────────────────────
# standard GPA recipe (no bio_filter; GC pull via DPS; max_beta=25)
C3_ARGS="--use_dps --dps_eta 3000 --dps_penalty_weight 0.30 --penalty_weight 0.30 --max_beta 25 --gc_dps_target 0.60 --gc_dps_weight 10"

# Dialed-down (no DPS, max_beta=10, no GC pull)
DIALED_ARGS="--penalty_weight 0.30 --max_beta 10"

CELLS=(hepg2 k562 sknsh)
SEEDS=(42 1042 2042 3042 4042)

echo "============================================================================="
echo "DNA-CRAFT MCTS pilot — 3 arms × 3 cells × 5 seeds = 45 jobs"
echo "QOS rotation: ${QOS_LIST[*]}"
echo "============================================================================="

# Arm A: UCB-MCTS stacked on standard GPA
echo
echo "--- Arm A: mcts_d2_ucb_stacked ((GPA + UCB-MCTS d=2 i=12)) ---"
for CELL in "${CELLS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "ucb_stacked" "$CELL" "$SEED" "01:30:00" "$C3_ARGS" 2 12 ucb 30
    done
done

# Arm B: uniform-MCTS stacked on standard GPA
echo
echo "--- Arm B: mcts_d2_uniform_stacked ((GPA + uniform-MCTS d=2 i=9)) ---"
for CELL in "${CELLS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "uniform_stacked" "$CELL" "$SEED" "01:30:00" "$C3_ARGS" 2 9 uniform 30
    done
done

# Arm C: dialed-down (matched aggressiveness)
echo
echo "--- Arm C: mcts_d2_uniform_dialed (max_beta=10, no DPS, max_steps=12) ---"
for CELL in "${CELLS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "uniform_dialed" "$CELL" "$SEED" "00:30:00" "$DIALED_ARGS" 2 9 uniform 12
    done
done

rm -rf "${TMPDIR_SCRIPTS}"
echo
echo "============================================================================="
echo "Submission complete."
echo "Outputs → results/rerd_comparison/dnacraft_pilot_*/gpa_output*.h5"
echo "============================================================================="
