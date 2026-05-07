#!/bin/bash
# UCB-MCTS scaling sweep on K562 — anchors the cost curve for the DNA-CRAFT
# runtime narrative (see dnacraft_pilot_runtime_and_structural_comparison.md).
#
# Two axes, each at three new points (plus the existing pilot anchor at i=12, K=8):
#   Iterations sweep (K=8 fixed, d=2):  i ∈ {18, 24, 36}
#   Branching sweep  (i=12 fixed, d=2): K ∈ {16, 24, 32}
# 2 seeds per point on K562 (cleanest variance in pilot) → 12 jobs total.
#
# Routing (per user 2026-05-03):
#   - 11 jobs → gpuq with --nodelist=gpunode20,21,24,25,26,27,28 (free of AG JAX work)
#   - 1 job  → partition_b + qos_long QOS (exactly one slot free)
#     The partition_b job is the longest projected (branch K=32 i=12 s42, ~90 min).
#
# Wall: 4h fixed (well above any projected runtime; under fast QOS cap of 4h).
# Memory: 4 CPU × 3 GB = 12 GB (matches pilot).
set -euo pipefail

PROJECT_DIR="${GPA_REPO_ROOT}"
cd "$PROJECT_DIR"

mkdir -p sbatch_out/rerd_comparison results/rerd_comparison
TMPDIR_SCRIPTS=$(mktemp -d)

SVDD_DIR="${HOME}/SVDD"
MDLM_CKPT="${SVDD_DIR}/artifacts/DNA_Diffusion:v0/last.ckpt"
ORACLE_CKPT="${SVDD_DIR}/artifacts/DRAKES_oracles/reward_oracle_ft.ckpt"
EVAL_CKPT="${SVDD_DIR}/artifacts/DRAKES_oracles/reward_oracle_eval.ckpt"
SEED_FILE="results/rerd_comparison/gosai_seeds_k562_gc45_55.h5"
NODELIST="gpunode20,gpunode21,gpunode24,gpunode25,gpunode26,gpunode27,gpunode28"

# standard GPA recipe args (same as pilot ucb_stacked)
C3_ARGS="--use_dps --dps_eta 3000 --dps_penalty_weight 0.30 --penalty_weight 0.30 --max_beta 25 --gc_dps_target 0.60 --gc_dps_weight 10"

# Round-robin QOS for gpuq jobs
QOS_LIST=(fast default qos_long bio_ai)
QOS_IDX=0
next_gpuq_qos() {
    local q="${QOS_LIST[$((QOS_IDX % ${#QOS_LIST[@]}))]}"
    QOS_IDX=$((QOS_IDX + 1))
    echo "$q"
}

# Args: TAG BRANCH_FACTOR MCTS_ITER PARTITION QOS NODELIST_ARG
submit_job() {
    local TAG="$1"
    local BRANCH_FACTOR="$2"
    local MCTS_ITER="$3"
    local PARTITION="$4"
    local QOS="$5"
    local NODELIST_ARG="$6"

    local OUTDIR="results/rerd_comparison/${TAG}"
    if [ -f "${OUTDIR}/gpa_output.h5" ]; then
        echo "  [SKIP] ${TAG} (output exists)"
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
    --target_cell k562 \\
    --population_size 5000 \\
    --max_steps 30 \\
    --noise_fraction 0.10 \\
    --seed ${SEED} \\
    --seed_pool ${SEED_FILE} \\
    --top_k_init \\
    --branch_factor ${BRANCH_FACTOR} \\
    --mcts_depth 2 \\
    --mcts_iterations ${MCTS_ITER} \\
    --mcts_select_mode ucb \\
    --output_dir ${OUTDIR} \\
    ${C3_ARGS}
EOF

    local JID
    JID=$(sbatch --parsable \
        --partition="${PARTITION}" \
        --qos="${QOS}" \
        --time=04:00:00 \
        --job-name="${TAG}" \
        --output="sbatch_out/rerd_comparison/${TAG}_%j.out" \
        --error="sbatch_out/rerd_comparison/${TAG}_%j.err" \
        ${NODELIST_ARG} \
        "${SCRIPT}" 2>&1) || { echo "  [ERROR] sbatch failed for ${TAG}: ${JID}"; return 1; }
    printf '  [%-7s %-13s] %-55s : %s\n' "${PARTITION}" "${QOS}" "${TAG}" "${JID}"
}

SEEDS=(42 1042)

echo "============================================================================="
echo "UCB-MCTS scaling sweep — K562, 12 jobs"
echo "  Iterations: i ∈ {18, 24, 36}, K=8, d=2"
echo "  Branching:  K ∈ {16, 24, 32}, i=12, d=2"
echo "  Routing: 11 → gpuq nodelist=${NODELIST}; 1 → partition_b/qos_long (longest job)"
echo "============================================================================="

# ── Iterations sweep (6 jobs, all gpuq nodelist) ────────────────────────
echo
echo "--- Iterations sweep (K=8, d=2) ---"
for I in 18 24 36; do
    for SEED in "${SEEDS[@]}"; do
        TAG="dnacraft_ucb_iter_i${I}_K8_d2_k562_s${SEED}"
        QOS=$(next_gpuq_qos)
        export SEED
        submit_job "${TAG}" 8 "${I}" gpuq "${QOS}" "--nodelist=${NODELIST}"
    done
done

# ── Branching sweep (6 jobs total: 5 gpuq nodelist + 1 partition_b/qos_long) ─────
echo
echo "--- Branching sweep (i=12, d=2) ---"
for K in 16 24 32; do
    for SEED in "${SEEDS[@]}"; do
        TAG="dnacraft_ucb_branch_K${K}_i12_d2_k562_s${SEED}"
        export SEED
        # Send the heaviest job (K=32 s42) to the partition_b/qos_long free slot
        if [ "${K}" = "32" ] && [ "${SEED}" = "42" ]; then
            submit_job "${TAG}" "${K}" 12 partition_b qos_long ""
        else
            QOS=$(next_gpuq_qos)
            submit_job "${TAG}" "${K}" 12 gpuq "${QOS}" "--nodelist=${NODELIST}"
        fi
    done
done

rm -rf "${TMPDIR_SCRIPTS}"
echo
echo "============================================================================="
echo "Submission complete."
echo "Outputs → results/rerd_comparison/dnacraft_ucb_{iter,branch}_*/"
echo "============================================================================="
