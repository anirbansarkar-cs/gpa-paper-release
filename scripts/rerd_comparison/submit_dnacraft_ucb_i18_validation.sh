#!/bin/bash
# i=18 ucb_stacked validation across all 3 cells × 5 seeds.
# 13 new jobs (K562 s42/s1042 already done at i=18 from the scaling sweep).
# Same recipe as pilot ucb_stacked but i=18 instead of i=12.
#
# Routing: nodelist=bamgpu20,21,24,25,26,27,28 + bamgpu19,bamgpu100 fallback
# (h100 nodes user doesn't use for AG JAX). 1 job to kooq/koolab if available.
set -euo pipefail

PROJECT_DIR="${GPA_REPO_ROOT}"
cd "$PROJECT_DIR"
mkdir -p sbatch_out/rerd_comparison
TMPDIR_SCRIPTS=$(mktemp -d)

SVDD_DIR="${HOME}/SVDD"
MDLM_CKPT="${SVDD_DIR}/artifacts/DNA_Diffusion:v0/last.ckpt"
ORACLE_CKPT="${SVDD_DIR}/artifacts/DRAKES_oracles/reward_oracle_ft.ckpt"
EVAL_CKPT="${SVDD_DIR}/artifacts/DRAKES_oracles/reward_oracle_eval.ckpt"

declare -A SEED_POOL=(
    [hepg2]="results/rerd_comparison/gosai_seeds_hepg2_gc45_55.h5"
    [k562]="results/rerd_comparison/gosai_seeds_k562_gc45_55.h5"
    [sknsh]="results/rerd_comparison/gosai_seeds_sknsh_gc45_55.h5"
)

NODELIST="bamgpu20,bamgpu21,bamgpu24,bamgpu25,bamgpu26,bamgpu27,bamgpu28,bamgpu19,bamgpu100"

C3_ARGS="--use_dps --dps_eta 3000 --dps_penalty_weight 0.30 --penalty_weight 0.30 --max_beta 25 --gc_dps_target 0.60 --gc_dps_weight 10"

# Args: TAG CELL SEED PART QOS NL_ARG GRES_ARG
submit_job() {
    local TAG="$1" CELL="$2" SEED="$3" PART="$4" QOS="$5" NL_ARG="$6" GRES_ARG="$7"
    local OUTDIR="results/rerd_comparison/${TAG}"
    if [ -f "${OUTDIR}/gpa_output.h5" ]; then echo "  [SKIP] ${TAG}"; return; fi
    local SEED_FILE="${SEED_POOL[$CELL]}"
    local SCRIPT="${TMPDIR_SCRIPTS}/${TAG}.sh"

    # Build SBATCH script. h100 constraint applied via GRES type for kooq;
    # via #SBATCH --constraint=h100 for gpuq.
    cat > "${SCRIPT}" <<SBATCH_EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=${GRES_ARG}
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=anonymous@example.com

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118
set -euo pipefail
SBATCH_EOF

    # Add h100 constraint only for gpuq (kooq uses gres type instead)
    if [ "${PART}" = "gpuq" ]; then
        sed -i '/#SBATCH --gres=/a #SBATCH --constraint=h100' "${SCRIPT}"
    fi

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
    --max_steps 30 \\
    --noise_fraction 0.10 \\
    --seed ${SEED} \\
    --seed_pool ${SEED_FILE} \\
    --top_k_init \\
    --branch_factor 8 \\
    --mcts_depth 2 \\
    --mcts_iterations 18 \\
    --mcts_select_mode ucb \\
    --output_dir ${OUTDIR} \\
    ${C3_ARGS}
EOF

    local JID
    JID=$(sbatch --parsable \
        --partition="${PART}" \
        --qos="${QOS}" \
        --time=04:00:00 \
        --job-name="${TAG}" \
        --output="sbatch_out/rerd_comparison/${TAG}_%j.out" \
        --error="sbatch_out/rerd_comparison/${TAG}_%j.err" \
        ${NL_ARG} \
        "${SCRIPT}" 2>&1) || { echo "  [ERROR] sbatch failed for ${TAG}: ${JID}"; return 1; }
    printf '  [%-7s %-13s] %-50s : %s\n' "${PART}" "${QOS}" "${TAG}" "${JID}"
}

NL_ARG="--nodelist=${NODELIST}"
GRES_GPUQ="gpu:1"
GRES_KOOQ="gpu:h100:1"

# Check if kooq has an available slot (no pending koolab jobs of mine, and at
# least one free GPU slot). If yes, send the first HepG2 job there.
USE_KOOQ_FOR_FIRST=0
if [ -z "$(squeue -u $(whoami) -p kooq -h 2>/dev/null)" ]; then
    KOOQ_FREE=$(scontrol show node bamgpu101 2>/dev/null | grep -oE "AllocTRES=.*gres/gpu=[0-9]+" | grep -oE "[0-9]+$" || echo 4)
    if [ "${KOOQ_FREE}" -lt "4" ]; then
        USE_KOOQ_FOR_FIRST=1
        echo "  kooq has free slot (alloc=${KOOQ_FREE}/4) — first HepG2 job → kooq"
    fi
fi

# 13 jobs: HepG2 ×5 + K562 ×3 (s2042/3042/4042 only; s42/s1042 done) + SKNSH ×5
# QOS rotation: explicit per-job to avoid the subshell bug from before
declare -a JOBS=(
    "dnacraft_ucb_iter_i18_K8_d2_hepg2_s42|hepg2|42|default"
    "dnacraft_ucb_iter_i18_K8_d2_hepg2_s1042|hepg2|1042|koolab_shared"
    "dnacraft_ucb_iter_i18_K8_d2_hepg2_s2042|hepg2|2042|bio_ai"
    "dnacraft_ucb_iter_i18_K8_d2_hepg2_s3042|hepg2|3042|default"
    "dnacraft_ucb_iter_i18_K8_d2_hepg2_s4042|hepg2|4042|koolab_shared"
    "dnacraft_ucb_iter_i18_K8_d2_k562_s2042|k562|2042|bio_ai"
    "dnacraft_ucb_iter_i18_K8_d2_k562_s3042|k562|3042|default"
    "dnacraft_ucb_iter_i18_K8_d2_k562_s4042|k562|4042|koolab_shared"
    "dnacraft_ucb_iter_i18_K8_d2_sknsh_s42|sknsh|42|bio_ai"
    "dnacraft_ucb_iter_i18_K8_d2_sknsh_s1042|sknsh|1042|default"
    "dnacraft_ucb_iter_i18_K8_d2_sknsh_s2042|sknsh|2042|koolab_shared"
    "dnacraft_ucb_iter_i18_K8_d2_sknsh_s3042|sknsh|3042|bio_ai"
    "dnacraft_ucb_iter_i18_K8_d2_sknsh_s4042|sknsh|4042|default"
)

idx=0
for entry in "${JOBS[@]}"; do
    IFS='|' read -r TAG CELL SEED QOS <<< "${entry}"
    if [ "${idx}" -eq 0 ] && [ "${USE_KOOQ_FOR_FIRST}" -eq 1 ]; then
        submit_job "${TAG}" "${CELL}" "${SEED}" kooq koolab "" "${GRES_KOOQ}"
    else
        submit_job "${TAG}" "${CELL}" "${SEED}" gpuq "${QOS}" "${NL_ARG}" "${GRES_GPUQ}"
    fi
    idx=$((idx + 1))
done

rm -rf "${TMPDIR_SCRIPTS}"
echo "Done."
