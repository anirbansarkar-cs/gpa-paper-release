#!/bin/bash
# DNA-CRAFT N-sweep ablation: standard GPA (no MCTS) at varying population sizes.
# Tests the SMC O(1/sqrt(N)) convergence claim cited in main.tex:119 / app A.1
# at the same recipe used for the Table 2 headline (recipe args, MDLM backbone).
#
# Sweep:
#   N in {128, 500, 1000, 2500, 5000, 10000, 20000}    (7 points)
#   cells in {hepg2, k562, sknsh}                       (3 cells)
#   seeds in {42, 1042, 2042, 3042, 4042}               (5 seeds, matches paper)
#
# Total: 7 * 3 * 5 = 105 jobs. K-branch+DPS only, mcts_depth=0 (no MCTS).
# Per-job runtime estimate (~3.6 min/seed at N=5000 from main.tex:280 paragraph):
#   N=128 ~10s, N=500 ~22s, N=1k ~45s, N=2.5k ~110s, N=5k ~3.6min,
#   N=10k ~7min, N=20k ~14min. Pad 1.5-2x per paper-push wall-time policy.
#
# Routing: all jobs fit under 4h wall -> rotate across 4 QOS
#   {fast, default, koolab_shared, bio_ai}. Spreads load so 'fast' (which
#   drains fastest at this small per-job size) doesn't bottleneck.
#   koolab QOS deliberately excluded per user. No GPU node exclusions.
# Run output: results/rerd_comparison/dnacraft_nsweep_n{N}_{cell}_s{SEED}/
# Score with run_batch_score.py --pattern 'dnacraft_nsweep_*' --source pool

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

# standard GPA recipe (identical to submit_dnacraft_ucb_i18_validation.sh, MCTS dropped).
C3_ARGS="--use_dps --dps_eta 3000 --dps_penalty_weight 0.30 --penalty_weight 0.30 --max_beta 25 --gc_dps_target 0.60 --gc_dps_weight 10"

CELLS=(hepg2 k562 sknsh)
SEEDS=(42 1042 2042 3042 4042)
N_VALUES=(128 500 1000 2500 5000 10000 20000)

# Wall-time per N (generous: ~5x typical runtime, 30-min floor).
declare -A WALL_BY_N=(
    [128]="00:30:00"
    [500]="00:30:00"
    [1000]="00:30:00"
    [2500]="00:30:00"
    [5000]="01:00:00"
    [10000]="02:00:00"
    [20000]="03:30:00"
)

# QOS rotation across 4 gpuq QOS. fast (4h) drains fastest; the 2-day pair
# (koolab_shared, bio_ai) and default (12h) cushion the rest. All 4 live on
# gpuq with --constraint=h100, so no per-QOS partition/GRES branching needed.
declare -A SLOT_COUNT=(
    [fast]=0 [default]=0 [koolab_shared]=0 [bio_ai]=0
)
QOS_ORDER=(fast)
choose_qos() {
    local TOTAL=0
    for q in "${QOS_ORDER[@]}"; do
        TOTAL=$((TOTAL + SLOT_COUNT[$q]))
    done
    local idx=$((TOTAL % ${#QOS_ORDER[@]}))
    echo "${QOS_ORDER[$idx]}"
}

submit_job() {
    local N="$1" CELL="$2" SEED="$3"
    local TAG="dnacraft_nsweep_n${N}_${CELL}_s${SEED}"
    local OUTDIR="results/rerd_comparison/${TAG}"
    if [ -f "${OUTDIR}/gpa_output.h5" ]; then echo "  [SKIP-DONE] ${TAG}"; return; fi
    # Avoid duplicate sbatch on rerun: skip if a job with this name is already
    # PENDING/RUNNING in the queue.
    if grep -Fxq "${TAG}" "${ACTIVE_NAMES_FILE}" 2>/dev/null; then
        echo "  [SKIP-QUEUED] ${TAG}"; return
    fi

    local SEED_FILE="${SEED_POOL[$CELL]}"
    local WALL="${WALL_BY_N[$N]}"
    local QOS=$(choose_qos)
    SLOT_COUNT[$QOS]=$((SLOT_COUNT[$QOS] + 1))
    local PART="gpuq"

    local SCRIPT="${TMPDIR_SCRIPTS}/${TAG}.sh"

    cat > "${SCRIPT}" <<SBATCH_EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=anonymous@example.com

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118
set -euo pipefail

cd ${PROJECT_DIR}
mkdir -p ${OUTDIR}

python scripts/rerd_comparison/run_rerd_gpa.py \\
    --backbone mdlm \\
    --mdlm_checkpoint ${MDLM_CKPT} \\
    --oracle_checkpoint ${ORACLE_CKPT} \\
    --eval_oracle_checkpoint ${EVAL_CKPT} \\
    --eval_checkpoint_interval 1 \\
    --mingap_archive_size 64 \\
    --perstep_archive_top_k 64 \\
    --target_cell ${CELL} \\
    --population_size ${N} \\
    --max_steps 30 \\
    --noise_fraction 0.10 \\
    --seed ${SEED} \\
    --seed_pool ${SEED_FILE} \\
    --top_k_init \\
    --branch_factor 8 \\
    --output_dir ${OUTDIR} \\
    ${C3_ARGS}
SBATCH_EOF

    local JID
    JID=$(sbatch --parsable \
        --partition="${PART}" \
        --qos="${QOS}" \
        --time="${WALL}" \
        --job-name="${TAG}" \
        --output="sbatch_out/rerd_comparison/${TAG}_%j.out" \
        --error="sbatch_out/rerd_comparison/${TAG}_%j.err" \
        "${SCRIPT}" 2>/dev/null) || JID=""
    if [ -n "${JID}" ]; then
        echo "  [${QOS} ${WALL}] ${JID}  ${TAG}"
    else
        # Roll back the slot count so rotation stays balanced
        SLOT_COUNT[$QOS]=$((SLOT_COUNT[$QOS] - 1))
        FAILED_JOBS+=("${N} ${CELL} ${SEED}")
        echo "  [FAIL ${QOS}] ${TAG} (QOS limit?)"
    fi
}

FAILED_JOBS=()
ACTIVE_NAMES_FILE=$(mktemp)
trap 'rm -f "${ACTIVE_NAMES_FILE}"' EXIT
squeue -u $(whoami) -h -o "%j" 2>/dev/null | grep "^dnacraft_nsweep_" > "${ACTIVE_NAMES_FILE}" || true

echo "=== DNA-CRAFT N-sweep (no MCTS) ==="
echo "N values: ${N_VALUES[*]}"
echo "Cells:    ${CELLS[*]}"
echo "Seeds:    ${SEEDS[*]}"
echo "Total:    $((${#N_VALUES[@]} * ${#CELLS[@]} * ${#SEEDS[@]})) jobs"
echo

for N in "${N_VALUES[@]}"; do
    for CELL in "${CELLS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            submit_job "${N}" "${CELL}" "${SEED}"
        done
    done
done

echo
echo "QOS distribution:"
for q in "${QOS_ORDER[@]}"; do
    echo "  ${q}: ${SLOT_COUNT[$q]}"
done
echo "Failed (QOS-limit overflow): ${#FAILED_JOBS[@]}"
if [ "${#FAILED_JOBS[@]}" -gt 0 ]; then
    {
        for spec in "${FAILED_JOBS[@]}"; do
            read -r N_FAIL CELL_FAIL SEED_FAIL <<< "${spec}"
            echo "n${N_FAIL}_${CELL_FAIL}_s${SEED_FAIL}"
        done
    } > sbatch_out/rerd_comparison/dnacraft_nsweep_failed_jobs.txt
    printf '  %s\n' "${FAILED_JOBS[@]}"
    echo "  -> failed list: sbatch_out/rerd_comparison/dnacraft_nsweep_failed_jobs.txt"
    echo "  -> rerun this script after queue drains; existence guard skips completed runs"
fi
echo "Sbatch tmpdir: ${TMPDIR_SCRIPTS}"
echo "Score later with:"
echo "  python scripts/rerd_comparison/dnacraft/run_batch_score.py \\"
echo "      --pattern 'dnacraft_nsweep_*' --source pool \\"
echo "      --selections by_composite --cells HepG2 K562 SKNSH --workers 16"
