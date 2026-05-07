#!/bin/bash
# ===========================================================================
# GPA v3t pool_A — 3-seed completion runs
# ===========================================================================
# After the pool_A noDPS pick was switched from v3e_softB → v3t (2026-05-04),
# v3t pool_A multi-seed coverage needs filling in to match pool_B's 3-seed
# grid.
#
# Existing on disk:
#   v3t pool_A seed=42  × 5 caps (cap10/20/30/50/uncap) — original baseline
#   v3t pool_A seed=43  × uncap-only — from run_gpa_v3_seed_verify.sh (job 2015776)
#
# Needed for full 3-seed × 5-cap grid:
#   v3t pool_A × seed=43 × {cap10, cap20, cap30, cap50}        = 4 jobs
#   v3t pool_A × seed=44 × {cap10, cap20, cap30, cap50, uncap} = 5 jobs
#   total = 9 jobs
#
# All runs use AG_STAGE=stage2 to stay consistent with paper plot values.
# Submit only after user says "go".
# Skip-if-output-exists guard handles re-runs.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p sbatch_out/gpa_v3

V2_ROOT="results/ledidi_comparison/gpa_vs_ism_ledidi_v2"
SEED_DIR="${V2_ROOT}/seed_pools"
OUT_ROOT="${V2_ROOT}/gpa_v3"

SGDD_DIR="${HOME}/SGDD"
MDLM_CKPT="${SGDD_DIR}/applications/drakes_dna/data_and_model/mdlm/outputs_lentimpra/2026.03.10/000831/checkpoints/best.ckpt"
ORACLE_CKPT="${GPA_REPO_ROOT}/model_zoo/lentimpra/oracle_models/best_model-epoch=24-val_pearson=0.814.ckpt"

WALL="${WALL_OVERRIDE:-04:00:00}"
QOS="${FORCE_QOS:-fast}"
# JAX AG-stage2 needs cuDNN-capable H100 driver (≥555). Exclude the driver-545
# nodes per reference_jax_alphagenome_cluster_setup.md. This is a hard infra
# constraint, NOT covered by feedback_no_gpu_exclusions.md.
EXCLUDE="--exclude=gpunode20,gpunode21,gpunode24,gpunode25,gpunode26,gpunode27,gpunode28,gpunode29"

SHARED_BASE=(
    SGDD_DIR="$SGDD_DIR" MDLM_CKPT="$MDLM_CKPT" ORACLE_CKPT="$ORACLE_CKPT"
    ORACLE_TYPE=legnet POPULATION_SIZE=5000
    HILL_CLIMB=false HILL_CLIMB_BUDGET=0
    CACHE_MUTATION_SCORES=true
    MAX_DELTA_BETA=6.67 HARD_MAX_STEPS=400 EXTEND_STEPS=10 EXTEND_ON_DELTA_BETA=0.5
    EVAL_AG=true EVAL_CHECKPOINT_INTERVAL=5 EVAL_EARLY_STOP_PATIENCE=10
    ARCHIVE_THRESHOLD=3.0
    AG_STAGE=stage2
)

# v3t_nodps_nobio_nogc_b2k: noDPS, BF=10, β=2000, NF=0.05, no bio_filter, no GC-DPS
v3t_args=(
    USE_DPS=false
    BRANCH_FACTOR=10 NOISE_FRACTION=0.05 MAX_BETA=2000.0
)

CAPS=("cap10:0.10" "cap20:0.20" "cap30:0.30" "cap50:0.50" "uncap:1.00")

declare -a JOBS
# seed=43: 4 caps (uncap already exists from job 2015776; skip-if-exists guard catches it)
for c in "${CAPS[@]}"; do
    JOBS+=("pool_A:${SEED_DIR}/pool_A_random_5k.h5:43:${c}")
done
# seed=44: all 5 caps
for c in "${CAPS[@]}"; do
    JOBS+=("pool_A:${SEED_DIR}/pool_A_random_5k.h5:44:${c}")
done

n=0
for spec in "${JOBS[@]}"; do
    IFS=':' read -r pool_tag pool_path seed cap_tag cap_frac <<< "$spec"
    full_recipe="v3t_nodps_nobio_nogc_b2k"
    out_name="${full_recipe}_${pool_tag}_${cap_tag}_seed${seed}_stage2"
    out_dir="${OUT_ROOT}/${out_name}"
    if [[ -f "${out_dir}/gpa_output_pool.h5" ]] && [[ "${FORCE:-0}" != "1" ]]; then
        echo "[skip $n] ${out_name} done"
        n=$((n+1)); continue
    fi
    if squeue -u "$USER" -h --name="gpa_v3_${out_name}" --format="%i" 2>/dev/null | grep -q .; then
        echo "[skip $n] ${out_name} queued"
        n=$((n+1)); continue
    fi
    CMD=(env "${SHARED_BASE[@]}" "${v3t_args[@]}"
        SEED="$seed"
        SEED_POOL="$pool_path" MAX_EDIT_FRAC="$cap_frac"
        RUN_TAG="${out_name}" OUTPUT_DIR="$out_dir"
        sbatch
        --job-name="gpa_v3_${out_name}"
        --output="sbatch_out/gpa_v3/${out_name}_%j.out"
        --error="sbatch_out/gpa_v3/${out_name}_%j.err"
        --time="$WALL"
        --qos="$QOS" --partition=gpuq --gres=gpu:1
        --constraint=h100 $EXCLUDE
        --ntasks=1 --cpus-per-task=6 --mem=60G
        --parsable
        scripts/k562_mdlm_gpa/run_k562_mdlm_gpa.sh)
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY [$n/$QOS] ${out_name}"
    else
        jid=$("${CMD[@]}")
        echo "  [$n/$QOS] ${out_name} -> Job ${jid}"
    fi
    n=$((n+1))
done
echo ""
echo "Submitted v3t pool_A 3-seed completion jobs (total iterated: $n)"
