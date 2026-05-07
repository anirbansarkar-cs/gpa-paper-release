#!/bin/bash
#SBATCH --job-name=k562_mdlm_gpa
#SBATCH --output=sbatch_out/k562_mdlm_gpa/%j_stdout.out
#SBATCH --error=sbatch_out/k562_mdlm_gpa/%j_stderr.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --qos=fast
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anonymous@example.com

# ===========================================================================
# GPA + DPS with LentiMPRA MDLM + K562 LegNet Oracle
# ===========================================================================

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p sbatch_out/k562_mdlm_gpa

# ---- Paths ----------------------------------------------------------------
SGDD_DIR="${SGDD_DIR:-${HOME}/SGDD}"
MDLM_CKPT="${MDLM_CKPT:-${SGDD_DIR}/applications/drakes_dna/data_and_model/mdlm/outputs_lentimpra/2026.03.10/000831/checkpoints/best.ckpt}"
ORACLE_CKPT="${ORACLE_CKPT:-${GPA_REPO_ROOT}/model_zoo/lentimpra/oracle_models/best_model-epoch=24-val_pearson=0.814.ckpt}"

# ---- Oracle type -----------------------------------------------------------
ORACLE_TYPE="${ORACLE_TYPE:-legnet}"  # legnet or alphagenome
AG_SERVER_BATCH_SIZE="${AG_SERVER_BATCH_SIZE:-64}"

# ---- Seed pool -------------------------------------------------------------
SEED_POOL="${SEED_POOL:-}"

# ---- GPA parameters --------------------------------------------------------
POPULATION_SIZE="${POPULATION_SIZE:-5000}"
MAX_BETA="${MAX_BETA:-200.0}"
ESS_THRESHOLD="${ESS_THRESHOLD:-0.5}"
MAX_STEPS="${MAX_STEPS:-30}"

# ---- Mutation parameters ---------------------------------------------------
NOISE_FRACTION="${NOISE_FRACTION:-0.05}"
MUTATION_BATCH_SIZE="${MUTATION_BATCH_SIZE:-128}"

# ---- DPS parameters --------------------------------------------------------
USE_DPS="${USE_DPS:-false}"
DPS_ETA="${DPS_ETA:-3000.0}"
DPS_TAU="${DPS_TAU:-1.0}"

# ---- DPS eta annealing -----------------------------------------------------
ETA_START="${ETA_START:-0.0}"
ETA_END="${ETA_END:-0.0}"

# ---- Diversity enforcement (duplicate re-mutation) -------------------------
DEDUP_THRESHOLD="${DEDUP_THRESHOLD:-0}"
DEDUP_NF_MULT="${DEDUP_NF_MULT:-2.0}"

# ---- Grammar-preserving DPS -----------------------------------------------
DPS_MODE="${DPS_MODE:-standard}"
KL_BUDGET="${KL_BUDGET:-1.0}"
KL_MODE="${KL_MODE:-per_position}"

# ---- Per-step edit budget --------------------------------------------------
MAX_EDITS_PER_STEP="${MAX_EDITS_PER_STEP:-0}"

# ---- Hill-climb ------------------------------------------------------------
HILL_CLIMB="${HILL_CLIMB:-false}"
HILL_CLIMB_BUDGET="${HILL_CLIMB_BUDGET:-0}"
HILL_CLIMB_FULL_VOCAB="${HILL_CLIMB_FULL_VOCAB:-false}"
HILL_CLIMB_POSITIONS="${HILL_CLIMB_POSITIONS:-0}"

# ---- MCTS ------------------------------------------------------------------
MCTS_DEPTH="${MCTS_DEPTH:-0}"
MCTS_ITERATIONS="${MCTS_ITERATIONS:-12}"
MCTS_C="${MCTS_C:-1.41}"

# ---- GC fitness penalty ----------------------------------------------------
GC_FITNESS_WEIGHT="${GC_FITNESS_WEIGHT:-0.0}"
GC_FITNESS_LOW="${GC_FITNESS_LOW:-0.40}"
GC_FITNESS_HIGH="${GC_FITNESS_HIGH:-0.60}"

# ---- NF annealing ----------------------------------------------------------
NF_START="${NF_START:-0.0}"
NF_END="${NF_END:-0.0}"

# ---- Adaptive NF boost ----------------------------------------------------
NF_BOOST_DELTA="${NF_BOOST_DELTA:-0.0}"
NF_BOOST_MAX="${NF_BOOST_MAX:-0.0}"

# ---- GC-aware DPS ----------------------------------------------------------
GC_DPS_WEIGHT="${GC_DPS_WEIGHT:-0.0}"
GC_DPS_TARGET="${GC_DPS_TARGET:-0.50}"

# ---- Direct GC pull --------------------------------------------------------
GC_PULL_WEIGHT="${GC_PULL_WEIGHT:-0.0}"
GC_PULL_TARGET="${GC_PULL_TARGET:-0.50}"

# ---- Elitism ----------------------------------------------------------------
ELITE_FRACTION="${ELITE_FRACTION:-0.0}"

# ---- Bio filter -------------------------------------------------------------
BIO_FILTER="${BIO_FILTER:-}"
GC_LOW="${GC_LOW:-0.40}"
GC_HIGH="${GC_HIGH:-0.70}"

# ---- GC-stratified resampling -----------------------------------------------
GC_BIN_EDGES="${GC_BIN_EDGES:-}"
GC_BIN_MIN_QUOTA="${GC_BIN_MIN_QUOTA:-50}"
GC_BIN_ALLOC_MODE="${GC_BIN_ALLOC_MODE:-raw}"
GC_BIN_ALLOC_ALPHA="${GC_BIN_ALLOC_ALPHA:-5.0}"

# ---- Adaptive continuation -------------------------------------------------
EXTEND_ON_DELTA_BETA="${EXTEND_ON_DELTA_BETA:-0.5}"
EXTEND_STEPS="${EXTEND_STEPS:-10}"
HARD_MAX_STEPS="${HARD_MAX_STEPS:-200}"
START_BETA="${START_BETA:-0.0}"

# ---- EDTS -------------------------------------------------------------------
BRANCH_FACTOR="${BRANCH_FACTOR:-1}"
MAX_EDIT_FRAC="${MAX_EDIT_FRAC:-1.0}"
EDIT_SELECT_MODE="${EDIT_SELECT_MODE:-efficiency}"

# ---- Initialization --------------------------------------------------------
TOP_K_INIT="${TOP_K_INIT:-false}"
FROM_RANDOM="${FROM_RANDOM:-false}"
GC_BALANCED_INIT="${GC_BALANCED_INIT:-false}"

# ---- Reproducibility -------------------------------------------------------
SEED="${SEED:-}"

# ---- AlphaGenome eval oracle ------------------------------------------------
EVAL_AG="${EVAL_AG:-false}"
AG_BATCH_SIZE="${AG_BATCH_SIZE:-64}"
EVAL_CHECKPOINT_INTERVAL="${EVAL_CHECKPOINT_INTERVAL:-5}"
EVAL_EARLY_STOP_PATIENCE="${EVAL_EARLY_STOP_PATIENCE:-0}"
ARCHIVE_THRESHOLD="${ARCHIVE_THRESHOLD:-}"

# ---- v7 optimization flags -------------------------------------------------
CACHE_MUTATION_SCORES="${CACHE_MUTATION_SCORES:-false}"
ENTROPY_HC="${ENTROPY_HC:-false}"
PREFILTER_TOPK="${PREFILTER_TOPK:-0}"
GPA_FITNESS_MODE="${GPA_FITNESS_MODE:-oracle}"
PLL_WARMUP_STEPS="${PLL_WARMUP_STEPS:-0}"
DIVERSITY_LAMBDA="${DIVERSITY_LAMBDA:-0.0}"
SURROGATE_PREFILTER="${SURROGATE_PREFILTER:-false}"
CASCADE_ORACLE="${CASCADE_ORACLE:-false}"

# ---- Lagrangian multi-objective fitness ------------------------------------
USE_LAGRANGIAN="${USE_LAGRANGIAN:-false}"
LAGRANGIAN_GC_TARGET="${LAGRANGIAN_GC_TARGET:-0.50}"
LAGRANGIAN_GC_TOLERANCE="${LAGRANGIAN_GC_TOLERANCE:-0.05}"
LAGRANGIAN_LR="${LAGRANGIAN_LR:-0.1}"

# ---- Diagnostics -----------------------------------------------------------
DIVERSITY_SUBSAMPLE="${DIVERSITY_SUBSAMPLE:-500}"

# ---- Output ----------------------------------------------------------------
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/k562_mdlm_gpa/run_${RUN_TAG}}"

# ===========================================================================
# Build command
# ===========================================================================
CMD=(
    python scripts/k562_mdlm_gpa/run_k562_mdlm_gpa.py
    --mdlm_checkpoint "$MDLM_CKPT"
    --oracle_type "$ORACLE_TYPE"
    --oracle_checkpoint "$ORACLE_CKPT"
    --ag_server_batch_size "$AG_SERVER_BATCH_SIZE"
    --sgdd_dir "$SGDD_DIR"
    --population_size "$POPULATION_SIZE"
    --max_beta "$MAX_BETA"
    --ess_threshold "$ESS_THRESHOLD"
    --max_steps "$MAX_STEPS"
    --noise_fraction "$NOISE_FRACTION"
    --mutation_batch_size "$MUTATION_BATCH_SIZE"
    --dps_eta "$DPS_ETA"
    --dps_tau "$DPS_TAU"
    --eta_start "$ETA_START"
    --eta_end "$ETA_END"
    --dedup_threshold "$DEDUP_THRESHOLD"
    --dedup_nf_mult "$DEDUP_NF_MULT"
    --dps_mode "$DPS_MODE"
    --kl_budget "$KL_BUDGET"
    --kl_mode "$KL_MODE"
    --max_edits_per_step "$MAX_EDITS_PER_STEP"
    --hill_climb_budget "$HILL_CLIMB_BUDGET"
    --hill_climb_positions "$HILL_CLIMB_POSITIONS"
    --mcts_depth "$MCTS_DEPTH"
    --mcts_iterations "$MCTS_ITERATIONS"
    --mcts_c "$MCTS_C"
    --elite_fraction "$ELITE_FRACTION"
    --gc_fitness_weight "$GC_FITNESS_WEIGHT"
    --gc_fitness_low "$GC_FITNESS_LOW"
    --gc_fitness_high "$GC_FITNESS_HIGH"
    --nf_start "$NF_START"
    --nf_end "$NF_END"
    --nf_boost_delta "$NF_BOOST_DELTA"
    --nf_boost_max "$NF_BOOST_MAX"
    --gc_dps_weight "$GC_DPS_WEIGHT"
    --gc_dps_target "$GC_DPS_TARGET"
    --gc_pull_weight "$GC_PULL_WEIGHT"
    --gc_pull_target "$GC_PULL_TARGET"
    --extend_on_delta_beta "$EXTEND_ON_DELTA_BETA"
    --extend_steps "$EXTEND_STEPS"
    --hard_max_steps "$HARD_MAX_STEPS"
    --start_beta "$START_BETA"
    --max_delta_beta "${MAX_DELTA_BETA:-0}"
    --diversity_subsample "$DIVERSITY_SUBSAMPLE"
    --branch_factor "$BRANCH_FACTOR"
    --max_edit_frac "$MAX_EDIT_FRAC"
    --edit_select_mode "$EDIT_SELECT_MODE"
    --prefilter_topk "$PREFILTER_TOPK"
    --gpa_fitness_mode "$GPA_FITNESS_MODE"
    --pll_warmup_steps "$PLL_WARMUP_STEPS"
    --diversity_lambda "$DIVERSITY_LAMBDA"
    --output_dir "$OUTPUT_DIR"
)

if [[ "$FROM_RANDOM" == "true" ]]; then
    CMD+=(--from_random)
elif [[ -n "$SEED_POOL" ]]; then
    CMD+=(--seed_pool "$SEED_POOL")
else
    echo "ERROR: Must set FROM_RANDOM=true or SEED_POOL=<path>"
    exit 1
fi

if [[ "$TOP_K_INIT" == "true" ]]; then
    CMD+=(--top_k_init)
fi

if [[ "$GC_BALANCED_INIT" == "true" ]]; then
    CMD+=(--gc_balanced_init)
fi

if [[ -n "$BIO_FILTER" ]]; then
    CMD+=(--bio_filter --gc_low "$GC_LOW" --gc_high "$GC_HIGH")
fi

if [[ "$USE_DPS" == "true" ]]; then
    CMD+=(--use_dps)
fi

if [[ "$HILL_CLIMB" == "true" ]]; then
    CMD+=(--hill_climb)
fi

if [[ "$HILL_CLIMB_FULL_VOCAB" == "true" ]]; then
    CMD+=(--hill_climb_full_vocab)
fi

if [[ "$CACHE_MUTATION_SCORES" == "true" ]]; then
    CMD+=(--cache_mutation_scores)
fi

if [[ "$ENTROPY_HC" == "true" ]]; then
    CMD+=(--entropy_hc)
fi

if [[ "$SURROGATE_PREFILTER" == "true" ]]; then
    CMD+=(--surrogate_prefilter)
fi

if [[ "$CASCADE_ORACLE" == "true" ]]; then
    CMD+=(--cascade_oracle)
fi

if [[ "$EVAL_AG" == "true" ]]; then
    CMD+=(--eval_ag --ag_batch_size "$AG_BATCH_SIZE"
          --eval_checkpoint_interval "$EVAL_CHECKPOINT_INTERVAL"
          --eval_early_stop_patience "$EVAL_EARLY_STOP_PATIENCE")
    if [[ -n "$ARCHIVE_THRESHOLD" ]]; then
        CMD+=(--archive_threshold "$ARCHIVE_THRESHOLD")
    fi
fi

if [[ -n "$SEED" ]]; then
    CMD+=(--seed "$SEED")
fi

if [[ -n "$GC_BIN_EDGES" ]]; then
    CMD+=(--gc_bin_edges "$GC_BIN_EDGES" --gc_bin_min_quota "$GC_BIN_MIN_QUOTA"
          --gc_bin_alloc_mode "$GC_BIN_ALLOC_MODE" --gc_bin_alloc_alpha "$GC_BIN_ALLOC_ALPHA")
fi

if [[ "$USE_LAGRANGIAN" == "true" ]]; then
    CMD+=(--use_lagrangian
          --lagrangian_gc_target "$LAGRANGIAN_GC_TARGET"
          --lagrangian_gc_tolerance "$LAGRANGIAN_GC_TOLERANCE"
          --lagrangian_lr "$LAGRANGIAN_LR")
fi

# ===========================================================================
# Run
# ===========================================================================
echo "========================================================================"
echo "GPA + DPS -- K562 LentiMPRA MDLM"
echo "========================================================================"
echo "SGDD dir:         $SGDD_DIR"
echo "Oracle type:      $ORACLE_TYPE"
echo "MDLM checkpoint:  $MDLM_CKPT"
echo "Oracle checkpoint: $ORACLE_CKPT"
echo "Population:       $POPULATION_SIZE"
echo "Max beta:         $MAX_BETA"
echo "Noise fraction:   $NOISE_FRACTION"
echo "DPS:              $USE_DPS (eta=$DPS_ETA)"
echo "GC fitness:       w=$GC_FITNESS_WEIGHT [$GC_FITNESS_LOW, $GC_FITNESS_HIGH]"
echo "Bio filter:       ${BIO_FILTER:-off}"
echo "From random:      $FROM_RANDOM"
echo "Seed pool:        ${SEED_POOL:-none}"
echo "Output dir:       $OUTPUT_DIR"
echo "Command: ${CMD[*]}"
echo "========================================================================"

"${CMD[@]}"
