#!/bin/bash
#SBATCH --job-name=rerd_gpa
#SBATCH --output=sbatch_out/rerd_comparison/%j_stdout.out
#SBATCH --error=sbatch_out/rerd_comparison/%j_stderr.out
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anonymous@example.com

# ===========================================================================
# GPA + DPS with SVDD MDLM + Enformer Oracle (RERD Comparison)
# ===========================================================================

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate d3_cuda118

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p sbatch_out/rerd_comparison

# ---- Backbone selection ---------------------------------------------------
BACKBONE="${BACKBONE:-mdlm}"

# ---- Paths ----------------------------------------------------------------
SVDD_DIR="${SVDD_DIR:-${HOME}/SVDD}"
SGDD_DIR="${SGDD_DIR:-${HOME}/SGDD}"
MDLM_CKPT="${MDLM_CKPT:-${SVDD_DIR}/artifacts/DNA_Diffusion:v0/last.ckpt}"
SEDD_U_CKPT="${SEDD_U_CKPT:-${SVDD_DIR}/artifacts/sedd_u_dna/dna_uniform}"
ORACLE_CKPT="${ORACLE_CKPT:-${SVDD_DIR}/artifacts/DNA_evaluation:v0/model.ckpt}"
EVAL_ORACLE_CKPT="${EVAL_ORACLE_CKPT:-}"

# ---- Seed pool -------------------------------------------------------------
SEED_POOL="${SEED_POOL:-results/rerd_comparison/gosai_seeds_hepg2.h5}"

# ---- Cell type specificity -------------------------------------------------
TARGET_CELL="${TARGET_CELL:-hepg2}"
PENALTY_WEIGHT="${PENALTY_WEIGHT:-0.5}"
FITNESS_MODE="${FITNESS_MODE:-linear}"
SPECIFICITY_THRESHOLD="${SPECIFICITY_THRESHOLD:-}"

# ---- GPA parameters --------------------------------------------------------
POPULATION_SIZE="${POPULATION_SIZE:-5000}"
MAX_BETA="${MAX_BETA:-50.0}"
ESS_THRESHOLD="${ESS_THRESHOLD:-0.5}"
MAX_STEPS="${MAX_STEPS:-30}"

# ---- Mutation parameters ---------------------------------------------------
NOISE_FRACTION="${NOISE_FRACTION:-0.10}"
MUTATION_BATCH_SIZE="${MUTATION_BATCH_SIZE:-128}"
STEPS="${STEPS:-}"

# ---- DPS parameters --------------------------------------------------------
USE_DPS="${USE_DPS:-false}"
DPS_ETA="${DPS_ETA:-3000.0}"
DPS_TAU="${DPS_TAU:-1.0}"
DPS_REWARD_MODE="${DPS_REWARD_MODE:-fitness}"
DPS_PENALTY_WEIGHT="${DPS_PENALTY_WEIGHT:-0.3}"
DPS_PENALTY_WEIGHT_K="${DPS_PENALTY_WEIGHT_K:-}"
DPS_PENALTY_WEIGHT_S="${DPS_PENALTY_WEIGHT_S:-}"
PENALTY_WEIGHT_K="${PENALTY_WEIGHT_K:-}"
PENALTY_WEIGHT_S="${PENALTY_WEIGHT_S:-}"

# ---- DPS gating (guide_start_frac) -----------------------------------------
GUIDE_START_FRAC="${GUIDE_START_FRAC:-0.0}"

# ---- GC fitness penalty ----------------------------------------------------
GC_FITNESS_WEIGHT="${GC_FITNESS_WEIGHT:-0.0}"
GC_FITNESS_LOW="${GC_FITNESS_LOW:-0.40}"
GC_FITNESS_HIGH="${GC_FITNESS_HIGH:-0.60}"
GC_CENTER_WEIGHT="${GC_CENTER_WEIGHT:-0.0}"

# ---- NF annealing (Approach D) ---------------------------------------------
NF_START="${NF_START:-0.0}"
NF_END="${NF_END:-0.0}"

# ---- GC-aware DPS (Approach A) ---------------------------------------------
GC_DPS_WEIGHT="${GC_DPS_WEIGHT:-0.0}"
GC_DPS_TARGET="${GC_DPS_TARGET:-0.50}"

# ---- Direct GC pull (v13, decoupled from DPS gradient) --------------------
GC_PULL_WEIGHT="${GC_PULL_WEIGHT:-0.0}"
GC_PULL_TARGET="${GC_PULL_TARGET:-0.50}"

# ---- EDTS (Edit-aware Diffusion Tree Search) --------------------------------
BRANCH_FACTOR="${BRANCH_FACTOR:-1}"
MAX_EDIT_FRAC="${MAX_EDIT_FRAC:-1.0}"
EDIT_SELECT_MODE="${EDIT_SELECT_MODE:-efficiency}"

# ---- MCTS (Monte Carlo Tree Search mutation) --------------------------------
MCTS_DEPTH="${MCTS_DEPTH:-0}"
MCTS_ITERATIONS="${MCTS_ITERATIONS:-12}"
MCTS_C="${MCTS_C:-1.41}"
MCTS_SELECT_MODE="${MCTS_SELECT_MODE:-ucb}"

# ---- Per-step edit budget ----------------------------------------------------
MAX_EDITS_PER_STEP="${MAX_EDITS_PER_STEP:-0}"

# ---- Hill-climb (per-position greedy optimization) ---------------------------
HILL_CLIMB="${HILL_CLIMB:-false}"
HILL_CLIMB_BUDGET="${HILL_CLIMB_BUDGET:-0}"
HILL_CLIMB_FULL_VOCAB="${HILL_CLIMB_FULL_VOCAB:-false}"
HILL_CLIMB_POSITIONS="${HILL_CLIMB_POSITIONS:-0}"

# ---- K-mer composition regularizer for HC -----------------------------------
HC_KMER_WEIGHT="${HC_KMER_WEIGHT:-0.0}"
GOSAI_CSV="${GOSAI_CSV:-}"
KMER_TARGET_CELL="${KMER_TARGET_CELL:-}"
KMER_TOP_FRAC="${KMER_TOP_FRAC:-0.10}"

# ---- Edit-efficiency --------------------------------------------------------
EDIT_DISCOUNT="${EDIT_DISCOUNT:-false}"
EDIT_DISCOUNT_MODE="${EDIT_DISCOUNT_MODE:-linear}"
EDIT_DISCOUNT_LAMBDA="${EDIT_DISCOUNT_LAMBDA:-1.0}"
PARETO_EDIT_FITNESS="${PARETO_EDIT_FITNESS:-false}"

# ---- Beta stepping cap ------------------------------------------------------
MAX_DELTA_BETA="${MAX_DELTA_BETA:-0}"

# ---- Grammar-preserving DPS modes -------------------------------------------
DPS_MODE="${DPS_MODE:-standard}"
GATE_SHARPNESS="${GATE_SHARPNESS:-1.0}"
KL_BUDGET="${KL_BUDGET:-1.0}"
KL_MODE="${KL_MODE:-per_position}"

# ---- Reproducibility -------------------------------------------------------
SEED="${SEED:-}"

# ---- Elitism ----------------------------------------------------------------
ELITE_FRACTION="${ELITE_FRACTION:-0.0}"

# ---- Bio filter -------------------------------------------------------------
BIO_FILTER="${BIO_FILTER:-}"
GC_LOW="${GC_LOW:-0.40}"
GC_HIGH="${GC_HIGH:-0.70}"

# ---- GC-stratified resampling (v16) ----------------------------------------
GC_BIN_EDGES="${GC_BIN_EDGES:-}"
GC_BIN_MIN_QUOTA="${GC_BIN_MIN_QUOTA:-50}"
GC_BIN_ALLOC_MODE="${GC_BIN_ALLOC_MODE:-raw}"
GC_BIN_ALLOC_ALPHA="${GC_BIN_ALLOC_ALPHA:-5.0}"

# ---- Adaptive continuation -------------------------------------------------
EXTEND_ON_DELTA_BETA="${EXTEND_ON_DELTA_BETA:-0.5}"
EXTEND_STEPS="${EXTEND_STEPS:-10}"
HARD_MAX_STEPS="${HARD_MAX_STEPS:-200}"
START_BETA="${START_BETA:-0.0}"
EVAL_CHECKPOINT_INTERVAL="${EVAL_CHECKPOINT_INTERVAL:-0}"

# ---- Initialization --------------------------------------------------------
TOP_K_INIT="${TOP_K_INIT:-false}"
FROM_RANDOM="${FROM_RANDOM:-false}"
GC_BALANCED_INIT="${GC_BALANCED_INIT:-false}"

# ---- Output ----------------------------------------------------------------
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/rerd_comparison/run_${RUN_TAG}}"

# ---- Diversity diagnostics -------------------------------------------------
DIVERSITY_SUBSAMPLE="${DIVERSITY_SUBSAMPLE:-500}"

# ===========================================================================
# Build command
# ===========================================================================
CMD=(
    python scripts/rerd_comparison/run_rerd_gpa.py
    --backbone "$BACKBONE"
    --oracle_checkpoint "$ORACLE_CKPT"
    --svdd_dir "$SVDD_DIR"
    --sgdd_dir "$SGDD_DIR"
    --target_cell "$TARGET_CELL"
    --penalty_weight "$PENALTY_WEIGHT"
    --fitness_mode "$FITNESS_MODE"
    --population_size "$POPULATION_SIZE"
    --max_beta "$MAX_BETA"
    --ess_threshold "$ESS_THRESHOLD"
    --max_steps "$MAX_STEPS"
    --noise_fraction "$NOISE_FRACTION"
    --mutation_batch_size "$MUTATION_BATCH_SIZE"
    --dps_eta "$DPS_ETA"
    --dps_tau "$DPS_TAU"
    --dps_reward_mode "$DPS_REWARD_MODE"
    --dps_penalty_weight "$DPS_PENALTY_WEIGHT"
    --guide_start_frac "$GUIDE_START_FRAC"
    --elite_fraction "$ELITE_FRACTION"
    --gc_fitness_weight "$GC_FITNESS_WEIGHT"
    --gc_fitness_low "$GC_FITNESS_LOW"
    --gc_fitness_high "$GC_FITNESS_HIGH"
    --gc_center_weight "$GC_CENTER_WEIGHT"
    --nf_start "$NF_START"
    --nf_end "$NF_END"
    --gc_dps_weight "$GC_DPS_WEIGHT"
    --gc_dps_target "$GC_DPS_TARGET"
    --gc_pull_weight "$GC_PULL_WEIGHT"
    --gc_pull_target "$GC_PULL_TARGET"
    --extend_on_delta_beta "$EXTEND_ON_DELTA_BETA"
    --extend_steps "$EXTEND_STEPS"
    --hard_max_steps "$HARD_MAX_STEPS"
    --start_beta "$START_BETA"
    --eval_checkpoint_interval "$EVAL_CHECKPOINT_INTERVAL"
    --diversity_subsample "$DIVERSITY_SUBSAMPLE"
    --dps_mode "$DPS_MODE"
    --gate_sharpness "$GATE_SHARPNESS"
    --kl_budget "$KL_BUDGET"
    --kl_mode "$KL_MODE"
    --branch_factor "$BRANCH_FACTOR"
    --max_edit_frac "$MAX_EDIT_FRAC"
    --edit_select_mode "$EDIT_SELECT_MODE"
    --mcts_depth "$MCTS_DEPTH"
    --mcts_iterations "$MCTS_ITERATIONS"
    --mcts_c "$MCTS_C"
    --mcts_select_mode "$MCTS_SELECT_MODE"
    --max_edits_per_step "$MAX_EDITS_PER_STEP"
    --hill_climb_budget "$HILL_CLIMB_BUDGET"
    --hill_climb_positions "$HILL_CLIMB_POSITIONS"
    --hc_kmer_weight "$HC_KMER_WEIGHT"
    --kmer_top_frac "$KMER_TOP_FRAC"
    --edit_discount_mode "$EDIT_DISCOUNT_MODE"
    --edit_discount_lambda "$EDIT_DISCOUNT_LAMBDA"
    --max_delta_beta "$MAX_DELTA_BETA"
    --output_dir "$OUTPUT_DIR"
)

if [[ -n "$STEPS" ]]; then
    CMD+=(--steps "$STEPS")
fi

if [[ "$BACKBONE" == "sedd_u" ]]; then
    CMD+=(--sedd_u_checkpoint "$SEDD_U_CKPT")
else
    CMD+=(--mdlm_checkpoint "$MDLM_CKPT")
fi

if [[ "$FROM_RANDOM" == "true" ]]; then
    CMD+=(--from_random)
else
    CMD+=(--seed_pool "$SEED_POOL")
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

if [[ -n "$EVAL_ORACLE_CKPT" ]]; then
    CMD+=(--eval_oracle_checkpoint "$EVAL_ORACLE_CKPT")
fi

if [[ -n "$SPECIFICITY_THRESHOLD" ]]; then
    CMD+=(--specificity_threshold "$SPECIFICITY_THRESHOLD")
fi

if [[ -n "$SEED" ]]; then
    CMD+=(--seed "$SEED")
fi

if [[ -n "$GC_BIN_EDGES" ]]; then
    CMD+=(--gc_bin_edges "$GC_BIN_EDGES" --gc_bin_min_quota "$GC_BIN_MIN_QUOTA"
          --gc_bin_alloc_mode "$GC_BIN_ALLOC_MODE" --gc_bin_alloc_alpha "$GC_BIN_ALLOC_ALPHA")
fi

if [[ "$HILL_CLIMB" == "true" ]]; then
    CMD+=(--hill_climb)
fi

if [[ "$HILL_CLIMB_FULL_VOCAB" == "true" ]]; then
    CMD+=(--hill_climb_full_vocab)
fi

if [[ -n "$GOSAI_CSV" ]]; then
    CMD+=(--gosai_csv "$GOSAI_CSV")
fi

if [[ -n "$KMER_TARGET_CELL" ]]; then
    CMD+=(--kmer_target_cell "$KMER_TARGET_CELL")
fi

if [[ "$EDIT_DISCOUNT" == "true" ]]; then
    CMD+=(--edit_discount)
fi

if [[ "$PARETO_EDIT_FITNESS" == "true" ]]; then
    CMD+=(--pareto_edit_fitness)
fi

if [[ -n "$DPS_PENALTY_WEIGHT_K" ]]; then
    CMD+=(--dps_penalty_weight_k "$DPS_PENALTY_WEIGHT_K")
fi

if [[ -n "$DPS_PENALTY_WEIGHT_S" ]]; then
    CMD+=(--dps_penalty_weight_s "$DPS_PENALTY_WEIGHT_S")
fi

if [[ -n "$PENALTY_WEIGHT_K" ]]; then
    CMD+=(--penalty_weight_k "$PENALTY_WEIGHT_K")
fi

if [[ -n "$PENALTY_WEIGHT_S" ]]; then
    CMD+=(--penalty_weight_s "$PENALTY_WEIGHT_S")
fi

# ===========================================================================
# Run
# ===========================================================================
echo "========================================================================"
echo "GPA + DPS — RERD Comparison"
echo "========================================================================"
echo "Backbone:         $BACKBONE"
echo "SVDD dir:         $SVDD_DIR"
echo "SGDD dir:         $SGDD_DIR"
echo "MDLM checkpoint:  $MDLM_CKPT"
echo "SEDD-U checkpoint: $SEDD_U_CKPT"
echo "Oracle checkpoint: $ORACLE_CKPT"
echo "Eval oracle:      ${EVAL_ORACLE_CKPT:-none}"
echo "Target cell:      $TARGET_CELL"
echo "Penalty weight:   $PENALTY_WEIGHT"
echo "Fitness mode:     $FITNESS_MODE"
echo "Spec. threshold:  ${SPECIFICITY_THRESHOLD:-none}"
echo "Population:       $POPULATION_SIZE"
echo "Max beta:         $MAX_BETA"
echo "Noise fraction:   $NOISE_FRACTION"
echo "Denoise steps:    ${STEPS:-default}"
echo "DPS:              $USE_DPS (eta=$DPS_ETA)"
echo "Guide start frac: $GUIDE_START_FRAC"
echo "DPS reward mode:  $DPS_REWARD_MODE (pw=$DPS_PENALTY_WEIGHT)"
echo "Elite fraction:   $ELITE_FRACTION"
echo "GC center weight: $GC_CENTER_WEIGHT"
echo "NF anneal:        $NF_START → $NF_END"
echo "GC DPS weight:    $GC_DPS_WEIGHT (target=$GC_DPS_TARGET)"
echo "GC pull weight:   $GC_PULL_WEIGHT (target=$GC_PULL_TARGET)"
echo "DPS mode:         $DPS_MODE"
echo "Gate sharpness:   $GATE_SHARPNESS"
echo "KL budget:        $KL_BUDGET ($KL_MODE)"
echo "EDTS:             K=$BRANCH_FACTOR, max_edit=$MAX_EDIT_FRAC, select=$EDIT_SELECT_MODE"
echo "MCTS:             depth=$MCTS_DEPTH, iter=$MCTS_ITERATIONS, C=$MCTS_C"
echo "Per-step cap:     $MAX_EDITS_PER_STEP"
echo "Hill-climb:       $HILL_CLIMB"
echo "HC budget:        $HILL_CLIMB_BUDGET"
echo "HC full-vocab:    $HILL_CLIMB_FULL_VOCAB"
echo "HC positions:     $HILL_CLIMB_POSITIONS"
echo "HC kmer weight:   $HC_KMER_WEIGHT"
echo "Gosai CSV:        ${GOSAI_CSV:-none}"
echo "Kmer target cell: ${KMER_TARGET_CELL:-auto}"
echo "Kmer top frac:    $KMER_TOP_FRAC"
echo "Edit discount:    $EDIT_DISCOUNT ($EDIT_DISCOUNT_MODE, lambda=$EDIT_DISCOUNT_LAMBDA)"
echo "Pareto fitness:   $PARETO_EDIT_FITNESS"
echo "Max delta beta:   $MAX_DELTA_BETA"
echo "Bio filter:       ${BIO_FILTER:-off}"
echo "Top-K init:       $TOP_K_INIT"
echo "Seed:             ${SEED:-none}"
echo "GC bin edges:     ${GC_BIN_EDGES:-none}"
echo "GC bin min quota: $GC_BIN_MIN_QUOTA"
echo "GC bin alloc:     $GC_BIN_ALLOC_MODE (alpha=$GC_BIN_ALLOC_ALPHA)"
echo "Start beta:       $START_BETA"
echo "Eval ckpt intv:   $EVAL_CHECKPOINT_INTERVAL"
echo "From random:      $FROM_RANDOM"
echo "Seed pool:        $SEED_POOL"
echo "Output dir:       $OUTPUT_DIR"
echo "Command: ${CMD[*]}"
echo "========================================================================"

"${CMD[@]}"
