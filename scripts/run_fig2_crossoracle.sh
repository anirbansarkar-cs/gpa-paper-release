#!/bin/bash
# Figure 2 — cross-oracle reward inflation on LentiMPRA K562.
#
# All three methods optimize the LegNet K562 oracle (LN); the held-out
# AlphaGenome-derived K562 oracle (AG, Appendix J) is used for evaluation only
# and never enters the optimization loop.
#
# Two 5,000-sequence seed pools (random and natural) x edit-budget caps of
# 20/40/60/100 bp plus each method's uncapped final checkpoint.
#
# GPA recipe `v3t_nodps_nobio_nogc_b2k`: K = 10, nf = 0.05, beta* = 2000, no DPS,
# no bio filter, no GC-centering, N = 5000, seeds 42/43/44.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GPA_DATA_ROOT:?set GPA_DATA_ROOT}"
: "${GPA_SHARED_ROOT:?set GPA_SHARED_ROOT}"
OUT_ROOT="${OUT_ROOT:-results/ledidi_comparison}"

MDLM_CKPT="${MDLM_CKPT:-${GPA_SHARED_ROOT}/mdlm/lentimpra/best.ckpt}"
LN_CKPT="${LN_CKPT:-${GPA_DATA_ROOT}/lentimpra/legnet_k562.ckpt}"
POOL_RANDOM="${POOL_RANDOM:-${GPA_DATA_ROOT}/lentimpra/pool_random_5k.h5}"
POOL_NATURAL="${POOL_NATURAL:-${GPA_DATA_ROOT}/lentimpra/pool_natural_5k.h5}"

# cap label : max edit fraction of the 200 bp core (0.10 -> 20 bp, ... 1.00 -> uncapped)
CAPS=("cap20:0.10" "cap40:0.20" "cap60:0.30" "cap100:0.50" "uncap:1.00")
SEEDS=(${SEEDS:-42 43 44})

for POOL_TAG in random natural; do
  case "${POOL_TAG}" in
    random)  POOL="${POOL_RANDOM}"  ;;
    natural) POOL="${POOL_NATURAL}" ;;
  esac

  # ---- GPA ----
  for SPEC in "${CAPS[@]}"; do
    CAP_TAG="${SPEC%%:*}"; CAP_FRAC="${SPEC##*:}"
    for SEED in "${SEEDS[@]}"; do
      OUT="${OUT_ROOT}/gpa_${POOL_TAG}_${CAP_TAG}_seed${SEED}"
      mkdir -p "${OUT}"
      python scripts/k562_mdlm_gpa/run_k562_mdlm_gpa.py \
          --mdlm_checkpoint "${MDLM_CKPT}" \
          --oracle_checkpoint "${LN_CKPT}" --oracle_type legnet \
          --seed_pool "${POOL}" \
          --population_size 5000 \
          --branch_factor 10 \
          --noise_fraction 0.05 \
          --max_beta 2000.0 \
          --max_delta_beta 6.67 \
          --hard_max_steps 400 --extend_steps 10 --extend_on_delta_beta 0.5 \
          --cache_mutation_scores \
          --max_edit_frac "${CAP_FRAC}" \
          --eval_ag --eval_checkpoint_interval 5 --eval_early_stop_patience 10 \
          --archive_threshold 3.0 \
          --seed "${SEED}" \
          --output_dir "${OUT}"
    done
  done

  # ---- ISM baseline: 5,000 greedy trajectories, 115 sequential edit steps ----
  OUT="${OUT_ROOT}/ism_${POOL_TAG}"
  mkdir -p "${OUT}"
  python scripts/ledidi_comparison/run_ism_pool.py \
      --pool "${POOL}" \
      --oracle_ckpt "${LN_CKPT}" --oracle_type legnet \
      --total_steps 115 \
      --output_dir "${OUT}"

  # ---- LEDIDI baseline: 1,000 gradient steps, seeds in chunks of 128 ----
  OUT="${OUT_ROOT}/ledidi_${POOL_TAG}"
  mkdir -p "${OUT}"
  python scripts/ledidi_comparison/run_ledidi_baseline.py \
      --seed_pool "${POOL}" --seed_mode pool \
      --oracle_ckpt "${LN_CKPT}" --oracle_type legnet \
      --max_iter 1000 --ledidi_batch_size 128 \
      --snapshot_iters 50,100,200,400,1000 \
      --output_dir "${OUT}"
done
