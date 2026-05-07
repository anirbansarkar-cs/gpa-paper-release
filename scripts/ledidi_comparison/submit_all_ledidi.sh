#!/bin/bash
# Submit all 3 LEDIDI baseline runs as separate SLURM jobs.
#
# Run 1: ISM seeds (105 natural + 90 random from ISM comparison)
# Run 2: Random init (100 uniform random 200bp sequences)
# Run 3: GPA seed (50 copies of single_seed_299823, tests Gumbel variance)

set -euo pipefail

SCRIPT_DIR=${GPA_REPO_ROOT}/scripts/ledidi_comparison
SBATCH_SCRIPT=${SCRIPT_DIR}/run_ledidi_baseline.sh

echo "Submitting 3 LEDIDI baseline jobs..."

# Run 1: ISM seeds
JOB1=$(SEED_MODE=ism sbatch --job-name=ledidi_ism ${SBATCH_SCRIPT} | awk '{print $NF}')
echo "  ISM seeds:    job ${JOB1}"

# Run 2: Random init
JOB2=$(SEED_MODE=random_init sbatch --job-name=ledidi_rand ${SBATCH_SCRIPT} | awk '{print $NF}')
echo "  Random init:  job ${JOB2}"

# Run 3: GPA seed
JOB3=$(SEED_MODE=gpa_seed sbatch --job-name=ledidi_gpa ${SBATCH_SCRIPT} | awk '{print $NF}')
echo "  GPA seed:     job ${JOB3}"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Results will be in:"
echo "  results/ledidi_comparison/ism/"
echo "  results/ledidi_comparison/random_init/"
echo "  results/ledidi_comparison/gpa_seed/"
