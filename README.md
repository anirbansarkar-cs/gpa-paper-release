# GPA: Generative Population Annealing

Code release accompanying the paper *"GPA: Generative Population Annealing for Test-Time Sequence Design with Pretrained Generative Models"*.

GPA is a backbone- and oracle-agnostic test-time SMC sampler over the reward-tilted distribution `pi_beta(x) ∝ p_theta(x) exp(beta · r(x))`. It runs unmodified on discrete-diffusion (MDLM, SEDD) and autoregressive language-model (HyenaDNA) backbones, and produces population-scale design pools (N ∈ [5,000, 20,000]) without any backbone fine-tuning.

## Repository layout

```
.
├── model/                          # Backbone-agnostic model components
├── model_zoo/                      # Per-dataset wrappers (lentimpra, promoter, deepstarr, ...)
├── scripts/
│   ├── gpa_sampling.py             # Core GPA sampler (Algorithm 1)
│   ├── cfg_sampling.py             # CFG mutation kernels
│   ├── dps_sampling.py             # Differentiable-policy-shift gradient option
│   ├── evaluate.py                 # Pool evaluation
│   ├── alphagenome/                # AlphaGenome held-out oracle wrapper
│   ├── dna_craft_comparison/       # Headline 1: DNA-CRAFT enhancer benchmark (Table 1)
│   ├── rerd_comparison/            # Headline 2: SMC-family Gosai-HepG2 (Table 2)
│   ├── ledidi_comparison/          # Headline 3: ISM/LEDIDI K562 baselines (Figure 1)
│   ├── k562_mdlm_gpa/              # Headline 3: GPA K562 driver
│   └── ctrl_dna_comparison/promoter/  # Headline 4: Ctrl-DNA Reddy promoter (Table 3)
└── paper/                          # LaTeX sources, figures, references
```

## Headline experiments

| # | Section | Driver | Selection script | Pool source |
|---|---------|--------|------------------|-------------|
| 1 | DNA-CRAFT enhancer (Table 1) | `scripts/rerd_comparison/run_rerd_gpa.py` (recipe `dps_pw030_b25_gct060_nobio`) | `scripts/rerd_comparison/dnacraft/build_top64_table.py` | top-64 by `by_pool_composite` |
| 2 | SMC family Gosai-HepG2 (Table 2) | `scripts/rerd_comparison/run_rerd_gpa.py` (β ∈ {25, 50}) | `scripts/rerd_comparison/_unified_rank_motif_kmer_atac.py` | App-LL ≥ −259 + composite rank top-640 |
| 3 | ISM/LEDIDI K562 (Figure 1) | `scripts/k562_mdlm_gpa/run_k562_mdlm_gpa.py` (v3t no-DPS) | `scripts/k562_mdlm_gpa/build_gpa_vs_ism_ledidi_docx.py` | 3 seeds {42,43,44}, AG-RC + LN-RC averaged |
| 4 | Ctrl-DNA Reddy promoter (Table 3) | `scripts/ctrl_dna_comparison/promoter/run_gpa_hyenadna_promoter.py` | `scripts/ctrl_dna_comparison/promoter/evaluate_promoter_pools.py` | 5 seeds, top-128 by target |

The universal recipe for headline 4 is `K=8` argmax + no-DPS + `pw=0.35` + `eta=3000` + no bio-filter + `lambda=1.0`.

## Setup

### Environment

```bash
conda env create -f environment.yml
conda activate gpa
pip install -e .
```

The repository expects `python ≥ 3.9`, `torch ≥ 2.0`, `lightning ≥ 2.0`, `flash-attn ≥ 2.0`. AlphaGenome scoring (`scripts/alphagenome/`) needs JAX with CUDA 11.8.

### Required data and checkpoints

The codebase expects the following directories to be set as environment variables:

```bash
export GPA_REPO_ROOT=$(pwd)
export GPA_DATA_ROOT=/path/to/datasets       # Gosai MPRA, Reddy MPRA, LentiMPRA
export GPA_EXTERNAL_ROOT=/path/to/external   # External baselines (DRAKES, Ctrl-DNA, LEDIDI)
export GPA_SHARED_ROOT=/path/to/shared       # Pretrained MDLM, AlphaGenome (public release)
```

External resources used in the paper:
- **MDLM backbone** — `sahoo2024mdlm` (pretrained on Gosai MPRA, public release).
- **HyenaDNA backbone** — `hyenadna2023` (HuggingFace).
- **AlphaGenome encoder (held-out cross-oracle)** — public JAX release at the project's official URL; place under `${GPA_SHARED_ROOT}/models/alphagenome_encoder/jax/`.
- **DRAKES split-oracle** — `drakes2024` (public release).
- **Ctrl-DNA / DNA-CRAFT / LEDIDI / ISM** — published baseline implementations; this repo includes wrappers, not the original baseline code.

## Running the headline experiments

### 1. DNA-CRAFT enhancer (Table 1)

```bash
# Sample
sbatch scripts/rerd_comparison/submit_gpa_recipes.sh

# Score and build Table 1
python scripts/rerd_comparison/dnacraft/score_dnacraft_protocol.py
python scripts/rerd_comparison/dnacraft/build_top64_table.py
```

### 2. SMC family on Gosai HepG2 (Table 2)

```bash
# Sample (recipes at beta=25, 50)
sbatch scripts/rerd_comparison/submit_gpa_recipes.sh

# Score the sampled pools under the DRAKES protocol
sbatch scripts/rerd_comparison/submit_drakes_scoring.sh

# Apply App-LL >= -259 + composite top-640 selection
python scripts/rerd_comparison/_unified_rank_motif_kmer_atac.py
python scripts/rerd_comparison/_unified_rank_app_filtered.py
```

Outputs: `top640_unified_motif_kmer_atac.csv` and `top640_unified_app_filtered.csv` under `${GPA_REPO_ROOT}/results/rerd_comparison/drakes_protocol/`.

### 3. ISM / LEDIDI on K562 (Figure 1)

```bash
# GPA v3t (no-DPS) on 3 seeds
bash scripts/ledidi_comparison/run_gpa_v3_pool_A_v3t_3seed.sh

# ISM and LEDIDI baselines
sbatch scripts/k562_mdlm_gpa/submit_ism_baseline.sh
python scripts/ledidi_comparison/run_ledidi_baseline.py

# Build Figure 1 + LN/AG ratio + joint-rank tables
python scripts/k562_mdlm_gpa/build_gpa_vs_ism_ledidi_docx.py
```

### 4. Ctrl-DNA on Reddy promoters (Table 3)

```bash
# GPA universal recipe, 5 seeds
sbatch scripts/ctrl_dna_comparison/promoter/submit_gpa_promoter.sh

# Ctrl-DNA baseline (200 PPO iterations)
sbatch scripts/ctrl_dna_comparison/promoter/submit_ctrldna_promoter.sh

# Evaluate and build Table 3
python scripts/ctrl_dna_comparison/promoter/evaluate_promoter_pools.py
python scripts/ctrl_dna_comparison/promoter/final_gpa_vs_ctrldna.py
```

## Notes on the SLURM scripts

The submit scripts under `scripts/*/submit_*.sh` are the original cluster batch files used to run the experiments. They use generic SLURM directives but assume:
- a CUDA-capable partition with H100 or A100 GPUs;
- `${HOME}/.bashrc` set up to load conda from `${CONDA_BASE}`.

You will likely need to edit `--partition`, `--qos`, and the conda activation line at the top of each submit script.

## Reproducibility

All headline numbers in the paper come from result CSVs produced by the corresponding selection scripts. The `_ref_cache/` reference data for the SMC-family scoring (top-0.1 % Gosai HepG2 anchor) is the same canonical DRAKES reference set used by `paniouli2025` and the DRAKES paper.

Wall-time per seed on a single H100 (paper Section 4):
- DNA-CRAFT enhancer: ~3.6 min/seed
- Gosai HepG2 GPA: ~30 min/seed
- K562 v3t: ~25 min/seed
- Reddy promoter (universal): ~2.2 h/seed

## License

This code is released under the MIT License for academic and non-commercial research use.
