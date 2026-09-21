# GPA: Generative Population Annealing

Code release for *"GPA: Generative Population Annealing for Test-Time Sequence
Design with Pretrained Generative Models"*.

GPA is a backbone- and oracle-agnostic test-time SMC sampler over the
reward-tilted distribution `pi_beta(x) ∝ p_theta(x) · exp(beta · r(x))`. It runs
unmodified on discrete-diffusion (MDLM) and autoregressive language-model
(HyenaDNA) backbones and returns population-scale design pools without any
backbone fine-tuning.

This repository contains the sampler and the four experiment drivers, plus one
shell script per reported experiment holding the exact commands used. It does
not include the analysis and table-building code.

## Layout

```
scripts/
  gpa_sampling.py                  Core GPA sampler (Algorithm 1)
  run_table1_dnacraft.sh           Table 1 — DNA-CRAFT enhancer benchmark
  run_table2_gosai.sh              Table 2 — Gosai HepG2, beta* in {5,10,15}
  run_fig2_crossoracle.sh          Figure 2 — LegNet/AlphaGenome cross-oracle
  run_table4_promoter.sh           Table 4 — Ctrl-DNA Reddy promoter
  rerd_comparison/                 MDLM backbone + Enformer oracle (Tables 1, 2)
  k562_mdlm_gpa/                   LentiMPRA K562 driver + LegNet/AlphaGenome oracles
  ledidi_comparison/               ISM and LEDIDI baselines
  ctrl_dna_comparison/             HyenaDNA backbone + Reddy promoter oracles
  dna_craft_comparison/            DiMamba/HyenaDNA enhancer driver
model_zoo/lentimpra/mpralegnet.py  LegNet definition used by the K562 oracle
bio_plausibility.py                Sequence-level bio filter
```

## Setup

```bash
conda env create -f environment.yml
conda activate gpa
pip install -e .
```

Requires `python >= 3.9`, `torch >= 2.0`, `lightning >= 2.0`, `flash-attn >= 2.0`.
The AlphaGenome held-out oracle additionally needs JAX with CUDA 11.8; it runs in
a separate environment and talks to the sampler over a Unix domain socket
(`ag_oracle_server.py` / `ag_oracle_client.py`).

Point the run scripts at your data and checkpoints:

```bash
export GPA_REPO_ROOT=$(pwd)
export GPA_DATA_ROOT=/path/to/datasets       # Gosai MPRA, Reddy MPRA, LentiMPRA
export GPA_EXTERNAL_ROOT=/path/to/external   # DRAKES release (MDLM prior + split oracles)
export GPA_SHARED_ROOT=/path/to/models       # pretrained MDLM / HyenaDNA / AlphaGenome
```

External resources, all from published releases: the MDLM backbone
(`sahoo2024mdlm`), the HyenaDNA backbone, the DRAKES split-oracle
(`drakes2024`), and the public AlphaGenome encoder. Ctrl-DNA, DNA-CRAFT, LEDIDI
and ISM are used through thin wrappers in this repo; the baseline
implementations themselves come from their own releases.

## Running

Each script takes one GPU per invocation and runs the full seed/condition grid
for its table. They are plain `python` loops with no scheduler directives — wrap
the inner command in whatever job system you use.

```bash
bash scripts/run_table1_dnacraft.sh      # 3 cells x 3 seeds, ~3.6 min/seed
bash scripts/run_table2_gosai.sh         # 3 betas x 3 reps, ~30 min/seed
bash scripts/run_fig2_crossoracle.sh     # 2 pools x 5 edit caps x 3 seeds + baselines
bash scripts/run_table4_promoter.sh      # 3 cells x 5 seeds, ~2.2 h/seed
```

Hyperparameters are set inline and match the appendix tables. Grid dimensions
can be overridden from the environment, e.g. `BETAS="5 10" bash
scripts/run_table2_gosai.sh`.

Each run writes its pool to `--output_dir`. Report held-out oracle scores from
`gpa_output_best_eval.h5`, not from the final-step population.

## License

MIT, for academic and non-commercial research use.
