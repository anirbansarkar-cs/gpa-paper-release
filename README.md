# GPA: Generative Population Annealing

Code release for *"GPA: Generative Population Annealing for Test-Time Sequence
Design with Pretrained Generative Models"*.

GPA is a test-time SMC sampler over the reward-tilted distribution
`pi_beta(x) ∝ p_theta(x) · exp(beta · r(x))`. It needs no fine-tuning of either
the generator or the oracle, and it returns a population-scale design pool
rather than a single optimised sequence.

The generator and the oracle enter through two callables, so GPA is not tied to
the backbones in the paper. If you want to apply it to your own model and
dataset, start at **Bring your own generator and oracle** below. If you want to
reproduce the paper, skip to **Reproducing the paper**.

---

## Bring your own generator and oracle

GPA asks two things of you:

```python
# 1. A mutation proposal: how your generator perturbs a sequence.
mutate_fn(model, x_clean, labels, branch_factor=1, **kwargs) -> x_new
#   x_clean : (B, L) long tensor of token indices, A=0 C=1 G=2 T=3
#   returns : (B, L), or (K, B, L) when branch_factor > 1

# 2. A reward: how good each sequence is.
oracle_fn(sequences_tensor) -> (scores, gc)
#   returns : two float arrays of length B
```

That is the whole interface. For masked diffusion, `mutate_fn` is
partial-corruption + re-denoise; for an autoregressive model it is
partial-context resampling; for anything else it is whatever keeps proposals
near your prior. `scores` is only ever ranked and exponentiated, so any monotone
scale works. `gc` is used for reporting and the optional GC controls — return
zeros if it is meaningless for your alphabet.

**Start here:** `scripts/example_custom_backbone.py` is a runnable ~100-line
template with a toy generator and a toy oracle. It needs only torch and numpy:

```bash
python scripts/example_custom_backbone.py
# [init]  mean motif hits 0.006   max 1
# [final] mean motif hits 1.5-1.7  max 2-3     (varies run to run)
```

**Then copy a real driver.** The closest to a general-purpose entry point is
`scripts/dna_craft_comparison/enhancer/run_gpa_dnacraft_enhancer.py` — it is the
smallest (362 lines) and the only one that already drives two different
backbones behind the same interface (`--backbone hyenadna|dimamba`), so it shows
the swap rather than describing it.

Reference implementations to imitate:

| You are writing | Look at |
|---|---|
| a masked-diffusion mutator | `scripts/rerd_comparison/mdlm_wrapper.py` (`MDLMMutator`, 72 lines) |
| a DiMamba / bidirectional mutator | `scripts/dna_craft_comparison/enhancer/mdlm_wrapper.py` |
| an autoregressive mutator | `scripts/ctrl_dna_comparison/hyenadna_wrapper.py` |
| an oracle (incl. the differentiable `dps_forward`) | `scripts/rerd_comparison/enformer_oracle.py` |
| an oracle behind a separate environment | `scripts/k562_mdlm_gpa/ag_oracle_{server,client}.py` |

The sampler itself is `scripts/gpa_sampling.py`; its class docstring documents
every argument, including the optional `fitness_fn` and `mutate_fn_factory`
hooks.

---

## Techniques that improve on standard GPA

Plain GPA — anneal, resample, mutate, return the final population — is the
baseline. Everything below is optional and each is a single flag. They are
roughly ordered by how much they buy for how little effort.

### 1. Keep the intermediate samples (usually the biggest free win)

**The final-step population is not your best pool.** GPA visits far more good
sequences than it ends with: annealing pushes the population past high-reward
regions, and resampling discards particles that were transiently excellent. Six
independent collection mechanisms exist, each writing its own file, all defined
on the `GPAHistory` dataclass at `scripts/gpa_sampling.py:113-148`:

| Flag | Writes | What it keeps |
|---|---|---|
| `--eval_checkpoint_interval N` | `gpa_output_best_eval.h5` | the whole population, snapshotted at its best held-out-oracle mean — **the authoritative pool; report from this, not the final step** |
| `--archive_threshold T` | `gpa_output_archive.h5` | every sequence scoring above `T` at any step |
| `--perstep_archive_top_k K` | `gpa_output_perstep_archive.h5` | top-K at each eval checkpoint, accumulated |
| `--mingap_archive_size N` | `gpa_output_mingap_archive.h5` | top-N by cell-specificity margin (the DNA-CRAFT `G*` rule) |
| `--budget_archive_size N` | `gpa_output_budget_archive.h5` | a capped streaming archive ranked by a composite of activity, k-mer fidelity and likelihood, with online eviction — bounded memory |
| `--topk_ag_size N` | `gpa_output_topk_ag.h5` | top-N by the held-out oracle, deduped, with eviction — catches transient peaks the end-of-run snapshot loses |

Use `--archive_threshold` if you have a natural score cutoff, `--budget_archive_size`
if you need a fixed output budget, `--topk_ag_size` if your evaluation oracle
differs from your design oracle. They compose; enabling several costs only disk.

### 2. Population size `N` — `--population_size`

The one knob that reliably improves every metric. The paper's ablation sweeps
128 → 20,000 and reports MinGap gains of +1.03 (HepG2), +1.08 (K562) and +2.98
(SK-N-SH), with k-mer, motif and diversity improving alongside. Cost is linear
in wall-time and memory. The paper's headline runs use 5,000 as a
runtime/benefit compromise; 20,000 is better on every axis if you can afford it.

### 3. Branch factor `K` — `--branch_factor`

Draw `K` proposals per parent from the same forward pass and keep the best. This
shapes the proposal by an order statistic rather than the raw model, so it
improves matched-budget metrics without changing the generator. `K=8` is the
paper's default across benchmarks; `K=10` in the cross-oracle runs. Cost is
roughly linear in `K` for the mutation step only. `--branch_selection_tau`
switches selection from argmax to softmax.

### 4. DPS gradient proposal — `--use_dps --dps_eta 3000`

Biases the proposal along the oracle gradient via a Gumbel-softmax relaxation.
**Requires a differentiable oracle** exposing `dps_forward(soft_onehot)` — see
`enformer_oracle.py:370`. The paper's ablation reports roughly +1.3 held-out
activity and +1.5 specificity over the bare sampler, with the specificity gain
appearing with and without an explicit off-target penalty. Skip it if your
oracle is black-box or API-only; the non-DPS setting is the honest reference
there.

### 5. The `beta*` dial — `--max_beta`

The single inference-time knob that moves you along the activity–fidelity
frontier. The paper sweeps 5 → 15 on Gosai HepG2 and reports predicted activity
rising 6.82 → 8.45 while 3-mer correlation falls 0.912 → 0.563. There is no
universally right value: pick the operating point your application needs, and
report the sweep rather than one point.

### 6. Multi-objective and constraint shaping

| Flag | Effect |
|---|---|
| `--penalty_weight` / `--dps_penalty_weight` | off-target penalty `pw`; a linear scalarisation with a Lagrangian reading. Sweep it to trace the activity–specificity trade-off |
| `--diversity_lambda` | population-conformity penalty `lambda`; discourages the population from collapsing onto one motif arrangement |
| `--gc_dps_target` / `--gc_dps_weight` | soft GC-centering inside the DPS update |
| `--bio_filter --gc_low --gc_high` | hard sequence-level filter (see `bio_plausibility.py`) |
| `--max_edit_frac` | cap edits as a fraction of length — use when designs must stay close to a natural seed |
| `--ess_threshold` | `alpha`; when to resample. 0.5 throughout the paper |

### 7. Diversity maintenance

GPA reaches 1.83–1.86 pairwise-Hamming diversity against a 1.98 benchmark
ceiling; this is the paper's acknowledged gap. If diversity matters for your
screen: `--dedup_threshold` (drop near-duplicates), `--rejuvenation_fraction`
(re-mutate a slice of the population each step), `--elite_fraction` (protect the
top fraction from resampling), and a larger `N`.

### 8. Local refinement and planning

`--hill_climb_budget` runs greedy single-position refinement on selected
particles. `--mcts_depth` / `--mcts_iterations` / `--mcts_c` enable UCB-style
lookahead over mutation sequences. Both cost oracle calls; the paper's headline
results use neither, and the MCTS scaling sweep is in the appendix.

### 9. Speed

`--cache_mutation_scores` reuses oracle scores across the branch-selection step.
`--mutation_batch_size` controls the GPU-memory/throughput trade (128 in the
cross-oracle runs). `--eval_early_stop_patience` stops once the held-out oracle
stops improving.

Not every flag is wired into every driver. `--topk_ag_size`,
`--cache_mutation_scores` and `--eval_early_stop_patience` are exposed only by
`run_k562_mdlm_gpa.py`; `--rejuvenation_fraction` and `--branch_selection_tau`
only by the promoter driver; `--max_edit_frac` and `--elite_fraction` by the
rerd and K562 drivers. All are available on the `DiffusionPopulationAnnealer`
API regardless — copy the argparse entry across if you need it elsewhere.

---

## Reproducing the paper

```
scripts/
  gpa_sampling.py                  Core GPA sampler (Algorithm 1)
  example_custom_backbone.py       Runnable template: custom generator + oracle
  run_table1_dnacraft.sh           Table 1 — DNA-CRAFT enhancer benchmark
  run_table2_gosai.sh              Table 2 — Gosai HepG2, beta* in {5,10,15}
  run_fig2_crossoracle.sh          Figure 2 — LegNet/AlphaGenome cross-oracle
  run_table4_promoter.sh           Table 4 — Ctrl-DNA Reddy promoter
  rerd_comparison/                 MDLM backbone + Enformer oracle (Tables 1, 2)
  k562_mdlm_gpa/                   LentiMPRA K562 driver + LegNet/AlphaGenome oracles
  ledidi_comparison/               ISM and LEDIDI baselines
  ctrl_dna_comparison/             HyenaDNA backbone + Reddy promoter oracles
  dna_craft_comparison/            DiMamba/HyenaDNA enhancer driver
  alphagenome/                     Held-out AlphaGenome scorer (JAX)
model_zoo/lentimpra/mpralegnet.py  LegNet definition used by the K562 oracle
bio_plausibility.py                Sequence-level bio filter
```

This release contains the sampler, the drivers and one script per reported
experiment. It does not contain the analysis and table-building code.

### Setup

```bash
conda env create -f environment.yml
conda activate gpa
pip install -e .
```

Requires `python >= 3.9`, `torch >= 2.0`, `lightning >= 2.0`, `flash-attn >= 2.0`.
The AlphaGenome held-out oracle additionally needs JAX with CUDA 11.8; it runs in
a separate environment and talks to the sampler over a Unix domain socket
(`ag_oracle_server.py` / `ag_oracle_client.py`).

```bash
export GPA_REPO_ROOT=$(pwd)
export GPA_DATA_ROOT=/path/to/datasets       # Gosai MPRA, Reddy MPRA, LentiMPRA
export GPA_EXTERNAL_ROOT=/path/to/external   # DRAKES release (MDLM prior + split oracles)
export GPA_SHARED_ROOT=/path/to/models       # pretrained MDLM / HyenaDNA / AlphaGenome
```

External resources, all from published releases: the MDLM backbone
(`sahoo2024mdlm`), the HyenaDNA backbone, the DRAKES split-oracle
(`drakes2024`), and the public AlphaGenome encoder. Ctrl-DNA, DNA-CRAFT, LEDIDI
and ISM are used through thin wrappers here; the baseline implementations come
from their own releases.

### Running

Each script takes one GPU per invocation and runs the full seed/condition grid
for its table. They are plain `python` loops with no scheduler directives — wrap
the inner command in whatever job system you use.

```bash
bash scripts/run_table1_dnacraft.sh      # 3 cells x 3 seeds, ~3.6 min/seed
bash scripts/run_table2_gosai.sh         # 3 betas x 3 reps, ~30 min/seed
bash scripts/run_fig2_crossoracle.sh     # 2 pools x 5 edit caps x 3 seeds + baselines
bash scripts/run_table4_promoter.sh      # 3 cells x 5 seeds, ~2.2 h/seed
```

Hyperparameters are inline and match the appendix tables. Grid dimensions can be
overridden from the environment, e.g. `BETAS="5 10" bash scripts/run_table2_gosai.sh`.

## License

MIT, for academic and non-commercial research use.
