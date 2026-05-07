# GPA paper — Overleaf upload guide

LaTeX source for "GPA: Discrete Population Annealing for Test-Time Sequence Design with Pretrained Discrete Diffusion Models". NeurIPS 2024 style.

## Files

```
paper/
├── main.tex          # Main 8-page paper
├── appendix.tex      # Supplementary material (proofs, hyperparameters, ablations)
├── references.bib    # All citations
├── figures/          # Place figure PDFs/PNGs here when added
└── README.md         # This file
```

## How to upload to Overleaf

The simplest path uses **Overleaf's built-in NeurIPS template** (which already includes `neurips_2024.sty`):

### Option A (recommended) — Use Overleaf's NeurIPS template as base

1. In Overleaf: **New Project → Templates → search "NeurIPS 2024"** → click "Open as Template"
2. Delete Overleaf's default `main.tex` and `references.bib` (the `neurips_2024.sty` stays)
3. Upload `main.tex`, `appendix.tex`, and `references.bib` from this zip
4. Set "Main document" to `main.tex` in Project menu → Settings
5. Compile

### Option B — Standalone upload (need to grab .sty separately)

1. Download `neurips_2024.sty` from <https://media.neurips.cc/Conferences/NeurIPS2024/Styles/neurips_2024.sty> on your local machine
2. Add it to the unzipped `paper/` directory
3. In Overleaf: **New Project → Upload Project** → drop the resulting zip

If you're targeting a different venue (ICLR, ICML), swap the `\documentclass` line and the `\usepackage{neurips_2024}` line in `main.tex` for that venue's style file.

## Step 3 — Compile

The paper compiles with `pdflatex` + `bibtex`. Overleaf default compiler should work without changes.

```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Status of tables / data

All numbers in `main.tex` are **live-pulled from the result files** as of 2026-04-28:

- **Table 1 (RERD comparison)**: numbers from `results/rerd_comparison/run_paper_{mean,spec}_random*` (4 reps each). Verified against `gpa_output.h5`.
- **Table 2 (one-knob curve)**: from v17b runs in `rerd_comparison`, GC≤50% subset (memory file `rerd_comparison.md`).
- **Table 3 (CtrlDNA HepG2/K562/SKNSH)**: HepG2 v5 and K562 v5 numbers verified live; SKNSH v5 is partial pending. `\ctrldna{} R200` numbers for K562/SKNSH are placeholders ("---") pending the resume run.
- **Table 4 (wall time)**: HepG2 only; cross-cell wall numbers not yet measured.
- **Table 5 (naturalness)**: from `naturalness_gpa_vs_ctrldna.md`.
- **Table 6 (K ablation)**: from `gpa_branch_factor_theory.md` (Reddy promoter benchmark).
- **Table 7 (DPS ablation)**: from RERD v17 ablation, same as Table 1's DPS-free rows.

## What needs filling in before submission

1. **Author block** — replace `Anonymous Authors` etc.
2. **Table 3 SK-N-SH \gpa{} v5 row** — pending the last few SLURM jobs to finish (jids 1904566–1904570). Run the analysis script to fill in.
3. **Table 3 \ctrldna{} K562/SKNSH rows** — pending RL-resume jobs to reach R200 (~24-30h after current runs end).
4. **Figures** — none included. Recommended additions:
   - Pareto frontier figure from Table 2 (pw curve).
   - GC distribution histogram (GPA vs CtrlDNA, Appendix~\ref{app:gc}).
   - Oracle trajectory plot (β over steps showing ESS-adaptive schedule).
   - Branch factor K saturation curve (Table 6 visualization).
5. **Specific paper-style edits** — citation key fixes once final venue is chosen, abstract polish, intro motivation tightening.

## Questions/decisions for the user

- **Title** — currently "GPA: Discrete Population Annealing ...". Alternatives:
  - "Test-Time Sequence Design via Population-Annealed Discrete Diffusion Sampling"
  - "Discrete Population Annealing: A Test-Time Sampler for Sequence Design"
- **GPA-Image inclusion** — currently NOT in main paper (per the soundness analysis: DPS reward-hacking requires explicit disclosure if included). If we want to include it, add Section 5b in main paper plus full reward-hacking section, or add as a separate appendix section "Cross-domain extension."
- **Promoter results** — currently mentioned only in Table 6 (K ablation) and conclusion. If we want a full promoter results section, add Section 5.X with the FINAL_gpa_vs_ctrldna_5seed.csv numbers for all 3 promoter cells.
