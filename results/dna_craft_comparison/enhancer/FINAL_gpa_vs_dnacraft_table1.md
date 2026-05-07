# GPA vs DNA-CRAFT — paper Table 1

Top-64 selected by `by_pool_composite` (paper-faithful matched-output-budget against DNA-CRAFT's G* at N_max=64).
GPA recipe: `dps_pw030_b25_gct060_nobio` (DPS, GC≤60%, no bio filter).

| Cell | Metric | SMC | CG | TDS | DRAKES | D3 | Ledidi | Ctrl-DNA | DNA-CRAFT | **GPA** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HepG2 | MinGap ↑ | +1.61 | −0.23 | +0.40 | −1.40 | +0.05 | +5.77 | **+7.79** | +4.35 | +6.91 ± 0.10 |
| HepG2 | Motif ↑ | 0.55 | 0.86 | 0.40 | 0.06 | 0.87 | 0.58 | 0.63 | **0.92** | **0.92 ± 0.01** |
| HepG2 | 3-mer ↑ | 0.81 | 0.97 | 0.74 | −0.36 | 0.98 | 0.76 | 0.49 | **0.98** | 0.95 ± 0.00 |
| HepG2 | Diversity ↑ | 0.83 | 1.98 | 0.96 | 1.86 | 1.98 | **1.98** | 1.90 | **1.98** | 1.86 ± 0.06 |
| K562 | MinGap ↑ | +4.12 | 0.00 | +1.62 | −0.20 | +0.18 | +7.66 | **+9.07** | +5.69 | +8.13 ± 0.05 |
| K562 | Motif ↑ | 0.45 | 0.85 | 0.51 | 0.14 | 0.86 | 0.65 | 0.63 | 0.93 | **0.94 ± 0.01** |
| K562 | 3-mer ↑ | 0.66 | 0.94 | 0.65 | −0.35 | 0.96 | 0.69 | 0.41 | **0.98** | 0.95 ± 0.02 |
| K562 | Diversity ↑ | 0.31 | 1.98 | 0.64 | 1.96 | 1.98 | 1.98 | 1.90 | **1.98** | 1.83 ± 0.08 |
| SK-N-SH | MinGap ↑ | +0.56 | −0.28 | +0.19 | +0.09 | −0.01 | +3.03 | +3.72 | +3.23 | **+4.98 ± 0.03** |
| SK-N-SH | Motif ↑ | 0.52 | 0.86 | 0.48 | 0.23 | 0.84 | 0.38 | 0.48 | 0.88 | **0.92 ± 0.01** |
| SK-N-SH | 3-mer ↑ | 0.78 | 0.95 | 0.72 | −0.38 | 0.93 | 0.37 | 0.20 | **0.97** | 0.94 ± 0.01 |
| SK-N-SH | Diversity ↑ | 1.27 | 1.98 | 0.92 | 1.83 | 1.97 | 1.98 | 1.86 | **1.98** | 1.86 ± 0.04 |

GPA values are mean ± std over 3 reps. Baseline rows (SMC … DNA-CRAFT) are reproduced verbatim from the DNA-CRAFT paper (arXiv:2604.20488 Table 2).
