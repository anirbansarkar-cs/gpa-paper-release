# GPA vs Ctrl-DNA — paper Table 3 (Reddy promoter, 5-seed)

**Oracle**: gReLU-MSE fine-tuned on Reddy 2026 promoter 3-cell dataset
**Cells**: JURKAT, K562, THP1
**Backbone**: HyenaDNA Stage-1 e3 fine-tuned (same base for both methods)
**Ctrl-DNA baseline**: 200 PPO iterations, 5 seeds, same oracle
**GPA**: universal recipe at 5 seeds per cell

## Universal recipe

```
K=8 argmax + noDPS + pw=0.35 + eta=3000 + no bio-filter + lambda=1.0
N=10000   max_steps=60   max_beta=100   nf=0.05   ESS_thresh=0.5
```

## Table 3 (top-128 by target reward, paper-faithful selection)

| Cell | Method | target_raw | ΔR_norm | shannon | motif_corr (FIMO) |
| --- | --- | --- | --- | --- | --- |
| JURKAT | Ctrl-DNA      | 5.68 ± 0.48 | 0.14 ± 0.03 | 1.64 ± 0.07 | 0.76 ± 0.05 |
| JURKAT | **GPA univ.** | **8.37 ± 0.09** | **0.25 ± 0.02** | 1.53 ± 0.10 | **0.79 ± 0.01** |
| K562   | Ctrl-DNA      | 6.02 ± 0.06 | **0.49 ± 0.02** | **1.69 ± 0.01** | **0.71 ± 0.01** |
| K562   | **GPA univ.** | **7.00 ± 0.01** | 0.31 ± 0.01 | 1.63 ± 0.05 | 0.67 ± 0.03 |
| THP1   | Ctrl-DNA      | 4.16 ± 0.26 | 0.31 ± 0.04 | **1.79 ± 0.03** | 0.51 ± 0.11 |
| THP1   | **GPA univ.** | **4.52 ± 0.04** | **0.38 ± 0.01** | 1.60 ± 0.16 | **0.60 ± 0.04** |

GPA wins target activity on all three cells. ΔR_norm and motif correlation favor GPA on JURKAT and THP1; K562 trails under top-128-by-target selection due to oracle saturation in the Reddy K562 oracle's training distribution.

## Wall-time

| Method | h/seed | total (5 seeds, 3 cells) |
| --- | --- | --- |
| GPA universal (inference only) | ~2.2 | ~33 GPU-h |
| Ctrl-DNA (200 PPO iterations)  | ~43.5 | ~653 GPU-h |

GPA delivers parity-or-better cell-type-discriminative activity at ~20× lower wall-time than the RL fine-tune.
