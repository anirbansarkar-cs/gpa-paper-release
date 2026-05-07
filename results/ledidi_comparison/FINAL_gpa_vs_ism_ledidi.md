# GPA vs ISM vs LEDIDI v2 — K562 results

Headline metrics. Oracle hacking diagnostics: lower z-score MSE / lower raw MSE / lower 1-Pearson(LN, AG) → more agreement between LegNet and AlphaGenome → less reward hacking.

## Pool: _smoke_snap — matched edit caps

| cap (edits) | method | recipe | n | mean LN | mean LN (RC) | mean AG | mean AG (RC) | max LN (RC) | max AG (RC) | 1-Pearson(LN,AG)_RC |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | LEDIDI | _smoke_snap | 13 | -0.017 | 0.035 | -0.288 | -0.286 | 0.862 | 0.297 | 0.273 |
| 60 | LEDIDI | _smoke_snap | 163 | -0.058 | -0.032 | -0.215 | -0.200 | 1.443 | 1.121 | 0.171 |
| 100 | LEDIDI | _smoke_snap | 22 | -0.212 | -0.290 | -0.358 | -0.344 | 0.248 | 0.122 | 0.095 |

### Pool _smoke_snap — method-specific tail rows

| method | recipe | cap (edits) | n | mean LN | max LN | mean AG | max AG | 1-Pearson |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LEDIDI | _smoke_snap | uncap/all | 700 | -0.123 | 1.130 | -0.264 | 1.188 | 0.177 |

## Pool: _smoke_snap_v2 — matched edit caps

| cap (edits) | method | recipe | n | mean LN | mean LN (RC) | mean AG | mean AG (RC) | max LN (RC) | max AG (RC) | 1-Pearson(LN,AG)_RC |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | LEDIDI | _smoke_snap_v2 | 2 | -0.235 | -0.213 | -0.265 | -0.225 | -0.113 | -0.145 | nan |
| 40 | LEDIDI | _smoke_snap_v2 | 70 | -0.024 | -0.021 | -0.205 | -0.192 | 0.843 | 0.719 | 0.244 |
| 60 | LEDIDI | _smoke_snap_v2 | 257 | -0.055 | -0.012 | -0.203 | -0.184 | 1.114 | 1.002 | 0.180 |
| 100 | LEDIDI | _smoke_snap_v2 | 92 | -0.109 | -0.102 | -0.231 | -0.228 | 0.957 | 0.836 | 0.278 |

### Pool _smoke_snap_v2 — method-specific tail rows

| method | recipe | cap (edits) | n | mean LN | max LN | mean AG | max AG | 1-Pearson |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LEDIDI | _smoke_snap_v2 | uncap/all | 1600 | -0.149 | 1.332 | -0.274 | 1.156 | 0.184 |

## Pool: pool_A — matched edit caps

| cap (edits) | method | recipe | n | mean LN | mean LN (RC) | mean AG | mean AG (RC) | max LN (RC) | max AG (RC) | 1-Pearson(LN,AG)_RC |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | GPA | v3t_nodps_nobio_nogc_b2k | 5000 | 5.071 | 4.503 | 3.211 | 3.006 | 4.666 | 3.148 | 0.579 |
| 20 | ISM | ism_greedy_115 | 5000 | 3.700 | 3.016 | 2.082 | 1.955 | 5.363 | 3.672 | 0.191 |
| 20 | LEDIDI | pool_A_l015 | 43 | 1.260 | 1.012 | 0.760 | 0.647 | 1.759 | 1.301 | 0.343 |
| 40 | GPA | v3t_nodps_nobio_nogc_b2k | 5000 | 5.527 | 5.000 | 3.623 | 3.620 | 5.175 | 3.727 | 0.074 |
| 40 | ISM | ism_greedy_115 | 5000 | 5.629 | 4.647 | 2.739 | 2.664 | 6.932 | 3.898 | 0.274 |
| 40 | LEDIDI | pool_A_l015 | 3396 | 1.392 | 1.182 | 1.024 | 0.938 | 3.842 | 2.852 | 0.236 |
| 60 | GPA | v3t_nodps_nobio_nogc_b2k | 5000 | 7.749 | 6.701 | 3.275 | 3.226 | 6.952 | 3.453 | 0.308 |
| 60 | ISM | ism_greedy_115 | 5000 | 6.806 | 5.599 | 2.934 | 2.882 | 8.134 | 3.922 | 0.379 |
| 60 | LEDIDI | pool_A_l015 | 11942 | 2.821 | 2.287 | 1.672 | 1.536 | 6.340 | 3.664 | 0.156 |
| 100 | GPA | v3t_nodps_nobio_nogc_b2k | 5000 | 6.751 | 5.935 | 3.084 | 3.039 | 6.979 | 3.500 | 0.319 |
| 100 | ISM | ism_greedy_115 | 5000 | 8.061 | 6.578 | 3.029 | 2.994 | 9.525 | 3.953 | 0.503 |
| 100 | LEDIDI | pool_A_l015 | 11232 | 7.669 | 6.455 | 2.875 | 2.834 | 10.135 | 3.867 | 0.444 |

### Pool pool_A — method-specific tail rows

| method | recipe | cap (edits) | n | mean LN | max LN | mean AG | max AG | 1-Pearson |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPA | v3t_nodps_nobio_nogc_b2k | uncap/all | 5000 | 6.114 | 7.912 | 2.945 | 3.531 | 0.389 |
| ISM | ism_greedy_115 | 115 | 5000 | 8.252 | 11.164 | 3.037 | 4.031 | 0.527 |
| LEDIDI | pool_A_l015 | uncap/all | 95000 | 3.818 | 11.940 | 1.588 | 3.969 | 0.058 |

## Pool: pool_B — matched edit caps

| cap (edits) | method | recipe | n | mean LN | mean LN (RC) | mean AG | mean AG (RC) | max LN (RC) | max AG (RC) | 1-Pearson(LN,AG)_RC |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | GPA | v3t_nodps_nobio_nogc_b2k | 5000 | 5.278 | 5.035 | 3.344 | 3.384 | 5.278 | 3.578 | 0.166 |
| 20 | ISM | ism_greedy_115 | 5000 | 3.874 | 3.183 | 2.274 | 2.131 | 5.281 | 3.727 | 0.174 |
| 20 | LEDIDI | pool_B_l015 | 6 | 1.192 | 0.924 | 0.617 | 0.619 | 1.448 | 1.281 | 0.076 |
| 40 | GPA | v3t_nodps_nobio_nogc_b2k | 5000 | 6.226 | 5.901 | 3.101 | 3.065 | 6.234 | 3.344 | 0.355 |
| 40 | ISM | ism_greedy_115 | 5000 | 5.719 | 4.735 | 2.824 | 2.742 | 6.893 | 3.812 | 0.283 |
| 40 | LEDIDI | pool_B_l015 | 1670 | 1.494 | 1.239 | 1.134 | 1.026 | 3.674 | 2.961 | 0.149 |
| 60 | GPA | v3t_nodps_nobio_nogc_b2k | 5000 | 6.806 | 6.445 | 3.360 | 3.400 | 6.985 | 3.617 | 0.402 |
| 60 | ISM | ism_greedy_115 | 5000 | 6.851 | 5.635 | 2.990 | 2.933 | 7.897 | 3.875 | 0.379 |
| 60 | LEDIDI | pool_B_l015 | 9320 | 3.000 | 2.456 | 1.860 | 1.723 | 6.298 | 3.695 | 0.114 |
| 100 | GPA | v3t_nodps_nobio_nogc_b2k | 5000 | 5.613 | 5.293 | 3.237 | 3.216 | 6.199 | 3.656 | 0.626 |
| 100 | ISM | ism_greedy_115 | 5000 | 8.068 | 6.581 | 3.071 | 3.033 | 8.923 | 3.953 | 0.522 |
| 100 | LEDIDI | pool_B_l015 | 10247 | 7.678 | 6.443 | 2.856 | 2.823 | 10.094 | 3.805 | 0.466 |

### Pool pool_B — method-specific tail rows

| method | recipe | cap (edits) | n | mean LN | max LN | mean AG | max AG | 1-Pearson |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPA | v3t_nodps_nobio_nogc_b2k | uncap/all | 5000 | 5.613 | 6.466 | 3.237 | 3.656 | 0.626 |
| ISM | ism_greedy_115 | 115 | 5000 | 8.252 | 10.840 | 3.074 | 3.984 | 0.545 |
| LEDIDI | pool_B_l015 | uncap/all | 95000 | 4.039 | 11.461 | 1.768 | 4.031 | 0.064 |

## Runtime

| pool | method | recipe | cap | n | total (s) | sec/seq |
|---|---|---|---:|---:|---:|---:|
| pool_A | GPA | v3t_nodps_nobio_nogc_b2k | 20 | 5000 | — | — |
| pool_A | GPA | v3t_nodps_nobio_nogc_b2k | 40 | 5000 | — | — |
| pool_A | GPA | v3t_nodps_nobio_nogc_b2k | 60 | 5000 | — | — |
| pool_A | GPA | v3t_nodps_nobio_nogc_b2k | 100 | 5000 | — | — |
| pool_A | GPA | v3t_nodps_nobio_nogc_b2k | 1000 | 5000 | — | — |
| pool_B | GPA | v3t_nodps_nobio_nogc_b2k | 20 | 5000 | — | — |
| pool_B | GPA | v3t_nodps_nobio_nogc_b2k | 40 | 5000 | — | — |
| pool_B | GPA | v3t_nodps_nobio_nogc_b2k | 60 | 5000 | — | — |
| pool_B | GPA | v3t_nodps_nobio_nogc_b2k | 100 | 5000 | — | — |
| pool_B | GPA | v3t_nodps_nobio_nogc_b2k | 1000 | 5000 | — | — |
| pool_A | ISM | ism_greedy_115 | 20 | 5000 | 1089.9 | 0.218 |
| pool_A | ISM | ism_greedy_115 | 40 | 5000 | 2179.9 | 0.436 |
| pool_A | ISM | ism_greedy_115 | 60 | 5000 | 3269.8 | 0.654 |
| pool_A | ISM | ism_greedy_115 | 100 | 5000 | 5449.7 | 1.090 |
| pool_A | ISM | ism_greedy_115 | 115 | 5000 | 6267.1 | 1.253 |
| pool_B | ISM | ism_greedy_115 | 20 | 5000 | 1155.3 | 0.231 |
| pool_B | ISM | ism_greedy_115 | 40 | 5000 | 2310.5 | 0.462 |
| pool_B | ISM | ism_greedy_115 | 60 | 5000 | 3465.8 | 0.693 |
| pool_B | ISM | ism_greedy_115 | 100 | 5000 | 5776.3 | 1.155 |
| pool_B | ISM | ism_greedy_115 | 115 | 5000 | 6642.7 | 1.329 |
| _smoke_snap | LEDIDI | _smoke_snap | 40 | 13 | 377.6 | 0.539 |
| _smoke_snap | LEDIDI | _smoke_snap | 60 | 163 | 377.6 | 0.539 |
| _smoke_snap | LEDIDI | _smoke_snap | 100 | 22 | 377.6 | 0.539 |
| _smoke_snap | LEDIDI | _smoke_snap | 1000 | 700 | — | — |
| _smoke_snap_v2 | LEDIDI | _smoke_snap_v2 | 20 | 2 | 3362.0 | 2.101 |
| _smoke_snap_v2 | LEDIDI | _smoke_snap_v2 | 40 | 70 | 3362.0 | 2.101 |
| _smoke_snap_v2 | LEDIDI | _smoke_snap_v2 | 60 | 257 | 3362.0 | 2.101 |
| _smoke_snap_v2 | LEDIDI | _smoke_snap_v2 | 100 | 92 | 3362.0 | 2.101 |
| _smoke_snap_v2 | LEDIDI | _smoke_snap_v2 | 1000 | 1600 | — | — |
| pool_A | LEDIDI | pool_A_l015 | 20 | 43 | 993902.4 | 10.462 |
| pool_A | LEDIDI | pool_A_l015 | 40 | 3396 | 993902.4 | 10.462 |
| pool_A | LEDIDI | pool_A_l015 | 60 | 11942 | 993902.4 | 10.462 |
| pool_A | LEDIDI | pool_A_l015 | 100 | 11232 | 993902.4 | 10.462 |
| pool_A | LEDIDI | pool_A_l015 | 1000 | 95000 | — | — |
| pool_B | LEDIDI | pool_B_l015 | 20 | 6 | 828498.9 | 8.721 |
| pool_B | LEDIDI | pool_B_l015 | 40 | 1670 | 828498.9 | 8.721 |
| pool_B | LEDIDI | pool_B_l015 | 60 | 9320 | 828498.9 | 8.721 |
| pool_B | LEDIDI | pool_B_l015 | 100 | 10247 | 828498.9 | 8.721 |
| pool_B | LEDIDI | pool_B_l015 | 1000 | 95000 | — | — |
