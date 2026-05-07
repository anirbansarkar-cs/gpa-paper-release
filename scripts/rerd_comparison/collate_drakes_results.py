#!/usr/bin/env python3
"""Collate per-pool DRAKES-protocol JSONs into a paper-ready table that
matches Pani-Ou-Li 2025 v4 Table 5 format.

Usage:
  python collate_drakes_results.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('${GPA_REPO_ROOT}/results/rerd_comparison/drakes_protocol')
RECIPES = {
    'GPA-DPS Mean (rand×4)': ['run_paper_mean_random', 'run_paper_mean_random_r2',
                              'run_paper_mean_random_r3', 'run_paper_mean_random_r4'],
    'GPA-DPS Mean (seeded)': ['run_paper_mean_seeded'],
    'GPA-DPS Spec (rand×4)': ['run_paper_spec_random', 'run_paper_spec_random_r2',
                              'run_paper_spec_random_r3', 'run_paper_spec_random_r4'],
    'GPA-DPS Spec (seeded)': ['run_paper_spec_seeded'],
    'GPA-noDPS Mean pw=0 (rand×3)': ['run_v17_nodps_mean_r1', 'run_v17_nodps_mean_r2', 'run_v17_nodps_mean_r3'],
    'GPA-noDPS Mean pw=0 (seeded)': ['run_v17_nodps_mean_seeded'],
    'GPA-noDPS Mean pw=0.10 (rand×2)': ['run_v17b_nodps_pw010_r1', 'run_v17b_nodps_pw010_r2'],
    'GPA-noDPS Spec pw=0.35 (rand×3)': ['run_v17_nodps_spec_r1', 'run_v17_nodps_spec_r2', 'run_v17_nodps_spec_r3'],
    'GPA-noDPS Spec pw=0.35 (seeded)': ['run_v17_nodps_spec_seeded'],
}

# Pani-Ou-Li v4 Table 5 baselines (verbatim)
BASELINES = [
    ('Pretrained',           0.17, 1.5,   -0.061, 0.249, -261),
    ('CG',                   3.30, 0.0,   -0.065, 0.212, -266),
    ('CFG',                  5.04, 92.1,   0.746, 0.864, -265),
    ('DRAKES (no KL)',       6.44, 82.5,   0.307, 0.557, -281),
    ('DRAKES',               5.61, 92.5,   0.887, 0.911, -264),
    ('SGDD (β=30)',          8.85, 90.9,   0.470, 0.466, -263),
    ('SGDD (β=50)',          9.32, 96.4,   0.370, 0.398, -269),
    ('SVDD (N=8)',           6.57, 67.4,   0.813, 0.753, -258),
    ('SVDD (N=16)',          6.89, 84.3,   0.891, 0.834, -260),
    ('SMC_amot (N=1)',       5.40, 82.1,   0.653, 0.778, -259),
    ('SMC_amot (N=8)',       6.35, 95.8,   0.736, 0.845, -261),
    ('SMC_amot (N=16)',      6.68, 97.6,   0.796, 0.886, -261),
]

def load_pool(name):
    p = ROOT / f'{name}.json'
    if not p.exists():
        return None
    return json.load(open(p))


def aggregate(recipe_label, pool_names):
    rows = []
    for n in pool_names:
        d = load_pool(n)
        if d is None:
            print(f'  SKIP {n}: not found at {ROOT}')
            continue
        rows.append([
            d.get('pred_activity_median'),
            d.get('atac_acc_pct'),
            d.get('kmer3_pearson'),
            d.get('jaspar_pearson'),
            d.get('app_log_lik_median'),
        ])
    arr = np.array([[v if v is not None else np.nan for v in r] for r in rows], dtype=float)
    if len(arr) == 0:
        return None
    if len(arr) == 1:
        return {'mean': arr[0], 'std': np.zeros(arr.shape[1]), 'n': 1}
    return {'mean': arr.mean(axis=0), 'std': arr.std(axis=0), 'n': len(arr)}


def fmt_cell(mean, std, n):
    if np.isnan(mean):
        return '—'
    if n > 1:
        return f'{mean:+.3f}±{std:.3f}'.replace('+', '')
    return f'{mean:.3f}'


def main():
    print('=' * 110)
    print('DRAKES-protocol comparison (Pani-Ou-Li 2025 v4 Table 5 format)')
    print('=' * 110)
    cols = ['Pred-Activity ↑', 'ATAC-Acc ↑', '3-mer Corr ↑', 'JASPAR Corr ↑', 'App-Log-Lik ↑']
    print(f'{"Method":<26} {"N":>3} ' + ' '.join(f'{c:>14}' for c in cols))
    print('-' * 110)

    # Baselines (from Pani v4 Table 5)
    for name, pa, atac, k3, jas, app in BASELINES:
        print(f'{name:<26} {"--":>3} ' + ' '.join(f'{v:>14.3f}' for v in [pa, atac, k3, jas, app]))
    print('-' * 110)

    # Our GPA rows
    rows_csv = []
    for label, pool_names in RECIPES.items():
        agg = aggregate(label, pool_names)
        if agg is None:
            print(f'{label:<26} {"--":>3} (no data yet)')
            rows_csv.append({'method': label, 'n': 0})
            continue
        print(f'{label:<26} {agg["n"]:>3} ' + ' '.join(fmt_cell(m, s, agg['n']).rjust(14) for m, s in zip(agg['mean'], agg['std'])))
        rows_csv.append({
            'method': label, 'n': agg['n'],
            'pred_activity': agg['mean'][0], 'pred_activity_std': agg['std'][0],
            'atac_acc': agg['mean'][1], 'atac_acc_std': agg['std'][1],
            'kmer3_corr': agg['mean'][2], 'kmer3_corr_std': agg['std'][2],
            'jaspar_corr': agg['mean'][3], 'jaspar_corr_std': agg['std'][3],
            'app_log_lik': agg['mean'][4], 'app_log_lik_std': agg['std'][4],
        })

    df = pd.DataFrame(rows_csv)
    csv_path = ROOT / 'drakes_protocol_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nWrote {csv_path}')


if __name__ == '__main__':
    main()
