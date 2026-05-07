#!/usr/bin/env python3
"""Resume — THP1 only (CtrlDNA + GPA universal) for top128_by_ΔR_norm + FIMO."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import quick_top128_by_deltaR as Q
import pandas as pd
import numpy as np


def main():
    Q.init_fimo()
    print(f"\n{'Cell':<8} {'Method':<14} target_raw     ΔR_norm        shannon       motif_corr_FIMO")
    print("-" * 105)
    out_rows = []
    cell = "THP1"
    for method, fn in [("CtrlDNA", Q.ctrldna_top128_by_delta_R), ("GPA universal", Q.gpa_top128_by_delta_R)]:
        seed_results = []
        for seed in range(5):
            r = fn(cell, seed)
            if r:
                seed_results.append(r)
        if not seed_results:
            continue
        agg = {k: (np.mean([r[k] for r in seed_results]),
                   np.std([r[k] for r in seed_results]))
               for k in seed_results[0].keys()}
        f = lambda k: f"{agg[k][0]:+.2f}±{agg[k][1]:.2f}"
        print(f"{cell:<8} {method:<14} {f('target_raw')}  {f('delta_R_norm')}  {f('shannon')}  {f('motif_corr')}", flush=True)
        out_rows.append({"cell": cell, "method": method, **{k: v[0] for k, v in agg.items()},
                         **{f"{k}_std": v[1] for k, v in agg.items()}})
    df = pd.DataFrame(out_rows)
    out_csv = Q.RESULTS / "top128_by_deltaR_norm_universal_THP1.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
