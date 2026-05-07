"""Canonical DNA-CRAFT paper Table 2 numbers (arXiv 2604.20488).

Single source of truth for all downstream scripts. Format:
  PAPER_TABLE2[(cell, method)][metric_key] = (mean, std)
where metric_key ∈ {"mingap", "motif", "kmer3", "div"} and cell uses the
display label "SK-N-SH" (not the internal "SKNSH").

If you find a discrepancy with the published table, fix it HERE — every
report script imports from this module.
"""
from __future__ import annotations

PAPER_METHODS = ["SMC", "CG", "TDS", "DRAKES", "D3", "Ledidi", "Ctrl-DNA", "DNA-CRAFT"]
PAPER_CELLS = ["HepG2", "K562", "SK-N-SH"]

PAPER_TABLE2: dict[tuple[str, str], dict[str, tuple[float, float]]] = {
    # ── HepG2 ────────────────────────────────────────────────────────────
    ("HepG2", "SMC"):       {"mingap": (1.614, 1.665), "motif": (0.554, 0.049), "kmer3": (0.808, 0.102), "div": (0.828, 0.432)},
    ("HepG2", "CG"):        {"mingap": (-0.226, 0.096), "motif": (0.860, 0.009), "kmer3": (0.968, 0.003), "div": (1.976, 0.002)},
    ("HepG2", "TDS"):       {"mingap": (0.404, 0.569), "motif": (0.397, 0.096), "kmer3": (0.744, 0.098), "div": (0.956, 0.096)},
    ("HepG2", "DRAKES"):    {"mingap": (-1.401, 0.054), "motif": (0.057, 0.013), "kmer3": (-0.361, 0.012), "div": (1.864, 0.002)},
    ("HepG2", "D3"):        {"mingap": (0.046, 0.026), "motif": (0.869, 0.011), "kmer3": (0.975, 0.001), "div": (1.976, 0.004)},
    ("HepG2", "Ledidi"):    {"mingap": (5.771, 0.053), "motif": (0.584, 0.025), "kmer3": (0.755, 0.013), "div": (1.981, 0.001)},
    ("HepG2", "Ctrl-DNA"):  {"mingap": (7.786, 0.070), "motif": (0.629, 0.045), "kmer3": (0.494, 0.028), "div": (1.897, 0.026)},
    ("HepG2", "DNA-CRAFT"): {"mingap": (4.346, 0.050), "motif": (0.921, 0.006), "kmer3": (0.980, 0.009), "div": (1.979, 0.000)},
    # ── K562 ─────────────────────────────────────────────────────────────
    ("K562", "SMC"):        {"mingap": (4.124, 0.893), "motif": (0.454, 0.025), "kmer3": (0.659, 0.133), "div": (0.309, 0.112)},
    ("K562", "CG"):         {"mingap": (-0.003, 0.046), "motif": (0.849, 0.026), "kmer3": (0.940, 0.010), "div": (1.977, 0.001)},
    ("K562", "TDS"):        {"mingap": (1.622, 1.611), "motif": (0.511, 0.130), "kmer3": (0.647, 0.198), "div": (0.637, 0.523)},
    ("K562", "DRAKES"):     {"mingap": (-0.202, 0.067), "motif": (0.143, 0.024), "kmer3": (-0.354, 0.007), "div": (1.958, 0.003)},
    ("K562", "D3"):         {"mingap": (0.178, 0.066), "motif": (0.861, 0.041), "kmer3": (0.964, 0.018), "div": (1.977, 0.003)},
    ("K562", "Ledidi"):     {"mingap": (7.662, 0.154), "motif": (0.647, 0.039), "kmer3": (0.689, 0.022), "div": (1.980, 0.001)},
    ("K562", "Ctrl-DNA"):   {"mingap": (9.067, 0.170), "motif": (0.634, 0.084), "kmer3": (0.413, 0.058), "div": (1.896, 0.021)},
    ("K562", "DNA-CRAFT"):  {"mingap": (5.686, 0.043), "motif": (0.933, 0.010), "kmer3": (0.976, 0.000), "div": (1.981, 0.001)},
    # ── SK-N-SH ──────────────────────────────────────────────────────────
    ("SK-N-SH", "SMC"):       {"mingap": (0.556, 0.146), "motif": (0.519, 0.155), "kmer3": (0.775, 0.035), "div": (1.269, 0.108)},
    ("SK-N-SH", "CG"):        {"mingap": (-0.278, 0.006), "motif": (0.855, 0.026), "kmer3": (0.949, 0.007), "div": (1.976, 0.002)},
    ("SK-N-SH", "TDS"):       {"mingap": (0.186, 0.332), "motif": (0.476, 0.092), "kmer3": (0.719, 0.030), "div": (0.918, 0.211)},
    ("SK-N-SH", "DRAKES"):    {"mingap": (0.094, 0.046), "motif": (0.226, 0.017), "kmer3": (-0.382, 0.001), "div": (1.826, 0.001)},
    ("SK-N-SH", "D3"):        {"mingap": (-0.007, 0.006), "motif": (0.836, 0.009), "kmer3": (0.931, 0.011), "div": (1.969, 0.001)},
    ("SK-N-SH", "Ledidi"):    {"mingap": (3.026, 0.222), "motif": (0.380, 0.043), "kmer3": (0.366, 0.019), "div": (1.981, 0.002)},
    ("SK-N-SH", "Ctrl-DNA"):  {"mingap": (3.720, 0.179), "motif": (0.477, 0.037), "kmer3": (0.201, 0.172), "div": (1.855, 0.091)},
    ("SK-N-SH", "DNA-CRAFT"): {"mingap": (3.230, 0.022), "motif": (0.881, 0.031), "kmer3": (0.969, 0.007), "div": (1.976, 0.002)},
}

# Means-only view, for legacy callers that don't need stds.
PAPER_BASELINES: dict[tuple[str, str], dict[str, float]] = {
    key: {m: v[0] for m, v in metrics.items()}
    for key, metrics in PAPER_TABLE2.items()
}
