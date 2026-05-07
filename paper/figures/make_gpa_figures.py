"""Generate all GPA paper figures (landscape small-multiples + walkthrough).

Mirrors the visual style of GPA_Introduction.pdf slides 7-21:
 - 3D landscape with green prior surface + orange oracle wireframe
 - Particles colored by fitness (blue -> orange -> red)
 - Walkthrough bars / ESS curve / resampling bubbles match slide colors

Run:  ${HOME}/.conda/envs/d3_cuda118/bin/python figures/make_gpa_figures.py
Outputs (PDFs alongside this script):
  fig_gpa_landscape_smallmults.pdf   -> Figure 1 right half
  fig_gpa_walkthrough_score.pdf      -> Appendix C, Step 1+2
  fig_gpa_walkthrough_ess.pdf        -> Appendix C, ESS bisection
  fig_gpa_walkthrough_resample.pdf   -> Appendix C, multinomial resample
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers projection

# ---------------------------------------------------------------------------
# Style: matches slide deck. Sans-serif, light gray frames, slide-deck palette.
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#888",
    "axes.labelcolor": "#222",
    "xtick.color": "#444",
    "ytick.color": "#444",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

PRIOR_GREEN  = "#7FBF7F"          # filled prior surface
PRIOR_EDGE   = "#3E8E3E"
ORACLE_ORG   = "#F4A340"          # oracle wireframe / dashed line
LOW_BLUE     = "#5B9BD5"          # low-fitness particles
MID_ORANGE   = "#ED7D31"          # mid-fitness particles
HIGH_RED     = "#C0392B"          # high-fitness particles
TARGET_GREEN = "#62B16D"          # walkthrough "highest" highlight
LOWEST_RED   = "#E36F6F"
TXT_GRAY     = "#5A5A5A"

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Shared landscape: prior p_theta(x, y) and oracle r(x, y)
# ---------------------------------------------------------------------------
def gauss(X, Y, mx, my, sx, sy, h):
    return h * np.exp(-((X - mx) ** 2) / (2 * sx ** 2)
                      - ((Y - my) ** 2) / (2 * sy ** 2))

def prior_surface(X, Y):
    z = gauss(X, Y, mx=-7,  my=-2, sx=4.5, sy=3.5, h=1.00)   # Start Valley (tallest)
    z += gauss(X, Y, mx= 6, my= 2.5, sx=2.5, sy=2.0, h=0.45)  # Main Target
    z += gauss(X, Y, mx= 2, my=-4, sx=1.6, sy=1.5, h=0.28)    # Secondary Target
    return z

def oracle_surface(X, Y):
    z  = gauss(X, Y, mx= 6, my= 2.5, sx=1.8, sy=1.6, h=0.95)  # Main Target (oracle peak)
    z += gauss(X, Y, mx= 2, my=-4, sx=1.2, sy=1.2, h=0.70)    # Secondary Target
    z += gauss(X, Y, mx=-3, my= 6, sx=1.5, sy=1.5, h=0.55)    # OOD Trap (no prior support)
    return z

# Coordinates of feature centers (used to place labels and particle clusters)
START_VALLEY = (-7, -2)
MAIN_TARGET  = ( 6,  2.5)
SECOND_TGT   = ( 2, -4)
OOD_TRAP     = (-3,  6)

def landscape_panel(ax, particles_xy, particle_colors, title, beta_label,
                    show_labels=True, draw_stars=None,
                    particle_sizes=None):
    """Render one 3D landscape panel."""
    x = np.linspace(-12, 12, 100)
    y = np.linspace(-9, 9, 90)
    X, Y = np.meshgrid(x, y)
    Zp = prior_surface(X, Y)
    Zr = oracle_surface(X, Y)

    # green prior surface (filled, with subtle shading)
    ax.plot_surface(X, Y, Zp, color=PRIOR_GREEN, alpha=0.62,
                    edgecolor=PRIOR_EDGE, linewidth=0.18, antialiased=True,
                    rstride=2, cstride=2, shade=True)
    # orange oracle wireframe (no fill, more transparent so particles show through)
    ax.plot_wireframe(X, Y, Zr, color=ORACLE_ORG, linewidth=0.45,
                      rstride=5, cstride=5, alpha=0.55)

    # particles: scatter on top of the prior surface at their (x,y)
    # Z lifted high enough to sit above the oracle wireframe at peaks.
    if len(particles_xy) > 0:
        px, py = zip(*particles_xy)
        px = np.array(px); py = np.array(py)
        pz = np.maximum(prior_surface(px, py), oracle_surface(px, py)) + 0.12
        sizes = particle_sizes if particle_sizes is not None else 60
        ax.scatter(px, py, pz, c=particle_colors, s=sizes, depthshade=False,
                   edgecolor="#202020", linewidths=0.55, zorder=10, alpha=0.95)

    # mutation stars (panel 3): drawn much larger
    if draw_stars:
        sx, sy_, sc = zip(*draw_stars)
        sx = np.array(sx); sy_ = np.array(sy_)
        sz = prior_surface(sx, sy_) + 0.18
        ax.scatter(sx, sy_, sz, c=sc, marker="*", s=180, depthshade=False,
                   edgecolor="#202020", linewidths=0.6, zorder=11)

    # peak labels — staggered positions so they don't collide in projection
    if show_labels:
        for (lx, ly, dz), txt in [
            ((-9.5, -2.5, 0.32),  "Start Valley"),
            ((  9,   3.5, 0.32),  "Main Target"),
            ((  2.5, -6, 0.18),   "Secondary"),
            (( -7,   8.5, 0.10),  "OOD Trap"),
        ]:
            zt = max(prior_surface(np.array(lx), np.array(ly)),
                     oracle_surface(np.array(lx), np.array(ly))) + dz
            ax.text(lx, ly, zt, txt, fontsize=7.5, color=TXT_GRAY,
                    ha="center", zorder=12)

    # cosmetics — exaggerate Z to make peaks visible
    ax.set_xlim(-12, 12); ax.set_ylim(-9, 9); ax.set_zlim(0, 1.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect((24, 18, 14))
    ax.view_init(elev=32, azim=-58)
    # transparent axis panes
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor("#cccccc")
        pane.pane.set_alpha(0.22)
    ax.grid(False)
    # title — anchored close to top of subplot
    ax.text2D(0.02, 0.96, title, transform=ax.transAxes,
              fontsize=11, fontweight="bold", color="#222",
              ha="left", va="top")
    ax.text2D(0.98, 0.96, beta_label, transform=ax.transAxes,
              fontsize=10, color=TXT_GRAY, ha="right", va="top")


# ---------------------------------------------------------------------------
# Particle distributions for each phase
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)

def sample_around(center, n, sigma=1.2):
    cx, cy = center
    return list(zip(cx + rng.normal(0, sigma, n),
                    cy + rng.normal(0, sigma * 0.7, n)))

def fig_landscape_smallmults():
    fig = plt.figure(figsize=(7.6, 4.6))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0,
                        wspace=-0.10, hspace=-0.05)

    # --- Panel 1: Init -----------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    init_particles = sample_around(START_VALLEY, 28, sigma=1.4)
    init_particles += sample_around(MAIN_TARGET, 2, sigma=0.6)
    init_particles += sample_around(SECOND_TGT, 1, sigma=0.4)
    init_colors = [LOW_BLUE] * 28 + [MID_ORANGE, HIGH_RED, MID_ORANGE]
    landscape_panel(ax1, init_particles, init_colors,
                    "1. Init", r"$\beta = 0$")

    # --- Panel 2: Reweight + Resample --------------------------------------
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    rw_particles = sample_around(START_VALLEY, 3, sigma=0.8)
    rw_particles += sample_around(MAIN_TARGET, 14, sigma=0.55)
    rw_particles += sample_around(SECOND_TGT, 7, sigma=0.5)
    rw_colors = (["#cfcfcf"] * 3 + [HIGH_RED] * 14 + [MID_ORANGE] * 7)
    rw_sizes  = ([20] * 3 + [85] * 14 + [70] * 7)
    landscape_panel(ax2, rw_particles, rw_colors,
                    "2. Reweight", r"$\beta\,\nearrow$",
                    particle_sizes=rw_sizes)

    # --- Panel 3: Mutate ---------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    mut_particles = sample_around(START_VALLEY, 3, sigma=0.9)
    mut_particles += sample_around(MAIN_TARGET, 9, sigma=0.85)
    mut_particles += sample_around(SECOND_TGT, 6, sigma=0.7)
    mut_colors = ([LOW_BLUE] * 3 + [HIGH_RED] * 9 + [MID_ORANGE] * 6)
    mut_sizes  = ([45] * 3 + [80] * 9 + [70] * 6)
    # mutation stars: outward burst from parent at (0, -1)
    stars = []
    parent = (0.5, -0.5)
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        rr = 4.5
        x = parent[0] + rr * np.cos(ang)
        y = parent[1] + rr * np.sin(ang) * 0.75
        x = float(np.clip(x, -10.5, 10.5))
        y = float(np.clip(y, -8, 8))
        oracle_val = float(oracle_surface(np.array([x]), np.array([y])))
        if oracle_val > 0.35:
            color = HIGH_RED
        elif oracle_val > 0.15:
            color = MID_ORANGE
        else:
            color = LOW_BLUE
        stars.append((x, y, color))
    landscape_panel(ax3, mut_particles, mut_colors,
                    "3. Mutate", r"$\beta = \beta_t$",
                    draw_stars=stars, particle_sizes=mut_sizes)

    # --- Panel 4: Converge -------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    # tightly clustered around Main Target — many overlapping particles
    conv_particles = sample_around(MAIN_TARGET, 35, sigma=0.45)
    conv_particles += sample_around(SECOND_TGT, 2, sigma=0.4)
    conv_colors = [HIGH_RED] * 35 + [MID_ORANGE] * 2
    conv_sizes  = [110] * 35 + [60] * 2
    landscape_panel(ax4, conv_particles, conv_colors,
                    "4. Converge", r"$\beta \to \beta_*$",
                    particle_sizes=conv_sizes)

    fig.savefig(OUT / "fig_gpa_landscape_smallmults.pdf", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Walkthrough Step 1+2: Score all particles + Reweight
# ---------------------------------------------------------------------------
def fig_walkthrough_score():
    scores = np.array([-0.45, -0.38, -0.52, -0.31, -0.48, -0.25, -0.41, -0.35])
    labels = [f"$x_{{{i+1}}}$" for i in range(8)]
    delta_beta = 15
    raw = np.exp(delta_beta * scores)
    weights = raw / raw.sum()
    uniform = 1.0 / 8

    fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(8.5, 3.0),
                                     gridspec_kw={"wspace": 0.28})
    y = np.arange(8)[::-1]  # x_1 on top

    # --- left: scores ---
    bar_colors = []
    for i, s in enumerate(scores):
        if s == scores.min():
            bar_colors.append(LOWEST_RED)
        elif s == scores.max():
            bar_colors.append(TARGET_GREEN)
        else:
            bar_colors.append(LOW_BLUE)
    ax_s.barh(y, scores, color=bar_colors, edgecolor="none", height=0.65)
    ax_s.set_yticks(y); ax_s.set_yticklabels(labels, fontsize=10)
    for yi, s in zip(y, scores):
        ax_s.text(s - 0.012, yi, f"{s:.2f}", va="center", ha="right",
                  fontsize=8.5, color="#333")
    ax_s.axvline(0, color="#888", linewidth=0.5)
    ax_s.set_xlim(-0.6, 0.05)
    ax_s.set_xlabel("oracle score $s_i$", fontsize=9)
    ax_s.set_title("Step 1: Score all particles",
                   loc="left", fontsize=11, color=ORACLE_ORG, fontweight="bold")
    ax_s.spines["left"].set_visible(False)

    # --- right: weights ---
    bar_colors_w = []
    for w in weights:
        if w == weights.max():
            bar_colors_w.append(TARGET_GREEN)
        elif w >= uniform:
            bar_colors_w.append(MID_ORANGE)
        else:
            bar_colors_w.append(LOWEST_RED)
    ax_w.barh(y, weights, color=bar_colors_w, edgecolor="none", height=0.65)
    ax_w.set_yticks(y); ax_w.set_yticklabels(labels, fontsize=10)
    for yi, w in zip(y, weights):
        ax_w.text(w + 0.005, yi, f"{w:.3f}", va="center", ha="left",
                  fontsize=8.5, color="#333")
    ax_w.axvline(uniform, color="#888", linestyle="--", linewidth=0.8)
    ax_w.text(uniform + 0.003, 7.6, f"uniform\n({uniform:.3f})", fontsize=7.5,
              color="#888", ha="left", va="top")
    ax_w.set_xlim(0, 0.6)
    ax_w.set_xlabel("normalised weight $w_i$", fontsize=9)
    ax_w.set_title(r"Step 2: Reweight $w_i\propto e^{\Delta\beta\,s_i}$, $\Delta\beta=15$",
                   loc="left", fontsize=11, color=LOW_BLUE, fontweight="bold")
    ax_w.spines["left"].set_visible(False)

    fig.suptitle(r"$x_6$ (best score $-0.25$) $\to$ weight $51.0\%$ "
                 r"— $4.1\times$ more likely to survive than uniform",
                 fontsize=9, color=TARGET_GREEN, y=-0.04)

    fig.savefig(OUT / "fig_gpa_walkthrough_score.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Walkthrough: ESS bisection curve
# ---------------------------------------------------------------------------
def fig_walkthrough_ess():
    scores = np.array([-0.45, -0.38, -0.52, -0.31, -0.48, -0.25, -0.41, -0.35])
    N = len(scores)
    betas = np.linspace(0.0, 40.0, 401)
    ess_vals = []
    for b in betas:
        w = np.exp(b * scores - (b * scores).max())
        w /= w.sum()
        ess_vals.append(1.0 / np.sum(w * w))
    ess_vals = np.array(ess_vals)

    target_ess = 4.0  # tau * N = 0.5 * 8
    # find auto-pick beta
    auto_beta = float(np.interp(-target_ess, -ess_vals, betas))  # ess monotone-decreasing
    # our example
    eg_beta = 15.0
    eg_w = np.exp(eg_beta * scores - (eg_beta * scores).max())
    eg_w /= eg_w.sum()
    eg_ess = 1.0 / np.sum(eg_w * eg_w)

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(betas, ess_vals, color=LOW_BLUE, linewidth=2.0, zorder=2)
    ax.axhline(target_ess, color=TARGET_GREEN, linestyle="--", linewidth=1.0,
               label=fr"$\tau\,N = {target_ess:.0f}$", zorder=1)
    ax.axhline(N, color="#bbb", linestyle=":", linewidth=0.8, zorder=1)
    ax.text(40, N - 0.15, f"$N = {N}$", fontsize=8, color="#888",
            ha="right", va="top")
    ax.axhline(1, color="#e0a4a4", linestyle=":", linewidth=0.8, zorder=1)
    ax.text(40, 1 + 0.15, "ESS = 1", fontsize=8, color="#cc6666",
            ha="right", va="bottom")

    # auto-pick green dot
    ax.scatter([auto_beta], [target_ess], s=70, color=TARGET_GREEN,
               edgecolor="#1f5d2c", linewidth=0.8, zorder=5)
    ax.annotate(fr"$\Delta\beta = {auto_beta:.1f}$" + "\n" + fr"ESS $= {target_ess:.0f}$",
                xy=(auto_beta, target_ess), xytext=(auto_beta + 4, target_ess + 1.2),
                fontsize=9, color="#1f5d2c", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#1f5d2c", linewidth=0.8))

    # our-example orange square
    ax.scatter([eg_beta], [eg_ess], s=60, marker="s", color=MID_ORANGE,
               edgecolor="#7a3a0d", linewidth=0.8, zorder=5)
    ax.annotate(fr"Our example: $\Delta\beta = {eg_beta:.0f}$" + "\n"
                + fr"ESS $= {eg_ess:.1f}$",
                xy=(eg_beta, eg_ess), xytext=(eg_beta + 5, eg_ess - 1.0),
                fontsize=9, color="#7a3a0d",
                arrowprops=dict(arrowstyle="-", color="#7a3a0d", linewidth=0.8))

    ax.set_xlabel(r"$\Delta\beta$", fontsize=10)
    ax.set_ylabel("ESS", fontsize=10)
    ax.set_xlim(0, 40); ax.set_ylim(0, 9)
    ax.set_title(r"ESS$(\Delta\beta)$ — bisection auto-picks $\Delta\beta$ at target",
                 loc="left", fontsize=11, color="#222", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.savefig(OUT / "fig_gpa_walkthrough_ess.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Walkthrough: multinomial / systematic resampling
# ---------------------------------------------------------------------------
def fig_walkthrough_resample():
    weights = np.array([0.025, 0.073, 0.009, 0.207, 0.016, 0.510, 0.046, 0.114])
    survives = weights >= 0.04
    counts = np.array([0, 1, 0, 2, 0, 4, 0, 1])
    labels = [f"$x_{{{i+1}}}$" for i in range(8)]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.set_xlim(0, 10); ax.set_ylim(-1.2, 8.5)
    ax.set_aspect("equal")  # circles render as circles
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    y_positions = np.arange(8)[::-1]

    # --- left column: before (size ∝ weight) ---
    for yi, lab, w, surv in zip(y_positions, labels, weights, survives):
        radius = 0.06 + np.sqrt(w) * 0.55
        circle = plt.Circle((2.4, yi), radius=radius,
                            color=TARGET_GREEN if surv else LOWEST_RED,
                            alpha=0.88, zorder=3,
                            ec="#333", linewidth=0.4)
        ax.add_patch(circle)
        ax.text(1.4, yi, lab, fontsize=10, va="center", ha="right",
                color="#333")
        ax.text(3.4, yi, f"{w:.3f}", fontsize=9, va="center", ha="left",
                color="#333")

    ax.text(2.4, 8.0, r"Before (size $\propto w_i$)",
            fontsize=10, ha="center", color="#222", fontweight="bold")

    # --- arrow ---
    arrow = FancyArrowPatch((5.0, 3.5), (6.4, 3.5),
                            arrowstyle="-|>", mutation_scale=18,
                            color="#444", linewidth=1.5)
    ax.add_patch(arrow)
    ax.text(5.7, 3.95, "draw\n$N=8$", fontsize=8.5, color="#444",
            ha="center", va="bottom")

    # --- right column: after (8 uniform-weight blue bubbles) ---
    out_idx = []
    for i, c in enumerate(counts):
        out_idx.extend([i] * int(c))
    out_y = np.arange(8)[::-1]
    out_x = 7.8
    radius_after = 0.32
    for yi, src in zip(out_y, out_idx):
        circle = plt.Circle((out_x, yi), radius=radius_after,
                            color=LOW_BLUE, alpha=0.88, zorder=3,
                            ec="#333", linewidth=0.4)
        ax.add_patch(circle)
        ax.text(out_x, yi, labels[src], fontsize=8.5, va="center", ha="center",
                color="white", fontweight="bold")

    grouped = [(i, c) for i, c in enumerate(counts) if c > 0]
    cum = 0
    for src, c in grouped:
        first_y = out_y[cum]
        last_y = out_y[cum + c - 1]
        mid_y = (first_y + last_y) / 2
        ax.text(out_x + 0.55, mid_y, fr"$\times {c}$",
                fontsize=10, va="center", ha="left", color="#444",
                fontweight="bold")
        cum += c

    ax.text(out_x, 8.0, "After (4 survive, 4 eliminated)",
            fontsize=10, ha="center", color="#222", fontweight="bold")
    ax.text(5.0, -1.0, r"Systematic resampling: $N$ pointers spaced $1/N$ apart "
            r"land in segments of width $w_i$",
            fontsize=8.5, ha="center", color=TXT_GRAY, style="italic")

    fig.savefig(OUT / "fig_gpa_walkthrough_resample.pdf")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating GPA paper figures...")
    fig_landscape_smallmults()
    print("  -> fig_gpa_landscape_smallmults.pdf")
    fig_walkthrough_score()
    print("  -> fig_gpa_walkthrough_score.pdf")
    fig_walkthrough_ess()
    print("  -> fig_gpa_walkthrough_ess.pdf")
    fig_walkthrough_resample()
    print("  -> fig_gpa_walkthrough_resample.pdf")
    print("Done.")
