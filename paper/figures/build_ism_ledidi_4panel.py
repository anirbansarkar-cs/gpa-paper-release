#!/usr/bin/env python3
"""Assemble the 4-panel ISM/LEDIDI figure for paper Sec. 6.3.

Layout (2 rows x 2 panels):
  Row 1 (Pool A, uid=181692): per-pool comparison plot (left), joint rank Pool A (right)
  Row 2 (Pool B, uid=50178):  per-pool comparison plot (left), joint rank Pool B (right)

Source images come from the addendum block of GPA_vs_ISM_LEDIDI_report.docx,
extracted to /tmp/ism_ledidi_imgs/ (image6 = per-pool A, image7 = per-pool B,
image8 = joint-rank combined wide image; we split image8 into A|B halves).

Output: figures/fig_ism_ledidi_4panel.pdf, sized for ~0.66 column width in main.tex.
"""
import shutil
import zipfile
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOCX = Path("${GPA_REPO_ROOT}/results/GPA_vs_ISM_LEDIDI_report.docx")
WORK = Path("/tmp/ism_ledidi_imgs")
OUT = Path(__file__).parent / "fig_ism_ledidi_4panel.pdf"


def extract_images():
    WORK.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DOCX) as z:
        for n in z.namelist():
            if n.startswith("word/media/"):
                target = WORK / Path(n).name
                with z.open(n) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def build_figure():
    img_per_pool_A = Image.open(WORK / "image6.png")  # per-pool Pool A
    img_per_pool_B = Image.open(WORK / "image7.png")  # per-pool Pool B
    img_joint = Image.open(WORK / "image8.png")       # combined joint-rank A|B
    w, h = img_joint.size
    img_joint_A = img_joint.crop((0, 0, w // 2, h))
    img_joint_B = img_joint.crop((w // 2, 0, w, h))

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.4),
                             gridspec_kw={"width_ratios": [1.15, 1.0],
                                          "wspace": 0.04, "hspace": 0.18})
    panels = [
        (axes[0, 0], img_per_pool_A, "Pool A — uid=181692 (natural seed)\nper-pool comparison"),
        (axes[0, 1], img_joint_A,    "Pool A — joint rank score"),
        (axes[1, 0], img_per_pool_B, "Pool B — uid=50178 (random seed)\nper-pool comparison"),
        (axes[1, 1], img_joint_B,    "Pool B — joint rank score"),
    ]
    for ax, im, title in panels:
        ax.imshow(im)
        ax.set_title(title, fontsize=9.5, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    fig.savefig(OUT, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    extract_images()
    build_figure()
