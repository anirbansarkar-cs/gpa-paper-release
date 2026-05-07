"""Build the GPA vs ISM/LEDIDI report as a Word document.

Output: results/GPA_vs_ISM_LEDIDI_report.docx
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


HEADER_FILL = "1F4E79"
ZEBRA_FILL = "F2F2F2"
SUBHEADER_FILL = "D9E1F2"


def shade_cell(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def set_cell_borders(cell) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:color"), "BFBFBF")
        borders.append(b)
    tc_pr.append(borders)


def add_heading(doc: Document, text: str, level: int) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)


def add_para(doc: Document, text: str, *, bold: bool = False, italic: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(11)


def add_md_paragraph(doc: Document, segments: list[tuple[str, dict]]) -> None:
    """Add a paragraph from a list of (text, style_kwargs) segments.

    style_kwargs may include bold=True, italic=True.
    """
    p = doc.add_paragraph()
    for text, style in segments:
        run = p.add_run(text)
        run.font.size = Pt(11)
        run.bold = style.get("bold", False)
        run.italic = style.get("italic", False)


def add_bullets(doc: Document, items: list) -> None:
    """items: list of either str or list[(text, style)] for rich segments."""
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        if isinstance(item, str):
            run = p.add_run(item)
            run.font.size = Pt(11)
        else:
            for text, style in item:
                run = p.add_run(text)
                run.font.size = Pt(11)
                run.bold = style.get("bold", False)
                run.italic = style.get("italic", False)


def parse_cell(text: str) -> tuple[str, bool]:
    """Strip surrounding **bold** markers, return (clean_text, is_bold)."""
    t = text.strip()
    if t.startswith("**") and t.endswith("**") and len(t) >= 4:
        return t[2:-2], True
    return t, False


def add_table(
    doc: Document,
    header: list[str],
    rows: list[list[str]],
    *,
    numeric_cols: set[int] | None = None,
    method_col: int = 0,
) -> None:
    """Add a styled table with header shading, zebra rows, and numeric alignment.

    Cells in `rows` may contain **bold** Markdown markers; those become bold runs.
    Empty `method_col` cells inherit the previous non-empty value visually (left blank).
    """
    numeric_cols = numeric_cols or set()
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # Header row
    hdr_cells = table.rows[0].cells
    for j, label in enumerate(header):
        hdr_cells[j].text = ""
        run = hdr_cells[j].paragraphs[0].add_run(label)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(10.5)
        hdr_cells[j].paragraphs[0].alignment = (
            WD_ALIGN_PARAGRAPH.RIGHT if j in numeric_cols else WD_ALIGN_PARAGRAPH.LEFT
        )
        hdr_cells[j].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shade_cell(hdr_cells[j], HEADER_FILL)
        set_cell_borders(hdr_cells[j])

    # Body rows
    for i, row in enumerate(rows):
        cells = table.rows[i + 1].cells
        zebra = (i % 2 == 1)
        for j, raw in enumerate(row):
            text, is_bold = parse_cell(raw)
            cells[j].text = ""
            para = cells[j].paragraphs[0]
            run = para.add_run(text)
            run.font.size = Pt(10.5)
            run.bold = is_bold
            para.alignment = (
                WD_ALIGN_PARAGRAPH.RIGHT if j in numeric_cols else WD_ALIGN_PARAGRAPH.LEFT
            )
            cells[j].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if zebra:
                shade_cell(cells[j], ZEBRA_FILL)
            set_cell_borders(cells[j])
    doc.add_paragraph()  # spacing after the table


def main() -> None:
    out_path = Path("${GPA_REPO_ROOT}/results/GPA_vs_ISM_LEDIDI_report.docx")

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("GPA vs ISM and LEDIDI — K562 Enhancer Design")
    run.bold = True
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub.add_run("Anirban Sarkar — 2026-04-30")
    sub_run.italic = True
    sub_run.font.size = Pt(11)

    doc.add_paragraph()

    # Summary
    add_heading(doc, "Summary", level=1)
    add_para(
        doc,
        "We compare three methods for K562 enhancer optimization on two seed sequences:",
    )
    add_bullets(
        doc,
        [
            [
                ("GPA", {"bold": True}),
                (" — population-scale guided diffusion, 5,000-sequence pools (gpa_output_pool.h5).", {}),
            ],
            [
                ("ISM", {"bold": True}),
                (" — greedy in-silico mutagenesis, single-sequence trajectory (115 generations).", {}),
            ],
            [
                ("LEDIDI", {"bold": True}),
                (" — gradient editor with Gumbel-softmax, 50 reruns per (target, λ) configuration.", {}),
            ],
        ],
    )
    add_para(doc, "Each method is scored under two oracles:")
    add_bullets(
        doc,
        [
            [("LN", {"bold": True}), (" — LegNet K562 (the oracle every method optimizes against).", {})],
            [("AG", {"bold": True}), (" — AlphaGenome K562 (held-out, JAX MPRA-optimal/stage1).", {})],
        ],
    )

    add_para(doc, "Key findings:", bold=True)
    add_bullets(
        doc,
        [
            [
                ("LEDIDI reward-hacks LN.", {"bold": True}),
                (" It reaches the highest LN scores (≈ 9–12) but ", {}),
                ("collapses on AG (−0.6 to −0.2)", {"bold": True}),
                (" — its solutions do not generalize to the held-out oracle.", {}),
            ],
            [
                ("ISM climbs both oracles", {"bold": True}),
                (" but produces only a single sequence per run (LN ≈ 8.7–9.6, AG ≈ 2.8–3.0).", {}),
            ],
            [
                ("GPA is the only method that delivers a pool of 5,000 sequences whose mean AG (up to 3.64) beats both baselines.", {"bold": True}),
                (" Pool means, not cherry-picks.", {}),
            ],
            [
                ("GPA is 60–90× cheaper per delivered sequence than ISM", {"bold": True}),
                (" and ", {}),
                ("10–15× cheaper than LEDIDI.", {"bold": True}),
            ],
        ],
    )
    add_md_paragraph(
        doc,
        [
            ("Two seeds are reported throughout: ", {}),
            ("uid=181692 (natural)", {"bold": True}),
            (" and ", {}),
            ("uid=50178 (random)", {"bold": True}),
            (".", {}),
        ],
    )

    # ---- Table 1 ----
    add_heading(doc, "Table 1 — Pool statistics (mean / max under each oracle)", level=1)

    header1 = ["Method", "Recipe / Gen", "Edits", "n", "LN mean", "LN max", "AG mean", "AG max"]
    numeric1 = {2, 3, 4, 5, 6, 7}

    add_heading(doc, "1a. Seed uid=181692 (natural)", level=2)
    rows_1a = [
        ["**GPA noDPS**", "cap10", "20", "5000", "4.43", "4.52", "2.56", "2.78"],
        ["",              "cap20", "40", "5000", "5.78", "5.87", "3.11", "3.34"],
        ["",              "cap30", "60", "5000", "6.42", "6.72", "2.58", "2.97"],
        ["",              "cap50", "99", "5000", "7.75", "7.93", "**3.64**", "**3.81**"],
        ["",              "uncap", "99", "5000", "7.71", "7.88", "3.53", "3.86"],
        ["**GPA +DPS**",  "cap10", "20", "5000", "4.46", "4.47", "2.55", "2.69"],
        ["",              "cap20", "40", "5000", "7.27", "7.33", "3.33", "3.47"],
        ["",              "cap30", "60", "5000", "7.50", "7.54", "**3.61**", "3.69"],
        ["",              "cap50", "100","5000", "**9.04**", "**9.48**", "3.44", "3.69"],
        ["",              "uncap", "97", "5000", "8.75", "9.26", "3.10", "3.58"],
        ["**ISM**",       "gen 57", "57", "1",   "6.01", "—", "3.03", "—"],
        ["",              "gen 115","115","1",   "8.74", "—", "2.83", "—"],
        ["**LEDIDI**",    "t10 λ=0.40 (best)", "66",  "1/50", "9.11", "—", "−0.59", "—"],
        ["",              "t15 λ=0.05 (best)", "125", "1/50", "9.81", "—", "−0.22", "—"],
    ]
    add_table(doc, header1, rows_1a, numeric_cols=numeric1)

    add_heading(doc, "1b. Seed uid=50178 (random)", level=2)
    rows_1b = [
        ["**GPA noDPS**", "cap10", "20", "5000", "3.46", "3.55", "1.87", "2.09"],
        ["",              "cap20", "40", "5000", "5.26", "5.41", "2.24", "2.48"],
        ["",              "cap30", "60", "5000", "6.93", "7.07", "2.85", "2.97"],
        ["",              "cap50", "89", "5000", "7.87", "8.10", "2.62", "2.91"],
        ["",              "uncap", "87", "5000", "7.15", "7.43", "**2.85**", "**3.08**"],
        ["**GPA +DPS**",  "cap10", "20", "5000", "3.81", "3.88", "1.95", "2.55"],
        ["",              "cap20", "40", "5000", "6.33", "6.37", "2.82", "3.31"],
        ["",              "cap30", "60", "5000", "7.78", "7.86", "**3.51**", "**3.64**"],
        ["",              "cap50", "100","5000", "8.99", "9.17", "3.40", "3.59"],
        ["",              "uncap", "115","5000", "**10.30**", "**10.67**", "2.88", "3.16"],
        ["**ISM**",       "gen 57", "57", "1",   "6.53", "—", "2.41", "—"],
        ["",              "gen 115","115","1",   "9.58", "—", "2.92", "—"],
        ["**LEDIDI**",    "t10 λ=0.40 (best)", "69", "1/50", "9.23",  "—", "−0.19", "—"],
        ["",              "t15 λ=0.05 (best)", "105","1/50", "11.66", "—", "−0.62", "—"],
    ]
    add_table(doc, header1, rows_1b, numeric_cols=numeric1)

    add_md_paragraph(
        doc,
        [
            ("Observations.", {"bold": True}),
            (
                " LEDIDI achieves the highest LN scores by aggressive editing, but those edits drive the held-out AG score "
                "negative — a textbook reward-hacking signature. ISM climbs both oracles but only ever returns one sequence. "
                "GPA pools meet or exceed ISM's LN at comparable edit counts, and dominate on AG with mean scores up to ",
                {},
            ),
            ("3.64", {"bold": True}),
            (" (noDPS, cap50, uid=181692) and ", {}),
            ("3.51", {"bold": True}),
            (" (+DPS, cap30, uid=50178).", {}),
        ],
    )

    # ---- Table 2 ----
    add_heading(doc, "Table 2 — Single-sequence comparison", level=1)
    add_md_paragraph(
        doc,
        [
            (
                "To compare GPA fairly against the inherently single-sequence ISM and LEDIDI, we select from each GPA pool the ",
                {},
            ),
            ("one sequence that maximizes AG − LN", {"bold": True}),
            (" — i.e., the best held-out activity at the lowest training-oracle inflation.", {}),
        ],
    )

    header2 = ["Method", "Recipe / Gen", "Edits", "LN", "AG"]
    numeric2 = {2, 3, 4}

    add_heading(doc, "2a. Seed uid=181692 (natural)", level=2)
    rows_2a = [
        ["**GPA noDPS**", "cap10", "20",  "1.85", "1.06"],
        ["",              "cap20", "39",  "3.70", "2.56"],
        ["",              "cap30", "58",  "3.68", "2.42"],
        ["",              "cap50", "100", "4.35", "2.86"],
        ["",              "uncap", "96",  "4.97", "**3.33**"],
        ["**GPA +DPS**",  "cap10", "20",  "4.20", "2.50"],
        ["",              "cap20", "39",  "6.46", "3.36"],
        ["",              "cap30", "60",  "6.21", "**3.47**"],
        ["",              "cap50", "100", "6.98", "3.66"],
        ["",              "uncap", "99",  "6.73", "3.20"],
        ["**ISM**",       "gen 57", "57", "6.01", "3.03"],
        ["",              "gen 115","115","8.74", "2.83"],
        ["**LEDIDI**",    "t10 λ=0.40", "66",  "9.11", "−0.59"],
        ["",              "t15 λ=0.05", "125", "9.81", "−0.22"],
    ]
    add_table(doc, header2, rows_2a, numeric_cols=numeric2)

    add_heading(doc, "2b. Seed uid=50178 (random)", level=2)
    rows_2b = [
        ["**GPA noDPS**", "cap10", "20",  "2.05", "1.56"],
        ["",              "cap20", "40",  "2.91", "1.86"],
        ["",              "cap30", "60",  "4.71", "**2.75**"],
        ["",              "cap50", "95",  "5.27", "2.45"],
        ["",              "uncap", "89",  "4.64", "2.27"],
        ["**GPA +DPS**",  "cap10", "20",  "3.74", "2.55"],
        ["",              "cap20", "40",  "4.93", "2.70"],
        ["",              "cap30", "59",  "6.58", "**3.61**"],
        ["",              "cap50", "98",  "5.89", "2.70"],
        ["",              "uncap", "115", "7.90", "2.67"],
        ["**ISM**",       "gen 57",  "57", "6.53", "2.41"],
        ["",              "gen 115", "115","9.58", "2.92"],
        ["**LEDIDI**",    "t10 λ=0.40", "69",  "9.23",  "−0.19"],
        ["",              "t15 λ=0.05", "105", "11.66", "−0.62"],
    ]
    add_table(doc, header2, rows_2b, numeric_cols=numeric2)

    add_md_paragraph(
        doc,
        [
            ("Observations.", {"bold": True}),
            (
                " A single GPA sequence (e.g. +DPS cap30 on uid=50178: LN 6.58, AG 3.61) beats ISM gen 115 on AG by ",
                {},
            ),
            ("+0.69", {"bold": True}),
            (" at ", {}),
            ("LN 3.0 lower", {"bold": True}),
            (
                " — i.e., better held-out activity and less reward-hacking. LEDIDI under any selection rule remains AG-negative; "
                "not a single one of its 50 reruns produces an AG-positive sequence on either seed.",
                {},
            ),
        ],
    )

    # ---- Table 3 ----
    add_heading(doc, "Table 3 — Runtime averaged per method", level=1)
    add_para(
        doc,
        "Single GPU, end-to-end wall time. Each cell is the mean across both seeds (and across recipes within a method, where applicable).",
    )
    header3 = ["Method", "Output", "Avg wall"]
    numeric3 = {2}
    rows_3 = [
        ["**GPA noDPS**", "5,000-sequence pool",   "18 min"],
        ["**GPA +DPS**",  "5,000-sequence pool",   "25.5 min"],
        ["**ISM**",       "1 sequence (115 gens)", "19.5 s"],
        ["**LEDIDI**",    "1 best-of-50 sequence", "2.6 min"],
    ]
    add_table(doc, header3, rows_3, numeric_cols=numeric3)

    add_md_paragraph(
        doc,
        [
            ("Aggregation.", {"italic": True, "bold": True}),
            (
                " GPA wall = median across {cap10, cap20, cap30, cap50, uncap}, averaged over both seeds. "
                "LEDIDI wall = mean over {t10 λ=0.40, t15 λ=0.05} × both seeds. "
                "ISM wall = mean over both seeds for a full 115-generation trajectory.",
                {"italic": True},
            ),
        ],
    )

    # ---- Table 4 ----
    add_heading(doc, "Table 4 — Per-sequence cost (apples-to-apples)", level=1)
    add_para(
        doc,
        "Wall time per delivered sequence. GPA produces 5,000 per run; ISM produces 1; LEDIDI produces 1 best-of-50.",
    )
    header4 = ["Method", "Per-sequence cost", "Note"]
    numeric4 = {1}
    rows_4 = [
        ["**GPA noDPS**", "**0.22 s/seq**", "5,000-sequence pool"],
        ["**GPA +DPS**",  "**0.31 s/seq**", "5,000-sequence pool"],
        ["**ISM**",       "19.5 s/seq",     "greedy, no parallelism"],
        ["**LEDIDI**",    "3.1 s/seq",      "per Gumbel-batch optimum"],
    ]
    add_table(doc, header4, rows_4, numeric_cols=numeric4)

    add_md_paragraph(
        doc,
        [
            ("Bottom line.", {"bold": True}),
            (" GPA is ", {}),
            ("~60–90× cheaper per delivered sequence than ISM", {"bold": True}),
            (" and ", {}),
            ("~10–15× cheaper than LEDIDI", {"bold": True}),
            (
                ", and it is the only method whose population holds up on the held-out AG oracle.",
                {},
            ),
        ],
    )

    # Conclusions
    add_heading(doc, "Conclusions", level=1)
    add_bullets(
        doc,
        [
            [
                ("LEDIDI is not a viable enhancer-design baseline on this task.", {"bold": True}),
                (
                    " Despite hitting the highest LN scores, it reward-hacks the training oracle: every \"best run\" lands at AG < 0.",
                    {},
                ),
            ],
            [
                ("ISM climbs both oracles but is single-sequence and slow per delivered design.", {"bold": True}),
                (" It cannot produce the sequence diversity needed for downstream screening.", {}),
            ],
            [
                ("GPA is the only method that combines (a) high held-out (AG) activity, (b) population-scale output (5,000 sequences), and (c) per-sequence cost an order of magnitude below either baseline.", {"bold": True}),
                (
                    " Both the noDPS and +DPS variants outperform the baselines; +DPS gives the strongest LN with comparable AG, "
                    "noDPS gives the highest AG with lower edit counts.",
                    {},
                ),
            ],
        ],
    )

    # Source files
    add_heading(doc, "Source files", level=1)
    sources = [
        "GPA pools — results/k562_mdlm_gpa/run_v1{3,4}*_ism{50178,181692}/gpa_output_pool.h5",
        "GPA wall times — gpa_history.json → wall_time[-1] in each run directory",
        "ISM trajectories — results/ism_baseline/ism_trajectory_uid{uid}_ag.csv (k562_ag_jax column)",
        "ISM wall times — results/ism_baseline/ism_summary_uid{uid}.json → total_wall_s",
        "LEDIDI — results/ledidi_comparison/matched/uid{uid}/ledidi_uid{uid}_t{t}_l{l}{,_ag}.csv (LN from edited_legnet, AG from k562_ag_jax, wall from elapsed_s)",
        "Single-seed reference — results/k562_mdlm_gpa/single_seed_ism_{uid}.h5",
        "Source decks — results/GPA_vs_ISM_LEDIDI_combined_uid{181692,50178}.pptx",
        "Plotting script — scripts/k562_mdlm_gpa/create_ism_ledidi_dual_plot_slides.py",
    ]
    add_bullets(doc, sources)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
