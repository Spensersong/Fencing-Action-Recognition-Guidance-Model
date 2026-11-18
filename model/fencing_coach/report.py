"""Build a coaching PDF from the recorded measurements."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

import reportlab
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .schema import ACTION_NAMES, DIMENSIONS
from .scoring import load_rubric

INK = colors.HexColor("#172B40")
TEAL = colors.HexColor("#007C83")
PALE = colors.HexColor("#EDF5F5")
AMBER = colors.HexColor("#935500")
EDGES = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
]


def skeleton_drawing(frame, width=330, height=190):
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=INK, strokeColor=None))
    import numpy as np

    if (
        frame is None
        or frame.image is None
        or frame.image.shape != (33, 5)
        or not np.isfinite(frame.image).all()
    ):
        drawing.add(
            String(
                15, height / 2, "No visible keyframe available", fillColor=colors.white, fontSize=10
            )
        )
        return drawing
    selected = [
        i
        for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        if frame.image[i, 3] >= 0.65
    ]
    if not selected:
        return drawing
    projected = frame.image[:, :2] * [frame.width, frame.height]
    low, high = np.min(projected[selected], axis=0), np.max(projected[selected], axis=0)
    span = np.maximum(high - low, 1)
    scale = min((width - 32) / span[0], (height - 40) / span[1])
    ox = (width - span[0] * scale) / 2 - low[0] * scale
    oy = (height - span[1] * scale) / 2 - low[1] * scale + 5

    def point(i):
        return ox + projected[i, 0] * scale, height - (oy + projected[i, 1] * scale)

    for a, b in EDGES:
        if min(frame.image[a, 3], frame.image[b, 3]) >= 0.65:
            x1, y1 = point(a)
            x2, y2 = point(b)
            drawing.add(
                Line(x1, y1, x2, y2, strokeColor=colors.HexColor("#60D5CB"), strokeWidth=1.5)
            )
    for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
        if frame.image[i, 3] >= 0.65:
            x, y = point(i)
            drawing.add(Circle(x, y, 2, fillColor=colors.white, strokeColor=None))
    drawing.add(
        String(
            9,
            8,
            f"Observed landmark projection | {frame.t:.2f} s",
            fillColor=colors.white,
            fontSize=8,
        )
    )
    return drawing


def write_pdf(result, path, frames=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    font_root = Path(reportlab.__file__).resolve().parent / "fonts"
    regular, bold = "Helvetica", "Helvetica-Bold"
    if (font_root / "Vera.ttf").is_file() and (font_root / "VeraBd.ttf").is_file():
        regular, bold = "EvidenceSans", "EvidenceSansBold"
        if regular not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(regular, str(font_root / "Vera.ttf")))
            pdfmetrics.registerFont(TTFont(bold, str(font_root / "VeraBd.ttf")))
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "ReportTitle",
            fontName=bold,
            fontSize=27,
            leading=31,
            textColor=INK,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionLabel",
            fontName=bold,
            fontSize=11,
            leading=14,
            textColor=TEAL,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "Fine", fontName=regular, fontSize=8, leading=11, textColor=INK, spaceAfter=5
        )
    )
    styles.add(
        ParagraphStyle(
            "ActionTitle",
            fontName=bold,
            fontSize=22,
            leading=26,
            textColor=INK,
            spaceAfter=8,
        )
    )
    styles.add(ParagraphStyle("Cell", fontName=regular, fontSize=8, leading=10, textColor=INK))
    styles.add(
        ParagraphStyle(
            "Score",
            fontName=bold,
            fontSize=15,
            leading=19,
            textColor=TEAL,
            alignment=TA_CENTER,
        )
    )
    styles["BodyText"].fontSize = 9
    styles["BodyText"].fontName = regular
    styles["BodyText"].leading = 13
    styles["BodyText"].textColor = INK

    def p(text, style="BodyText"):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])

    def measured(value, unit):
        return f"{value:.1f}" if unit.startswith("deg") else f"{value:.3f}"

    def table(rows, widths, header=True):
        cells = [[p(cell, "Cell") for cell in row] for row in rows]
        item = Table(cells, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
        rules = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, 0), 1, TEAL),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ]
        if header:
            rules.append(("BACKGROUND", (0, 0), (-1, 0), PALE))
        item.setStyle(TableStyle(rules))
        return item

    doc = SimpleDocTemplate(
        str(path),
        pagesize=(210 * mm, 297 * mm),
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=19 * mm,
        title="Fencing Practice Report",
        author="Fencing Practice Review",
    )
    story = [
        p("Fencing practice report", "ReportTitle"),
    ]
    if result.simulated:
        story.append(p("SIMULATED DATA / SAMPLE DRILL", "SectionLabel"))
    source_display = (
        result.source_name
        if result.source_name.isascii()
        else "Uploaded video (original filename retained in analysis.json)"
    )
    story += [
        p(f"Source: {source_display}"),
        p(
            f"Duration: {result.duration:.2f} s | Usable sampled frames: {result.observed_fraction:.0%} | View: {result.calibration.view}"
        ),
        p(
            f"Weapon: {result.config['weapon']} | Weapon hand: {result.config['weapon_side']} | Front leg: {result.config['front_side']}"
        ),
        Spacer(1, 9 * mm),
    ]
    story.append(p("How to read this report", "SectionLabel"))
    story.append(
        p(
            "Use the movement times to find the relevant part of your recording. Each review point shows a measurement, its reference range, and a suggestion for what to check in the clip. Items without a usable measurement are marked unassessed."
        )
    )
    story.append(
        p(
            "The report assigns 45 points to joint configuration, 25 to timing and coordination, 20 to movement control, and 10 to fluency and consistency. Fluency uses changes in posture or reach over the movement."
        )
    )
    story.append(
        p(
            "The ranges are intended for beginner solo drills and still need to be checked against independent coach scores. A total requires a usable measurement for every weighted item and a visible finish."
        )
    )
    story.append(p("Session summary", "SectionLabel"))
    counts = Counter(a.action.kind for a in result.actions)
    rows = [["Movement", "Segments", "Overall scores / 100"]]
    for kind, count in counts.items():
        totals = [a.total for a in result.actions if a.action.kind == kind and a.total is not None]
        score = ", ".join(f"{v:.1f}" for v in totals[:8]) if totals else "Withheld"
        rows.append([ACTION_NAMES[kind], str(count), score])
    if len(rows) == 1:
        rows.append(["No supported movement identified", "0", "Withheld"])
    story.append(table(rows, [190, 60, 243]))
    story.append(p("Capture notes", "SectionLabel"))
    for warning in result.warnings:
        story.append(p("- " + warning, "Fine"))
    story.append(
        p(
            "Joint positions are estimated from a single camera. Review the measurements alongside your clip or with a fencing coach.",
            "Fine",
        )
    )

    for assessment in result.actions:
        action = assessment.action
        story.append(PageBreak())
        story += [
            p(ACTION_NAMES[action.kind].upper(), "SectionLabel"),
            p(f"{action.start:.2f} - {action.end:.2f} seconds", "ActionTitle"),
        ]
        score_text = (
            f"{assessment.total:.1f} / 100"
            if assessment.total is not None
            else "Overall score withheld"
        )
        story.append(p(score_text, "Score"))
        story.append(
            p(
                f"Weighted evidence coverage: {assessment.coverage:.0%}. Finish observed: {'yes' if action.complete else 'no'}.",
                "Fine",
            )
        )
        for note in assessment.notes:
            story.append(p(note, "Fine"))
        if action.kind == "unknown":
            story.append(
                p(
                    "No movement-specific correction or score is issued for this interval. Record a separate supported drill with a clear starting guard, an observable finish, and a fixed camera."
                )
            )
            continue
        dimension_rows = [["Category", "Points"]]
        for key, (label, weight) in DIMENSIONS.items():
            value = assessment.dimensions.get(key)
            dimension_rows.append(
                [
                    label,
                    f"{value:.1f} / {weight}" if value is not None else f"Unassessed / {weight}",
                ]
            )
        nearest = min(frames, key=lambda f: abs(f.t - action.peak)) if frames else None
        if nearest is not None and abs(nearest.t - action.peak) <= 0.15:
            panel = Table(
                [
                    [
                        skeleton_drawing(nearest, width=225, height=150),
                        table(dimension_rows, [170, 90]),
                    ]
                ],
                colWidths=[233, 260],
                hAlign="LEFT",
            )
            panel.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ]
                )
            )
            story.append(panel)
            story.append(
                p(
                    "The diagram shows joint positions in the image. The angles below use estimated 3D positions.",
                    "Fine",
                )
            )
        else:
            story.append(table(dimension_rows, [340, 153]))
        story.append(p("Measurements", "SectionLabel"))
        metric_rows = [["Measured item", "Estimate", "Reference band", "Time", "Points"]]
        for m in assessment.metrics:
            value = f"{measured(m.value, m.unit)} {m.unit}" if m.value is not None else "Unassessed"
            metric_rows.append(
                [
                    m.label,
                    value,
                    f"{m.target[0]:g} - {m.target[1]:g} {m.unit}",
                    f"{m.observed_at:.2f} s" if m.observed_at is not None else "-",
                    f"{m.score * m.weight / 100:.1f}/{m.weight:g}" if m.score is not None else "-",
                ]
            )
        story.append(table(metric_rows, [174, 94, 99, 51, 75]))
        assessed = [m for m in assessment.metrics if m.score is not None]
        findings = [m for m in assessed if m.score < 99.5]
        if assessed:
            story.append(p("What to review", "SectionLabel"))
            if not findings:
                story.append(
                    p("The assessed measurements were within or close to the reference ranges.")
                )
        for m in sorted(findings, key=lambda item: item.score * item.weight / 100 - item.weight):
            text = (
                f"Review marker {m.observed_at:.2f} s. The estimate is {measured(m.value, m.unit)} {m.unit} "
                f"and the reference range is {m.target[0]:g}-{m.target[1]:g} {m.unit}. "
                f"{m.cue} [{m.source}]"
            )
            story.append(KeepTogether([p(m.label, "SectionLabel"), p(text)]))
        missing = [m for m in assessment.metrics if m.score is None]
        if missing:
            story.append(p("Unassessed measurements", "SectionLabel"))
            grouped_reasons = {}
            for m in missing:
                grouped_reasons.setdefault(m.reason, []).append(m.label)
            for reason, labels in grouped_reasons.items():
                story.append(p(f"{', '.join(labels)}. {reason}", "Fine"))

    story.append(PageBreak())
    story.append(p("Method and references", "ReportTitle"))
    story.append(
        p(
            f"Software {result.software_version} | Result schema {result.schema_version} | Rubric {result.rubric_version}"
        )
    )
    story.append(
        p(
            "The program recognises each drill from changes in joint positions over time. It estimates joint angles from MediaPipe's 3D coordinates and uses the starting guard to set the forward direction and body scale. Timing comparisons need separate, visible starts for each movement."
        )
    )
    story.append(
        p(
            "A front view can show some joint angles. Image-based movement measurements require a fixed camera and a clear view of forward and backward travel. Gaps in the recording can interrupt an action."
        )
    )
    story.append(
        p(
            "A measurement earns full points inside its reference range. Points decrease linearly outside the range until the measurement reaches the outer bound. Unassessed items keep their assigned weight. Compare scores for the same drill, rubric and capture setup."
        )
    )
    for key, ref in load_rubric()["sources"].items():
        story += [
            p(key + " / " + ref["title"], "SectionLabel"),
            p(ref["scope"]),
            p(ref["url"], "Fine"),
        ]
    story += [
        p("MediaPipe model and coordinate documentation", "SectionLabel"),
        p("https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python", "Fine"),
    ]
    if result.provenance:
        story.append(p("Recording and model details", "SectionLabel"))
        for key, value in result.provenance.items():
            story.append(p(f"{key}: {value}", "Fine"))

    def page_footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(TEAL)
        canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
        canvas.setFont(regular, 8)
        canvas.setFillColor(INK)
        canvas.drawString(18 * mm, 10 * mm, "FENCING PRACTICE REPORT")
        canvas.drawRightString(192 * mm, 10 * mm, str(document.page))
        if result.simulated:
            canvas.setFillColor(AMBER)
            canvas.setFont(bold, 8)
            canvas.drawRightString(192 * mm, 287 * mm, "SIMULATED DATA / SAMPLE DRILL")
        canvas.restoreState()

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return path
