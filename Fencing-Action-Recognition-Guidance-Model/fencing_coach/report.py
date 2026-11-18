"""English PDF reports made exclusively from structured analysis evidence."""

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
            spaceBefore=13,
            spaceAfter=8,
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
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
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
        title="Fencing Movement Evidence Report",
        author="Fencing Evidence Coach",
    )
    story = [
        p("FENCING / MOVEMENT REVIEW", "SectionLabel"),
        p("Evidence-led\ncoaching report", "ReportTitle"),
    ]
    if result.simulated:
        story.append(p("SIMULATED DATA - SOFTWARE DEMONSTRATION ONLY", "SectionLabel"))
    source_display = (
        result.source_name
        if result.source_name.isascii()
        else "Uploaded video (original filename retained in analysis.json)"
    )
    story += [
        p(f"Source: {source_display}"),
        p(
            f"Duration: {result.duration:.2f} s | Visible sampled frames: {result.observed_fraction:.0%} | View: {result.calibration.view}"
        ),
        p(
            f"Weapon: {result.config['weapon']} | Weapon hand: {result.config['weapon_side']} | Front leg: {result.config['front_side']}"
        ),
        Spacer(1, 9 * mm),
    ]
    story.append(p("How to read this report", "SectionLabel"))
    story.append(
        p(
            "Every finding below is linked to an estimated measurement, a time interval, and a published rule configuration. No generative model writes the coaching text. Missing evidence is shown as unassessed; it is never replaced with a predicted fault or an ideal value."
        )
    )
    story.append(
        p(
            "The 100-point reference rubric allocates 45 points to joint configuration, 25 to timing and coordination, 20 to movement control, and 10 to fluency and consistency. The last category uses measurable consistency proxies; it does not judge appearance or body type."
        )
    )
    story.append(
        p(
            "The rubric is a project-defined beginner solo-drill aid. Its numerical ranges and weights are not official federation standards and have not been calibrated against independent coach scores. An overall score is issued only when every weighted item and the action finish are observed."
        )
    )
    story.append(p("Session observations", "SectionLabel"))
    counts = Counter(a.action.kind for a in result.actions)
    rows = [["Movement", "Segments", "Fully supported /100 scores"]]
    for kind, count in counts.items():
        totals = [a.total for a in result.actions if a.action.kind == kind and a.total is not None]
        score = ", ".join(f"{v:.1f}" for v in totals[:8]) if totals else "Withheld"
        rows.append([ACTION_NAMES[kind], str(count), score])
    if len(rows) == 1:
        rows.append(["No supported movement identified", "0", "Withheld"])
    story.append(table(rows, [190, 60, 243]))
    story.append(p("Capture and evidence limits", "SectionLabel"))
    for warning in result.warnings:
        story.append(p("- " + warning, "Fine"))
    story.append(
        p(
            "Pose coordinates are monocular estimates. This report does not determine hits, blade contact, tactical correctness, priority, medical conditions, or injury risk. It provides observations for review with a fencing coach.",
            "Fine",
        )
    )

    for i, assessment in enumerate(result.actions, 1):
        action = assessment.action
        story.append(PageBreak())
        story += [
            p(f"MOVEMENT {i:02d} / {ACTION_NAMES[action.kind].upper()}", "SectionLabel"),
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
        dimension_rows = [["Dimension", "Points / available weight"]]
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
                    "Overlay: image-space landmarks. Angles below use estimated 3D world coordinates, not this 2D projection.",
                    "Fine",
                )
            )
        else:
            story.append(table(dimension_rows, [340, 153]))
        story.append(p("Measurement evidence", "SectionLabel"))
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
        findings = [m for m in assessment.metrics if m.score is not None and m.score < 99.5]
        story.append(p("Evidence-linked coaching review", "SectionLabel"))
        if not findings:
            story.append(
                p(
                    "No deviation from the configured bands was observed among assessed items. This does not certify overall technique or unmeasured aspects of the movement."
                )
            )
        for m in sorted(findings, key=lambda item: item.score * item.weight / 100 - item.weight):
            text = f"During {action.start:.2f}-{action.end:.2f} s (evidence marker {m.observed_at:.2f} s), {m.label.lower()} was estimated at {measured(m.value, m.unit)} {m.unit}; the configured band is {m.target[0]:g}-{m.target[1]:g} {m.unit}. {m.cue} Reference: {m.source}; numerical tolerance: PROJECT."
            story.append(KeepTogether([p(m.label, "SectionLabel"), p(text)]))
        missing = [m for m in assessment.metrics if m.score is None]
        if missing:
            story.append(p("Items deliberately left unassessed", "SectionLabel"))
            grouped_reasons = {}
            for m in missing:
                grouped_reasons.setdefault(m.reason, []).append(m.label)
            for reason, labels in grouped_reasons.items():
                story.append(p(f"{'; '.join(labels)}: {reason}", "Fine"))

    story.append(PageBreak())
    story.append(p("METHOD / REFERENCES", "SectionLabel"))
    story.append(p("Audit trail", "ReportTitle"))
    story.append(
        p(
            f"Software {result.software_version} | Result schema {result.schema_version} | Rubric {result.rubric_version}"
        )
    )
    story.append(
        p(
            "Action labels are produced by temporal geometry rules, not a newly trained fencing classifier. Joint angles use shoulder-elbow-wrist or hip-knee-ankle triplets in MediaPipe world coordinates. The initial guard defines the attack axis and body-scale normalization. Timing values require separate sustained onsets; unresolved ordering is not graded."
        )
    )
    story.append(
        p(
            "Missing observations break action sequences. A frontal view can retain visible 3D posture observations but does not establish forward/backward travel from hip-relative coordinates. Projection-dependent items are withheld for frontal or moving-camera capture."
        )
    )
    story.append(
        p(
            "Scoring: each metric receives 100 within its reference band, decreases linearly to zero at the configured outer bounds, and contributes metric_score x metric_weight / 100. Unknown metrics retain their original weight and are never silently renormalized. Scores should only be compared for the same drill, rubric, and capture setup."
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
        story.append(p("Input and model provenance", "SectionLabel"))
        for key, value in result.provenance.items():
            story.append(p(f"{key}: {value}", "Fine"))

    def page_footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(TEAL)
        canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
        canvas.setFont(regular, 8)
        canvas.setFillColor(INK)
        canvas.drawString(18 * mm, 10 * mm, "FENCING EVIDENCE COACH | Prototype reference rubric")
        canvas.drawRightString(192 * mm, 10 * mm, str(document.page))
        if result.simulated:
            canvas.setFillColor(AMBER)
            canvas.setFont(bold, 8)
            canvas.drawRightString(192 * mm, 287 * mm, "SIMULATED DATA - NOT AN ATHLETE ASSESSMENT")
        canvas.restoreState()

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return path
