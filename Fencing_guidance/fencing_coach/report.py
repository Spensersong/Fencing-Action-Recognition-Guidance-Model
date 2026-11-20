"""Put the clip times, practice notes and measurements in a printable PDF."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import reportlab
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Table,
    TableStyle,
)

from .schema import ACTION_NAMES, DIMENSIONS
from .scoring import load_rubric

RULE = colors.HexColor("#AAAAAA")
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


def _report_fonts():
    # Embed the fonts that come with ReportLab so text spacing stays consistent across viewers.
    directory = Path(reportlab.__file__).resolve().parent / "fonts"
    regular, bold = directory / "Vera.ttf", directory / "VeraBd.ttf"
    if regular.is_file() and bold.is_file():
        pdfmetrics.registerFont(TTFont("PracticeSans", str(regular)))
        pdfmetrics.registerFont(TTFont("PracticeSansBold", str(bold)))
        return "PracticeSans", "PracticeSansBold"
    return "Helvetica", "Helvetica-Bold"


REGULAR, BOLD = _report_fonts()
BODY = ParagraphStyle(
    "Body", fontName=REGULAR, fontSize=10, leading=14, textColor=colors.black, spaceAfter=6
)
TITLE = ParagraphStyle("Title", parent=BODY, fontName=BOLD, fontSize=24, leading=30, spaceAfter=14)
SECTION = ParagraphStyle(
    "Section",
    parent=BODY,
    fontName=BOLD,
    fontSize=12,
    leading=16,
    spaceBefore=13,
    spaceAfter=6,
    keepWithNext=True,
)
MOVEMENT = ParagraphStyle("Movement", parent=SECTION, fontSize=15, leading=20, spaceAfter=3)
SUBHEAD = ParagraphStyle(
    "Subhead", parent=SECTION, fontSize=10, leading=13, spaceBefore=7, spaceAfter=3
)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.8, leading=12, spaceAfter=4)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=9, leading=12, spaceAfter=0)
CELL_RIGHT = ParagraphStyle("CellRight", parent=CELL, alignment=TA_RIGHT)
TABLE_HEAD = ParagraphStyle("TableHead", parent=CELL, fontName=BOLD)
TABLE_HEAD_RIGHT = ParagraphStyle("TableHeadRight", parent=TABLE_HEAD, alignment=TA_RIGHT)


def _text(value, style=BODY):
    return Paragraph(escape(str(value)).replace("\n", "<br/>"), style)


def _table(rows, widths, right_columns=()):
    cells = []
    for row_number, row in enumerate(rows):
        left, right = (TABLE_HEAD, TABLE_HEAD_RIGHT) if row_number == 0 else (CELL, CELL_RIGHT)
        cells.append(
            [_text(value, right if col in right_columns else left) for col, value in enumerate(row)]
        )
    item = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    item.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
            ]
        )
    )
    return item


def skeleton_drawing(frame, width=330, height=190):
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=None, strokeColor=RULE, strokeWidth=0.5))
    selected = []
    if frame is not None and frame.image is not None:
        if frame.image.shape == (33, 5) and np.isfinite(frame.image).all():
            selected = [
                i
                for i in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
                if frame.image[i, 3] >= 0.65
            ]
    if not selected:
        drawing.add(
            String(
                width / 2,
                height / 2,
                "No visible landmarks at this frame",
                textAnchor="middle",
                fontName=REGULAR,
                fontSize=8,
                fillColor=colors.black,
            )
        )
        return drawing

    projected = frame.image[:, :2] * [frame.width, frame.height]
    low, high = np.min(projected[selected], axis=0), np.max(projected[selected], axis=0)
    span = np.maximum(high - low, 1)
    scale = min((width - 24) / span[0], (height - 29) / span[1])
    ox = (width - span[0] * scale) / 2 - low[0] * scale
    oy = (height - span[1] * scale) / 2 - low[1] * scale - 6

    def point(index):
        return ox + projected[index, 0] * scale, height - (oy + projected[index, 1] * scale)

    for a, b in EDGES:
        if a in selected and b in selected:
            x1, y1 = point(a)
            x2, y2 = point(b)
            drawing.add(Line(x1, y1, x2, y2, strokeColor=colors.black, strokeWidth=0.9))
    for index in selected:
        if 11 <= index <= 28:
            x, y = point(index)
            drawing.add(Circle(x, y, 1.6, fillColor=colors.black, strokeColor=None))
    drawing.add(
        String(
            7,
            6,
            f"Joint positions at {frame.t:.2f} s",
            fontName=REGULAR,
            fontSize=8,
            fillColor=colors.black,
        )
    )
    return drawing


def _seconds(action):
    return f"{action.start:.2f} - {action.end:.2f}"


def _estimate(metric):
    if metric.value is None:
        return "Unassessed"
    precision = 1 if metric.unit.startswith("deg") else 3
    return f"{metric.value:.{precision}f} {metric.unit}"


def _range(metric):
    return f"{metric.target[0]:g} - {metric.target[1]:g} {metric.unit}"


def _session_details(result, width):
    source = (
        result.source_name
        if result.source_name.isascii()
        else "Uploaded video (filename in analysis.json)"
    )
    config = result.config
    rows = [
        ["Recording", source, "", ""],
        ["View", result.calibration.view.capitalize(), "Duration", f"{result.duration:.2f} s"],
        [
            "Weapon",
            config["weapon"].capitalize(),
            "Weapon hand",
            config["weapon_side"].capitalize(),
        ],
        [
            "Front leg",
            config["front_side"].capitalize(),
            "Usable samples",
            f"{result.observed_fraction:.0%}",
        ],
    ]
    cells = [
        [_text(value, TABLE_HEAD if col in (0, 2) else CELL) for col, value in enumerate(row)]
        for row in rows
    ]
    item = Table(
        cells, colWidths=[width * part for part in (0.17, 0.33, 0.18, 0.32)], hAlign="LEFT"
    )
    item.setStyle(
        TableStyle(
            [
                ("SPAN", (1, 0), (3, 0)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return item


def _review_notes(actions, references):
    notes = []
    for assessment in actions:
        findings = [
            metric
            for metric in assessment.metrics
            if metric.score is not None and metric.score < 99.5
        ]
        findings.sort(
            key=lambda metric: (
                metric.observed_at if metric.observed_at is not None else float("inf")
            )
        )
        for metric in findings:
            name = ACTION_NAMES[assessment.action.kind]
            marker = (
                f", review at {metric.observed_at:.2f} s" if metric.observed_at is not None else ""
            )
            ref = f" [{references[metric.source]}]" if metric.source in references else ""
            notes.append(
                KeepTogether(
                    [
                        _text(f"{name}{marker}  |  {metric.label}", SUBHEAD),
                        _text(
                            f"{_estimate(metric)} (reference {_range(metric)}). {metric.cue}{ref}"
                        ),
                    ]
                )
            )
    return notes


def _movement_details(assessment, frames, width):
    action = assessment.action
    state = f"{assessment.total:.1f} / 100" if assessment.total is not None else "Score withheld"
    finish = "Complete interval" if action.complete else "Incomplete interval"
    parts = [
        _text(ACTION_NAMES[action.kind], MOVEMENT),
        _text(
            f"{_seconds(action)} s    |    {state}    |    Scored weight {assessment.coverage:.0%}    |    {finish}",
            SMALL,
        ),
    ]
    parts.extend(_text(note, SMALL) for note in assessment.notes)
    if not assessment.metrics:
        if action.kind == "unknown":
            parts.append(
                _text(
                    "Try another clip with a clear starting guard, a visible finish and a fixed camera.",
                    SMALL,
                )
            )
        return KeepTogether(parts)

    category_rows = [["Score by category", "Points"]]
    for key, (label, weight) in DIMENSIONS.items():
        points = assessment.dimensions.get(key)
        category_rows.append(
            [label, f"{points:.1f} / {weight}" if points is not None else "Unassessed"]
        )

    nearest = min(frames, key=lambda frame: abs(frame.t - action.peak)) if frames else None
    if nearest is not None and abs(nearest.t - action.peak) <= 0.15:
        gap = 16
        sketch_width = width * 0.37
        category_width = width - sketch_width - gap
        panel = Table(
            [
                [
                    _table(category_rows, [category_width * 0.72, category_width * 0.28], (1,)),
                    "",
                    skeleton_drawing(nearest, width=sketch_width, height=100),
                ]
            ],
            colWidths=[category_width, gap, sketch_width],
            hAlign="LEFT",
        )
        panel.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        parts.append(panel)
    else:
        parts.append(_table(category_rows, [width * 0.72, width * 0.28], (1,)))

    rows = [["Measurement", "Estimate", "Reference", "At (s)", "Points"]]
    for metric in assessment.metrics:
        rows.append(
            [
                metric.label,
                _estimate(metric),
                _range(metric),
                f"{metric.observed_at:.2f}" if metric.observed_at is not None else "-",
                f"{metric.score * metric.weight / 100:.1f} / {metric.weight:g}"
                if metric.score is not None
                else "-",
            ]
        )
    parts += [
        _text("Measured values", SUBHEAD),
        _table(rows, [width * part for part in (0.43, 0.16, 0.20, 0.08, 0.13)], (1, 2, 3, 4)),
    ]
    missing = {}
    for metric in assessment.metrics:
        if metric.score is None:
            missing.setdefault(metric.reason, []).append(metric.label)
    if missing:
        parts.append(_text("Unassessed measurements", SUBHEAD))
        for reason, labels in missing.items():
            parts.append(_text(f"{', '.join(labels)}. {reason}", SMALL))
    return KeepTogether(parts)


def write_pdf(result, path, frames=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=20 * mm,
        bottomMargin=19 * mm,
        title="Fencing Practice Report",
        author="Fencing Practice Review",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.45)
        left, right = document.leftMargin, document.pagesize[0] - document.rightMargin
        canvas.line(left, 14 * mm, right, 14 * mm)
        canvas.setFont(REGULAR, 8)
        canvas.setFillColor(colors.black)
        label = "SIMULATED DATA / SAMPLE DRILL" if result.simulated else "Fencing practice report"
        canvas.drawString(left, 10 * mm, label)
        canvas.drawRightString(right, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=footer))
    sources = load_rubric()["sources"]
    references = {key: number for number, key in enumerate(sources, 1)}
    story = [_text("Fencing practice report", TITLE), _session_details(result, doc.width)]

    rows = [["Movement", "Time in clip (s)", "Score / 100"]]
    for assessment in result.actions:
        action = assessment.action
        rows.append(
            [
                ACTION_NAMES[action.kind],
                _seconds(action),
                f"{assessment.total:.1f}" if assessment.total is not None else "Withheld",
            ]
        )
    if not result.actions:
        rows.append(["No supported movement identified", "-", "Withheld"])
    story += [
        _text("Movement summary", SECTION),
        _table(rows, [doc.width * part for part in (0.53, 0.25, 0.22)], (1, 2)),
    ]

    notes = _review_notes(result.actions, references)
    if notes:
        story += [_text("What to check in the clip", SECTION), *notes]
    elif any(metric.score is not None for item in result.actions for metric in item.metrics):
        story += [
            _text("Review notes", SECTION),
            _text(
                "The measurements available for scoring were within or close to the reference ranges."
            ),
        ]
    if result.warnings:
        story.append(_text("Recording notes", SECTION))
        story.extend(_text(warning, SMALL) for warning in result.warnings)

    if result.actions:
        story += [PageBreak(), _text("Movement measurements", TITLE)]
        story.extend(
            _movement_details(assessment, frames, doc.width) for assessment in result.actions
        )

    story += [
        PageBreak(),
        _text("Scoring notes and references", TITLE),
        _text(
            "How the measurements were taken",
            SECTION,
        ),
        _text(
            "The movement labels come from changes in joint positions through the clip. Joint angles use MediaPipe's estimated 3D coordinates. The starting guard sets the forward direction and body scale. The sketches show the joint positions in the video image."
        ),
        _text(
            "A front view can show some joint angles. Image-based travel measurements need a fixed camera and a view that shows movement forward and backward. Separate starts must be visible to compare timing."
        ),
        _text("Reading the score", SECTION),
        _text(
            "Joint configuration is worth 45 points, timing and coordination 25, movement control 20, and fluency and consistency 10. The fluency measures track changes in posture or reach over the movement."
        ),
        _text(
            "A measurement earns full points within its reference range. Points fall linearly outside the range to zero at the outer limit. Scored weight is the share of points for which a measurement is available. A total requires a complete movement and all weighted measurements."
        ),
        _text("For comparisons, use the same drill, rubric and recording setup.", SMALL),
        _text("References", SECTION),
    ]
    for key, source in sources.items():
        story.append(
            KeepTogether(
                [
                    _text(f"[{references[key]}] {source['title']} ({key})", SUBHEAD),
                    _text(source["scope"], SMALL),
                    _text(source["url"], SMALL),
                ]
            )
        )
    story += [
        _text("MediaPipe model and coordinate documentation", SUBHEAD),
        _text(
            "https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python", SMALL
        ),
        _text("Report settings", SECTION),
        _text(
            f"Software {result.software_version}    |    Schema {result.schema_version}    |    Rubric {result.rubric_version}",
            SMALL,
        ),
    ]
    for key, value in result.provenance.items():
        story.append(_text(f"{key.replace('_', ' ').capitalize()}: {value}", SMALL))

    doc.build(story)
    return path
