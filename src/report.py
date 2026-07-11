"""
report.py
=========
Generates the downloadable PDF report required by the brief: a summary of
model predictions, confidence/XAI notes, and the human reviewer's final
approve/reject/modify decisions, plus the business snapshot.

Uses reportlab only (no external binaries needed, works fully offline).
An optional `narrative` string (produced by an LLM call in app.py, purely
for wording — never for the underlying numbers) can be inserted as an
executive summary paragraph; if none is supplied, a template summary is
used instead so the report always generates even without API access.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
)


def build_pdf_report(output_path: str, df_reviewed, kpis: dict, narrative: str = None,
                       chart_image_path: str = None):
    """
    df_reviewed: DataFrame with at least
        ['SKU', 'Supplier name', 'Risk Score', 'Risk Label',
         'human_decision', 'human_note']
    kpis: dict of headline numbers to show at the top
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                              topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=colors.HexColor("#1a237e"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#283593"))
    body = styles["BodyText"]

    story = []
    story.append(Paragraph("Supply Chain Risk Co-Pilot — Report", title_style))
    story.append(Paragraph(datetime.now().strftime("Generated %Y-%m-%d %H:%M"), body))
    story.append(Spacer(1, 0.5 * cm))

    # --- Executive summary --------------------------------------------------
    story.append(Paragraph("Executive Summary", h2))
    if not narrative:
        narrative = (
            f"Out of {kpis.get('total', 0)} shipments analyzed, "
            f"{kpis.get('high', 0)} were flagged High risk, "
            f"{kpis.get('medium', 0)} Medium risk, and {kpis.get('low', 0)} Low risk. "
            f"The human reviewer approved {kpis.get('approved', 0)} AI recommendations, "
            f"modified {kpis.get('modified', 0)}, and rejected {kpis.get('rejected', 0)}."
        )
    story.append(Paragraph(narrative, body))
    story.append(Spacer(1, 0.4 * cm))

    # --- KPI table -----------------------------------------------------------
    kpi_rows = [["Metric", "Value"]] + [[k.replace("_", " ").title(), str(v)] for k, v in kpis.items()]
    kpi_table = Table(kpi_rows, colWidths=[8 * cm, 6 * cm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3949ab")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.5 * cm))

    if chart_image_path and os.path.exists(chart_image_path):
        story.append(Paragraph("Risk Distribution", h2))
        story.append(RLImage(chart_image_path, width=14 * cm, height=7 * cm))
        story.append(Spacer(1, 0.4 * cm))

    # --- Human-in-the-loop decision log --------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Human-in-the-Loop Decision Log", h2))
    cols = ["SKU", "Supplier", "AI Risk", "Score", "Decision", "Reviewer Note"]
    table_data = [cols]
    for _, r in df_reviewed.iterrows():
        table_data.append([
            str(r.get("SKU", "")),
            str(r.get("Supplier name", ""))[:14],
            str(r.get("Risk Label", "")),
            f"{r.get('Risk Score', 0):.0f}",
            str(r.get("human_decision", "Pending")),
            str(r.get("human_note", ""))[:28],
        ])
    t = Table(table_data, colWidths=[2 * cm, 3 * cm, 2 * cm, 1.7 * cm, 2.3 * cm, 6 * cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3949ab")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)

    doc.build(story)
    return output_path


if __name__ == "__main__":
    import pandas as pd
    demo = pd.DataFrame([
        {"SKU": "SKU0", "Supplier name": "Supplier 3", "Risk Label": "Medium",
         "Risk Score": 55, "human_decision": "Approved", "human_note": "Matches QA log"},
        {"SKU": "SKU4", "Supplier name": "Supplier 1", "Risk Label": "High",
         "Risk Score": 82, "human_decision": "Modified -> Medium", "human_note": "Backup supplier lined up"},
    ])
    kpis = {"total": 100, "high": 17, "medium": 73, "low": 10,
            "approved": 70, "modified": 20, "rejected": 10}
    out = build_pdf_report("/home/claude/supply_chain_copilot/outputs/demo_report.pdf", demo, kpis)
    print("wrote", out)
