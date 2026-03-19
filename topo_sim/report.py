from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


SECTION_FILL = colors.HexColor("#12343b")
GRID_LINE = colors.HexColor("#93a6ac")
ALT_ROW = colors.HexColor("#f7f2ea")


def _fmt_value(value: Any, unit: str = "") -> str:
    if isinstance(value, float):
        rendered = f"{value:,.2f}" if abs(value) >= 1000 else f"{value:.4g}"
    else:
        rendered = str(value)
    return f"{rendered}{unit}" if unit else rendered


def _styled_table(rows: list[list[str]], col_widths: list[float] | None = None) -> Table:
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SECTION_FILL),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID_LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
            ]
        )
    )
    return table


def _add_key_value_section(
    story: list[Any],
    title: str,
    data: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> None:
    story.append(Paragraph(title, styles["Heading3"]))
    rows = [["Field", "Value"]]
    for key, value in data.items():
        label = key.replace("_", " ").title()
        rows.append([label, _fmt_value(value)])
    story.append(_styled_table(rows, [62 * mm, 110 * mm]))
    story.append(Spacer(1, 6))


def _metric_rows(title: str, metrics: dict[str, float]) -> list[list[str]]:
    rows = [[title, "Value"]]
    for key, value in metrics.items():
        label = key.replace("_", " ").title()
        rows.append([label, _fmt_value(value)])
    return rows


def build_pdf_report(results: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], leading=15, fontName="Helvetica")

    story: list[Any] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph("SSU-Centric Topology Analysis Report", styles["Title"]))
    story.append(Paragraph(f"Generated at: {now}", styles["Italic"]))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "This report compares 2D-FullMesh, 2D-Torus, 3D-Torus, and Clos topologies under the shared 8 SSU + 2 Union exchange-node model. Metrics are evaluated at SSU-to-SSU granularity and grouped into structural, A2A, and sparse 1-to-N communication views.",
            body_style,
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    summary_rows = [[
        "Topology",
        "Diameter",
        "Average Hops",
        "Bisection BW (Gbps)",
        "A2A Per-SSU Throughput (Gbps)",
        "Sparse P95 Completion (ms)",
    ]]
    for item in results:
        summary_rows.append(
            [
                item["name"],
                _fmt_value(item["structural_metrics"]["diameter"]),
                _fmt_value(item["structural_metrics"]["average_hops"]),
                _fmt_value(item["structural_metrics"]["bisection_bandwidth_gbps"]),
                _fmt_value(item["communication_metrics"]["A2A"]["per_ssu_throughput_gbps"]),
                _fmt_value(item["communication_metrics"]["Sparse 1-to-N"]["completion_time_p95_s"] * 1e3),
            ]
        )
    story.append(_styled_table(summary_rows, [32 * mm, 22 * mm, 24 * mm, 30 * mm, 42 * mm, 38 * mm]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Route And Model Notes", styles["Heading2"]))
    story.append(
        Paragraph(
            "1. Same-exchange SSU traffic stays inside the exchange node via Union switching. 2. Inter-exchange SSU traffic follows source SSU -> source Union -> backend topology -> destination Union -> destination SSU. 3. Direct-connect topologies can use DOR or PORT_BALANCED, while Clos can use ECMP over equal-cost shortest paths.",
            body_style,
        )
    )
    story.append(Spacer(1, 10))

    for item in results:
        story.append(Paragraph(item["name"], styles["Heading2"]))
        _add_key_value_section(story, "Hardware And Topology Configuration", {**item["hardware"], **item["topology"]}, styles)
        _add_key_value_section(story, "Routing And Workload Configuration", {
            "routing_mode": item["routing"]["mode"],
            "message_size_mb": item["workloads"]["message_size_mb"],
            "a2a_scope": item["workloads"]["a2a_scope"],
            "sparse_active_ratio": item["workloads"]["sparse_active_ratio"],
            "sparse_target_count": item["workloads"]["sparse_target_count"],
        }, styles)

        story.append(Paragraph("Structural Metric Comparison", styles["Heading3"]))
        story.append(_styled_table(_metric_rows("Metric", item["structural_metrics"]), [72 * mm, 100 * mm]))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Communication Metric Comparison", styles["Heading3"]))
        story.append(_styled_table(_metric_rows("A2A", item["communication_metrics"]["A2A"]), [72 * mm, 100 * mm]))
        story.append(Spacer(1, 4))
        story.append(_styled_table(_metric_rows("Sparse 1-to-N", item["communication_metrics"]["Sparse 1-to-N"]), [72 * mm, 100 * mm]))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Key Observations", styles["Heading3"]))
        for note in item["observations"]:
            story.append(Paragraph(f"- {note}", body_style))
        for note in item["routing"]["notes"]:
            story.append(Paragraph(f"- {note}", body_style))
        story.append(Spacer(1, 10))

    doc.build(story)
    return output_path
