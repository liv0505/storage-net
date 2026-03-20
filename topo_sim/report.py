from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PAGE_BG = colors.HexColor("#020617")
CARD_BG = colors.HexColor("#111827")
CARD_BG_ALT = colors.HexColor("#0f172a")
CARD_BG_SOFT = colors.HexColor("#0b1220")
TEXT = colors.HexColor("#e5eef9")
MUTED = colors.HexColor("#95a3b8")
CYAN = colors.HexColor("#38bdf8")
AMBER = colors.HexColor("#f59e0b")
GRID_LINE = colors.HexColor("#243244")


def _paint_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(PAGE_BG)
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#08101f"))
    canvas.rect(0, doc.pagesize[1] - 36 * mm, doc.pagesize[0], 24 * mm, stroke=0, fill=1)
    canvas.restoreState()


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
                ("BACKGROUND", (0, 0), (-1, 0), CARD_BG_SOFT),
                ("TEXTCOLOR", (0, 0), (-1, 0), TEXT),
                ("GRID", (0, 0), (-1, -1), 0.45, GRID_LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CARD_BG, CARD_BG_ALT]),
                ("TEXTCOLOR", (0, 1), (-1, -1), TEXT),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.6),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
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
    story.append(Paragraph(title, styles["SectionHeading"]))
    rows = [["Field", "Value"]]
    for key, value in data.items():
        rows.append([key.replace("_", " ").title(), _fmt_value(value)])
    story.append(_styled_table(rows, [62 * mm, 110 * mm]))
    story.append(Spacer(1, 7))


def _structural_rows(metrics: dict[str, float]) -> list[list[str]]:
    rows = [["Metric", "Value"]]
    for key, value in metrics.items():
        rows.append([key.replace("_", " ").title(), _fmt_value(value)])
    return rows


def _default_throughput_rows(item: dict[str, Any]) -> list[list[str]]:
    highlight = item["default_routing_highlight"]
    label = highlight["label"]
    return [
        ["Workload", "Route", "Per SSU Throughput (Gbps)"],
        ["A2A", label, f"{highlight['a2a_per_ssu_throughput_gbps']:.2f}"],
        ["Sparse 1-to-N", label, f"{highlight['sparse_per_ssu_throughput_gbps']:.2f}"],
    ]


def _comparison_table_rows(section: dict[str, Any], columns: list[dict[str, str]]) -> list[list[str]]:
    rows = [["Mode", *[column["label"] for column in columns]]]
    for row in section["rows"]:
        rendered = [row["mode"]]
        for column in columns:
            value = row[column["key"]]
            if column["key"] == "per_ssu_throughput_gbps":
                rendered.append(f"{value:.2f} Gbps")
            elif column["key"] in {"completion_time_s", "completion_time_p95_s"}:
                rendered.append(f"{value:.4f} s")
            elif column["key"] == "max_link_utilization":
                rendered.append(f"{value * 100:.2f}%")
            else:
                rendered.append(f"{value:.3f}")
        rows.append(rendered)
    return rows


def _single_route_summary_rows(item: dict[str, Any]) -> list[list[str]]:
    return [
        ["Workload", "Per SSU Throughput", "Completion Time", "P95 Completion", "Max Link Utilization", "Link Utilization CV"],
        [
            "A2A",
            f"{item['communication_metrics']['A2A']['per_ssu_throughput_gbps']:.2f} Gbps",
            f"{item['communication_metrics']['A2A']['completion_time_s']:.4f} s",
            f"{item['communication_metrics']['A2A']['completion_time_p95_s']:.4f} s",
            f"{item['communication_metrics']['A2A']['max_link_utilization'] * 100:.2f}%",
            f"{item['communication_metrics']['A2A']['link_utilization_cv']:.3f}",
        ],
        [
            "Sparse 1-to-N",
            f"{item['communication_metrics']['Sparse 1-to-N']['per_ssu_throughput_gbps']:.2f} Gbps",
            f"{item['communication_metrics']['Sparse 1-to-N']['completion_time_s']:.4f} s",
            f"{item['communication_metrics']['Sparse 1-to-N']['completion_time_p95_s']:.4f} s",
            f"{item['communication_metrics']['Sparse 1-to-N']['max_link_utilization'] * 100:.2f}%",
            f"{item['communication_metrics']['Sparse 1-to-N']['link_utilization_cv']:.3f}",
        ],
    ]


def build_pdf_report(results: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        pageCompression=0,
    )
    base_styles = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {
        "Title": ParagraphStyle(
            "DarkTitle",
            parent=base_styles["Title"],
            textColor=TEXT,
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            spaceAfter=6,
        ),
        "Meta": ParagraphStyle(
            "DarkMeta",
            parent=base_styles["Italic"],
            textColor=MUTED,
            fontName="Helvetica",
            fontSize=9.5,
            leading=12,
            spaceAfter=6,
        ),
        "Body": ParagraphStyle(
            "DarkBody",
            parent=base_styles["BodyText"],
            textColor=TEXT,
            fontName="Helvetica",
            fontSize=10,
            leading=15,
        ),
        "Heading": ParagraphStyle(
            "DarkHeading",
            parent=base_styles["Heading2"],
            textColor=CYAN,
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            spaceBefore=6,
            spaceAfter=6,
        ),
        "SectionHeading": ParagraphStyle(
            "DarkSectionHeading",
            parent=base_styles["Heading3"],
            textColor=AMBER,
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=3,
            spaceAfter=5,
        ),
        "Bullet": ParagraphStyle(
            "DarkBullet",
            parent=base_styles["BodyText"],
            textColor=TEXT,
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            leftIndent=8,
            bulletIndent=0,
        ),
        "Muted": ParagraphStyle(
            "DarkMuted",
            parent=base_styles["BodyText"],
            textColor=MUTED,
            fontName="Helvetica",
            fontSize=9.2,
            leading=13,
        ),
    }

    story: list[Any] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph("SSU-Centric Topology Analysis Report", styles["Title"]))
    story.append(Paragraph(f"Generated at: {now}", styles["Meta"]))
    story.append(Spacer(1, 5))
    story.append(
        Paragraph(
            "This report follows the same dark, topology-first presentation as the HTML dashboard. It compares 2D-FullMesh, 2D-Torus, 3D-Torus, and Clos under the shared 8 SSU + 2 Union exchange-node model.",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("Topology Snapshot", styles["Heading"]))
    summary_rows = [[
        "Topology",
        "Diameter",
        "Average Hops",
        "Bisection BW (Gbps)",
        "Bisection BW / SSU",
        "Default Route Throughput (Gbps)",
        "Sparse P95 Completion (ms)",
    ]]
    for item in results:
        summary_rows.append(
            [
                item["name"],
                _fmt_value(item["structural_metrics"]["diameter"]),
                _fmt_value(item["structural_metrics"]["average_hops"]),
                _fmt_value(item["structural_metrics"]["bisection_bandwidth_gbps"]),
                _fmt_value(item["structural_metrics"]["bisection_bandwidth_gbps_per_ssu"]),
                _fmt_value(item["default_routing_highlight"]["a2a_per_ssu_throughput_gbps"]),
                _fmt_value(item["communication_metrics"]["Sparse 1-to-N"]["completion_time_p95_s"] * 1e3),
            ]
        )
    story.append(_styled_table(summary_rows, [24 * mm, 16 * mm, 20 * mm, 26 * mm, 24 * mm, 38 * mm, 30 * mm]))
    story.append(Spacer(1, 10))

    for item in results:
        story.append(Paragraph(item["name"], styles["Heading"]))
        story.append(
            Paragraph(
                "Topology figure stays first in the HTML dashboard; the PDF mirrors that hierarchy with compact configuration and routing tables before the smaller observation notes.",
                styles["Muted"],
            )
        )
        story.append(Spacer(1, 4))

        _add_key_value_section(
            story,
            "Hardware And Topology Configuration",
            {**item["hardware"], **item["topology"]},
            styles,
        )
        _add_key_value_section(
            story,
            "Routing And Workload Configuration",
            {
                "routing_mode": item["routing"]["mode"],
                "message_size_mb": item["workloads"]["message_size_mb"],
                "a2a_scope": item["workloads"]["a2a_scope"],
                "sparse_active_ratio": item["workloads"]["sparse_active_ratio"],
                "sparse_target_count": item["workloads"]["sparse_target_count"],
            },
            styles,
        )

        if item.get("routing_diversity"):
            story.append(Paragraph("Routing Diversity Snapshot", styles["SectionHeading"]))
            story.append(Paragraph(item["routing_diversity"]["summary"], styles["Body"]))
            diversity_rows = [["Mode", "Avg Paths", "Peak Paths"]]
            for mode_item in item["routing_diversity"]["modes"]:
                diversity_rows.append(
                    [
                        mode_item["mode"],
                        _fmt_value(mode_item["avg_path_count"]),
                        _fmt_value(mode_item["max_path_count"]),
                    ]
                )
            story.append(_styled_table(diversity_rows, [52 * mm, 55 * mm, 55 * mm]))
            story.append(Spacer(1, 7))

        story.append(Paragraph("Default Route Throughput", styles["SectionHeading"]))
        story.append(_styled_table(_default_throughput_rows(item), [40 * mm, 54 * mm, 70 * mm]))
        story.append(Spacer(1, 7))

        story.append(Paragraph("Structural Metrics", styles["SectionHeading"]))
        story.append(_styled_table(_structural_rows(item["structural_metrics"]), [72 * mm, 100 * mm]))
        story.append(Spacer(1, 7))

        if item.get("routing_comparison"):
            for section in item["routing_comparison"]["sections"]:
                story.append(Paragraph(section["title"], styles["SectionHeading"]))
                story.append(
                    _styled_table(
                        _comparison_table_rows(section, item["routing_comparison"]["columns"]),
                        [28 * mm, 28 * mm, 24 * mm, 24 * mm, 28 * mm, 24 * mm],
                    )
                )
                story.append(Spacer(1, 7))
        else:
            story.append(Paragraph("ECMP Workload Summary", styles["SectionHeading"]))
            story.append(
                _styled_table(
                    _single_route_summary_rows(item),
                    [28 * mm, 30 * mm, 28 * mm, 28 * mm, 34 * mm, 26 * mm],
                )
            )
            story.append(Spacer(1, 7))

        story.append(Paragraph("Key Observations", styles["SectionHeading"]))
        for note in item["routing"]["notes"]:
            story.append(Paragraph(f"- {note}", styles["Bullet"]))
        for note in item["observations"]:
            story.append(Paragraph(f"- {note}", styles["Bullet"]))
        story.append(Spacer(1, 10))

    doc.build(story, onFirstPage=_paint_page, onLaterPages=_paint_page)
    return output_path
