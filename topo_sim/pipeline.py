from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import networkx as nx

from .config import AnalysisConfig
from .metrics import compute_structural_metrics, evaluate_workload
from .report import build_pdf_report
from .routing import compute_paths, normalize_routing_mode
from .topologies import available_topologies, build_topology
from .traffic import build_a2a_demands, build_sparse_random_demands
from .visualization import render_html_dashboard


_COMPARISON_COLUMNS: list[dict[str, str]] = [
    {"key": "per_ssu_throughput_gbps", "label": "Per SSU Throughput", "unit": "Gbps"},
    {"key": "completion_time_s", "label": "Completion Time", "unit": "s"},
    {"key": "completion_time_p95_s", "label": "P95 Completion", "unit": "s"},
    {"key": "max_link_utilization", "label": "Max Link Utilization", "unit": "ratio"},
    {"key": "link_utilization_cv", "label": "Link Utilization CV", "unit": "ratio"},
]


def _prefix_workload_metrics(prefix: str, workload: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}_completion_time_s": workload["completion_time_s"],
        f"{prefix}_completion_time_p50_s": workload["completion_time_p50_s"],
        f"{prefix}_completion_time_p95_s": workload["completion_time_p95_s"],
        f"{prefix}_per_ssu_throughput_gbps": workload["per_ssu_throughput_gbps"],
        f"{prefix}_max_link_utilization": workload["max_link_utilization"],
        f"{prefix}_link_utilization_cv": workload["link_utilization_cv"],
    }


def _workload_group(prefix: str, machine_metrics: dict[str, float]) -> dict[str, float]:
    return {
        "completion_time_s": machine_metrics[f"{prefix}_completion_time_s"],
        "completion_time_p50_s": machine_metrics[f"{prefix}_completion_time_p50_s"],
        "completion_time_p95_s": machine_metrics[f"{prefix}_completion_time_p95_s"],
        "per_ssu_throughput_gbps": machine_metrics[f"{prefix}_per_ssu_throughput_gbps"],
        "max_link_utilization": machine_metrics[f"{prefix}_max_link_utilization"],
        "link_utilization_cv": machine_metrics[f"{prefix}_link_utilization_cv"],
    }


def _analysis_mode_for_topology(name: str, cfg: AnalysisConfig) -> str:
    if name == "Clos":
        return "ECMP"
    return normalize_routing_mode(cfg.routing_mode)


def _workload_demands(g: nx.Graph, cfg: AnalysisConfig) -> dict[str, list[Any]]:
    return {
        "A2A": build_a2a_demands(g, cfg),
        "Sparse 1-to-N": build_sparse_random_demands(g, cfg),
    }


def _evaluate_named_workloads(
    g: nx.Graph,
    cfg: AnalysisConfig,
    routing_mode: str,
    demands: dict[str, list[Any]] | None = None,
) -> dict[str, dict[str, float]]:
    workload_demands = demands or _workload_demands(g, cfg)
    return {
        workload_name: evaluate_workload(g, demand_list, routing_mode=routing_mode, cfg=cfg)
        for workload_name, demand_list in workload_demands.items()
    }


def _build_machine_metrics(name: str, g: nx.Graph, cfg: AnalysisConfig) -> dict[str, float]:
    structural_metrics = compute_structural_metrics(g)
    routing_mode = _analysis_mode_for_topology(name, cfg)
    workload_metrics = _evaluate_named_workloads(g, cfg, routing_mode)
    return {
        **structural_metrics,
        **_prefix_workload_metrics("a2a", workload_metrics["A2A"]),
        **_prefix_workload_metrics("sparse", workload_metrics["Sparse 1-to-N"]),
    }


def _selected_topologies(cfg: AnalysisConfig, topologies: list[str] | None) -> list[str]:
    if topologies is None:
        selected = list(cfg.topology_names or available_topologies())
    else:
        selected = list(topologies)

    if not selected:
        raise ValueError("At least one topology must be selected for analysis")
    return selected


def _hardware_assumptions() -> dict[str, Any]:
    return {
        "exchange_node_unit": "8 SSU + 2 Union",
        "ssus_per_exchange_node": 8,
        "unions_per_exchange_node": 2,
        "internal_ssu_union_link_gbps": 200.0,
        "backend_link_gbps": 400.0,
        "union_total_ub_ports": 18,
        "union_ports_reserved_for_ssu": 8,
        "union_ports_reserved_for_frontend_nics": 4,
        "backend_ports_available_per_union": 6,
    }


def _topology_scale(name: str) -> str:
    if name == "2D-FullMesh":
        return "4 x 4 exchange nodes"
    if name == "2D-Torus":
        return "4 x 4 exchange nodes"
    if name == "3D-Torus":
        return "4 x 4 x 4 exchange nodes"
    return "18 exchange nodes"


def _topology_pattern(name: str, cfg: AnalysisConfig) -> str:
    if name == "2D-FullMesh":
        return (
            "Each Union uses 6 backend ports: 3 row links on the X axis and 3 column links on the Y axis, "
            "so both Union chips carry the same 4x4 full-mesh structure."
        )
    if name == "2D-Torus":
        return (
            "Each Union uses 4 backend ports: X+/X- and Y+/Y- wrap-around torus links, "
            "so both Union chips carry the same 4x4 torus plane."
        )
    if name == "3D-Torus":
        return (
            "Each Union uses 6 backend ports: X+/X-, Y+/Y-, and Z+/Z- wrap-around torus links, "
            "so both Union chips carry the same 4x4x4 torus volume."
        )
    return (
        "Each exchange node keeps two independent Union planes; each Union uplinks via "
        f"{cfg.clos_uplinks_per_exchange_node} x 400 Gbps into its own Clos spine pool, "
        f"for {cfg.clos_uplinks_per_exchange_node * 2} x 400 Gbps total per exchange node."
    )


def _topology_configuration(name: str, g: nx.Graph, cfg: AnalysisConfig) -> dict[str, Any]:
    exchange_node_count = len(
        {
            data.get("exchange_node_id")
            for _, data in g.nodes(data=True)
            if data.get("exchange_node_id") is not None
        }
    )
    ssu_count = sum(1 for _, data in g.nodes(data=True) if data.get("node_role") == "ssu")
    union_count = sum(1 for _, data in g.nodes(data=True) if data.get("node_role") == "union")
    backend_link_count = sum(
        1 for _, _, data in g.edges(data=True) if data.get("link_kind") == "backend_interconnect"
    )
    internal_link_count = sum(
        1 for _, _, data in g.edges(data=True) if data.get("link_kind") == "internal_ssu_uplink"
    )

    backend_ports_per_union = {
        "2D-FullMesh": 6,
        "2D-Torus": 4,
        "3D-Torus": 6,
        "Clos": cfg.clos_uplinks_per_exchange_node,
    }[name]

    topology_cfg = {
        "scale": _topology_scale(name),
        "exchange_node_count": exchange_node_count,
        "ssu_count": ssu_count,
        "union_count": union_count,
        "internal_link_count": internal_link_count,
        "backend_link_count": backend_link_count,
        "backend_ports_per_union": backend_ports_per_union,
        "backend_pattern": _topology_pattern(name, cfg),
    }
    if name == "Clos":
        topology_cfg["clos_uplinks_per_union_plane"] = cfg.clos_uplinks_per_exchange_node
        topology_cfg["clos_total_uplinks_per_exchange_node"] = cfg.clos_uplinks_per_exchange_node * 2
    return topology_cfg


def _routing_configuration(name: str, cfg: AnalysisConfig) -> dict[str, Any]:
    mode = _analysis_mode_for_topology(name, cfg)
    direct_connect = name in {"2D-FullMesh", "2D-Torus", "3D-Torus"}
    notes = [
        "same-exchange SSU traffic stays inside the exchange node via Union switching",
        "source SSU traffic evenly splits across both local 200 Gbps uplinks into the two Union planes",
        "inter-exchange SSU traffic is modeled as source SSU -> source Union -> backend topology -> destination Union -> destination SSU",
        "destination-side traffic evenly splits from the two destination Unions down to the target SSU",
    ]
    if direct_connect:
        notes.append(
            "both Union chips expose the same direct-connect backend plane, so routing decisions are evaluated independently on each Union plane"
        )
    if mode == "DOR":
        if name == "2D-Torus":
            notes.append("DOR keeps each Union plane on deterministic dimension-order shortest routing in X -> Y order")
        elif name == "3D-Torus":
            notes.append("DOR keeps each Union plane on deterministic dimension-order shortest routing in X -> Y -> Z order")
        elif name == "2D-FullMesh":
            notes.append("DOR keeps each Union plane on deterministic dimension-order shortest routing in X -> Y order across the full-mesh axes")
        else:
            notes.append("DOR keeps each Union plane on deterministic dimension-order shortest routing")
    elif mode == "SHORTEST_PATH":
        notes.append("SHORTEST_PATH splits each Union plane across all shortest Union-to-Union paths without a fixed dimension order")
    elif mode == "FULL_PATH":
        notes.append("each Union plane can use every available backend egress port")
        notes.append("FULL_PATH uniformly splits across all available backend egress ports on each Union plane, with one path per egress port")
        notes.append("for each selected egress port, routing prefers a shortest path to the destination Union")
        notes.append("if a selected egress has no shortest path, routing falls back to the least-hop non-shortest path to the destination node")
        notes.append(
            f"non-shortest fallback is limited to {cfg.port_balanced_max_detour_hops} additional hop(s) beyond shortest-path distance when fallback is required"
        )
    elif mode == "ECMP":
        notes.append("ECMP splits Clos traffic across equal-cost shortest paths through the upper Union stage")
        notes.append(
            f"each exchange contributes {cfg.clos_uplinks_per_exchange_node} x 400 Gbps from union0 and {cfg.clos_uplinks_per_exchange_node} x 400 Gbps from union1 into separate Clos spine pools"
        )

    return {
        "mode": mode,
        "requested_mode": cfg.routing_mode,
        "notes": notes,
    }


def _exchange_ids(g: nx.Graph) -> list[str]:
    return sorted(
        {
            str(data.get("exchange_node_id"))
            for _, data in g.nodes(data=True)
            if data.get("exchange_node_id") is not None
        },
        key=lambda value: int(value.removeprefix("en")),
    )


def _routing_diversity_snapshot(name: str, g: nx.Graph, cfg: AnalysisConfig) -> dict[str, Any] | None:
    if name not in {"2D-FullMesh", "2D-Torus", "3D-Torus"}:
        return None

    exchange_ids = _exchange_ids(g)
    if len(exchange_ids) < 2:
        return None

    pair_count = 0
    per_mode_counts: dict[str, list[int]] = {
        "DOR": [],
        "SHORTEST_PATH": [],
        "FULL_PATH": [],
    }

    for index, src_exchange in enumerate(exchange_ids):
        src_ssu = f"{src_exchange}:ssu0"
        for dst_exchange in exchange_ids[index + 1 :]:
            dst_ssu = f"{dst_exchange}:ssu0"
            pair_count += 1
            for mode in per_mode_counts:
                per_mode_counts[mode].append(len(compute_paths(g, src_ssu, dst_ssu, mode, cfg)))

    if pair_count == 0:
        return None

    modes: list[dict[str, Any]] = []
    for mode, counts in per_mode_counts.items():
        average = float(sum(counts) / len(counts)) if counts else 0.0
        peak = int(max(counts)) if counts else 0
        modes.append(
            {
                "mode": mode,
                "avg_path_count": average,
                "max_path_count": peak,
            }
        )

    mode_map = {item["mode"]: item for item in modes}
    dor_avg = max(mode_map["DOR"]["avg_path_count"], 1e-9)
    shortest_avg = mode_map["SHORTEST_PATH"]["avg_path_count"]
    full_avg = mode_map["FULL_PATH"]["avg_path_count"]

    summary = (
        f"SHORTEST_PATH vs DOR: {shortest_avg / dor_avg:.2f}x average path diversity "
        f"({shortest_avg:.2f} vs {mode_map['DOR']['avg_path_count']:.2f} end-to-end paths per inter-exchange SSU pair); "
        f"FULL_PATH reaches {full_avg:.2f} on average with peak {mode_map['FULL_PATH']['max_path_count']} paths."
    )

    return {
        "pair_count": pair_count,
        "modes": modes,
        "summary": summary,
    }


def _workload_configuration(cfg: AnalysisConfig) -> dict[str, Any]:
    return {
        "message_size_mb": cfg.message_size_mb,
        "a2a_scope": "all SSUs send to every other SSU",
        "sparse_active_ratio": cfg.sparse_active_ratio,
        "sparse_target_count": cfg.sparse_target_count,
    }


def _comparison_modes_for_topology(name: str) -> list[str]:
    if name == "Clos":
        return ["ECMP"]
    return ["DOR", "SHORTEST_PATH", "FULL_PATH"]


def _default_highlight_mode(name: str) -> str:
    return "ECMP" if name == "Clos" else "DOR"


def _comparison_metric_subset(metrics: dict[str, float]) -> dict[str, float]:
    return {column["key"]: metrics[column["key"]] for column in _COMPARISON_COLUMNS}


def _routing_comparison_payload(name: str, g: nx.Graph, cfg: AnalysisConfig) -> dict[str, Any] | None:
    if name == "Clos":
        return None

    demands = _workload_demands(g, cfg)
    per_mode_results = {
        mode: _evaluate_named_workloads(g, cfg, mode, demands)
        for mode in _comparison_modes_for_topology(name)
    }

    sections = []
    for title in ("A2A", "Sparse 1-to-N"):
        sections.append(
            {
                "title": f"{title} Routing Comparison",
                "workload": title,
                "rows": [
                    {
                        "mode": mode,
                        **_comparison_metric_subset(per_mode_results[mode][title]),
                    }
                    for mode in _comparison_modes_for_topology(name)
                ],
            }
        )

    return {
        "columns": list(_COMPARISON_COLUMNS),
        "sections": sections,
    }


def _default_routing_highlight(
    name: str,
    cfg: AnalysisConfig,
    comparison_payload: dict[str, Any] | None,
    active_a2a_metrics: dict[str, float],
    active_sparse_metrics: dict[str, float],
) -> dict[str, Any]:
    default_mode = _default_highlight_mode(name)
    if comparison_payload is not None:
        by_workload = {section["workload"]: section for section in comparison_payload["sections"]}
        a2a_row = next(row for row in by_workload["A2A"]["rows"] if row["mode"] == default_mode)
        sparse_row = next(row for row in by_workload["Sparse 1-to-N"]["rows"] if row["mode"] == default_mode)
        return {
            "mode": default_mode,
            "label": f"{default_mode} Throughput",
            "a2a_per_ssu_throughput_gbps": a2a_row["per_ssu_throughput_gbps"],
            "sparse_per_ssu_throughput_gbps": sparse_row["per_ssu_throughput_gbps"],
        }

    return {
        "mode": default_mode,
        "label": f"{default_mode} Throughput",
        "a2a_per_ssu_throughput_gbps": active_a2a_metrics["per_ssu_throughput_gbps"],
        "sparse_per_ssu_throughput_gbps": active_sparse_metrics["per_ssu_throughput_gbps"],
    }


def _build_observations(
    structural_metrics: dict[str, float],
    a2a_metrics: dict[str, float],
    sparse_metrics: dict[str, float],
) -> list[str]:
    return [
        (
            "A2A per-SSU throughput reaches "
            f"{a2a_metrics['per_ssu_throughput_gbps']:.2f} Gbps with maximum backend link utilization "
            f"{a2a_metrics['max_link_utilization']:.2%}."
        ),
        (
            "Sparse 1-to-N communication completes in "
            f"{sparse_metrics['completion_time_p95_s'] * 1e3:.2f} ms at p95 with link-utilization CV "
            f"{sparse_metrics['link_utilization_cv']:.3f}."
        ),
        (
            "Structural span is diameter "
            f"{structural_metrics['diameter']:.0f} and average hops {structural_metrics['average_hops']:.2f}."
        ),
    ]


def _build_render_result(
    name: str,
    g: nx.Graph,
    cfg: AnalysisConfig,
    machine_metrics: dict[str, float],
    layout_seed: int,
) -> dict[str, Any]:
    structural_metrics = {
        "diameter": machine_metrics["diameter"],
        "average_hops": machine_metrics["average_hops"],
        "bisection_bandwidth_gbps": machine_metrics["bisection_bandwidth_gbps"],
        "bisection_bandwidth_gbps_per_ssu": machine_metrics["bisection_bandwidth_gbps_per_ssu"],
    }
    a2a_metrics = _workload_group("a2a", machine_metrics)
    sparse_metrics = _workload_group("sparse", machine_metrics)
    routing_comparison = _routing_comparison_payload(name, g, cfg)

    return {
        "name": name,
        "graph": g,
        "layout_seed": layout_seed,
        "hardware": _hardware_assumptions(),
        "topology": _topology_configuration(name, g, cfg),
        "routing": _routing_configuration(name, cfg),
        "routing_diversity": _routing_diversity_snapshot(name, g, cfg),
        "workloads": _workload_configuration(cfg),
        "structural_metrics": structural_metrics,
        "communication_metrics": {
            "A2A": a2a_metrics,
            "Sparse 1-to-N": sparse_metrics,
        },
        "default_routing_highlight": _default_routing_highlight(
            name,
            cfg,
            routing_comparison,
            a2a_metrics,
            sparse_metrics,
        ),
        "routing_comparison": routing_comparison,
        "observations": _build_observations(structural_metrics, a2a_metrics, sparse_metrics),
    }


def run_full_analysis(cfg: AnalysisConfig, topologies: list[str] | None = None) -> dict[str, Path]:
    selected = _selected_topologies(cfg, topologies)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    render_results: list[dict[str, Any]] = []

    for offset, name in enumerate(selected):
        g = build_topology(name, cfg)
        machine_metrics = _build_machine_metrics(name, g, cfg)

        summary_rows.append({"topology": name, **machine_metrics})
        render_results.append(
            _build_render_result(
                name=name,
                g=g,
                cfg=cfg,
                machine_metrics=machine_metrics,
                layout_seed=cfg.random_seed + offset,
            )
        )

    csv_path = output_dir / "summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    html_path = render_html_dashboard(render_results, output_dir / "topology_dashboard.html")
    pdf_path = build_pdf_report(render_results, output_dir / "topology_report.pdf")

    config_path = output_dir / "run_config.json"
    config_dict = asdict(cfg)
    config_dict["output_dir"] = str(config_dict["output_dir"])
    config_dict["selected_topologies"] = selected
    config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    return {"csv": csv_path, "html": html_path, "pdf": pdf_path, "config": config_path}
