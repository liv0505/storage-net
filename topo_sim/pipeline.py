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
from .topologies import available_topologies, build_topology
from .traffic import build_a2a_demands, build_sparse_random_demands
from .visualization import render_html_dashboard


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


def _build_machine_metrics(g: nx.Graph, cfg: AnalysisConfig) -> dict[str, float]:
    structural_metrics = compute_structural_metrics(g)
    a2a_metrics = evaluate_workload(
        g,
        build_a2a_demands(g, cfg),
        routing_mode=cfg.routing_mode,
        cfg=cfg,
    )
    sparse_metrics = evaluate_workload(
        g,
        build_sparse_random_demands(g, cfg),
        routing_mode=cfg.routing_mode,
        cfg=cfg,
    )
    return {
        **structural_metrics,
        **_prefix_workload_metrics("a2a", a2a_metrics),
        **_prefix_workload_metrics("sparse", sparse_metrics),
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
        return "Row-wise full mesh on Union0 and column-wise full mesh on Union1"
    if name == "2D-Torus":
        return "Wrap-around X on Union0 and wrap-around Y on Union1"
    if name == "3D-Torus":
        return "Wrap-around X and Z on Union0, wrap-around Y on Union1"
    return (
        "Two-stage Clos with an upper Union stage; each exchange node uses "
        f"{cfg.clos_uplinks_per_exchange_node} x 400 Gbps uplinks"
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

    topology_cfg = {
        "scale": _topology_scale(name),
        "exchange_node_count": exchange_node_count,
        "ssu_count": ssu_count,
        "union_count": union_count,
        "internal_link_count": internal_link_count,
        "backend_link_count": backend_link_count,
        "backend_pattern": _topology_pattern(name, cfg),
    }
    if name == "Clos":
        topology_cfg["clos_uplinks_per_exchange_node"] = cfg.clos_uplinks_per_exchange_node
    return topology_cfg


def _routing_configuration(name: str, cfg: AnalysisConfig) -> dict[str, Any]:
    mode = cfg.routing_mode
    direct_connect = name in {"2D-FullMesh", "2D-Torus", "3D-Torus"}
    notes = [
        "same-exchange SSU traffic stays inside the exchange node via Union switching",
        "inter-exchange SSU traffic is modeled as source SSU -> source Union -> backend topology -> destination Union -> destination SSU",
    ]
    if direct_connect:
        notes.append(
            "PORT_BALANCED evenly splits traffic across source backend ports and can use non-shortest paths only when a selected port has no shortest path"
        )
    if name == "Clos":
        notes.append("ECMP splits traffic across equal-cost shortest paths on the Clos fabric")
    if mode == "DOR":
        notes.append("DOR follows a deterministic dimension order on torus topologies")
    elif mode == "PORT_BALANCED":
        notes.append(
            f"PORT_BALANCED allows up to {cfg.port_balanced_max_detour_hops} additional hop(s) beyond shortest-path distance"
        )
    elif mode == "ECMP":
        notes.append("ECMP keeps traffic on equal-cost shortest paths")
    else:
        notes.append("MIN_HOPS uses a single shortest path as the baseline route")

    return {
        "mode": mode,
        "notes": notes,
    }


def _workload_configuration(cfg: AnalysisConfig) -> dict[str, Any]:
    return {
        "message_size_mb": cfg.message_size_mb,
        "a2a_scope": "all SSUs send to every other SSU",
        "sparse_active_ratio": cfg.sparse_active_ratio,
        "sparse_target_count": cfg.sparse_target_count,
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
    }
    a2a_metrics = _workload_group("a2a", machine_metrics)
    sparse_metrics = _workload_group("sparse", machine_metrics)

    return {
        "name": name,
        "graph": g,
        "layout_seed": layout_seed,
        "hardware": _hardware_assumptions(),
        "topology": _topology_configuration(name, g, cfg),
        "routing": _routing_configuration(name, cfg),
        "workloads": _workload_configuration(cfg),
        "structural_metrics": structural_metrics,
        "communication_metrics": {
            "A2A": a2a_metrics,
            "Sparse 1-to-N": sparse_metrics,
        },
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
        machine_metrics = _build_machine_metrics(g, cfg)

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
