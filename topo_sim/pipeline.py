from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import networkx as nx

from .config import AnalysisConfig
from .labels import display_workload_name
from .metrics import compute_structural_metrics, evaluate_workload, evaluate_workload_with_details
from .report import build_pdf_report
from .routing import compute_paths, normalize_routing_mode
from .topologies import available_topologies, build_topology
from .traffic import build_a2a_demands, build_sparse_random_demands, load_custom_traffic_profile
from .visualization import render_html_dashboard


_COMPARISON_COLUMNS: list[dict[str, str]] = [
    {"key": "per_ssu_throughput_gbps", "label": "Per SSU Throughput", "unit": "Gbps"},
    {"key": "completion_time_s", "label": "Completion Time", "unit": "s"},
    {"key": "completion_time_p95_s", "label": "P95 Completion", "unit": "s"},
    {"key": "average_hops", "label": "Average Hops", "unit": "count"},
    {"key": "max_link_utilization", "label": "Max Link Utilization", "unit": "ratio"},
    {"key": "link_utilization_cv", "label": "Link Utilization CV", "unit": "ratio"},
]

_HOP_VOLUME_DISTRIBUTION_HEADERS = [
    "topology",
    "workload",
    "routing_mode",
    "hop_count",
    "offered_volume_gb",
    "offered_volume_pct",
]

_LINK_VOLUME_DISTRIBUTION_HEADERS = [
    "topology",
    "workload",
    "routing_mode",
    "offered_volume_gb",
    "link_count",
    "link_ratio_pct",
]


def _is_df_name(name: str) -> bool:
    upper = str(name).strip().upper()
    return upper == "DF" or upper.startswith("DF-")


def _node_sort_key(node_id: str) -> tuple[int, str, int]:
    exchange_id, local_id = str(node_id).split(":", 1)
    prefix = "".join(ch for ch in local_id if not ch.isdigit())
    digits = "".join(ch for ch in local_id if ch.isdigit())
    return (
        int(exchange_id.removeprefix("en")),
        prefix,
        int(digits) if digits else 0,
    )


def _prefix_workload_metrics(prefix: str, workload: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}_completion_time_s": workload["completion_time_s"],
        f"{prefix}_completion_time_p50_s": workload["completion_time_p50_s"],
        f"{prefix}_completion_time_p95_s": workload["completion_time_p95_s"],
        f"{prefix}_per_ssu_throughput_gbps": workload["per_ssu_throughput_gbps"],
        f"{prefix}_average_hops": workload["average_hops"],
        f"{prefix}_max_link_utilization": workload["max_link_utilization"],
        f"{prefix}_link_utilization_cv": workload["link_utilization_cv"],
    }


def _workload_group(prefix: str, machine_metrics: dict[str, float]) -> dict[str, float]:
    return {
        "completion_time_s": machine_metrics[f"{prefix}_completion_time_s"],
        "completion_time_p50_s": machine_metrics[f"{prefix}_completion_time_p50_s"],
        "completion_time_p95_s": machine_metrics[f"{prefix}_completion_time_p95_s"],
        "per_ssu_throughput_gbps": machine_metrics[f"{prefix}_per_ssu_throughput_gbps"],
        "average_hops": machine_metrics[f"{prefix}_average_hops"],
        "max_link_utilization": machine_metrics[f"{prefix}_max_link_utilization"],
        "link_utilization_cv": machine_metrics[f"{prefix}_link_utilization_cv"],
    }


def _analysis_mode_for_topology(name: str, cfg: AnalysisConfig) -> str:
    if name == "Clos":
        return "ECMP"
    if _is_df_name(name):
        return "SHORTEST_PATH"
    return normalize_routing_mode(cfg.routing_mode)


def _custom_traffic_profile(g: nx.Graph, cfg: AnalysisConfig) -> dict[str, Any] | None:
    if not cfg.custom_traffic_file:
        return None
    profile = load_custom_traffic_profile(
        g,
        cfg.custom_traffic_file,
        workload_name=cfg.custom_traffic_name,
    )
    return {
        "name": profile.name,
        "description": profile.description,
        "input_path": profile.input_path,
        "demands": profile.demands,
    }


def _workload_demands(g: nx.Graph, cfg: AnalysisConfig) -> dict[str, list[Any]]:
    workloads = {
        "A2A": build_a2a_demands(g, cfg),
        "Sparse 1-to-N": build_sparse_random_demands(g, cfg),
    }
    custom_profile = _custom_traffic_profile(g, cfg)
    if custom_profile is not None:
        workloads[str(custom_profile["name"])] = list(custom_profile["demands"])
    return workloads


def _evaluate_named_workloads(
    g: nx.Graph,
    cfg: AnalysisConfig,
    routing_mode: str,
    demands: dict[str, list[Any]] | None = None,
) -> dict[str, dict[str, float]]:
    workload_details = _evaluate_named_workloads_with_details(g, cfg, routing_mode, demands)
    return {
        workload_name: dict(detail["metrics"])
        for workload_name, detail in workload_details.items()
    }


def _evaluate_named_workloads_with_details(
    g: nx.Graph,
    cfg: AnalysisConfig,
    routing_mode: str,
    demands: dict[str, list[Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    workload_demands = demands or _workload_demands(g, cfg)
    workload_details: dict[str, dict[str, Any]] = {}
    for workload_name, demand_list in workload_demands.items():
        demand_sequence = demand_list if isinstance(demand_list, list) else list(demand_list)
        detail = evaluate_workload_with_details(
            g,
            demand_sequence,
            routing_mode=routing_mode,
            cfg=cfg,
        )
        detail["active_sources"] = sorted(
            {str(demand.src) for demand in demand_sequence},
            key=_node_sort_key,
        )
        detail["target_ssus"] = sorted(
            {str(demand.dst) for demand in demand_sequence},
            key=_node_sort_key,
        )
        workload_details[workload_name] = detail
    return workload_details


def _distribution_detail_rows(
    topology_name: str,
    routing_mode: str,
    workload_details: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hop_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []

    for workload_name, detail in workload_details.items():
        for item in detail.get("hop_volume_distribution", []):
            hop_rows.append(
                {
                    "topology": topology_name,
                    "workload": workload_name,
                    "routing_mode": routing_mode,
                    "hop_count": item["hop_count"],
                    "offered_volume_gb": item["offered_volume_gb"],
                    "offered_volume_pct": item["offered_volume_pct"],
                }
            )
        for item in detail.get("link_volume_distribution", []):
            link_rows.append(
                {
                    "topology": topology_name,
                    "workload": workload_name,
                    "routing_mode": routing_mode,
                    "offered_volume_gb": item["offered_volume_gb"],
                    "link_count": item["link_count"],
                    "link_ratio_pct": item["link_ratio_pct"],
                }
            )

    return hop_rows, link_rows


def _build_machine_metrics(
    name: str,
    g: nx.Graph,
    cfg: AnalysisConfig,
    workload_details: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    structural_metrics = compute_structural_metrics(g)
    if workload_details is None:
        routing_mode = _analysis_mode_for_topology(name, cfg)
        workload_details = _evaluate_named_workloads_with_details(g, cfg, routing_mode)
    workload_metrics = {
        workload_name: dict(detail["metrics"])
        for workload_name, detail in workload_details.items()
    }
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


def _topology_scale(name: str, g: nx.Graph) -> str:
    if name == "2D-FullMesh":
        return "4 x 4 exchange nodes"
    if name == "2D-Torus":
        logical_unions = int(g.graph.get("logical_plane_union_count", 0))
        logical_ssus = int(g.graph.get("logical_plane_ssu_count", 0))
        return f"single torus plane | {logical_unions} Union | {logical_ssus} SSU"
    if name == "3D-Torus":
        logical_unions = int(g.graph.get("logical_plane_union_count", 0))
        logical_ssus = int(g.graph.get("logical_plane_ssu_count", 0))
        return f"single torus plane | {logical_unions} Union | {logical_ssus} SSU"
    if _is_df_name(name):
        plane_count = int(g.graph.get("df_plane_count", 1))
        server_count = int(g.graph.get("df_server_count", 0))
        group_count = int(g.graph.get("df_group_count", 0))
        unions_per_plane = int(g.graph.get("df_union_count_per_plane", 0))
        return (
            f"{group_count} shared exchange groups | {plane_count} DF planes | "
            f"{server_count} local 4-Union groups/plane | {unions_per_plane} Union/plane"
        )
    return "18 exchange nodes"


def _topology_pattern(name: str, g: nx.Graph, cfg: AnalysisConfig) -> str:
    if name == "2D-FullMesh":
        return (
            "Each Union uses 6 backend ports: 3 row links on the X axis and 3 column links on the Y axis, "
            "so both Union chips carry the same 4x4 full-mesh structure."
        )
    if name == "2D-Torus":
        return (
            "Each Union uses 4 backend ports on one 4x4 torus of Union chips, and the in-group Union-to-Union link is one of the torus X-direction edges rather than an extra local link."
        )
    if name == "3D-Torus":
        return (
            "Each Union uses 6 backend ports on one 4x4x4 torus of Union chips, and the in-group Union-to-Union link is one of the torus Z-direction edges rather than an extra local link."
        )
    if _is_df_name(name):
        local_ports = int(g.graph.get("df_local_ports_per_union", 0))
        global_ports = int(g.graph.get("df_global_ports_per_union", 0))
        local_topology = str(g.graph.get("df_local_topology", "fullmesh"))
        plane_count = int(g.graph.get("df_plane_count", 1))
        group_count = int(g.graph.get("df_group_count", 0))
        plane_phrase = (
            f"{plane_count} Dragon-Fly planes over {group_count} shared 8-SSU + 2-Union groups"
            if plane_count > 1
            else "one Dragon-Fly plane"
        )
        if name == "DF":
            return (
                f"Each Union uses {local_ports} local 400 Gbps links and {global_ports} global 400 Gbps links inside {plane_phrase}, "
                "and every physical group contributes its union0 to plane 0 and union1 to plane 1."
            )
        if name == "DF-Shuffled":
            return (
                f"Each Union still uses {local_ports} local 400 Gbps links and {global_ports} global 400 Gbps links inside {plane_phrase}, "
                "but the inter-group links rotate their destination Union choice to spread gateway pressure more evenly within each plane."
            )
        if name == "DF-ScaleUp":
            local_label = "ring" if local_topology == "ring" else local_topology
            return (
                f"Each Union uses {local_ports} local 400 Gbps links and {global_ports} global 400 Gbps links inside {plane_phrase}, "
                f"with {local_label} server-local connectivity to trade some in-server path richness for higher server radix."
            )
        if name == "DF-2P-Double-4Global":
            return (
                f"Each 2-Union exchange unit uses one aggregated 800 Gbps local bond (2 parallel 400 Gbps links), "
                f"and each Union keeps 4 global 400 Gbps links inside {plane_phrase}."
            )
        if name == "DF-2P-Triple-3Global":
            return (
                f"Each 2-Union exchange unit uses one aggregated 1200 Gbps local bond (3 parallel 400 Gbps links), "
                f"and each Union keeps 3 global 400 Gbps links inside {plane_phrase}."
            )
        if name == "DF-2P-Double-Bridge-3Global":
            return (
                f"Each 2-Union exchange unit uses one aggregated 800 Gbps local bond, "
                f"and the two 2P units inside a server add one 400 Gbps bridge per Union lane before exposing 3 global 400 Gbps links inside {plane_phrase}."
            )
        return f"Each Union uses {local_ports} local 400 Gbps links and {global_ports} global 400 Gbps links inside {plane_phrase}."
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
        int(data.get("parallel_links", 1))
        for _, _, data in g.edges(data=True)
        if data.get("link_kind") == "backend_interconnect"
    )
    internal_link_count = sum(
        1 for _, _, data in g.edges(data=True) if data.get("link_kind") == "internal_ssu_uplink"
    )

    if _is_df_name(name):
        backend_ports_per_union = int(g.graph.get("df_backend_ports_per_union", 0))
    else:
        backend_ports_per_union = {
            "2D-FullMesh": 6,
            "2D-Torus": 4,
            "3D-Torus": 6,
            "Clos": cfg.clos_uplinks_per_exchange_node,
        }[name]

    topology_cfg = {
        "scale": _topology_scale(name, g),
        "exchange_node_count": exchange_node_count,
        "ssu_count": ssu_count,
        "union_count": union_count,
        "internal_link_count": internal_link_count,
        "backend_link_count": backend_link_count,
        "backend_ports_per_union": backend_ports_per_union,
        "backend_pattern": _topology_pattern(name, g, cfg),
    }
    if name == "Clos":
        topology_cfg["clos_uplinks_per_union_plane"] = cfg.clos_uplinks_per_exchange_node
        topology_cfg["clos_total_uplinks_per_exchange_node"] = cfg.clos_uplinks_per_exchange_node * 2
    if _is_df_name(name):
        topology_cfg["server_count"] = int(g.graph.get("df_total_server_count", 0))
        topology_cfg["server_count_per_plane"] = int(g.graph.get("df_server_count", 0))
        topology_cfg["df_total_server_count"] = int(g.graph.get("df_total_server_count", 0))
        topology_cfg["df_group_count"] = int(g.graph.get("df_group_count", 0))
        topology_cfg["exchange_nodes_per_server"] = int(g.graph.get("df_exchange_nodes_per_server", 0))
        topology_cfg["unions_per_server"] = int(g.graph.get("df_unions_per_server", 0))
        topology_cfg["df_external_servers_per_union"] = int(
            g.graph.get("df_external_servers_per_union", 0)
        )
        topology_cfg["df_variant"] = str(g.graph.get("df_variant", name))
        topology_cfg["df_plane_count"] = int(g.graph.get("df_plane_count", 1))
        topology_cfg["df_exchange_count_per_plane"] = int(g.graph.get("df_exchange_count_per_plane", 0))
        topology_cfg["df_union_count_per_plane"] = int(g.graph.get("df_union_count_per_plane", 0))
        topology_cfg["df_ssu_count_per_plane"] = int(g.graph.get("df_ssu_count_per_plane", 0))
        topology_cfg["df_local_topology"] = str(g.graph.get("df_local_topology", "fullmesh"))
        topology_cfg["df_local_ports_per_union"] = int(g.graph.get("df_local_ports_per_union", 0))
        topology_cfg["df_global_ports_per_union"] = int(g.graph.get("df_global_ports_per_union", 0))
    return topology_cfg


def _routing_configuration(name: str, cfg: AnalysisConfig) -> dict[str, Any]:
    mode = _analysis_mode_for_topology(name, cfg)
    direct_connect = name in {"2D-FullMesh", "2D-Torus", "3D-Torus"}
    if _is_df_name(name):
        return {
            "mode": mode,
            "requested_mode": cfg.routing_mode,
            "notes": [
                "same-exchange SSU traffic stays inside the local 8 SSU + 2 Union unit and splits across all shortest local Union choices",
                "each exchange group shares its 8 SSUs across two independent Dragon-Fly Union planes, with union0 on plane 0 and union1 on plane 1",
                "routing always follows shortest Union-to-Union paths on the backend graph, so traffic may choose either plane when both planes offer the same minimum hop count",
                "within each plane, local 4-Union groups and inter-group Dragon-Fly links are built independently, but the two planes only meet at the shared SSUs",
                "if local Union connectivity is not fully bridged, traffic between groups in the same local 4-Union block can legitimately detour through the global Dragon-Fly plane",
                "completion and utilization metrics are evaluated with single-direction traffic accounting",
            ],
        }
    notes = [
        "same-exchange SSU traffic stays inside the exchange node via Union switching",
        "source SSU traffic evenly splits across both local 200 Gbps uplinks into the two Union planes",
        "inter-exchange SSU traffic is modeled as source SSU -> source Union -> backend topology -> destination Union -> destination SSU",
        "destination-side traffic evenly splits from the two destination Unions down to the target SSU",
    ]
    if name == "2D-FullMesh":
        notes.append(
            "both Union chips expose the same direct-connect backend plane, so routing decisions are evaluated independently on each Union plane"
        )
    elif name in {"2D-Torus", "3D-Torus"}:
        notes.append(
            "the two Union chips inside each exchange node are locally coupled into one connected torus backend, so routing can enter the torus through either Union"
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
    payload = {
        "message_size_mb": cfg.message_size_mb,
        "a2a_scope": "all SSUs send to every other SSU",
        "sparse_active_ratio": cfg.sparse_active_ratio,
        "sparse_target_count": cfg.sparse_target_count,
    }
    if cfg.custom_traffic_file:
        payload["custom_traffic_file"] = cfg.custom_traffic_file
        payload["custom_traffic_name"] = cfg.custom_traffic_name
    return payload


def _workload_description_payload(
    g: nx.Graph,
    cfg: AnalysisConfig,
    workload_details: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    total_ssus = sum(1 for _, data in g.nodes(data=True) if data.get("node_role") == "ssu")
    sparse_sources = len(workload_details["Sparse 1-to-N"].get("active_sources", []))
    sparse_ratio_pct = cfg.sparse_active_ratio * 100.0
    component_ssu_counts = sorted(
        [
            sum(1 for node_id in component if g.nodes[node_id].get("node_role") == "ssu")
            for component in nx.connected_components(g)
        ]
    )
    component_ssu_counts = [count for count in component_ssu_counts if count > 0]
    component_count = len(component_ssu_counts)
    component_summary = ""
    if component_count > 1:
        if len(set(component_ssu_counts)) == 1:
            component_summary = f"{component_count} disconnected components x {component_ssu_counts[0]} SSUs"
        else:
            component_summary = ", ".join(str(count) for count in component_ssu_counts)

    payload = [
        {
            "title": "A2A",
            "description": (
                (
                    f"A2A runs independently inside {component_summary}; each SSU sends "
                    f"{cfg.message_size_mb:.2f} MB to every other SSU in its own component."
                )
                if component_count > 1
                else f"{total_ssus} SSUs each send {cfg.message_size_mb:.2f} MB to every other SSU."
            ),
        },
        {
            "title": display_workload_name("Sparse 1-to-N"),
            "description": (
                (
                    f"{sparse_sources} active sources ({sparse_ratio_pct:.0f}%) each send "
                    f"{cfg.message_size_mb:.2f} MB to {cfg.sparse_target_count} destinations inside their own component."
                )
                if component_count > 1
                else f"{sparse_sources} active sources ({sparse_ratio_pct:.0f}%) each send "
                f"{cfg.message_size_mb:.2f} MB to {cfg.sparse_target_count} destinations."
            ),
        },
    ]

    custom_profile = _custom_traffic_profile(g, cfg)
    if custom_profile is not None:
        payload.append(
            {
                "title": str(custom_profile["name"]),
                "description": str(custom_profile["description"]),
            }
        )
    return payload


def _routing_mode_description(mode: str, topology_name: str, cfg: AnalysisConfig) -> str:
    if mode == "DOR":
        if topology_name == "2D-Torus":
            return "Fixed shortest routing in X -> Y order."
        if topology_name == "3D-Torus":
            return "Fixed shortest routing in X -> Y -> Z order."
        if topology_name == "2D-FullMesh":
            return "Fixed shortest routing that walks X first, then Y."
        return "Fixed shortest routing with deterministic dimension order."
    if mode == "SHORTEST_PATH":
        if _is_df_name(topology_name):
            return "Single shortest-path routing on the Dragon-Fly Union graph."
        return "Evenly splits traffic across all shortest paths."
    if mode == "FULL_PATH":
        return f"Uses every backend egress; prefers shortest paths and allows up to {cfg.port_balanced_max_detour_hops} extra hop(s) if needed."
    if mode == "ECMP":
        return "Splits Clos traffic across equal-cost spine paths."
    return "Topology-specific routing behavior."


def _routing_mode_description_payload(name: str, cfg: AnalysisConfig) -> list[dict[str, str]]:
    return [
        {
            "mode": mode,
            "description": _routing_mode_description(mode, name, cfg),
        }
        for mode in _comparison_modes_for_topology(name)
    ]


def _comparison_modes_for_topology(name: str) -> list[str]:
    if name == "Clos":
        return ["ECMP"]
    if _is_df_name(name):
        return ["SHORTEST_PATH"]
    return ["DOR", "SHORTEST_PATH", "FULL_PATH"]


def _default_highlight_mode(name: str) -> str:
    if name == "Clos":
        return "ECMP"
    if _is_df_name(name):
        return "SHORTEST_PATH"
    return "DOR"


def _comparison_metric_subset(metrics: dict[str, float]) -> dict[str, float]:
    return {column["key"]: metrics[column["key"]] for column in _COMPARISON_COLUMNS}


def _routing_comparison_payload(name: str, g: nx.Graph, cfg: AnalysisConfig) -> dict[str, Any] | None:
    if name == "Clos" or _is_df_name(name):
        return None

    demands = _workload_demands(g, cfg)
    workload_names = list(demands.keys())
    per_mode_results = {
        mode: _evaluate_named_workloads(g, cfg, mode, demands)
        for mode in _comparison_modes_for_topology(name)
    }

    sections = []
    for title in workload_names:
        sections.append(
            {
                "title": f"{display_workload_name(title)} Routing Comparison",
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
            f"{display_workload_name('Sparse 1-to-N')} communication completes in "
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
    workload_details: dict[str, dict[str, Any]],
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
    communication_metrics = {
        "A2A": a2a_metrics,
        "Sparse 1-to-N": sparse_metrics,
    }
    for workload_name, detail in workload_details.items():
        if workload_name in communication_metrics:
            continue
        communication_metrics[workload_name] = dict(detail["metrics"])
    workload_metric_rows = [
        {
            "workload": workload_name,
            "display_name": display_workload_name(workload_name),
            **metrics,
        }
        for workload_name, metrics in communication_metrics.items()
    ]

    return {
        "name": name,
        "graph": g,
        "layout_seed": layout_seed,
        "hardware": _hardware_assumptions(),
        "topology": _topology_configuration(name, g, cfg),
        "routing": _routing_configuration(name, cfg),
        "routing_diversity": _routing_diversity_snapshot(name, g, cfg),
        "workloads": _workload_configuration(cfg),
        "workload_descriptions": _workload_description_payload(g, cfg, workload_details),
        "routing_mode_descriptions": _routing_mode_description_payload(name, cfg),
        "structural_metrics": structural_metrics,
        "communication_metrics": communication_metrics,
        "workload_metric_rows": workload_metric_rows,
        "traffic_workload_names": list(communication_metrics.keys()),
        "traffic_details": workload_details,
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
    hop_distribution_rows: list[dict[str, Any]] = []
    link_distribution_rows: list[dict[str, Any]] = []
    render_results: list[dict[str, Any]] = []

    for offset, name in enumerate(selected):
        g = build_topology(name, cfg)
        routing_mode = _analysis_mode_for_topology(name, cfg)
        workload_details = _evaluate_named_workloads_with_details(g, cfg, routing_mode)
        machine_metrics = _build_machine_metrics(name, g, cfg, workload_details=workload_details)
        hop_rows, link_rows = _distribution_detail_rows(name, routing_mode, workload_details)

        summary_rows.append({"topology": name, **machine_metrics})
        hop_distribution_rows.extend(hop_rows)
        link_distribution_rows.extend(link_rows)
        render_results.append(
            _build_render_result(
                name=name,
                g=g,
                cfg=cfg,
                machine_metrics=machine_metrics,
                workload_details=workload_details,
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

    hop_volume_csv_path = output_dir / "hop_volume_distribution.csv"
    with hop_volume_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HOP_VOLUME_DISTRIBUTION_HEADERS)
        writer.writeheader()
        writer.writerows(hop_distribution_rows)

    link_volume_csv_path = output_dir / "link_volume_distribution.csv"
    with link_volume_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_LINK_VOLUME_DISTRIBUTION_HEADERS)
        writer.writeheader()
        writer.writerows(link_distribution_rows)

    html_path = render_html_dashboard(render_results, output_dir / "topology_dashboard.html")
    pdf_path = build_pdf_report(render_results, output_dir / "topology_report.pdf")

    config_path = output_dir / "run_config.json"
    config_dict = asdict(cfg)
    config_dict["output_dir"] = str(config_dict["output_dir"])
    config_dict["selected_topologies"] = selected
    config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    return {
        "csv": csv_path,
        "hop_volume_csv": hop_volume_csv_path,
        "link_volume_csv": link_volume_csv_path,
        "html": html_path,
        "pdf": pdf_path,
        "config": config_path,
    }
