from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Iterable

import networkx as nx
import numpy as np

from .config import AnalysisConfig
from .routing import RoutedPath, compute_paths
from .traffic import FlowDemand, build_a2a_demands


Edge = tuple[object, object]


def _edge_key(u: object, v: object) -> Edge:
    return tuple(sorted((u, v), key=lambda x: str(x)))


def _ssu_nodes(g: nx.Graph) -> list[str]:
    return [str(node_id) for node_id, data in g.nodes(data=True) if data.get("node_role") == "ssu"]


def _bisection_bandwidth_gbps(g: nx.Graph) -> float:
    if g.number_of_nodes() < 2 or g.number_of_edges() == 0:
        return 0.0

    weighted = nx.Graph()
    weighted.add_nodes_from(g.nodes())
    for u, v, data in g.edges(data=True):
        weighted.add_edge(u, v, weight=float(data.get("bandwidth_gbps", 0.0)))

    if weighted.number_of_edges() == 0:
        return 0.0
    if not nx.is_connected(weighted):
        return 0.0

    cut_value, _ = nx.stoer_wagner(weighted, weight="weight")
    return float(cut_value)


def compute_structural_metrics(g: nx.Graph) -> dict[str, float]:
    ssus = _ssu_nodes(g)
    if len(ssus) < 2:
        return {
            "diameter": 0.0,
            "average_hops": 0.0,
            "bisection_bandwidth_gbps": _bisection_bandwidth_gbps(g),
        }

    pair_hops: list[float] = []
    reachable_pairs = 0
    total_pairs = len(ssus) * (len(ssus) - 1) // 2

    for idx, src in enumerate(ssus):
        lengths = nx.single_source_shortest_path_length(g, src)
        for dst in ssus[idx + 1 :]:
            hop_count = lengths.get(dst)
            if hop_count is None:
                continue
            pair_hops.append(float(hop_count))
            reachable_pairs += 1

    if pair_hops:
        average_hops = float(statistics.fmean(pair_hops))
        diameter = float(max(pair_hops))
    else:
        average_hops = 0.0
        diameter = 0.0

    if reachable_pairs < total_pairs:
        diameter = float("inf")

    return {
        "diameter": diameter,
        "average_hops": average_hops,
        "bisection_bandwidth_gbps": _bisection_bandwidth_gbps(g),
    }


def _edge_capacity_bits_per_s(edge_data: dict[str, float], cfg: AnalysisConfig) -> float:
    bandwidth_gbps = float(edge_data.get("bandwidth_gbps", cfg.link_bandwidth_gbps))
    return max(bandwidth_gbps * 1e9, 1.0)


def _completion_time_from_edge_loads(
    g: nx.Graph,
    edge_load_bits: dict[Edge, float],
    cfg: AnalysisConfig,
) -> float:
    edge_times: list[float] = []
    for edge_key, offered_bits in edge_load_bits.items():
        u, v = edge_key
        edge_data = g.get_edge_data(u, v) or {}
        capacity_bps = _edge_capacity_bits_per_s(edge_data, cfg)
        edge_times.append(offered_bits / capacity_bps)
    return float(max(edge_times) if edge_times else 0.0)


def evaluate_workload(
    g: nx.Graph,
    demands: Iterable[FlowDemand],
    routing_mode: str,
    cfg: AnalysisConfig,
) -> dict[str, float]:
    edge_load_bits: dict[Edge, float] = defaultdict(float)
    source_edge_load_bits: dict[str, dict[Edge, float]] = defaultdict(lambda: defaultdict(float))
    path_cache: dict[tuple[str, str], list[RoutedPath]] = {}

    routed_demand_bits = 0.0
    active_sources: set[str] = set()

    for demand in demands:
        bits = float(demand.bits)
        if bits <= 0:
            continue

        active_sources.add(demand.src)

        pair = (demand.src, demand.dst)
        if pair not in path_cache:
            path_cache[pair] = compute_paths(g, demand.src, demand.dst, routing_mode, cfg)

        paths = [path for path in path_cache[pair] if path.weight > 0]
        if not paths:
            continue

        total_weight = sum(path.weight for path in paths)
        if total_weight <= 0:
            continue

        routed_demand_bits += bits

        for path in paths:
            split_weight = path.weight / total_weight
            split_bits = bits * split_weight
            for u, v in zip(path.nodes[:-1], path.nodes[1:]):
                key = _edge_key(u, v)
                edge_load_bits[key] += split_bits
                source_edge_load_bits[demand.src][key] += split_bits

    completion_time_s = _completion_time_from_edge_loads(g, edge_load_bits, cfg)

    source_completion_times_s: list[float] = []
    for source in active_sources:
        source_completion_times_s.append(
            _completion_time_from_edge_loads(g, source_edge_load_bits[source], cfg)
        )

    if source_completion_times_s:
        completion_time_p50_s = float(np.percentile(source_completion_times_s, 50))
        completion_time_p95_s = float(np.percentile(source_completion_times_s, 95))
    else:
        completion_time_p50_s = 0.0
        completion_time_p95_s = 0.0

    backend_link_utilization: list[float] = []
    for u, v, edge_data in g.edges(data=True):
        if edge_data.get("link_kind") != "backend_interconnect":
            continue

        key = _edge_key(u, v)
        offered_bits = edge_load_bits.get(key, 0.0)
        capacity_bps = _edge_capacity_bits_per_s(edge_data, cfg)

        if completion_time_s > 0:
            utilization = offered_bits / (capacity_bps * completion_time_s)
        else:
            utilization = 0.0

        backend_link_utilization.append(float(utilization))

    max_link_utilization = float(max(backend_link_utilization) if backend_link_utilization else 0.0)

    mean_backend_util = (
        float(statistics.fmean(backend_link_utilization)) if backend_link_utilization else 0.0
    )
    if mean_backend_util > 0:
        link_utilization_cv = float(
            statistics.pstdev(backend_link_utilization) / mean_backend_util
        )
    else:
        link_utilization_cv = 0.0

    active_source_count = len(active_sources)
    if completion_time_s > 0 and active_source_count > 0:
        per_ssu_throughput_gbps = float(
            (routed_demand_bits / float(active_source_count)) / completion_time_s / 1e9
        )
    else:
        per_ssu_throughput_gbps = 0.0

    return {
        "completion_time_s": completion_time_s,
        "completion_time_p50_s": completion_time_p50_s,
        "completion_time_p95_s": completion_time_p95_s,
        "per_ssu_throughput_gbps": per_ssu_throughput_gbps,
        "max_link_utilization": max_link_utilization,
        "link_utilization_cv": link_utilization_cv,
    }


def compute_topology_metrics(g: nx.Graph, cfg: AnalysisConfig) -> dict[str, float]:
    structural = compute_structural_metrics(g)
    workload = evaluate_workload(
        g,
        build_a2a_demands(g, cfg),
        routing_mode=cfg.routing_mode,
        cfg=cfg,
    )
    return {**structural, **workload}
