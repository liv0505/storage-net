from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Iterable

import networkx as nx
import numpy as np

from .config import AnalysisConfig
from .routing import RoutedPath, compute_paths, normalize_routing_mode
from .traffic import FlowDemand, build_a2a_demands


Edge = tuple[object, object]


def _edge_key(u: object, v: object) -> Edge:
    return tuple(sorted((u, v), key=lambda x: str(x)))


def _ssu_nodes(g: nx.Graph) -> list[str]:
    return [str(node_id) for node_id, data in g.nodes(data=True) if data.get("node_role") == "ssu"]


def _exchange_id(node_id: str) -> str:
    return str(node_id).split(":", 1)[0]


def _union_label(union_id: str) -> str:
    return str(union_id).split(":", 1)[1]


def _source_union_ids(g: nx.Graph, src_ssu: str) -> list[str]:
    unions = [
        str(neighbor)
        for neighbor in g.neighbors(src_ssu)
        if g.nodes[neighbor].get("node_role") == "union"
    ]
    return sorted(unions, key=_union_label)


def _infer_direct_topology_kind(g: nx.Graph) -> str | None:
    backend_roles = {
        str(data.get("topology_role"))
        for _, _, data in g.edges(data=True)
        if data.get("link_kind") == "backend_interconnect"
    }
    if {"2d_fullmesh_x", "2d_fullmesh_y"}.issubset(backend_roles):
        return "2D-FULLMESH"
    if {"2d_torus_x", "2d_torus_y"}.issubset(backend_roles):
        return "2D-TORUS"
    if {"3d_torus_x", "3d_torus_y", "3d_torus_z"}.issubset(backend_roles):
        return "3D-TORUS"
    return None


def _build_union_plane_graph(g: nx.Graph, union_label: str) -> nx.Graph:
    plane_nodes = {
        node_id
        for node_id, data in g.nodes(data=True)
        if data.get("node_role") == "union" and _union_label(str(node_id)) == union_label
    }
    plane_graph = nx.Graph()
    plane_graph.add_nodes_from(plane_nodes)

    for u, v, data in g.edges(data=True):
        if data.get("link_kind") != "backend_interconnect":
            continue
        if u in plane_nodes and v in plane_nodes:
            plane_graph.add_edge(u, v, **data)

    return plane_graph


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


def _accumulate_routed_paths(
    edge_load_bits: dict[Edge, float],
    source_edge_load_bits: dict[str, dict[Edge, float]],
    demand: FlowDemand,
    paths: list[RoutedPath],
) -> bool:
    paths = [path for path in paths if path.weight > 0]
    if not paths:
        return False

    total_weight = sum(path.weight for path in paths)
    if total_weight <= 0:
        return False

    bits = float(demand.bits)
    for path in paths:
        split_weight = path.weight / total_weight
        split_bits = bits * split_weight
        for u, v in zip(path.nodes[:-1], path.nodes[1:]):
            key = _edge_key(u, v)
            edge_load_bits[key] += split_bits
            source_edge_load_bits[demand.src][key] += split_bits
    return True


def _should_use_exact_shortest_path_fast_path(g: nx.Graph, routing_mode: str) -> bool:
    if normalize_routing_mode(routing_mode) != "SHORTEST_PATH":
        return False
    return _infer_direct_topology_kind(g) in {"2D-FULLMESH", "2D-TORUS", "3D-TORUS"}


def _should_use_direct_projection_fast_path(g: nx.Graph, routing_mode: str) -> bool:
    if normalize_routing_mode(routing_mode) not in {"DOR", "FULL_PATH"}:
        return False
    return _infer_direct_topology_kind(g) in {"2D-FULLMESH", "2D-TORUS", "3D-TORUS"}


def _shortest_edge_fractions(
    plane_graph: nx.Graph,
    src_union: str,
    dst_union: str,
) -> list[tuple[Edge, float]]:
    src_dist = nx.single_source_shortest_path_length(plane_graph, src_union)
    shortest_distance = src_dist.get(dst_union)
    if shortest_distance is None:
        return []

    dst_dist = nx.single_source_shortest_path_length(plane_graph, dst_union)
    dag_nodes = {
        node
        for node, dist in src_dist.items()
        if dist + dst_dist.get(node, shortest_distance + 1) == shortest_distance
    }

    levels: dict[int, list[str]] = defaultdict(list)
    successors: dict[str, list[str]] = defaultdict(list)
    for node in dag_nodes:
        levels[src_dist[node]].append(node)

    max_level = max(levels) if levels else 0
    for level in range(max_level):
        for node in levels[level]:
            for neighbor in plane_graph.neighbors(node):
                if neighbor not in dag_nodes:
                    continue
                if src_dist.get(neighbor) != level + 1:
                    continue
                successors[node].append(str(neighbor))

    forward_counts: dict[str, int] = {src_union: 1}
    for level in range(max_level):
        for node in levels[level]:
            count = forward_counts.get(str(node), 0)
            if count == 0:
                continue
            for successor in successors.get(str(node), []):
                forward_counts[successor] = forward_counts.get(successor, 0) + count

    total_paths = forward_counts.get(dst_union, 0)
    if total_paths <= 0:
        return []

    backward_counts: dict[str, int] = {dst_union: 1}
    for level in range(max_level - 1, -1, -1):
        for node in levels[level]:
            backward_counts[str(node)] = sum(
                backward_counts.get(successor, 0) for successor in successors.get(str(node), [])
            )

    fractions: list[tuple[Edge, float]] = []
    for level in range(max_level):
        for node in levels[level]:
            node_str = str(node)
            node_count = forward_counts.get(node_str, 0)
            if node_count == 0:
                continue
            for successor in successors.get(node_str, []):
                edge_paths = node_count * backward_counts.get(successor, 0)
                if edge_paths <= 0:
                    continue
                fractions.append((_edge_key(node_str, successor), edge_paths / float(total_paths)))

    return fractions


def _evaluate_shortest_path_workload_exact_fast(
    g: nx.Graph,
    demands: Iterable[FlowDemand],
    cfg: AnalysisConfig,
) -> dict[str, float]:
    edge_load_bits: dict[Edge, float] = defaultdict(float)
    source_edge_load_bits: dict[str, dict[Edge, float]] = defaultdict(lambda: defaultdict(float))
    fallback_path_cache: dict[tuple[str, str], list[RoutedPath]] = {}
    plane_graph_cache: dict[str, nx.Graph] = {}
    fraction_cache: dict[tuple[str, str, str], list[tuple[Edge, float]]] = {}

    routed_demand_bits = 0.0
    active_sources: set[str] = set()

    for demand in demands:
        bits = float(demand.bits)
        if bits <= 0:
            continue

        active_sources.add(demand.src)
        src_exchange = _exchange_id(demand.src)
        dst_exchange = _exchange_id(demand.dst)

        if src_exchange == dst_exchange:
            pair = (demand.src, demand.dst)
            if pair not in fallback_path_cache:
                fallback_path_cache[pair] = compute_paths(g, demand.src, demand.dst, "SHORTEST_PATH", cfg)
            if _accumulate_routed_paths(edge_load_bits, source_edge_load_bits, demand, fallback_path_cache[pair]):
                routed_demand_bits += bits
            continue

        plane_payloads: list[tuple[str, str, list[tuple[Edge, float]]]] = []
        for source_union in _source_union_ids(g, demand.src):
            union_label = _union_label(source_union)
            dst_union = f"{dst_exchange}:{union_label}"
            cache_key = (union_label, src_exchange, dst_exchange)
            if cache_key not in fraction_cache:
                plane_graph = plane_graph_cache.setdefault(union_label, _build_union_plane_graph(g, union_label))
                fraction_cache[cache_key] = _shortest_edge_fractions(plane_graph, source_union, dst_union)
            fractions = fraction_cache[cache_key]
            if fractions:
                plane_payloads.append((source_union, dst_union, fractions))

        if not plane_payloads:
            continue

        routed_demand_bits += bits
        plane_bits = bits / float(len(plane_payloads))

        for source_union, dst_union, fractions in plane_payloads:
            src_internal = _edge_key(demand.src, source_union)
            edge_load_bits[src_internal] += plane_bits
            source_edge_load_bits[demand.src][src_internal] += plane_bits

            dst_internal = _edge_key(dst_union, demand.dst)
            edge_load_bits[dst_internal] += plane_bits
            source_edge_load_bits[demand.src][dst_internal] += plane_bits

            for edge_key, fraction in fractions:
                split_bits = plane_bits * fraction
                edge_load_bits[edge_key] += split_bits
                source_edge_load_bits[demand.src][edge_key] += split_bits

    return _finalize_workload_metrics(g, edge_load_bits, source_edge_load_bits, routed_demand_bits, active_sources, cfg)


def _projection_from_paths(paths: list[RoutedPath]) -> tuple[dict[str, float], dict[str, float], dict[Edge, float]]:
    usable_paths = [path for path in paths if path.weight > 0]
    if not usable_paths:
        return {}, {}, {}

    total_weight = sum(path.weight for path in usable_paths)
    if total_weight <= 0:
        return {}, {}, {}

    source_union_fractions: dict[str, float] = defaultdict(float)
    destination_union_fractions: dict[str, float] = defaultdict(float)
    backend_edge_fractions: dict[Edge, float] = defaultdict(float)

    for path in usable_paths:
        normalized_weight = path.weight / total_weight
        if len(path.nodes) < 4:
            continue
        source_union = str(path.nodes[1])
        destination_union = str(path.nodes[-2])
        source_union_fractions[source_union] += normalized_weight
        destination_union_fractions[destination_union] += normalized_weight

        for u, v in zip(path.nodes[1:-2], path.nodes[2:-1]):
            backend_edge_fractions[_edge_key(u, v)] += normalized_weight

    return dict(source_union_fractions), dict(destination_union_fractions), dict(backend_edge_fractions)


def _evaluate_direct_workload_projection_fast(
    g: nx.Graph,
    demands: Iterable[FlowDemand],
    routing_mode: str,
    cfg: AnalysisConfig,
) -> dict[str, float]:
    edge_load_bits: dict[Edge, float] = defaultdict(float)
    source_edge_load_bits: dict[str, dict[Edge, float]] = defaultdict(lambda: defaultdict(float))
    same_exchange_path_cache: dict[tuple[str, str], list[RoutedPath]] = {}
    projection_cache: dict[tuple[str, str, str], tuple[dict[str, float], dict[str, float], dict[Edge, float]]] = {}

    normalized_mode = normalize_routing_mode(routing_mode)
    routed_demand_bits = 0.0
    active_sources: set[str] = set()

    for demand in demands:
        bits = float(demand.bits)
        if bits <= 0:
            continue

        active_sources.add(demand.src)
        src_exchange = _exchange_id(demand.src)
        dst_exchange = _exchange_id(demand.dst)

        if src_exchange == dst_exchange:
            pair = (demand.src, demand.dst)
            if pair not in same_exchange_path_cache:
                same_exchange_path_cache[pair] = compute_paths(g, demand.src, demand.dst, normalized_mode, cfg)
            if _accumulate_routed_paths(edge_load_bits, source_edge_load_bits, demand, same_exchange_path_cache[pair]):
                routed_demand_bits += bits
            continue

        cache_key = (src_exchange, dst_exchange, normalized_mode)
        if cache_key not in projection_cache:
            representative_src = f"{src_exchange}:ssu0"
            representative_dst = f"{dst_exchange}:ssu0"
            projection_cache[cache_key] = _projection_from_paths(
                compute_paths(g, representative_src, representative_dst, normalized_mode, cfg)
            )

        source_union_fractions, destination_union_fractions, backend_edge_fractions = projection_cache[cache_key]
        if not source_union_fractions and not destination_union_fractions and not backend_edge_fractions:
            continue

        routed_demand_bits += bits

        for source_union, fraction in source_union_fractions.items():
            split_bits = bits * fraction
            key = _edge_key(demand.src, source_union)
            edge_load_bits[key] += split_bits
            source_edge_load_bits[demand.src][key] += split_bits

        for destination_union, fraction in destination_union_fractions.items():
            split_bits = bits * fraction
            key = _edge_key(destination_union, demand.dst)
            edge_load_bits[key] += split_bits
            source_edge_load_bits[demand.src][key] += split_bits

        for edge_key, fraction in backend_edge_fractions.items():
            split_bits = bits * fraction
            edge_load_bits[edge_key] += split_bits
            source_edge_load_bits[demand.src][edge_key] += split_bits

    return _finalize_workload_metrics(g, edge_load_bits, source_edge_load_bits, routed_demand_bits, active_sources, cfg)


def _finalize_workload_metrics(
    g: nx.Graph,
    edge_load_bits: dict[Edge, float],
    source_edge_load_bits: dict[str, dict[Edge, float]],
    routed_demand_bits: float,
    active_sources: set[str],
    cfg: AnalysisConfig,
) -> dict[str, float]:
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


def _evaluate_workload_via_explicit_paths(
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

        if _accumulate_routed_paths(edge_load_bits, source_edge_load_bits, demand, path_cache[pair]):
            routed_demand_bits += bits

    return _finalize_workload_metrics(g, edge_load_bits, source_edge_load_bits, routed_demand_bits, active_sources, cfg)


def evaluate_workload(
    g: nx.Graph,
    demands: Iterable[FlowDemand],
    routing_mode: str,
    cfg: AnalysisConfig,
) -> dict[str, float]:
    if _should_use_exact_shortest_path_fast_path(g, routing_mode):
        return _evaluate_shortest_path_workload_exact_fast(g, demands, cfg)
    if _should_use_direct_projection_fast_path(g, routing_mode):
        return _evaluate_direct_workload_projection_fast(g, demands, routing_mode, cfg)
    return _evaluate_workload_via_explicit_paths(g, demands, routing_mode, cfg)


def compute_topology_metrics(g: nx.Graph, cfg: AnalysisConfig) -> dict[str, float]:
    structural = compute_structural_metrics(g)
    workload = evaluate_workload(
        g,
        build_a2a_demands(g, cfg),
        routing_mode=cfg.routing_mode,
        cfg=cfg,
    )
    return {**structural, **workload}
