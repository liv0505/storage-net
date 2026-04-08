from __future__ import annotations

import statistics
from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable

import networkx as nx
import numpy as np

from .config import AnalysisConfig
from .routing import RoutedPath, compute_paths, normalize_routing_mode
from .traffic import FlowDemand, build_a2a_demands


Edge = tuple[object, object]
WorkloadDetails = dict[str, Any]
_VOLUME_BUCKET_DECIMALS = 4


def _edge_key(u: object, v: object) -> Edge:
    return (u, v)


def _ssu_nodes(g: nx.Graph) -> list[str]:
    return [str(node_id) for node_id, data in g.nodes(data=True) if data.get("node_role") == "ssu"]


def _ssu_components(g: nx.Graph) -> list[list[str]]:
    components: list[list[str]] = []
    for component_nodes in nx.connected_components(g):
        component_ssus = sorted(
            [
                str(node_id)
                for node_id in component_nodes
                if g.nodes[node_id].get("node_role") == "ssu"
            ]
        )
        if component_ssus:
            components.append(component_ssus)
    return components


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


def _is_df_topology(g: nx.Graph) -> bool:
    family = str(g.graph.get("topology_family", "")).upper()
    if family == "DF":
        return True
    topology_name = str(g.graph.get("topology_name", "")).upper()
    return topology_name == "DF" or topology_name.startswith("DF-")


def _server_id(g: nx.Graph, node_id: str) -> int | None:
    value = g.nodes[node_id].get("server_id")
    if value is None:
        return None
    return int(value)


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


def _exchange_ids(g: nx.Graph) -> list[str]:
    return sorted(
        {
            str(data.get("exchange_node_id"))
            for _, data in g.nodes(data=True)
            if data.get("exchange_node_id") is not None
        },
        key=lambda value: int(value.removeprefix("en")),
    )


def _torus_side_length(exchange_count: int, dimensions: int) -> int:
    if dimensions == 2:
        size = int(round(exchange_count ** 0.5))
    elif dimensions == 3:
        size = int(round(exchange_count ** (1 / 3)))
    else:
        raise ValueError(f"Unsupported torus dimension count: {dimensions}")

    if size <= 0 or size**dimensions != exchange_count:
        raise ValueError(
            f"Exchange count {exchange_count} does not form a {dimensions}D torus grid"
        )
    return size


def _exchange_to_coord_3d(exchange_id: str, size: int) -> tuple[int, int, int]:
    idx = int(exchange_id.removeprefix("en"))
    x = idx // (size * size)
    rem = idx % (size * size)
    y = rem // size
    z = rem % size
    return x, y, z


def _single_plane_torus_candidate_exchange_partitions(
    g: nx.Graph,
    exchange_ids: list[str],
) -> list[frozenset[str]]:
    grid_shape = g.graph.get("torus_exchange_grid_shape")
    if not grid_shape:
        return []

    normalized_shape = tuple(int(value) for value in grid_shape)
    exchange_coords: dict[str, tuple[int, ...]] = {}
    for exchange_id in exchange_ids:
        coord = (g.nodes.get(f"{exchange_id}:union0") or {}).get("exchange_grid_coord")
        if coord is None:
            return []
        exchange_coords[exchange_id] = tuple(int(value) for value in coord)

    half = len(exchange_ids) // 2
    partitions: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for axis, axis_size in enumerate(normalized_shape):
        if axis_size <= 1 or axis_size % 2 != 0:
            continue
        midpoint = axis_size // 2
        side_a = frozenset(
            exchange_id
            for exchange_id, coord in exchange_coords.items()
            if coord[axis] < midpoint
        )
        if len(side_a) == half and side_a not in seen:
            seen.add(side_a)
            partitions.append(side_a)

    return partitions


def _candidate_exchange_partitions(g: nx.Graph) -> list[frozenset[str]]:
    exchange_ids = _exchange_ids(g)
    exchange_count = len(exchange_ids)
    if exchange_count == 0 or exchange_count % 2 != 0:
        return []

    topology_kind = _infer_direct_topology_kind(g)
    half = exchange_count // 2

    if topology_kind in {"2D-TORUS", "3D-TORUS"} and str(g.graph.get("direct_backend_mode", "")) == "single_plane":
        partitions = _single_plane_torus_candidate_exchange_partitions(g, exchange_ids)
        if partitions:
            return partitions

    if topology_kind == "3D-TORUS" and exchange_count > 18:
        size = _torus_side_length(exchange_count, dimensions=3)
        midpoint = size // 2
        partitions: list[frozenset[str]] = []
        for axis in range(3):
            side_a = {
                exchange_id
                for exchange_id in exchange_ids
                if _exchange_to_coord_3d(exchange_id, size)[axis] < midpoint
            }
            partitions.append(frozenset(side_a))
        return partitions

    if _is_df_topology(g):
        partitions = _df_candidate_exchange_partitions(g, exchange_ids, half)
        if partitions:
            return partitions

    if exchange_count > 18:
        return _sliding_window_exchange_partitions(exchange_ids, half)

    fixed = exchange_ids[0]
    return [
        frozenset((fixed, *combo))
        for combo in combinations(exchange_ids[1:], half - 1)
    ]


def _df_candidate_exchange_partitions(
    g: nx.Graph,
    exchange_ids: list[str],
    half: int,
) -> list[frozenset[str]]:
    exchanges_by_server: dict[int, list[str]] = defaultdict(list)
    for exchange_id in exchange_ids:
        server = _server_id(g, f"{exchange_id}:union0")
        if server is None:
            return []
        exchanges_by_server[server].append(exchange_id)

    ordered_servers = sorted(exchanges_by_server)
    if not ordered_servers:
        return []

    ordered_groups = [sorted(exchanges_by_server[server], key=lambda value: int(value.removeprefix("en"))) for server in ordered_servers]
    partitions: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()

    for start in range(len(ordered_groups)):
        selected: list[str] = []
        cursor = start
        while len(selected) < half:
            group = ordered_groups[cursor % len(ordered_groups)]
            remaining = half - len(selected)
            selected.extend(group[:remaining])
            cursor += 1
        partition = frozenset(selected)
        if len(partition) == half and partition not in seen:
            seen.add(partition)
            partitions.append(partition)

    return partitions


def _sliding_window_exchange_partitions(
    exchange_ids: list[str],
    half: int,
) -> list[frozenset[str]]:
    if not exchange_ids or half <= 0:
        return []

    circular = exchange_ids + exchange_ids[: half - 1]
    seen: set[frozenset[str]] = set()
    partitions: list[frozenset[str]] = []
    for start in range(len(exchange_ids)):
        partition = frozenset(circular[start : start + half])
        if len(partition) == half and partition not in seen:
            seen.add(partition)
            partitions.append(partition)
    return partitions


def _backend_component_subgraphs(g: nx.Graph) -> list[nx.Graph]:
    backend_graph = nx.Graph()
    for u, v, data in g.edges(data=True):
        if data.get("link_kind") != "backend_interconnect":
            continue
        backend_graph.add_edge(str(u), str(v), **data)

    if backend_graph.number_of_edges() == 0:
        return []

    return [g.subgraph({str(node_id) for node_id in component_nodes}).copy() for component_nodes in nx.connected_components(backend_graph)]


def _balanced_partition_candidates(items: list[object]) -> list[frozenset[object]]:
    if len(items) < 2:
        return []

    ordered = list(items)
    side_a_size = len(ordered) // 2
    if side_a_size <= 0:
        return []

    if len(ordered) % 2 == 0:
        fixed = ordered[0]
        return [
            frozenset((fixed, *combo))
            for combo in combinations(ordered[1:], side_a_size - 1)
        ]
    return [frozenset(combo) for combo in combinations(ordered, side_a_size)]


def _df_local_group_graph(g: nx.Graph) -> nx.Graph:
    group_graph = nx.Graph()
    for node_id, data in g.nodes(data=True):
        if data.get("node_role") != "union":
            continue
        server_id = data.get("server_id")
        if server_id is not None:
            group_graph.add_node(int(server_id))

    for u, v, data in g.edges(data=True):
        if data.get("link_kind") != "backend_interconnect":
            continue
        server_u = g.nodes[u].get("server_id")
        server_v = g.nodes[v].get("server_id")
        if server_u is None or server_v is None or int(server_u) == int(server_v):
            continue
        left = int(server_u)
        right = int(server_v)
        bandwidth = float(data.get("bandwidth_gbps", 0.0))
        if group_graph.has_edge(left, right):
            group_graph[left][right]["bandwidth_gbps"] += bandwidth
        else:
            group_graph.add_edge(left, right, bandwidth_gbps=bandwidth)
    return group_graph


def _df_group_level_bisection_bandwidth_component_gbps(g: nx.Graph) -> float:
    group_graph = _df_local_group_graph(g)
    group_ids = sorted(group_graph.nodes())
    candidate_partitions = _balanced_partition_candidates(group_ids)
    if not candidate_partitions:
        return 0.0

    best_cut = float("inf")
    for side_a in candidate_partitions:
        cut_value = 0.0
        for left, right, data in group_graph.edges(data=True):
            if (left in side_a) != (right in side_a):
                cut_value += float(data.get("bandwidth_gbps", 0.0))
        best_cut = min(best_cut, cut_value)

    return 0.0 if best_cut == float("inf") else float(best_cut)


def _backend_only_balanced_bisection_bandwidth_component_gbps(g: nx.Graph) -> float:
    if _is_df_topology(g):
        return _df_group_level_bisection_bandwidth_component_gbps(g)

    candidate_partitions = _candidate_exchange_partitions(g)
    if not candidate_partitions:
        return 0.0

    best_cut = float("inf")
    for side_a in candidate_partitions:
        cut_value = 0.0
        neutral_loads: dict[str, dict[str, float]] = defaultdict(lambda: {"A": 0.0, "B": 0.0})

        for u, v, data in g.edges(data=True):
            if data.get("link_kind") != "backend_interconnect":
                continue

            bandwidth = float(data.get("bandwidth_gbps", 0.0))
            exchange_u = g.nodes[u].get("exchange_node_id")
            exchange_v = g.nodes[v].get("exchange_node_id")

            if exchange_u is not None and exchange_v is not None:
                if (exchange_u in side_a) != (exchange_v in side_a):
                    cut_value += bandwidth
                continue

            if exchange_u is None and exchange_v is None:
                continue

            neutral_node = str(u if exchange_u is None else v)
            fixed_exchange = str(exchange_v if exchange_u is None else exchange_u)
            fixed_side = "A" if fixed_exchange in side_a else "B"
            neutral_loads[neutral_node][fixed_side] += bandwidth

        cut_value += sum(min(loads["A"], loads["B"]) for loads in neutral_loads.values())
        best_cut = min(best_cut, cut_value)

    return 0.0 if best_cut == float("inf") else float(best_cut)


def _backend_only_balanced_bisection_bandwidth_gbps(g: nx.Graph) -> float:
    component_subgraphs = _backend_component_subgraphs(g)
    if not component_subgraphs:
        return 0.0
    return float(
        sum(_backend_only_balanced_bisection_bandwidth_component_gbps(subgraph) for subgraph in component_subgraphs)
    )


def compute_structural_metrics(g: nx.Graph) -> dict[str, float]:
    ssus = _ssu_nodes(g)
    total_ssu_count = len(ssus)
    bisection_bandwidth_gbps = _backend_only_balanced_bisection_bandwidth_gbps(g)
    bisection_bandwidth_gbps_per_ssu = (
        (2.0 * bisection_bandwidth_gbps / float(total_ssu_count)) if total_ssu_count > 0 else 0.0
    )
    if total_ssu_count < 2:
        return {
            "diameter": 0.0,
            "average_hops": 0.0,
            "bisection_bandwidth_gbps": bisection_bandwidth_gbps,
            "bisection_bandwidth_gbps_per_ssu": bisection_bandwidth_gbps_per_ssu,
        }

    pair_hops: list[float] = []
    for component_ssus in _ssu_components(g):
        if len(component_ssus) < 2:
            continue
        for idx, src in enumerate(component_ssus):
            lengths = nx.single_source_shortest_path_length(g, src)
            for dst in component_ssus[idx + 1 :]:
                hop_count = lengths.get(dst)
                if hop_count is None:
                    continue
                pair_hops.append(float(hop_count))

    if pair_hops:
        average_hops = float(statistics.fmean(pair_hops))
        diameter = float(max(pair_hops))
    else:
        average_hops = 0.0
        diameter = 0.0

    return {
        "diameter": diameter,
        "average_hops": average_hops,
        "bisection_bandwidth_gbps": bisection_bandwidth_gbps,
        "bisection_bandwidth_gbps_per_ssu": bisection_bandwidth_gbps_per_ssu,
    }


def _edge_capacity_bits_per_s(edge_data: dict[str, float], cfg: AnalysisConfig) -> float:
    bandwidth_gbps = float(edge_data.get("bandwidth_gbps", cfg.link_bandwidth_gbps))
    return max(bandwidth_gbps * 1e9, 1.0)


def _bits_to_gb(bits: float) -> float:
    return float(bits) / 8e9


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


def _expected_hops_from_paths(paths: list[RoutedPath]) -> float:
    usable_paths = [path for path in paths if path.weight > 0]
    if not usable_paths:
        return 0.0

    total_weight = sum(path.weight for path in usable_paths)
    if total_weight <= 0:
        return 0.0

    return float(
        sum(path.hops * (path.weight / total_weight) for path in usable_paths)
    )


def _accumulate_routed_paths(
    edge_load_bits: dict[Edge, float],
    source_edge_load_bits: dict[str, dict[Edge, float]],
    hop_load_bits: dict[int, float],
    demand: FlowDemand,
    paths: list[RoutedPath],
) -> tuple[bool, float]:
    paths = [path for path in paths if path.weight > 0]
    if not paths:
        return False, 0.0

    total_weight = sum(path.weight for path in paths)
    if total_weight <= 0:
        return False, 0.0

    bits = float(demand.bits)
    expected_hops = 0.0
    for path in paths:
        split_weight = path.weight / total_weight
        expected_hops += path.hops * split_weight
        split_bits = bits * split_weight
        hop_load_bits[int(path.hops)] += split_bits
        for u, v in zip(path.nodes[:-1], path.nodes[1:]):
            key = _edge_key(u, v)
            edge_load_bits[key] += split_bits
            source_edge_load_bits[demand.src][key] += split_bits
    return True, float(expected_hops)


def _should_use_exact_shortest_path_fast_path(g: nx.Graph, routing_mode: str) -> bool:
    if normalize_routing_mode(routing_mode) != "SHORTEST_PATH":
        return False
    return _infer_direct_topology_kind(g) == "2D-FULLMESH"


def _should_use_direct_projection_fast_path(g: nx.Graph, routing_mode: str) -> bool:
    mode = normalize_routing_mode(routing_mode)
    if mode in {"DOR", "FULL_PATH"}:
        return _infer_direct_topology_kind(g) in {"2D-FULLMESH", "2D-TORUS", "3D-TORUS"}
    if mode == "SHORTEST_PATH" and _is_single_plane_direct_torus(g):
        return True
    return False


def _is_single_plane_direct_torus(g: nx.Graph) -> bool:
    topology_kind = _infer_direct_topology_kind(g)
    if topology_kind not in {"2D-TORUS", "3D-TORUS"}:
        return False
    return str(g.graph.get("direct_backend_mode", "dual_plane")) == "single_plane"


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
) -> WorkloadDetails:
    edge_load_bits: dict[Edge, float] = defaultdict(float)
    source_edge_load_bits: dict[str, dict[Edge, float]] = defaultdict(lambda: defaultdict(float))
    hop_load_bits: dict[int, float] = defaultdict(float)
    fallback_path_cache: dict[tuple[str, str], list[RoutedPath]] = {}
    plane_graph_cache: dict[str, nx.Graph] = {}
    fraction_cache: dict[tuple[str, str, str], list[tuple[Edge, float]]] = {}

    routed_demand_bits = 0.0
    routed_hop_weighted_sum = 0.0
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
            routed, expected_hops = _accumulate_routed_paths(
                edge_load_bits,
                source_edge_load_bits,
                hop_load_bits,
                demand,
                fallback_path_cache[pair],
            )
            if routed:
                routed_demand_bits += bits
                routed_hop_weighted_sum += bits * expected_hops
            continue

        plane_payloads: list[tuple[str, str, list[tuple[Edge, float]], float]] = []
        for source_union in _source_union_ids(g, demand.src):
            union_label = _union_label(source_union)
            dst_union = f"{dst_exchange}:{union_label}"
            cache_key = (union_label, src_exchange, dst_exchange)
            if cache_key not in fraction_cache:
                plane_graph = plane_graph_cache.setdefault(union_label, _build_union_plane_graph(g, union_label))
                fraction_cache[cache_key] = _shortest_edge_fractions(plane_graph, source_union, dst_union)
            fractions = fraction_cache[cache_key]
            if fractions:
                plane_graph = plane_graph_cache.setdefault(union_label, _build_union_plane_graph(g, union_label))
                backend_hops = nx.shortest_path_length(plane_graph, source_union, dst_union)
                plane_payloads.append((source_union, dst_union, fractions, float(backend_hops + 2)))

        if not plane_payloads:
            continue

        routed_demand_bits += bits
        routed_hop_weighted_sum += bits * (
            sum(payload[3] for payload in plane_payloads) / float(len(plane_payloads))
        )
        plane_bits = bits / float(len(plane_payloads))

        for source_union, dst_union, fractions, total_hops_value in plane_payloads:
            total_hops = int(round(total_hops_value))
            hop_load_bits[total_hops] += plane_bits
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

    return _finalize_workload_details(
        g,
        edge_load_bits,
        source_edge_load_bits,
        hop_load_bits,
        routed_demand_bits,
        routed_hop_weighted_sum,
        active_sources,
        cfg,
    )


def _projection_from_paths(
    paths: list[RoutedPath],
) -> tuple[dict[str, float], dict[str, float], dict[Edge, float], dict[int, float], float]:
    usable_paths = [path for path in paths if path.weight > 0]
    if not usable_paths:
        return {}, {}, {}, {}, 0.0

    total_weight = sum(path.weight for path in usable_paths)
    if total_weight <= 0:
        return {}, {}, {}, {}, 0.0

    source_union_fractions: dict[str, float] = defaultdict(float)
    destination_union_fractions: dict[str, float] = defaultdict(float)
    backend_edge_fractions: dict[Edge, float] = defaultdict(float)
    hop_fractions: dict[int, float] = defaultdict(float)
    expected_hops = 0.0

    for path in usable_paths:
        normalized_weight = path.weight / total_weight
        expected_hops += path.hops * normalized_weight
        hop_fractions[int(path.hops)] += normalized_weight
        if len(path.nodes) < 4:
            continue
        source_union = str(path.nodes[1])
        destination_union = str(path.nodes[-2])
        source_union_fractions[source_union] += normalized_weight
        destination_union_fractions[destination_union] += normalized_weight

        for u, v in zip(path.nodes[1:-2], path.nodes[2:-1]):
            backend_edge_fractions[_edge_key(u, v)] += normalized_weight

    return (
        dict(source_union_fractions),
        dict(destination_union_fractions),
        dict(backend_edge_fractions),
        dict(hop_fractions),
        float(expected_hops),
    )


def _evaluate_direct_workload_projection_fast(
    g: nx.Graph,
    demands: Iterable[FlowDemand],
    routing_mode: str,
    cfg: AnalysisConfig,
) -> WorkloadDetails:
    edge_load_bits: dict[Edge, float] = defaultdict(float)
    source_edge_load_bits: dict[str, dict[Edge, float]] = defaultdict(lambda: defaultdict(float))
    hop_load_bits: dict[int, float] = defaultdict(float)
    same_exchange_path_cache: dict[tuple[str, str], list[RoutedPath]] = {}
    projection_cache: dict[
        tuple[str, str, str],
        tuple[dict[str, float], dict[str, float], dict[Edge, float], dict[int, float], float],
    ] = {}

    normalized_mode = normalize_routing_mode(routing_mode)
    routed_demand_bits = 0.0
    routed_hop_weighted_sum = 0.0
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
            routed, expected_hops = _accumulate_routed_paths(
                edge_load_bits,
                source_edge_load_bits,
                hop_load_bits,
                demand,
                same_exchange_path_cache[pair],
            )
            if routed:
                routed_demand_bits += bits
                routed_hop_weighted_sum += bits * expected_hops
            continue

        cache_key = (src_exchange, dst_exchange, normalized_mode)
        if cache_key not in projection_cache:
            representative_src = f"{src_exchange}:ssu0"
            representative_dst = f"{dst_exchange}:ssu0"
            projection_cache[cache_key] = _projection_from_paths(
                compute_paths(g, representative_src, representative_dst, normalized_mode, cfg)
            )

        (
            source_union_fractions,
            destination_union_fractions,
            backend_edge_fractions,
            hop_fractions,
            expected_hops,
        ) = projection_cache[cache_key]
        if (
            not source_union_fractions
            and not destination_union_fractions
            and not backend_edge_fractions
            and not hop_fractions
        ):
            continue

        routed_demand_bits += bits
        routed_hop_weighted_sum += bits * expected_hops
        for hop_count, fraction in hop_fractions.items():
            hop_load_bits[int(hop_count)] += bits * fraction

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

    return _finalize_workload_details(
        g,
        edge_load_bits,
        source_edge_load_bits,
        hop_load_bits,
        routed_demand_bits,
        routed_hop_weighted_sum,
        active_sources,
        cfg,
    )


def _hop_volume_distribution_rows(
    hop_load_bits: dict[int, float],
    routed_demand_bits: float,
) -> list[dict[str, float]]:
    if routed_demand_bits <= 0:
        return []
    return [
        {
            "hop_count": int(hop_count),
            "offered_volume_gb": _bits_to_gb(offered_bits),
            "offered_volume_pct": (float(offered_bits) / float(routed_demand_bits)) * 100.0,
        }
        for hop_count, offered_bits in sorted(hop_load_bits.items())
        if offered_bits > 0
    ]


def _link_volume_distribution_rows(
    edge_load_bits: dict[Edge, float],
) -> list[dict[str, float]]:
    positive_loads = [float(offered_bits) for offered_bits in edge_load_bits.values() if offered_bits > 0]
    if not positive_loads:
        return []

    volume_buckets: dict[float, int] = defaultdict(int)
    for offered_bits in positive_loads:
        volume_bucket = round(_bits_to_gb(offered_bits), _VOLUME_BUCKET_DECIMALS)
        volume_buckets[volume_bucket] += 1

    total_links = len(positive_loads)
    return [
        {
            "offered_volume_gb": float(volume_gb),
            "link_count": int(link_count),
            "link_ratio_pct": (float(link_count) / float(total_links)) * 100.0,
        }
        for volume_gb, link_count in sorted(volume_buckets.items())
    ]


def _finalize_workload_details(
    g: nx.Graph,
    edge_load_bits: dict[Edge, float],
    source_edge_load_bits: dict[str, dict[Edge, float]],
    hop_load_bits: dict[int, float],
    routed_demand_bits: float,
    routed_hop_weighted_sum: float,
    active_sources: set[str],
    cfg: AnalysisConfig,
) -> WorkloadDetails:
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

        capacity_bps = _edge_capacity_bits_per_s(edge_data, cfg)
        for src_node, dst_node in ((u, v), (v, u)):
            offered_bits = edge_load_bits.get(_edge_key(src_node, dst_node), 0.0)
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

    average_hops = (
        float(routed_hop_weighted_sum / routed_demand_bits) if routed_demand_bits > 0 else 0.0
    )

    metrics = {
        "completion_time_s": completion_time_s,
        "completion_time_p50_s": completion_time_p50_s,
        "completion_time_p95_s": completion_time_p95_s,
        "per_ssu_throughput_gbps": per_ssu_throughput_gbps,
        "average_hops": average_hops,
        "max_link_utilization": max_link_utilization,
        "link_utilization_cv": link_utilization_cv,
    }
    return {
        "metrics": metrics,
        "edge_load_bits": {
            edge_key: float(offered_bits)
            for edge_key, offered_bits in edge_load_bits.items()
            if offered_bits > 0
        },
        "hop_volume_distribution": _hop_volume_distribution_rows(hop_load_bits, routed_demand_bits),
        "link_volume_distribution": _link_volume_distribution_rows(edge_load_bits),
    }


def _evaluate_workload_via_explicit_paths(
    g: nx.Graph,
    demands: Iterable[FlowDemand],
    routing_mode: str,
    cfg: AnalysisConfig,
) -> WorkloadDetails:
    edge_load_bits: dict[Edge, float] = defaultdict(float)
    source_edge_load_bits: dict[str, dict[Edge, float]] = defaultdict(lambda: defaultdict(float))
    hop_load_bits: dict[int, float] = defaultdict(float)
    path_cache: dict[tuple[str, str], list[RoutedPath]] = {}

    routed_demand_bits = 0.0
    routed_hop_weighted_sum = 0.0
    active_sources: set[str] = set()

    for demand in demands:
        bits = float(demand.bits)
        if bits <= 0:
            continue

        active_sources.add(demand.src)

        pair = (demand.src, demand.dst)
        if pair not in path_cache:
            path_cache[pair] = compute_paths(g, demand.src, demand.dst, routing_mode, cfg)

        routed, expected_hops = _accumulate_routed_paths(
            edge_load_bits,
            source_edge_load_bits,
            hop_load_bits,
            demand,
            path_cache[pair],
        )
        if routed:
            routed_demand_bits += bits
            routed_hop_weighted_sum += bits * expected_hops

    return _finalize_workload_details(
        g,
        edge_load_bits,
        source_edge_load_bits,
        hop_load_bits,
        routed_demand_bits,
        routed_hop_weighted_sum,
        active_sources,
        cfg,
    )


def evaluate_workload_with_details(
    g: nx.Graph,
    demands: Iterable[FlowDemand],
    routing_mode: str,
    cfg: AnalysisConfig,
) -> WorkloadDetails:
    if _should_use_exact_shortest_path_fast_path(g, routing_mode):
        return _evaluate_shortest_path_workload_exact_fast(g, demands, cfg)
    if _should_use_direct_projection_fast_path(g, routing_mode):
        return _evaluate_direct_workload_projection_fast(g, demands, routing_mode, cfg)
    return _evaluate_workload_via_explicit_paths(g, demands, routing_mode, cfg)


def evaluate_workload(
    g: nx.Graph,
    demands: Iterable[FlowDemand],
    routing_mode: str,
    cfg: AnalysisConfig,
) -> dict[str, float]:
    return dict(
        evaluate_workload_with_details(
            g,
            demands,
            routing_mode=routing_mode,
            cfg=cfg,
        )["metrics"]
    )


def compute_topology_metrics(g: nx.Graph, cfg: AnalysisConfig) -> dict[str, float]:
    structural = compute_structural_metrics(g)
    workload = evaluate_workload(
        g,
        build_a2a_demands(g, cfg),
        routing_mode=cfg.routing_mode,
        cfg=cfg,
    )
    return {**structural, **workload}
