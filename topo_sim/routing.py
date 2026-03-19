from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from .config import AnalysisConfig


_SUPPORTED_ROUTING_MODES = {"DOR", "PORT_BALANCED", "ECMP", "MIN_HOPS"}


@dataclass(slots=True)
class RoutedPath:
    nodes: tuple[str, ...]
    weight: float

    @property
    def hops(self) -> int:
        return max(0, len(self.nodes) - 1)


def compute_paths(
    g: nx.Graph,
    src_ssu: str,
    dst_ssu: str,
    routing_mode: str,
    cfg: AnalysisConfig,
) -> list[RoutedPath]:
    if not isinstance(routing_mode, str):
        raise ValueError("routing_mode must be a string")

    mode = routing_mode.strip().upper()
    if mode not in _SUPPORTED_ROUTING_MODES:
        raise ValueError(f"Unsupported routing mode: {routing_mode}")

    if src_ssu not in g or dst_ssu not in g:
        return []

    if src_ssu == dst_ssu:
        return [RoutedPath(nodes=(src_ssu,), weight=1.0)]

    if _exchange_id(src_ssu) == _exchange_id(dst_ssu):
        return _compute_same_exchange_internal_paths(g, src_ssu, dst_ssu, mode)

    if not nx.has_path(g, src_ssu, dst_ssu):
        return []

    try:
        if mode == "DOR":
            return _compute_dor_paths(g, src_ssu, dst_ssu)
        if mode == "PORT_BALANCED":
            return _compute_port_balanced_paths(g, src_ssu, dst_ssu, cfg)
        if mode == "ECMP":
            return _compute_ecmp_paths(g, src_ssu, dst_ssu)
        return _compute_min_hops_path(g, src_ssu, dst_ssu)
    except nx.NetworkXNoPath:
        return []


def _compute_same_exchange_internal_paths(
    g: nx.Graph,
    src_ssu: str,
    dst_ssu: str,
    mode: str,
) -> list[RoutedPath]:
    exchange_id = _exchange_id(src_ssu)
    internal_graph = _build_exchange_internal_graph(g, exchange_id)

    if src_ssu not in internal_graph or dst_ssu not in internal_graph:
        return []
    if not nx.has_path(internal_graph, src_ssu, dst_ssu):
        return []

    if mode == "ECMP":
        return _compute_ecmp_paths(internal_graph, src_ssu, dst_ssu)
    return _compute_min_hops_path(internal_graph, src_ssu, dst_ssu)


def _build_exchange_internal_graph(g: nx.Graph, exchange_id: str) -> nx.Graph:
    exchange_nodes = {
        node_id
        for node_id, data in g.nodes(data=True)
        if data.get("exchange_node_id") == exchange_id
    }
    internal_graph = g.subgraph(exchange_nodes).copy()

    for u, v, data in list(internal_graph.edges(data=True)):
        if data.get("link_kind") != "internal_ssu_uplink":
            internal_graph.remove_edge(u, v)

    return internal_graph


def _compute_min_hops_path(g: nx.Graph, src_ssu: str, dst_ssu: str) -> list[RoutedPath]:
    if src_ssu == dst_ssu:
        return [RoutedPath(nodes=(src_ssu,), weight=1.0)]
    nodes = tuple(nx.shortest_path(g, src_ssu, dst_ssu))
    return [RoutedPath(nodes=nodes, weight=1.0)]


def _compute_ecmp_paths(g: nx.Graph, src_ssu: str, dst_ssu: str) -> list[RoutedPath]:
    if src_ssu == dst_ssu:
        return [RoutedPath(nodes=(src_ssu,), weight=1.0)]

    shortest_paths = [tuple(path) for path in nx.all_shortest_paths(g, src_ssu, dst_ssu)]
    if not shortest_paths:
        return []

    weight = 1.0 / len(shortest_paths)
    return [RoutedPath(nodes=path, weight=weight) for path in shortest_paths]


def _compute_dor_paths(g: nx.Graph, src_ssu: str, dst_ssu: str) -> list[RoutedPath]:
    if src_ssu == dst_ssu:
        return [RoutedPath(nodes=(src_ssu,), weight=1.0)]

    backend_roles = {
        str(data.get("topology_role"))
        for _, _, data in g.edges(data=True)
        if data.get("link_kind") == "backend_interconnect"
    }

    src_exchange = _exchange_id(src_ssu)
    dst_exchange = _exchange_id(dst_ssu)

    if src_exchange == dst_exchange:
        return _compute_min_hops_path(g, src_ssu, dst_ssu)

    if {"2d_torus_x", "2d_torus_y"}.issubset(backend_roles):
        size = _torus_side_length(_exchange_count(g), dimensions=2)
        src_coord = _exchange_to_coord_2d(src_exchange, size)
        dst_coord = _exchange_to_coord_2d(dst_exchange, size)
        dims = ((1, "union0"), (0, "union1"))
        coord_to_exchange = lambda c: _coord_to_exchange_2d(c, size)
    elif {"3d_torus_x", "3d_torus_y", "3d_torus_z"}.issubset(backend_roles):
        size = _torus_side_length(_exchange_count(g), dimensions=3)
        src_coord = _exchange_to_coord_3d(src_exchange, size)
        dst_coord = _exchange_to_coord_3d(dst_exchange, size)
        dims = ((0, "union0"), (1, "union1"), (2, "union0"))
        coord_to_exchange = lambda c: _coord_to_exchange_3d(c, size)
    else:
        # Fall back to shortest path when topology is not torus-shaped.
        return _compute_min_hops_path(g, src_ssu, dst_ssu)

    nodes: list[str] = [src_ssu]
    current_node = src_ssu
    current_exchange = src_exchange
    current_coord = tuple(src_coord)

    for axis, union_label in dims:
        signed_steps = _torus_signed_steps(current_coord[axis], dst_coord[axis], size)
        if signed_steps == 0:
            continue

        current_union = f"{current_exchange}:{union_label}"
        current_node = _move_to_union(g, nodes, current_node, current_exchange, current_union)

        direction = 1 if signed_steps > 0 else -1
        for _ in range(abs(signed_steps)):
            next_coord = list(current_coord)
            next_coord[axis] = (next_coord[axis] + direction) % size
            next_coord_tuple = tuple(next_coord)
            next_exchange = coord_to_exchange(next_coord_tuple)
            next_union = f"{next_exchange}:{union_label}"

            if not g.has_edge(current_union, next_union):
                raise ValueError(
                    f"DOR expected backend edge missing between {current_union} and {next_union}"
                )

            nodes.append(next_union)
            current_coord = next_coord_tuple
            current_exchange = next_exchange
            current_union = next_union
            current_node = next_union

    if current_node != dst_ssu:
        if not g.has_edge(current_node, dst_ssu):
            raise ValueError(f"DOR could not reach destination SSU from {current_node} to {dst_ssu}")
        nodes.append(dst_ssu)

    return [RoutedPath(nodes=tuple(nodes), weight=1.0)]


def _compute_port_balanced_paths(
    g: nx.Graph,
    src_ssu: str,
    dst_ssu: str,
    cfg: AnalysisConfig,
) -> list[RoutedPath]:
    if src_ssu == dst_ssu:
        return [RoutedPath(nodes=(src_ssu,), weight=1.0)]

    shortest_hops = nx.shortest_path_length(g, src_ssu, dst_ssu)
    max_hops = shortest_hops + max(0, int(cfg.port_balanced_max_detour_hops))

    source_unions = [
        neighbor
        for neighbor in g.neighbors(src_ssu)
        if g.nodes[neighbor].get("node_role") == "union"
    ]

    egress_ports: list[tuple[str, str]] = []
    for source_union in source_unions:
        for neighbor in g.neighbors(source_union):
            edge = g.get_edge_data(source_union, neighbor) or {}
            if edge.get("link_kind") != "backend_interconnect":
                continue
            egress_ports.append((source_union, neighbor))

    selected_paths: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for source_union, egress_neighbor in egress_ports:
        path = _shortest_path_via_egress(g, src_ssu, dst_ssu, source_union, egress_neighbor)
        if path is None:
            continue
        hops = len(path) - 1
        if hops > max_hops:
            continue
        if path in seen:
            continue
        seen.add(path)
        selected_paths.append(path)

    if not selected_paths:
        return _compute_min_hops_path(g, src_ssu, dst_ssu)

    weight = 1.0 / len(selected_paths)
    return [RoutedPath(nodes=path, weight=weight) for path in selected_paths]


def _shortest_path_via_egress(
    g: nx.Graph,
    src_ssu: str,
    dst_ssu: str,
    source_union: str,
    egress_neighbor: str,
) -> tuple[str, ...] | None:
    if not g.has_edge(src_ssu, source_union):
        return None
    edge = g.get_edge_data(source_union, egress_neighbor) or {}
    if edge.get("link_kind") != "backend_interconnect":
        return None

    prefix = [src_ssu, source_union, egress_neighbor]
    blocked = set(prefix[:-1])

    reduced = g.copy()
    reduced.remove_nodes_from(blocked)

    try:
        suffix = nx.shortest_path(reduced, egress_neighbor, dst_ssu)
    except nx.NetworkXNoPath:
        return None

    return tuple(prefix[:-1] + suffix)


def _move_to_union(
    g: nx.Graph,
    nodes: list[str],
    current_node: str,
    exchange_id: str,
    target_union: str,
) -> str:
    if current_node == target_union:
        return current_node

    if _exchange_id(current_node) != exchange_id:
        raise ValueError(
            f"Current node {current_node} does not belong to exchange {exchange_id}"
        )

    if _is_ssu(current_node):
        if not g.has_edge(current_node, target_union):
            raise ValueError(
                f"DOR expected internal edge missing between {current_node} and {target_union}"
            )
        nodes.append(target_union)
        return target_union

    transfer_ssu = f"{exchange_id}:ssu0"
    if current_node != transfer_ssu:
        if not g.has_edge(current_node, transfer_ssu):
            raise ValueError(
                f"DOR expected internal edge missing between {current_node} and {transfer_ssu}"
            )
        nodes.append(transfer_ssu)
    if not g.has_edge(transfer_ssu, target_union):
        raise ValueError(
            f"DOR expected internal edge missing between {transfer_ssu} and {target_union}"
        )
    nodes.append(target_union)
    return target_union


def _exchange_id(node_id: str) -> str:
    return str(node_id).split(":", 1)[0]


def _is_ssu(node_id: str) -> bool:
    return ":ssu" in str(node_id)


def _exchange_count(g: nx.Graph) -> int:
    exchanges = {
        data.get("exchange_node_id")
        for _, data in g.nodes(data=True)
        if data.get("exchange_node_id") is not None
    }
    return len(exchanges)


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


def _torus_signed_steps(src: int, dst: int, size: int) -> int:
    forward = (dst - src) % size
    backward = (src - dst) % size
    if forward <= backward:
        return forward
    return -backward


def _exchange_to_coord_2d(exchange_id: str, size: int) -> tuple[int, int]:
    idx = int(exchange_id.removeprefix("en"))
    row = idx // size
    col = idx % size
    return row, col


def _coord_to_exchange_2d(coord: Iterable[int], size: int) -> str:
    row, col = tuple(coord)
    return f"en{row * size + col}"


def _exchange_to_coord_3d(exchange_id: str, size: int) -> tuple[int, int, int]:
    idx = int(exchange_id.removeprefix("en"))
    x = idx // (size * size)
    rem = idx % (size * size)
    y = rem // size
    z = rem % size
    return x, y, z


def _coord_to_exchange_3d(coord: Iterable[int], size: int) -> str:
    x, y, z = tuple(coord)
    return f"en{(x * size * size) + (y * size) + z}"

