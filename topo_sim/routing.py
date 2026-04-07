from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from .config import AnalysisConfig


_ROUTING_ALIASES = {
    "MIN_HOPS": "SHORTEST_PATH",
    "PORT_BALANCED": "FULL_PATH",
}

_SUPPORTED_ROUTING_MODES = {
    "DOR",
    "SHORTEST_PATH",
    "FULL_PATH",
    "ECMP",
    *_ROUTING_ALIASES.keys(),
}


@dataclass(slots=True)
class RoutedPath:
    nodes: tuple[str, ...]
    weight: float

    @property
    def hops(self) -> int:
        return max(0, len(self.nodes) - 1)


def normalize_routing_mode(routing_mode: str) -> str:
    if not isinstance(routing_mode, str):
        raise ValueError("routing_mode must be a string")

    mode = routing_mode.strip().upper()
    if mode not in _SUPPORTED_ROUTING_MODES:
        raise ValueError(f"Unsupported routing mode: {routing_mode}")
    return _ROUTING_ALIASES.get(mode, mode)


def compute_paths(
    g: nx.Graph,
    src_ssu: str,
    dst_ssu: str,
    routing_mode: str,
    cfg: AnalysisConfig,
) -> list[RoutedPath]:
    mode = normalize_routing_mode(routing_mode)

    if src_ssu not in g or dst_ssu not in g:
        return []

    if src_ssu == dst_ssu:
        return [RoutedPath(nodes=(src_ssu,), weight=1.0)]

    if _exchange_id(src_ssu) == _exchange_id(dst_ssu):
        if _is_df_topology(g):
            return _compute_df_same_exchange_paths(g, src_ssu, dst_ssu)
        return _compute_same_exchange_internal_paths(g, src_ssu, dst_ssu, mode)

    if _is_df_topology(g):
        return _compute_df_paths(g, src_ssu, dst_ssu)

    if not nx.has_path(g, src_ssu, dst_ssu):
        return []

    try:
        if mode == "DOR":
            return _compute_dor_paths(g, src_ssu, dst_ssu)
        if mode == "SHORTEST_PATH":
            return _compute_dual_plane_shortest_paths(g, src_ssu, dst_ssu)
        if mode == "FULL_PATH":
            return _compute_dual_plane_full_paths(g, src_ssu, dst_ssu)
        return _compute_ecmp_paths(g, src_ssu, dst_ssu)
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
    topology_kind = _infer_direct_topology_kind(g)
    if topology_kind not in {"2D-FULLMESH", "2D-TORUS", "3D-TORUS"}:
        return _compute_dual_plane_shortest_paths(g, src_ssu, dst_ssu)

    src_exchange = _exchange_id(src_ssu)
    dst_exchange = _exchange_id(dst_ssu)
    plane_paths: dict[str, list[tuple[str, ...]]] = {}

    for source_union in _source_union_ids(g, src_ssu):
        union_label = _union_label(source_union)
        plane_path = _compute_dor_backend_union_path(
            g,
            topology_kind=topology_kind,
            src_exchange=src_exchange,
            dst_exchange=dst_exchange,
            union_label=union_label,
        )
        plane_paths[source_union] = [plane_path]

    return _weight_wrapped_plane_paths(src_ssu, dst_ssu, plane_paths)


def _compute_dual_plane_shortest_paths(g: nx.Graph, src_ssu: str, dst_ssu: str) -> list[RoutedPath]:
    if _infer_direct_topology_kind(g) is None:
        return _compute_ecmp_paths(g, src_ssu, dst_ssu)

    src_exchange = _exchange_id(src_ssu)
    dst_exchange = _exchange_id(dst_ssu)
    plane_paths: dict[str, list[tuple[str, ...]]] = {}

    for source_union in _source_union_ids(g, src_ssu):
        union_label = _union_label(source_union)
        dst_union = f"{dst_exchange}:{union_label}"
        plane_graph = _build_union_plane_graph(g, union_label)
        shortest_paths = [tuple(path) for path in nx.all_shortest_paths(plane_graph, source_union, dst_union)]
        plane_paths[source_union] = shortest_paths

    return _weight_wrapped_plane_paths(src_ssu, dst_ssu, plane_paths)


def _compute_dual_plane_full_paths(g: nx.Graph, src_ssu: str, dst_ssu: str) -> list[RoutedPath]:
    if _infer_direct_topology_kind(g) is None:
        return _compute_dual_plane_shortest_paths(g, src_ssu, dst_ssu)

    src_exchange = _exchange_id(src_ssu)
    dst_exchange = _exchange_id(dst_ssu)
    plane_paths: dict[str, list[tuple[str, ...]]] = {}

    for source_union in _source_union_ids(g, src_ssu):
        union_label = _union_label(source_union)
        dst_union = f"{dst_exchange}:{union_label}"
        plane_graph = _build_union_plane_graph(g, union_label)
        selected_paths: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()

        for egress_neighbor in sorted(plane_graph.neighbors(source_union)):
            path = _least_hops_path_via_egress(plane_graph, source_union, dst_union, egress_neighbor)
            if path is None or path in seen:
                continue
            seen.add(path)
            selected_paths.append(path)

        plane_paths[source_union] = selected_paths

    return _weight_wrapped_plane_paths(src_ssu, dst_ssu, plane_paths)


def _least_hops_path_via_egress(
    plane_graph: nx.Graph,
    source_union: str,
    dst_union: str,
    egress_neighbor: str,
) -> tuple[str, ...] | None:
    if not plane_graph.has_edge(source_union, egress_neighbor):
        return None

    if egress_neighbor == dst_union:
        return (source_union, dst_union)

    reduced = plane_graph.copy()
    reduced.remove_node(source_union)

    try:
        suffix = nx.shortest_path(reduced, egress_neighbor, dst_union)
    except nx.NetworkXNoPath:
        return None

    return tuple([source_union, *suffix])


def _weight_wrapped_plane_paths(
    src_ssu: str,
    dst_ssu: str,
    plane_paths: dict[str, list[tuple[str, ...]]],
) -> list[RoutedPath]:
    non_empty = [(source_union, paths) for source_union, paths in plane_paths.items() if paths]
    if not non_empty:
        return []

    plane_weight = 1.0 / len(non_empty)
    routed_paths: list[RoutedPath] = []

    for _, backend_paths in non_empty:
        per_path_weight = plane_weight / len(backend_paths)
        for backend_path in backend_paths:
            full_nodes = (src_ssu, *backend_path, dst_ssu)
            routed_paths.append(RoutedPath(nodes=full_nodes, weight=per_path_weight))

    return routed_paths


def _compute_df_same_exchange_paths(g: nx.Graph, src_ssu: str, dst_ssu: str) -> list[RoutedPath]:
    exchange_id = _exchange_id(src_ssu)
    internal_graph = _build_exchange_internal_graph(g, exchange_id)
    return _compute_ecmp_paths(internal_graph, src_ssu, dst_ssu)


def _compute_df_paths(g: nx.Graph, src_ssu: str, dst_ssu: str) -> list[RoutedPath]:
    src_server = _server_id(g, src_ssu)
    dst_server = _server_id(g, dst_ssu)
    if src_server is None or dst_server is None:
        return _compute_ecmp_paths(g, src_ssu, dst_ssu)

    if src_server == dst_server:
        return _compute_df_same_server_paths(g, src_ssu, dst_ssu)

    gateway_map = g.graph.get("df_inter_server_gateways", {})
    gateway_pair = gateway_map.get((src_server, dst_server))
    if gateway_pair is None:
        return []

    src_gateway, dst_gateway = gateway_pair
    if not g.has_edge(src_gateway, dst_gateway):
        return []
    source_prefixes = _df_paths_from_ssu_to_union(g, src_ssu, src_gateway)
    destination_suffixes = _df_paths_from_union_to_ssu(g, dst_gateway, dst_ssu)
    if not source_prefixes or not destination_suffixes:
        return []

    total_paths = len(source_prefixes) * len(destination_suffixes)
    weight = 1.0 / float(total_paths)
    return [
        RoutedPath(nodes=tuple(prefix + suffix), weight=weight)
        for prefix in source_prefixes
        for suffix in destination_suffixes
    ]


def _compute_df_same_server_paths(g: nx.Graph, src_ssu: str, dst_ssu: str) -> list[RoutedPath]:
    source_unions = _source_union_ids(g, src_ssu)
    destination_unions = _source_union_ids(g, dst_ssu)
    if not source_unions or not destination_unions:
        return []

    routed_nodes = [
        (src_ssu, source_union, destination_union, dst_ssu)
        for source_union in source_unions
        for destination_union in destination_unions
        if g.has_edge(source_union, destination_union)
    ]
    if not routed_nodes:
        return []

    weight = 1.0 / float(len(routed_nodes))
    return [RoutedPath(nodes=nodes, weight=weight) for nodes in routed_nodes]


def _df_paths_from_ssu_to_union(
    g: nx.Graph,
    src_ssu: str,
    target_union: str,
) -> list[tuple[str, ...]]:
    source_unions = _source_union_ids(g, src_ssu)
    if target_union in source_unions:
        return [(src_ssu, target_union)]

    return [
        (src_ssu, source_union, target_union)
        for source_union in source_unions
        if g.has_edge(source_union, target_union)
    ]


def _df_paths_from_union_to_ssu(
    g: nx.Graph,
    src_union: str,
    dst_ssu: str,
) -> list[tuple[str, ...]]:
    destination_unions = _source_union_ids(g, dst_ssu)
    if src_union in destination_unions:
        return [(src_union, dst_ssu)]

    return [
        (src_union, destination_union, dst_ssu)
        for destination_union in destination_unions
        if g.has_edge(src_union, destination_union)
    ]


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


def _compute_dor_backend_union_path(
    g: nx.Graph,
    *,
    topology_kind: str,
    src_exchange: str,
    dst_exchange: str,
    union_label: str,
) -> tuple[str, ...]:
    if topology_kind == "2D-FULLMESH":
        size = _torus_side_length(_exchange_count(g), dimensions=2)
        current_coord = _exchange_to_coord_2d(src_exchange, size)
        dst_coord = _exchange_to_coord_2d(dst_exchange, size)
        current_union = f"{src_exchange}:{union_label}"
        nodes: list[str] = [current_union]

        if current_coord[1] != dst_coord[1]:
            next_exchange = _coord_to_exchange_2d((current_coord[0], dst_coord[1]), size)
            next_union = f"{next_exchange}:{union_label}"
            if not g.has_edge(current_union, next_union):
                raise ValueError(
                    f"DOR expected backend edge missing between {current_union} and {next_union}"
                )
            nodes.append(next_union)
            current_coord = (current_coord[0], dst_coord[1])
            current_union = next_union

        if current_coord[0] != dst_coord[0]:
            next_exchange = _coord_to_exchange_2d((dst_coord[0], current_coord[1]), size)
            next_union = f"{next_exchange}:{union_label}"
            if not g.has_edge(current_union, next_union):
                raise ValueError(
                    f"DOR expected backend edge missing between {current_union} and {next_union}"
                )
            nodes.append(next_union)

        return tuple(nodes)

    if topology_kind == "2D-TORUS":
        size = _torus_side_length(_exchange_count(g), dimensions=2)
        current_coord = _exchange_to_coord_2d(src_exchange, size)
        dst_coord = _exchange_to_coord_2d(dst_exchange, size)
        axes = (1, 0)
        coord_to_exchange = lambda c: _coord_to_exchange_2d(c, size)
    else:
        size = _torus_side_length(_exchange_count(g), dimensions=3)
        current_coord = _exchange_to_coord_3d(src_exchange, size)
        dst_coord = _exchange_to_coord_3d(dst_exchange, size)
        axes = (0, 1, 2)
        coord_to_exchange = lambda c: _coord_to_exchange_3d(c, size)

    current_exchange = src_exchange
    current_union = f"{current_exchange}:{union_label}"
    nodes: list[str] = [current_union]

    for axis in axes:
        signed_steps = _torus_signed_steps(current_coord[axis], dst_coord[axis], size)
        if signed_steps == 0:
            continue

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

    return tuple(nodes)


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
    return str(g.graph.get("topology_name", "")).upper() == "DF"


def _source_union_ids(g: nx.Graph, src_ssu: str) -> list[str]:
    unions = [
        str(neighbor)
        for neighbor in g.neighbors(src_ssu)
        if g.nodes[neighbor].get("node_role") == "union"
    ]
    return sorted(unions, key=_union_label)


def _union_label(union_id: str) -> str:
    return str(union_id).split(":", 1)[1]


def _exchange_id(node_id: str) -> str:
    return str(node_id).split(":", 1)[0]


def _server_id(g: nx.Graph, node_id: str) -> int | None:
    value = g.nodes[node_id].get("server_id")
    if value is None:
        return None
    return int(value)


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
