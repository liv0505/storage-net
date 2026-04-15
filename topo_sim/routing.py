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
    if mode == "DOR" and (bool(g.graph.get("torus_twisted", False)) or _is_sparsemesh_topology(g)):
        mode = "SHORTEST_PATH"
    if mode == "ECMP" and _is_sparsemesh_topology(g):
        mode = "SHORTEST_PATH"

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

    if _is_single_plane_direct_torus(g):
        try:
            if mode == "DOR":
                return _compute_single_plane_torus_dor_paths(g, src_ssu, dst_ssu)
            if mode == "SHORTEST_PATH":
                return _compute_single_plane_direct_shortest_paths(g, src_ssu, dst_ssu)
            if mode == "FULL_PATH":
                return _compute_single_plane_direct_full_paths(g, src_ssu, dst_ssu)
        except nx.NetworkXNoPath:
            return []

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
    if _infer_direct_topology_kind(g) is None and not _is_sparsemesh_topology(g):
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
    if _infer_direct_topology_kind(g) is None and not _is_sparsemesh_topology(g):
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


def _build_backend_union_graph(g: nx.Graph) -> nx.Graph:
    union_nodes = {
        str(node_id)
        for node_id, data in g.nodes(data=True)
        if data.get("node_role") == "union"
    }
    backend_graph = nx.Graph()
    backend_graph.add_nodes_from(union_nodes)

    for u, v, data in g.edges(data=True):
        if data.get("link_kind") != "backend_interconnect":
            continue
        if u in union_nodes and v in union_nodes:
            backend_graph.add_edge(str(u), str(v), **data)

    return backend_graph


def _shortest_union_paths_in_graph(
    graph: nx.Graph,
    source_unions: list[str],
    destination_unions: list[str],
) -> list[tuple[str, ...]]:
    best_hops: int | None = None
    selected_paths: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    for source_union in source_unions:
        for destination_union in destination_unions:
            if source_union == destination_union:
                candidate_paths = [(source_union,)]
            else:
                if source_union not in graph or destination_union not in graph:
                    continue
                try:
                    candidate_paths = [
                        tuple(str(node_id) for node_id in path)
                        for path in nx.all_shortest_paths(graph, source_union, destination_union)
                    ]
                except nx.NetworkXNoPath:
                    continue

            if not candidate_paths:
                continue

            hop_count = len(candidate_paths[0]) - 1
            if best_hops is None or hop_count < best_hops:
                best_hops = hop_count
                selected_paths = []
                seen.clear()
            if hop_count != best_hops:
                continue
            for candidate in candidate_paths:
                if candidate in seen:
                    continue
                seen.add(candidate)
                selected_paths.append(candidate)

    return selected_paths


def _compute_single_plane_direct_shortest_paths(
    g: nx.Graph,
    src_ssu: str,
    dst_ssu: str,
) -> list[RoutedPath]:
    backend_graph = _build_backend_union_graph(g)
    source_unions = _source_union_ids(g, src_ssu)
    destination_unions = _source_union_ids(g, dst_ssu)
    routed_nodes = [
        (src_ssu, *union_path, dst_ssu)
        for union_path in _shortest_union_paths_in_graph(
            backend_graph,
            source_unions,
            destination_unions,
        )
    ]
    if not routed_nodes:
        return []

    weight = 1.0 / float(len(routed_nodes))
    return [RoutedPath(nodes=nodes, weight=weight) for nodes in routed_nodes]


def _torus_union_grid_shape(g: nx.Graph) -> tuple[int, ...]:
    shape = g.graph.get("torus_union_grid_shape")
    if not shape:
        raise ValueError("Single-plane torus graph is missing torus_union_grid_shape metadata")
    return tuple(int(value) for value in shape)


def _torus_union_coord(g: nx.Graph, union_id: str) -> tuple[int, ...]:
    coord = g.nodes[union_id].get("torus_union_coord")
    if coord is None:
        raise ValueError(f"Union '{union_id}' is missing torus_union_coord metadata")
    return tuple(int(value) for value in coord)


def _torus_coord_to_union_map(g: nx.Graph) -> dict[tuple[int, ...], str]:
    cache = g.graph.setdefault("_torus_coord_to_union_map_cache", {})
    cached = cache.get("map")
    if cached is not None:
        return cached

    coord_map = {
        tuple(int(value) for value in data.get("torus_union_coord")): str(node_id)
        for node_id, data in g.nodes(data=True)
        if data.get("node_role") == "union" and data.get("torus_union_coord") is not None
    }
    cache["map"] = coord_map
    return coord_map


def _single_plane_torus_axis_order(topology_kind: str) -> tuple[int, ...]:
    if topology_kind == "2D-TORUS":
        return (1, 0)
    if topology_kind == "3D-TORUS":
        return (0, 1, 2)
    raise ValueError(f"Unsupported single-plane torus kind: {topology_kind}")


def _single_plane_torus_dor_union_path(
    g: nx.Graph,
    topology_kind: str,
    source_union: str,
    destination_union: str,
) -> tuple[str, ...]:
    shape = _torus_union_grid_shape(g)
    current_coord = _torus_union_coord(g, source_union)
    dst_coord = _torus_union_coord(g, destination_union)
    coord_to_union = _torus_coord_to_union_map(g)
    axes = _single_plane_torus_axis_order(topology_kind)

    nodes: list[str] = [source_union]
    for axis in axes:
        signed_steps = _torus_signed_steps(current_coord[axis], dst_coord[axis], shape[axis])
        if signed_steps == 0:
            continue

        direction = 1 if signed_steps > 0 else -1
        for _ in range(abs(signed_steps)):
            next_coord = list(current_coord)
            next_coord[axis] = (next_coord[axis] + direction) % shape[axis]
            current_coord = tuple(next_coord)
            next_union = coord_to_union.get(current_coord)
            if next_union is None or not g.has_edge(nodes[-1], next_union):
                raise nx.NetworkXNoPath(
                    f"Missing single-plane torus edge between {nodes[-1]} and {next_union}"
                )
            nodes.append(next_union)
    return tuple(nodes)


def _compute_single_plane_torus_dor_paths(
    g: nx.Graph,
    src_ssu: str,
    dst_ssu: str,
) -> list[RoutedPath]:
    topology_kind = _infer_direct_topology_kind(g)
    if topology_kind not in {"2D-TORUS", "3D-TORUS"}:
        return _compute_single_plane_direct_shortest_paths(g, src_ssu, dst_ssu)

    source_unions = _source_union_ids(g, src_ssu)
    destination_unions = _source_union_ids(g, dst_ssu)
    best_hops: int | None = None
    selected_paths: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    for source_union in source_unions:
        for destination_union in destination_unions:
            union_path = _single_plane_torus_dor_union_path(
                g,
                topology_kind,
                source_union,
                destination_union,
            )
            hop_count = len(union_path) - 1
            if best_hops is None or hop_count < best_hops:
                best_hops = hop_count
                selected_paths = []
                seen.clear()
            if hop_count != best_hops:
                continue
            if union_path in seen:
                continue
            seen.add(union_path)
            selected_paths.append(union_path)

    routed_nodes = [(src_ssu, *union_path, dst_ssu) for union_path in selected_paths]
    if not routed_nodes:
        return []

    weight = 1.0 / float(len(routed_nodes))
    return [RoutedPath(nodes=nodes, weight=weight) for nodes in routed_nodes]


def _least_hops_path_via_egress_to_destinations(
    backend_graph: nx.Graph,
    source_union: str,
    destination_unions: list[str],
    egress_neighbor: str,
) -> tuple[str, ...] | None:
    if not backend_graph.has_edge(source_union, egress_neighbor):
        return None

    if egress_neighbor in destination_unions:
        return (source_union, egress_neighbor)

    reduced = backend_graph.copy()
    reduced.remove_node(source_union)

    best_path: tuple[str, ...] | None = None
    for destination_union in destination_unions:
        try:
            suffix = nx.shortest_path(reduced, egress_neighbor, destination_union)
        except nx.NetworkXNoPath:
            continue
        candidate = tuple([source_union, *suffix])
        if best_path is None or len(candidate) < len(best_path) or (
            len(candidate) == len(best_path) and candidate < best_path
        ):
            best_path = candidate

    return best_path


def _compute_single_plane_direct_full_paths(
    g: nx.Graph,
    src_ssu: str,
    dst_ssu: str,
) -> list[RoutedPath]:
    backend_graph = _build_backend_union_graph(g)
    source_unions = _source_union_ids(g, src_ssu)
    destination_unions = _source_union_ids(g, dst_ssu)
    selected_paths: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    for source_union in source_unions:
        for egress_neighbor in sorted(backend_graph.neighbors(source_union)):
            path = _least_hops_path_via_egress_to_destinations(
                backend_graph,
                source_union,
                destination_unions,
                egress_neighbor,
            )
            if path is None or path in seen:
                continue
            seen.add(path)
            selected_paths.append(path)

    if not selected_paths:
        return []

    weight = 1.0 / float(len(selected_paths))
    return [
        RoutedPath(nodes=(src_ssu, *backend_path, dst_ssu), weight=weight)
        for backend_path in selected_paths
    ]


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


def _df_local_topology(g: nx.Graph) -> str:
    return str(g.graph.get("df_local_topology", "fullmesh"))


def _df_backend_union_graph(g: nx.Graph) -> nx.Graph:
    cache = g.graph.setdefault("_df_backend_union_graph_cache", {})
    cached = cache.get("graph")
    if cached is not None:
        return cached

    union_nodes = {
        str(node_id)
        for node_id, data in g.nodes(data=True)
        if data.get("node_role") == "union"
    }
    backend_graph = nx.Graph()
    backend_graph.add_nodes_from(union_nodes)
    for u, v, data in g.edges(data=True):
        if data.get("link_kind") != "backend_interconnect":
            continue
        if u in union_nodes and v in union_nodes:
            backend_graph.add_edge(str(u), str(v), **data)

    cache["graph"] = backend_graph
    return backend_graph


def _df_server_union_graph(g: nx.Graph, server_id: int) -> nx.Graph:
    cache = g.graph.setdefault("_df_server_union_graph_cache", {})
    cached = cache.get(server_id)
    if cached is not None:
        return cached

    union_nodes = {
        str(node_id)
        for node_id, data in g.nodes(data=True)
        if data.get("node_role") == "union" and _server_id(g, str(node_id)) == server_id
    }
    server_graph = nx.Graph()
    server_graph.add_nodes_from(union_nodes)
    for u, v, data in g.edges(data=True):
        if data.get("link_kind") != "backend_interconnect":
            continue
        if u in union_nodes and v in union_nodes:
            server_graph.add_edge(str(u), str(v), **data)

    cache[server_id] = server_graph
    return server_graph


def _df_shortest_union_paths(
    g: nx.Graph,
    server_id: int,
    source_unions: list[str],
    destination_unions: list[str],
) -> list[tuple[str, ...]]:
    server_graph = _df_server_union_graph(g, server_id)
    if not source_unions or not destination_unions:
        return []

    best_hops: int | None = None
    selected_paths: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    for source_union in source_unions:
        for destination_union in destination_unions:
            if source_union == destination_union:
                candidate_paths = [(source_union,)]
            else:
                if source_union not in server_graph or destination_union not in server_graph:
                    continue
                try:
                    candidate_paths = [
                        tuple(str(node_id) for node_id in path)
                        for path in nx.all_shortest_paths(server_graph, source_union, destination_union)
                    ]
                except nx.NetworkXNoPath:
                    continue

            if not candidate_paths:
                continue

            hop_count = len(candidate_paths[0]) - 1
            if best_hops is None or hop_count < best_hops:
                best_hops = hop_count
                selected_paths = []
                seen.clear()
            if hop_count != best_hops:
                continue
            for candidate in candidate_paths:
                if candidate in seen:
                    continue
                seen.add(candidate)
                selected_paths.append(candidate)

    return selected_paths


def _df_shortest_backend_union_paths(
    g: nx.Graph,
    source_unions: list[str],
    destination_unions: list[str],
) -> list[tuple[str, ...]]:
    backend_graph = _df_backend_union_graph(g)
    if not source_unions or not destination_unions:
        return []

    best_hops: int | None = None
    selected_paths: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    for source_union in source_unions:
        for destination_union in destination_unions:
            if source_union == destination_union:
                candidate_paths = [(source_union,)]
            else:
                if source_union not in backend_graph or destination_union not in backend_graph:
                    continue
                try:
                    candidate_paths = [
                        tuple(str(node_id) for node_id in path)
                        for path in nx.all_shortest_paths(backend_graph, source_union, destination_union)
                    ]
                except nx.NetworkXNoPath:
                    continue

            if not candidate_paths:
                continue

            hop_count = len(candidate_paths[0]) - 1
            if best_hops is None or hop_count < best_hops:
                best_hops = hop_count
                selected_paths = []
                seen.clear()
            if hop_count != best_hops:
                continue
            for candidate in candidate_paths:
                if candidate in seen:
                    continue
                seen.add(candidate)
                selected_paths.append(candidate)

    return selected_paths


def _compute_df_paths(g: nx.Graph, src_ssu: str, dst_ssu: str) -> list[RoutedPath]:
    if _df_local_topology(g) in {"pair_double", "pair_triple"}:
        return _compute_df_paths_via_backend_shortest(g, src_ssu, dst_ssu)

    src_server = _server_id(g, src_ssu)
    dst_server = _server_id(g, dst_ssu)
    if src_server is None or dst_server is None:
        return _compute_df_paths_via_backend_shortest(g, src_ssu, dst_ssu)

    if src_server == dst_server:
        local_paths = _compute_df_same_server_local_paths(g, src_ssu, dst_ssu)
        if local_paths:
            return local_paths
        return _compute_df_paths_via_backend_shortest(g, src_ssu, dst_ssu)

    gateway_map = g.graph.get("df_inter_server_gateways", {})
    plane_path_nodes: list[list[tuple[str, ...]]] = []
    for plane_index in range(int(g.graph.get("df_plane_count", 1))):
        gateway_pair = gateway_map.get((plane_index, src_server, dst_server))
        if gateway_pair is None:
            continue

        src_gateway, dst_gateway = gateway_pair
        if not g.has_edge(src_gateway, dst_gateway):
            continue

        source_prefixes = _df_paths_from_ssu_to_union(g, src_ssu, src_gateway)
        destination_suffixes = _df_paths_from_union_to_ssu(g, dst_gateway, dst_ssu)
        if not source_prefixes or not destination_suffixes:
            continue

        plane_path_nodes.append(
            [
                tuple(prefix + suffix)
                for prefix in source_prefixes
                for suffix in destination_suffixes
            ]
        )

    if not plane_path_nodes:
        return _compute_df_paths_via_backend_shortest(g, src_ssu, dst_ssu)

    per_plane_weight = 1.0 / float(len(plane_path_nodes))
    routed_paths: list[RoutedPath] = []
    for nodes_list in plane_path_nodes:
        weight = per_plane_weight / float(len(nodes_list))
        routed_paths.extend(
            RoutedPath(nodes=nodes, weight=weight)
            for nodes in nodes_list
        )
    return routed_paths


def _compute_df_paths_via_backend_shortest(g: nx.Graph, src_ssu: str, dst_ssu: str) -> list[RoutedPath]:
    source_unions = _source_union_ids(g, src_ssu)
    destination_unions = _source_union_ids(g, dst_ssu)
    if not source_unions or not destination_unions:
        return []

    routed_nodes = [
        (src_ssu, *union_path, dst_ssu)
        for union_path in _df_shortest_backend_union_paths(g, source_unions, destination_unions)
    ]
    if not routed_nodes:
        return []

    weight = 1.0 / float(len(routed_nodes))
    return [RoutedPath(nodes=nodes, weight=weight) for nodes in routed_nodes]


def _compute_df_same_server_local_paths(g: nx.Graph, src_ssu: str, dst_ssu: str) -> list[RoutedPath]:
    source_unions = _source_union_ids(g, src_ssu)
    destination_unions = _source_union_ids(g, dst_ssu)
    src_server = _server_id(g, src_ssu)
    if src_server is None or not source_unions or not destination_unions:
        return []

    routed_nodes = [
        (src_ssu, *union_path, dst_ssu)
        for union_path in _df_shortest_union_paths(g, src_server, source_unions, destination_unions)
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
    src_server = _server_id(g, src_ssu)
    if src_server is None:
        return []
    return [
        (src_ssu, *union_path)
        for union_path in _df_shortest_union_paths(g, src_server, source_unions, [target_union])
    ]


def _df_paths_from_union_to_ssu(
    g: nx.Graph,
    src_union: str,
    dst_ssu: str,
) -> list[tuple[str, ...]]:
    destination_unions = _source_union_ids(g, dst_ssu)
    dst_server = _server_id(g, dst_ssu)
    if dst_server is None:
        return []
    return [
        (*union_path, dst_ssu)
        for union_path in _df_shortest_union_paths(g, dst_server, [src_union], destination_unions)
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

    shape = _torus_exchange_grid_shape(g)
    current_coord = _torus_exchange_coord(g, src_exchange)
    dst_coord = _torus_exchange_coord(g, dst_exchange)
    if topology_kind == "2D-TORUS":
        axes = (1, 0)
    else:
        axes = (0, 1, 2)

    current_exchange = src_exchange
    current_union = f"{current_exchange}:{union_label}"
    nodes: list[str] = [current_union]

    for axis in axes:
        signed_steps = _torus_signed_steps(current_coord[axis], dst_coord[axis], shape[axis])
        if signed_steps == 0:
            continue

        direction = 1 if signed_steps > 0 else -1
        for _ in range(abs(signed_steps)):
            next_coord = list(current_coord)
            next_coord[axis] = (next_coord[axis] + direction) % shape[axis]
            next_coord_tuple = tuple(next_coord)
            next_exchange = _torus_coord_to_exchange(g, next_coord_tuple)
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
    family = str(g.graph.get("topology_family", "")).upper()
    if family == "DF":
        return True
    topology_name = str(g.graph.get("topology_name", "")).upper()
    return topology_name == "DF" or topology_name.startswith("DF-")


def _is_sparsemesh_topology(g: nx.Graph) -> bool:
    family = str(g.graph.get("topology_family", "")).upper()
    if family == "SPARSEMESH":
        return True
    topology_name = str(g.graph.get("topology_name", "")).upper()
    return topology_name.startswith("SPARSEMESH")


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


def _torus_exchange_grid_shape(g: nx.Graph) -> tuple[int, ...]:
    shape = g.graph.get("torus_exchange_grid_shape")
    if not shape:
        raise ValueError("Torus graph is missing torus_exchange_grid_shape metadata")
    return tuple(int(value) for value in shape)


def _torus_exchange_coord(g: nx.Graph, exchange_id: str) -> tuple[int, ...]:
    for _, data in g.nodes(data=True):
        if str(data.get("exchange_node_id")) != exchange_id:
            continue
        coord = data.get("exchange_grid_coord")
        if coord is not None:
            return tuple(int(value) for value in coord)
    raise ValueError(f"Exchange '{exchange_id}' is missing exchange_grid_coord metadata")


def _torus_coord_to_exchange(g: nx.Graph, coord: tuple[int, ...]) -> str:
    cache = g.graph.setdefault("_torus_coord_to_exchange_cache", {})
    coord_map = cache.get("map")
    if coord_map is None:
        coord_map = {}
        for node_id, data in g.nodes(data=True):
            if data.get("node_role") != "union":
                continue
            exchange_id = data.get("exchange_node_id")
            exchange_coord = data.get("exchange_grid_coord")
            if exchange_id is None or exchange_coord is None:
                continue
            coord_map[tuple(int(value) for value in exchange_coord)] = str(exchange_id)
        cache["map"] = coord_map

    exchange_id = coord_map.get(tuple(int(value) for value in coord))
    if exchange_id is None:
        raise ValueError(f"Unknown torus exchange coord: {coord}")
    return exchange_id


def _is_single_plane_direct_torus(g: nx.Graph) -> bool:
    topology_kind = _infer_direct_topology_kind(g)
    if topology_kind not in {"2D-TORUS", "3D-TORUS"}:
        return False
    return str(g.graph.get("direct_backend_mode", "dual_plane")) == "single_plane"


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
