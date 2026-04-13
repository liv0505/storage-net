from __future__ import annotations

import math
from typing import Callable, Iterator

import networkx as nx

from .config import AnalysisConfig


TopologyBuilder = Callable[[AnalysisConfig], nx.Graph]

_BACKEND_BW_GBPS = 400.0
_INTERNAL_BW_GBPS = 200.0
_MIN_CLOS_UPLINKS_PER_PLANE = 1
_MAX_CLOS_UPLINKS_PER_PLANE = 6
_CLOS_EXCHANGE_NODE_COUNT = 18
_MIN_DF_UNIONS_PER_SERVER = 2
_MIN_DF_RING_UNIONS_PER_SERVER = 4


def _annotate_graph(g: nx.Graph, cfg: AnalysisConfig) -> nx.Graph:
    for _, data in g.nodes(data=True):
        data.setdefault("node_type", "switch")
    for u, v, data in g.edges(data=True):
        data.setdefault("bandwidth_gbps", cfg.link_bandwidth_gbps)
        data.setdefault("link_cost", cfg.link_cost)
        data.setdefault("weight", 1.0)
        g.edges[u, v].update(data)
    return g


def _iter_backend_edges(g: nx.Graph) -> Iterator[tuple[str, str, dict]]:
    for u, v, data in g.edges(data=True):
        if data.get("link_kind") == "backend_interconnect":
            yield u, v, data


def _validate_backend_uniformity(g: nx.Graph, topology_name: str) -> None:
    role_counts: dict[str, int] = {}
    uplinks_by_exchange: dict[str, int] = {}

    for u, v, data in _iter_backend_edges(g):
        role = str(data.get("topology_role", "backend_interconnect"))
        if not role.endswith("_local"):
            role_counts[role] = role_counts.get(role, 0) + 1

        for node in (u, v):
            exchange_node_id = g.nodes[node].get("exchange_node_id")
            if exchange_node_id is not None:
                uplinks_by_exchange[exchange_node_id] = (
                    uplinks_by_exchange.get(exchange_node_id, 0) + 1
                )

    is_df_family = topology_name.upper() == "DF" or topology_name.upper().startswith("DF-")
    is_single_plane_torus = topology_name in {"2D-Torus", "3D-Torus"}
    if not is_df_family and not is_single_plane_torus and len(role_counts) > 1 and len(set(role_counts.values())) != 1:
        raise ValueError(
            f"{topology_name} backend directions must stay uniform, got role counts: {role_counts}"
        )

    if uplinks_by_exchange and len(set(uplinks_by_exchange.values())) != 1:
        raise ValueError(
            f"{topology_name} exchange-node uplink counts must stay uniform, got: {uplinks_by_exchange}"
        )


def _validate_clos_uplink_budget(cfg: AnalysisConfig) -> None:
    if not (
        _MIN_CLOS_UPLINKS_PER_PLANE
        <= cfg.clos_uplinks_per_exchange_node
        <= _MAX_CLOS_UPLINKS_PER_PLANE
    ):
        raise ValueError(
            "clos_uplinks_per_exchange_node must be in "
            f"[{_MIN_CLOS_UPLINKS_PER_PLANE}, {_MAX_CLOS_UPLINKS_PER_PLANE}] per union plane"
        )


def _validate_clos_spine_fanout(g: nx.Graph) -> None:
    for node_id, node_data in g.nodes(data=True):
        if node_data.get("node_role") != "clos_spine":
            continue

        fanout = sum(
            1
            for _, _, edge_data in g.edges(node_id, data=True)
            if edge_data.get("link_kind") == "backend_interconnect"
        )
        if fanout != _CLOS_EXCHANGE_NODE_COUNT:
            raise ValueError(
                "Clos spine fanout must be exactly "
                f"{_CLOS_EXCHANGE_NODE_COUNT}, got {fanout} for {node_id}"
            )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _validate_df_server_shape(cfg: AnalysisConfig) -> None:
    unions_per_server = int(cfg.df_unions_per_server)
    if unions_per_server < _MIN_DF_UNIONS_PER_SERVER or unions_per_server % 2 != 0:
        raise ValueError("df_unions_per_server must be an even integer >= 2")
    if not _is_power_of_two(unions_per_server):
        raise ValueError("df_unions_per_server must be a power of two")
    if int(cfg.df_external_servers_per_union) <= 0:
        raise ValueError("df_external_servers_per_union must be > 0")


def _validate_df_ring_server_shape(cfg: AnalysisConfig) -> None:
    _validate_df_server_shape(cfg)
    unions_per_server = int(cfg.df_unions_per_server)
    if unions_per_server < _MIN_DF_RING_UNIONS_PER_SERVER:
        raise ValueError("DF ring-local variants require df_unions_per_server >= 4")


def _validate_df_pair_bridge_shape(cfg: AnalysisConfig) -> None:
    _validate_df_server_shape(cfg)
    if int(cfg.df_unions_per_server) != 4:
        raise ValueError("2P bridge variants currently require df_unions_per_server == 4")


def _add_exchange_node(g: nx.Graph, exchange_node_id: str, cfg: AnalysisConfig) -> dict[str, list[str]]:
    union_ids: list[str] = []
    ssu_ids: list[str] = []

    for union_index in range(2):
        union_id = f"{exchange_node_id}:union{union_index}"
        g.add_node(
            union_id,
            node_type="switch",
            node_role="union",
            exchange_node_id=exchange_node_id,
            local_index=union_index,
        )
        union_ids.append(union_id)

    for ssu_index in range(8):
        ssu_id = f"{exchange_node_id}:ssu{ssu_index}"
        g.add_node(
            ssu_id,
            node_type="endpoint",
            node_role="ssu",
            exchange_node_id=exchange_node_id,
            local_index=ssu_index,
        )
        ssu_ids.append(ssu_id)

        for union_id in union_ids:
            g.add_edge(
                ssu_id,
                union_id,
                bandwidth_gbps=_INTERNAL_BW_GBPS,
                link_kind="internal_ssu_uplink",
                topology_role="exchange_internal",
            )

    return {"ssus": ssu_ids, "unions": union_ids}


def _add_backend_link(
    g: nx.Graph,
    src_union_id: str,
    dst_union_id: str,
    topology_role: str,
    *,
    bandwidth_gbps: float = _BACKEND_BW_GBPS,
    parallel_links: int = 1,
) -> None:
    g.add_edge(
        src_union_id,
        dst_union_id,
        bandwidth_gbps=bandwidth_gbps,
        link_kind="backend_interconnect",
        topology_role=topology_role,
        parallel_links=parallel_links,
    )


def _set_exchange_grid_coord(
    g: nx.Graph,
    exchange: dict[str, list[str]],
    coord: tuple[int, ...],
) -> None:
    normalized = tuple(int(value) for value in coord)
    for node_id in [*exchange["ssus"], *exchange["unions"]]:
        g.nodes[node_id]["exchange_grid_coord"] = normalized


def _set_torus_union_coord(
    g: nx.Graph,
    union_id: str,
    coord: tuple[int, ...],
) -> None:
    g.nodes[union_id]["torus_union_coord"] = tuple(int(value) for value in coord)


def _torus_role_for_axis(
    topology_kind: str,
    axis: int,
    *,
    is_local_pair: bool,
) -> str:
    if topology_kind == "2D-Torus":
        if axis == 1 and is_local_pair:
            return "2d_torus_local"
        return {
            0: "2d_torus_y",
            1: "2d_torus_x",
        }[axis]
    if topology_kind == "3D-Torus":
        if axis == 2 and is_local_pair:
            return "3d_torus_local"
        return {
            0: "3d_torus_x",
            1: "3d_torus_y",
            2: "3d_torus_z",
        }[axis]
    raise ValueError(f"Unsupported torus topology kind: {topology_kind}")


def _build_single_plane_2d_torus(cfg: AnalysisConfig) -> nx.Graph:
    g = nx.Graph()
    union_rows = 4
    union_cols = 4
    exchange_cols = union_cols // 2
    exchanges: dict[tuple[int, int], dict[str, list[str]]] = {}
    coord_to_union: dict[tuple[int, int], str] = {}

    for row in range(union_rows):
        for exchange_col in range(exchange_cols):
            exchange_id = f"en{(row * exchange_cols) + exchange_col}"
            exchange = _add_exchange_node(g, exchange_id, cfg)
            _set_exchange_grid_coord(g, exchange, (row, exchange_col))
            exchanges[(row, exchange_col)] = exchange

            union_coords = (
                (row, exchange_col * 2),
                (row, (exchange_col * 2) + 1),
            )
            for union_id, union_coord in zip(exchange["unions"], union_coords):
                _set_torus_union_coord(g, union_id, union_coord)
                coord_to_union[union_coord] = union_id

    for row in range(union_rows):
        for col in range(union_cols):
            src_union = coord_to_union[(row, col)]
            x_neighbor_coord = (row, (col + 1) % union_cols)
            y_neighbor_coord = ((row + 1) % union_rows, col)

            for axis, dst_coord in ((1, x_neighbor_coord), (0, y_neighbor_coord)):
                dst_union = coord_to_union[dst_coord]
                if g.has_edge(src_union, dst_union):
                    continue
                is_local_pair = (
                    g.nodes[src_union].get("exchange_node_id") == g.nodes[dst_union].get("exchange_node_id")
                )
                _add_backend_link(
                    g,
                    src_union,
                    dst_union,
                    topology_role=_torus_role_for_axis("2D-Torus", axis, is_local_pair=is_local_pair),
                )

    g.graph["direct_backend_mode"] = "single_plane"
    g.graph["direct_plane_count"] = 1
    g.graph["logical_plane_union_count"] = union_rows * union_cols
    g.graph["logical_plane_ssu_count"] = (union_rows * union_cols) * 4
    g.graph["torus_union_grid_shape"] = (union_rows, union_cols)
    g.graph["torus_exchange_grid_shape"] = (union_rows, exchange_cols)
    g.graph["torus_pair_axis"] = 1
    return _annotate_graph(g, cfg)


def _build_single_plane_3d_torus(cfg: AnalysisConfig) -> nx.Graph:
    g = nx.Graph()
    size = 4
    exchange_depth = size // 2
    exchanges: dict[tuple[int, int, int], dict[str, list[str]]] = {}
    coord_to_union: dict[tuple[int, int, int], str] = {}

    for x in range(size):
        for y in range(size):
            for z_block in range(exchange_depth):
                exchange_id = f"en{(x * size * exchange_depth) + (y * exchange_depth) + z_block}"
                exchange = _add_exchange_node(g, exchange_id, cfg)
                _set_exchange_grid_coord(g, exchange, (x, y, z_block))
                exchanges[(x, y, z_block)] = exchange

                union_coords = (
                    (x, y, z_block * 2),
                    (x, y, (z_block * 2) + 1),
                )
                for union_id, union_coord in zip(exchange["unions"], union_coords):
                    _set_torus_union_coord(g, union_id, union_coord)
                    coord_to_union[union_coord] = union_id

    for x in range(size):
        for y in range(size):
            for z in range(size):
                src_union = coord_to_union[(x, y, z)]
                neighbor_coords = (
                    (0, ((x + 1) % size, y, z)),
                    (1, (x, (y + 1) % size, z)),
                    (2, (x, y, (z + 1) % size)),
                )
                for axis, dst_coord in neighbor_coords:
                    dst_union = coord_to_union[dst_coord]
                    if g.has_edge(src_union, dst_union):
                        continue
                    is_local_pair = (
                        g.nodes[src_union].get("exchange_node_id") == g.nodes[dst_union].get("exchange_node_id")
                    )
                    _add_backend_link(
                        g,
                        src_union,
                        dst_union,
                        topology_role=_torus_role_for_axis("3D-Torus", axis, is_local_pair=is_local_pair),
                    )

    g.graph["direct_backend_mode"] = "single_plane"
    g.graph["direct_plane_count"] = 1
    g.graph["logical_plane_union_count"] = size * size * size
    g.graph["logical_plane_ssu_count"] = size * size * size * 4
    g.graph["torus_union_grid_shape"] = (size, size, size)
    g.graph["torus_exchange_grid_shape"] = (size, size, exchange_depth)
    g.graph["torus_pair_axis"] = 2
    return _annotate_graph(g, cfg)


def _grid_coord_to_index(coord: tuple[int, ...], shape: tuple[int, ...]) -> int:
    index = 0
    stride = 1
    for axis_size, axis_value in zip(reversed(shape), reversed(coord)):
        index += int(axis_value) * stride
        stride *= int(axis_size)
    return index


def _iter_grid_coords(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    if len(shape) == 2:
        rows, cols = shape
        for row in range(rows):
            for col in range(cols):
                yield (row, col)
        return
    if len(shape) == 3:
        x_size, y_size, z_size = shape
        for x in range(x_size):
            for y in range(y_size):
                for z in range(z_size):
                    yield (x, y, z)
        return
    raise ValueError(f"Unsupported grid shape: {shape}")


def _build_dual_plane_torus(
    cfg: AnalysisConfig,
    *,
    topology_kind: str,
    shape: tuple[int, ...],
) -> nx.Graph:
    if topology_kind not in {"2D-Torus", "3D-Torus"}:
        raise ValueError(f"Unsupported torus topology kind: {topology_kind}")

    g = nx.Graph()
    exchanges: dict[tuple[int, ...], dict[str, list[str]]] = {}
    axis_count = len(shape)

    for coord in _iter_grid_coords(shape):
        exchange_id = f"en{_grid_coord_to_index(coord, shape)}"
        exchange = _add_exchange_node(g, exchange_id, cfg)
        _set_exchange_grid_coord(g, exchange, coord)
        exchanges[coord] = exchange

    for union_index in range(2):
        for coord in _iter_grid_coords(shape):
            src_union = exchanges[coord]["unions"][union_index]
            for axis in range(axis_count):
                next_coord = list(coord)
                next_coord[axis] = (next_coord[axis] + 1) % shape[axis]
                dst_union = exchanges[tuple(next_coord)]["unions"][union_index]
                parallel_links = 2 if int(shape[axis]) == 2 else 1
                _add_backend_link(
                    g,
                    src_union,
                    dst_union,
                    topology_role=_torus_role_for_axis(
                        topology_kind,
                        axis,
                        is_local_pair=False,
                    ),
                    bandwidth_gbps=_BACKEND_BW_GBPS * float(parallel_links),
                    parallel_links=parallel_links,
                )

    g.graph["direct_backend_mode"] = "dual_plane"
    g.graph["direct_plane_count"] = 2
    g.graph["logical_plane_union_count"] = math.prod(shape) * 2
    g.graph["logical_plane_ssu_count"] = math.prod(shape) * 8
    g.graph["torus_exchange_grid_shape"] = tuple(int(value) for value in shape)
    return _annotate_graph(g, cfg)


def _add_df_server_fullmesh_links(g: nx.Graph, union_ids: list[str]) -> None:
    for src_index, src_union_id in enumerate(union_ids):
        for dst_union_id in union_ids[src_index + 1 :]:
            _add_backend_link(
                g,
                src_union_id,
                dst_union_id,
                topology_role="df_server_fullmesh",
            )


def _add_df_server_ring_links(g: nx.Graph, union_ids: list[str]) -> None:
    union_count = len(union_ids)
    if union_count < _MIN_DF_RING_UNIONS_PER_SERVER:
        raise ValueError("DF ring-local variants require at least four Unions per server")

    for src_index in range(union_count):
        dst_index = (src_index + 1) % union_count
        if src_index < dst_index or dst_index == 0:
            _add_backend_link(
                g,
                union_ids[src_index],
                union_ids[dst_index],
                topology_role="df_server_ring",
            )


def _add_df_exchange_pair_links(
    g: nx.Graph,
    exchange_units: list[list[str]],
    *,
    pair_parallel_links: int,
    topology_role: str,
) -> None:
    pair_bandwidth = _BACKEND_BW_GBPS * float(pair_parallel_links)
    for unit_unions in exchange_units:
        if len(unit_unions) != 2:
            raise ValueError("Each DF exchange unit must contain exactly two Unions")
        _add_backend_link(
            g,
            unit_unions[0],
            unit_unions[1],
            topology_role=topology_role,
            bandwidth_gbps=pair_bandwidth,
            parallel_links=pair_parallel_links,
        )


def _add_df_server_pair_bridges(g: nx.Graph, exchange_units: list[list[str]]) -> None:
    if len(exchange_units) != 2:
        raise ValueError("2P bridge variants currently support exactly two exchange units per server")

    left_unit, right_unit = exchange_units
    _add_backend_link(
        g,
        left_unit[0],
        right_unit[0],
        topology_role="df_server_bridge",
    )
    _add_backend_link(
        g,
        left_unit[1],
        right_unit[1],
        topology_role="df_server_bridge",
    )


def _df_local_ports_per_union(local_topology: str) -> int:
    return {
        "fullmesh": 3,
        "ring": 2,
        "pair_double": 2,
        "pair_triple": 3,
        "pair_double_bridge": 3,
    }[local_topology]


def _df_target_union_index(
    src_union_index: int,
    unions_per_server: int,
) -> int:
    return unions_per_server - 1 - src_union_index


def _df_relative_server_offsets(
    src_union_index: int,
    unions_per_server: int,
    external_servers_per_union: int,
    global_pattern: str,
) -> list[int]:
    if global_pattern == "contiguous":
        return [
            (src_union_index * external_servers_per_union) + offset + 1
            for offset in range(external_servers_per_union)
        ]
    if global_pattern == "interleaved":
        return [
            src_union_index + 1 + (offset * unions_per_server)
            for offset in range(external_servers_per_union)
        ]
    raise ValueError(f"Unsupported DF global pattern: {global_pattern}")


def _build_df_variant(
    cfg: AnalysisConfig,
    *,
    topology_key: str,
    local_topology: str,
    global_pattern: str,
    external_servers_per_union: int,
    plane_count: int,
) -> nx.Graph:
    if local_topology == "ring":
        _validate_df_ring_server_shape(cfg)
    elif local_topology == "pair_double_bridge":
        _validate_df_pair_bridge_shape(cfg)
    else:
        _validate_df_server_shape(cfg)

    g = nx.Graph()
    unions_per_server = int(cfg.df_unions_per_server)
    server_count_per_plane = (unions_per_server * external_servers_per_union) + 1
    if plane_count != 2:
        raise ValueError("DF variants currently require exactly two Union planes per exchange group")

    exchange_nodes_per_server = unions_per_server
    exchange_count_per_plane = server_count_per_plane * exchange_nodes_per_server
    inter_server_gateways: dict[tuple[int, int, int], tuple[str, str]] = {}
    unions_by_plane_server: dict[int, dict[int, list[str]]] = {
        plane_index: {} for plane_index in range(plane_count)
    }
    pair_units_by_plane_server: dict[int, dict[int, list[list[str]]]] = {
        plane_index: {} for plane_index in range(plane_count)
    }

    exchange_index = 0
    for logical_server_id in range(server_count_per_plane):
        plane_unions: dict[int, list[str]] = {plane_index: [] for plane_index in range(plane_count)}

        for group_local_index in range(exchange_nodes_per_server):
            exchange_id = f"en{exchange_index}"
            exchange = _add_exchange_node(g, exchange_id, cfg)

            for ssu_id in exchange["ssus"]:
                g.nodes[ssu_id].update(
                    server_id=logical_server_id,
                    plane_local_server_id=logical_server_id,
                    df_plane_index=None,
                    df_group_index=exchange_index,
                    df_group_local_index=group_local_index,
                )

            for plane_index, union_id in enumerate(exchange["unions"][:plane_count]):
                g.nodes[union_id].update(
                    server_id=logical_server_id,
                    plane_local_server_id=logical_server_id,
                    df_plane_index=plane_index,
                    server_local_union_index=group_local_index,
                    df_group_index=exchange_index,
                    df_group_local_index=group_local_index,
                )
                plane_unions[plane_index].append(union_id)

            exchange_index += 1

        for plane_index in range(plane_count):
            unions_by_plane_server[plane_index][logical_server_id] = list(plane_unions[plane_index])
            pair_units_by_plane_server[plane_index][logical_server_id] = [
                plane_unions[plane_index][offset : offset + 2]
                for offset in range(0, len(plane_unions[plane_index]), 2)
            ]

    for plane_index in range(plane_count):
        for logical_server_id, union_ids in unions_by_plane_server[plane_index].items():
            exchange_units = pair_units_by_plane_server[plane_index][logical_server_id]
            if local_topology == "fullmesh":
                _add_df_server_fullmesh_links(g, union_ids)
            elif local_topology == "ring":
                _add_df_server_ring_links(g, union_ids)
            elif local_topology == "pair_double":
                _add_df_exchange_pair_links(
                    g,
                    exchange_units,
                    pair_parallel_links=2,
                    topology_role="df_pair_double",
                )
            elif local_topology == "pair_triple":
                _add_df_exchange_pair_links(
                    g,
                    exchange_units,
                    pair_parallel_links=3,
                    topology_role="df_pair_triple",
                )
            elif local_topology == "pair_double_bridge":
                _add_df_exchange_pair_links(
                    g,
                    exchange_units,
                    pair_parallel_links=2,
                    topology_role="df_pair_double",
                )
                _add_df_server_pair_bridges(g, exchange_units)
            else:
                raise ValueError(f"Unsupported DF local topology: {local_topology}")

        for src_server in range(server_count_per_plane):
            for src_union_index, src_union_id in enumerate(unions_by_plane_server[plane_index][src_server]):
                dst_union_index = _df_target_union_index(src_union_index, unions_per_server)
                for relative_offset in _df_relative_server_offsets(
                    src_union_index,
                    unions_per_server,
                    external_servers_per_union,
                    global_pattern,
                ):
                    dst_server = (src_server + relative_offset) % server_count_per_plane
                    dst_union_id = unions_by_plane_server[plane_index][dst_server][dst_union_index]
                    if g.has_edge(src_union_id, dst_union_id):
                        continue
                    _add_backend_link(
                        g,
                        src_union_id,
                        dst_union_id,
                        topology_role="df_inter_server",
                    )
                    inter_server_gateways[(plane_index, src_server, dst_server)] = (
                        src_union_id,
                        dst_union_id,
                    )
                    inter_server_gateways[(plane_index, dst_server, src_server)] = (
                        dst_union_id,
                        src_union_id,
                    )

    base_local_ports = _df_local_ports_per_union(local_topology)
    base_global_ports = external_servers_per_union
    g.graph["topology_family"] = "DF"
    g.graph["df_variant"] = topology_key
    g.graph["df_local_topology"] = local_topology
    g.graph["df_global_pattern"] = global_pattern
    g.graph["df_plane_count"] = plane_count
    g.graph["df_server_count"] = server_count_per_plane
    g.graph["df_total_server_count"] = server_count_per_plane * plane_count
    g.graph["df_exchange_nodes_per_server"] = exchange_nodes_per_server
    g.graph["df_group_count"] = exchange_count_per_plane
    g.graph["df_exchange_count_per_plane"] = exchange_count_per_plane
    g.graph["df_union_count_per_plane"] = exchange_count_per_plane
    g.graph["df_ssu_count_per_plane"] = exchange_count_per_plane * 4
    g.graph["logical_plane_union_count"] = g.graph["df_union_count_per_plane"]
    g.graph["logical_plane_ssu_count"] = g.graph["df_ssu_count_per_plane"]
    g.graph["df_unions_per_server"] = unions_per_server
    g.graph["df_external_servers_per_union"] = external_servers_per_union
    g.graph["df_base_local_ports_per_union"] = base_local_ports
    g.graph["df_base_global_ports_per_union"] = base_global_ports
    g.graph["df_local_ports_per_union"] = base_local_ports
    g.graph["df_global_ports_per_union"] = base_global_ports
    g.graph["df_backend_ports_per_union"] = (
        g.graph["df_local_ports_per_union"] + g.graph["df_global_ports_per_union"]
    )
    g.graph["df_inter_server_gateways"] = inter_server_gateways
    return _annotate_graph(g, cfg)


def build_2d_fullmesh(cfg: AnalysisConfig) -> nx.Graph:
    g = nx.Graph()
    rows = 4
    cols = 4
    exchanges: dict[tuple[int, int], dict[str, list[str]]] = {}

    for r in range(rows):
        for c in range(cols):
            exchange_id = f"en{r * cols + c}"
            exchanges[(r, c)] = _add_exchange_node(g, exchange_id, cfg)

    for union_index in range(2):
        for r in range(rows):
            for c1 in range(cols):
                for c2 in range(c1 + 1, cols):
                    _add_backend_link(
                        g,
                        exchanges[(r, c1)]["unions"][union_index],
                        exchanges[(r, c2)]["unions"][union_index],
                        topology_role="2d_fullmesh_x",
                    )

        for c in range(cols):
            for r1 in range(rows):
                for r2 in range(r1 + 1, rows):
                    _add_backend_link(
                        g,
                        exchanges[(r1, c)]["unions"][union_index],
                        exchanges[(r2, c)]["unions"][union_index],
                        topology_role="2d_fullmesh_y",
                    )

    g.graph["direct_backend_mode"] = "dual_plane"
    g.graph["direct_plane_count"] = 2
    g.graph["logical_plane_union_count"] = rows * cols * 2
    g.graph["logical_plane_ssu_count"] = rows * cols * 8
    return _annotate_graph(g, cfg)


def build_2d_torus(cfg: AnalysisConfig) -> nx.Graph:
    return _build_dual_plane_torus(
        cfg,
        topology_kind="2D-Torus",
        shape=(2, 4),
    )


def build_3d_torus(cfg: AnalysisConfig) -> nx.Graph:
    return _build_dual_plane_torus(
        cfg,
        topology_kind="3D-Torus",
        shape=(2, 4, 4),
    )


def build_clos(cfg: AnalysisConfig) -> nx.Graph:
    _validate_clos_uplink_budget(cfg)

    g = nx.Graph()
    exchange_nodes = [_add_exchange_node(g, f"en{idx}", cfg) for idx in range(_CLOS_EXCHANGE_NODE_COUNT)]

    plane_spine_ids: dict[int, list[str]] = {}
    for plane_index in range(2):
        spine_ids = [
            f"clos_spine_plane{plane_index}_uplink{index}"
            for index in range(cfg.clos_uplinks_per_exchange_node)
        ]
        plane_spine_ids[plane_index] = spine_ids
        for spine_id in spine_ids:
            g.add_node(
                spine_id,
                node_type="switch",
                node_role="clos_spine",
                union_plane=plane_index,
            )

    for exchange in exchange_nodes:
        for plane_index, union_id in enumerate(exchange["unions"]):
            for spine_id in plane_spine_ids[plane_index]:
                _add_backend_link(
                    g,
                    union_id,
                    spine_id,
                    topology_role="clos_uplink",
                )

    _validate_clos_spine_fanout(g)
    return _annotate_graph(g, cfg)


def build_df(cfg: AnalysisConfig) -> nx.Graph:
    return _build_df_variant(
        cfg,
        topology_key="DF",
        local_topology="fullmesh",
        global_pattern="contiguous",
        external_servers_per_union=int(cfg.df_external_servers_per_union),
        plane_count=2,
    )


def build_df_shuffled(cfg: AnalysisConfig) -> nx.Graph:
    return _build_df_variant(
        cfg,
        topology_key="DF-Shuffled",
        local_topology="fullmesh",
        global_pattern="interleaved",
        external_servers_per_union=int(cfg.df_external_servers_per_union),
        plane_count=2,
    )


def build_df_scaleup(cfg: AnalysisConfig) -> nx.Graph:
    return _build_df_variant(
        cfg,
        topology_key="DF-ScaleUp",
        local_topology="ring",
        global_pattern="interleaved",
        external_servers_per_union=int(cfg.df_external_servers_per_union) + 1,
        plane_count=2,
    )


def build_df_2p_double_4global(cfg: AnalysisConfig) -> nx.Graph:
    return _build_df_variant(
        cfg,
        topology_key="DF-2P-Double-4Global",
        local_topology="pair_double",
        global_pattern="contiguous",
        external_servers_per_union=int(cfg.df_external_servers_per_union) + 1,
        plane_count=2,
    )


def build_df_2p_triple_3global(cfg: AnalysisConfig) -> nx.Graph:
    return _build_df_variant(
        cfg,
        topology_key="DF-2P-Triple-3Global",
        local_topology="pair_triple",
        global_pattern="contiguous",
        external_servers_per_union=int(cfg.df_external_servers_per_union),
        plane_count=2,
    )


def build_df_2p_double_bridge_3global(cfg: AnalysisConfig) -> nx.Graph:
    return _build_df_variant(
        cfg,
        topology_key="DF-2P-Double-Bridge-3Global",
        local_topology="pair_double_bridge",
        global_pattern="contiguous",
        external_servers_per_union=int(cfg.df_external_servers_per_union),
        plane_count=2,
    )


BUILDERS: dict[str, TopologyBuilder] = {
    "2D-FullMesh": build_2d_fullmesh,
    "2D-Torus": build_2d_torus,
    "3D-Torus": build_3d_torus,
    "Clos": build_clos,
    "DF": build_df,
    "DF-Shuffled": build_df_shuffled,
    "DF-ScaleUp": build_df_scaleup,
    "DF-2P-Double-4Global": build_df_2p_double_4global,
    "DF-2P-Triple-3Global": build_df_2p_triple_3global,
    "DF-2P-Double-Bridge-3Global": build_df_2p_double_bridge_3global,
}


NORMALIZED_BUILDERS: dict[str, TopologyBuilder] = {name.lower(): builder for name, builder in BUILDERS.items()}


def build_topology(name: str, cfg: AnalysisConfig) -> nx.Graph:
    if not isinstance(name, str):
        valid = ", ".join(BUILDERS.keys())
        raise ValueError(f"Unknown topology '{name}'. Valid: {valid}")

    key = name.lower().strip()
    builder = NORMALIZED_BUILDERS.get(key)
    if builder is None:
        valid = ", ".join(BUILDERS.keys())
        raise ValueError(f"Unknown topology '{name}'. Valid: {valid}")

    g = builder(cfg)
    canonical_name = next(k for k in BUILDERS if k.lower() == key)
    g.graph["topology_name"] = canonical_name
    _validate_backend_uniformity(g, canonical_name)
    return g


def available_topologies() -> list[str]:
    return list(BUILDERS.keys())
