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
_CLOS_SCALED_EXCHANGE_COUNTS: dict[str, int] = {
    "Clos-64": 8,
    "Clos-128": 16,
    "Clos-192": 24,
    "Clos-256": 32,
}
_CLOS_4P_FULLMESH_GROUP_SIZE = 4
_CLOS_4P_FULLMESH_GROUPS_PER_PLANE = 32
_CLOS_4P_FULLMESH_PLANES = 2
_CLOS_4P_FULLMESH_LEAFS_PER_PLANE = 2
_CLOS_4P_FULLMESH_LEAF_UPLINK_BW_GBPS = 400.0
_CLOS_4P_RING_LEAF_UPLINK_BW_GBPS = 800.0
_CLOS_4P_RING_LEAF_UPLINK_PARALLEL_LINKS = 2
_CLOS_4P_FULLMESH_EXCHANGE_NODE_COUNT = (
    _CLOS_4P_FULLMESH_GROUP_SIZE * _CLOS_4P_FULLMESH_GROUPS_PER_PLANE
)
_MIN_DF_UNIONS_PER_SERVER = 2
_MIN_DF_RING_UNIONS_PER_SERVER = 4
_FULLMESH_SHAPES: dict[str, tuple[int, int]] = {
    "2D-FullMesh": (4, 4),
    "2D-FullMesh-2x4": (2, 4),
}
_SPARSEMESH_VARIANTS: dict[str, dict[str, int | bool]] = {
    "SparseMesh-Local": {
        "sparsity": 5,
        "stride_count": 2,
        "sparser": False,
    },
    "SparseMesh-Global": {
        "sparsity": 3,
        "stride_count": 4,
        "sparser": False,
    },
}
_TORUS_SHAPES: dict[str, tuple[int, ...]] = {
    "2D-Torus": (2, 4),
    "3D-Torus": (2, 4, 4),
    "3D-Torus-2x4x3": (2, 4, 3),
    "3D-Torus-2x4x2": (2, 4, 2),
    "3D-Torus-2x4x1": (2, 4, 1),
}
_TORUS_BEST_TWIST_OFFSETS: dict[str, tuple[tuple[int, ...], ...]] = {
    "2D-Torus-BestTwist": ((0, 2), (0, 0)),
    "3D-Torus-BestTwist": ((0, 2, 2), (0, 0, 0), (0, 0, 0)),
    "3D-Torus-2x4x3-BestTwist": ((0, 2, 0), (0, 0, 0), (0, 0, 0)),
    "3D-Torus-2x4x2-BestTwist": ((0, 0, 1), (1, 0, 1), (0, 2, 0)),
    "3D-Torus-2x4x1-BestTwist": ((0, 2, 0), (0, 0, 0), (0, 0, 0)),
}
_TORUS_VARIANT_SHAPE_NAMES: dict[str, str] = {
    **{name: name for name in _TORUS_SHAPES},
    "2D-Torus-BestTwist": "2D-Torus",
    "3D-Torus-BestTwist": "3D-Torus",
    "3D-Torus-2x4x3-BestTwist": "3D-Torus-2x4x3",
    "3D-Torus-2x4x2-BestTwist": "3D-Torus-2x4x2",
    "3D-Torus-2x4x1-BestTwist": "3D-Torus-2x4x1",
}
_TORUS_VARIANT_FAMILY_NAMES: dict[str, str] = {
    "2D-Torus": "2D-Torus",
    "2D-Torus-BestTwist": "2D-Torus",
    "3D-Torus": "3D-Torus",
    "3D-Torus-BestTwist": "3D-Torus",
    "3D-Torus-2x4x3": "3D-Torus",
    "3D-Torus-2x4x3-BestTwist": "3D-Torus",
    "3D-Torus-2x4x2": "3D-Torus",
    "3D-Torus-2x4x2-BestTwist": "3D-Torus",
    "3D-Torus-2x4x1": "3D-Torus",
    "3D-Torus-2x4x1-BestTwist": "3D-Torus",
}


def torus_base_name(topology_name: str) -> str | None:
    if not isinstance(topology_name, str):
        return None

    normalized = topology_name.strip().lower()
    for variant_name, family_name in _TORUS_VARIANT_FAMILY_NAMES.items():
        if variant_name.lower() == normalized:
            return family_name
    return None


def _torus_shape_variant_name(topology_name: str) -> str | None:
    if not isinstance(topology_name, str):
        return None

    normalized = topology_name.strip().lower()
    for variant_name, shape_name in _TORUS_VARIANT_SHAPE_NAMES.items():
        if variant_name.lower() == normalized:
            return shape_name
    return None


def is_fullmesh_topology_name(topology_name: str) -> bool:
    if not isinstance(topology_name, str):
        return False
    normalized = topology_name.strip().lower()
    return any(name.lower() == normalized for name in _FULLMESH_SHAPES)


def fullmesh_shape(topology_name: str) -> tuple[int, int]:
    if not isinstance(topology_name, str):
        valid = ", ".join(sorted(_FULLMESH_SHAPES))
        raise ValueError(f"Unsupported fullmesh topology '{topology_name}'. Valid: {valid}")

    normalized = topology_name.strip().lower()
    for variant_name, shape in _FULLMESH_SHAPES.items():
        if variant_name.lower() == normalized:
            return tuple(int(value) for value in shape)

    valid = ", ".join(sorted(_FULLMESH_SHAPES))
    raise ValueError(f"Unsupported fullmesh topology '{topology_name}'. Valid: {valid}")


def is_torus_topology_name(topology_name: str) -> bool:
    return torus_base_name(topology_name) is not None


def is_best_twisted_torus_name(topology_name: str) -> bool:
    if not isinstance(topology_name, str):
        return False

    normalized = topology_name.strip().lower()
    return any(name.lower() == normalized for name in _TORUS_BEST_TWIST_OFFSETS)


def is_sparsemesh_topology_name(topology_name: str) -> bool:
    if not isinstance(topology_name, str):
        return False
    normalized = topology_name.strip().lower()
    return any(name.lower() == normalized for name in _SPARSEMESH_VARIANTS)


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
    uplinks_by_exchange: dict[str, int] = {}

    for u, v, data in _iter_backend_edges(g):
        for node in (u, v):
            exchange_node_id = g.nodes[node].get("exchange_node_id")
            if exchange_node_id is not None:
                uplinks_by_exchange[exchange_node_id] = (
                    uplinks_by_exchange.get(exchange_node_id, 0) + 1
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


def _validate_clos_spine_fanout(g: nx.Graph, expected_exchange_count: int) -> None:
    for node_id, node_data in g.nodes(data=True):
        if node_data.get("node_role") != "clos_spine":
            continue

        fanout = sum(
            1
            for _, _, edge_data in g.edges(node_id, data=True)
            if edge_data.get("link_kind") == "backend_interconnect"
        )
        if fanout != expected_exchange_count:
            raise ValueError(
                "Clos spine fanout must be exactly "
                f"{expected_exchange_count}, got {fanout} for {node_id}"
            )


def _validate_clos_leaf_fanout(g: nx.Graph) -> None:
    for node_id, node_data in g.nodes(data=True):
        if node_data.get("node_role") != "clos_leaf":
            continue

        fanout = sum(
            1
            for _, _, edge_data in g.edges(node_id, data=True)
            if edge_data.get("link_kind") == "backend_interconnect"
        )
        if fanout != _CLOS_4P_FULLMESH_EXCHANGE_NODE_COUNT:
            raise ValueError(
                "Clos 4P leaf fanout must be exactly "
                f"{_CLOS_4P_FULLMESH_EXCHANGE_NODE_COUNT}, got {fanout} for {node_id}"
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


def _clos_exchange_grid_shape(exchange_count: int) -> tuple[int, int]:
    predefined = {
        8: (4, 2),
        16: (4, 4),
        18: (6, 3),
        24: (6, 4),
        32: (8, 4),
    }
    if exchange_count in predefined:
        return predefined[exchange_count]

    cols = max(1, math.ceil(math.sqrt(exchange_count)))
    rows = max(1, math.ceil(exchange_count / cols))
    while cols > rows + 2 and exchange_count % (cols - 1) == 0:
        cols -= 1
        rows = max(1, math.ceil(exchange_count / cols))
    return cols, rows


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


def _normalize_torus_wrap_offsets(
    shape: tuple[int, ...],
    wrap_offsets_by_axis: tuple[tuple[int, ...], ...] | None,
) -> tuple[tuple[int, ...], ...]:
    axis_count = len(shape)
    if wrap_offsets_by_axis is None:
        return tuple(tuple(0 for _ in range(axis_count)) for _ in range(axis_count))

    if len(wrap_offsets_by_axis) != axis_count:
        raise ValueError("wrap_offsets_by_axis must include one offset vector per torus axis")

    normalized: list[tuple[int, ...]] = []
    for axis, axis_offsets in enumerate(wrap_offsets_by_axis):
        if len(axis_offsets) != axis_count:
            raise ValueError("Each torus wrap offset vector must match the torus dimension count")
        axis_vector: list[int] = []
        for dim, raw_offset in enumerate(axis_offsets):
            if dim == axis:
                axis_vector.append(0)
                continue
            axis_vector.append(int(raw_offset) % int(shape[dim]))
        normalized.append(tuple(axis_vector))
    return tuple(normalized)


def _torus_next_coord(
    coord: tuple[int, ...],
    shape: tuple[int, ...],
    axis: int,
    wrap_offsets_by_axis: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    next_coord = list(coord)
    did_wrap = int(coord[axis]) == int(shape[axis]) - 1
    next_coord[axis] = (int(next_coord[axis]) + 1) % int(shape[axis])
    if did_wrap:
        wrap_offsets = wrap_offsets_by_axis[axis]
        for dim, offset in enumerate(wrap_offsets):
            if dim == axis or int(offset) == 0:
                continue
            next_coord[dim] = (int(next_coord[dim]) + int(offset)) % int(shape[dim])
    return tuple(next_coord)


def _build_dual_plane_torus(
    cfg: AnalysisConfig,
    *,
    topology_kind: str,
    shape: tuple[int, ...],
    wrap_offsets_by_axis: tuple[tuple[int, ...], ...] | None = None,
) -> nx.Graph:
    if topology_kind not in {"2D-Torus", "3D-Torus"}:
        raise ValueError(f"Unsupported torus topology kind: {topology_kind}")

    g = nx.Graph()
    exchanges: dict[tuple[int, ...], dict[str, list[str]]] = {}
    axis_count = len(shape)
    normalized_wrap_offsets = _normalize_torus_wrap_offsets(shape, wrap_offsets_by_axis)

    for coord in _iter_grid_coords(shape):
        exchange_id = f"en{_grid_coord_to_index(coord, shape)}"
        exchange = _add_exchange_node(g, exchange_id, cfg)
        _set_exchange_grid_coord(g, exchange, coord)
        exchanges[coord] = exchange

    for union_index in range(2):
        for coord in _iter_grid_coords(shape):
            src_union = exchanges[coord]["unions"][union_index]
            for axis in range(axis_count):
                if int(shape[axis]) <= 1:
                    continue
                dst_coord = _torus_next_coord(coord, shape, axis, normalized_wrap_offsets)
                dst_union = exchanges[dst_coord]["unions"][union_index]
                axis_wrap_offsets = normalized_wrap_offsets[axis]
                has_twisted_wrap = any(int(offset) != 0 for dim, offset in enumerate(axis_wrap_offsets) if dim != axis)
                parallel_links = 2 if int(shape[axis]) == 2 and not has_twisted_wrap else 1
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
    g.graph["torus_wrap_offsets"] = normalized_wrap_offsets
    g.graph["direct_topology_kind"] = "2D-TORUS" if topology_kind == "2D-Torus" else "3D-TORUS"
    g.graph["topology_family"] = "TORUS"
    g.graph["torus_twisted"] = any(
        int(offset) != 0
        for axis_offsets in normalized_wrap_offsets
        for offset in axis_offsets
    )
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


def _build_sparsemesh_neighbor_offsets(
    *,
    sparsity: int,
    stride_count: int,
    sparser: bool,
) -> tuple[tuple[int, ...], int]:
    ring_size = (sparsity * stride_count + sparsity) if sparser else (sparsity * stride_count + 1)
    if ring_size <= 0:
        raise ValueError("SparseMesh ring_size must be positive")

    near_count_per_direction = int(sparsity / 2) if ring_size > 1 else 0
    offsets: set[int] = set()

    for distance in range(1, near_count_per_direction + 1):
        offsets.add(int(distance))

    for stride_index in range(stride_count):
        offset = (
            sparsity * stride_index + (2 * near_count_per_direction) + 1
            if sparser
            else sparsity * stride_index + near_count_per_direction + 1
        )
        normalized_offset = int(offset) % ring_size
        if normalized_offset == 0:
            continue
        canonical_offset = min(normalized_offset, ring_size - normalized_offset)
        offsets.add(int(canonical_offset))

    normalized = tuple(sorted(offset for offset in offsets if 0 < offset <= ring_size // 2))
    return normalized, int(ring_size)


def _build_sparsemesh_variant(
    cfg: AnalysisConfig,
    *,
    topology_key: str,
    sparsity: int,
    stride_count: int,
    sparser: bool,
) -> nx.Graph:
    g = nx.Graph()
    offsets, exchange_count = _build_sparsemesh_neighbor_offsets(
        sparsity=sparsity,
        stride_count=stride_count,
        sparser=sparser,
    )
    exchanges: dict[int, dict[str, list[str]]] = {}

    for exchange_index in range(exchange_count):
        exchange_id = f"en{exchange_index}"
        exchange = _add_exchange_node(g, exchange_id, cfg)
        _set_exchange_grid_coord(g, exchange, (exchange_index,))
        exchanges[exchange_index] = exchange

        for union_index, union_id in enumerate(exchange["unions"]):
            g.nodes[union_id].update(
                sparsemesh_plane_index=union_index,
                sparsemesh_exchange_index=exchange_index,
                sparsemesh_ring_position=exchange_index,
            )
        for ssu_id in exchange["ssus"]:
            g.nodes[ssu_id].update(
                sparsemesh_exchange_index=exchange_index,
            )

    for plane_index in range(2):
        for offset in offsets:
            for exchange_index in range(exchange_count):
                dst_exchange_index = (exchange_index + int(offset)) % exchange_count
                src_union = exchanges[exchange_index]["unions"][plane_index]
                dst_union = exchanges[dst_exchange_index]["unions"][plane_index]
                if g.has_edge(src_union, dst_union):
                    continue
                _add_backend_link(
                    g,
                    src_union,
                    dst_union,
                    topology_role=f"sparsemesh_o{int(offset)}",
                )
                g.edges[src_union, dst_union]["sparsemesh_offset"] = int(offset)

    g.graph["topology_family"] = "SparseMesh"
    g.graph["sparsemesh_variant"] = topology_key
    g.graph["sparsemesh_offsets"] = tuple(int(offset) for offset in offsets)
    g.graph["sparsemesh_sparsity"] = int(sparsity)
    g.graph["sparsemesh_stride_count"] = int(stride_count)
    g.graph["sparsemesh_sparser"] = bool(sparser)
    g.graph["sparsemesh_ring_node_count"] = int(exchange_count)
    g.graph["sparsemesh_plane_count"] = 2
    g.graph["sparsemesh_exchange_count_per_plane"] = int(exchange_count)
    g.graph["sparsemesh_union_count_per_plane"] = int(exchange_count)
    g.graph["sparsemesh_ssu_count_per_plane"] = int(exchange_count) * 4
    g.graph["logical_plane_union_count"] = int(exchange_count)
    g.graph["logical_plane_ssu_count"] = int(exchange_count) * 4
    g.graph["direct_backend_mode"] = "dual_plane"
    g.graph["direct_plane_count"] = 2
    return _annotate_graph(g, cfg)


def _build_2d_fullmesh_variant(
    cfg: AnalysisConfig,
    *,
    topology_key: str,
    rows: int,
    cols: int,
) -> nx.Graph:
    g = nx.Graph()
    exchanges: dict[tuple[int, int], dict[str, list[str]]] = {}

    for r in range(rows):
        for c in range(cols):
            exchange_id = f"en{r * cols + c}"
            exchange = _add_exchange_node(g, exchange_id, cfg)
            _set_exchange_grid_coord(g, exchange, (r, c))
            exchanges[(r, c)] = exchange

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
    g.graph["fullmesh_exchange_grid_shape"] = (rows, cols)
    g.graph["direct_topology_kind"] = "2D-FULLMESH"
    g.graph["topology_family"] = "FULLMESH"
    g.graph["topology_variant"] = topology_key
    return _annotate_graph(g, cfg)


def build_2d_fullmesh(cfg: AnalysisConfig) -> nx.Graph:
    rows, cols = _FULLMESH_SHAPES["2D-FullMesh"]
    return _build_2d_fullmesh_variant(
        cfg,
        topology_key="2D-FullMesh",
        rows=int(rows),
        cols=int(cols),
    )


def build_2d_fullmesh_2x4(cfg: AnalysisConfig) -> nx.Graph:
    rows, cols = _FULLMESH_SHAPES["2D-FullMesh-2x4"]
    return _build_2d_fullmesh_variant(
        cfg,
        topology_key="2D-FullMesh-2x4",
        rows=int(rows),
        cols=int(cols),
    )


def build_2d_torus(cfg: AnalysisConfig) -> nx.Graph:
    return build_twisted_torus(cfg, "2D-Torus")


def build_3d_torus(cfg: AnalysisConfig) -> nx.Graph:
    return build_twisted_torus(cfg, "3D-Torus")


def torus_shape(topology_name: str) -> tuple[int, ...]:
    shape_name = _torus_shape_variant_name(topology_name)
    if shape_name is None:
        valid = ", ".join(sorted(_TORUS_VARIANT_SHAPE_NAMES))
        raise ValueError(f"Unsupported torus topology '{topology_name}'. Valid: {valid}")

    return tuple(int(value) for value in _TORUS_SHAPES[shape_name])


def build_twisted_torus(
    cfg: AnalysisConfig,
    topology_name: str,
    *,
    wrap_offsets_by_axis: tuple[tuple[int, ...], ...] | None = None,
) -> nx.Graph:
    shape = torus_shape(topology_name)
    family_name = torus_base_name(topology_name)
    if family_name is None:
        valid = ", ".join(sorted(_TORUS_VARIANT_FAMILY_NAMES))
        raise ValueError(f"Unsupported torus topology '{topology_name}'. Valid: {valid}")
    g = _build_dual_plane_torus(
        cfg,
        topology_kind=family_name,
        shape=shape,
        wrap_offsets_by_axis=wrap_offsets_by_axis,
    )
    g.graph["topology_name"] = family_name
    g.graph["torus_shape_variant"] = _torus_shape_variant_name(topology_name)
    _validate_backend_uniformity(g, family_name)
    return g


def build_2d_torus_best_twist(cfg: AnalysisConfig) -> nx.Graph:
    g = build_twisted_torus(
        cfg,
        "2D-Torus",
        wrap_offsets_by_axis=_TORUS_BEST_TWIST_OFFSETS["2D-Torus-BestTwist"],
    )
    g.graph["torus_twist_label"] = "axis0=[0, 2] | axis1=[0, 0]"
    return g


def build_3d_torus_best_twist(cfg: AnalysisConfig) -> nx.Graph:
    g = build_twisted_torus(
        cfg,
        "3D-Torus",
        wrap_offsets_by_axis=_TORUS_BEST_TWIST_OFFSETS["3D-Torus-BestTwist"],
    )
    g.graph["torus_twist_label"] = "axis0=[0, 2, 2] | axis1=[0, 0, 0] | axis2=[0, 0, 0]"
    return g


def build_3d_torus_2x4x2(cfg: AnalysisConfig) -> nx.Graph:
    return build_twisted_torus(cfg, "3D-Torus-2x4x2")


def build_3d_torus_2x4x3(cfg: AnalysisConfig) -> nx.Graph:
    return build_twisted_torus(cfg, "3D-Torus-2x4x3")


def build_3d_torus_2x4x3_best_twist(cfg: AnalysisConfig) -> nx.Graph:
    g = build_twisted_torus(
        cfg,
        "3D-Torus-2x4x3",
        wrap_offsets_by_axis=_TORUS_BEST_TWIST_OFFSETS["3D-Torus-2x4x3-BestTwist"],
    )
    g.graph["torus_twist_label"] = "axis0=[0, 2, 0] | axis1=[0, 0, 0] | axis2=[0, 0, 0]"
    return g


def build_3d_torus_2x4x2_best_twist(cfg: AnalysisConfig) -> nx.Graph:
    g = build_twisted_torus(
        cfg,
        "3D-Torus-2x4x2",
        wrap_offsets_by_axis=_TORUS_BEST_TWIST_OFFSETS["3D-Torus-2x4x2-BestTwist"],
    )
    g.graph["torus_twist_label"] = "axis0=[0, 0, 1] | axis1=[1, 0, 1] | axis2=[0, 2, 0]"
    return g


def build_3d_torus_2x4x1(cfg: AnalysisConfig) -> nx.Graph:
    return build_twisted_torus(cfg, "3D-Torus-2x4x1")


def build_3d_torus_2x4x1_best_twist(cfg: AnalysisConfig) -> nx.Graph:
    g = build_twisted_torus(
        cfg,
        "3D-Torus-2x4x1",
        wrap_offsets_by_axis=_TORUS_BEST_TWIST_OFFSETS["3D-Torus-2x4x1-BestTwist"],
    )
    g.graph["torus_twist_label"] = "axis0=[0, 2, 0] | axis1=[0, 0, 0] | axis2=[0, 0, 0]"
    return g


def build_clos(cfg: AnalysisConfig) -> nx.Graph:
    return _build_clos_variant(
        cfg,
        topology_key="Clos",
        exchange_count=_CLOS_EXCHANGE_NODE_COUNT,
    )


def build_clos_64(cfg: AnalysisConfig) -> nx.Graph:
    return _build_clos_variant(
        cfg,
        topology_key="Clos-64",
        exchange_count=int(_CLOS_SCALED_EXCHANGE_COUNTS["Clos-64"]),
    )


def build_clos_128(cfg: AnalysisConfig) -> nx.Graph:
    return _build_clos_variant(
        cfg,
        topology_key="Clos-128",
        exchange_count=int(_CLOS_SCALED_EXCHANGE_COUNTS["Clos-128"]),
    )


def build_clos_192(cfg: AnalysisConfig) -> nx.Graph:
    return _build_clos_variant(
        cfg,
        topology_key="Clos-192",
        exchange_count=int(_CLOS_SCALED_EXCHANGE_COUNTS["Clos-192"]),
    )


def build_clos_256(cfg: AnalysisConfig) -> nx.Graph:
    return _build_clos_variant(
        cfg,
        topology_key="Clos-256",
        exchange_count=int(_CLOS_SCALED_EXCHANGE_COUNTS["Clos-256"]),
    )


def _build_clos_variant(
    cfg: AnalysisConfig,
    *,
    topology_key: str,
    exchange_count: int,
) -> nx.Graph:
    _validate_clos_uplink_budget(cfg)

    g = nx.Graph()
    exchange_nodes = [_add_exchange_node(g, f"en{idx}", cfg) for idx in range(exchange_count)]

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

    grid_cols, grid_rows = _clos_exchange_grid_shape(exchange_count)

    _validate_clos_spine_fanout(g, exchange_count)
    g.graph["topology_family"] = "CLOS"
    g.graph["clos_variant"] = topology_key
    g.graph["clos_plane_count"] = 2
    g.graph["clos_exchange_count"] = exchange_count
    g.graph["clos_exchange_grid_shape"] = (grid_cols, grid_rows)
    g.graph["clos_switch_count_per_plane"] = int(cfg.clos_uplinks_per_exchange_node)
    g.graph["clos_total_switch_count"] = int(cfg.clos_uplinks_per_exchange_node) * 2
    g.graph["clos_switch_role"] = "clos_spine"
    g.graph["clos_local_group_size"] = 1
    g.graph["clos_local_group_count_per_plane"] = exchange_count
    g.graph["clos_uplink_bandwidth_gbps"] = _BACKEND_BW_GBPS
    g.graph["exchange_projection_fast_path"] = True
    return _annotate_graph(g, cfg)


def _build_clos_4p_leaf_variant(
    cfg: AnalysisConfig,
    *,
    topology_key: str,
    local_topology: str,
    uplink_bandwidth_gbps: float,
    uplink_parallel_links: int = 1,
) -> nx.Graph:
    g = nx.Graph()
    exchange_nodes = [
        _add_exchange_node(g, f"en{idx}", cfg)
        for idx in range(_CLOS_4P_FULLMESH_EXCHANGE_NODE_COUNT)
    ]

    plane_leaf_ids: dict[int, list[str]] = {}
    for plane_index in range(_CLOS_4P_FULLMESH_PLANES):
        leaf_ids = [
            f"clos_leaf_plane{plane_index}_uplink{leaf_index}"
            for leaf_index in range(_CLOS_4P_FULLMESH_LEAFS_PER_PLANE)
        ]
        plane_leaf_ids[plane_index] = leaf_ids
        for leaf_id in leaf_ids:
            g.add_node(
                leaf_id,
                node_type="switch",
                node_role="clos_leaf",
                union_plane=plane_index,
            )

    for exchange_index, exchange in enumerate(exchange_nodes):
        local_group_id = exchange_index // _CLOS_4P_FULLMESH_GROUP_SIZE
        local_group_slot = exchange_index % _CLOS_4P_FULLMESH_GROUP_SIZE
        for plane_index, union_id in enumerate(exchange["unions"]):
            g.nodes[union_id].update(
                clos_plane_index=plane_index,
                clos_local_group_id=local_group_id,
                clos_local_group_slot=local_group_slot,
            )

    for plane_index in range(_CLOS_4P_FULLMESH_PLANES):
        for group_index in range(_CLOS_4P_FULLMESH_GROUPS_PER_PLANE):
            group_start = group_index * _CLOS_4P_FULLMESH_GROUP_SIZE
            union_ids = [
                exchange_nodes[group_start + offset]["unions"][plane_index]
                for offset in range(_CLOS_4P_FULLMESH_GROUP_SIZE)
            ]
            if local_topology == "fullmesh":
                _add_df_server_fullmesh_links(g, union_ids)
                for src_index, src_union_id in enumerate(union_ids):
                    for dst_union_id in union_ids[src_index + 1 :]:
                        g.edges[src_union_id, dst_union_id]["topology_role"] = "clos4p_local_fullmesh"
            elif local_topology == "ring":
                _add_df_server_ring_links(g, union_ids)
                for src_index in range(len(union_ids)):
                    dst_index = (src_index + 1) % len(union_ids)
                    if src_index < dst_index or dst_index == 0:
                        g.edges[union_ids[src_index], union_ids[dst_index]]["topology_role"] = "clos4p_local_ring"
            else:
                raise ValueError(f"Unsupported Clos 4P local topology: {local_topology}")

        for exchange in exchange_nodes:
            union_id = exchange["unions"][plane_index]
            for leaf_id in plane_leaf_ids[plane_index]:
                _add_backend_link(
                    g,
                    union_id,
                    leaf_id,
                    topology_role="clos4p_leaf_uplink",
                    bandwidth_gbps=uplink_bandwidth_gbps,
                    parallel_links=uplink_parallel_links,
                )

    _validate_clos_leaf_fanout(g)
    g.graph["topology_family"] = "CLOS"
    g.graph["clos_variant"] = topology_key
    g.graph["clos_plane_count"] = _CLOS_4P_FULLMESH_PLANES
    g.graph["clos_exchange_count"] = _CLOS_4P_FULLMESH_EXCHANGE_NODE_COUNT
    g.graph["clos_switch_count_per_plane"] = _CLOS_4P_FULLMESH_LEAFS_PER_PLANE
    g.graph["clos_total_switch_count"] = _CLOS_4P_FULLMESH_PLANES * _CLOS_4P_FULLMESH_LEAFS_PER_PLANE
    g.graph["clos_switch_role"] = "clos_leaf"
    g.graph["clos_local_topology"] = local_topology
    g.graph["clos_local_group_size"] = _CLOS_4P_FULLMESH_GROUP_SIZE
    g.graph["clos_local_group_count_per_plane"] = _CLOS_4P_FULLMESH_GROUPS_PER_PLANE
    g.graph["clos_local_bandwidth_gbps"] = _BACKEND_BW_GBPS
    g.graph["clos_uplink_bandwidth_gbps"] = uplink_bandwidth_gbps
    g.graph["clos_uplink_parallel_links"] = uplink_parallel_links
    g.graph["exchange_projection_fast_path"] = True
    return _annotate_graph(g, cfg)


def build_clos_4p_fullmesh(cfg: AnalysisConfig) -> nx.Graph:
    return _build_clos_4p_leaf_variant(
        cfg,
        topology_key="Clos-4P-FullMesh",
        local_topology="fullmesh",
        uplink_bandwidth_gbps=_CLOS_4P_FULLMESH_LEAF_UPLINK_BW_GBPS,
    )


def build_clos_4p_ring(cfg: AnalysisConfig) -> nx.Graph:
    return _build_clos_4p_leaf_variant(
        cfg,
        topology_key="Clos-4P-Ring",
        local_topology="ring",
        uplink_bandwidth_gbps=_CLOS_4P_RING_LEAF_UPLINK_BW_GBPS,
        uplink_parallel_links=_CLOS_4P_RING_LEAF_UPLINK_PARALLEL_LINKS,
    )


def build_df(cfg: AnalysisConfig) -> nx.Graph:
    return _build_df_variant(
        cfg,
        topology_key="DF",
        local_topology="fullmesh",
        global_pattern="contiguous",
        external_servers_per_union=int(cfg.df_external_servers_per_union),
        plane_count=2,
    )


def build_df_3local_2global(cfg: AnalysisConfig) -> nx.Graph:
    return _build_df_variant(
        cfg,
        topology_key="DF-3Local-2Global",
        local_topology="fullmesh",
        global_pattern="contiguous",
        external_servers_per_union=2,
        plane_count=2,
    )


def build_df_3local_1global(cfg: AnalysisConfig) -> nx.Graph:
    return _build_df_variant(
        cfg,
        topology_key="DF-3Local-1Global",
        local_topology="fullmesh",
        global_pattern="contiguous",
        external_servers_per_union=1,
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


def build_sparsemesh_local(cfg: AnalysisConfig) -> nx.Graph:
    return _build_sparsemesh_variant(
        cfg,
        topology_key="SparseMesh-Local",
        sparsity=int(_SPARSEMESH_VARIANTS["SparseMesh-Local"]["sparsity"]),
        stride_count=int(_SPARSEMESH_VARIANTS["SparseMesh-Local"]["stride_count"]),
        sparser=bool(_SPARSEMESH_VARIANTS["SparseMesh-Local"]["sparser"]),
    )


def build_sparsemesh_global(cfg: AnalysisConfig) -> nx.Graph:
    return _build_sparsemesh_variant(
        cfg,
        topology_key="SparseMesh-Global",
        sparsity=int(_SPARSEMESH_VARIANTS["SparseMesh-Global"]["sparsity"]),
        stride_count=int(_SPARSEMESH_VARIANTS["SparseMesh-Global"]["stride_count"]),
        sparser=bool(_SPARSEMESH_VARIANTS["SparseMesh-Global"]["sparser"]),
    )


BUILDERS: dict[str, TopologyBuilder] = {
    "2D-FullMesh": build_2d_fullmesh,
    "2D-FullMesh-2x4": build_2d_fullmesh_2x4,
    "2D-Torus": build_2d_torus,
    "2D-Torus-BestTwist": build_2d_torus_best_twist,
    "3D-Torus": build_3d_torus,
    "3D-Torus-BestTwist": build_3d_torus_best_twist,
    "3D-Torus-2x4x3": build_3d_torus_2x4x3,
    "3D-Torus-2x4x3-BestTwist": build_3d_torus_2x4x3_best_twist,
    "3D-Torus-2x4x2": build_3d_torus_2x4x2,
    "3D-Torus-2x4x2-BestTwist": build_3d_torus_2x4x2_best_twist,
    "3D-Torus-2x4x1": build_3d_torus_2x4x1,
    "3D-Torus-2x4x1-BestTwist": build_3d_torus_2x4x1_best_twist,
    "Clos": build_clos,
    "Clos-64": build_clos_64,
    "Clos-128": build_clos_128,
    "Clos-192": build_clos_192,
    "Clos-256": build_clos_256,
    "Clos-4P-FullMesh": build_clos_4p_fullmesh,
    "Clos-4P-Ring": build_clos_4p_ring,
    "DF": build_df,
    "DF-3Local-2Global": build_df_3local_2global,
    "DF-3Local-1Global": build_df_3local_1global,
    "SparseMesh-Local": build_sparsemesh_local,
    "SparseMesh-Global": build_sparsemesh_global,
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
