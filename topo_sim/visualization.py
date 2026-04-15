from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .labels import display_topology_name, display_workload_name
from .topologies import (
    fullmesh_shape,
    is_fullmesh_topology_name,
    is_sparsemesh_topology_name,
    is_torus_topology_name,
    torus_base_name,
)


_INTERNAL_EDGE_COLOR = "rgba(92, 108, 136, 0.22)"
_BACKEND_EDGE_COLOR = "rgba(86, 166, 255, 0.30)"
_HIGHLIGHT_EDGE_COLOR = "rgba(255, 198, 112, 0.96)"
_DEFAULT_NODE_COLOR = "#8b97aa"
_UNION_NODE_COLOR = "#ffbe5c"
_SSU_NODE_COLOR = "#34d399"
_SPINE_NODE_COLOR = "#ff8e5a"
_TRAFFIC_IDLE_SSU_COLOR = "#667085"
_TRAFFIC_SOURCE_COLOR = "#7dd3fc"
_TRAFFIC_DESTINATION_COLOR = "#22c55e"
_TRAFFIC_DESTINATION_RING_COLOR = "#facc15"
_PLOT_BG_COLOR = "#000000"


def _is_df_name(topology_name: str) -> bool:
    upper = str(topology_name).strip().upper()
    return upper == "DF" or upper.startswith("DF-")


def _is_torus_name(topology_name: str) -> bool:
    return is_torus_topology_name(topology_name)


def _torus_family_name(topology_name: str) -> str | None:
    return torus_base_name(topology_name)


def _is_sparsemesh_name(topology_name: str) -> bool:
    return is_sparsemesh_topology_name(topology_name)


def _fallback_positions(g: nx.Graph) -> dict[Any, tuple[float, float]]:
    circular = nx.circular_layout(g)
    return {k: (float(v[0]), float(v[1])) for k, v in circular.items()}


def _exchange_local_positions(base_x: float, base_y: float, exchange_node_id: str) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    ssu_spacing = 0.62
    union_spacing = 1.2

    for ssu_index in range(8):
        positions[f"{exchange_node_id}:ssu{ssu_index}"] = (base_x + (ssu_index * ssu_spacing), base_y)

    union_center = base_x + 3.5 * ssu_spacing
    positions[f"{exchange_node_id}:union0"] = (union_center - (union_spacing / 2.0), base_y + 1.15)
    positions[f"{exchange_node_id}:union1"] = (union_center + (union_spacing / 2.0), base_y + 1.15)
    return positions


def _torus_exchange_local_positions(base_x: float, base_y: float, exchange_node_id: str) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    ssu_spacing = 0.62
    union_spacing = 1.36

    for ssu_index in range(8):
        positions[f"{exchange_node_id}:ssu{ssu_index}"] = (base_x + (ssu_index * ssu_spacing), base_y)

    union_center = base_x + 3.5 * ssu_spacing
    positions[f"{exchange_node_id}:union0"] = (
        union_center - (union_spacing / 2.0) - 0.12,
        base_y + 1.02,
    )
    positions[f"{exchange_node_id}:union1"] = (
        union_center + (union_spacing / 2.0) + 0.12,
        base_y + 1.34,
    )
    return positions


def _oriented_exchange_local_positions(
    center_x: float,
    center_y: float,
    exchange_node_id: str,
    tangent: tuple[float, float],
    inward: tuple[float, float],
    reverse_union_order: bool = False,
) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    ssu_spacing = 0.62
    union_spacing = 1.2

    tangent_x, tangent_y = tangent
    inward_x, inward_y = inward

    def _project(local_x: float, local_y: float) -> tuple[float, float]:
        return (
            center_x + (local_x * tangent_x) + (local_y * inward_x),
            center_y + (local_x * tangent_y) + (local_y * inward_y),
        )

    for ssu_index in range(8):
        local_x = (ssu_index - 3.5) * ssu_spacing
        positions[f"{exchange_node_id}:ssu{ssu_index}"] = _project(local_x, 0.0)

    if reverse_union_order:
        positions[f"{exchange_node_id}:union0"] = _project(union_spacing / 2.0, 1.15)
        positions[f"{exchange_node_id}:union1"] = _project(-(union_spacing / 2.0), 1.15)
    else:
        positions[f"{exchange_node_id}:union0"] = _project(-(union_spacing / 2.0), 1.15)
        positions[f"{exchange_node_id}:union1"] = _project(union_spacing / 2.0, 1.15)
    return positions


def _oriented_df_exchange_local_positions(
    center_x: float,
    center_y: float,
    exchange_node_id: str,
    tangent: tuple[float, float],
    inward: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    ssu_spacing = 0.62
    tangent_x, tangent_y = tangent
    inward_x, inward_y = inward

    def _project(local_x: float, local_y: float) -> tuple[float, float]:
        return (
            center_x + (local_x * tangent_x) + (local_y * inward_x),
            center_y + (local_x * tangent_y) + (local_y * inward_y),
        )

    for ssu_index in range(8):
        local_x = (ssu_index - 3.5) * ssu_spacing
        positions[f"{exchange_node_id}:ssu{ssu_index}"] = _project(local_x, 0.0)

    # Two Union chips belong to different DF planes but share the same 8-SSU group.
    # Render them on two staggered inner rings so the front and rear DF circles stay visible.
    positions[f"{exchange_node_id}:union0"] = _project(-0.28, 1.08)
    positions[f"{exchange_node_id}:union1"] = _project(0.28, 1.86)
    return positions


def _exchange_grid_positions_2d(rows: int, cols: int) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    cell_x = 7.2
    cell_y = 3.8

    for row in range(rows):
        for col in range(cols):
            exchange_id = f"en{row * cols + col}"
            base_x = col * cell_x
            base_y = -row * cell_y
            positions.update(_exchange_local_positions(base_x, base_y, exchange_id))

    return positions


def _torus_exchange_coords(
    g: nx.Graph,
) -> dict[str, tuple[int, ...]]:
    coords: dict[str, tuple[int, ...]] = {}
    for node_id, data in g.nodes(data=True):
        if data.get("node_role") != "union":
            continue
        exchange_id = data.get("exchange_node_id")
        coord = data.get("exchange_grid_coord")
        if exchange_id is None or coord is None:
            continue
        coords[str(exchange_id)] = tuple(int(value) for value in coord)
    return coords


def _exchange_grid_positions_2d_torus(g: nx.Graph) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    exchange_coords = _torus_exchange_coords(g)
    if not exchange_coords:
        exchange_coords = {
            f"en{row * 4 + col}": (row, col)
            for row in range(4)
            for col in range(4)
        }
    cell_x = 7.2
    cell_y = 3.8

    for exchange_id, (row, col) in sorted(exchange_coords.items(), key=lambda item: int(item[0].removeprefix("en"))):
        base_x = col * cell_x
        base_y = -row * cell_y
        positions.update(_torus_exchange_local_positions(base_x, base_y, exchange_id))

    return positions


def _exchange_grid_positions_3d() -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    cell_x = 6.6
    cell_y = 3.7
    block_gap_x = 31.0
    block_gap_y = 18.0
    size = 4

    for x in range(size):
        for y in range(size):
            for z in range(size):
                exchange_index = (x * size * size) + (y * size) + z
                exchange_id = f"en{exchange_index}"
                block_col = z % 2
                block_row = z // 2
                base_x = (block_col * block_gap_x) + (y * cell_x)
                base_y = -(block_row * block_gap_y) - (x * cell_y)
                positions.update(_exchange_local_positions(base_x, base_y, exchange_id))

    return positions


def _exchange_grid_positions_3d_torus(g: nx.Graph) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    exchange_coords = _torus_exchange_coords(g)
    if not exchange_coords:
        exchange_coords = {
            f"en{(x * 16) + (y * 4) + z}": (x, y, z)
            for x in range(4)
            for y in range(4)
            for z in range(4)
        }
    cell_x = 6.6
    cell_y = 3.7
    block_gap_x = 31.0
    block_gap_y = 18.0

    z_levels = sorted({coord[2] for coord in exchange_coords.values()})
    z_index_map = {z_level: index for index, z_level in enumerate(z_levels)}
    blocks_per_row = 2 if len(z_levels) > 1 else 1

    for exchange_id, coord in sorted(exchange_coords.items(), key=lambda item: int(item[0].removeprefix("en"))):
        x, y, z = coord
        z_index = z_index_map[z]
        block_col = z_index % blocks_per_row
        block_row = z_index // blocks_per_row
        base_x = (block_col * block_gap_x) + (y * cell_x)
        base_y = -(block_row * block_gap_y) - (x * cell_y)
        positions.update(_torus_exchange_local_positions(base_x, base_y, exchange_id))

    return positions


def _clos_spine_sort_key(node_id: str) -> tuple[int, int]:
    parts = str(node_id).split("_")
    plane = next((int(part.removeprefix("plane")) for part in parts if part.startswith("plane")), 0)
    uplink = next((int(part.removeprefix("uplink")) for part in parts if part.startswith("uplink")), 0)
    return plane, uplink


def _exchange_grid_positions_clos(g: nx.Graph) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    cols = 6
    cell_x = 7.4
    cell_y = 3.9

    for index in range(18):
        row = index // cols
        col = index % cols
        exchange_id = f"en{index}"
        base_x = col * cell_x
        base_y = -(row * cell_y) - 2.8
        positions.update(_exchange_local_positions(base_x, base_y, exchange_id))

    spine_nodes = sorted(
        [node_id for node_id, data in g.nodes(data=True) if data.get("node_role") == "clos_spine"],
        key=_clos_spine_sort_key,
    )
    plane_spacing_y = 2.0
    spine_spacing_x = 8.2
    spine_origin_x = 8.5
    spine_origin_y = 3.2
    for node_id in spine_nodes:
        plane, uplink = _clos_spine_sort_key(node_id)
        positions[node_id] = (
            spine_origin_x + (uplink * spine_spacing_x),
            spine_origin_y + (plane * plane_spacing_y),
        )

    return positions


def _exchange_grid_positions_df(g: nx.Graph) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    exchanges_by_server: dict[int, list[tuple[int, str]]] = {}
    for node_id, data in g.nodes(data=True):
        if data.get("node_role") != "union":
            continue
        exchange_id = data.get("exchange_node_id")
        server_id = data.get("server_id")
        group_local_index = data.get("df_group_local_index")
        if exchange_id is None or server_id is None or group_local_index is None:
            continue
        server_exchanges = exchanges_by_server.setdefault(int(server_id), [])
        exchange_pair = (int(group_local_index), str(exchange_id))
        if exchange_pair not in server_exchanges:
            server_exchanges.append(exchange_pair)

    ordered_servers = sorted(exchanges_by_server)
    if not ordered_servers:
        return positions

    exchange_sequence: list[str] = []
    group_breaks: list[int] = []
    for server_id in ordered_servers:
        ordered_exchanges = [
            exchange_id
            for _, exchange_id in sorted(
                exchanges_by_server[server_id],
                key=lambda item: item[0],
                reverse=True,
            )
        ]
        exchange_sequence.extend(ordered_exchanges)
        group_breaks.append(len(exchange_sequence))

    server_gap_units = 1.24
    slot_count = len(exchange_sequence) + (len(ordered_servers) * server_gap_units)
    radius = max(42.0, slot_count * 1.48)

    slot_cursor = 0.0
    for exchange_index, exchange_id in enumerate(exchange_sequence):
        angle = -2.0 * math.pi * (slot_cursor / slot_count)
        radial = (math.cos(angle), math.sin(angle))
        tangent = (-radial[1], radial[0])
        inward = (-radial[0], -radial[1])
        center_x = radius * radial[0]
        center_y = radius * radial[1]
        positions.update(
            _oriented_df_exchange_local_positions(
                center_x=center_x,
                center_y=center_y,
                exchange_node_id=exchange_id,
                tangent=tangent,
                inward=inward,
            )
        )

        slot_cursor += 1.0
        if exchange_index + 1 in group_breaks:
            slot_cursor += server_gap_units

    return positions


def _exchange_grid_positions_sparsemesh(g: nx.Graph) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    exchange_ids = sorted(
        {
            str(data.get("exchange_node_id"))
            for _, data in g.nodes(data=True)
            if data.get("exchange_node_id") is not None
        },
        key=lambda value: int(value.removeprefix("en")),
    )
    exchange_count = len(exchange_ids)
    if exchange_count == 0:
        return positions

    radius = max(40.0, exchange_count * 1.08)
    for index, exchange_id in enumerate(exchange_ids):
        angle = -2.0 * math.pi * (index / float(exchange_count))
        radial = (math.cos(angle), math.sin(angle))
        tangent = (-radial[1], radial[0])
        inward = (-radial[0], -radial[1])
        center_x = radius * radial[0]
        center_y = radius * radial[1]
        positions.update(
            _oriented_df_exchange_local_positions(
                center_x=center_x,
                center_y=center_y,
                exchange_node_id=exchange_id,
                tangent=tangent,
                inward=inward,
            )
        )

    return positions


def _explicit_positions(topology_name: str, g: nx.Graph) -> dict[Any, tuple[float, float]] | None:
    if is_fullmesh_topology_name(topology_name):
        shape = tuple(int(value) for value in g.graph.get("fullmesh_exchange_grid_shape", fullmesh_shape(topology_name)))
        return _exchange_grid_positions_2d(int(shape[0]), int(shape[1]))
    if _torus_family_name(topology_name) == "2D-Torus":
        return _exchange_grid_positions_2d_torus(g)
    if _torus_family_name(topology_name) == "3D-Torus":
        return _exchange_grid_positions_3d_torus(g)
    if topology_name == "Clos":
        return _exchange_grid_positions_clos(g)
    if _is_df_name(topology_name):
        return _exchange_grid_positions_df(g)
    if _is_sparsemesh_name(topology_name):
        return _exchange_grid_positions_sparsemesh(g)
    return None


def _positions(g: nx.Graph, topology_name: str, seed: int) -> dict[Any, tuple[float, float]]:
    existing = nx.get_node_attributes(g, "pos")
    if existing:
        return {k: (float(v[0]), float(v[1])) for k, v in existing.items()}

    explicit = _explicit_positions(topology_name, g)
    if explicit is not None:
        return explicit

    try:
        spring = nx.spring_layout(g, seed=seed)
    except (ImportError, ModuleNotFoundError):
        return _fallback_positions(g)
    return {k: (float(v[0]), float(v[1])) for k, v in spring.items()}


def _node_color(node_data: dict[str, Any]) -> str:
    role = node_data.get("node_role")
    if role == "ssu":
        return _SSU_NODE_COLOR
    if role == "union":
        return _UNION_NODE_COLOR
    if role == "clos_spine":
        return _SPINE_NODE_COLOR
    return _DEFAULT_NODE_COLOR


def _df_union_label(node_data: dict[str, Any]) -> str | None:
    local_union_index = node_data.get("server_local_union_index")
    if local_union_index is None:
        return None
    return str(int(local_union_index))


def _df_plane_indices(g: nx.Graph) -> list[int]:
    planes = sorted(
        {
            int(data.get("df_plane_index"))
            for _, data in g.nodes(data=True)
            if data.get("df_plane_index") is not None
        }
    )
    return planes or [0]


def _df_front_plane_index(g: nx.Graph) -> int:
    return _df_plane_indices(g)[0]


def _df_plane_draw_order(g: nx.Graph) -> dict[int, int]:
    back_to_front = list(reversed(_df_plane_indices(g)))
    return {plane_index: order for order, plane_index in enumerate(back_to_front)}


def _df_node_draw_rank(g: nx.Graph, node_id: str) -> int:
    plane_index = g.nodes[node_id].get("df_plane_index")
    if plane_index is None:
        return len(_df_plane_draw_order(g))
    return _df_plane_draw_order(g).get(int(plane_index), 0)


def _df_plane_node_opacity(g: nx.Graph, node_data: dict[str, Any]) -> float:
    plane_index = node_data.get("df_plane_index")
    if plane_index is None:
        return 1.0
    draw_order = _df_plane_draw_order(g)
    plane_count = max(len(draw_order), 1)
    if plane_count <= 1:
        return 1.0
    frontness = draw_order.get(int(plane_index), plane_count - 1) / float(plane_count - 1)
    return 0.42 + (0.58 * frontness)


def _df_server_label_positions(
    g: nx.Graph,
    pos: dict[Any, tuple[float, float]],
    *,
    plane_index: int | None = None,
) -> tuple[list[float], list[float], list[str]]:
    server_union_points: dict[int, list[tuple[float, float]]] = {}
    for node_id, node_data in g.nodes(data=True):
        if node_data.get("node_role") != "union":
            continue
        if plane_index is not None and int(node_data.get("df_plane_index", 0)) != int(plane_index):
            continue
        server_id = node_data.get("server_id")
        if server_id is None:
            continue
        server_union_points.setdefault(int(server_id), []).append(pos[node_id])

    label_x: list[float] = []
    label_y: list[float] = []
    label_text: list[str] = []
    for server_id in sorted(server_union_points):
        coords = server_union_points[server_id]
        if not coords:
            continue
        center_x = sum(x for x, _ in coords) / float(len(coords))
        center_y = sum(y for _, y in coords) / float(len(coords))
        norm = math.hypot(center_x, center_y)
        if norm <= 1e-9:
            outward_x = 0.0
            outward_y = 3.6
        else:
            outward_scale = 3.6 / norm
            outward_x = center_x * outward_scale
            outward_y = center_y * outward_scale
        label_x.append(center_x + outward_x)
        label_y.append(center_y + outward_y)
        label_text.append(f"server{server_id}")

    return label_x, label_y, label_text


def _edge_curve_factor(
    g: nx.Graph,
    topology_name: str,
    u: str,
    v: str,
    edge_data: dict[str, Any],
) -> float:
    if edge_data.get("link_kind") != "backend_interconnect":
        return 0.0

    topology_role = str(edge_data.get("topology_role", ""))
    if is_fullmesh_topology_name(topology_name) and topology_role in {"2d_fullmesh_x", "2d_fullmesh_y"}:
        union_plane = int(g.nodes[u].get("local_index", 0))
        plane_sign = -1.0 if union_plane == 0 else 1.0
        axis_sign = 1.0 if topology_role.endswith("_x") else -1.0
        return 0.18 * plane_sign * axis_sign

    if _is_df_name(topology_name) and topology_role in {"df_server_fullmesh", "df_server_ring"}:
        src_local = int(g.nodes[u].get("server_local_union_index", 0))
        dst_local = int(g.nodes[v].get("server_local_union_index", 0))
        gap = abs(src_local - dst_local)
        parity_sign = -1.0 if ((src_local + dst_local) % 2 == 0) else 1.0
        curve = min(0.34, 0.16 + (0.05 * gap))
        if topology_role == "df_server_ring":
            curve = min(0.22, 0.10 + (0.03 * gap))
        return parity_sign * curve
    if _is_df_name(topology_name) and topology_role == "df_server_bridge":
        src_local = int(g.nodes[u].get("server_local_union_index", 0))
        dst_local = int(g.nodes[v].get("server_local_union_index", 0))
        parity_sign = -1.0 if ((src_local + dst_local) % 2 == 0) else 1.0
        return 0.12 * parity_sign

    if _torus_family_name(topology_name) == "2D-Torus" and topology_role in {"2d_torus_x", "2d_torus_y"}:
        union_plane = int(g.nodes[u].get("local_index", 0))
        plane_sign = -1.0 if union_plane == 0 else 1.0
        axis_curve = 0.12 if topology_role.endswith("_x") else -0.12
        return plane_sign * axis_curve
    if _torus_family_name(topology_name) == "2D-Torus" and topology_role == "2d_torus_local":
        return 0.14

    if _torus_family_name(topology_name) == "3D-Torus" and topology_role in {"3d_torus_x", "3d_torus_y", "3d_torus_z"}:
        union_plane = int(g.nodes[u].get("local_index", 0))
        plane_sign = -1.0 if union_plane == 0 else 1.0
        axis_curve = {
            "3d_torus_x": 0.08,
            "3d_torus_y": -0.08,
            "3d_torus_z": 0.15,
        }[topology_role]
        return plane_sign * axis_curve
    if _torus_family_name(topology_name) == "3D-Torus" and topology_role == "3d_torus_local":
        return 0.12

    if _is_sparsemesh_name(topology_name) and topology_role.startswith("sparsemesh_o"):
        plane_index = int(g.nodes[u].get("sparsemesh_plane_index", g.nodes[u].get("local_index", 0)))
        plane_sign = -1.0 if plane_index == 0 else 1.0
        offset = int(edge_data.get("sparsemesh_offset", 1))
        curve = {
            1: 0.08,
            2: 0.14,
            3: 0.20,
            5: 0.28,
        }.get(offset, 0.10 + (0.03 * min(offset, 6)))
        return plane_sign * curve

    return 0.0


def _edge_path_points(
    g: nx.Graph,
    pos: dict[Any, tuple[float, float]],
    topology_name: str,
    u: str,
    v: str,
    edge_data: dict[str, Any],
    *,
    point_count: int = 13,
    curve_bias: float = 0.0,
) -> list[tuple[float, float]]:
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    curve_factor = _edge_curve_factor(g, topology_name, u, v, edge_data) + float(curve_bias)
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if point_count <= 2 or length <= 1e-9:
        return [(x0, y0), (x1, y1)]
    if abs(curve_factor) <= 1e-9:
        return [
            (
                x0 + (dx * (index / float(point_count - 1))),
                y0 + (dy * (index / float(point_count - 1))),
            )
            for index in range(point_count)
        ]

    perp_x = -dy / length
    perp_y = dx / length
    mid_x = (x0 + x1) / 2.0
    mid_y = (y0 + y1) / 2.0
    control_x = mid_x + (perp_x * length * curve_factor)
    control_y = mid_y + (perp_y * length * curve_factor)

    points: list[tuple[float, float]] = []
    for index in range(point_count):
        t = index / float(max(point_count - 1, 1))
        omt = 1.0 - t
        points.append(
            (
                (omt * omt * x0) + (2.0 * omt * t * control_x) + (t * t * x1),
                (omt * omt * y0) + (2.0 * omt * t * control_y) + (t * t * y1),
            )
        )
    return points


def _parallel_edge_curve_biases(edge_data: dict[str, Any]) -> list[float]:
    parallel_links = max(1, int(edge_data.get("parallel_links", 1)))
    if parallel_links <= 1:
        return [0.0]

    step = 0.085
    midpoint = (parallel_links - 1) / 2.0
    return [(index - midpoint) * step for index in range(parallel_links)]


def _parallel_edge_paths(
    g: nx.Graph,
    pos: dict[Any, tuple[float, float]],
    topology_name: str,
    u: str,
    v: str,
    edge_data: dict[str, Any],
    *,
    point_count: int = 13,
) -> list[list[tuple[float, float]]]:
    return [
        _edge_path_points(
            g,
            pos,
            topology_name,
            u,
            v,
            edge_data,
            point_count=point_count,
            curve_bias=curve_bias,
        )
        for curve_bias in _parallel_edge_curve_biases(edge_data)
    ]


def _path_midpoint(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    midpoint_index = len(points) // 2
    return points[midpoint_index]


def _edge_draw_rank(g: nx.Graph, topology_name: str, u: str, v: str) -> int:
    if not _is_df_name(topology_name):
        return 0
    draw_order = _df_plane_draw_order(g)
    plane_index = g.nodes[u].get("df_plane_index", g.nodes[v].get("df_plane_index", 0))
    return draw_order.get(int(plane_index), 0)


def _trace_from_edges(
    g: nx.Graph,
    pos: dict[Any, tuple[float, float]],
    *,
    topology_name: str,
    link_kind: str,
    color: str,
    width: float,
    name: str,
) -> go.Scatter:
    edge_x: list[float] = []
    edge_y: list[float] = []
    segments: list[tuple[int, list[tuple[float, float]]]] = []

    for u, v, data in g.edges(data=True):
        if data.get("link_kind") != link_kind:
            continue
        for points in _parallel_edge_paths(g, pos, topology_name, str(u), str(v), data):
            segments.append((_edge_draw_rank(g, topology_name, str(u), str(v)), points))

    for _, points in sorted(segments, key=lambda item: item[0]):
        edge_x.extend([point[0] for point in points] + [None])
        edge_y.extend([point[1] for point in points] + [None])

    return go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=width, color=color),
        hoverinfo="skip",
        showlegend=False,
        name=name,
    )


def _build_interaction_payload(
    g: nx.Graph,
    pos: dict[Any, tuple[float, float]],
    node_ids: list[str],
    *,
    base_node_opacities: list[float] | None = None,
) -> dict[str, Any]:
    topology_name = str(g.graph.get("topology_name", ""))
    node_points: dict[str, Any] = {}
    neighbors: dict[str, list[str]] = {}
    incident_edges: dict[str, list[dict[str, list[float | None]]]] = {}
    incident_link_labels: dict[str, list[dict[str, Any]]] = {}

    for node_id in node_ids:
        node_data = g.nodes[node_id]
        degree = g.degree[node_id]
        x, y = pos[node_id]
        node_points[node_id] = {
            "x": x,
            "y": y,
            "color": _node_color(node_data),
            "size": 12 if node_data.get("node_role") == "ssu" else 15,
            "text": "<br>".join(
                [
                    f"node={node_id}",
                    f"role={node_data.get('node_role', 'unknown')}",
                    f"degree={degree}",
                ]
            ),
        }
        neighbors[node_id] = [str(neighbor) for neighbor in g.neighbors(node_id)]

        edge_segments: list[dict[str, list[float | None]]] = []
        label_items: list[dict[str, Any]] = []
        for neighbor in g.neighbors(node_id):
            edge_data = g.get_edge_data(node_id, neighbor) or {}
            bandwidth = float(edge_data.get("bandwidth_gbps", 0.0))
            parallel_links = max(1, int(edge_data.get("parallel_links", 1)))
            per_link_bandwidth_gbps = bandwidth / float(parallel_links)
            bandwidth_label = f"{per_link_bandwidth_gbps:.0f} Gbps"
            for points in _parallel_edge_paths(g, pos, topology_name, str(node_id), str(neighbor), edge_data):
                edge_segments.append(
                    {
                        "x": [point[0] for point in points] + [None],
                        "y": [point[1] for point in points] + [None],
                    }
                )
                label_x, label_y = _path_midpoint(points)
                label_items.append(
                    {
                        "x": label_x,
                        "y": label_y,
                        "text": bandwidth_label,
                        "bandwidth_label": bandwidth_label,
                    }
                )
        incident_edges[node_id] = edge_segments
        incident_link_labels[node_id] = label_items

    return {
        "node_points": node_points,
        "neighbors": neighbors,
        "incident_edges": incident_edges,
        "incident_link_labels": incident_link_labels,
        "base_node_opacities": list(base_node_opacities or ([1.0] * len(node_ids))),
    }


def _layout_notes(topology_name: str) -> list[str]:
    base_note = "SSUs stay on the bottom row and Unions sit on the layer above inside each exchange node."
    if _torus_family_name(topology_name) == "3D-Torus":
        return [base_note, "Exchange nodes are grouped into paired z-layers, each rendered as one 4x4 plane block."]
    if topology_name == "Clos":
        return [base_note, "Clos spine layer sits above the exchange-node Union layer for structured two-level viewing."]
    if _is_df_name(topology_name):
        return [
            base_note,
            "Dragon-Fly exchange nodes are rendered as staggered circular plane copies in server order so the front ring and the rear ring stay visually separable.",
        ]
    return [base_note, "Exchange nodes are arranged on a structured horizontal and vertical grid."]


def _sparse_focus_traces(
    g: nx.Graph,
    pos: dict[Any, tuple[float, float]],
    traffic_details: dict[str, Any],
) -> list[go.Scatter]:
    traces: list[go.Scatter] = []
    source_set = {
        str(node_id)
        for node_id in traffic_details.get("active_sources", [])
        if str(node_id) in pos and g.nodes[str(node_id)].get("node_role") == "ssu"
    }
    destination_set = {
        str(node_id)
        for node_id in traffic_details.get("target_ssus", [])
        if str(node_id) in pos and g.nodes[str(node_id)].get("node_role") == "ssu"
    }
    highlight_specs = [
        (
            sorted(source_set - destination_set),
            "Source SSU",
            _TRAFFIC_SOURCE_COLOR,
            "#f8fafc",
        ),
        (
            sorted(destination_set - source_set),
            "Destination SSU",
            _TRAFFIC_DESTINATION_COLOR,
            _TRAFFIC_DESTINATION_RING_COLOR,
        ),
        (
            sorted(source_set & destination_set),
            "Source + Destination SSU",
            _TRAFFIC_SOURCE_COLOR,
            _TRAFFIC_DESTINATION_RING_COLOR,
        ),
    ]

    for node_ids, label, fill_color, border_color in highlight_specs:
        if not node_ids:
            continue
        traces.append(
            go.Scatter(
                x=[pos[node_id][0] for node_id in node_ids],
                y=[pos[node_id][1] for node_id in node_ids],
                mode="markers",
                marker=dict(
                    size=15,
                    symbol="circle",
                    color=fill_color,
                    opacity=0.92,
                    line=dict(width=1.8, color=border_color),
                ),
                text=node_ids,
                hovertemplate=f"{label}: %{{text}}<extra></extra>",
                showlegend=True,
                name=label,
            )
        )
    return traces


def _interaction_script(plot_id: str, payload: dict[str, Any]) -> str:
    interaction_json = json.dumps(payload, separators=(",", ":"))
    return f"""
(function() {{
  const plotId = {json.dumps(plot_id)};
  const interaction = {interaction_json};
  const plot = document.getElementById(plotId);
  if (!plot) return;

  function flattenEdgeSegments(segments) {{
    const x = [];
    const y = [];
    segments.forEach((segment) => {{
      x.push(...segment.x);
      y.push(...segment.y);
    }});
    return {{ x, y }};
  }}

  function resetHighlight() {{
    Plotly.restyle(plot, {{opacity: 1}}, [0, 1, 3]);
    Plotly.restyle(plot, {{x: [[]], y: [[]]}}, [2]);
    Plotly.restyle(plot, {{x: [[]], y: [[]], text: [[]], customdata: [[]]}}, [4, 5]);
    const defaultOpacity = interaction.base_node_opacities || interaction.baseNodeIds.map(() => 1);
    Plotly.restyle(plot, {{'marker.opacity': [defaultOpacity]}}, [3]);
    plot.dataset.activeNodeId = '';
  }}

  function highlightNode(nodeId) {{
    const activeIds = [nodeId].concat(interaction.neighbors[nodeId] || []);
    const edgeSegments = interaction.incident_edges[nodeId] || [];
    const flatEdges = flattenEdgeSegments(edgeSegments);
    const linkLabels = interaction.incident_link_labels[nodeId] || [];
    const highlightPoints = activeIds.map((id) => interaction.node_points[id]).filter(Boolean);

    Plotly.restyle(plot, {{opacity: 0.14}}, [0, 1]);
    Plotly.restyle(plot, {{opacity: 0.30}}, [3]);
    const baseOpacity = interaction.base_node_opacities || interaction.baseNodeIds.map(() => 1);
    const nodeOpacity = interaction.baseNodeIds.map((id, index) => activeIds.includes(id) ? 1 : Math.max(0.10, baseOpacity[index] * 0.24));
    Plotly.restyle(plot, {{'marker.opacity': [nodeOpacity]}}, [3]);
    Plotly.restyle(plot, {{x: [flatEdges.x], y: [flatEdges.y]}}, [2]);
    Plotly.restyle(plot, {{
      x: [linkLabels.map((item) => item.x)],
      y: [linkLabels.map((item) => item.y)],
      text: [linkLabels.map((item) => item.text)],
      customdata: [linkLabels.map((item) => item.bandwidth_label)]
    }}, [4]);
    Plotly.restyle(plot, {{
      x: [highlightPoints.map((item) => item.x)],
      y: [highlightPoints.map((item) => item.y)],
      text: [highlightPoints.map((item) => item.text)],
      customdata: [activeIds]
    }}, [5]);
    plot.dataset.activeNodeId = nodeId;
  }}

  plot.on('plotly_click', function(event) {{
    const point = event && event.points && event.points[0];
    if (!point || !point.customdata) return;
    const nodeId = point.customdata;
    if (plot.dataset.activeNodeId === nodeId) {{
      resetHighlight();
      return;
    }}
    highlightNode(nodeId);
  }});

  plot.on('plotly_doubleclick', function() {{
    resetHighlight();
    return false;
  }});

  window.registerTopologyHighlight = window.registerTopologyHighlight || true;
}})();
"""


def _topology_node_traces(
    g: nx.Graph,
    pos: dict[Any, tuple[float, float]],
    title: str,
    *,
    idle_ssu_color: str | None = None,
) -> tuple[go.Scatter, go.Scatter, go.Scatter, list[str], list[float]]:
    node_ids = [str(node_id) for node_id in g.nodes()]
    node_x: list[float] = []
    node_y: list[float] = []
    node_color: list[str] = []
    node_size: list[int] = []
    node_opacity: list[float] = []
    node_text: list[str] = []
    union_label_x: list[float] = []
    union_label_y: list[float] = []
    union_label_text: list[str] = []
    server_label_x: list[float] = []
    server_label_y: list[float] = []
    server_label_text: list[str] = []
    show_df_union_labels = _is_df_name(title)
    front_plane_index = _df_front_plane_index(g) if show_df_union_labels else None

    if show_df_union_labels:
        node_ids.sort(
            key=lambda node_id: (
                _df_node_draw_rank(g, node_id),
                0 if g.nodes[node_id].get("node_role") == "ssu" else 1,
                str(node_id),
            )
        )

    for node_id in node_ids:
        node_data = g.nodes[node_id]
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        if node_data.get("node_role") == "ssu" and idle_ssu_color is not None:
            node_color.append(idle_ssu_color)
        else:
            node_color.append(_node_color(node_data))
        node_size.append(11 if node_data.get("node_role") == "ssu" else 14)
        node_opacity.append(_df_plane_node_opacity(g, node_data) if show_df_union_labels else 1.0)
        node_text.append(
            "<br>".join(
                [
                    f"node={node_id}",
                    f"role={node_data.get('node_role', 'unknown')}",
                    f"degree={g.degree[node_id]}",
                ]
            )
        )
        if (
            show_df_union_labels
            and node_data.get("node_role") == "union"
            and int(node_data.get("df_plane_index", 0)) == int(front_plane_index)
        ):
            label = _df_union_label(node_data)
            if label is not None:
                union_label_x.append(x)
                union_label_y.append(y)
                union_label_text.append(label)

    if show_df_union_labels:
        server_label_x, server_label_y, server_label_text = _df_server_label_positions(
            g,
            pos,
            plane_index=front_plane_index,
        )

    base_nodes = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color="#232b34"),
            opacity=node_opacity,
        ),
        hovertemplate="%{text}<extra></extra>",
        text=node_text,
        customdata=node_ids,
        showlegend=False,
        name="nodes",
    )
    union_labels = go.Scatter(
        x=union_label_x,
        y=union_label_y,
        mode="text",
        text=union_label_text,
        textposition="middle center",
        textfont=dict(color="#eaf6ff", size=11, family="Space Grotesk, Segoe UI, sans-serif"),
        hoverinfo="skip",
        showlegend=False,
        name="union-labels",
    )
    server_labels = go.Scatter(
        x=server_label_x,
        y=server_label_y,
        mode="text",
        text=server_label_text,
        textposition="middle center",
        textfont=dict(color="#8ec5ff", size=12, family="Space Grotesk, Segoe UI, sans-serif"),
        hoverinfo="skip",
        showlegend=False,
        name="server-labels",
    )
    return base_nodes, union_labels, server_labels, node_ids, node_opacity


def _hardware_legend_traces(g: nx.Graph) -> list[go.Scatter]:
    present_roles = {str(data.get("node_role")) for _, data in g.nodes(data=True)}
    legend_items = [
        ("ssu", "SSU", _SSU_NODE_COLOR, 11),
        ("union", "Union", _UNION_NODE_COLOR, 14),
        ("clos_spine", "Clos Spine", _SPINE_NODE_COLOR, 14),
    ]
    traces: list[go.Scatter] = []
    for role, label, color, size in legend_items:
        if role not in present_roles:
            continue
        traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=size, color=color, line=dict(width=1, color="#232b34")),
                hoverinfo="skip",
                showlegend=True,
                name=label,
            )
        )
    return traces


def _topology_subtitle(
    structural_metrics: dict[str, float],
    communication_metrics: dict[str, dict[str, float]],
) -> str:
    a2a = communication_metrics["A2A"]
    sparse = communication_metrics["Sparse 1-to-N"]
    return (
        f"Diameter: {structural_metrics['diameter']:.0f} | "
        f"Avg Hops: {structural_metrics['average_hops']:.2f} | "
        f"A2A Throughput: {a2a['per_ssu_throughput_gbps']:.2f} Gbps | "
        f"{display_workload_name('Sparse 1-to-N')} P95: {sparse['completion_time_p95_s'] * 1e3:.2f} ms"
    )


def _apply_figure_layout(
    fig: go.Figure,
    title: str,
    subtitle: str,
    *,
    height: int,
    clickmode: str | None = None,
    legend_position: str = "top",
    show_heading: bool = True,
) -> None:
    legend_layout: dict[str, Any]
    right_margin = 16
    if legend_position == "right":
        right_margin = 148
        legend_layout = dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(0, 0, 0, 0.86)",
            bordercolor="rgba(86, 166, 255, 0.18)",
            borderwidth=1,
            font=dict(size=11),
            itemsizing="constant",
        )
    elif legend_position == "inside_top_right":
        legend_layout = dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0, 0, 0, 0.82)",
            bordercolor="rgba(86, 166, 255, 0.20)",
            borderwidth=1,
            font=dict(size=11),
            itemsizing="constant",
        )
    else:
        legend_layout = dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0, 0, 0, 0.86)",
            bordercolor="rgba(86, 166, 255, 0.18)",
            borderwidth=1,
            font=dict(size=11),
        )

    plot_title = f"{title}<br><sup>{subtitle}</sup>" if show_heading else None
    layout_args: dict[str, Any] = {
        "title": plot_title,
        "template": "plotly_dark",
        "margin": dict(l=16, r=right_margin, t=72 if show_heading else 16, b=16),
        "xaxis": dict(showgrid=False, zeroline=False, visible=False),
        "yaxis": dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": _PLOT_BG_COLOR,
        "height": height,
        "hovermode": "closest",
        "hoverdistance": 28,
        "dragmode": "pan",
        "font": dict(color="#e8f4ff", family="Space Grotesk, Segoe UI, sans-serif"),
        "hoverlabel": dict(
            bgcolor="rgba(0, 0, 0, 0.97)",
            bordercolor="rgba(86, 166, 255, 0.24)",
            font=dict(color="#e8f4ff", size=12),
        ),
        "legend": legend_layout,
    }
    if clickmode is not None:
        layout_args["clickmode"] = clickmode
    fig.update_layout(**layout_args)


def _plot_config() -> dict[str, Any]:
    return {
        "displaylogo": False,
        "displayModeBar": False,
        "responsive": True,
        "scrollZoom": True,
        "doubleClick": "reset",
    }


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _rgb_to_rgba(rgb: tuple[float, float, float], alpha: float) -> str:
    red, green, blue = (max(0, min(255, int(round(channel)))) for channel in rgb)
    return f"rgba({red}, {green}, {blue}, {alpha:.3f})"


def _scale_rgba_alpha(color: str, factor: float) -> str:
    prefix = "rgba("
    if not color.startswith(prefix) or not color.endswith(")"):
        return color
    parts = [part.strip() for part in color[len(prefix):-1].split(",")]
    if len(parts) != 4:
        return color
    try:
        alpha = max(0.0, min(1.0, float(parts[3]) * factor))
    except ValueError:
        return color
    return f"rgba({parts[0]}, {parts[1]}, {parts[2]}, {alpha:.3f})"


def _traffic_rate_color(total_rate_gbps: float, max_rate_gbps: float) -> str:
    if max_rate_gbps <= 0.0 or total_rate_gbps <= 0.0:
        return "rgba(66, 74, 92, 0.28)"

    stops = [
        (0.0, _hex_to_rgb("#101623")),
        (0.20, _hex_to_rgb("#244a86")),
        (0.48, _hex_to_rgb("#3b82f6")),
        (0.78, _hex_to_rgb("#78a9ff")),
        (1.0, _hex_to_rgb("#ffbe5c")),
    ]
    normalized = min(max(total_rate_gbps / max_rate_gbps, 0.0), 1.0)
    for (left_ratio, left_rgb), (right_ratio, right_rgb) in zip(stops, stops[1:]):
        if normalized > right_ratio:
            continue
        span = max(right_ratio - left_ratio, 1e-9)
        blend = (normalized - left_ratio) / span
        mixed = tuple(
            left_rgb[index] + ((right_rgb[index] - left_rgb[index]) * blend)
            for index in range(3)
        )
        return _rgb_to_rgba(mixed, 0.92)
    return _rgb_to_rgba(stops[-1][1], 0.92)


def _traffic_edge_traces(
    g: nx.Graph,
    pos: dict[Any, tuple[float, float]],
    topology_name: str,
    edge_load_bits: dict[tuple[object, object], float],
    completion_time_s: float,
) -> list[go.Scatter]:
    edge_payloads: list[dict[str, Any]] = []
    plane_draw_order = _df_plane_draw_order(g) if _is_df_name(topology_name) else {}
    plane_count = max(len(plane_draw_order), 1)

    for left, right in g.edges():
        left = str(left)
        right = str(right)
        edge_data = g.get_edge_data(left, right) or {}
        bandwidth_gbps = float(edge_data.get("bandwidth_gbps", 0.0))
        parallel_links = max(1, int(edge_data.get("parallel_links", 1)))
        per_link_bandwidth_gbps = bandwidth_gbps / float(parallel_links)
        bandwidth_label = f"{per_link_bandwidth_gbps:.0f} Gbps"
        forward_bits = float(edge_load_bits.get((left, right), 0.0))
        reverse_bits = float(edge_load_bits.get((right, left), 0.0))
        per_link_forward_bits = forward_bits / float(parallel_links)
        per_link_reverse_bits = reverse_bits / float(parallel_links)
        forward_rate_gbps = (per_link_forward_bits / completion_time_s / 1e9) if completion_time_s > 0 else 0.0
        reverse_rate_gbps = (per_link_reverse_bits / completion_time_s / 1e9) if completion_time_s > 0 else 0.0
        forward_utilization = (forward_rate_gbps / per_link_bandwidth_gbps) if per_link_bandwidth_gbps > 0 else 0.0
        reverse_utilization = (reverse_rate_gbps / per_link_bandwidth_gbps) if per_link_bandwidth_gbps > 0 else 0.0
        plane_index = int(g.nodes[left].get("df_plane_index", g.nodes[right].get("df_plane_index", 0)))
        draw_rank = plane_draw_order.get(plane_index, 0)
        if plane_count <= 1:
            plane_opacity = 1.0
        else:
            frontness = draw_rank / float(plane_count - 1)
            plane_opacity = 0.42 + (0.58 * frontness)
        for points in _parallel_edge_paths(g, pos, topology_name, left, right, edge_data):
            edge_payloads.append(
                {
                    "left": left,
                    "right": right,
                    "x": [point[0] for point in points],
                    "y": [point[1] for point in points],
                    "bandwidth_gbps": per_link_bandwidth_gbps,
                    "bandwidth_label": bandwidth_label,
                    "forward_rate_gbps": forward_rate_gbps,
                    "reverse_rate_gbps": reverse_rate_gbps,
                    "forward_utilization_pct": forward_utilization * 100.0,
                    "reverse_utilization_pct": reverse_utilization * 100.0,
                    "forward_volume_gb": per_link_forward_bits / 8e9,
                    "reverse_volume_gb": per_link_reverse_bits / 8e9,
                    "total_rate_gbps": forward_rate_gbps + reverse_rate_gbps,
                    "line_width": 2.4 if edge_data.get("link_kind") == "backend_interconnect" else 2.0,
                    "draw_rank": draw_rank,
                    "plane_opacity": plane_opacity,
                }
            )

    edge_payloads.sort(key=lambda payload: payload["draw_rank"])
    max_total_rate_gbps = max(
        (payload["total_rate_gbps"] for payload in edge_payloads),
        default=0.0,
    )
    hover_template = (
        "Link: %{customdata[0]} <-> %{customdata[1]}"
        "<br>Theoretical Bandwidth: %{customdata[2]}"
        "<br>%{customdata[0]} -> %{customdata[1]} Carried Rate: %{customdata[3]:.2f} Gbps"
        "<br>%{customdata[1]} -> %{customdata[0]} Carried Rate: %{customdata[4]:.2f} Gbps"
        "<br>%{customdata[0]} -> %{customdata[1]} Utilization: %{customdata[5]:.1f}%"
        "<br>%{customdata[1]} -> %{customdata[0]} Utilization: %{customdata[6]:.1f}%"
        "<br>%{customdata[0]} -> %{customdata[1]} Offered Volume: %{customdata[7]:.2f} GB"
        "<br>%{customdata[1]} -> %{customdata[0]} Offered Volume: %{customdata[8]:.2f} GB"
        "<br>Total Carried Rate: %{customdata[9]:.2f} Gbps"
        "<extra></extra>"
    )
    traces: list[go.Scatter] = []
    for payload in edge_payloads:
        custom_row = [
            payload["left"],
            payload["right"],
            payload["bandwidth_label"],
            payload["forward_rate_gbps"],
            payload["reverse_rate_gbps"],
            payload["forward_utilization_pct"],
            payload["reverse_utilization_pct"],
            payload["forward_volume_gb"],
            payload["reverse_volume_gb"],
            payload["total_rate_gbps"],
        ]
        visible_color = _scale_rgba_alpha(
            _traffic_rate_color(payload["total_rate_gbps"], max_total_rate_gbps),
            payload["plane_opacity"],
        )
        traces.append(
            go.Scatter(
                x=payload["x"],
                y=payload["y"],
                mode="lines",
                line=dict(
                    width=payload["line_width"],
                    color=visible_color,
                ),
                hoverinfo="skip",
                showlegend=False,
                name="traffic-edge-visible",
            )
        )
        traces.append(
            go.Scatter(
                x=payload["x"],
                y=payload["y"],
                mode="lines",
                line=dict(
                    width=max(payload["line_width"] + 8.0, 11.0),
                    color="rgba(255, 255, 255, 0.004)",
                ),
                customdata=[custom_row] * len(payload["x"]),
                hovertemplate=hover_template,
                showlegend=False,
                name="traffic-edge-hitbox",
            )
        )
    return traces


def create_topology_figure(
    g: nx.Graph,
    topology_name: str,
    display_title: str,
    structural_metrics: dict[str, float],
    communication_metrics: dict[str, dict[str, float]],
    layout_seed: int,
) -> tuple[go.Figure, dict[str, Any], str]:
    pos = _positions(g, topology_name, layout_seed)

    internal_edges = _trace_from_edges(
        g,
        pos,
        topology_name=topology_name,
        link_kind="internal_ssu_uplink",
        color=_INTERNAL_EDGE_COLOR,
        width=1.0,
        name="internal",
    )
    backend_edges = _trace_from_edges(
        g,
        pos,
        topology_name=topology_name,
        link_kind="backend_interconnect",
        color=_BACKEND_EDGE_COLOR,
        width=1.7,
        name="backend",
    )
    highlight_edges = go.Scatter(
        x=[],
        y=[],
        mode="lines",
        line=dict(width=3.2, color=_HIGHLIGHT_EDGE_COLOR),
        hoverinfo="skip",
        showlegend=False,
        name="highlight-edges",
    )
    highlight_edge_labels = go.Scatter(
        x=[],
        y=[],
        mode="text",
        text=[],
        customdata=[],
        textposition="top center",
        textfont=dict(color="#ffd58f", size=11, family="Space Grotesk, Segoe UI, sans-serif"),
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
        name="highlight-edge-labels",
    )
    base_nodes, union_labels, server_labels, node_ids, node_opacity = _topology_node_traces(
        g,
        pos,
        topology_name,
    )
    hardware_legend = _hardware_legend_traces(g)
    highlight_nodes = go.Scatter(
        x=[],
        y=[],
        mode="markers",
        marker=dict(size=18, color="#ffd58f", line=dict(width=2, color="#fff0cf")),
        hovertemplate="%{text}<extra></extra>",
        text=[],
        customdata=[],
        showlegend=False,
        name="highlight-nodes",
    )
    subtitle = _topology_subtitle(structural_metrics, communication_metrics)

    fig = go.Figure(
        data=[
            internal_edges,
            backend_edges,
            highlight_edges,
            base_nodes,
            highlight_edge_labels,
            highlight_nodes,
            union_labels,
            server_labels,
            *hardware_legend,
        ]
    )
    _apply_figure_layout(
        fig,
        display_title,
        subtitle,
        height=560,
        clickmode="event",
        legend_position="right",
    )

    interaction = _build_interaction_payload(
        g,
        pos,
        node_ids,
        base_node_opacities=node_opacity,
    )
    interaction["baseNodeIds"] = node_ids
    plot_id = f"plot-{topology_name.lower().replace(' ', '-').replace('_', '-')}"
    return fig, interaction, plot_id


def create_traffic_figure(
    g: nx.Graph,
    topology_name: str,
    workload_name: str,
    workload_metrics: dict[str, float],
    traffic_details: dict[str, Any],
    layout_seed: int,
) -> tuple[go.Figure, str]:
    pos = _positions(g, topology_name, layout_seed)
    traffic_edges = _traffic_edge_traces(
        g,
        pos,
        topology_name,
        traffic_details.get("edge_load_bits", {}),
        float(workload_metrics.get("completion_time_s", 0.0)),
    )
    if workload_name != "A2A":
        base_nodes, union_labels, server_labels, _, _ = _topology_node_traces(
            g,
            pos,
            topology_name,
            idle_ssu_color=_TRAFFIC_IDLE_SSU_COLOR,
        )
        focus_traces = _sparse_focus_traces(g, pos, traffic_details)
    else:
        base_nodes, union_labels, server_labels, _, _ = _topology_node_traces(g, pos, topology_name)
        focus_traces = []

    subtitle = (
        f"Completion: {workload_metrics['completion_time_s'] * 1e3:.2f} ms | "
        f"P95: {workload_metrics['completion_time_p95_s'] * 1e3:.2f} ms | "
        f"Max Backend Utilization: {workload_metrics['max_link_utilization'] * 100:.1f}%"
    )
    fig = go.Figure(
        data=[
            *traffic_edges,
            base_nodes,
            union_labels,
            server_labels,
            *focus_traces,
        ]
    )
    _apply_figure_layout(
        fig,
        f"{display_workload_name(workload_name)} Directional Traffic",
        subtitle,
        height=560,
        legend_position="inside_top_right" if workload_name != "A2A" else "top",
        show_heading=False,
    )
    plot_id = (
        f"plot-{topology_name.lower().replace(' ', '-').replace('_', '-')}"
        f"-{workload_name.lower().replace(' ', '-').replace('_', '-')}-traffic"
    )
    return fig, plot_id


def _join_display_names(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}和{names[1]}"
    return "、".join(names[:-1]) + f"和{names[-1]}"


def _all_topology_page_order(topology_name: str) -> int:
    order = {
        "Clos": 0,
        "2D-FullMesh": 1,
        "2D-FullMesh-2x4": 2,
        "2D-Torus": 3,
        "2D-Torus-BestTwist": 4,
        "3D-Torus": 5,
        "3D-Torus-BestTwist": 6,
        "3D-Torus-2x4x2": 7,
        "3D-Torus-2x4x2-BestTwist": 8,
        "3D-Torus-2x4x1": 9,
        "3D-Torus-2x4x1-BestTwist": 10,
        "DF": 11,
        "SparseMesh-Local": 12,
        "SparseMesh-Global": 13,
    }
    return int(order.get(str(topology_name), 10_000))


def _all_topology_comparison_summary(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not results:
        return []

    result_by_name = {str(item["name"]): item for item in results}

    def get_item(name: str) -> dict[str, Any] | None:
        return result_by_name.get(name)

    def a2a_tp(item: dict[str, Any] | None) -> float:
        if item is None:
            return 0.0
        return float(item["communication_metrics"]["A2A"]["per_ssu_throughput_gbps"])

    def a2a_cv(item: dict[str, Any] | None) -> float:
        if item is None:
            return 0.0
        return float(item["communication_metrics"]["A2A"]["link_utilization_cv"])

    def bisection(item: dict[str, Any] | None) -> float:
        if item is None:
            return 0.0
        return float(item["structural_metrics"]["bisection_bandwidth_gbps_per_ssu"])

    def hops(item: dict[str, Any] | None) -> float:
        if item is None:
            return 0.0
        return float(item["communication_metrics"]["A2A"]["average_hops"])

    def ssu_count(item: dict[str, Any] | None) -> int:
        if item is None:
            return 0
        return int(item["topology"]["ssu_count"])

    clos = get_item("Clos")
    fullmesh = get_item("2D-FullMesh")
    fullmesh_small = get_item("2D-FullMesh-2x4")
    sparse_global = get_item("SparseMesh-Global")
    sparse_local = get_item("SparseMesh-Local")
    torus2_twist = get_item("2D-Torus-BestTwist")
    torus3_twist = get_item("3D-Torus-BestTwist")
    torus3_2x4x2 = get_item("3D-Torus-2x4x2")
    torus3_2x4x2_twist = get_item("3D-Torus-2x4x2-BestTwist")
    torus3_2x4x1 = get_item("3D-Torus-2x4x1")
    torus3_2x4x1_twist = get_item("3D-Torus-2x4x1-BestTwist")
    dragonfly = get_item("DF")
    torus3 = get_item("3D-Torus")
    torus2 = get_item("2D-Torus")

    cards: list[dict[str, str]] = []

    if clos is not None or fullmesh is not None:
        first_parts: list[str] = []
        if clos is not None:
            first_parts.append(
                f"{clos['display_name']} 的 A2A 每 SSU 吞吐最高，为 {a2a_tp(clos):.0f} Gbps；"
                f"每 SSU 对分带宽达到 {bisection(clos):.0f} Gbps，链路利用率 CV 为 {a2a_cv(clos):.3f}。"
            )
        if fullmesh is not None:
            first_parts.append(
                f"{fullmesh['display_name']} 以 {a2a_tp(fullmesh):.0f} Gbps 紧随其后，"
                f"同样接近零热点，平均跳数只有 {hops(fullmesh):.2f}。"
            )
        if fullmesh_small is not None:
            first_parts.append(
                f"{fullmesh_small['display_name']} 缩到 2x4 后，每 SSU 对分带宽降到 {bisection(fullmesh_small):.0f} Gbps，"
                f"A2A 每 SSU 吞吐也回落到 {a2a_tp(fullmesh_small):.0f} Gbps。"
            )
        first_parts.append("这一档的共同特点是后端切分能力强、流量分布均匀，因此 A2A 表现最稳。")
        cards.append(
            {
                "title": "第一档：Clos 与 2D-FullMesh",
                "body": "".join(first_parts),
            }
        )

    if sparse_global is not None or sparse_local is not None:
        sparse_parts: list[str] = []
        if sparse_global is not None and sparse_local is not None:
            sparse_parts.append(
                f"{sparse_global['display_name']} 和 {sparse_local['display_name']} 的 A2A 每 SSU 吞吐分别为 "
                f"{a2a_tp(sparse_global):.0f} / {a2a_tp(sparse_local):.0f} Gbps，已经逼近第一档。"
            )
            sparse_parts.append(
                f"它们的规模分别是 {ssu_count(sparse_global)} / {ssu_count(sparse_local)} 个 SSU，"
                f"每 SSU 对分带宽为 {bisection(sparse_global):.0f} / {bisection(sparse_local):.0f} Gbps。"
            )
        sparse_parts.append(
            "SparseMesh 虽然 fanout 只有 6，但单平面直径可控制在 2 跳，路径短、切分能力也不弱，"
            "所以在当前这组规模下，每 SSU 的 A2A 吞吐表现会明显好于常规 Torus。"
        )
        if torus3_2x4x2_twist is not None or torus3_2x4x1_twist is not None:
            compact_twist_notes: list[str] = []
            if torus3_2x4x2_twist is not None:
                compact_twist_notes.append(
                    f"{torus3_2x4x2_twist['display_name']} {a2a_tp(torus3_2x4x2_twist):.0f} Gbps"
                )
            if torus3_2x4x1_twist is not None:
                compact_twist_notes.append(
                    f"{torus3_2x4x1_twist['display_name']} {a2a_tp(torus3_2x4x1_twist):.0f} Gbps"
                )
            sparse_parts.append(
                "同样属于这一档的还有更小规模的 Best Twist 3D-Torus："
                + "、".join(compact_twist_notes)
                + "，说明缩小 plane 尺寸后，只要热点能被打散，A2A 仍然可以保持较高效率。"
            )
        cards.append(
            {
                "title": "第二档：SparseMesh",
                "body": "".join(sparse_parts),
            }
        )

    if torus2_twist is not None or torus3_twist is not None:
        twist_parts: list[str] = []
        if torus2_twist is not None and torus2 is not None:
            twist_parts.append(
                f"{torus2_twist['display_name']} 相比 {torus2['display_name']}，"
                f"A2A 每 SSU 吞吐从 {a2a_tp(torus2):.0f} 提升到 {a2a_tp(torus2_twist):.0f} Gbps，"
                f"CV 从 {a2a_cv(torus2):.3f} 降到 {a2a_cv(torus2_twist):.3f}。"
            )
        if torus3_twist is not None and torus3 is not None:
            twist_parts.append(
                f"{torus3_twist['display_name']} 相比 {torus3['display_name']}，"
                f"A2A 每 SSU 吞吐从 {a2a_tp(torus3):.0f} 提升到 {a2a_tp(torus3_twist):.0f} Gbps，"
                f"CV 从 {a2a_cv(torus3):.3f} 降到 {a2a_cv(torus3_twist):.3f}。"
            )
        if torus3_2x4x2_twist is not None and torus3_2x4x2 is not None:
            twist_parts.append(
                f"{torus3_2x4x2_twist['display_name']} 相比 {torus3_2x4x2['display_name']}，"
                f"A2A 每 SSU 吞吐从 {a2a_tp(torus3_2x4x2):.0f} 提升到 {a2a_tp(torus3_2x4x2_twist):.0f} Gbps。"
            )
        if torus3_2x4x1_twist is not None and torus3_2x4x1 is not None:
            twist_parts.append(
                f"{torus3_2x4x1_twist['display_name']} 相比 {torus3_2x4x1['display_name']}，"
                f"A2A 每 SSU 吞吐从 {a2a_tp(torus3_2x4x1):.0f} 提升到 {a2a_tp(torus3_2x4x1_twist):.0f} Gbps。"
            )
        twist_parts.append(
            "这一档的价值主要体现在 A2A：Twist 不增加端口预算，而是通过重排 wrap 链路把热点打散，"
            "把原始 Torus 的不均衡问题基本消掉。"
        )
        cards.append(
            {
                "title": "第三档：Best Twist Torus",
                "body": "".join(twist_parts),
            }
        )

    fourth_names = [
        item
        for item in (dragonfly, torus3, torus2, fullmesh_small, torus3_2x4x2, torus3_2x4x1)
        if item is not None
    ]
    if fourth_names:
        fourth_parts: list[str] = []
        if dragonfly is not None:
            fourth_parts.append(
                f"{dragonfly['display_name']} 的规模最大，共 {ssu_count(dragonfly)} 个 SSU，"
                f"A2A 每 SSU 吞吐为 {a2a_tp(dragonfly):.0f} Gbps。"
            )
        torus_notes: list[str] = []
        if torus3 is not None:
            torus_notes.append(f"{torus3['display_name']} {a2a_tp(torus3):.0f} Gbps")
        if torus2 is not None:
            torus_notes.append(f"{torus2['display_name']} {a2a_tp(torus2):.0f} Gbps")
        if fullmesh_small is not None:
            torus_notes.append(f"{fullmesh_small['display_name']} {a2a_tp(fullmesh_small):.0f} Gbps")
        if torus3_2x4x2 is not None:
            torus_notes.append(f"{torus3_2x4x2['display_name']} {a2a_tp(torus3_2x4x2):.0f} Gbps")
        if torus3_2x4x1 is not None:
            torus_notes.append(f"{torus3_2x4x1['display_name']} {a2a_tp(torus3_2x4x1):.0f} Gbps")
        if torus_notes:
            fourth_parts.append("这一档里较受后端切分限制的版本包括 " + "、".join(torus_notes) + "。")
        fourth_parts.append(
            "这一档并不代表拓扑设计差，而是说明当网络规模更大、或者对分带宽和流量均衡度更受限时，"
            "A2A 这种全局最重负载模式会更早暴露后端瓶颈。"
        )
        cards.append(
            {
                "title": "第四档：Dragon-Fly 与原始 Torus",
                "body": "".join(fourth_parts),
            }
        )

    return cards


def render_html_dashboard(results: list[dict[str, Any]], output_path: Path) -> Path:
    template_dir = Path(__file__).resolve().parent.parent / "templates"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("dashboard.html.j2")

    blocks = []
    for item in results:
        display_name = display_topology_name(item["name"])
        fig, interaction, plot_id = create_topology_figure(
            g=item["graph"],
            topology_name=item["name"],
            display_title=display_name,
            structural_metrics=item["structural_metrics"],
            communication_metrics=item["communication_metrics"],
            layout_seed=item["layout_seed"],
        )
        traffic_plots = []
        for workload_name in item.get("traffic_workload_names", ("A2A", "Sparse 1-to-N")):
            traffic_fig, traffic_plot_id = create_traffic_figure(
                g=item["graph"],
                topology_name=item["name"],
                workload_name=workload_name,
                workload_metrics=item["communication_metrics"][workload_name],
                traffic_details=item["traffic_details"][workload_name],
                layout_seed=item["layout_seed"],
            )
            workload_metrics = item["communication_metrics"][workload_name]
            traffic_plots.append(
                {
                    "title": f"{display_workload_name(workload_name)} Directional Traffic",
                    "notes": [
                        "Hover any link to inspect theoretical bandwidth plus forward and reverse carried rate, utilization, and offered volume.",
                        "Edge color encodes the sum of forward and reverse carried rate on that link or parallel-link bundle.",
                        "Scroll to zoom, drag to pan, and double-click to reset the view.",
                        (
                            f"Completion {workload_metrics['completion_time_s'] * 1e3:.2f} ms | "
                            f"P95 {workload_metrics['completion_time_p95_s'] * 1e3:.2f} ms | "
                            f"Max backend utilization {workload_metrics['max_link_utilization'] * 100:.1f}%"
                        ),
                    ],
                    "plot_id": traffic_plot_id,
                    "plot_html": traffic_fig.to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        config=_plot_config(),
                        div_id=traffic_plot_id,
                    ),
                }
            )
        blocks.append(
            {
                "name": item["name"],
                "display_name": display_name,
                "hardware": item["hardware"],
                "topology": item["topology"],
                "routing": item["routing"],
                "routing_diversity": item.get("routing_diversity"),
                "workloads": item["workloads"],
                "workload_descriptions": item.get("workload_descriptions", []),
                "routing_mode_descriptions": item.get("routing_mode_descriptions", []),
                "structural_metrics": item["structural_metrics"],
                "communication_metrics": item["communication_metrics"],
                "workload_metric_rows": item.get("workload_metric_rows", []),
                "default_routing_highlight": item.get("default_routing_highlight"),
                "routing_comparison": item.get("routing_comparison"),
                "observations": item["observations"],
                "layout_notes": _layout_notes(item["name"]),
                "traffic_plots": traffic_plots,
                "plot_id": plot_id,
                "interaction_mode": "neighbors",
                "interaction_script": _interaction_script(plot_id, interaction),
                "plot_html": fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config=_plot_config(),
                    div_id=plot_id,
                ),
            }
        )

    html = template.render(
        results=blocks,
        comparison_results=sorted(
            blocks,
            key=lambda item: (_all_topology_page_order(str(item["name"])), str(item["display_name"])),
        ),
        comparison_summary=_all_topology_comparison_summary(blocks),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    return output_path
