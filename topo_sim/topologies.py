from __future__ import annotations

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
        role_counts[role] = role_counts.get(role, 0) + 1

        for node in (u, v):
            exchange_node_id = g.nodes[node].get("exchange_node_id")
            if exchange_node_id is not None:
                uplinks_by_exchange[exchange_node_id] = (
                    uplinks_by_exchange.get(exchange_node_id, 0) + 1
                )

    if len(role_counts) > 1 and len(set(role_counts.values())) != 1:
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
) -> None:
    g.add_edge(
        src_union_id,
        dst_union_id,
        bandwidth_gbps=_BACKEND_BW_GBPS,
        link_kind="backend_interconnect",
        topology_role=topology_role,
    )


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

    return _annotate_graph(g, cfg)


def build_2d_torus(cfg: AnalysisConfig) -> nx.Graph:
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
            for c in range(cols):
                _add_backend_link(
                    g,
                    exchanges[(r, c)]["unions"][union_index],
                    exchanges[(r, (c + 1) % cols)]["unions"][union_index],
                    topology_role="2d_torus_x",
                )
                _add_backend_link(
                    g,
                    exchanges[(r, c)]["unions"][union_index],
                    exchanges[((r + 1) % rows, c)]["unions"][union_index],
                    topology_role="2d_torus_y",
                )

    return _annotate_graph(g, cfg)


def build_3d_torus(cfg: AnalysisConfig) -> nx.Graph:
    g = nx.Graph()
    size = 4
    exchanges: dict[tuple[int, int, int], dict[str, list[str]]] = {}

    for x in range(size):
        for y in range(size):
            for z in range(size):
                exchange_id = f"en{(x * size * size) + (y * size) + z}"
                exchanges[(x, y, z)] = _add_exchange_node(g, exchange_id, cfg)

    for union_index in range(2):
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    _add_backend_link(
                        g,
                        exchanges[(x, y, z)]["unions"][union_index],
                        exchanges[((x + 1) % size, y, z)]["unions"][union_index],
                        topology_role="3d_torus_x",
                    )
                    _add_backend_link(
                        g,
                        exchanges[(x, y, z)]["unions"][union_index],
                        exchanges[(x, (y + 1) % size, z)]["unions"][union_index],
                        topology_role="3d_torus_y",
                    )
                    _add_backend_link(
                        g,
                        exchanges[(x, y, z)]["unions"][union_index],
                        exchanges[(x, y, (z + 1) % size)]["unions"][union_index],
                        topology_role="3d_torus_z",
                    )

    return _annotate_graph(g, cfg)


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
    _validate_df_server_shape(cfg)

    g = nx.Graph()
    unions_per_server = int(cfg.df_unions_per_server)
    exchange_nodes_per_server = unions_per_server // 2
    external_servers_per_union = int(cfg.df_external_servers_per_union)
    server_count = (unions_per_server * external_servers_per_union) + 1
    inter_server_gateways: dict[tuple[int, int], tuple[str, str]] = {}
    unions_by_server: dict[int, list[str]] = {}

    exchange_index = 0
    for server_id in range(server_count):
        server_unions: list[str] = []
        for exchange_offset in range(exchange_nodes_per_server):
            exchange_id = f"en{exchange_index}"
            exchange = _add_exchange_node(g, exchange_id, cfg)

            for ssu_id in exchange["ssus"]:
                g.nodes[ssu_id].update(
                    server_id=server_id,
                    server_local_exchange_index=exchange_offset,
                )

            for union_offset, union_id in enumerate(exchange["unions"]):
                server_local_union_index = (exchange_offset * 2) + union_offset
                g.nodes[union_id].update(
                    server_id=server_id,
                    server_local_exchange_index=exchange_offset,
                    server_local_union_index=server_local_union_index,
                )
                server_unions.append(union_id)

            exchange_index += 1

        unions_by_server[server_id] = server_unions

    for server_id, union_ids in unions_by_server.items():
        for src_index, src_union_id in enumerate(union_ids):
            for dst_union_id in union_ids[src_index + 1 :]:
                _add_backend_link(
                    g,
                    src_union_id,
                    dst_union_id,
                    topology_role="df_server_fullmesh",
                )

    for src_server in range(server_count):
        for src_union_index, src_union_id in enumerate(unions_by_server[src_server]):
            start_dst_server = src_server + (external_servers_per_union * src_union_index) + 1
            dst_union_index = unions_per_server - 1 - src_union_index
            for offset in range(external_servers_per_union):
                dst_server = (start_dst_server + offset) % server_count
                dst_union_id = unions_by_server[dst_server][dst_union_index]
                if g.has_edge(src_union_id, dst_union_id):
                    continue
                _add_backend_link(
                    g,
                    src_union_id,
                    dst_union_id,
                    topology_role="df_inter_server",
                )
                inter_server_gateways[(src_server, dst_server)] = (src_union_id, dst_union_id)
                inter_server_gateways[(dst_server, src_server)] = (dst_union_id, src_union_id)

    g.graph["df_server_count"] = server_count
    g.graph["df_exchange_nodes_per_server"] = exchange_nodes_per_server
    g.graph["df_unions_per_server"] = unions_per_server
    g.graph["df_external_servers_per_union"] = external_servers_per_union
    g.graph["df_inter_server_gateways"] = inter_server_gateways
    return _annotate_graph(g, cfg)


BUILDERS: dict[str, TopologyBuilder] = {
    "2D-FullMesh": build_2d_fullmesh,
    "2D-Torus": build_2d_torus,
    "3D-Torus": build_3d_torus,
    "Clos": build_clos,
    "DF": build_df,
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
