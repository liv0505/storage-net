import networkx as nx
import pytest

from topo_sim.config import AnalysisConfig
from topo_sim.routing import compute_paths, normalize_routing_mode
from topo_sim.topologies import build_topology


def _iter_path_edges(path_nodes: tuple[str, ...]) -> list[tuple[str, str]]:
    return list(zip(path_nodes[:-1], path_nodes[1:]))


def _assert_path_stays_internal(g, path_nodes: tuple[str, ...]) -> None:
    traversed_kinds = []
    for u, v in _iter_path_edges(path_nodes):
        edge = g.get_edge_data(u, v) or {}
        link_kind = edge.get("link_kind")
        traversed_kinds.append(link_kind)
        assert link_kind == "internal_ssu_uplink"
    assert "backend_interconnect" not in traversed_kinds


def _backend_roles_for_path(g, path_nodes: tuple[str, ...]) -> list[str]:
    roles = []
    for u, v in _iter_path_edges(path_nodes):
        edge = g.get_edge_data(u, v) or {}
        if edge.get("link_kind") == "backend_interconnect":
            roles.append(str(edge.get("topology_role")))
    return roles


def test_dor_returns_two_plane_paths_for_2d_torus():
    g = build_topology("2D-Torus", AnalysisConfig())
    paths = compute_paths(g, "en0:ssu0", "en5:ssu0", routing_mode="DOR", cfg=AnalysisConfig())

    assert len(paths) == 2
    assert {path.nodes[1] for path in paths} == {"en0:union0", "en0:union1"}
    assert {path.nodes[-2] for path in paths} == {"en5:union0", "en5:union1"}
    assert {round(path.weight, 8) for path in paths} == {0.5}
    assert abs(sum(path.weight for path in paths) - 1.0) < 1e-9
    assert {path.hops for path in paths} == {4}
    for path in paths:
        assert path.nodes[0] == "en0:ssu0"
        assert path.nodes[-1] == "en5:ssu0"
        assert _backend_roles_for_path(g, path.nodes) == ["2d_torus_x", "2d_torus_y"]


def test_shortest_path_routing_splits_across_shortest_paths_in_both_planes():
    g = build_topology("2D-Torus", AnalysisConfig())
    paths = compute_paths(
        g,
        "en0:ssu0",
        "en5:ssu0",
        routing_mode="SHORTEST_PATH",
        cfg=AnalysisConfig(),
    )

    assert len(paths) == 4
    assert {path.nodes[1] for path in paths} == {"en0:union0", "en0:union1"}
    assert {path.nodes[-2] for path in paths} == {"en5:union0", "en5:union1"}
    assert {path.hops for path in paths} == {4}
    assert {round(path.weight, 8) for path in paths} == {0.25}
    assert abs(sum(path.weight for path in paths) - 1.0) < 1e-9


def test_2d_fullmesh_shortest_path_differs_from_dor():
    g = build_topology("2D-FullMesh", AnalysisConfig())

    dor_paths = compute_paths(
        g,
        "en0:ssu0",
        "en15:ssu0",
        routing_mode="DOR",
        cfg=AnalysisConfig(),
    )
    shortest_paths = compute_paths(
        g,
        "en0:ssu0",
        "en15:ssu0",
        routing_mode="SHORTEST_PATH",
        cfg=AnalysisConfig(),
    )

    assert len(dor_paths) == 2
    assert {tuple(_backend_roles_for_path(g, path.nodes)) for path in dor_paths} == {
        ("2d_fullmesh_x", "2d_fullmesh_y"),
    }
    assert {round(path.weight, 8) for path in dor_paths} == {0.5}

    assert len(shortest_paths) == 4
    assert {tuple(_backend_roles_for_path(g, path.nodes)) for path in shortest_paths} == {
        ("2d_fullmesh_x", "2d_fullmesh_y"),
        ("2d_fullmesh_y", "2d_fullmesh_x"),
    }
    assert {round(path.weight, 8) for path in shortest_paths} == {0.25}


def test_2d_torus_shortest_path_differs_from_dor():
    g = build_topology("2D-Torus", AnalysisConfig())

    dor_paths = compute_paths(
        g,
        "en0:ssu0",
        "en5:ssu0",
        routing_mode="DOR",
        cfg=AnalysisConfig(),
    )
    shortest_paths = compute_paths(
        g,
        "en0:ssu0",
        "en5:ssu0",
        routing_mode="SHORTEST_PATH",
        cfg=AnalysisConfig(),
    )

    assert len(dor_paths) == 2
    assert {tuple(_backend_roles_for_path(g, path.nodes)) for path in dor_paths} == {
        ("2d_torus_x", "2d_torus_y"),
    }
    assert {round(path.weight, 8) for path in dor_paths} == {0.5}

    assert len(shortest_paths) == 4
    assert {tuple(_backend_roles_for_path(g, path.nodes)) for path in shortest_paths} == {
        ("2d_torus_x", "2d_torus_y"),
        ("2d_torus_y", "2d_torus_x"),
    }
    assert {round(path.weight, 8) for path in shortest_paths} == {0.25}


def test_3d_torus_shortest_path_differs_from_dor():
    g = build_topology("3D-Torus", AnalysisConfig())

    dor_paths = compute_paths(
        g,
        "en0:ssu0",
        "en21:ssu0",
        routing_mode="DOR",
        cfg=AnalysisConfig(),
    )
    shortest_paths = compute_paths(
        g,
        "en0:ssu0",
        "en21:ssu0",
        routing_mode="SHORTEST_PATH",
        cfg=AnalysisConfig(),
    )

    assert len(dor_paths) == 2
    assert {tuple(_backend_roles_for_path(g, path.nodes)) for path in dor_paths} == {
        ("3d_torus_x", "3d_torus_y", "3d_torus_z"),
    }
    assert {round(path.weight, 8) for path in dor_paths} == {0.5}

    assert len(shortest_paths) == 12
    assert {tuple(sorted(_backend_roles_for_path(g, path.nodes))) for path in shortest_paths} == {
        ("3d_torus_x", "3d_torus_y", "3d_torus_z"),
    }
    assert {round(path.weight, 8) for path in shortest_paths} == {round(1.0 / 12.0, 8)}


def test_full_path_uses_all_available_egress_ports_across_both_planes():
    g = build_topology("2D-FullMesh", AnalysisConfig())
    paths = compute_paths(
        g,
        "en0:ssu0",
        "en15:ssu0",
        routing_mode="FULL_PATH",
        cfg=AnalysisConfig(),
    )

    expected_egress = {
        (union_id, neighbor)
        for union_id in ("en0:union0", "en0:union1")
        for neighbor in g.neighbors(union_id)
        if (g.get_edge_data(union_id, neighbor) or {}).get("link_kind") == "backend_interconnect"
    }
    selected_egress = {(path.nodes[1], path.nodes[2]) for path in paths}

    assert len(paths) == len(expected_egress) == 12
    assert selected_egress == expected_egress
    assert all(path.nodes[0] == "en0:ssu0" and path.nodes[-1] == "en15:ssu0" for path in paths)
    assert {round(path.weight, 8) for path in paths} == {round(1.0 / 12.0, 8)}
    assert abs(sum(path.weight for path in paths) - 1.0) < 1e-9


def test_df_same_exchange_shortest_path_stays_internal_and_splits_locally():
    g = build_topology("DF", AnalysisConfig())
    paths = compute_paths(
        g,
        "en0:ssu0",
        "en0:ssu1",
        routing_mode="SHORTEST_PATH",
        cfg=AnalysisConfig(),
    )

    assert len(paths) == 2
    assert {round(path.weight, 8) for path in paths} == {0.5}
    for path in paths:
        _assert_path_stays_internal(g, path.nodes)


def test_df_same_server_different_exchange_uses_server_local_fullmesh_only():
    g = build_topology("DF", AnalysisConfig())
    paths = compute_paths(
        g,
        "en0:ssu0",
        "en1:ssu0",
        routing_mode="SHORTEST_PATH",
        cfg=AnalysisConfig(),
    )

    assert len(paths) == 2
    assert {path.hops for path in paths} == {3}
    assert {round(path.weight, 8) for path in paths} == {0.5}
    assert {
        tuple(_backend_roles_for_path(g, path.nodes))
        for path in paths
    } == {("df_server_fullmesh",)}


def test_df_cross_server_routing_uses_one_unique_inter_server_link_with_local_split():
    g = build_topology("DF", AnalysisConfig())
    paths = compute_paths(
        g,
        "en1:ssu0",
        "en4:ssu0",
        routing_mode="DOR",
        cfg=AnalysisConfig(),
    )

    assert len(paths) == 2
    assert {round(path.weight, 8) for path in paths} == {0.5}
    assert abs(sum(path.weight for path in paths) - 1.0) < 1e-9
    assert {
        tuple(_backend_roles_for_path(g, path.nodes))
        for path in paths
    } == {("df_server_fullmesh", "df_inter_server", "df_server_fullmesh")}


def test_df_shortest_path_uses_true_backend_min_hop_route():
    g = build_topology("DF", AnalysisConfig())
    src_ssu = "en10:ssu0"
    dst_ssu = "en14:ssu0"
    paths = compute_paths(
        g,
        src_ssu,
        dst_ssu,
        routing_mode="SHORTEST_PATH",
        cfg=AnalysisConfig(),
    )

    expected_paths = {
        tuple(path_nodes) for path_nodes in nx.all_shortest_paths(g, src_ssu, dst_ssu)
    }

    assert {path.nodes for path in paths} == expected_paths
    assert {path.hops for path in paths} == {nx.shortest_path_length(g, src_ssu, dst_ssu)}
    assert {round(path.weight, 8) for path in paths} == {round(1.0 / len(expected_paths), 8)}


def test_df_shuffled_cross_server_routing_keeps_one_inter_server_backend_edge():
    g = build_topology("DF-Shuffled", AnalysisConfig())
    src_ssu = "en1:ssu0"
    dst_ssu = "en4:ssu0"
    paths = compute_paths(
        g,
        src_ssu,
        dst_ssu,
        routing_mode="SHORTEST_PATH",
        cfg=AnalysisConfig(),
    )

    assert paths
    assert abs(sum(path.weight for path in paths) - 1.0) < 1e-9
    assert {
        path.nodes for path in paths
    } == {tuple(path_nodes) for path_nodes in nx.all_shortest_paths(g, src_ssu, dst_ssu)}
    assert {path.hops for path in paths} == {nx.shortest_path_length(g, src_ssu, dst_ssu)}


def test_df_scaleup_same_server_paths_follow_ring_shortest_choices():
    g = build_topology("DF-ScaleUp", AnalysisConfig())
    paths = compute_paths(
        g,
        "en0:ssu0",
        "en1:ssu0",
        routing_mode="SHORTEST_PATH",
        cfg=AnalysisConfig(),
    )

    assert len(paths) == 2
    assert {path.hops for path in paths} == {3}
    assert {round(path.weight, 8) for path in paths} == {0.5}
    assert {
        tuple(_backend_roles_for_path(g, path.nodes))
        for path in paths
    } == {("df_server_ring",)}


def test_df_2p_double_4global_same_server_cross_unit_traffic_detours_via_global_plane():
    g = build_topology("DF-2P-Double-4Global", AnalysisConfig())
    paths = compute_paths(
        g,
        "en0:ssu0",
        "en2:ssu0",
        routing_mode="SHORTEST_PATH",
        cfg=AnalysisConfig(),
    )

    assert paths
    assert abs(sum(path.weight for path in paths) - 1.0) < 1e-9
    for path in paths:
        roles = _backend_roles_for_path(g, path.nodes)
        assert roles.count("df_inter_server") >= 2


def test_df_2p_triple_3global_same_server_cross_unit_traffic_detours_via_global_plane():
    g = build_topology("DF-2P-Triple-3Global", AnalysisConfig())
    paths = compute_paths(
        g,
        "en0:ssu0",
        "en2:ssu0",
        routing_mode="SHORTEST_PATH",
        cfg=AnalysisConfig(),
    )

    assert paths
    assert abs(sum(path.weight for path in paths) - 1.0) < 1e-9
    for path in paths:
        roles = _backend_roles_for_path(g, path.nodes)
        assert roles.count("df_inter_server") >= 2


def test_df_2p_double_bridge_3global_same_server_cross_unit_traffic_stays_local():
    g = build_topology("DF-2P-Double-Bridge-3Global", AnalysisConfig())
    paths = compute_paths(
        g,
        "en0:ssu0",
        "en2:ssu0",
        routing_mode="SHORTEST_PATH",
        cfg=AnalysisConfig(),
    )

    assert len(paths) == 2
    assert {round(path.weight, 8) for path in paths} == {0.5}
    assert {
        tuple(_backend_roles_for_path(g, path.nodes))
        for path in paths
    } == {("df_server_bridge",)}


def test_sparsemesh_shortest_path_stays_on_union_planes_without_intermediate_ssus():
    g = build_topology("SparseMesh-Local", AnalysisConfig())
    paths = compute_paths(
        g,
        "en0:ssu0",
        "en3:ssu0",
        routing_mode="SHORTEST_PATH",
        cfg=AnalysisConfig(),
    )

    assert len(paths) == 2
    assert {round(path.weight, 8) for path in paths} == {0.5}
    assert {path.hops for path in paths} == {3}
    for path in paths:
        assert path.nodes[0] == "en0:ssu0"
        assert path.nodes[-1] == "en3:ssu0"
        assert all(
            g.nodes[node_id].get("node_role") == "union"
            for node_id in path.nodes[1:-1]
        )
        assert _backend_roles_for_path(g, path.nodes) == ["sparsemesh_o3"]


def test_sparsemesh_full_path_uses_all_backend_egress_ports_in_both_planes():
    g = build_topology("SparseMesh-Global", AnalysisConfig())
    paths = compute_paths(
        g,
        "en0:ssu0",
        "en11:ssu0",
        routing_mode="FULL_PATH",
        cfg=AnalysisConfig(),
    )

    assert len(paths) == 12
    assert {round(path.weight, 8) for path in paths} == {round(1.0 / 12.0, 8)}
    assert abs(sum(path.weight for path in paths) - 1.0) < 1e-9
    assert all(path.nodes[0] == "en0:ssu0" and path.nodes[-1] == "en11:ssu0" for path in paths)
    assert all(
        all(g.nodes[node_id].get("node_role") == "union" for node_id in path.nodes[1:-1])
        for path in paths
    )


@pytest.mark.parametrize(
    "routing_mode",
    ["DOR", "SHORTEST_PATH", "FULL_PATH", "ECMP", "MIN_HOPS", "PORT_BALANCED"],
)
def test_same_exchange_routing_stays_internal_for_direct_connect_topologies(routing_mode: str):
    cfg = AnalysisConfig(port_balanced_max_detour_hops=20)
    g = build_topology("2D-Torus", cfg)
    paths = compute_paths(g, "en0:ssu0", "en0:ssu1", routing_mode=routing_mode, cfg=cfg)

    assert paths
    assert all(path.nodes[0] == "en0:ssu0" and path.nodes[-1] == "en0:ssu1" for path in paths)
    for path in paths:
        _assert_path_stays_internal(g, path.nodes)


def test_normalize_routing_mode_maps_legacy_aliases():
    assert normalize_routing_mode("MIN_HOPS") == "SHORTEST_PATH"
    assert normalize_routing_mode("PORT_BALANCED") == "FULL_PATH"


def test_compute_paths_normalizes_dor_routing_mode():
    g = build_topology("2D-Torus", AnalysisConfig())
    paths = compute_paths(g, "en0:ssu0", "en5:ssu0", routing_mode=" dor ", cfg=AnalysisConfig())
    assert len(paths) == 2


def test_compute_paths_accepts_legacy_aliases_for_direct_topologies():
    g = build_topology("2D-FullMesh", AnalysisConfig())
    shortest_paths = compute_paths(g, "en0:ssu0", "en15:ssu0", routing_mode="SHORTEST_PATH", cfg=AnalysisConfig())
    alias_shortest_paths = compute_paths(g, "en0:ssu0", "en15:ssu0", routing_mode="MIN_HOPS", cfg=AnalysisConfig())
    full_paths = compute_paths(g, "en0:ssu0", "en15:ssu0", routing_mode="FULL_PATH", cfg=AnalysisConfig())
    alias_full_paths = compute_paths(g, "en0:ssu0", "en15:ssu0", routing_mode="PORT_BALANCED", cfg=AnalysisConfig())

    assert [path.nodes for path in shortest_paths] == [path.nodes for path in alias_shortest_paths]
    assert [path.nodes for path in full_paths] == [path.nodes for path in alias_full_paths]


def test_ecmp_returns_equal_cost_paths_for_clos():
    cfg = AnalysisConfig()
    g = build_topology("Clos", cfg)
    paths = compute_paths(g, "en0:ssu0", "en1:ssu0", routing_mode="ECMP", cfg=cfg)
    assert len(paths) == cfg.clos_uplinks_per_exchange_node * 2
    assert len({path.hops for path in paths}) == 1
    assert len({round(path.weight, 8) for path in paths}) == 1
    assert abs(sum(path.weight for path in paths) - 1.0) < 1e-9


def test_compute_paths_rejects_invalid_routing_mode():
    g = build_topology("2D-Torus", AnalysisConfig())
    with pytest.raises(ValueError, match="Unsupported routing mode"):
        compute_paths(g, "en0:ssu0", "en5:ssu0", routing_mode="BAD_MODE", cfg=AnalysisConfig())


def test_compute_paths_rejects_non_string_routing_mode():
    g = build_topology("2D-Torus", AnalysisConfig())
    with pytest.raises(ValueError, match="routing_mode"):
        compute_paths(g, "en0:ssu0", "en5:ssu0", routing_mode=123, cfg=AnalysisConfig())


@pytest.mark.parametrize(
    "routing_mode",
    ["DOR", "SHORTEST_PATH", "FULL_PATH", "ECMP", "MIN_HOPS", "PORT_BALANCED"],
)
def test_compute_paths_returns_empty_list_when_disconnected(routing_mode: str):
    g = build_topology("2D-Torus", AnalysisConfig())
    for u, v, data in list(g.edges(data=True)):
        if data.get("link_kind") == "backend_interconnect":
            g.remove_edge(u, v)

    paths = compute_paths(g, "en0:ssu0", "en7:ssu0", routing_mode=routing_mode, cfg=AnalysisConfig())
    assert paths == []
