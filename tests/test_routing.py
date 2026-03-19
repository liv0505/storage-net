import pytest

from topo_sim.config import AnalysisConfig
from topo_sim.routing import compute_paths
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


def test_dor_returns_dimension_order_path_for_2d_torus():
    g = build_topology("2D-Torus", AnalysisConfig())
    paths = compute_paths(g, "en0:ssu0", "en5:ssu0", routing_mode="DOR", cfg=AnalysisConfig())
    assert len(paths) == 1
    path = paths[0]
    assert path.nodes[0] == "en0:ssu0"
    assert path.nodes[-1] == "en5:ssu0"

    backend_roles = []
    for u, v in zip(path.nodes[:-1], path.nodes[1:]):
        edge = g.get_edge_data(u, v) or {}
        if edge.get("link_kind") == "backend_interconnect":
            backend_roles.append(edge.get("topology_role"))
    assert backend_roles == ["2d_torus_x", "2d_torus_y"]


def test_dor_routes_same_exchange_ssus_internally():
    g = build_topology("2D-Torus", AnalysisConfig())
    paths = compute_paths(g, "en0:ssu0", "en0:ssu1", routing_mode="DOR", cfg=AnalysisConfig())

    assert len(paths) == 1
    path = paths[0]
    assert path.nodes[0] == "en0:ssu0"
    assert path.nodes[-1] == "en0:ssu1"
    assert len(path.nodes) == 3
    assert path.nodes[1] in {"en0:union0", "en0:union1"}
    _assert_path_stays_internal(g, path.nodes)


@pytest.mark.parametrize("topology_name", ["2D-FullMesh", "2D-Torus", "3D-Torus"])
@pytest.mark.parametrize("routing_mode", ["DOR", "MIN_HOPS", "ECMP", "PORT_BALANCED"])
def test_same_exchange_routing_stays_internal_for_direct_connect_topologies(
    topology_name: str,
    routing_mode: str,
):
    cfg = AnalysisConfig(port_balanced_max_detour_hops=20)
    g = build_topology(topology_name, cfg)
    paths = compute_paths(g, "en0:ssu0", "en0:ssu1", routing_mode=routing_mode, cfg=cfg)

    assert paths
    assert all(path.nodes[0] == "en0:ssu0" and path.nodes[-1] == "en0:ssu1" for path in paths)

    for path in paths:
        _assert_path_stays_internal(g, path.nodes)


def test_compute_paths_normalizes_dor_routing_mode():
    g = build_topology("2D-Torus", AnalysisConfig())
    paths = compute_paths(g, "en0:ssu0", "en5:ssu0", routing_mode=" dor ", cfg=AnalysisConfig())
    assert len(paths) == 1


def test_port_balanced_splits_evenly_across_available_egress_ports():
    g = build_topology("2D-FullMesh", AnalysisConfig())
    paths = compute_paths(
        g,
        "en0:ssu0",
        "en15:ssu0",
        routing_mode="PORT_BALANCED",
        cfg=AnalysisConfig(),
    )

    expected_egress = {
        (union_id, neighbor)
        for union_id in ("en0:union0", "en0:union1")
        for neighbor in g.neighbors(union_id)
        if (g.get_edge_data(union_id, neighbor) or {}).get("link_kind")
        == "backend_interconnect"
    }
    selected_egress = {(path.nodes[1], path.nodes[2]) for path in paths}

    assert len(paths) == len(expected_egress)
    assert selected_egress == expected_egress
    assert all(path.nodes[0] == "en0:ssu0" and path.nodes[-1] == "en15:ssu0" for path in paths)

    weights = {round(path.weight, 8) for path in paths}
    assert len(weights) == 1
    assert abs(sum(path.weight for path in paths) - 1.0) < 1e-9


def test_port_balanced_respects_detour_cap():
    g = build_topology("2D-FullMesh", AnalysisConfig())
    cfg = AnalysisConfig(port_balanced_max_detour_hops=0)
    paths = compute_paths(
        g,
        "en0:ssu0",
        "en15:ssu0",
        routing_mode="PORT_BALANCED",
        cfg=cfg,
    )
    shortest = min(path.hops for path in paths)
    assert all(path.hops == shortest for path in paths)


def test_ecmp_returns_equal_cost_paths_for_clos():
    cfg = AnalysisConfig()
    g = build_topology("Clos", cfg)
    paths = compute_paths(g, "en0:ssu0", "en1:ssu0", routing_mode="ECMP", cfg=cfg)
    assert len(paths) == cfg.clos_uplinks_per_exchange_node
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


@pytest.mark.parametrize("routing_mode", ["DOR", "PORT_BALANCED", "ECMP", "MIN_HOPS"])
def test_compute_paths_returns_empty_list_when_disconnected(routing_mode: str):
    g = build_topology("2D-Torus", AnalysisConfig())
    for u, v, data in list(g.edges(data=True)):
        if data.get("link_kind") == "backend_interconnect":
            g.remove_edge(u, v)

    paths = compute_paths(g, "en0:ssu0", "en15:ssu0", routing_mode=routing_mode, cfg=AnalysisConfig())
    assert paths == []