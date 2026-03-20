import pytest

from topo_sim.config import AnalysisConfig
from topo_sim.topologies import available_topologies, build_topology


def test_available_topologies_only_exposes_new_names():
    assert available_topologies() == ["2D-FullMesh", "2D-Torus", "3D-Torus", "Clos"]


def test_2d_fullmesh_builds_exchange_nodes_with_expected_parts():
    g = build_topology("2D-FullMesh", AnalysisConfig())
    ssu_nodes = [n for n, d in g.nodes(data=True) if d["node_role"] == "ssu"]
    union_nodes = [n for n, d in g.nodes(data=True) if d["node_role"] == "union"]
    assert len(ssu_nodes) == 16 * 8
    assert len(union_nodes) == 16 * 2


def test_2d_fullmesh_has_expected_backend_structure():
    g = build_topology("2D-FullMesh", AnalysisConfig())
    backend = [
        data
        for _, _, data in g.edges(data=True)
        if data["link_kind"] == "backend_interconnect"
    ]
    assert len(backend) == 96
    assert {data["topology_role"] for data in backend} == {"2d_fullmesh_x", "2d_fullmesh_y"}


def test_2d_torus_has_expected_backend_structure():
    g = build_topology("2D-Torus", AnalysisConfig())
    backend = [
        data
        for _, _, data in g.edges(data=True)
        if data["link_kind"] == "backend_interconnect"
    ]
    assert len(backend) == 64
    assert {data["topology_role"] for data in backend} == {"2d_torus_x", "2d_torus_y"}


def test_3d_torus_has_uniform_backend_bandwidth_per_direction():
    g = build_topology("3D-Torus", AnalysisConfig())
    backend = [
        data["bandwidth_gbps"]
        for _, _, data in g.edges(data=True)
        if data["link_kind"] == "backend_interconnect"
    ]
    assert set(backend) == {400.0}
    assert len(backend) == 384


def test_2d_fullmesh_gives_each_union_six_backend_ports():
    g = build_topology("2D-FullMesh", AnalysisConfig())
    union_backend_degree = {
        node_id: sum(
            1
            for _, _, data in g.edges(node_id, data=True)
            if data["link_kind"] == "backend_interconnect"
        )
        for node_id, node_data in g.nodes(data=True)
        if node_data["node_role"] == "union"
    }
    assert union_backend_degree
    assert set(union_backend_degree.values()) == {6}


def test_2d_torus_gives_each_union_four_backend_ports():
    g = build_topology("2D-Torus", AnalysisConfig())
    union_backend_degree = {
        node_id: sum(
            1
            for _, _, data in g.edges(node_id, data=True)
            if data["link_kind"] == "backend_interconnect"
        )
        for node_id, node_data in g.nodes(data=True)
        if node_data["node_role"] == "union"
    }
    assert union_backend_degree
    assert set(union_backend_degree.values()) == {4}


def test_3d_torus_gives_each_union_six_backend_ports():
    g = build_topology("3D-Torus", AnalysisConfig())
    union_backend_degree = {
        node_id: sum(
            1
            for _, _, data in g.edges(node_id, data=True)
            if data["link_kind"] == "backend_interconnect"
        )
        for node_id, node_data in g.nodes(data=True)
        if node_data["node_role"] == "union"
    }
    assert union_backend_degree
    assert set(union_backend_degree.values()) == {6}


def test_clos_uses_18_exchange_nodes_with_plane_local_uplinks():
    cfg = AnalysisConfig()
    g = build_topology("Clos", cfg)
    backend = [
        (u, v, data)
        for u, v, data in g.edges(data=True)
        if data["link_kind"] == "backend_interconnect"
    ]
    uplinks_by_exchange = {}
    for u, v, data in backend:
        for node in (u, v):
            exchange_node = g.nodes[node].get("exchange_node_id")
            if exchange_node is not None:
                uplinks_by_exchange[exchange_node] = uplinks_by_exchange.get(exchange_node, 0) + 1
    assert len(uplinks_by_exchange) == 18
    assert all(count == cfg.clos_uplinks_per_exchange_node * 2 for count in uplinks_by_exchange.values())

    union_backend_degree = {
        node_id: sum(
            1
            for _, _, data in g.edges(node_id, data=True)
            if data.get("link_kind") == "backend_interconnect"
        )
        for node_id, node_data in g.nodes(data=True)
        if node_data.get("node_role") == "union"
    }
    assert union_backend_degree
    assert set(union_backend_degree.values()) == {cfg.clos_uplinks_per_exchange_node}


def test_clos_spines_have_expected_count_and_exact_fanout_in_default_case():
    cfg = AnalysisConfig()
    g = build_topology("Clos", cfg)
    spine_nodes = [n for n, d in g.nodes(data=True) if d.get("node_role") == "clos_spine"]
    assert len(spine_nodes) == cfg.clos_uplinks_per_exchange_node * 2

    downlinks = [
        sum(
            1
            for _, _, data in g.edges(node, data=True)
            if data["link_kind"] == "backend_interconnect"
        )
        for node in spine_nodes
    ]
    assert all(count == 18 for count in downlinks)


def test_clos_rejects_more_than_six_uplinks_per_exchange_node():
    cfg = AnalysisConfig(clos_uplinks_per_exchange_node=7)
    with pytest.raises(ValueError, match="clos_uplinks_per_exchange_node"):
        build_topology("Clos", cfg)


def test_clos_rejects_zero_uplinks_per_exchange_node():
    cfg = AnalysisConfig(clos_uplinks_per_exchange_node=0)
    with pytest.raises(ValueError, match="clos_uplinks_per_exchange_node"):
        build_topology("Clos", cfg)


def test_clos_rejects_negative_uplinks_per_exchange_node():
    cfg = AnalysisConfig(clos_uplinks_per_exchange_node=-1)
    with pytest.raises(ValueError, match="clos_uplinks_per_exchange_node"):
        build_topology("Clos", cfg)


def test_build_topology_is_case_insensitive_and_trims_whitespace():
    g = build_topology("  2d-tOrUs  ", AnalysisConfig())
    ssu_nodes = [n for n, d in g.nodes(data=True) if d["node_role"] == "ssu"]
    assert len(ssu_nodes) == 16 * 8


def test_build_topology_rejects_invalid_name_type():
    with pytest.raises(ValueError, match="Unknown topology"):
        build_topology(None, AnalysisConfig())
