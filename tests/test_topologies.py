import networkx as nx
import pytest

from topo_sim.config import AnalysisConfig
from topo_sim.topologies import available_topologies, build_topology


def _backend_ports_for_union(g, node_id: str) -> int:
    return sum(
        int(data.get("parallel_links", 1))
        for _, _, data in g.edges(node_id, data=True)
        if data["link_kind"] == "backend_interconnect"
    )


def test_available_topologies_only_exposes_new_names():
    assert available_topologies() == [
        "2D-FullMesh",
        "2D-Torus",
        "3D-Torus",
        "Clos",
        "DF",
        "DF-Shuffled",
        "DF-ScaleUp",
        "DF-2P-Double-4Global",
        "DF-2P-Triple-3Global",
        "DF-2P-Double-Bridge-3Global",
    ]


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
    assert len(backend) == 192


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


def test_df_builds_expected_server_scoped_counts():
    cfg = AnalysisConfig()
    g = build_topology("DF", cfg)

    ssu_nodes = [n for n, d in g.nodes(data=True) if d["node_role"] == "ssu"]
    union_nodes = [n for n, d in g.nodes(data=True) if d["node_role"] == "union"]

    assert len(ssu_nodes) == 52 * 8
    assert len(union_nodes) == 52 * 2
    assert g.graph["df_server_count"] == 13
    assert g.graph["df_total_server_count"] == 26
    assert g.graph["df_exchange_nodes_per_server"] == 4
    assert g.graph["df_group_count"] == 52
    assert g.graph["df_ssu_count_per_plane"] == 208
    assert g.graph["df_union_count_per_plane"] == 52


def test_df_has_expected_backend_structure():
    g = build_topology("DF", AnalysisConfig())
    backend = [
        data
        for _, _, data in g.edges(data=True)
        if data["link_kind"] == "backend_interconnect"
    ]
    assert len(backend) == 312
    assert {data["topology_role"] for data in backend} == {"df_server_fullmesh", "df_inter_server"}


def test_df_gives_each_union_six_backend_ports_per_plane_copy_by_default():
    g = build_topology("DF", AnalysisConfig())
    union_backend_degree = {
        node_id: _backend_ports_for_union(g, node_id)
        for node_id, node_data in g.nodes(data=True)
        if node_data["node_role"] == "union"
    }
    assert union_backend_degree
    assert g.graph["df_plane_count"] == 2
    assert set(union_backend_degree.values()) == {6}


def test_df_shuffled_matches_df_scale_and_port_budget():
    g = build_topology("DF-Shuffled", AnalysisConfig())

    assert g.graph["df_server_count"] == 13
    assert g.graph["df_total_server_count"] == 26
    assert g.graph["df_local_topology"] == "fullmesh"
    assert g.graph["df_plane_count"] == 2
    assert g.graph["df_base_global_ports_per_union"] == 3
    assert g.graph["df_global_ports_per_union"] == 3

    union_backend_degree = {
        node_id: _backend_ports_for_union(g, node_id)
        for node_id, node_data in g.nodes(data=True)
        if node_data["node_role"] == "union"
    }
    assert union_backend_degree
    assert set(union_backend_degree.values()) == {6}


def test_df_scaleup_uses_ring_local_links_and_more_servers():
    g = build_topology("DF-ScaleUp", AnalysisConfig())

    assert g.graph["df_server_count"] == 17
    assert g.graph["df_total_server_count"] == 34
    assert g.graph["df_local_topology"] == "ring"
    assert g.graph["df_plane_count"] == 2
    assert g.graph["df_base_global_ports_per_union"] == 4
    assert g.graph["df_global_ports_per_union"] == 4

    local_roles = {
        data["topology_role"]
        for _, _, data in g.edges(data=True)
        if data["link_kind"] == "backend_interconnect"
        and str(data["topology_role"]).startswith("df_server_")
    }
    assert local_roles == {"df_server_ring"}

    union_backend_degree = {
        node_id: _backend_ports_for_union(g, node_id)
        for node_id, node_data in g.nodes(data=True)
        if node_data["node_role"] == "union"
    }
    assert union_backend_degree
    assert set(union_backend_degree.values()) == {6}


def test_df_2p_double_4global_increases_scale_with_800g_local_bonds():
    g = build_topology("DF-2P-Double-4Global", AnalysisConfig())

    assert g.graph["df_server_count"] == 17
    assert g.graph["df_total_server_count"] == 34
    assert g.graph["df_local_topology"] == "pair_double"
    assert g.graph["df_plane_count"] == 2
    assert g.graph["df_base_global_ports_per_union"] == 4
    assert g.graph["df_global_ports_per_union"] == 4

    local_roles = {
        data["topology_role"]
        for _, _, data in g.edges(data=True)
        if data["link_kind"] == "backend_interconnect"
        and str(data["topology_role"]).startswith("df_")
        and data["topology_role"] != "df_inter_server"
    }
    assert local_roles == {"df_pair_double"}

    pair_bond_bandwidths = {
        float(data["bandwidth_gbps"])
        for _, _, data in g.edges(data=True)
        if data.get("topology_role") == "df_pair_double"
    }
    assert pair_bond_bandwidths == {800.0}

    union_backend_degree = {
        node_id: _backend_ports_for_union(g, node_id)
        for node_id, node_data in g.nodes(data=True)
        if node_data["node_role"] == "union"
    }
    assert set(union_backend_degree.values()) == {6}


def test_df_2p_triple_3global_uses_1200g_pair_bonds_with_six_total_ports_per_plane_copy():
    g = build_topology("DF-2P-Triple-3Global", AnalysisConfig())

    assert g.graph["df_server_count"] == 13
    assert g.graph["df_total_server_count"] == 26
    assert g.graph["df_local_topology"] == "pair_triple"
    assert g.graph["df_plane_count"] == 2
    assert g.graph["df_base_global_ports_per_union"] == 3
    assert g.graph["df_global_ports_per_union"] == 3

    pair_bond_bandwidths = {
        float(data["bandwidth_gbps"])
        for _, _, data in g.edges(data=True)
        if data.get("topology_role") == "df_pair_triple"
    }
    assert pair_bond_bandwidths == {1200.0}

    union_backend_degree = {
        node_id: _backend_ports_for_union(g, node_id)
        for node_id, node_data in g.nodes(data=True)
        if node_data["node_role"] == "union"
    }
    assert set(union_backend_degree.values()) == {6}


def test_df_2p_double_bridge_3global_adds_server_bridge_with_dual_plane_budget():
    g = build_topology("DF-2P-Double-Bridge-3Global", AnalysisConfig())

    assert g.graph["df_server_count"] == 13
    assert g.graph["df_total_server_count"] == 26
    assert g.graph["df_local_topology"] == "pair_double_bridge"
    assert g.graph["df_plane_count"] == 2
    assert g.graph["df_base_global_ports_per_union"] == 3
    assert g.graph["df_global_ports_per_union"] == 3

    local_roles = {
        data["topology_role"]
        for _, _, data in g.edges(data=True)
        if data["link_kind"] == "backend_interconnect"
        and str(data["topology_role"]).startswith("df_")
        and data["topology_role"] != "df_inter_server"
    }
    assert local_roles == {"df_pair_double", "df_server_bridge"}

    union_backend_degree = {
        node_id: _backend_ports_for_union(g, node_id)
        for node_id, node_data in g.nodes(data=True)
        if node_data["node_role"] == "union"
    }
    assert set(union_backend_degree.values()) == {6}


@pytest.mark.parametrize(
    ("name", "expected_plane_unions", "expected_total_ssus", "expected_total_unions"),
    [
        ("DF", 52, 416, 104),
        ("DF-Shuffled", 52, 416, 104),
        ("DF-ScaleUp", 68, 544, 136),
        ("DF-2P-Double-4Global", 68, 544, 136),
        ("DF-2P-Triple-3Global", 52, 416, 104),
        ("DF-2P-Double-Bridge-3Global", 52, 416, 104),
    ],
)
def test_df_variants_share_ssus_across_two_backend_plane_components(
    name: str,
    expected_plane_unions: int,
    expected_total_ssus: int,
    expected_total_unions: int,
):
    g = build_topology(name, AnalysisConfig())

    assert g.graph["df_plane_count"] == 2
    assert nx.number_connected_components(g) == 1

    component = next(iter(nx.connected_components(g)))
    ssu_count = sum(1 for node_id in component if g.nodes[node_id].get("node_role") == "ssu")
    union_count = sum(1 for node_id in component if g.nodes[node_id].get("node_role") == "union")
    assert (ssu_count, union_count) == (expected_total_ssus, expected_total_unions)

    backend_graph = nx.Graph()
    for u, v, data in g.edges(data=True):
        if data.get("link_kind") == "backend_interconnect":
            backend_graph.add_edge(u, v)
    backend_component_sizes = sorted(len(component) for component in nx.connected_components(backend_graph))

    assert backend_component_sizes == [expected_plane_unions, expected_plane_unions]


@pytest.mark.parametrize(
        ("name", "expected_ssus", "expected_unions", "expected_ssu_union_links", "expected_union_union_links"),
    [
        ("2D-FullMesh", 128, 32, 256, 96),
        ("2D-Torus", 128, 32, 256, 64),
        ("3D-Torus", 256, 64, 512, 192),
        ("Clos", 144, 36, 288, 144),
        ("DF", 416, 104, 832, 312),
        ("DF-Shuffled", 416, 104, 832, 312),
        ("DF-ScaleUp", 544, 136, 1088, 408),
        ("DF-2P-Double-4Global", 544, 136, 1088, 408),
        ("DF-2P-Triple-3Global", 416, 104, 832, 312),
        ("DF-2P-Double-Bridge-3Global", 416, 104, 832, 312),
    ],
)
def test_topology_hardware_inventory_counts_match_expected_totals(
    name: str,
    expected_ssus: int,
    expected_unions: int,
    expected_ssu_union_links: int,
    expected_union_union_links: int,
):
    g = build_topology(name, AnalysisConfig())

    actual_ssus = sum(1 for _, data in g.nodes(data=True) if data.get("node_role") == "ssu")
    actual_unions = sum(1 for _, data in g.nodes(data=True) if data.get("node_role") == "union")
    actual_ssu_union_links = sum(
        1 for _, _, data in g.edges(data=True) if data.get("link_kind") == "internal_ssu_uplink"
    )
    actual_union_union_links = sum(
        int(data.get("parallel_links", 1))
        for _, _, data in g.edges(data=True)
        if data.get("link_kind") == "backend_interconnect"
    )

    assert actual_ssus == expected_ssus
    assert actual_unions == expected_unions
    assert actual_ssu_union_links == expected_ssu_union_links
    assert actual_union_union_links == expected_union_union_links


def test_df_rejects_non_power_of_two_union_count_per_server():
    with pytest.raises(ValueError, match="df_unions_per_server"):
        build_topology("DF", AnalysisConfig(df_unions_per_server=6))


def test_df_2p_bridge_variant_rejects_non_four_union_servers():
    with pytest.raises(ValueError, match="df_unions_per_server == 4"):
        build_topology("DF-2P-Double-Bridge-3Global", AnalysisConfig(df_unions_per_server=8))


def test_build_topology_is_case_insensitive_and_trims_whitespace():
    g = build_topology("  2d-tOrUs  ", AnalysisConfig())
    ssu_nodes = [n for n, d in g.nodes(data=True) if d["node_role"] == "ssu"]
    assert len(ssu_nodes) == 16 * 8


def test_build_topology_rejects_invalid_name_type():
    with pytest.raises(ValueError, match="Unknown topology"):
        build_topology(None, AnalysisConfig())
