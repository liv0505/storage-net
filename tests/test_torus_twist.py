from topo_sim.config import AnalysisConfig
from topo_sim.topologies import build_topology
from topo_sim.torus_twist import build_torus_twist_graph, generate_google_torus_twist_candidates


def _backend_ports_for_union(g, node_id: str) -> int:
    return sum(
        int(data.get("parallel_links", 1))
        for _, _, data in g.edges(node_id, data=True)
        if data.get("link_kind") == "backend_interconnect"
    )


def _backend_edge_signature(g) -> list[tuple[str, str, str, float, int]]:
    return sorted(
        (
            min(str(u), str(v)),
            max(str(u), str(v)),
            str(data.get("topology_role")),
            float(data.get("bandwidth_gbps", 0.0)),
            int(data.get("parallel_links", 1)),
        )
        for u, v, data in g.edges(data=True)
        if data.get("link_kind") == "backend_interconnect"
    )


def test_2d_torus_google_twist_candidate_count_matches_expected_space():
    candidates = generate_google_torus_twist_candidates("2D-Torus")
    assert len(candidates) == 4
    assert sum(1 for spec in candidates if spec.is_baseline) == 1


def test_3d_torus_google_twist_candidate_count_matches_expected_space():
    candidates = generate_google_torus_twist_candidates("3D-Torus")
    assert len(candidates) == 64
    assert sum(1 for spec in candidates if spec.is_baseline) == 1


def test_3d_torus_2x4x2_google_twist_candidate_count_matches_expected_space():
    candidates = generate_google_torus_twist_candidates("3D-Torus-2x4x2")
    assert len(candidates) == 64
    assert sum(1 for spec in candidates if spec.is_baseline) == 1


def test_3d_torus_2x4x3_google_twist_candidates_keep_baseline_and_best_twist():
    candidates = generate_google_torus_twist_candidates("3D-Torus-2x4x3")
    assert candidates
    assert sum(1 for spec in candidates if spec.is_baseline) == 1
    assert any(spec.wrap_offsets_by_axis == ((0, 2, 0), (0, 0, 0), (0, 0, 0)) for spec in candidates)


def test_3d_torus_2x4x1_google_twist_candidate_count_matches_expected_space():
    candidates = generate_google_torus_twist_candidates("3D-Torus-2x4x1")
    assert len(candidates) == 4
    assert sum(1 for spec in candidates if spec.is_baseline) == 1


def test_2d_torus_baseline_twist_matches_current_default_builder():
    cfg = AnalysisConfig()
    baseline = next(
        spec
        for spec in generate_google_torus_twist_candidates("2D-Torus")
        if spec.is_baseline
    )

    twisted = build_torus_twist_graph(cfg, baseline)
    current = build_topology("2D-Torus", cfg)

    assert twisted.number_of_nodes() == current.number_of_nodes()
    assert twisted.number_of_edges() == current.number_of_edges()
    assert _backend_edge_signature(twisted) == _backend_edge_signature(current)
    assert twisted.graph["torus_twisted"] is False


def test_twisted_2d_torus_preserves_union_backend_port_budget():
    cfg = AnalysisConfig()
    twisted_spec = next(
        spec
        for spec in generate_google_torus_twist_candidates("2D-Torus")
        if spec.wrap_offsets_by_axis == ((0, 2), (1, 0))
    )

    g = build_torus_twist_graph(cfg, twisted_spec)
    union_backend_ports = {
        str(node_id): _backend_ports_for_union(g, str(node_id))
        for node_id, node_data in g.nodes(data=True)
        if node_data.get("node_role") == "union"
    }

    backend_bandwidths = {
        float(data.get("bandwidth_gbps", 0.0))
        for _, _, data in g.edges(data=True)
        if data.get("link_kind") == "backend_interconnect"
    }

    assert union_backend_ports
    assert set(union_backend_ports.values()) == {4}
    assert sum(
        int(data.get("parallel_links", 1))
        for _, _, data in g.edges(data=True)
        if data.get("link_kind") == "backend_interconnect"
    ) == 32
    assert backend_bandwidths == {400.0}
    assert g.graph["torus_twisted"] is True


def test_twisted_3d_torus_preserves_union_backend_port_budget():
    cfg = AnalysisConfig()
    twisted_spec = next(
        spec
        for spec in generate_google_torus_twist_candidates("3D-Torus")
        if spec.wrap_offsets_by_axis == ((0, 2, 2), (1, 0, 2), (1, 2, 0))
    )

    g = build_torus_twist_graph(cfg, twisted_spec)
    union_backend_ports = {
        str(node_id): _backend_ports_for_union(g, str(node_id))
        for node_id, node_data in g.nodes(data=True)
        if node_data.get("node_role") == "union"
    }

    assert union_backend_ports
    assert set(union_backend_ports.values()) == {6}
    assert sum(
        int(data.get("parallel_links", 1))
        for _, _, data in g.edges(data=True)
        if data.get("link_kind") == "backend_interconnect"
    ) == 192
    assert g.graph["torus_twisted"] is True


def test_twisted_3d_torus_2x4x2_preserves_union_backend_port_budget():
    cfg = AnalysisConfig()
    twisted_spec = next(
        spec
        for spec in generate_google_torus_twist_candidates("3D-Torus-2x4x2")
        if spec.wrap_offsets_by_axis == ((0, 0, 1), (1, 0, 1), (0, 2, 0))
    )

    g = build_torus_twist_graph(cfg, twisted_spec)
    union_backend_ports = {
        str(node_id): _backend_ports_for_union(g, str(node_id))
        for node_id, node_data in g.nodes(data=True)
        if node_data.get("node_role") == "union"
    }

    assert union_backend_ports
    assert set(union_backend_ports.values()) == {6}
    assert sum(
        int(data.get("parallel_links", 1))
        for _, _, data in g.edges(data=True)
        if data.get("link_kind") == "backend_interconnect"
    ) == 96
    assert g.graph["torus_twisted"] is True


def test_twisted_3d_torus_2x4x3_preserves_union_backend_port_budget():
    cfg = AnalysisConfig()
    twisted_spec = next(
        spec
        for spec in generate_google_torus_twist_candidates("3D-Torus-2x4x3")
        if spec.wrap_offsets_by_axis == ((0, 2, 0), (0, 0, 0), (0, 0, 0))
    )

    g = build_torus_twist_graph(cfg, twisted_spec)
    union_backend_ports = {
        str(node_id): _backend_ports_for_union(g, str(node_id))
        for node_id, node_data in g.nodes(data=True)
        if node_data.get("node_role") == "union"
    }

    assert union_backend_ports
    assert set(union_backend_ports.values()) == {6}
    assert sum(
        int(data.get("parallel_links", 1))
        for _, _, data in g.edges(data=True)
        if data.get("link_kind") == "backend_interconnect"
    ) == 144
    assert g.graph["torus_twisted"] is True


def test_twisted_3d_torus_2x4x1_preserves_union_backend_port_budget():
    cfg = AnalysisConfig()
    twisted_spec = next(
        spec
        for spec in generate_google_torus_twist_candidates("3D-Torus-2x4x1")
        if spec.wrap_offsets_by_axis == ((0, 2, 0), (0, 0, 0), (0, 0, 0))
    )

    g = build_torus_twist_graph(cfg, twisted_spec)
    union_backend_ports = {
        str(node_id): _backend_ports_for_union(g, str(node_id))
        for node_id, node_data in g.nodes(data=True)
        if node_data.get("node_role") == "union"
    }

    assert union_backend_ports
    assert set(union_backend_ports.values()) == {4}
    assert sum(
        int(data.get("parallel_links", 1))
        for _, _, data in g.edges(data=True)
        if data.get("link_kind") == "backend_interconnect"
    ) == 32
    assert g.graph["torus_twisted"] is True
