import networkx as nx
import pytest

from topo_sim.config import AnalysisConfig
from topo_sim.metrics import compute_structural_metrics, evaluate_workload, evaluate_workload_with_details
from topo_sim.topologies import build_topology
from topo_sim.traffic import FlowDemand, build_a2a_demands


def _message_bits(cfg: AnalysisConfig) -> float:
    return cfg.message_size_mb * 8_000_000.0


def test_structural_metrics_use_ssu_pairs():
    g = build_topology("2D-FullMesh", AnalysisConfig())
    metrics = compute_structural_metrics(g)
    assert "diameter" in metrics
    assert "average_hops" in metrics
    assert "bisection_bandwidth_gbps" in metrics


def test_structural_metrics_use_backend_only_balanced_bisection_bandwidth():
    g = build_topology("2D-Torus", AnalysisConfig())
    metrics = compute_structural_metrics(g)

    assert metrics["bisection_bandwidth_gbps"] == pytest.approx(3200.0)
    assert metrics["bisection_bandwidth_gbps_per_ssu"] == pytest.approx(100.0)


def test_2d_fullmesh_bisection_matches_two_plane_cut_formula():
    g = build_topology("2D-FullMesh", AnalysisConfig())
    metrics = compute_structural_metrics(g)

    assert metrics["bisection_bandwidth_gbps"] == pytest.approx(12800.0)
    assert metrics["bisection_bandwidth_gbps_per_ssu"] == pytest.approx(200.0)


def test_2d_fullmesh_2x4_bisection_matches_two_plane_cut_formula():
    g = build_topology("2D-FullMesh-2x4", AnalysisConfig())
    metrics = compute_structural_metrics(g)

    assert metrics["bisection_bandwidth_gbps"] == pytest.approx(3200.0)
    assert metrics["bisection_bandwidth_gbps_per_ssu"] == pytest.approx(100.0)


def test_3d_torus_bisection_matches_surface_cut_formula():
    g = build_topology("3D-Torus", AnalysisConfig())
    metrics = compute_structural_metrics(g)

    assert metrics["bisection_bandwidth_gbps"] == pytest.approx(12800.0)
    assert metrics["bisection_bandwidth_gbps_per_ssu"] == pytest.approx(100.0)


def test_3d_torus_2x4x2_bisection_matches_surface_cut_formula():
    g = build_topology("3D-Torus-2x4x2", AnalysisConfig())
    metrics = compute_structural_metrics(g)

    assert metrics["bisection_bandwidth_gbps"] == pytest.approx(6400.0)
    assert metrics["bisection_bandwidth_gbps_per_ssu"] == pytest.approx(100.0)


def test_3d_torus_2x4x1_bisection_matches_surface_cut_formula():
    g = build_topology("3D-Torus-2x4x1", AnalysisConfig())
    metrics = compute_structural_metrics(g)

    assert metrics["bisection_bandwidth_gbps"] == pytest.approx(3200.0)
    assert metrics["bisection_bandwidth_gbps_per_ssu"] == pytest.approx(100.0)


def test_bisection_bandwidth_ignores_internal_ssu_uplinks_when_backend_is_removed():
    g = build_topology("2D-Torus", AnalysisConfig())
    for u, v, data in list(g.edges(data=True)):
        if data.get("link_kind") == "backend_interconnect":
            g.remove_edge(u, v)

    metrics = compute_structural_metrics(g)

    assert metrics["bisection_bandwidth_gbps"] == 0.0
    assert metrics["bisection_bandwidth_gbps_per_ssu"] == 0.0


def test_a2a_metrics_return_throughput_completion_and_percentiles():
    cfg = AnalysisConfig()
    g = build_topology("Clos", cfg)
    result = evaluate_workload(g, build_a2a_demands(g, cfg), routing_mode="ECMP", cfg=cfg)

    assert result["completion_time_s"] > 0
    assert result["completion_time_p50_s"] >= 0
    assert result["completion_time_p95_s"] >= 0
    assert result["completion_time_p95_s"] >= result["completion_time_p50_s"]
    assert result["per_ssu_throughput_gbps"] > 0
    assert result["max_link_utilization"] > 0
    assert result["link_utilization_cv"] >= 0


def test_throughput_uses_active_source_count_not_total_ssus():
    cfg = AnalysisConfig()
    g = build_topology("Clos", cfg)
    demands = [
        FlowDemand(src="en0:ssu0", dst="en1:ssu0", bits=_message_bits(cfg)),
        FlowDemand(src="en0:ssu0", dst="en2:ssu0", bits=_message_bits(cfg)),
    ]

    result = evaluate_workload(g, demands, routing_mode="ECMP", cfg=cfg)

    total_bits = sum(d.bits for d in demands)
    expected_active_source_throughput = (total_bits / 1.0) / result["completion_time_s"] / 1e9
    total_ssus = sum(1 for _, data in g.nodes(data=True) if data.get("node_role") == "ssu")
    wrong_total_ssu_throughput = (total_bits / float(total_ssus)) / result["completion_time_s"] / 1e9

    assert result["per_ssu_throughput_gbps"] == pytest.approx(expected_active_source_throughput)
    assert result["per_ssu_throughput_gbps"] > wrong_total_ssu_throughput


def test_backend_utilization_metrics_ignore_internal_only_routes():
    cfg = AnalysisConfig()
    g = build_topology("2D-Torus", cfg)
    demands = [FlowDemand(src="en0:ssu0", dst="en0:ssu1", bits=_message_bits(cfg))]

    result = evaluate_workload(g, demands, routing_mode="MIN_HOPS", cfg=cfg)

    assert result["completion_time_s"] > 0
    assert result["max_link_utilization"] == 0.0
    assert result["link_utilization_cv"] == 0.0
    assert result["completion_time_p50_s"] >= 0
    assert result["completion_time_p95_s"] >= 0


def test_link_utilization_cv_counts_parallel_links_as_separate_physical_samples():
    cfg = AnalysisConfig()
    g = nx.Graph()

    for exchange_id in ("en0", "en1", "en2", "en3"):
        g.add_node(f"{exchange_id}:union0", node_role="union", exchange_node_id=exchange_id)
        g.add_node(f"{exchange_id}:ssu0", node_role="ssu", exchange_node_id=exchange_id)

    for exchange_id in ("en0", "en1", "en2", "en3"):
        g.add_edge(
            f"{exchange_id}:ssu0",
            f"{exchange_id}:union0",
            link_kind="internal_ssu_uplink",
            bandwidth_gbps=200.0,
        )

    g.add_edge(
        "en0:union0",
        "en1:union0",
        link_kind="backend_interconnect",
        bandwidth_gbps=800.0,
        parallel_links=2,
    )
    g.add_edge(
        "en2:union0",
        "en3:union0",
        link_kind="backend_interconnect",
        bandwidth_gbps=400.0,
        parallel_links=1,
    )

    result = evaluate_workload(
        g,
        [FlowDemand(src="en0:ssu0", dst="en1:ssu0", bits=200_000_000_000.0)],
        routing_mode="ECMP",
        cfg=cfg,
    )

    assert result["max_link_utilization"] == pytest.approx(0.25)
    assert result["link_utilization_cv"] == pytest.approx(2 ** 0.5)


def test_completion_time_uses_directional_edge_loads():
    cfg = AnalysisConfig()
    g = build_topology("2D-FullMesh", cfg)
    bidirectional_demands = [
        FlowDemand(src="en0:ssu0", dst="en1:ssu0", bits=_message_bits(cfg)),
        FlowDemand(src="en1:ssu0", dst="en0:ssu0", bits=_message_bits(cfg)),
    ]
    single_direction_demand = [FlowDemand(src="en0:ssu0", dst="en1:ssu0", bits=_message_bits(cfg))]

    forward_only = evaluate_workload(g, single_direction_demand, routing_mode="DOR", cfg=cfg)
    bidirectional = evaluate_workload(g, bidirectional_demands, routing_mode="DOR", cfg=cfg)

    assert bidirectional["completion_time_s"] == pytest.approx(forward_only["completion_time_s"])
    assert bidirectional["max_link_utilization"] == pytest.approx(
        forward_only["max_link_utilization"]
    )


def test_workload_details_keep_opposite_directions_separate():
    cfg = AnalysisConfig()
    g = build_topology("2D-FullMesh", cfg)
    demands = [
        FlowDemand(src="en0:ssu0", dst="en1:ssu0", bits=_message_bits(cfg)),
        FlowDemand(src="en1:ssu0", dst="en0:ssu0", bits=_message_bits(cfg)),
    ]

    details = evaluate_workload_with_details(g, demands, routing_mode="DOR", cfg=cfg)
    metrics_only = evaluate_workload(g, demands, routing_mode="DOR", cfg=cfg)
    edge_load_bits = details["edge_load_bits"]

    assert details["metrics"] == pytest.approx(metrics_only)
    assert ("en0:union0", "en1:union0") in edge_load_bits
    assert ("en1:union0", "en0:union0") in edge_load_bits
    assert edge_load_bits[("en0:union0", "en1:union0")] > 0
    assert edge_load_bits[("en1:union0", "en0:union0")] > 0


def test_hop_volume_distribution_weights_buckets_by_offered_volume():
    cfg = AnalysisConfig(message_size_mb=1.0)
    g = build_topology("2D-FullMesh", cfg)
    same_exchange_bits = _message_bits(cfg)
    remote_bits = same_exchange_bits * 3
    demands = [
        FlowDemand(src="en0:ssu0", dst="en0:ssu1", bits=same_exchange_bits),
        FlowDemand(src="en0:ssu0", dst="en1:ssu0", bits=remote_bits),
    ]

    details = evaluate_workload_with_details(g, demands, routing_mode="DOR", cfg=cfg)
    by_hop = {
        row["hop_count"]: row for row in details["hop_volume_distribution"]
    }

    assert by_hop[2]["offered_volume_gb"] == pytest.approx(same_exchange_bits / 8e9)
    assert by_hop[2]["offered_volume_pct"] == pytest.approx(25.0)
    assert by_hop[3]["offered_volume_gb"] == pytest.approx(remote_bits / 8e9)
    assert by_hop[3]["offered_volume_pct"] == pytest.approx(75.0)


def test_link_volume_distribution_groups_directional_offered_volume():
    cfg = AnalysisConfig(message_size_mb=1.0)
    g = build_topology("2D-FullMesh", cfg)
    demands = [FlowDemand(src="en0:ssu0", dst="en0:ssu1", bits=_message_bits(cfg))]

    details = evaluate_workload_with_details(g, demands, routing_mode="MIN_HOPS", cfg=cfg)
    distribution = details["link_volume_distribution"]

    assert len(distribution) == 1
    row = distribution[0]
    assert row["offered_volume_gb"] == pytest.approx(_message_bits(cfg) / 8e9)
    assert row["link_count"] == 2
    assert row["link_ratio_pct"] == pytest.approx(100.0)


def test_df_structural_metrics_complete_with_server_aware_bisection_candidates():
    g = build_topology("DF", AnalysisConfig())
    metrics = compute_structural_metrics(g)

    assert metrics["bisection_bandwidth_gbps"] == pytest.approx(33600.0)
    assert metrics["bisection_bandwidth_gbps_per_ssu"] == pytest.approx(33600.0 / 208.0)


@pytest.mark.parametrize("routing_mode", ["DOR", "FULL_PATH"])
def test_direct_projection_fast_path_matches_explicit_paths_for_repeated_exchange_pairs(monkeypatch, routing_mode: str):
    import topo_sim.metrics as metrics_module

    cfg = AnalysisConfig()
    g = build_topology("2D-Torus", cfg)
    demands = [
        FlowDemand(src="en0:ssu0", dst="en5:ssu0", bits=_message_bits(cfg)),
        FlowDemand(src="en0:ssu1", dst="en5:ssu2", bits=_message_bits(cfg)),
        FlowDemand(src="en0:ssu2", dst="en5:ssu3", bits=_message_bits(cfg)),
    ]

    fast = evaluate_workload(g, demands, routing_mode=routing_mode, cfg=cfg)

    monkeypatch.setattr(
        metrics_module,
        "_should_use_direct_projection_fast_path",
        lambda _g, _routing_mode: False,
    )
    explicit = evaluate_workload(g, demands, routing_mode=routing_mode, cfg=cfg)

    assert fast == pytest.approx(explicit)


@pytest.mark.parametrize("routing_mode", ["DOR", "FULL_PATH"])
def test_direct_projection_fast_path_reuses_exchange_pair_routes(monkeypatch, routing_mode: str):
    import topo_sim.metrics as metrics_module

    cfg = AnalysisConfig()
    g = build_topology("2D-Torus", cfg)
    demands = [
        FlowDemand(src="en0:ssu0", dst="en5:ssu0", bits=_message_bits(cfg)),
        FlowDemand(src="en0:ssu1", dst="en5:ssu2", bits=_message_bits(cfg)),
        FlowDemand(src="en0:ssu2", dst="en5:ssu3", bits=_message_bits(cfg)),
    ]

    observed_calls: list[tuple[str, str, str]] = []
    original_compute_paths = metrics_module.compute_paths

    def counting_compute_paths(g, src_ssu, dst_ssu, active_mode, active_cfg):
        observed_calls.append((src_ssu, dst_ssu, active_mode))
        return original_compute_paths(g, src_ssu, dst_ssu, active_mode, active_cfg)

    monkeypatch.setattr(metrics_module, "compute_paths", counting_compute_paths)

    result = evaluate_workload(g, demands, routing_mode=routing_mode, cfg=cfg)

    assert result["per_ssu_throughput_gbps"] > 0
    assert len(observed_calls) <= 1
