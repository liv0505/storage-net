import pytest

from topo_sim.config import AnalysisConfig
from topo_sim.metrics import compute_structural_metrics, evaluate_workload
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
