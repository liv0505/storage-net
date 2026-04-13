import networkx as nx
import pytest

import topo_sim.metrics as metrics_module
from topo_sim.config import AnalysisConfig
from topo_sim.metrics import evaluate_workload
from topo_sim.topologies import build_topology
from topo_sim.traffic import FlowDemand, build_sparse_random_demands


def _message_bits(cfg: AnalysisConfig) -> float:
    return cfg.message_size_mb * 8_000_000.0


def test_exact_shortest_path_flow_aggregation_matches_explicit_path_splitting_on_2d_torus(monkeypatch):
    cfg = AnalysisConfig()
    g = build_topology("2D-Torus", cfg)
    demands = [
        FlowDemand(src="en0:ssu0", dst="en5:ssu0", bits=_message_bits(cfg)),
        FlowDemand(src="en1:ssu3", dst="en6:ssu2", bits=_message_bits(cfg)),
    ]

    monkeypatch.setattr(metrics_module, "_should_use_exact_shortest_path_fast_path", lambda graph, mode: False)
    explicit = evaluate_workload(g, demands, routing_mode="SHORTEST_PATH", cfg=cfg)

    monkeypatch.setattr(metrics_module, "_should_use_exact_shortest_path_fast_path", lambda graph, mode: True)
    accelerated = evaluate_workload(g, demands, routing_mode="SHORTEST_PATH", cfg=cfg)

    assert accelerated == pytest.approx(explicit)


def test_3d_torus_shortest_path_uses_exact_accelerated_flow_projection(monkeypatch):
    cfg = AnalysisConfig()
    g = build_topology("3D-Torus", cfg)
    demands = [FlowDemand(src="en0:ssu0", dst="en21:ssu0", bits=_message_bits(cfg))]

    monkeypatch.setattr(metrics_module, "_should_use_exact_shortest_path_fast_path", lambda graph, mode: True)

    def fail_if_all_shortest_paths(*args, **kwargs):
        raise AssertionError("explicit shortest-path enumeration should not run here")

    monkeypatch.setattr(nx, "all_shortest_paths", fail_if_all_shortest_paths)

    result = evaluate_workload(g, demands, routing_mode="SHORTEST_PATH", cfg=cfg)

    assert result["per_ssu_throughput_gbps"] > 0


def test_accelerated_shortest_path_projection_still_emits_all_required_metrics(monkeypatch):
    cfg = AnalysisConfig(sparse_active_ratio=0.05, sparse_target_count=1, random_seed=7)
    g = build_topology("3D-Torus", cfg)

    monkeypatch.setattr(metrics_module, "_should_use_exact_shortest_path_fast_path", lambda graph, mode: True)

    result = evaluate_workload(g, build_sparse_random_demands(g, cfg), routing_mode="SHORTEST_PATH", cfg=cfg)

    assert set(result) >= {
        "completion_time_s",
        "completion_time_p50_s",
        "completion_time_p95_s",
        "per_ssu_throughput_gbps",
        "max_link_utilization",
        "link_utilization_cv",
    }
