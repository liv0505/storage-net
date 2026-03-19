from collections import Counter, defaultdict
import math

import pytest

from topo_sim.config import AnalysisConfig
from topo_sim.topologies import build_topology
from topo_sim.traffic import build_a2a_demands, build_sparse_random_demands


def test_a2a_builds_exactly_one_flow_for_each_distinct_ordered_ssu_pair():
    cfg = AnalysisConfig(message_size_mb=2.5)
    g = build_topology("2D-FullMesh", cfg)
    ssus = [node_id for node_id, data in g.nodes(data=True) if data["node_role"] == "ssu"]
    demands = build_a2a_demands(g, cfg)

    expected_pairs = {(src, dst) for src in ssus for dst in ssus if src != dst}
    pairs = [(d.src, d.dst) for d in demands]
    pair_counts = Counter(pairs)

    assert all(d.src != d.dst for d in demands)
    assert set(pairs) == expected_pairs
    assert set(pair_counts.values()) == {1}
    assert len(demands) == len(expected_pairs)
    assert {d.bits for d in demands} == {cfg.message_size_mb * 8_000_000.0}


def test_sparse_random_uses_ceiling_policy_for_active_source_count():
    cfg = AnalysisConfig(sparse_active_ratio=0.01, sparse_target_count=1, random_seed=7)
    g = build_topology("2D-Torus", cfg)
    ssu_count = sum(1 for _, data in g.nodes(data=True) if data["node_role"] == "ssu")
    demands = build_sparse_random_demands(g, cfg)
    active_sources = {d.src for d in demands}

    assert len(active_sources) == math.ceil(ssu_count * cfg.sparse_active_ratio)


def test_sparse_random_uses_unique_non_self_targets_per_source():
    cfg = AnalysisConfig(sparse_active_ratio=0.25, sparse_target_count=3, random_seed=7)
    g = build_topology("2D-Torus", cfg)
    demands = build_sparse_random_demands(g, cfg)

    targets_by_src: dict[str, list[str]] = defaultdict(list)
    for demand in demands:
        targets_by_src[demand.src].append(demand.dst)
        assert demand.src != demand.dst
        assert demand.bits == cfg.message_size_mb * 8_000_000.0

    assert len(targets_by_src) > 0
    assert all(len(targets) == cfg.sparse_target_count for targets in targets_by_src.values())
    assert all(len(set(targets)) == len(targets) for targets in targets_by_src.values())


def test_sparse_random_is_deterministic_with_same_seed():
    cfg = AnalysisConfig(sparse_active_ratio=0.25, sparse_target_count=2, random_seed=19)
    g = build_topology("2D-Torus", cfg)

    demands_a = build_sparse_random_demands(g, cfg)
    demands_b = build_sparse_random_demands(g, cfg)

    assert demands_a == demands_b


@pytest.mark.parametrize("message_size_mb", [0.0, -1.0])
def test_a2a_rejects_non_positive_message_size(message_size_mb: float):
    cfg = AnalysisConfig(message_size_mb=message_size_mb)
    g = build_topology("2D-FullMesh", cfg)

    with pytest.raises(ValueError, match="message_size_mb"):
        build_a2a_demands(g, cfg)


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1])
def test_sparse_random_rejects_invalid_active_ratio(ratio: float):
    cfg = AnalysisConfig(sparse_active_ratio=ratio)
    g = build_topology("2D-Torus", cfg)

    with pytest.raises(ValueError, match="sparse_active_ratio"):
        build_sparse_random_demands(g, cfg)


def test_sparse_random_rejects_negative_target_count():
    cfg = AnalysisConfig(sparse_target_count=-1)
    g = build_topology("2D-Torus", cfg)

    with pytest.raises(ValueError, match="sparse_target_count"):
        build_sparse_random_demands(g, cfg)


@pytest.mark.parametrize("target_count", [1.5, True, False])
def test_sparse_random_rejects_non_integer_target_count(target_count: object):
    cfg = AnalysisConfig(sparse_target_count=target_count)
    g = build_topology("2D-Torus", cfg)

    with pytest.raises(ValueError, match="sparse_target_count must be an integer >= 0"):
        build_sparse_random_demands(g, cfg)


def test_sparse_random_allows_zero_target_count_and_emits_no_demands():
    cfg = AnalysisConfig(sparse_active_ratio=1.0, sparse_target_count=0, random_seed=11)
    g = build_topology("2D-Torus", cfg)

    demands = build_sparse_random_demands(g, cfg)

    assert demands == []


def test_sparse_random_truncates_target_count_above_available_peers_deterministically():
    cfg = AnalysisConfig(sparse_active_ratio=1.0, sparse_target_count=999, random_seed=13)
    g = build_topology("2D-Torus", cfg)

    demands_a = build_sparse_random_demands(g, cfg)
    demands_b = build_sparse_random_demands(g, cfg)

    ssus = [node_id for node_id, data in g.nodes(data=True) if data["node_role"] == "ssu"]
    expected_targets_per_source = len(ssus) - 1
    targets_by_src: dict[str, list[str]] = defaultdict(list)

    for demand in demands_a:
        targets_by_src[demand.src].append(demand.dst)

    assert demands_a == demands_b
    assert set(targets_by_src) == set(ssus)
    assert all(len(targets) == expected_targets_per_source for targets in targets_by_src.values())
    assert all(len(set(targets)) == expected_targets_per_source for targets in targets_by_src.values())
