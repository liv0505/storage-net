from collections import Counter, defaultdict
import math
from pathlib import Path

import pytest

from topo_sim.config import AnalysisConfig
from topo_sim.topologies import build_topology
from topo_sim.traffic import (
    build_a2a_demands,
    build_controlled_m2n_demands,
    build_sparse_random_demands,
    load_custom_traffic_profile,
    select_ssu_nodes,
)


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


def test_df_a2a_demands_cover_all_shared_ssus_across_both_planes():
    cfg = AnalysisConfig(message_size_mb=2.5)
    g = build_topology("DF", cfg)
    demands = build_a2a_demands(g, cfg)

    total_ssu_count = sum(1 for _, data in g.nodes(data=True) if data["node_role"] == "ssu")
    expected_pairs = total_ssu_count * (total_ssu_count - 1)

    assert len(demands) == expected_pairs


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


def test_df_sparse_random_targets_can_span_the_shared_df_machine():
    cfg = AnalysisConfig(sparse_active_ratio=0.25, sparse_target_count=2, random_seed=19)
    g = build_topology("DF", cfg)

    demands = build_sparse_random_demands(g, cfg)

    assert demands
    assert all(demand.src != demand.dst for demand in demands)


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


def test_select_ssu_nodes_supports_server_exchange_and_local_filters():
    cfg = AnalysisConfig()
    g = build_topology("DF", cfg)

    selected = select_ssu_nodes(
        g,
        server_ids=[0],
        local_indices=[0, 1],
        limit=3,
    )

    assert selected == ["en0:ssu0", "en0:ssu1", "en1:ssu0"]

    exchange_selected = select_ssu_nodes(
        g,
        exchange_ids=["en1"],
        local_indices=[6, 7],
    )
    assert exchange_selected == ["en1:ssu6", "en1:ssu7"]


def test_controlled_m2n_builder_supports_pair_level_volume_overrides():
    cfg = AnalysisConfig(message_size_mb=2.0)
    g = build_topology("2D-Torus", cfg)
    sources = select_ssu_nodes(g, exchange_ids=["en0"], local_indices=[0, 1])
    destinations = select_ssu_nodes(g, exchange_ids=["en5"], local_indices=[0, 1])
    message_bits = cfg.message_size_mb * 8_000_000.0

    demands = build_controlled_m2n_demands(
        g,
        cfg,
        source_ssus=sources,
        destination_ssus=destinations,
        pair_bits={
            ("en0:ssu0", "en5:ssu1"): message_bits * 3,
        },
    )

    pair_to_bits = {(demand.src, demand.dst): demand.bits for demand in demands}
    assert len(demands) == 4
    assert pair_to_bits[("en0:ssu0", "en5:ssu0")] == pytest.approx(message_bits)
    assert pair_to_bits[("en0:ssu0", "en5:ssu1")] == pytest.approx(message_bits * 3)
    assert pair_to_bits[("en0:ssu1", "en5:ssu0")] == pytest.approx(message_bits)
    assert pair_to_bits[("en0:ssu1", "en5:ssu1")] == pytest.approx(message_bits)


def test_load_custom_traffic_profile_from_csv(tmp_path: Path):
    cfg = AnalysisConfig(message_size_mb=2.0)
    g = build_topology("2D-Torus", cfg)
    traffic_path = tmp_path / "custom.csv"
    traffic_path.write_text(
        "src,dst,gb\n"
        "en0:ssu0,en5:ssu0,4\n"
        "en0:ssu1,en5:ssu1,2\n",
        encoding="utf-8",
    )

    profile = load_custom_traffic_profile(g, traffic_path, workload_name="CSV Traffic")

    assert profile.name == "CSV Traffic"
    assert len(profile.demands) == 2
    assert profile.demands[0].bits == pytest.approx(32_000_000_000.0)
    assert profile.demands[1].bits == pytest.approx(16_000_000_000.0)


def test_load_custom_traffic_profile_from_json_groups_and_overrides(tmp_path: Path):
    cfg = AnalysisConfig()
    g = build_topology("DF", cfg)
    traffic_path = tmp_path / "custom.json"
    traffic_path.write_text(
        """
        {
          "name": "JSON Traffic",
          "groups": [
            {
              "sources": {"exchange_ids": ["en0"], "local_indices": [0, 1]},
              "destinations": {"exchange_ids": ["en2"], "local_indices": [0, 1]},
              "default_gb": 1,
              "pair_overrides": [
                {"src": "en0:ssu0", "dst": "en2:ssu0", "gb": 3}
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    profile = load_custom_traffic_profile(g, traffic_path)
    pair_to_bits = {(d.src, d.dst): d.bits for d in profile.demands}

    assert profile.name == "JSON Traffic"
    assert len(profile.demands) == 4
    assert pair_to_bits[("en0:ssu0", "en2:ssu0")] == pytest.approx(24_000_000_000.0)
    assert pair_to_bits[("en0:ssu0", "en2:ssu1")] == pytest.approx(8_000_000_000.0)
