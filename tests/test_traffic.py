from collections import Counter, defaultdict
import math
from pathlib import Path

import pytest

from topo_sim.config import AnalysisConfig
from topo_sim.topologies import build_topology
from topo_sim.traffic import (
    build_a2a_demands,
    build_controlled_m2n_demands,
    build_npu_write_local_1to1_demands,
    build_npu_write_local_1to1_pooling_demands,
    build_npu_write_local_1to1_sharding_demands,
    build_npu_write_rack_target_set_direct_demands,
    build_npu_write_rack_target_set_pooling_demands,
    build_npu_write_rack_target_set_sharding_demands,
    build_npu_write_single_ssu_hotspot_direct_demands,
    build_npu_write_single_ssu_hotspot_pooling_demands,
    build_npu_write_single_ssu_hotspot_sharding_demands,
    build_rack_stripe_random_demands,
    build_rack_stripe_topology_aware_demands,
    build_replica3_random_demands,
    build_replica3_topology_aware_demands,
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


@pytest.mark.parametrize("topology_name", ["Clos", "2D-FullMesh", "2D-Torus", "3D-Torus", "DF"])
def test_replica3_random_uses_two_distinct_remote_exchange_nodes(topology_name: str):
    cfg = AnalysisConfig(random_seed=23)
    g = build_topology(topology_name, cfg)
    demands = build_replica3_random_demands(g, cfg)
    ssus = [node_id for node_id, data in g.nodes(data=True) if data["node_role"] == "ssu"]
    targets_by_src: dict[str, list[str]] = defaultdict(list)

    for demand in demands:
        targets_by_src[demand.src].append(demand.dst)
        assert demand.bits == cfg.message_size_mb * 8_000_000.0

    assert set(targets_by_src) == set(ssus)
    assert all(len(targets) == 2 for targets in targets_by_src.values())
    for src, targets in targets_by_src.items():
        exchange_ids = {src.split(":", 1)[0], *(target.split(":", 1)[0] for target in targets)}
        assert len(exchange_ids) == 3


@pytest.mark.parametrize("topology_name", ["Clos", "2D-FullMesh", "2D-Torus", "3D-Torus", "DF"])
def test_replica3_topology_aware_uses_two_distinct_remote_exchange_nodes(topology_name: str):
    cfg = AnalysisConfig()
    g = build_topology(topology_name, cfg)
    demands = build_replica3_topology_aware_demands(g, cfg)
    ssus = [node_id for node_id, data in g.nodes(data=True) if data["node_role"] == "ssu"]
    targets_by_src: dict[str, list[str]] = defaultdict(list)

    for demand in demands:
        targets_by_src[demand.src].append(demand.dst)

    assert set(targets_by_src) == set(ssus)
    assert all(len(targets) == 2 for targets in targets_by_src.values())
    for src, targets in targets_by_src.items():
        exchange_ids = {src.split(":", 1)[0], *(target.split(":", 1)[0] for target in targets)}
        assert len(exchange_ids) == 3


def test_replica3_topology_aware_torus_prefers_one_hop_neighbor_exchanges():
    cfg = AnalysisConfig()
    g = build_topology("2D-Torus", cfg)
    demands = build_replica3_topology_aware_demands(g, cfg)
    targets_by_src: dict[str, list[str]] = defaultdict(list)
    coord_by_exchange = {
        node_id.split(":", 1)[0]: tuple(data["exchange_grid_coord"])
        for node_id, data in g.nodes(data=True)
        if data.get("node_role") == "ssu"
    }
    shape = tuple(int(value) for value in g.graph["torus_exchange_grid_shape"])

    for demand in demands:
        targets_by_src[demand.src].append(demand.dst)

    for src, targets in targets_by_src.items():
        src_coord = coord_by_exchange[src.split(":", 1)[0]]
        for target in targets:
            dst_coord = coord_by_exchange[target.split(":", 1)[0]]
            axis_distances = [
                min((a - b) % size, (b - a) % size)
                for a, b, size in zip(src_coord, dst_coord, shape)
            ]
            assert sum(axis_distances) == 1


def test_rack_stripe_random_uses_requested_source_count_and_target_count():
    cfg = AnalysisConfig(
        enable_rack_stripe_workloads=True,
        rack_stripe_target_count=4,
        message_size_mb=8.0,
        random_seed=11,
    )
    g = build_topology("3D-Torus", cfg)

    demands = build_rack_stripe_random_demands(g, cfg, source_count=8)
    targets_by_src: dict[str, list[str]] = defaultdict(list)

    for demand in demands:
        targets_by_src[demand.src].append(demand.dst)

    assert len(targets_by_src) == 8
    assert all(len(targets) == 4 for targets in targets_by_src.values())
    assert all(len(set(targets)) == 4 for targets in targets_by_src.values())
    assert {demand.bits for demand in demands} == {cfg.message_size_mb * 8_000_000.0 / 4.0}


def test_rack_stripe_topology_aware_spreads_targets_across_exchange_groups():
    cfg = AnalysisConfig(
        enable_rack_stripe_workloads=True,
        rack_stripe_target_count=4,
        random_seed=17,
    )
    g = build_topology("2D-FullMesh", cfg)

    demands = build_rack_stripe_topology_aware_demands(g, cfg, source_count=8)
    targets_by_src: dict[str, list[str]] = defaultdict(list)
    for demand in demands:
        targets_by_src[demand.src].append(demand.dst)

    assert len(targets_by_src) == 8
    for targets in targets_by_src.values():
        exchange_ids = {target.split(":", 1)[0] for target in targets}
        assert len(targets) == 4
        assert len(exchange_ids) == 4


def test_npu_local_1to1_uses_local_exchange_targets_with_even_union_split():
    cfg = AnalysisConfig(enable_npu_write_workloads=True, message_size_mb=8.0)
    g = build_topology("2D-Torus", cfg)

    demands = build_npu_write_local_1to1_demands(g, cfg, source_count=64)
    sources = {demand.source_id for demand in demands}

    assert len(sources) == 64
    assert all(demand.source_id is not None for demand in demands)
    assert all(g.nodes[demand.src]["node_role"] == "dpu" for demand in demands)
    assert all(g.nodes[demand.dst]["node_role"] == "ssu" for demand in demands)
    assert all(demand.explicit_paths is not None and len(demand.explicit_paths) == 2 for demand in demands)
    for demand in demands[:8]:
        assert demand.explicit_paths is not None
        for path in demand.explicit_paths:
            assert g.nodes[path[0]]["node_role"] == "npu"
            assert path[1] == demand.src
            assert path[-1] == demand.dst
            assert path[0].split(":", 1)[0] == demand.dst.split(":", 1)[0]
    assert {demand.bits for demand in demands} == {cfg.message_size_mb * 8_000_000.0}


def test_npu_local_1to1_pooling_uses_same_source_rack_union_pool():
    cfg = AnalysisConfig(enable_npu_write_workloads=True)
    g = build_topology("2D-Torus", cfg)

    direct_demands = build_npu_write_local_1to1_demands(g, cfg, source_count=64)
    pooled_demands = build_npu_write_local_1to1_pooling_demands(g, cfg, source_count=64)

    assert len(pooled_demands) == len(direct_demands) == 64
    direct_by_source = {str(demand.source_id): demand for demand in direct_demands}
    for demand in pooled_demands:
        direct_match = direct_by_source[str(demand.source_id)]
        assert demand.dst == direct_match.dst
        assert demand.explicit_paths is not None
        assert direct_match.explicit_paths is not None
        assert len(demand.explicit_paths) >= len(direct_match.explicit_paths)
        assert any(len(path) > len(direct_match.explicit_paths[0]) for path in demand.explicit_paths)


def test_npu_local_1to1_sharding_splits_across_four_source_rack_targets():
    cfg = AnalysisConfig(enable_npu_write_workloads=True, message_size_mb=16.0)
    g = build_topology("3D-Torus", cfg)

    demands = build_npu_write_local_1to1_sharding_demands(g, cfg, source_count=64)
    targets_by_source: dict[str, list[str]] = defaultdict(list)
    bits_by_source: dict[str, list[float]] = defaultdict(list)
    for demand in demands:
        assert demand.source_id is not None
        targets_by_source[str(demand.source_id)].append(demand.dst)
        bits_by_source[str(demand.source_id)].append(demand.bits)
        assert demand.explicit_paths is not None
        assert len(demand.explicit_paths) == 2

    assert len(targets_by_source) == 64
    assert all(len(targets) == 4 for targets in targets_by_source.values())
    assert all(len({target.split(':', 1)[0] for target in targets}) == 4 for targets in targets_by_source.values())
    assert all(all(bits == 4.0 * 8_000_000.0 for bits in bits_list) for bits_list in bits_by_source.values())


def test_npu_single_ssu_hotspot_pooling_clos_paths_can_traverse_leaf_switches():
    cfg = AnalysisConfig(enable_npu_write_workloads=True)
    g = build_topology("Clos-4P-FullMesh", cfg)

    demands = build_npu_write_single_ssu_hotspot_pooling_demands(g, cfg, source_count=64)

    assert demands
    assert any(
        "clos_leaf" in node_id
        for demand in demands
        for path in (demand.explicit_paths or ())
        for node_id in path
    )


def test_npu_single_ssu_hotspot_direct_uses_one_fixed_target():
    cfg = AnalysisConfig(enable_npu_write_workloads=True)
    g = build_topology("2D-Torus", cfg)

    demands = build_npu_write_single_ssu_hotspot_direct_demands(g, cfg, source_count=64)

    assert len(demands) == 64
    assert len({demand.dst for demand in demands}) == 1
    assert all(demand.explicit_paths is not None and len(demand.explicit_paths) == 2 for demand in demands)
    assert all(all(path[-1] == demand.dst for path in demand.explicit_paths or ()) for demand in demands)


def test_npu_single_ssu_hotspot_pooling_uses_target_rack_union_pool():
    cfg = AnalysisConfig(enable_npu_write_workloads=True)
    g = build_topology("2D-Torus", cfg)

    direct_demands = build_npu_write_single_ssu_hotspot_direct_demands(g, cfg, source_count=64)
    pooled_demands = build_npu_write_single_ssu_hotspot_pooling_demands(g, cfg, source_count=64)

    demands_by_source: dict[str, list] = defaultdict(list)
    for demand in pooled_demands:
        assert demand.source_id is not None
        demands_by_source[demand.source_id].append(demand)

    assert all(len(source_demands) == 1 for source_demands in demands_by_source.values())
    for source_demands in demands_by_source.values():
        demand = source_demands[0]
        assert len({demand.dst for demand in source_demands}) == 1
        assert demand.explicit_paths is not None
        direct_match = next(item for item in direct_demands if item.source_id == demand.source_id)
        assert direct_match.explicit_paths is not None
        assert len(demand.explicit_paths) > len(direct_match.explicit_paths)
        assert any(len(path) > 4 for path in demand.explicit_paths)
        assert math.isclose(
            sum(demand.bits for demand in source_demands),
            cfg.message_size_mb * 8_000_000.0,
        )
    assert pooled_demands[0].dst == direct_demands[0].dst
    for demand in pooled_demands[:8]:
        assert demand.explicit_paths is not None
        path = demand.explicit_paths[0]
        assert g.nodes[path[0]]["node_role"] == "npu"
        assert g.nodes[path[1]]["node_role"] == "dpu"
        assert path[1] == demand.src
        assert path[-1] == demand.dst
        assert len(path) >= 4


def test_npu_single_ssu_hotspot_sharding_splits_each_source_evenly_across_four_targets():
    cfg = AnalysisConfig(enable_npu_write_workloads=True, message_size_mb=16.0)
    g = build_topology("3D-Torus", cfg)

    demands = build_npu_write_single_ssu_hotspot_sharding_demands(g, cfg, source_count=64)
    targets_by_source: dict[str, list[str]] = defaultdict(list)
    bits_by_source: dict[str, list[float]] = defaultdict(list)
    for demand in demands:
        assert demand.source_id is not None
        targets_by_source[demand.source_id].append(demand.dst)
        bits_by_source[demand.source_id].append(demand.bits)
        assert g.nodes[demand.src]["node_role"] == "dpu"
        assert demand.explicit_paths is not None
        assert len(demand.explicit_paths) == 2

    assert len(targets_by_source) == 64
    assert all(len(targets) == 4 for targets in targets_by_source.values())
    assert all(len({target.split(':', 1)[0] for target in targets}) == 4 for targets in targets_by_source.values())
    assert all(all(bits == 4.0 * 8_000_000.0 for bits in bits_list) for bits_list in bits_by_source.values())


def test_npu_rack_target_set_workloads_keep_same_target_set_semantics():
    cfg = AnalysisConfig(enable_npu_write_workloads=True, message_size_mb=16.0)
    g = build_topology("3D-Torus", cfg)

    direct = build_npu_write_rack_target_set_direct_demands(g, cfg, source_count=64)
    pooled = build_npu_write_rack_target_set_pooling_demands(g, cfg, source_count=64)
    sharded = build_npu_write_rack_target_set_sharding_demands(g, cfg, source_count=64)

    assert len(direct) == 64
    assert len(pooled) == 64
    assert len(sharded) == 64 * 4
    assert len({demand.dst for demand in direct}) == 4
    assert len({demand.dst for demand in pooled}) == 4
    assert len({demand.dst for demand in sharded}) == 4
    assert all(demand.explicit_paths is not None and len(demand.explicit_paths) == 2 for demand in direct)
    assert all(demand.explicit_paths is not None and len(demand.explicit_paths) >= 8 for demand in pooled)
    sharded_targets: dict[str, list[str]] = defaultdict(list)
    for demand in sharded:
        sharded_targets[str(demand.source_id)].append(demand.dst)
    assert all(len(targets) == 4 for targets in sharded_targets.values())


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
