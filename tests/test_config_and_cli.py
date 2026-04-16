import argparse
from pathlib import Path

import pytest

import main as main_module
from main import parse_args
from topo_sim.config import AnalysisConfig


def test_analysis_config_defaults_match_ssu_design():
    cfg = AnalysisConfig()
    assert cfg.topology_names == [
        "2D-FullMesh",
        "2D-FullMesh-2x4",
        "2D-Torus",
        "2D-Torus-BestTwist",
        "3D-Torus",
        "3D-Torus-BestTwist",
        "3D-Torus-2x4x3",
        "3D-Torus-2x4x3-BestTwist",
        "3D-Torus-2x4x2",
        "3D-Torus-2x4x2-BestTwist",
        "3D-Torus-2x4x1",
        "3D-Torus-2x4x1-BestTwist",
        "Clos",
        "DF",
        "SparseMesh-Local",
        "SparseMesh-Global",
    ]
    assert cfg.sparse_active_ratio == 0.25
    assert cfg.sparse_target_count == 2
    assert cfg.port_balanced_max_detour_hops == 1
    assert cfg.clos_uplinks_per_exchange_node == 4
    assert cfg.df_unions_per_server == 4
    assert cfg.df_external_servers_per_union == 3
    assert cfg.custom_traffic_file is None
    assert cfg.custom_traffic_name == "Custom M-to-N"


def test_parse_args_accepts_new_topology_names(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--topologies",
            "2D-FullMesh,Clos",
            "--routing-mode",
            "PORT_BALANCED",
            "--sparse-active-ratio",
            "0.5",
        ],
    )
    args = parse_args()
    assert args.topologies == "2D-FullMesh,Clos"
    assert args.routing_mode == "PORT_BALANCED"
    assert args.sparse_active_ratio == 0.5


def test_parse_args_accepts_df_topology(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--topologies", "DF"])
    args = parse_args()
    assert args.topologies == "DF"


def test_parse_args_accepts_best_twist_torus_variants(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--topologies", "2D-Torus-BestTwist,3D-Torus-BestTwist"],
    )
    args = parse_args()
    assert args.topologies == "2D-Torus-BestTwist,3D-Torus-BestTwist"


def test_parse_args_accepts_new_fullmesh_and_3d_torus_variants(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--topologies",
            "2D-FullMesh-2x4,3D-Torus-2x4x3,3D-Torus-2x4x1-BestTwist",
        ],
    )
    args = parse_args()
    assert args.topologies == "2D-FullMesh-2x4,3D-Torus-2x4x3,3D-Torus-2x4x1-BestTwist"


def test_parse_args_accepts_df_variants(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--topologies", "DF-Shuffled,DF-ScaleUp"])
    args = parse_args()
    assert args.topologies == "DF-Shuffled,DF-ScaleUp"


def test_parse_args_accepts_sparsemesh_variants(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--topologies", "SparseMesh-Local,SparseMesh-Global"],
    )
    args = parse_args()
    assert args.topologies == "SparseMesh-Local,SparseMesh-Global"


def test_parse_args_accepts_2p_df_variants(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--topologies", "DF-2P-Double-4Global,DF-2P-Double-Bridge-3Global"],
    )
    args = parse_args()
    assert args.topologies == "DF-2P-Double-4Global,DF-2P-Double-Bridge-3Global"


def test_parse_args_accepts_custom_traffic_file_and_name(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--custom-traffic-file",
            "traffic/custom.json",
            "--custom-traffic-name",
            "Hotspot",
        ],
    )
    args = parse_args()
    assert args.custom_traffic_file == "traffic/custom.json"
    assert args.custom_traffic_name == "Hotspot"


@pytest.mark.parametrize("routing_mode", ["DOR", "ECMP", "SHORTEST_PATH", "FULL_PATH", "MIN_HOPS", "PORT_BALANCED"])
def test_parse_args_accepts_supported_routing_modes(monkeypatch, routing_mode):
    monkeypatch.setattr("sys.argv", ["main.py", "--routing-mode", routing_mode])
    args = parse_args()
    assert args.routing_mode == routing_mode


def test_parse_args_default_topologies_match_analysis_config(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py"])
    args = parse_args()
    assert args.topologies == ",".join(AnalysisConfig().topology_names)


def test_parse_args_default_routing_and_sparse_match_analysis_config(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py"])
    args = parse_args()
    cfg = AnalysisConfig()
    assert args.routing_mode == cfg.routing_mode
    assert args.sparse_active_ratio == cfg.sparse_active_ratio


def test_parse_args_rejects_invalid_routing_mode(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--routing-mode", "NOT_A_MODE"])
    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize("value", ["0", "-0.1", "1.5"])
def test_parse_args_rejects_invalid_sparse_active_ratio(monkeypatch, value):
    monkeypatch.setattr("sys.argv", ["main.py", "--sparse-active-ratio", value])
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_non_finite_sparse_active_ratio(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--sparse-active-ratio", "nan"])
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_invalid_topology_name(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--topologies", "2D-FullMesh,Unknown"])
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_empty_topologies(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--topologies", " , , "])
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_non_positive_message_size(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--message-size-mb", "0"])
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_non_finite_message_size(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--message-size-mb", "nan"])
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_non_positive_bandwidth(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--bandwidth-gbps", "0"])
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_non_finite_bandwidth(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--bandwidth-gbps", "inf"])
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_non_positive_traffic_samples(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--traffic-samples", "0"])
    with pytest.raises(SystemExit):
        parse_args()


def test_main_passes_cli_values_into_config_and_run_full_analysis(monkeypatch):
    captured: dict[str, object] = {}

    def fake_parse_args() -> argparse.Namespace:
        return argparse.Namespace(
            topologies="2D-FullMesh,Clos",
            routing_mode="FULL_PATH",
            sparse_active_ratio=0.5,
            sparse_target_count=3,
            port_balanced_max_detour_hops=2,
            clos_uplinks_per_exchange_node=4,
            df_unions_per_server=4,
            df_external_servers_per_union=3,
            custom_traffic_file="traffic/custom.csv",
            custom_traffic_name="My Traffic",
            message_size_mb=8.0,
            bandwidth_gbps=200.0,
            traffic_samples=123,
            output_dir="outputs/test",
            seed=777,
        )

    def fake_run_full_analysis(cfg, topology_list):
        captured["cfg"] = cfg
        captured["topology_list"] = topology_list
        return {"summary": "outputs/test/summary.html"}

    monkeypatch.setattr(main_module, "parse_args", fake_parse_args)
    monkeypatch.setattr(main_module, "run_full_analysis", fake_run_full_analysis)

    main_module.main()

    cfg = captured["cfg"]
    assert captured["topology_list"] == ["2D-FullMesh", "Clos"]
    assert cfg.link_bandwidth_gbps == 200.0
    assert cfg.message_size_mb == 8.0
    assert cfg.traffic_samples == 123
    assert cfg.random_seed == 777
    assert cfg.output_dir == Path("outputs/test")
    assert cfg.sparse_active_ratio == 0.5
    assert cfg.sparse_target_count == 3
    assert cfg.port_balanced_max_detour_hops == 2
    assert cfg.clos_uplinks_per_exchange_node == 4
    assert cfg.df_unions_per_server == 4
    assert cfg.df_external_servers_per_union == 3
    assert cfg.custom_traffic_file == "traffic/custom.csv"
    assert cfg.custom_traffic_name == "My Traffic"
    assert cfg.routing_mode == "FULL_PATH"
