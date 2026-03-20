import csv
import json
import shutil
import uuid
from pathlib import Path

import pytest

from topo_sim.config import AnalysisConfig
from topo_sim.pipeline import _routing_configuration, _routing_diversity_snapshot, run_full_analysis
import topo_sim.visualization as visualization


EXPECTED_SUMMARY_HEADERS = [
    "topology",
    "diameter",
    "average_hops",
    "bisection_bandwidth_gbps",
    "bisection_bandwidth_gbps_per_ssu",
    "a2a_completion_time_s",
    "a2a_completion_time_p50_s",
    "a2a_completion_time_p95_s",
    "a2a_per_ssu_throughput_gbps",
    "a2a_max_link_utilization",
    "a2a_link_utilization_cv",
    "sparse_completion_time_s",
    "sparse_completion_time_p50_s",
    "sparse_completion_time_p95_s",
    "sparse_per_ssu_throughput_gbps",
    "sparse_max_link_utilization",
    "sparse_link_utilization_cv",
]


@pytest.fixture
def output_dir() -> Path:
    base = Path('.tmp_tests')
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_pipeline_writes_new_metric_columns(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir)
    paths = run_full_analysis(cfg, ["2D-FullMesh"])

    with paths["csv"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == EXPECTED_SUMMARY_HEADERS
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["topology"] == "2D-FullMesh"


def test_pipeline_writes_routing_and_workload_config(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir, routing_mode="PORT_BALANCED")
    paths = run_full_analysis(cfg, ["Clos"])

    payload = json.loads(paths["config"].read_text(encoding="utf-8"))
    assert payload["routing_mode"] == "PORT_BALANCED"
    assert payload["sparse_active_ratio"] == 0.25
    assert payload["selected_topologies"] == ["Clos"]


def test_dashboard_and_report_use_new_labels(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir, routing_mode="PORT_BALANCED")
    paths = run_full_analysis(cfg, ["2D-Torus"])

    html = paths["html"].read_text(encoding="utf-8")
    assert "A2A Per SSU Throughput" in html
    assert "Bisection BW / SSU" in html
    assert "Sparse 1-to-N" in html
    assert "Routing Mode" in html
    assert "same-exchange SSU traffic stays inside the exchange node via Union switching" in html
    assert "source SSU traffic evenly splits across both local 200 Gbps uplinks into the two Union planes" in html
    assert "each Union plane can use every available backend egress port" in html
    assert "FULL_PATH" in html
    assert "PORT_BALANCED" not in html
    assert "A2Av Efficiency" not in html

    assert paths["pdf"].exists()
    assert paths["pdf"].stat().st_size > 0
    pdf_text = paths["pdf"].read_bytes().decode("latin1", errors="ignore")
    assert "FULL_PATH" in pdf_text
    assert "PORT_BALANCED" not in pdf_text
    assert "same-exchange SSU traffic stays inside the exchange node via Union switching" in pdf_text


def test_route_notes_distinguish_dor_shortest_path_and_full_path_for_torus():
    dor_2d = _routing_configuration("2D-Torus", AnalysisConfig(routing_mode="DOR"))
    dor_3d = _routing_configuration("3D-Torus", AnalysisConfig(routing_mode="DOR"))
    shortest_2d = _routing_configuration("2D-Torus", AnalysisConfig(routing_mode="SHORTEST_PATH"))
    full_2d = _routing_configuration("2D-Torus", AnalysisConfig(routing_mode="FULL_PATH"))
    full_3d = _routing_configuration("3D-Torus", AnalysisConfig(routing_mode="FULL_PATH"))

    assert any(
        "X -> Y order" in note
        for note in dor_2d["notes"]
    )
    assert any(
        "X -> Y -> Z order" in note
        for note in dor_3d["notes"]
    )
    assert any(
        "all shortest Union-to-Union paths without a fixed dimension order" in note
        for note in shortest_2d["notes"]
    )
    assert any(
        "uniformly splits across all available backend egress ports" in note
        for note in full_2d["notes"]
    )
    assert any(
        "one path per egress port" in note
        for note in full_3d["notes"]
    )
    assert any(
        "if a selected egress has no shortest path, routing falls back to the least-hop non-shortest path" in note
        for note in full_2d["notes"]
    )
    assert not any("X -> Y order" in note for note in shortest_2d["notes"])



def test_routing_diversity_snapshot_quantifies_path_diversity():
    from topo_sim.topologies import build_topology

    cfg = AnalysisConfig(routing_mode="SHORTEST_PATH")
    g = build_topology("2D-Torus", cfg)
    snapshot = _routing_diversity_snapshot("2D-Torus", g, cfg)

    assert snapshot is not None
    assert snapshot["summary"]
    by_mode = {item["mode"]: item for item in snapshot["modes"]}
    assert {"DOR", "SHORTEST_PATH", "FULL_PATH"}.issubset(by_mode)
    assert by_mode["SHORTEST_PATH"]["avg_path_count"] > by_mode["DOR"]["avg_path_count"]
    assert by_mode["FULL_PATH"]["avg_path_count"] > by_mode["DOR"]["avg_path_count"]
    assert "SHORTEST_PATH vs DOR" in snapshot["summary"]


def test_dashboard_includes_routing_diversity_snapshot(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir, routing_mode="SHORTEST_PATH")
    paths = run_full_analysis(cfg, ["2D-Torus"])

    html = paths["html"].read_text(encoding="utf-8")
    assert "Routing Diversity Snapshot" in html
    assert "SHORTEST_PATH vs DOR" in html


def test_direct_topology_outputs_compact_routing_comparison_sections(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir, routing_mode="DOR")
    paths = run_full_analysis(cfg, ["2D-Torus"])

    html = paths["html"].read_text(encoding="utf-8")
    assert "A2A Routing Comparison" in html
    assert "Sparse 1-to-N Routing Comparison" in html
    assert "Per SSU Throughput" in html
    assert "Completion Time" in html
    assert "P95 Completion" in html
    assert "Max Link Utilization" in html
    assert "Link Utilization CV" in html
    assert "DOR" in html
    assert "SHORTEST_PATH" in html
    assert "FULL_PATH" in html
    assert "Default Route Throughput" in html
    assert "DOR Throughput" in html

    pdf_text = paths["pdf"].read_bytes().decode("latin1", errors="ignore")
    assert "A2A Routing Comparison" in pdf_text
    assert "Sparse 1-to-N Routing Comparison" in pdf_text
    assert "Default Route Throughput" in pdf_text
    assert "DOR Throughput" in pdf_text


def test_clos_outputs_only_ecmp_without_direct_routing_comparison_sections(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir, routing_mode="ECMP")
    paths = run_full_analysis(cfg, ["Clos"])

    html = paths["html"].read_text(encoding="utf-8")
    assert "A2A Routing Comparison" not in html
    assert "Sparse 1-to-N Routing Comparison" not in html
    assert "Default Route Throughput" in html
    assert "ECMP Throughput" in html
    assert "DOR Throughput" not in html

    pdf_text = paths["pdf"].read_bytes().decode("latin1", errors="ignore")
    assert "A2A Routing Comparison" not in pdf_text
    assert "Sparse 1-to-N Routing Comparison" not in pdf_text
    assert "ECMP Throughput" in pdf_text


def test_dashboard_click_interaction_payload_includes_incident_bandwidth_labels(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir, routing_mode="DOR")
    paths = run_full_analysis(cfg, ["Clos"])

    html = paths["html"].read_text(encoding="utf-8")
    assert "Click a node to highlight one-hop neighbors, incident links, and link bandwidth" in html
    assert "incident_link_labels" in html
    assert "bandwidth_gbps" in html

def test_dashboard_uses_black_theme_and_interaction_hooks(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir)
    paths = run_full_analysis(cfg, ["2D-Torus", "Clos"])

    html = paths["html"].read_text(encoding="utf-8")
    assert "plot-card plot-card-dark" in html
    assert "data-highlight-mode=\"neighbors\"" in html
    assert "registerTopologyHighlight" in html
    assert html.index("Topology Figure") < html.index("Routing Mode")


def test_dashboard_mentions_layered_layout_rules(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir, routing_mode="DOR")
    paths = run_full_analysis(cfg, ["3D-Torus", "Clos"])

    html = paths["html"].read_text(encoding="utf-8")
    assert "4 z-layers" in html
    assert "Clos spine layer" in html
    assert "SSUs stay on the bottom row" in html


def test_dashboard_layout_falls_back_when_spring_layout_needs_scipy(output_dir: Path, monkeypatch):
    def raising_spring_layout(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'scipy'")

    monkeypatch.setattr(visualization.nx, "spring_layout", raising_spring_layout)

    cfg = AnalysisConfig(output_dir=output_dir, routing_mode="DOR")
    paths = run_full_analysis(cfg, ["3D-Torus"])

    assert paths["html"].exists()
    html = paths["html"].read_text(encoding="utf-8")
    assert "3D-Torus" in html
