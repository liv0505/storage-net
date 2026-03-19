import csv
import json
import shutil
import uuid
from pathlib import Path

import pytest

from topo_sim.config import AnalysisConfig
from topo_sim.pipeline import run_full_analysis
import topo_sim.visualization as visualization


EXPECTED_SUMMARY_HEADERS = [
    "topology",
    "diameter",
    "average_hops",
    "bisection_bandwidth_gbps",
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
    assert "Sparse 1-to-N" in html
    assert "Routing Mode" in html
    assert "same-exchange SSU traffic stays inside the exchange node via Union switching" in html
    assert "A2Av Efficiency" not in html

    assert paths["pdf"].exists()
    assert paths["pdf"].stat().st_size > 0


def test_dashboard_uses_black_theme_and_interaction_hooks(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir)
    paths = run_full_analysis(cfg, ["2D-Torus", "Clos"])

    html = paths["html"].read_text(encoding="utf-8")
    assert "plot-card plot-card-dark" in html
    assert "data-highlight-mode=\"neighbors\"" in html
    assert "registerTopologyHighlight" in html
    assert html.index("Topology Figure") < html.index("Routing Mode")


def test_dashboard_mentions_layered_layout_rules(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir)
    paths = run_full_analysis(cfg, ["3D-Torus", "Clos"])

    html = paths["html"].read_text(encoding="utf-8")
    assert "4 z-layers" in html
    assert "Clos spine layer" in html
    assert "SSUs stay on the bottom row" in html


def test_dashboard_layout_falls_back_when_spring_layout_needs_scipy(output_dir: Path, monkeypatch):
    def raising_spring_layout(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'scipy'")

    monkeypatch.setattr(visualization.nx, "spring_layout", raising_spring_layout)

    cfg = AnalysisConfig(output_dir=output_dir)
    paths = run_full_analysis(cfg, ["3D-Torus"])

    assert paths["html"].exists()
    html = paths["html"].read_text(encoding="utf-8")
    assert "3D-Torus" in html
