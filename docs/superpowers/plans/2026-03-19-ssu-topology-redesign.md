# SSU Topology Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current abstract topology toolkit with an SSU-centric model for `2D-FullMesh`, `2D-Torus`, `3D-Torus`, and `Clos`, and emit the new routing-aware metrics and reports.

**Architecture:** Build every topology from a shared `8 SSU + 2 Union` exchange-node template, then layer routing and workload generation on top of the explicit graph so all structural and communication metrics come from the same `SSU -> SSU` model. Keep topology generation, routing, traffic generation, metric calculation, and reporting in separate modules so each concern can be tested independently.

**Tech Stack:** Python 3, `networkx`, `numpy`, `plotly`, `jinja2`, `reportlab`, `pytest`

---

## File Structure

### Files To Create

- `docs/superpowers/plans/2026-03-19-ssu-topology-redesign.md`
- `topo_sim/routing.py`
- `topo_sim/traffic.py`
- `tests/test_config_and_cli.py`
- `tests/test_topologies.py`
- `tests/test_routing.py`
- `tests/test_traffic.py`
- `tests/test_metrics.py`
- `tests/test_pipeline_outputs.py`

### Files To Modify

- `requirements.txt`
- `README.md`
- `main.py`
- `topo_sim/config.py`
- `topo_sim/topologies.py`
- `topo_sim/metrics.py`
- `topo_sim/simulation.py`
- `topo_sim/pipeline.py`
- `topo_sim/report.py`
- `topo_sim/visualization.py`
- `templates/dashboard.html.j2`

### Responsibility Map

- `topo_sim/config.py`: centralize hardware defaults, topology defaults, routing mode, and workload knobs.
- `topo_sim/topologies.py`: expose only the four supported display names and build explicit SSU/Union graphs.
- `topo_sim/routing.py`: compute `DOR`, `PORT_BALANCED`, and `ECMP` path sets and traffic splits.
- `topo_sim/traffic.py`: generate `A2A` and sparse random `1-to-N` SSU demand matrices.
- `topo_sim/simulation.py`: host shared load projection helpers if needed by metrics or pipeline.
- `topo_sim/metrics.py`: compute structural metrics and workload performance metrics from routed traffic.
- `topo_sim/pipeline.py`: orchestrate topology build, route selection, traffic evaluation, and exports.
- `topo_sim/report.py`: write PDF sections for topology, routing, workload, and metrics.
- `topo_sim/visualization.py` and `templates/dashboard.html.j2`: update HTML dashboard labels and cards to the new metric set.
- `main.py`: update CLI defaults and argument surface to the new topology and workload model.
- `tests/*.py`: pin the new behavior with focused test coverage.

### Task 1: Add Test Harness And New Config Surface

**Files:**
- Modify: `requirements.txt`
- Modify: `main.py`
- Modify: `topo_sim/config.py`
- Test: `tests/test_config_and_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from main import parse_args
from topo_sim.config import AnalysisConfig


def test_analysis_config_defaults_match_ssu_design():
    cfg = AnalysisConfig()
    assert cfg.topology_names == ["2D-FullMesh", "2D-Torus", "3D-Torus", "Clos"]
    assert cfg.sparse_active_ratio == 0.25
    assert cfg.sparse_target_count == 2
    assert cfg.port_balanced_max_detour_hops == 1
    assert cfg.clos_uplinks_per_exchange_node == 4


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_and_cli.py -v`
Expected: FAIL with missing config fields or unsupported CLI arguments.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(slots=True)
class AnalysisConfig:
    topology_names: list[str] = field(
        default_factory=lambda: ["2D-FullMesh", "2D-Torus", "3D-Torus", "Clos"]
    )
    sparse_active_ratio: float = 0.25
    sparse_target_count: int = 2
    port_balanced_max_detour_hops: int = 1
    clos_uplinks_per_exchange_node: int = 4
```

- [ ] **Step 4: Update dependencies for TDD**

Add `pytest>=8.0` to `requirements.txt` so the new tests can run in a clean environment.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config_and_cli.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt main.py topo_sim/config.py tests/test_config_and_cli.py
git commit -m "test: add config and cli coverage for ssu topology redesign"
```

If this workspace is still outside a git repository, note that commit is skipped and continue.

### Task 2: Replace Legacy Topology Catalog With Exchange-Node Builders

**Files:**
- Modify: `topo_sim/topologies.py`
- Test: `tests/test_topologies.py`

- [ ] **Step 1: Write the failing tests**

```python
from topo_sim.config import AnalysisConfig
from topo_sim.topologies import available_topologies, build_topology


def test_available_topologies_only_exposes_new_names():
    assert available_topologies() == ["2D-FullMesh", "2D-Torus", "3D-Torus", "Clos"]


def test_2d_fullmesh_builds_exchange_nodes_with_expected_parts():
    g = build_topology("2D-FullMesh", AnalysisConfig())
    ssu_nodes = [n for n, d in g.nodes(data=True) if d["node_role"] == "ssu"]
    union_nodes = [n for n, d in g.nodes(data=True) if d["node_role"] == "union"]
    assert len(ssu_nodes) == 16 * 8
    assert len(union_nodes) == 16 * 2


def test_3d_torus_has_uniform_backend_bandwidth_per_direction():
    g = build_topology("3D-Torus", AnalysisConfig())
    backend = [
        data["bandwidth_gbps"]
        for _, _, data in g.edges(data=True)
        if data["link_kind"] == "backend_interconnect"
    ]
    assert set(backend) == {400.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topologies.py -v`
Expected: FAIL because legacy builders and node metadata are still in place.

- [ ] **Step 3: Implement the shared exchange-node template**

```python
def _add_exchange_node(g: nx.Graph, exchange_node_id: str, cfg: AnalysisConfig) -> dict[str, list[str]]:
    union_ids = []
    ssu_ids = []
    for union_index in range(2):
        union_id = f"{exchange_node_id}:union{union_index}"
        g.add_node(union_id, node_type="switch", node_role="union", exchange_node_id=exchange_node_id)
        union_ids.append(union_id)
    for ssu_index in range(8):
        ssu_id = f"{exchange_node_id}:ssu{ssu_index}"
        g.add_node(ssu_id, node_type="endpoint", node_role="ssu", exchange_node_id=exchange_node_id)
        ssu_ids.append(ssu_id)
        for union_id in union_ids:
            g.add_edge(ssu_id, union_id, bandwidth_gbps=200.0, link_kind="internal_ssu_uplink")
    return {"ssus": ssu_ids, "unions": union_ids}
```

- [ ] **Step 4: Implement the new topology catalog**

Add dedicated builders for:

- `2D-FullMesh`
- `2D-Torus`
- `3D-Torus`

Remove `ring`, `star`, `mesh`, old `torus`, and legacy `fat_tree` from the user-facing builder map.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_topologies.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add topo_sim/topologies.py tests/test_topologies.py
git commit -m "feat: build explicit ssu exchange-node topologies"
```

### Task 3: Add Clos Builder And Port-Budget Validation

**Files:**
- Modify: `topo_sim/topologies.py`
- Test: `tests/test_topologies.py`

- [ ] **Step 1: Extend the failing tests**

```python
def test_clos_uses_18_exchange_nodes_with_four_uplinks_each():
    g = build_topology("Clos", AnalysisConfig())
    backend = [
        (u, v, data)
        for u, v, data in g.edges(data=True)
        if data["link_kind"] == "backend_interconnect"
    ]
    uplinks_by_exchange = {}
    for u, v, data in backend:
        for node in (u, v):
            exchange_node = g.nodes[node].get("exchange_node_id")
            if exchange_node is not None:
                uplinks_by_exchange[exchange_node] = uplinks_by_exchange.get(exchange_node, 0) + 1
    assert all(count == 4 for count in uplinks_by_exchange.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topologies.py::test_clos_uses_18_exchange_nodes_with_four_uplinks_each -v`
Expected: FAIL because the legacy Clos/fat-tree implementation does not match the new Union-stage design.

- [ ] **Step 3: Implement the minimal code**

```python
def build_clos(cfg: AnalysisConfig) -> nx.Graph:
    g = nx.Graph()
    exchange_nodes = [_add_exchange_node(g, f"en{index}", cfg) for index in range(18)]
    spine_ids = [f"clos_spine_union{index}" for index in range(cfg.clos_uplinks_per_exchange_node)]
    for spine_id in spine_ids:
        g.add_node(spine_id, node_type="switch", node_role="clos_spine")
    for exchange in exchange_nodes:
        for uplink_index, spine_id in enumerate(spine_ids):
            union_id = exchange["unions"][uplink_index % 2]
            g.add_edge(union_id, spine_id, bandwidth_gbps=400.0, link_kind="backend_interconnect")
    return g
```

- [ ] **Step 4: Add validation**

Validate:

- `clos_uplinks_per_exchange_node <= 6`
- the upper-layer Union fanout does not exceed `18`
- all backend directions remain uniform within a topology instance

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_topologies.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add topo_sim/topologies.py tests/test_topologies.py
git commit -m "feat: add clos topology and port budget validation"
```

### Task 4: Implement Routing Modes

**Files:**
- Create: `topo_sim/routing.py`
- Test: `tests/test_routing.py`

- [ ] **Step 1: Write the failing tests**

```python
from topo_sim.config import AnalysisConfig
from topo_sim.routing import compute_paths
from topo_sim.topologies import build_topology


def test_dor_returns_dimension_order_path_for_2d_torus():
    g = build_topology("2D-Torus", AnalysisConfig())
    paths = compute_paths(g, "en0:ssu0", "en5:ssu0", routing_mode="DOR", cfg=AnalysisConfig())
    assert len(paths) == 1
    assert paths[0].hops >= 1


def test_port_balanced_splits_evenly_across_available_egress_ports():
    g = build_topology("2D-FullMesh", AnalysisConfig())
    paths = compute_paths(g, "en0:ssu0", "en15:ssu0", routing_mode="PORT_BALANCED", cfg=AnalysisConfig())
    weights = {round(path.weight, 5) for path in paths}
    assert len(paths) >= 2
    assert len(weights) == 1


def test_ecmp_returns_equal_cost_paths_for_clos():
    g = build_topology("Clos", AnalysisConfig())
    paths = compute_paths(g, "en0:ssu0", "en1:ssu0", routing_mode="ECMP", cfg=AnalysisConfig())
    assert len(paths) == AnalysisConfig().clos_uplinks_per_exchange_node
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_routing.py -v`
Expected: FAIL because `topo_sim.routing` does not exist.

- [ ] **Step 3: Implement the routing primitives**

```python
@dataclass(slots=True)
class RoutedPath:
    nodes: tuple[str, ...]
    weight: float


def compute_paths(g: nx.Graph, src_ssu: str, dst_ssu: str, routing_mode: str, cfg: AnalysisConfig) -> list[RoutedPath]:
    if routing_mode == "DOR":
        return _compute_dor_paths(g, src_ssu, dst_ssu, cfg)
    if routing_mode == "PORT_BALANCED":
        return _compute_port_balanced_paths(g, src_ssu, dst_ssu, cfg)
    if routing_mode == "ECMP":
        return _compute_ecmp_paths(g, src_ssu, dst_ssu)
    raise ValueError(f"Unsupported routing mode: {routing_mode}")
```

- [ ] **Step 4: Add detour and egress-port rules**

Implement:

- source-side egress grouping by Union-facing backend edge
- shortest-path preference when available for an egress port
- fallback to non-shortest path when no shortest path exists for that port
- detour cap from `cfg.port_balanced_max_detour_hops`

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_routing.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add topo_sim/routing.py tests/test_routing.py
git commit -m "feat: add ssu routing modes"
```

### Task 5: Implement SSU Workload Generation

**Files:**
- Create: `topo_sim/traffic.py`
- Test: `tests/test_traffic.py`

- [ ] **Step 1: Write the failing tests**

```python
from topo_sim.config import AnalysisConfig
from topo_sim.traffic import build_a2a_demands, build_sparse_random_demands
from topo_sim.topologies import build_topology


def test_a2a_builds_one_flow_per_ordered_ssu_pair():
    g = build_topology("2D-FullMesh", AnalysisConfig())
    demands = build_a2a_demands(g, AnalysisConfig())
    ssu_count = sum(1 for _, data in g.nodes(data=True) if data["node_role"] == "ssu")
    assert len(demands) == ssu_count * (ssu_count - 1)


def test_sparse_random_respects_active_ratio_and_target_count():
    cfg = AnalysisConfig(sparse_active_ratio=0.25, sparse_target_count=2, random_seed=7)
    g = build_topology("2D-Torus", cfg)
    demands = build_sparse_random_demands(g, cfg)
    active_sources = {d.src for d in demands}
    assert len(active_sources) > 0
    assert all(sum(1 for d in demands if d.src == src) == 2 for src in active_sources)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_traffic.py -v`
Expected: FAIL because `topo_sim.traffic` does not exist.

- [ ] **Step 3: Implement the minimal workload module**

```python
@dataclass(slots=True)
class FlowDemand:
    src: str
    dst: str
    bits: float


def build_sparse_random_demands(g: nx.Graph, cfg: AnalysisConfig) -> list[FlowDemand]:
    rng = np.random.default_rng(cfg.random_seed)
    ssus = [node for node, data in g.nodes(data=True) if data["node_role"] == "ssu"]
    active_count = max(1, int(round(len(ssus) * cfg.sparse_active_ratio)))
    active_sources = list(rng.choice(ssus, size=active_count, replace=False))
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_traffic.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add topo_sim/traffic.py tests/test_traffic.py
git commit -m "feat: add a2a and sparse ssu traffic generation"
```

### Task 6: Redesign Metric Computation Around Routed Loads

**Files:**
- Modify: `topo_sim/metrics.py`
- Modify: `topo_sim/simulation.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
from topo_sim.config import AnalysisConfig
from topo_sim.metrics import compute_structural_metrics, evaluate_workload
from topo_sim.topologies import build_topology
from topo_sim.traffic import build_a2a_demands


def test_structural_metrics_use_ssu_pairs():
    g = build_topology("2D-FullMesh", AnalysisConfig())
    metrics = compute_structural_metrics(g)
    assert "diameter" in metrics
    assert "average_hops" in metrics
    assert "bisection_bandwidth_gbps" in metrics


def test_a2a_metrics_return_throughput_and_completion_time():
    cfg = AnalysisConfig()
    g = build_topology("Clos", cfg)
    result = evaluate_workload(g, build_a2a_demands(g, cfg), routing_mode="ECMP", cfg=cfg)
    assert result["completion_time_s"] > 0
    assert result["per_ssu_throughput_gbps"] > 0
    assert result["max_link_utilization"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL because the legacy metric API and field set are still present.

- [ ] **Step 3: Implement the new metric split**

```python
def compute_structural_metrics(g: nx.Graph) -> dict[str, float]:
    return {
        "diameter": ...,
        "average_hops": ...,
        "bisection_bandwidth_gbps": ...,
    }


def evaluate_workload(g: nx.Graph, demands: list[FlowDemand], routing_mode: str, cfg: AnalysisConfig) -> dict[str, float]:
    routed = route_demands(g, demands, routing_mode, cfg)
    link_loads = project_link_loads(g, routed)
    completion_time_s = max(load_bits / capacity_bits_per_s for load_bits, capacity_bits_per_s in ...)
    return {
        "completion_time_s": completion_time_s,
        "per_ssu_throughput_gbps": ...,
        "max_link_utilization": ...,
        "link_utilization_cv": ...,
    }
```

- [ ] **Step 4: Remove old metric fields**

Delete or stop exporting:

- `network_cost`
- `estimated_latency_us`
- `a2av_efficiency`
- `ideal_alltoall_us`
- `topology_alltoall_us`
- shortest-path-only random-traffic summary fields from the old model

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add topo_sim/metrics.py topo_sim/simulation.py tests/test_metrics.py
git commit -m "feat: compute routing-aware ssu metrics"
```

### Task 7: Rewire The Analysis Pipeline And Machine-Readable Outputs

**Files:**
- Modify: `topo_sim/pipeline.py`
- Modify: `main.py`
- Test: `tests/test_pipeline_outputs.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from topo_sim.config import AnalysisConfig
from topo_sim.pipeline import run_full_analysis


def test_pipeline_writes_new_metric_columns(tmp_path: Path):
    cfg = AnalysisConfig(output_dir=tmp_path)
    paths = run_full_analysis(cfg, ["2D-FullMesh"])
    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "a2a_per_ssu_throughput_gbps" in csv_text
    assert "sparse_completion_time_p95" in csv_text
    assert "estimated_latency_us" not in csv_text


def test_pipeline_writes_routing_and_workload_config(tmp_path: Path):
    cfg = AnalysisConfig(output_dir=tmp_path)
    paths = run_full_analysis(cfg, ["Clos"])
    payload = paths["config"].read_text(encoding="utf-8")
    assert "routing_mode" in payload
    assert "sparse_active_ratio" in payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_outputs.py -v`
Expected: FAIL because the old pipeline still merges legacy metrics and random-traffic simulation fields.

- [ ] **Step 3: Implement the minimal pipeline changes**

```python
def run_full_analysis(cfg: AnalysisConfig, topologies: list[str] | None = None) -> dict[str, Path]:
    selected = topologies if topologies else cfg.topology_names
    for name in selected:
        g = build_topology(name, cfg)
        structural = compute_structural_metrics(g)
        a2a = evaluate_workload(g, build_a2a_demands(g, cfg), routing_mode=select_routing_mode(name, cfg), cfg=cfg)
        sparse = evaluate_workload(g, build_sparse_random_demands(g, cfg), routing_mode=select_routing_mode(name, cfg), cfg=cfg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_outputs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py topo_sim/pipeline.py tests/test_pipeline_outputs.py
git commit -m "feat: wire ssu workloads into analysis pipeline"
```

### Task 8: Redesign PDF, HTML, And README Outputs

**Files:**
- Modify: `topo_sim/report.py`
- Modify: `topo_sim/visualization.py`
- Modify: `templates/dashboard.html.j2`
- Modify: `README.md`
- Test: `tests/test_pipeline_outputs.py`

- [ ] **Step 1: Extend the failing tests**

```python
def test_dashboard_and_report_use_new_labels(tmp_path: Path):
    cfg = AnalysisConfig(output_dir=tmp_path)
    paths = run_full_analysis(cfg, ["2D-Torus"])
    html = paths["html"].read_text(encoding="utf-8")
    assert "A2A Per SSU Throughput" in html
    assert "Routing" in html
    assert "A2Av Efficiency" not in html
    assert paths["pdf"].exists()
    assert paths["pdf"].stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_outputs.py::test_dashboard_and_report_use_new_labels -v`
Expected: FAIL because the old report and dashboard still render legacy metrics and labels.

- [ ] **Step 3: Implement the output refresh**

Update:

- `topo_sim/report.py` to add sections for hardware assumptions, topology configuration, routing configuration, workload configuration, structural metrics, communication metrics, and observations
- `topo_sim/visualization.py` to pass the new metric groups into the template
- `templates/dashboard.html.j2` to display the new topology names and metric cards
- `README.md` to document the four supported topologies, routing modes, sparse workload knobs, and output artifacts

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_outputs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add topo_sim/report.py topo_sim/visualization.py templates/dashboard.html.j2 README.md tests/test_pipeline_outputs.py
git commit -m "feat: update reports and dashboard for ssu topology metrics"
```

### Task 9: Full Verification And Cleanup

**Files:**
- Modify: `topo_sim/__init__.py` if exports need refresh
- Modify: any touched file from prior tasks for final cleanup
- Test: `tests/test_config_and_cli.py`
- Test: `tests/test_topologies.py`
- Test: `tests/test_routing.py`
- Test: `tests/test_traffic.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_pipeline_outputs.py`

- [ ] **Step 1: Run the focused test suite**

Run: `pytest tests/test_config_and_cli.py tests/test_topologies.py tests/test_routing.py tests/test_traffic.py tests/test_metrics.py tests/test_pipeline_outputs.py -v`
Expected: PASS

- [ ] **Step 2: Run an end-to-end CLI check**

Run: `python main.py --topologies 2D-FullMesh,2D-Torus,3D-Torus,Clos --output-dir outputs_plan_check`
Expected: command completes successfully and writes `summary.csv`, `topology_dashboard.html`, `topology_report.pdf`, and `run_config.json` under `outputs_plan_check`

- [ ] **Step 3: Sanity-check generated outputs**

Inspect:

- `outputs_plan_check/summary.csv` for only the new metric columns
- `outputs_plan_check/run_config.json` for routing and sparse-workload settings
- `outputs_plan_check/topology_dashboard.html` for new topology names and labels
- `outputs_plan_check/topology_report.pdf` for topology, routing, workload, and metric sections

- [ ] **Step 4: Update README examples if CLI flags changed during implementation**

```bash
python main.py --topologies 2D-FullMesh,Clos --routing-mode PORT_BALANCED --sparse-active-ratio 0.25 --sparse-target-count 2 --output-dir outputs_example
```

- [ ] **Step 5: Final commit**

```bash
git add requirements.txt README.md main.py topo_sim templates tests
git commit -m "feat: ship ssu-centric topology and routing redesign"
```

If git is unavailable in the current workspace export, record the skipped commit in the task log and finish with the passing verification results.
