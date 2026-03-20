# Multi-Routing Comparison and 3D-Torus Shortest-Path Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add compact multi-routing performance comparison tables for direct-connect topologies, keep `Clos` on `ECMP` only with corrected per-Union uplinks, add selected-link bandwidth labels to the dashboard interaction, and make exact `3D-Torus + SHORTEST_PATH` analysis fast enough for the default full run to complete.

**Architecture:** Keep the current explicit SSU/Union graph model, but split the work into three coordinated layers: a topology correction layer that fixes `Clos` to use `4 x 400 Gbps` uplinks per Union plane, a routing/metrics optimization layer that computes exact shortest-path traffic splitting without explicit path explosion, and a presentation layer that augments each topology result with compact routing comparison tables plus selected-link bandwidth labels. Direct-connect topologies will internally evaluate `DOR`, `SHORTEST_PATH`, and `FULL_PATH` for both `A2A` and sparse `1-to-N`, while `Clos` remains `ECMP` only.

**Tech Stack:** Python 3, `networkx`, `numpy`, `jinja2`, `plotly`, `reportlab`, `pytest`

---

## File Structure

### Files To Modify

- `topo_sim/routing.py`
- `topo_sim/metrics.py`
- `topo_sim/pipeline.py`
- `topo_sim/visualization.py`
- `templates/dashboard.html.j2`
- `topo_sim/report.py`
- `main.py`
- `README.md`
- `tests/test_metrics.py`
- `tests/test_pipeline_outputs.py`
- `tests/test_config_and_cli.py`
- `tests/test_topologies.py`
- `tests/test_routing.py`

### Optional Helper Files To Create

- `tests/test_shortest_path_acceleration.py`
  - Use if the shortest-path DAG tests become too large for `tests/test_metrics.py`

### Responsibility Map

- `topo_sim/topologies.py`: corrected Clos construction with `4 x 400 Gbps` uplinks per Union plane and `8 x 400 Gbps` total uplinks per exchange node
- `topo_sim/routing.py`: shortest-path DAG helpers, direct-topology routing helpers, exact shortest-path flow propagation primitives
- `topo_sim/metrics.py`: workload evaluation, fast exact aggregation for heavy shortest-path cases, routing-aware metric outputs
- `topo_sim/pipeline.py`: orchestrate per-topology multi-routing analyses and build result payloads for dashboard / report
- `topo_sim/visualization.py`: pass new routing-comparison payloads into the template context
- `templates/dashboard.html.j2`: compact multi-routing comparison tables and smaller default throughput highlight cards
- `topo_sim/report.py`: compact comparison tables and default-throughput-only main summary in the PDF
- `main.py`: preserve CLI behavior while default full analysis now emits complete direct-topology multi-routing comparisons
- `README.md`: document new output structure and default per-flow traffic volume interpretation
- `tests/*`: protect correctness, output structure, runtime-sensitive behavior, and the new selected-link bandwidth labels

---

### Task 1: Add Failing Tests For Multi-Routing Comparison Payloads And Output Shape

**Files:**
- Modify: `tests/test_pipeline_outputs.py`
- Modify: `tests/test_config_and_cli.py`
- Test: `tests/test_pipeline_outputs.py`
- Test: `tests/test_config_and_cli.py`

- [ ] **Step 1: Write the failing tests for direct-topology routing comparison payloads**

```python
def test_pipeline_emits_direct_topology_routing_comparisons(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir)
    paths = run_full_analysis(cfg, ["2D-Torus", "Clos"])

    html = paths["html"].read_text(encoding="utf-8")
    assert "A2A Routing Comparison" in html
    assert "Sparse 1-to-N Routing Comparison" in html
    assert "DOR" in html
    assert "SHORTEST_PATH" in html
    assert "FULL_PATH" in html
    assert "ECMP" in html
```

- [ ] **Step 2: Write the failing tests for compact metric columns**

```python
def test_dashboard_comparison_tables_only_show_expected_metric_columns(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir)
    paths = run_full_analysis(cfg, ["2D-FullMesh"])

    html = paths["html"].read_text(encoding="utf-8")
    assert "Per SSU Throughput" in html
    assert "Completion Time" in html
    assert "P95 Completion" in html
    assert "Max Link Utilization" in html
    assert "Link Utilization CV" in html
    assert "Completion Time P50" not in html
```

- [ ] **Step 3: Write the failing tests for default-route highlight behavior**

```python
def test_main_result_block_uses_dor_for_direct_topologies_and_ecmp_for_clos(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir)
    paths = run_full_analysis(cfg, ["2D-Torus", "Clos"])

    html = paths["html"].read_text(encoding="utf-8")
    assert "Default Route Throughput" in html
    assert "DOR Throughput" in html
    assert "ECMP Throughput" in html
```

- [ ] **Step 4: Write the failing tests for `Clos` comparison exclusion**

```python
def test_clos_does_not_render_direct_routing_comparison_tables(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir)
    paths = run_full_analysis(cfg, ["Clos"])

    html = paths["html"].read_text(encoding="utf-8")
    assert "A2A Routing Comparison" not in html
    assert "Sparse 1-to-N Routing Comparison" not in html
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_outputs.py tests/test_config_and_cli.py -k "routing_comparison or default_route or compact_metric_columns or clos_does_not_render" -v`
Expected: FAIL because the pipeline still assumes a single routing mode per topology and the new compact comparison tables do not exist.

- [ ] **Step 6: Commit the failing-test checkpoint**

```bash
git add tests/test_pipeline_outputs.py tests/test_config_and_cli.py
git commit -m "test: add multi-routing output expectations"
```

---

### Task 2: Add Failing Tests For Exact 3D-Torus Shortest-Path Acceleration

**Files:**
- Modify: `tests/test_metrics.py`
- Create: `tests/test_shortest_path_acceleration.py` (optional if needed)
- Test: `tests/test_metrics.py`
- Test: `tests/test_shortest_path_acceleration.py`

- [ ] **Step 1: Write the failing test that compares explicit shortest-path splitting with fast aggregation on a tractable case**

```python
def test_exact_shortest_path_flow_aggregation_matches_explicit_path_splitting_on_2d_torus():
    cfg = AnalysisConfig()
    g = build_topology("2D-Torus", cfg)
    demands = [FlowDemand(src="en0:ssu0", dst="en5:ssu0", bits=cfg.message_size_mb * 8_000_000.0)]

    explicit = evaluate_workload(g, demands, routing_mode="SHORTEST_PATH", cfg=cfg)
    accelerated = evaluate_workload(g, demands, routing_mode="SHORTEST_PATH", cfg=cfg)

    assert accelerated == explicit
```

- [ ] **Step 2: Write the failing test that exercises the 3D-Torus fast path without enumerating all shortest paths**

```python
def test_3d_torus_shortest_path_uses_exact_accelerated_flow_projection(monkeypatch):
    cfg = AnalysisConfig()
    g = build_topology("3D-Torus", cfg)

    def fail_if_all_shortest_paths(*args, **kwargs):
        raise AssertionError("explicit shortest-path enumeration should not run here")

    monkeypatch.setattr("networkx.all_shortest_paths", fail_if_all_shortest_paths)

    result = evaluate_workload(g, build_a2a_demands(g, cfg), routing_mode="SHORTEST_PATH", cfg=cfg)
    assert result["per_ssu_throughput_gbps"] > 0
```

- [ ] **Step 3: Write the failing test for fast-path metric completeness**

```python
def test_accelerated_shortest_path_projection_still_emits_all_required_metrics():
    cfg = AnalysisConfig()
    g = build_topology("3D-Torus", cfg)
    result = evaluate_workload(g, build_sparse_random_demands(g, cfg), routing_mode="SHORTEST_PATH", cfg=cfg)

    assert set(result) >= {
        "completion_time_s",
        "completion_time_p50_s",
        "completion_time_p95_s",
        "per_ssu_throughput_gbps",
        "max_link_utilization",
        "link_utilization_cv",
    }
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py tests/test_shortest_path_acceleration.py -k "accelerated or exact_shortest_path_flow" -v`
Expected: FAIL because there is no dedicated exact shortest-path aggregation path yet.

- [ ] **Step 5: Commit the failing-test checkpoint**

```bash
git add tests/test_metrics.py tests/test_shortest_path_acceleration.py
git commit -m "test: add exact shortest-path acceleration coverage"
```

---

### Task 3: Implement Exact Shortest-Path DAG Helpers In `routing.py`

**Files:**
- Modify: `topo_sim/routing.py`
- Test: `tests/test_routing.py`
- Test: `tests/test_shortest_path_acceleration.py`

- [ ] **Step 1: Add failing routing-helper tests for shortest-path DAG primitives**

```python
def test_shortest_path_dag_contains_only_edges_on_shortest_paths():
    ...

def test_shortest_path_dag_path_counts_match_known_small_torus_case():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_routing.py tests/test_shortest_path_acceleration.py -k "dag or path_counts" -v`
Expected: FAIL because the helper APIs do not exist.

- [ ] **Step 3: Implement minimal shortest-path DAG helpers**

```python
def build_shortest_path_dag(plane_graph: nx.Graph, src: str, dst: str) -> nx.DiGraph:
    src_dist = nx.single_source_shortest_path_length(plane_graph, src)
    dst_dist = nx.single_source_shortest_path_length(plane_graph, dst)
    shortest = src_dist[dst]
    dag = nx.DiGraph()
    for node, dist in src_dist.items():
        if dist + dst_dist.get(node, shortest + 1) != shortest:
            continue
        dag.add_node(node)
    for u, v in plane_graph.edges():
        if u in dag and v in dag:
            if src_dist[u] + 1 == src_dist[v]:
                dag.add_edge(u, v)
            if src_dist[v] + 1 == src_dist[u]:
                dag.add_edge(v, u)
    return dag
```

- [ ] **Step 4: Add exact shortest-path count propagation helper**

```python
def count_shortest_paths_in_dag(dag: nx.DiGraph, src: str) -> dict[str, int]:
    counts = {src: 1}
    for node in nx.topological_sort(dag):
        for succ in dag.successors(node):
            counts[succ] = counts.get(succ, 0) + counts.get(node, 0)
    return counts
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_routing.py tests/test_shortest_path_acceleration.py -k "dag or path_counts" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add topo_sim/routing.py tests/test_routing.py tests/test_shortest_path_acceleration.py
git commit -m "feat: add exact shortest-path dag helpers"
```

---

### Task 4: Implement Exact Accelerated Workload Projection For Heavy `SHORTEST_PATH`

**Files:**
- Modify: `topo_sim/metrics.py`
- Modify: `topo_sim/routing.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_shortest_path_acceleration.py`

- [ ] **Step 1: Add the failing tests for metric equivalence and non-enumerating 3D execution**

Reuse the tests added in Task 2 and extend them if necessary.

- [ ] **Step 2: Run tests to verify they still fail**

Run: `python -m pytest tests/test_metrics.py tests/test_shortest_path_acceleration.py -k "accelerated or exact_shortest_path_flow" -v`
Expected: FAIL

- [ ] **Step 3: Implement a specialized exact aggregation path in `metrics.py`**

```python
def _evaluate_shortest_path_workload_exact_fast(
    g: nx.Graph,
    demands: Iterable[FlowDemand],
    cfg: AnalysisConfig,
) -> dict[str, float]:
    # 1. Group demand bits by (src_ssu, dst_ssu)
    # 2. Split demand 1:1 across the two source-side Union planes
    # 3. For each Union plane, build shortest-path DAG once per exchange pair
    # 4. Propagate exact flow mass edge-by-edge instead of enumerating path objects
    # 5. Reuse existing completion-time and utilization logic
```

- [ ] **Step 4: Add automatic selection logic for the fast path**

```python
if normalize_routing_mode(routing_mode) == "SHORTEST_PATH" and _should_use_exact_shortest_path_fast_path(g):
    return _evaluate_shortest_path_workload_exact_fast(g, demands, cfg)
```

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py tests/test_shortest_path_acceleration.py -k "accelerated or exact_shortest_path_flow" -v`
Expected: PASS

- [ ] **Step 6: Run the broader routing/metrics regression suite**

Run: `python -m pytest tests/test_routing.py tests/test_metrics.py tests/test_traffic.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add topo_sim/metrics.py topo_sim/routing.py tests/test_metrics.py tests/test_shortest_path_acceleration.py tests/test_routing.py
git commit -m "feat: accelerate exact shortest-path workload projection"
```

---

### Task 5: Extend The Pipeline To Compute Multi-Routing Comparisons

**Files:**
- Modify: `topo_sim/pipeline.py`
- Test: `tests/test_pipeline_outputs.py`

- [ ] **Step 1: Add failing tests for direct-topology routing-comparison payloads**

```python
def test_pipeline_builds_routing_comparison_payload_for_direct_topologies(output_dir: Path):
    ...

def test_pipeline_keeps_clos_single_route_payload(output_dir: Path):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_outputs.py -k "comparison_payload or single_route_payload" -v`
Expected: FAIL

- [ ] **Step 3: Implement compact comparison payload builders**

```python
def _comparison_modes_for_topology(name: str) -> list[str]:
    if name == "Clos":
        return ["ECMP"]
    return ["DOR", "SHORTEST_PATH", "FULL_PATH"]


def _routing_comparison_for_topology(name: str, g: nx.Graph, cfg: AnalysisConfig) -> dict[str, Any] | None:
    ...
```

- [ ] **Step 4: Add main-result throughput highlight payloads**

```python
def _default_highlight_mode(name: str) -> str:
    return "ECMP" if name == "Clos" else "DOR"
```

- [ ] **Step 5: Thread the new payloads into `_build_render_result`**

```python
return {
    ...,
    "default_routing_highlight": default_highlight,
    "routing_comparison": comparison_payload,
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline_outputs.py -k "comparison_payload or single_route_payload or default_route" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add topo_sim/pipeline.py tests/test_pipeline_outputs.py
git commit -m "feat: add per-topology multi-routing analysis payloads"
```

---

### Task 6: Render Compact Routing Comparison Tables In The Dashboard

**Files:**
- Modify: `topo_sim/visualization.py`
- Modify: `templates/dashboard.html.j2`
- Test: `tests/test_pipeline_outputs.py`

- [ ] **Step 1: Add failing HTML-output tests for compact comparison tables**

```python
def test_dashboard_renders_compact_a2a_and_sparse_routing_tables(output_dir: Path):
    ...

def test_dashboard_shows_only_default_throughput_in_main_result_highlight(output_dir: Path):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_outputs.py -k "compact_a2a_and_sparse_routing_tables or default_throughput_in_main_result" -v`
Expected: FAIL

- [ ] **Step 3: Pass the new payloads through `visualization.py`**

```python
blocks.append(
    {
        ...,
        "default_routing_highlight": item["default_routing_highlight"],
        "routing_comparison": item.get("routing_comparison"),
    }
)
```

- [ ] **Step 4: Add compact dashboard tables in `dashboard.html.j2`**

```html
{% if r.routing_comparison %}
<div class="compact-table-panel">
  <h4>A2A Routing Comparison</h4>
  ...
</div>
<div class="compact-table-panel">
  <h4>Sparse 1-to-N Routing Comparison</h4>
  ...
</div>
{% endif %}
```

- [ ] **Step 5: Keep the default throughput highlight small and route-specific**

```html
<div class="metric"><span class="k">DOR Throughput</span><span class="v">...</span></div>
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline_outputs.py -k "compact_a2a_and_sparse_routing_tables or default_throughput_in_main_result or dashboard_includes_routing_diversity_snapshot" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add topo_sim/visualization.py templates/dashboard.html.j2 tests/test_pipeline_outputs.py
git commit -m "feat: render compact routing comparison tables in dashboard"
```

---

### Task 7: Render Matching Compact Comparison Tables In The PDF

**Files:**
- Modify: `topo_sim/report.py`
- Test: `tests/test_pipeline_outputs.py`

- [ ] **Step 1: Add failing PDF-output tests for comparison tables and default throughput summaries**

```python
def test_pdf_includes_compact_routing_comparison_sections(output_dir: Path):
    ...

def test_pdf_keeps_clos_out_of_direct_routing_comparison(output_dir: Path):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_outputs.py -k "pdf_includes_compact_routing_comparison_sections or pdf_keeps_clos_out" -v`
Expected: FAIL

- [ ] **Step 3: Implement compact PDF comparison sections**

```python
if item.get("routing_comparison"):
    story.append(Paragraph("A2A Routing Comparison", styles["SectionHeading"]))
    story.append(_styled_table(...))
    story.append(Paragraph("Sparse 1-to-N Routing Comparison", styles["SectionHeading"]))
    story.append(_styled_table(...))
```

- [ ] **Step 4: Keep the main summary throughput-only**

```python
story.append(Paragraph("Default Route Throughput", styles["SectionHeading"]))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline_outputs.py -k "pdf_includes_compact_routing_comparison_sections or pdf_keeps_clos_out or dashboard_and_report_use_new_labels" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add topo_sim/report.py tests/test_pipeline_outputs.py
git commit -m "feat: add compact routing comparison tables to pdf report"
```

---

### Task 8: Update CLI Semantics And Documentation

**Files:**
- Modify: `main.py`
- Modify: `README.md`
- Modify: `tests/test_config_and_cli.py`
- Test: `tests/test_config_and_cli.py`

- [ ] **Step 1: Add failing CLI tests for default full multi-routing behavior**

```python
def test_main_default_full_analysis_includes_direct_topology_multi_routing(monkeypatch):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config_and_cli.py -k "default_full_analysis_includes_direct_topology_multi_routing" -v`
Expected: FAIL

- [ ] **Step 3: Adjust CLI/config plumbing as needed without breaking existing options**

```python
# Keep --routing-mode for focused runs / compatibility, but default full analysis now emits the direct-topology comparison payloads.
```

- [ ] **Step 4: Update `README.md`**

Include:

- direct-topology comparison tables now compare `DOR`, `SHORTEST_PATH`, `FULL_PATH`
- `Clos` stays `ECMP` only
- each `SSU -> SSU` demand carries `4 MB` by default
- `A2A` and sparse traffic interpretation for per-SSU data volume

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_config_and_cli.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add main.py README.md tests/test_config_and_cli.py
git commit -m "docs: explain multi-routing comparisons and traffic defaults"
```

---

### Task 9: Full Verification And End-To-End Generation

**Files:**
- Modify: none expected
- Test: entire relevant suite

- [ ] **Step 1: Run the focused regression suite**

Run: `python -m pytest tests/test_routing.py tests/test_traffic.py tests/test_metrics.py tests/test_pipeline_outputs.py tests/test_config_and_cli.py -v`
Expected: PASS

- [ ] **Step 2: Run the default full analysis end-to-end**

Run: `python main.py --output-dir outputs_full_validation`
Expected: process completes and writes:
- `outputs_full_validation/summary.csv`
- `outputs_full_validation/topology_dashboard.html`
- `outputs_full_validation/topology_report.pdf`
- `outputs_full_validation/run_config.json`

- [ ] **Step 3: Spot-check the generated outputs**

Verify:

- direct topologies show compact `A2A` and `Sparse 1-to-N` routing comparison tables
- `Clos` shows only `ECMP`
- direct-topology main result blocks show only default-route throughput (`DOR`)
- `3D-Torus` section completes and renders successfully

- [ ] **Step 4: Commit the final validated state**

```bash
git add .
git commit -m "feat: compare direct-topology routing modes and accelerate 3d torus shortest path"
```
