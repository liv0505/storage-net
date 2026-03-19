# Dashboard Layered Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the dashboard around explicit layered topology layouts, a black theme, topology-first card ordering, and one-hop node highlight interaction.

**Architecture:** Keep the existing pipeline output contract intact, but replace the visualization coordinate generation and dashboard template structure. Compute deterministic positions per topology in `visualization.py`, emit richer Plotly metadata for node/edge adjacency, and update the Jinja template to render the new dark layout and click-highlighting script.

**Tech Stack:** Python 3, networkx, plotly, jinja2, pytest

---

## File Structure

### Files To Modify

- `topo_sim/visualization.py`
- `templates/dashboard.html.j2`
- `tests/test_pipeline_outputs.py`

### Responsibility Map

- `topo_sim/visualization.py`: explicit layered coordinates, node/edge metadata, interactive Plotly figure generation.
- `templates/dashboard.html.j2`: dark dashboard shell, topology-first card order, reduced note styling, click highlight script.
- `tests/test_pipeline_outputs.py`: pin dark theme labels, topology-first ordering cues, and interactive script markers.

### Task 1: Lock The New Dashboard Contract With Tests

**Files:**
- Modify: `tests/test_pipeline_outputs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_dashboard_uses_black_theme_and_interaction_hooks(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir)
    paths = run_full_analysis(cfg, ["2D-Torus", "Clos"])
    html = paths["html"].read_text(encoding="utf-8")
    assert "Topology Figure" in html
    assert "data-highlight-mode=\"neighbors\"" in html
    assert "plot-card plot-card-dark" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_outputs.py::test_dashboard_uses_black_theme_and_interaction_hooks -v`
Expected: FAIL because the current template lacks the new structure and interaction markers.

- [ ] **Step 3: Extend layout-specific assertions**

```python
def test_dashboard_mentions_layered_layout_rules(output_dir: Path):
    cfg = AnalysisConfig(output_dir=output_dir)
    paths = run_full_analysis(cfg, ["3D-Torus", "Clos"])
    html = paths["html"].read_text(encoding="utf-8")
    assert "4 z-layers" in html
    assert "Clos spine layer" in html
```

- [ ] **Step 4: Run the targeted tests again**

Run: `python -m pytest tests/test_pipeline_outputs.py -v`
Expected: FAIL on the new dashboard assertions.

### Task 2: Replace Force-Directed Placement With Explicit Layered Coordinates

**Files:**
- Modify: `topo_sim/visualization.py`
- Test: `tests/test_pipeline_outputs.py`

- [ ] **Step 1: Implement topology-aware coordinate builders**

Create helpers for:
- exchange-internal `SSU` and `Union` rows
- `2D-FullMesh` / `2D-Torus` `4x4` exchange-node grids
- `3D-Torus` four `4x4` `z` blocks
- `Clos` exchange-node grid plus upper spine layer

- [ ] **Step 2: Replace the current `_positions(...)` logic**

Ensure the dashboard prefers explicit layered positions and only keeps the old fallback for unknown future topologies.

- [ ] **Step 3: Run the focused tests**

Run: `python -m pytest tests/test_pipeline_outputs.py::test_dashboard_mentions_layered_layout_rules -v`
Expected: PASS

### Task 3: Add One-Hop Node Highlight Interaction And Dark Card Layout

**Files:**
- Modify: `topo_sim/visualization.py`
- Modify: `templates/dashboard.html.j2`
- Test: `tests/test_pipeline_outputs.py`

- [ ] **Step 1: Emit adjacency metadata into the figure / page payload**

Include enough information for front-end click handlers to identify neighboring nodes and incident edges.

- [ ] **Step 2: Redesign the template**

Update the HTML to:
- use a black theme
- render the topology figure first in each card
- show metrics next
- shrink explanatory text styling
- add a small script that listens for Plotly clicks and applies one-hop highlight / dim behavior

- [ ] **Step 3: Run dashboard output tests**

Run: `python -m pytest tests/test_pipeline_outputs.py -v`
Expected: PASS

### Task 4: Full Verification

**Files:**
- Modify: any touched files above if needed
- Test: `tests/test_pipeline_outputs.py`
- Test: `tests/test_config_and_cli.py`
- Test: `tests/test_topologies.py`
- Test: `tests/test_routing.py`
- Test: `tests/test_traffic.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Run the focused suite**

Run: `python -m pytest tests/test_config_and_cli.py tests/test_topologies.py tests/test_routing.py tests/test_traffic.py tests/test_metrics.py tests/test_pipeline_outputs.py -q`
Expected: PASS

- [ ] **Step 2: Run an end-to-end dashboard generation check**

Run: `python main.py --topologies 2D-FullMesh,2D-Torus,3D-Torus,Clos --output-dir outputs_dashboard_layout_check`
Expected: command completes and writes updated dashboard/report artifacts.
