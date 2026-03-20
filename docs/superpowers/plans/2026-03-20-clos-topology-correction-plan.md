# Clos Topology Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Realign the Clos builder and tests so each exchange exposes two union planes with four plane-local uplinks, giving eight 400 Gbps uplinks per exchange and preserving spine fanout limits.

**Architecture:** Extend `topo_sim/topologies.py` to create two spine pools (one per union plane) and wire each union to its plane's switches, then update topology/routing tests so they encode the higher path and port counts before the builder changes are applied.

**Tech Stack:** Python 3, NetworkX, pytest, dataclasses in `topo_sim/config.py`.

---

### Task 1: Capture the new Clos expectations in topology tests

**Files:**
- Modify: `tests/test_topologies.py`

- [ ] **Step 1:** Add assertions that each exchange has `2 * cfg.clos_uplinks_per_exchange_node` backend edges and that each union node exposes exactly `cfg.clos_uplinks_per_exchange_node` backend uplinks.
```python
assert all(count == cfg.clos_uplinks_per_exchange_node * 2 for count in uplinks_by_exchange.values())
```
```python
assert set(union_backend_degree.values()) == {cfg.clos_uplinks_per_exchange_node}
```
- [ ] **Step 2:** Run `pytest tests/test_topologies.py -k clos` (expect failures because the builder currently only provides four uplinks per exchange).
- [ ] **Step 3:** Keep the test failures in mind as the driver for Task 3 (fixing the builder) and record the observed assertion message for reference.

### Task 2: Make the routing test reflect the corrected ECMP diversity

**Files:**
- Modify: `tests/test_routing.py`

- [ ] **Step 1:** Update `test_ecmp_returns_equal_cost_paths_for_clos` so it expects `cfg.clos_uplinks_per_exchange_node * 2` paths and documents the intended weight normalization.
```python
assert len(paths) == cfg.clos_uplinks_per_exchange_node * 2
```
- [ ] **Step 2:** Run `pytest tests/test_routing.py::test_ecmp_returns_equal_cost_paths_for_clos` and note the mismatch (the old builder returns only `cfg.clos_uplinks_per_exchange_node` paths).
- [ ] **Step 3:** Keep the failure output so Task 3 can verify the corrected builder now exposes the additional paths.

### Task 3: Update the Clos builder to emit dual-plane spines

**Files:**
- Modify: `topo_sim/topologies.py`

- [ ] **Step 1:** Replace the single `spine_ids` list with two plane-specific lists of size `cfg.clos_uplinks_per_exchange_node`, then add both planes' nodes with `node_role="clos_spine"`.
```python
plane_spine_ids = {plane: [f"clos_spine_plane{plane}_uplink{idx}" for idx in range(cfg.clos_uplinks_per_exchange_node)] for plane in (0, 1)}
```
- [ ] **Step 2:** Wire each union to every spine in its plane so that each union provides the configured uplink count and each exchange has double that number.
```python
for plane_index, union_id in enumerate(exchange["unions"]):
    for spine_id in plane_spine_ids[plane_index]:
        g.add_edge(...)
```
- [ ] **Step 3:** Run the previously failing subset (`pytest tests/test_topologies.py -k clos` and `pytest tests/test_routing.py::test_ecmp_returns_equal_cost_paths_for_clos`) to verify the builder now satisfies the new assertions.

### Task 4: Smoke-check the targeted test suite

**Files:**
- None (running commands)

- [ ] **Step 1:** Run `pytest tests/test_topologies.py::test_clos_uses_18_exchange_nodes ...` (or use `-k clos` as before) to confirm the topology tests pass.
- [ ] **Step 2:** Run `pytest tests/test_routing.py::test_ecmp_returns_equal_cost_paths_for_clos` and expect PASS.

### Task 5: Wrap up and capture work history

**Files:**
- Modify: `docs/superpowers/plans/2026-03-20-clos-topology-correction-plan.md`
- Modify: `docs/superpowers/specs/2026-03-20-clos-topology-design.md` (already created)
- Modify: `topo_sim/topologies.py`
- Modify: `tests/test_topologies.py`
- Modify: `tests/test_routing.py`

- [ ] **Step 1:** Stage the modified files with `git add topo_sim/topologies.py tests/test_topologies.py tests/test_routing.py docs/superpowers/specs/2026-03-20-clos-topology-design.md docs/superpowers/plans/2026-03-20-clos-topology-correction-plan.md`.
- [ ] **Step 2:** Run `git status -sb` to confirm only the relevant files are staged.
- [ ] **Step 3:** Commit with a descriptive message such as `feat: fix Clos per-plane uplinks`.

---
Plan complete and saved to `docs/superpowers/plans/2026-03-20-clos-topology-correction-plan.md`. Execution approach: inline (executing-plans) within this session. Spec reference: `docs/superpowers/specs/2026-03-20-clos-topology-design.md`.
