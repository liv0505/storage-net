# Multi-Routing Comparison and 3D-Torus Shortest-Path Acceleration Design

**Date:** 2026-03-20

## Summary

This design extends the SSU-centric topology analysis project in two coordinated directions:

1. For direct-connect topologies (`2D-FullMesh`, `2D-Torus`, `3D-Torus`), the dashboard and PDF report will compare routing performance across `DOR`, `SHORTEST_PATH`, and `FULL_PATH` for both `A2A` and sparse `1-to-N` workloads.
2. The `3D-Torus + SHORTEST_PATH` evaluation path will be optimized so the default full analysis can finish while preserving exact shortest-path traffic splitting semantics.

The design keeps `Clos` separate: it continues to evaluate only `ECMP` for `A2A` and sparse `1-to-N` traffic.

## Approved Decisions

### Routing Comparison Scope

For the direct-connect topologies:

- `2D-FullMesh`
- `2D-Torus`
- `3D-Torus`

The project will compute three routing modes:

- `DOR`
- `SHORTEST_PATH`
- `FULL_PATH`

For each of those routing modes, the project will evaluate:

- `A2A`
- `Sparse 1-to-N`

For `Clos`:

- only `ECMP` is evaluated
- only `A2A` and `Sparse 1-to-N` performance are reported
- `Clos` does not participate in the multi-routing comparison tables
- each exchange node uses two Union planes
- each Union plane uplinks through `4 x 400 Gbps` to its own set of L2 Union switches
- each exchange node therefore has `8 x 400 Gbps` total Clos uplink bandwidth

### Main Result Block

Each topology keeps a single primary result block in the dashboard and report.

That main result block contains:

- topology figure
- topology configuration summary
- workload configuration summary
- structural metrics
- routing semantics summary
- routing diversity snapshot
- a compact default routing throughput highlight

The default routing performance shown in the main result block is intentionally minimal:

- direct-connect topologies show only `DOR` throughput
- `Clos` shows only `ECMP` throughput
- the only performance value shown there is `per_ssu_throughput_gbps`

The main result block does not show the other routing-performance metrics.

### Multi-Routing Comparison Tables

For each direct-connect topology, add two compact routing comparison tables:

1. `A2A Routing Comparison`
2. `Sparse 1-to-N Routing Comparison`

Each table contains one row per routing mode:

- `DOR`
- `SHORTEST_PATH`
- `FULL_PATH`

Each row contains the following metrics only:

- `per_ssu_throughput_gbps`
- `completion_time_s`
- `completion_time_p95_s`
- `max_link_utilization`
- `link_utilization_cv`

The tables should be visually compact and should not dominate the topology figure or the page layout.

### Current Traffic Volume Defaults

The current workload model keeps a fixed message size per `SSU -> SSU` demand:

- `message_size_mb = 4.0` by default
- each flow therefore carries `4 MB`
- each flow therefore carries `32,000,000 bits`

Under current defaults:

- in `A2A`, each SSU sends `4 MB` to every other SSU
- in sparse `1-to-N`, each active SSU sends `4 MB` to each of its selected targets
- default sparse settings are:
  - `sparse_active_ratio = 0.25`
  - `sparse_target_count = 2`

### Link Balance Metric

`link_utilization_cv` remains part of the comparison tables.

Definition:

- it is the coefficient of variation of backend-link utilization
- computed as backend-link utilization population standard deviation divided by backend-link utilization mean
- `0` means perfectly uniform backend load
- larger values indicate increasingly uneven link utilization and stronger hot spots

### Clos Topology Clarification

The Clos topology must be modeled with per-Union uplink fanout, not per-exchange total fanout.

For each exchange node:

- `2` Union chips are present
- each Union connects downward to the local `8` SSUs through `8 x 200 Gbps`
- each Union connects upward through `4 x 400 Gbps`
- those `4` uplinks terminate on `4` L2-layer Union switches belonging to the same Union plane

Therefore:

- each exchange node has `8 x 400 Gbps` aggregate uplink bandwidth into the Clos network
- the two local Union planes remain structurally separate above the exchange node
- each L2 Union switch still fans out to at most `18` exchange nodes within its own plane

## Goals

- Keep the dashboard and PDF focused on one result block per topology.
- Add compact multi-routing comparison tables for the three direct-connect topologies.
- Preserve exact routing semantics for `DOR`, `SHORTEST_PATH`, `FULL_PATH`, and `ECMP`.
- Make `3D-Torus + SHORTEST_PATH` practical enough that the default full analysis can complete.
- Keep the HTML and PDF styles aligned.

## Non-Goals

- Changing the SSU/Union hardware model.
- Changing the workload definitions for `A2A` or sparse `1-to-N`.
- Adding new routing modes beyond the already approved set.
- Making `Clos` participate in direct-connect routing comparison tables.
- Switching to approximate shortest-path traffic splitting.

## Current Problem

The current analysis pipeline is still centered on a single selected routing mode per run. This causes two issues:

- the dashboard and report cannot yet compare routing modes side-by-side for the direct topologies
- `3D-Torus + SHORTEST_PATH` can become too slow because explicit enumeration of all shortest paths scales poorly

## Design

### 1. Analysis Model Upgrade

The analysis pipeline should distinguish between:

- topology-level structural analysis
- per-workload performance analysis
- per-routing-mode comparison analysis

For each topology result object:

- retain one primary routing summary for the main result block
- add a routing-comparison payload for `A2A`
- add a routing-comparison payload for `Sparse 1-to-N`

For direct-connect topologies, the routing-comparison payload includes all three direct routing modes.
For `Clos`, only a single `ECMP` payload is needed and no comparison table is produced.

### 2. Main Result Block Content

#### Direct-Connect Topologies

Each main result block should contain:

- layered topology figure
- topology configuration
- workload configuration
- structural metrics:
  - `diameter`
  - `average_hops`
  - `bisection_bandwidth_gbps`
- routing semantics summary
- routing diversity snapshot
- throughput highlight cards using `DOR` only:
  - `A2A per_ssu_throughput_gbps`
  - `Sparse 1-to-N per_ssu_throughput_gbps`

#### Clos

The `Clos` main result block should contain the same structural and configuration sections, but the throughput highlight cards use `ECMP`.

### 3. Compact Comparison Tables

#### Dashboard

For each direct-connect topology card, add two compact tables under the main summary area:

- `A2A Routing Comparison`
- `Sparse 1-to-N Routing Comparison`

Presentation requirements:

- small, dense table styling
- minimal vertical padding
- no repeated explanatory prose inside the table itself
- preserve the topology figure as the dominant visual element
- clicking a node should continue to highlight one-hop neighbors and incident edges
- when a node is selected, the highlighted incident links should also display their link bandwidth labels

#### PDF

For each direct-connect topology section, add matching compact comparison tables:

- `A2A Routing Comparison`
- `Sparse 1-to-N Routing Comparison`

The tables should use the same dark visual language as the HTML dashboard and should remain narrow enough to avoid overwhelming the surrounding topology summary.

### 4. Exact `SHORTEST_PATH` Acceleration for 3D-Torus

The optimization target is exactness with better runtime, not approximation.

#### Existing Slow Path

The current `SHORTEST_PATH` implementation expands all shortest Union-to-Union paths explicitly. For `3D-Torus`, this can cause path-count explosion and make the default full run impractical.

#### Proposed Fast Path

Introduce a specialized exact evaluation path for shortest-path splitting on direct topologies, especially `3D-Torus`.

Instead of enumerating every shortest path as a full node list, compute shortest-path traffic splitting over the shortest-path DAG of each Union plane.

For a given source Union and destination Union:

1. Build the plane graph.
2. Compute shortest-path distance labels.
3. Restrict the graph to shortest-path edges only.
4. Count shortest paths dynamically.
5. Propagate flow mass across the DAG so every shortest path receives equal weight.
6. Accumulate exact backend-link load contributions without materializing every path as a `RoutedPath` object.

This keeps the semantics identical to:

- split across all shortest Union-to-Union paths
- no fixed dimension order
- source SSU traffic first splits evenly across the two source-side Unions
- destination-side traffic still splits evenly down through the two destination Unions

#### Expected Scope of Change

This acceleration should be used where the full path list is not needed for presentation and only aggregate load metrics are required.

That means the project may keep explicit path enumeration for:

- routing tests
- small topologies
- routing diversity samples

but use the DAG-based evaluator for workload-performance projection in the heavy `3D-Torus + SHORTEST_PATH` case.

### 5. Routing Diversity Snapshot

Keep the recently added routing-diversity summary for direct-connect topologies.

It should continue to summarize path diversity differences between:

- `DOR`
- `SHORTEST_PATH`
- `FULL_PATH`

This snapshot remains qualitative-supporting context, while the new routing comparison tables become the primary quantitative performance comparison mechanism.

## Data Model Changes

### `AnalysisConfig`

No new mandatory user-facing routing modes are required.

Optional internal controls may be added for:

- enabling the shortest-path DAG fast path
- selecting thresholds for when to use the fast path automatically

Those controls should default to the safe, exact, optimized behavior so the normal CLI remains simple.

### Pipeline Result Payload

Each topology result payload should be extended with something conceptually equivalent to:

- `default_routing_highlight`
- `routing_comparison`

Recommended shape:

- `default_routing_highlight`
  - `mode`
  - `A2A.per_ssu_throughput_gbps`
  - `Sparse 1-to-N.per_ssu_throughput_gbps`
- `routing_comparison`
  - `A2A.rows[]`
  - `Sparse 1-to-N.rows[]`

Each row should contain:

- `mode`
- `per_ssu_throughput_gbps`
- `completion_time_s`
- `completion_time_p95_s`
- `max_link_utilization`
- `link_utilization_cv`

## File-Level Design

### `topo_sim/routing.py`

- preserve existing public routing semantics
- add internal helpers for shortest-path DAG extraction and path-count propagation
- keep explicit path enumeration for tests and lightweight use cases
- expose enough internal structure for metrics to evaluate exact shortest-path splitting efficiently

### `topo_sim/metrics.py`

- add a fast, exact evaluation path for heavy shortest-path workloads
- keep current output metric definitions unchanged
- ensure `per_ssu_throughput_gbps`, `completion_time_s`, `completion_time_p95_s`, `max_link_utilization`, and `link_utilization_cv` match existing semantics

### `topo_sim/pipeline.py`

- change direct-topology analysis from single-routing-mode execution to multi-routing comparison execution
- keep `Clos` on `ECMP` only
- populate compact comparison-table payloads
- populate main-result throughput highlights using `DOR` for direct topologies and `ECMP` for `Clos`
- update Clos topology metadata so per-Union uplinks are reported correctly and total exchange-node uplink bandwidth is `8 x 400 Gbps`

### `templates/dashboard.html.j2`

- add compact routing comparison tables beneath the main topology summary for direct-connect topologies
- keep table styling dense and secondary to the topology figure
- preserve dark theme and current layout hierarchy
- add a compact selected-node link-bandwidth readout for highlighted incident edges

### `topo_sim/report.py`

- add PDF sections for compact routing comparison tables
- keep them visually aligned with the dark dashboard style
- ensure `Clos` omits direct-routing comparison sections

### `main.py`

- keep the existing CLI working
- default full analysis should produce the complete multi-routing direct-topology comparison automatically
- retain `--routing-mode` for focused analysis / compatibility as needed

### `README.md`

Update documentation to explain:

- direct-connect topologies now compare `DOR`, `SHORTEST_PATH`, and `FULL_PATH`
- `Clos` evaluates `ECMP` only
- the meaning of the compact routing comparison tables
- the default per-flow message size and resulting interpretation of per-SSU data volume

## Validation Plan

### Correctness Validation

- verify `DOR`, `SHORTEST_PATH`, and `FULL_PATH` remain semantically distinct on:
  - `2D-FullMesh`
  - `2D-Torus`
  - `3D-Torus`
- verify `Clos` remains `ECMP` only
- verify the fast shortest-path evaluator matches explicit shortest-path splitting on tractable cases

### Output Validation

- dashboard shows compact `A2A` and `Sparse 1-to-N` routing comparison tables for direct topologies
- dashboard does not show those direct-topology comparison tables for `Clos`
- PDF shows matching compact comparison tables
- main result blocks only show default throughput highlights, not the full five-metric comparison set

### Runtime Validation

- default full analysis run must complete for all four topologies
- `3D-Torus + SHORTEST_PATH` must no longer be the blocker preventing full completion

## Acceptance Criteria

This design is complete when all of the following are true:

- `2D-FullMesh`, `2D-Torus`, and `3D-Torus` each show compact routing comparison tables for `A2A` and `Sparse 1-to-N`
- those tables compare `DOR`, `SHORTEST_PATH`, and `FULL_PATH`
- each row contains exactly:
  - `per_ssu_throughput_gbps`
  - `completion_time_s`
  - `completion_time_p95_s`
  - `max_link_utilization`
  - `link_utilization_cv`
- the main result block shows only throughput highlights for the default route view
- the default route view is `DOR` for direct topologies and `ECMP` for `Clos`
- `Clos` shows only `ECMP` results for `A2A` and `Sparse 1-to-N`
- `Clos` reports `4 x 400 Gbps` uplinks per Union and `8 x 400 Gbps` aggregate uplinks per exchange node
- clicking a node in the dashboard highlights neighboring links and shows the bandwidth of each highlighted incident link
- the default full run completes without needing a special preview workflow
- the optimized `3D-Torus + SHORTEST_PATH` path remains exact, not approximate
