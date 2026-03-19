# SSU-Centric Topology, Routing, and Metrics Redesign

## Summary

This design replaces the current generic topology toolkit with a hardware-constrained, SSU-centric model tailored to four target topologies:

- `2D-FullMesh`
- `2D-Torus`
- `3D-Torus`
- `Clos`

The redesign changes three core aspects of the project:

1. Topology construction is based on a real data-exchange-node template rather than abstract switch graphs.
2. Performance evaluation is defined from the perspective of `SSU -> SSU` communication.
3. Reports and exports are updated to describe topology configuration, routing configuration, workload configuration, and quantitative metrics.

This design also removes the current legacy metrics centered on generic latency/cost estimation.

## Goals

- Keep only the four required topology options.
- Model each topology using a shared data-exchange-node hardware template.
- Measure topology structure and communication performance at SSU granularity.
- Support multiple routing strategies depending on topology type.
- Evaluate both all-to-all and sparse random communication workloads.
- Generate machine-readable summaries and a PDF report that explains the evaluated configuration and results.

## Non-Goals

- Modeling front-end network traffic through the `1825` NICs as explicit graph nodes.
- Performing cycle-accurate or packet-level simulation.
- Preserving backward compatibility with the current `ring`, `star`, `mesh`, or old `torus` naming and metrics.
- Keeping the current cost-centric comparison model.

## Current Project Problems

- Existing topologies are abstract graph templates and do not reflect the Union/SSU hardware structure.
- The current metrics mix node-level and switch-level viewpoints.
- The current simulation assumes shortest-path random traffic only.
- The PDF report is tied to legacy fields such as estimated latency, cost, and old A2Av efficiency.

## Hardware Model

### Data Exchange Node Template

Each data exchange node is modeled explicitly with:

- `8` SSU chips
- `2` Union switching chips

Each SSU has:

- `2 x 200 Gbps` UB ports
- One UB port connects to Union A
- One UB port connects to Union B
- Total SSU southbound-to-node-network attachment bandwidth: `400 Gbps`

Each Union has up to `18 x 400 Gbps` UB ports:

- `8` ports are used to connect downward to the `8` SSUs
- These downward links are modeled at `200 Gbps` per SSU-facing connection so that they match the SSU single-lane rate
- `4` ports are reserved for `1825` NIC attachment to maintain a `1:1` bandwidth ratio between SSU interconnect bandwidth and front-end NIC bandwidth
- Up to `6` ports remain available for back-end topology interconnect

`1825` NICs are not represented as graph nodes in this redesign. They remain a configuration constraint that consumes Union ports and reduces back-end port budget.

### Internal Graph Representation

The graph explicitly contains:

- `SSU` nodes
- `Union` nodes
- Internal `SSU <-> Union` links
- External `Union <-> Union` back-end links

Node metadata should distinguish at least:

- `node_type`
- `node_role`
- `exchange_node_id`
- `local_index`

Link metadata should distinguish at least:

- `bandwidth_gbps`
- `link_kind`
- `topology_role`

Recommended `link_kind` values:

- `internal_ssu_uplink`
- `backend_interconnect`

## Supported Topologies

### Display Names

Externally visible topology names should use:

- `2D-FullMesh`
- `2D-Torus`
- `3D-Torus`
- `Clos`

The code may map these names to internal identifiers, but reports and user-facing outputs should keep the display names above.

### 2D-FullMesh

- Scale: `4 x 4` data exchange nodes, total `16` exchange nodes
- Definition: each exchange node connects to all nodes in the same row and all nodes in the same column
- Logical neighbor count per exchange node: `6`
- Default interconnect bandwidth per logical neighbor: `1 x 400 Gbps`

This interpretation is not a complete graph over all `16` exchange nodes. It is a row-and-column full-connect topology on a `4 x 4` placement.

### 2D-Torus

- Scale: `4 x 4` data exchange nodes, total `16` exchange nodes
- Four periodic directions
- Default interconnect bandwidth per logical direction: `1 x 400 Gbps`

### 3D-Torus

- Scale: `4 x 4 x 4` data exchange nodes, total `64` exchange nodes
- Six periodic directions
- Default interconnect bandwidth per logical direction: `1 x 400 Gbps`

### Clos

- Scale: `18` data exchange nodes
- A higher-layer Union switching stage is added to expand network scale
- Each upper-layer Union switch can connect downward to at most `18` data exchange nodes
- Each data exchange node uses `4 x 400 Gbps` uplinks to the Clos fabric

This design treats `4` uplinks per exchange node as the required default because it is sufficient to achieve non-oversubscribed SSU interconnect for the target Clos case.

## Port Allocation Rules

- Default rule: each logical neighbor or direction should have identical interconnect bandwidth.
- Default starting point: `1 x 400 Gbps` per logical neighbor or direction.
- If logical interconnect demand exceeds directly available physical links, link aggregation is allowed.
- When aggregation is used, bandwidth must remain uniform across all logical neighbors or directions within the same topology instance.
- For Clos, the default uplink count is fixed at `4 x 400 Gbps` per exchange node in this phase.

## Routing Design

All communication is evaluated at `SSU -> SSU` granularity.

An end-to-end `SSU -> SSU` route is composed of:

1. Source SSU to one of the source node's two Union chips
2. Inter-node Union-to-Union back-end traversal
3. Destination Union to destination SSU

The project should introduce a routing layer that computes path sets independently from topology generation and traffic generation.

### Routing Modes

#### DOR

Used for:

- `2D-Torus`
- `3D-Torus`

Definition:

- Dimension-order routing along a deterministic dimension sequence
- Example: `X -> Y` in 2D, `X -> Y -> Z` in 3D
- Uses shortest-path dimension progress only

#### PORT_BALANCED

Used for:

- `2D-FullMesh`
- Optionally torus topologies when comparing path-diverse routing against DOR

Definition:

- Traffic is evenly split across all available source-side egress ports
- Each egress port corresponds to one selected path
- If a shortest path exists through that egress port, a shortest path is used
- If no shortest path exists through that egress port, a non-shortest path may be used
- Non-shortest routing is therefore a fallback used to preserve even distribution across egress options

Recommended default control:

- Maximum additional detour compared with the shortest path: `1` hop

#### ECMP

Used for:

- `Clos`

Definition:

- Equal-cost multi-path routing across shortest paths
- Traffic is evenly split over equal-cost alternatives

## Workload Design

### All-to-All

All SSUs participate.

- Every SSU sends to every other SSU
- Demand is symmetric and equal-sized per source-destination pair
- Used to compute:
  - per-SSU effective throughput
  - completion time
  - route-dependent link balance

### Sparse Random 1-to-N

Only a subset of SSUs participate.

- Default active SSU ratio: `25%`
- Each active SSU randomly selects `N` targets
- Default `N = 2`
- Communication pattern is `1-to-N`

This workload is used to evaluate non-uniform, sparse communication behavior under realistic partial activation.

## Metric Definitions

Legacy metrics are removed in this redesign. The following metrics become the primary outputs.

### Structural Metrics

#### `diameter`

Maximum shortest-path hop count over all `SSU -> SSU` pairs in the evaluated graph.

#### `average_hops`

Average shortest-path hop count over all `SSU -> SSU` pairs.

#### `bisection_bandwidth_gbps`

Bandwidth of the minimum weighted cut that splits the SSU set into two balanced halves, computed on the explicit SSU/Union graph using link bandwidth as edge capacity.

#### `max_link_utilization`

Maximum utilization over all back-end interconnect links under the selected workload and routing mode.

#### `link_utilization_cv`

Coefficient of variation of back-end link utilization under the selected workload and routing mode.

This metric serves as the primary link-balance indicator.

### Communication Metrics

#### `a2a_per_ssu_throughput_gbps`

Effective per-SSU throughput achieved during all-to-all communication.

#### `a2a_completion_time`

Completion time for the all-to-all workload.

#### `sparse_per_ssu_throughput_gbps`

Average effective throughput among active SSUs under sparse random `1-to-N` traffic.

#### `sparse_completion_time_p50`

Median completion time across sparse random communications.

#### `sparse_completion_time_p95`

P95 completion time across sparse random communications.

#### `sparse_max_link_utilization`

Maximum back-end link utilization during sparse random communication.

#### `sparse_link_balance_cv`

Coefficient of variation of back-end link utilization during sparse random communication.

## Metrics to Remove

The following current outputs should be removed from the main comparison pipeline:

- `network_cost`
- `estimated_latency_us`
- legacy `a2av_efficiency`
- legacy `ideal_alltoall_us`
- legacy `topology_alltoall_us`
- shortest-path-only random-traffic summary fields that are tied to the old simulation model

## Performance Computation Model

The redesigned performance pipeline should use a route-aware demand projection model:

1. Generate a flow demand matrix at `SSU -> SSU` granularity
2. Expand each demand into one or more routed paths according to the selected routing mode
3. Split traffic across path sets according to the routing policy
4. Aggregate carried load per link
5. Derive bottleneck completion time from per-link offered load and link capacity
6. Compute effective throughput from total delivered data divided by completion time
7. Compute link-balance metrics from the resulting utilization distribution

This replaces the current queueing-style approximation as the primary evaluation method for the new hardware-constrained topologies.

## Configuration Changes

`AnalysisConfig` should be updated to represent hardware defaults, routing choices, and workload options needed by the redesign.

Recommended additions include:

- topology display-name selection or mapping
- sparse workload active ratio with default `0.25`
- sparse workload target count with default `2`
- routing mode selection
- maximum detour hops for `PORT_BALANCED`, default `1`
- Clos uplink count per exchange node, default `4`
- topology size defaults for:
  - `2D-FullMesh = 4 x 4`
  - `2D-Torus = 4 x 4`
  - `3D-Torus = 4 x 4 x 4`
  - `Clos = 18` exchange nodes
- message size and random seed

## Module Structure

The redesign should separate concerns into focused modules.

### `topologies.py`

Responsibilities:

- define the exchange-node hardware template
- build the four supported topologies
- attach standardized node/link metadata

### `routing.py`

New module responsibilities:

- implement `DOR`
- implement `PORT_BALANCED`
- implement `ECMP`
- enumerate candidate paths and traffic splits

### `traffic.py`

New module responsibilities:

- generate all-to-all demands
- generate sparse random `1-to-N` demands
- normalize flow objects before routing

### `metrics.py`

Responsibilities after redesign:

- compute structural metrics
- compute link-balance metrics
- compute workload performance metrics from routed traffic results

`metrics.py` should stop owning traffic-generation or legacy cost/latency definitions.

### `simulation.py`

Recommended role after redesign:

- host shared demand-to-link projection utilities if a separate execution engine is still desired
- otherwise shrink significantly or be replaced by `traffic.py` and route-evaluation helpers

### `pipeline.py`

Responsibilities:

- build topologies
- apply selected routing policies
- run the defined workloads
- aggregate outputs for CSV, JSON, HTML, and PDF export

## Reporting Requirements

The PDF report remains a required output and should be redefined around the new model.

The report should include:

- project title and generation timestamp
- evaluated topology names
- hardware assumptions for a data exchange node
- topology-specific configuration
- routing configuration per topology
- workload configuration:
  - all-to-all settings
  - sparse random active ratio
  - sparse random target count
- quantitative metric tables
- route/model notes explaining how DOR, PORT_BALANCED, and ECMP are interpreted

Recommended PDF sections:

1. Executive summary
2. Hardware and topology configuration
3. Routing and workload configuration
4. Structural metric comparison
5. Communication metric comparison
6. Key observations

The machine-readable outputs should also be updated:

- `summary.csv` should export only the redesigned metric set
- `run_config.json` should include topology, routing, and workload settings
- HTML output should reflect the same topology naming and metric definitions

## Implementation Notes

- Existing legacy topology builders should be removed or replaced rather than left as primary options.
- User-facing names should stay in the new display format even if internal identifiers are normalized.
- The first implementation can prioritize correctness and clarity over heavy optimization.
- The design should preserve extensibility for future additions such as alternate torus routing modes or more detailed front-end modeling.

## Acceptance Criteria

The redesign is complete when all of the following are true:

- only `2D-FullMesh`, `2D-Torus`, `3D-Torus`, and `Clos` are exposed as supported topologies
- each topology is built from the shared `8 SSU + 2 Union` exchange-node template
- `1825` NICs are treated as port-budget constraints, not graph nodes
- routing is topology-aware and supports `DOR`, `PORT_BALANCED`, and `ECMP`
- all-to-all and sparse random `1-to-N` workloads are both evaluated
- legacy metrics are removed from the main pipeline
- the new structural and communication metrics are exported
- the PDF report documents topology configuration, routing configuration, workload configuration, and resulting metrics
