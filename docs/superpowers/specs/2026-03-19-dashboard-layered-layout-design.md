# Dashboard Layered Layout Redesign

**Date:** 2026-03-19

## Goal

Redesign the HTML dashboard so topology diagrams lead the page visually, explanatory text is deemphasized, and each topology uses an explicit layered layout instead of force-directed placement.

## Approved Decisions

- Overall page uses a black-first theme.
- Use a single-page dashboard with one primary card per topology.
- Within each topology card, content order is:
  1. topology figure
  2. key metrics
  3. routing and workload configuration
  4. smaller analysis / observation text
- Node click interaction only highlights one-hop neighboring nodes and directly incident edges.
- `3D-Torus` uses four `4x4` plane blocks, one per `z` layer.

## Layout Requirements

### Page Layout

- Compress the top hero area so the page reaches the topology figures quickly.
- Put the topology figures at the start of each topology card.
- Reduce font size and visual weight of explanatory text and long notes.
- Keep the metrics readable, but secondary to the figures.

### Topology Figure Layout

Do not use spring / force-directed / spherical placement for the main dashboard figures.

#### Direct-connect topologies

For each exchange node:
- `8` SSUs on the bottom row in a straight horizontal line
- `2` Unions on the row above

For exchange-node placement:
- `2D-FullMesh` and `2D-Torus`: exchange nodes arranged in a `4x4` grid
- `3D-Torus`: arranged as four separate `4x4` blocks, one block per `z` layer

#### Clos topology

For each exchange node:
- `8` SSUs on the bottom row in a straight horizontal line
- `2` Unions on the row above

For topology-level placement:
- exchange nodes occupy the lower area in a structured grid
- Clos spine / second-layer switches occupy a higher layer above all exchange nodes

## Interaction Requirements

- Clicking a node highlights:
  - the clicked node
  - its one-hop neighboring nodes
  - directly connected edges
- Non-selected nodes and unrelated edges are dimmed.
- Clicking empty space should reset the view if practical; otherwise a second click on the same node may restore default styling.

## Technical Direction

- Generate deterministic per-topology coordinates directly in `visualization.py`.
- Prefer explicit geometry over physics-based layout.
- Use Plotly traces and client-side JavaScript hooks to support node click highlighting.
- Preserve existing HTML generation entry points so the pipeline does not need a structural rewrite.

## Validation Targets

- HTML contains dark-theme structure and new layout labels.
- Topology figures use deterministic layered coordinates.
- Dashboard remains renderable without `scipy`.
- Clicking nodes in the generated dashboard can highlight immediate neighbors and incident links.
