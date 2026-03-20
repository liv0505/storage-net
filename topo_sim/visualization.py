from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape


_INTERNAL_EDGE_COLOR = "rgba(120, 138, 160, 0.22)"
_BACKEND_EDGE_COLOR = "rgba(95, 227, 193, 0.28)"
_HIGHLIGHT_EDGE_COLOR = "rgba(255, 190, 92, 0.96)"
_DEFAULT_NODE_COLOR = "#7dd3fc"
_UNION_NODE_COLOR = "#f59e0b"
_SSU_NODE_COLOR = "#34d399"
_SPINE_NODE_COLOR = "#fb7185"


def _fallback_positions(g: nx.Graph) -> dict[Any, tuple[float, float]]:
    circular = nx.circular_layout(g)
    return {k: (float(v[0]), float(v[1])) for k, v in circular.items()}


def _exchange_local_positions(base_x: float, base_y: float, exchange_node_id: str) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    ssu_spacing = 0.62
    union_spacing = 1.2

    for ssu_index in range(8):
        positions[f"{exchange_node_id}:ssu{ssu_index}"] = (base_x + (ssu_index * ssu_spacing), base_y)

    union_center = base_x + 3.5 * ssu_spacing
    positions[f"{exchange_node_id}:union0"] = (union_center - (union_spacing / 2.0), base_y + 1.15)
    positions[f"{exchange_node_id}:union1"] = (union_center + (union_spacing / 2.0), base_y + 1.15)
    return positions


def _exchange_grid_positions_2d(rows: int, cols: int) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    cell_x = 7.2
    cell_y = 3.8

    for row in range(rows):
        for col in range(cols):
            exchange_id = f"en{row * cols + col}"
            base_x = col * cell_x
            base_y = -row * cell_y
            positions.update(_exchange_local_positions(base_x, base_y, exchange_id))

    return positions


def _exchange_grid_positions_3d() -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    cell_x = 6.6
    cell_y = 3.7
    block_gap_x = 31.0
    block_gap_y = 18.0
    size = 4

    for x in range(size):
        for y in range(size):
            for z in range(size):
                exchange_index = (x * size * size) + (y * size) + z
                exchange_id = f"en{exchange_index}"
                block_col = z % 2
                block_row = z // 2
                base_x = (block_col * block_gap_x) + (y * cell_x)
                base_y = -(block_row * block_gap_y) - (x * cell_y)
                positions.update(_exchange_local_positions(base_x, base_y, exchange_id))

    return positions


def _clos_spine_sort_key(node_id: str) -> tuple[int, int]:
    parts = str(node_id).split("_")
    plane = next((int(part.removeprefix("plane")) for part in parts if part.startswith("plane")), 0)
    uplink = next((int(part.removeprefix("uplink")) for part in parts if part.startswith("uplink")), 0)
    return plane, uplink


def _exchange_grid_positions_clos(g: nx.Graph) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    cols = 6
    cell_x = 7.4
    cell_y = 3.9

    for index in range(18):
        row = index // cols
        col = index % cols
        exchange_id = f"en{index}"
        base_x = col * cell_x
        base_y = -(row * cell_y) - 2.8
        positions.update(_exchange_local_positions(base_x, base_y, exchange_id))

    spine_nodes = sorted(
        [node_id for node_id, data in g.nodes(data=True) if data.get("node_role") == "clos_spine"],
        key=_clos_spine_sort_key,
    )
    plane_spacing_y = 2.0
    spine_spacing_x = 8.2
    spine_origin_x = 8.5
    spine_origin_y = 3.2
    for node_id in spine_nodes:
        plane, uplink = _clos_spine_sort_key(node_id)
        positions[node_id] = (
            spine_origin_x + (uplink * spine_spacing_x),
            spine_origin_y + (plane * plane_spacing_y),
        )

    return positions


def _explicit_positions(topology_name: str, g: nx.Graph) -> dict[Any, tuple[float, float]] | None:
    if topology_name in {"2D-FullMesh", "2D-Torus"}:
        return _exchange_grid_positions_2d(4, 4)
    if topology_name == "3D-Torus":
        return _exchange_grid_positions_3d()
    if topology_name == "Clos":
        return _exchange_grid_positions_clos(g)
    return None


def _positions(g: nx.Graph, topology_name: str, seed: int) -> dict[Any, tuple[float, float]]:
    existing = nx.get_node_attributes(g, "pos")
    if existing:
        return {k: (float(v[0]), float(v[1])) for k, v in existing.items()}

    explicit = _explicit_positions(topology_name, g)
    if explicit is not None:
        return explicit

    try:
        spring = nx.spring_layout(g, seed=seed)
    except (ImportError, ModuleNotFoundError):
        return _fallback_positions(g)
    return {k: (float(v[0]), float(v[1])) for k, v in spring.items()}


def _node_color(node_data: dict[str, Any]) -> str:
    role = node_data.get("node_role")
    if role == "ssu":
        return _SSU_NODE_COLOR
    if role == "union":
        return _UNION_NODE_COLOR
    if role == "clos_spine":
        return _SPINE_NODE_COLOR
    return _DEFAULT_NODE_COLOR


def _trace_from_edges(
    g: nx.Graph,
    pos: dict[Any, tuple[float, float]],
    *,
    link_kind: str,
    color: str,
    width: float,
    name: str,
) -> go.Scatter:
    edge_x: list[float] = []
    edge_y: list[float] = []

    for u, v, data in g.edges(data=True):
        if data.get("link_kind") != link_kind:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    return go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=width, color=color),
        hoverinfo="skip",
        showlegend=False,
        name=name,
    )


def _build_interaction_payload(
    g: nx.Graph,
    pos: dict[Any, tuple[float, float]],
    node_ids: list[str],
) -> dict[str, Any]:
    node_points: dict[str, Any] = {}
    neighbors: dict[str, list[str]] = {}
    incident_edges: dict[str, list[list[float | None]]] = {}
    incident_link_labels: dict[str, list[dict[str, Any]]] = {}

    for node_id in node_ids:
        node_data = g.nodes[node_id]
        degree = g.degree[node_id]
        x, y = pos[node_id]
        node_points[node_id] = {
            "x": x,
            "y": y,
            "color": _node_color(node_data),
            "size": 12 if node_data.get("node_role") == "ssu" else 15,
            "text": "<br>".join(
                [
                    f"node={node_id}",
                    f"role={node_data.get('node_role', 'unknown')}",
                    f"degree={degree}",
                ]
            ),
        }
        neighbors[node_id] = [str(neighbor) for neighbor in g.neighbors(node_id)]

        edge_segments: list[list[float | None]] = []
        label_items: list[dict[str, Any]] = []
        for neighbor in g.neighbors(node_id):
            x0, y0 = pos[node_id]
            x1, y1 = pos[neighbor]
            edge_segments.append([x0, x1, None, y0, y1, None])
            edge_data = g.get_edge_data(node_id, neighbor) or {}
            bandwidth = float(edge_data.get("bandwidth_gbps", 0.0))
            label_items.append(
                {
                    "x": (x0 + x1) / 2.0,
                    "y": (y0 + y1) / 2.0,
                    "text": f"{bandwidth:.0f} Gbps",
                    "bandwidth_gbps": bandwidth,
                }
            )
        incident_edges[node_id] = edge_segments
        incident_link_labels[node_id] = label_items

    return {
        "node_points": node_points,
        "neighbors": neighbors,
        "incident_edges": incident_edges,
        "incident_link_labels": incident_link_labels,
    }


def _layout_notes(topology_name: str) -> list[str]:
    base_note = "SSUs stay on the bottom row and Unions sit on the layer above inside each exchange node."
    if topology_name == "3D-Torus":
        return [base_note, "Exchange nodes are grouped into 4 z-layers, each rendered as one 4x4 plane block."]
    if topology_name == "Clos":
        return [base_note, "Clos spine layer sits above the exchange-node Union layer for structured two-level viewing."]
    return [base_note, "Exchange nodes are arranged on a structured horizontal and vertical grid."]


def _interaction_script(plot_id: str, payload: dict[str, Any]) -> str:
    interaction_json = json.dumps(payload, separators=(",", ":"))
    return f"""
(function() {{
  const plotId = {json.dumps(plot_id)};
  const interaction = {interaction_json};
  const plot = document.getElementById(plotId);
  if (!plot) return;

  function flattenEdgeSegments(segments) {{
    const x = [];
    const y = [];
    segments.forEach((segment) => {{
      x.push(segment[0], segment[1], segment[2]);
      y.push(segment[3], segment[4], segment[5]);
    }});
    return {{ x, y }};
  }}

  function resetHighlight() {{
    Plotly.restyle(plot, {{opacity: 1}}, [0, 1, 3]);
    Plotly.restyle(plot, {{x: [[]], y: [[]]}}, [2]);
    Plotly.restyle(plot, {{x: [[]], y: [[]], text: [[]], customdata: [[]]}}, [4, 5]);
    const defaultOpacity = interaction.baseNodeIds.map(() => 1);
    Plotly.restyle(plot, {{'marker.opacity': [defaultOpacity]}}, [3]);
    plot.dataset.activeNodeId = '';
  }}

  function highlightNode(nodeId) {{
    const activeIds = [nodeId].concat(interaction.neighbors[nodeId] || []);
    const edgeSegments = interaction.incident_edges[nodeId] || [];
    const flatEdges = flattenEdgeSegments(edgeSegments);
    const linkLabels = interaction.incident_link_labels[nodeId] || [];
    const highlightPoints = activeIds.map((id) => interaction.node_points[id]).filter(Boolean);

    Plotly.restyle(plot, {{opacity: 0.14}}, [0, 1]);
    Plotly.restyle(plot, {{opacity: 0.30}}, [3]);
    const nodeOpacity = interaction.baseNodeIds.map((id) => activeIds.includes(id) ? 1 : 0.12);
    Plotly.restyle(plot, {{'marker.opacity': [nodeOpacity]}}, [3]);
    Plotly.restyle(plot, {{x: [flatEdges.x], y: [flatEdges.y]}}, [2]);
    Plotly.restyle(plot, {{
      x: [linkLabels.map((item) => item.x)],
      y: [linkLabels.map((item) => item.y)],
      text: [linkLabels.map((item) => item.text)],
      customdata: [linkLabels.map((item) => item.bandwidth_gbps)]
    }}, [4]);
    Plotly.restyle(plot, {{
      x: [highlightPoints.map((item) => item.x)],
      y: [highlightPoints.map((item) => item.y)],
      text: [highlightPoints.map((item) => item.text)],
      customdata: [activeIds]
    }}, [5]);
    plot.dataset.activeNodeId = nodeId;
  }}

  plot.on('plotly_click', function(event) {{
    const point = event && event.points && event.points[0];
    if (!point || !point.customdata) return;
    const nodeId = point.customdata;
    if (plot.dataset.activeNodeId === nodeId) {{
      resetHighlight();
      return;
    }}
    highlightNode(nodeId);
  }});

  plot.on('plotly_doubleclick', function() {{
    resetHighlight();
    return false;
  }});

  window.registerTopologyHighlight = window.registerTopologyHighlight || true;
}})();
"""


def create_topology_figure(
    g: nx.Graph,
    title: str,
    structural_metrics: dict[str, float],
    communication_metrics: dict[str, dict[str, float]],
    layout_seed: int,
) -> tuple[go.Figure, dict[str, Any], str]:
    pos = _positions(g, title, layout_seed)

    internal_edges = _trace_from_edges(
        g,
        pos,
        link_kind="internal_ssu_uplink",
        color=_INTERNAL_EDGE_COLOR,
        width=1.0,
        name="internal",
    )
    backend_edges = _trace_from_edges(
        g,
        pos,
        link_kind="backend_interconnect",
        color=_BACKEND_EDGE_COLOR,
        width=1.7,
        name="backend",
    )
    highlight_edges = go.Scatter(
        x=[],
        y=[],
        mode="lines",
        line=dict(width=3.2, color=_HIGHLIGHT_EDGE_COLOR),
        hoverinfo="skip",
        showlegend=False,
        name="highlight-edges",
    )
    highlight_edge_labels = go.Scatter(
        x=[],
        y=[],
        mode="text",
        text=[],
        customdata=[],
        textposition="top center",
        textfont=dict(color="#fde68a", size=11, family="Space Grotesk, Segoe UI, sans-serif"),
        hovertemplate="%{customdata:.0f} Gbps<extra></extra>",
        showlegend=False,
        name="highlight-edge-labels",
    )

    node_ids = [str(node_id) for node_id in g.nodes()]
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []

    for node_id in node_ids:
        node_data = g.nodes[node_id]
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        node_color.append(_node_color(node_data))
        node_size.append(11 if node_data.get("node_role") == "ssu" else 14)
        node_text.append(
            "<br>".join(
                [
                    f"node={node_id}",
                    f"role={node_data.get('node_role', 'unknown')}",
                    f"degree={g.degree[node_id]}",
                ]
            )
        )

    base_nodes = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#030712"), opacity=1),
        hovertemplate="%{text}<extra></extra>",
        text=node_text,
        customdata=node_ids,
        showlegend=False,
        name="nodes",
    )
    highlight_nodes = go.Scatter(
        x=[],
        y=[],
        mode="markers",
        marker=dict(size=18, color="#fde68a", line=dict(width=2, color="#fff7ed")),
        hovertemplate="%{text}<extra></extra>",
        text=[],
        customdata=[],
        showlegend=False,
        name="highlight-nodes",
    )

    a2a = communication_metrics["A2A"]
    sparse = communication_metrics["Sparse 1-to-N"]
    subtitle = (
        f"Diameter: {structural_metrics['diameter']:.0f} | "
        f"Avg Hops: {structural_metrics['average_hops']:.2f} | "
        f"A2A Throughput: {a2a['per_ssu_throughput_gbps']:.2f} Gbps | "
        f"Sparse P95: {sparse['completion_time_p95_s'] * 1e3:.2f} ms"
    )

    fig = go.Figure(
        data=[internal_edges, backend_edges, highlight_edges, base_nodes, highlight_edge_labels, highlight_nodes]
    )
    fig.update_layout(
        title=f"{title}<br><sup>{subtitle}</sup>",
        template="plotly_dark",
        margin=dict(l=16, r=16, t=72, b=16),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#020617",
        height=560,
        clickmode="event",
    )

    interaction = _build_interaction_payload(g, pos, node_ids)
    interaction["baseNodeIds"] = node_ids
    plot_id = f"plot-{title.lower().replace(' ', '-').replace('_', '-')}"
    return fig, interaction, plot_id


def render_html_dashboard(results: list[dict[str, Any]], output_path: Path) -> Path:
    template_dir = Path(__file__).resolve().parent.parent / "templates"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("dashboard.html.j2")

    blocks = []
    for item in results:
        fig, interaction, plot_id = create_topology_figure(
            g=item["graph"],
            title=item["name"],
            structural_metrics=item["structural_metrics"],
            communication_metrics=item["communication_metrics"],
            layout_seed=item["layout_seed"],
        )
        blocks.append(
            {
                "name": item["name"],
                "hardware": item["hardware"],
                "topology": item["topology"],
                "routing": item["routing"],
                "routing_diversity": item.get("routing_diversity"),
                "workloads": item["workloads"],
                "structural_metrics": item["structural_metrics"],
                "communication_metrics": item["communication_metrics"],
                "default_routing_highlight": item.get("default_routing_highlight"),
                "routing_comparison": item.get("routing_comparison"),
                "observations": item["observations"],
                "layout_notes": _layout_notes(item["name"]),
                "plot_id": plot_id,
                "interaction_mode": "neighbors",
                "interaction_script": _interaction_script(plot_id, interaction),
                "plot_html": fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config={"displaylogo": False, "responsive": True},
                    div_id=plot_id,
                ),
            }
        )

    html = template.render(results=blocks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    return output_path
