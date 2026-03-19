from __future__ import annotations

import statistics
from collections import defaultdict

import networkx as nx
import numpy as np

from .config import AnalysisConfig


Edge = tuple[object, object]


def canonical_edge_key(u: object, v: object) -> Edge:
    return tuple(sorted((u, v), key=lambda x: str(x)))


def simulate_random_traffic(g: nx.Graph, cfg: AnalysisConfig) -> dict[str, float]:
    if g.number_of_nodes() < 2 or g.number_of_edges() == 0:
        return {
            "sim_avg_latency_ms": 0.0,
            "sim_p95_latency_ms": 0.0,
            "sim_max_link_utilization": 0.0,
            "sim_dropped_ratio": 0.0,
        }

    rng = np.random.default_rng(cfg.random_seed)
    nodes = list(g.nodes())
    msg_bits = cfg.message_size_mb * 8_000_000.0

    flows: list[list[object]] = []
    edge_bits = defaultdict(float)

    for _ in range(cfg.traffic_samples):
        src, dst = rng.choice(nodes, size=2, replace=False)
        path = nx.shortest_path(g, source=src, target=dst)
        flows.append(path)
        for u, v in zip(path[:-1], path[1:]):
            edge_bits[canonical_edge_key(u, v)] += msg_bits

    edge_util: dict[Edge, float] = {}
    for u, v, data in g.edges(data=True):
        key = canonical_edge_key(u, v)
        bw = float(data.get("bandwidth_gbps", cfg.link_bandwidth_gbps)) * 1e9
        capacity = bw * cfg.simulation_window_s
        util = edge_bits[key] / max(capacity, 1.0)
        edge_util[key] = util

    latencies_ms = []
    dropped = 0
    for path in flows:
        total_s = 0.0
        for u, v in zip(path[:-1], path[1:]):
            key = canonical_edge_key(u, v)
            edge_data = g.get_edge_data(u, v) or {}
            bw = float(edge_data.get("bandwidth_gbps", cfg.link_bandwidth_gbps)) * 1e9
            service_s = msg_bits / max(bw, 1.0)

            rho = min(edge_util[key], 0.98)
            queue_s = (rho * service_s) / max(1.0 - rho, 1e-6)
            prop_s = (cfg.hop_latency_us + cfg.switch_latency_us) * 1e-6
            total_s += prop_s + service_s + queue_s

            if edge_util[key] > 1.0:
                dropped += 1

        latencies_ms.append(total_s * 1e3)

    p95 = float(np.percentile(latencies_ms, 95))
    return {
        "sim_avg_latency_ms": float(statistics.fmean(latencies_ms)),
        "sim_p95_latency_ms": p95,
        "sim_max_link_utilization": float(max(edge_util.values()) if edge_util else 0.0),
        "sim_dropped_ratio": dropped / max(len(flows), 1),
    }
