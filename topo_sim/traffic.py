from __future__ import annotations

from dataclasses import dataclass
import math
import random

import networkx as nx

from .config import AnalysisConfig


@dataclass(slots=True)
class FlowDemand:
    src: str
    dst: str
    bits: float


def _ssu_nodes(g: nx.Graph) -> list[str]:
    return [node_id for node_id, data in g.nodes(data=True) if data.get("node_role") == "ssu"]


def _validated_message_bits(cfg: AnalysisConfig) -> float:
    if cfg.message_size_mb <= 0:
        raise ValueError("message_size_mb must be > 0")
    return cfg.message_size_mb * 8_000_000.0


def _validate_sparse_inputs(cfg: AnalysisConfig) -> None:
    if not (0 < cfg.sparse_active_ratio <= 1):
        raise ValueError("sparse_active_ratio must be in (0, 1]")
    if type(cfg.sparse_target_count) is not int or cfg.sparse_target_count < 0:
        raise ValueError("sparse_target_count must be an integer >= 0")


def _active_source_count(ssu_count: int, sparse_active_ratio: float) -> int:
    # Explicit policy: ceil(ssu_count * ratio), capped by available SSUs.
    return min(ssu_count, math.ceil(ssu_count * sparse_active_ratio))


def build_a2a_demands(g: nx.Graph, cfg: AnalysisConfig) -> list[FlowDemand]:
    ssus = _ssu_nodes(g)
    bits = _validated_message_bits(cfg)
    demands: list[FlowDemand] = []

    for src in ssus:
        for dst in ssus:
            if src == dst:
                continue
            demands.append(FlowDemand(src=src, dst=dst, bits=bits))

    return demands


def build_sparse_random_demands(g: nx.Graph, cfg: AnalysisConfig) -> list[FlowDemand]:
    bits = _validated_message_bits(cfg)
    _validate_sparse_inputs(cfg)

    ssus = _ssu_nodes(g)
    if not ssus:
        return []

    rng = random.Random(cfg.random_seed)
    active_count = _active_source_count(len(ssus), cfg.sparse_active_ratio)
    targets_per_source = cfg.sparse_target_count

    demands: list[FlowDemand] = []
    active_sources = rng.sample(ssus, k=active_count)

    for src in active_sources:
        candidates = [dst for dst in ssus if dst != src]
        target_count = min(len(candidates), targets_per_source)
        for dst in rng.sample(candidates, k=target_count):
            demands.append(FlowDemand(src=src, dst=dst, bits=bits))

    return demands
