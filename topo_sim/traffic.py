from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import networkx as nx

from .config import AnalysisConfig


@dataclass(slots=True)
class FlowDemand:
    src: str
    dst: str
    bits: float


@dataclass(slots=True)
class CustomTrafficProfile:
    name: str
    demands: list[FlowDemand]
    description: str
    input_path: str


_VOLUME_FACTORS = {
    "bits": 1.0,
    "bytes": 8.0,
    "kb": 8_000.0,
    "mb": 8_000_000.0,
    "gb": 8_000_000_000.0,
}


def _ssu_sort_key(node_id: str) -> tuple[int, int]:
    exchange_id, local_id = str(node_id).split(":", 1)
    local_digits = "".join(ch for ch in local_id if ch.isdigit())
    return (
        int(exchange_id.removeprefix("en")),
        int(local_digits) if local_digits else 0,
    )


def _ssu_nodes(g: nx.Graph) -> list[str]:
    return sorted(
        [str(node_id) for node_id, data in g.nodes(data=True) if data.get("node_role") == "ssu"],
        key=_ssu_sort_key,
    )


def _ssu_components(g: nx.Graph) -> list[list[str]]:
    components: list[list[str]] = []
    for component_nodes in nx.connected_components(g):
        component_ssus = sorted(
            [
                str(node_id)
                for node_id in component_nodes
                if g.nodes[node_id].get("node_role") == "ssu"
            ],
            key=_ssu_sort_key,
        )
        if component_ssus:
            components.append(component_ssus)
    components.sort(key=lambda nodes: (_ssu_sort_key(nodes[0]), len(nodes)))
    return components


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


def _validated_non_negative_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be an integer >= 0")
    return limit


def _validated_ssu_node(g: nx.Graph, node_id: str) -> str:
    normalized = str(node_id)
    if normalized not in g:
        raise ValueError(f"Unknown node '{normalized}'")
    if g.nodes[normalized].get("node_role") != "ssu":
        raise ValueError(f"Node '{normalized}' is not an SSU")
    return normalized


def _normalized_ssu_selection(
    g: nx.Graph,
    node_ids: Sequence[str] | None,
) -> list[str] | None:
    if node_ids is None:
        return None
    deduped = {_validated_ssu_node(g, node_id) for node_id in node_ids}
    return sorted(deduped, key=_ssu_sort_key)


def _validated_positive_bits(bits: float, label: str) -> float:
    value = float(bits)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be > 0")
    return value


def select_ssu_nodes(
    g: nx.Graph,
    *,
    node_ids: Sequence[str] | None = None,
    exchange_ids: Sequence[str] | None = None,
    server_ids: Sequence[int] | None = None,
    local_indices: Sequence[int] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Select SSUs by explicit ids and/or topology-local filters.

    Multiple filters are combined by intersection. If no filters are provided,
    all SSUs are returned in topology order.
    """

    explicit_nodes = _normalized_ssu_selection(g, node_ids)
    exchange_filter = {str(exchange_id) for exchange_id in exchange_ids} if exchange_ids else None
    server_filter = {int(server_id) for server_id in server_ids} if server_ids else None
    local_index_filter = {int(local_index) for local_index in local_indices} if local_indices else None
    node_limit = _validated_non_negative_limit(limit)

    selected: list[str] = []
    for node_id in _ssu_nodes(g):
        node_data = g.nodes[node_id]
        if explicit_nodes is not None and node_id not in explicit_nodes:
            continue
        if exchange_filter is not None and str(node_data.get("exchange_node_id")) not in exchange_filter:
            continue
        if server_filter is not None and node_data.get("server_id") not in server_filter:
            continue
        if local_index_filter is not None and int(node_data.get("local_index", -1)) not in local_index_filter:
            continue
        selected.append(node_id)

    if node_limit is not None:
        return selected[:node_limit]
    return selected


def build_controlled_m2n_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_ssus: Sequence[str] | None = None,
    destination_ssus: Sequence[str] | None = None,
    pair_bits: Mapping[tuple[str, str], float] | None = None,
    default_bits: float | None = None,
    include_self: bool = False,
) -> list[FlowDemand]:
    """Build an explicit M-to-N workload with per-pair traffic control.

    Usage patterns:
    - Provide `source_ssus` and `destination_ssus` for a Cartesian-product M-to-N workload.
    - Provide `pair_bits` for fully explicit per-pair traffic sizes.
    - Provide both to set a default M-to-N traffic matrix and override selected pairs.
    """

    if source_ssus is None and destination_ssus is None and not pair_bits:
        raise ValueError(
            "At least one of source_ssus/destination_ssus or pair_bits must be provided"
        )

    normalized_sources = (
        _normalized_ssu_selection(g, source_ssus) if source_ssus is not None else None
    )
    normalized_destinations = (
        _normalized_ssu_selection(g, destination_ssus) if destination_ssus is not None else None
    )
    if normalized_sources is None and normalized_destinations is not None:
        normalized_sources = _ssu_nodes(g)
    if normalized_destinations is None and normalized_sources is not None:
        normalized_destinations = _ssu_nodes(g)

    demand_map: dict[tuple[str, str], float] = {}
    if normalized_sources is not None and normalized_destinations is not None:
        base_pair_bits = (
            _validated_message_bits(cfg)
            if default_bits is None
            else _validated_positive_bits(default_bits, "default_bits")
        )
        for src in normalized_sources:
            for dst in normalized_destinations:
                if src == dst and not include_self:
                    continue
                demand_map[(src, dst)] = base_pair_bits

    for pair, bits in (pair_bits or {}).items():
        if len(pair) != 2:
            raise ValueError("pair_bits keys must be (src_ssu, dst_ssu) tuples")
        src = _validated_ssu_node(g, pair[0])
        dst = _validated_ssu_node(g, pair[1])
        if src == dst and not include_self:
            continue
        demand_map[(src, dst)] = _validated_positive_bits(bits, "pair_bits value")

    return [
        FlowDemand(src=src, dst=dst, bits=bits)
        for (src, dst), bits in sorted(
            demand_map.items(),
            key=lambda item: (_ssu_sort_key(item[0][0]), _ssu_sort_key(item[0][1])),
        )
    ]


def _selector_kwargs(selector: Mapping[str, Any] | None) -> dict[str, Any]:
    if selector is None:
        return {}
    allowed = {"node_ids", "exchange_ids", "server_ids", "local_indices", "limit"}
    unknown = set(selector) - allowed
    if unknown:
        raise ValueError(f"Unsupported selector keys: {', '.join(sorted(unknown))}")
    return {
        key: selector.get(key)
        for key in ("node_ids", "exchange_ids", "server_ids", "local_indices", "limit")
        if key in selector
    }


def _volume_bits_from_mapping(
    payload: Mapping[str, Any],
    *,
    keys: Sequence[str],
    label: str,
) -> float:
    present = [
        key for key in keys
        if key in payload and payload.get(key) not in (None, "")
    ]
    if not present:
        raise ValueError(f"{label} must include one of: {', '.join(keys)}")
    if len(present) > 1:
        raise ValueError(f"{label} must specify exactly one volume field")

    field = present[0]
    unit = field.removeprefix("default_").lower()
    factor = _VOLUME_FACTORS.get(unit)
    if factor is None:
        raise ValueError(f"Unsupported volume field '{field}' in {label}")
    value = float(payload[field]) * factor
    return _validated_positive_bits(value, label)


def _build_explicit_flow_demands(
    g: nx.Graph,
    flow_items: Sequence[Mapping[str, Any]],
    *,
    label_prefix: str,
) -> list[FlowDemand]:
    demands: list[FlowDemand] = []
    for index, item in enumerate(flow_items):
        label = f"{label_prefix}[{index}]"
        src = _validated_ssu_node(g, str(item.get("src")))
        dst = _validated_ssu_node(g, str(item.get("dst")))
        bits = _volume_bits_from_mapping(item, keys=tuple(_VOLUME_FACTORS.keys()), label=label)
        if src == dst:
            continue
        demands.append(FlowDemand(src=src, dst=dst, bits=bits))
    return demands


def _custom_profile_description(
    name: str,
    demands: Sequence[FlowDemand],
    input_path: Path,
    description: str | None = None,
) -> str:
    if description:
        return description
    total_volume_gb = sum(demand.bits for demand in demands) / 8e9
    source_count = len({demand.src for demand in demands})
    destination_count = len({demand.dst for demand in demands})
    return (
        f"{len(demands)} directed flows from {source_count} source SSUs to "
        f"{destination_count} destination SSUs, total offered volume {total_volume_gb:.2f} GB "
        f"(from {input_path.name})."
    )


def _load_custom_traffic_from_csv(
    g: nx.Graph,
    input_path: Path,
    *,
    workload_name: str,
) -> CustomTrafficProfile:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    demands = _build_explicit_flow_demands(g, rows, label_prefix="csv_flow")
    if not demands:
        raise ValueError(f"No valid demands found in {input_path}")
    return CustomTrafficProfile(
        name=workload_name,
        demands=demands,
        description=_custom_profile_description(workload_name, demands, input_path),
        input_path=str(input_path),
    )


def _load_custom_traffic_from_json(
    g: nx.Graph,
    input_path: Path,
    *,
    workload_name: str,
) -> CustomTrafficProfile:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"flows": payload}
    if not isinstance(payload, dict):
        raise ValueError("Custom traffic JSON must be an object or a list of flows")

    profile_name = str(payload.get("name") or workload_name)
    description = payload.get("description")
    demands: list[FlowDemand] = []

    flow_items = payload.get("flows", [])
    if flow_items:
        if not isinstance(flow_items, list):
            raise ValueError("'flows' must be a list")
        demands.extend(_build_explicit_flow_demands(g, flow_items, label_prefix="json_flow"))

    group_items = payload.get("groups", [])
    if group_items:
        if not isinstance(group_items, list):
            raise ValueError("'groups' must be a list")
        for index, group in enumerate(group_items):
            if not isinstance(group, dict):
                raise ValueError(f"groups[{index}] must be an object")
            sources = select_ssu_nodes(g, **_selector_kwargs(group.get("sources")))
            destinations = select_ssu_nodes(g, **_selector_kwargs(group.get("destinations")))
            if not sources:
                raise ValueError(f"groups[{index}] selected no source SSUs")
            if not destinations:
                raise ValueError(f"groups[{index}] selected no destination SSUs")

            override_map: dict[tuple[str, str], float] = {}
            pair_overrides = group.get("pair_overrides", [])
            if pair_overrides:
                if not isinstance(pair_overrides, list):
                    raise ValueError(f"groups[{index}].pair_overrides must be a list")
                for override_index, item in enumerate(pair_overrides):
                    label = f"groups[{index}].pair_overrides[{override_index}]"
                    src = _validated_ssu_node(g, str(item.get("src")))
                    dst = _validated_ssu_node(g, str(item.get("dst")))
                    override_map[(src, dst)] = _volume_bits_from_mapping(
                        item,
                        keys=tuple(_VOLUME_FACTORS.keys()),
                        label=label,
                    )

            demands.extend(
                build_controlled_m2n_demands(
                    g,
                    cfg=AnalysisConfig(message_size_mb=1.0),
                    source_ssus=sources,
                    destination_ssus=destinations,
                    pair_bits=override_map,
                    default_bits=_volume_bits_from_mapping(
                        group,
                        keys=tuple(f"default_{key}" for key in _VOLUME_FACTORS),
                        label=f"groups[{index}]",
                    ),
                    include_self=bool(group.get("include_self", False)),
                )
            )

    if not demands:
        raise ValueError(f"No valid demands found in {input_path}")

    demand_map: dict[tuple[str, str], float] = {}
    for demand in demands:
        demand_map[(demand.src, demand.dst)] = demand_map.get((demand.src, demand.dst), 0.0) + demand.bits
    merged_demands = [
        FlowDemand(src=src, dst=dst, bits=bits)
        for (src, dst), bits in sorted(
            demand_map.items(),
            key=lambda item: (_ssu_sort_key(item[0][0]), _ssu_sort_key(item[0][1])),
        )
    ]

    return CustomTrafficProfile(
        name=profile_name,
        demands=merged_demands,
        description=_custom_profile_description(profile_name, merged_demands, input_path, description),
        input_path=str(input_path),
    )


def load_custom_traffic_profile(
    g: nx.Graph,
    traffic_path: str | Path,
    *,
    workload_name: str = "Custom M-to-N",
) -> CustomTrafficProfile:
    input_path = Path(traffic_path)
    if not input_path.exists():
        raise ValueError(f"Custom traffic file does not exist: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return _load_custom_traffic_from_csv(g, input_path, workload_name=workload_name)
    if suffix == ".json":
        return _load_custom_traffic_from_json(g, input_path, workload_name=workload_name)
    raise ValueError("Custom traffic file must end with .csv or .json")


def build_a2a_demands(g: nx.Graph, cfg: AnalysisConfig) -> list[FlowDemand]:
    bits = _validated_message_bits(cfg)
    demands: list[FlowDemand] = []

    for component_ssus in _ssu_components(g):
        for src in component_ssus:
            for dst in component_ssus:
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
    component_by_ssu = {
        ssu_id: component_ssus
        for component_ssus in _ssu_components(g)
        for ssu_id in component_ssus
    }

    demands: list[FlowDemand] = []
    active_sources = rng.sample(ssus, k=active_count)

    for src in active_sources:
        candidates = [dst for dst in component_by_ssu.get(src, []) if dst != src]
        target_count = min(len(candidates), targets_per_source)
        for dst in rng.sample(candidates, k=target_count):
            demands.append(FlowDemand(src=src, dst=dst, bits=bits))

    return demands
