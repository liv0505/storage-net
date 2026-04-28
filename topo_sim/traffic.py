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
    source_id: str | None = None
    explicit_paths: tuple[tuple[str, ...], ...] | None = None


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


def _exchange_sort_key(exchange_id: str) -> int:
    return int(str(exchange_id).removeprefix("en"))


def _exchange_id(node_id: str) -> str:
    return str(node_id).split(":", 1)[0]


def _local_ssu_index(node_id: str) -> int:
    local_id = str(node_id).split(":", 1)[1]
    digits = "".join(ch for ch in local_id if ch.isdigit())
    return int(digits) if digits else 0


def _local_node_index(node_id: str) -> int:
    local_id = str(node_id).split(":", 1)[1]
    digits = "".join(ch for ch in local_id if ch.isdigit())
    return int(digits) if digits else 0


def _union_sort_key(node_id: str) -> tuple[int, int]:
    exchange_id, local_id = str(node_id).split(":", 1)
    digits = "".join(ch for ch in local_id if ch.isdigit())
    return (
        int(exchange_id.removeprefix("en")),
        int(digits) if digits else 0,
    )


def _ssu_nodes(g: nx.Graph) -> list[str]:
    return sorted(
        [str(node_id) for node_id, data in g.nodes(data=True) if data.get("node_role") == "ssu"],
        key=_ssu_sort_key,
    )


def _union_nodes(g: nx.Graph) -> list[str]:
    return sorted(
        [str(node_id) for node_id, data in g.nodes(data=True) if data.get("node_role") == "union"],
        key=_union_sort_key,
    )


def _npu_nodes(g: nx.Graph) -> list[str]:
    return sorted(
        [str(node_id) for node_id, data in g.nodes(data=True) if data.get("node_role") == "npu"],
        key=_union_sort_key,
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


def _exchange_ssus(g: nx.Graph) -> dict[str, list[str]]:
    exchange_map: dict[str, list[str]] = {}
    for ssu_id in _ssu_nodes(g):
        exchange_map.setdefault(_exchange_id(ssu_id), []).append(ssu_id)
    return {
        exchange_id: sorted(nodes, key=_ssu_sort_key)
        for exchange_id, nodes in sorted(exchange_map.items(), key=lambda item: _exchange_sort_key(item[0]))
    }


def _exchange_ids(g: nx.Graph) -> list[str]:
    return list(_exchange_ssus(g).keys())


def _exchange_coord(g: nx.Graph, exchange_id: str) -> tuple[int, ...] | None:
    probe_node = f"{exchange_id}:ssu0"
    if probe_node not in g:
        probe_node = f"{exchange_id}:union0"
    coord = g.nodes[probe_node].get("exchange_grid_coord") if probe_node in g else None
    if coord is None:
        return None
    return tuple(int(value) for value in coord)


def _component_exchanges_by_exchange(g: nx.Graph) -> dict[str, list[str]]:
    component_map: dict[str, list[str]] = {}
    for component_nodes in nx.connected_components(g):
        exchange_ids = sorted(
            {
                _exchange_id(str(node_id))
                for node_id in component_nodes
                if g.nodes[node_id].get("node_role") == "ssu"
            },
            key=_exchange_sort_key,
        )
        for exchange_id in exchange_ids:
            component_map[exchange_id] = exchange_ids
    return component_map


def _target_ssu(exchange_map: dict[str, list[str]], exchange_id: str, local_index: int) -> str:
    ssus = exchange_map[exchange_id]
    return ssus[int(local_index) % len(ssus)]


def _spread_sample(items: Sequence[str], count: int) -> list[str]:
    ordered = list(items)
    if count >= len(ordered):
        return ordered
    selected: list[str] = []
    used: set[str] = set()
    total = len(ordered)
    for index in range(count):
        position = int(index * total / count)
        candidate = ordered[position]
        if candidate in used:
            for fallback in ordered:
                if fallback not in used:
                    candidate = fallback
                    break
        selected.append(candidate)
        used.add(candidate)
    return selected


def _chunked_exchange_groups(exchange_ids: Sequence[str], group_size: int) -> list[list[str]]:
    return [
        list(exchange_ids[start : start + group_size])
        for start in range(0, len(exchange_ids), group_size)
        if exchange_ids[start : start + group_size]
    ]


def _rack_exchange_groups(g: nx.Graph) -> list[list[str]]:
    family = str(g.graph.get("topology_family", "")).upper()
    exchange_ids = _exchange_ids(g)
    if not exchange_ids:
        return []

    if family == "TORUS":
        shape = tuple(int(value) for value in g.graph.get("torus_exchange_grid_shape", ()))
        if len(shape) == 3 and int(shape[0]) * int(shape[1]) == 8:
            groups: list[list[str]] = []
            for z_index in range(int(shape[2])):
                group = sorted(
                    [
                        exchange_id
                        for exchange_id in exchange_ids
                        if (_exchange_coord(g, exchange_id) or (None, None, None))[2] == z_index
                    ],
                    key=lambda exchange_id: _exchange_coord(g, exchange_id) or (),
                )
                if group:
                    groups.append(group)
            if groups:
                return groups

        ordered = sorted(
            exchange_ids,
            key=lambda exchange_id: _exchange_coord(g, exchange_id) or (_exchange_sort_key(exchange_id),),
        )
        return _chunked_exchange_groups(ordered, 8)

    if family == "FULLMESH":
        ordered = sorted(
            exchange_ids,
            key=lambda exchange_id: _exchange_coord(g, exchange_id) or (_exchange_sort_key(exchange_id),),
        )
        return _chunked_exchange_groups(ordered, 8)

    if family == "CLOS":
        grouped: dict[int, list[str]] = {}
        for exchange_id in exchange_ids:
            probe_union = f"{exchange_id}:union0"
            local_group_id = g.nodes[probe_union].get("clos_local_group_id") if probe_union in g else None
            if local_group_id is None:
                continue
            rack_id = int(local_group_id) // 2
            grouped.setdefault(rack_id, []).append(exchange_id)
        if grouped:
            return [
                sorted(grouped[rack_id], key=_exchange_sort_key)
                for rack_id in sorted(grouped)
            ]

    ordered = sorted(exchange_ids, key=_exchange_sort_key)
    return _chunked_exchange_groups(ordered, 8)


def _source_target_rack_index(src_ssu: str, source_rack_index: int, rack_count: int) -> int:
    if rack_count <= 1:
        return source_rack_index
    offset = 1 + (
        (_exchange_sort_key(_exchange_id(src_ssu)) + _local_ssu_index(src_ssu)) % (rack_count - 1)
    )
    return (source_rack_index + offset) % rack_count


def _random_rack_targets(
    g: nx.Graph,
    src_ssu: str,
    rack_exchange_ids: Sequence[str],
    target_count: int,
    *,
    seed: int,
) -> list[str]:
    exchange_map = _exchange_ssus(g)
    candidates = sorted(
        [
            ssu_id
            for exchange_id in rack_exchange_ids
            for ssu_id in exchange_map.get(exchange_id, [])
            if ssu_id != src_ssu
        ],
        key=_ssu_sort_key,
    )
    if not candidates:
        return []
    rng = random.Random(seed)
    return sorted(
        rng.sample(candidates, k=min(len(candidates), int(target_count))),
        key=_ssu_sort_key,
    )


def _aware_rack_targets(
    g: nx.Graph,
    src_ssu: str,
    rack_exchange_ids: Sequence[str],
    target_count: int,
) -> list[str]:
    exchange_map = _exchange_ssus(g)
    ordered_exchanges = sorted(
        rack_exchange_ids,
        key=lambda exchange_id: _exchange_coord(g, exchange_id) or (_exchange_sort_key(exchange_id),),
    )
    if not ordered_exchanges:
        return []

    desired_count = min(len(ordered_exchanges), int(target_count))
    start = (_exchange_sort_key(_exchange_id(src_ssu)) + _local_ssu_index(src_ssu)) % len(ordered_exchanges)
    step = max(1, len(ordered_exchanges) // max(1, desired_count))
    chosen_exchanges: list[str] = []
    for index in [start + (step * offset) for offset in range(len(ordered_exchanges))]:
        exchange_id = ordered_exchanges[index % len(ordered_exchanges)]
        if exchange_id not in chosen_exchanges:
            chosen_exchanges.append(exchange_id)
        if len(chosen_exchanges) == desired_count:
            break

    local_index = _local_ssu_index(src_ssu)
    targets: list[str] = []
    for rank, exchange_id in enumerate(chosen_exchanges):
        ssus = exchange_map[exchange_id]
        slot = (local_index + rank) % len(ssus)
        target = ssus[slot]
        if target == src_ssu:
            target = ssus[(slot + 1) % len(ssus)]
        targets.append(target)
    return sorted(_unique_ssus(targets), key=_ssu_sort_key)


def _unique_ssus(node_ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for node_id in node_ids:
        if node_id in seen:
            continue
        seen.add(node_id)
        unique.append(node_id)
    return unique


def _home_unions_for_ssu(g: nx.Graph, ssu_id: str) -> list[str]:
    return sorted(
        [
            str(neighbor)
            for neighbor in g.neighbors(ssu_id)
            if g.nodes[neighbor].get("node_role") == "union"
        ],
        key=_union_sort_key,
    )


def _attached_dpu_for_npu(g: nx.Graph, npu_id: str) -> str:
    for neighbor in g.neighbors(npu_id):
        if g.nodes[neighbor].get("node_role") == "dpu":
            return str(neighbor)
    raise ValueError(f"NPU '{npu_id}' is not attached to a DPU")


def _union_neighbors_of_dpu(g: nx.Graph, dpu_id: str) -> list[str]:
    return sorted(
        [
            str(neighbor)
            for neighbor in g.neighbors(dpu_id)
            if g.nodes[neighbor].get("node_role") == "union"
        ],
        key=_union_sort_key,
    )


def _union_local_index(union_id: str) -> int:
    return _local_node_index(union_id)


def _npu_local_index(npu_id: str) -> int:
    return _local_node_index(npu_id)


def _logical_npu_sources(g: nx.Graph, source_count: int) -> list[tuple[str, str]]:
    npu_ids = _npu_nodes(g)
    count = _validated_rack_stripe_source_count(source_count)
    if not npu_ids:
        return []
    selected = (
        _spread_sample(npu_ids, count)
        if count <= len(npu_ids)
        else [npu_ids[index % len(npu_ids)] for index in range(count)]
    )
    return [(f"npu{index}", selected[index]) for index in range(len(selected))]


def _target_rack_index_for_union(
    source_union: str,
    source_index: int,
    rack_by_exchange: Mapping[str, int],
    rack_count: int,
) -> int:
    source_rack = int(rack_by_exchange.get(_exchange_id(source_union), 0))
    if rack_count <= 1:
        return source_rack
    offset = 1 + (
        (_exchange_sort_key(_exchange_id(source_union)) + _union_local_index(source_union) + int(source_index))
        % (rack_count - 1)
    )
    return (source_rack + offset) % rack_count


def _ordered_rack_exchanges(g: nx.Graph, rack_exchange_ids: Sequence[str]) -> list[str]:
    return sorted(
        rack_exchange_ids,
        key=lambda exchange_id: _exchange_coord(g, exchange_id) or (_exchange_sort_key(exchange_id),),
    )


def _target_ssus_in_rack(
    g: nx.Graph,
    source_union: str,
    rack_exchange_ids: Sequence[str],
    *,
    target_count: int,
    source_index: int,
) -> list[str]:
    exchange_map = _exchange_ssus(g)
    ordered_exchanges = _ordered_rack_exchanges(g, rack_exchange_ids)
    if not ordered_exchanges:
        return []

    desired_count = min(int(target_count), len(ordered_exchanges))
    start = (
        _exchange_sort_key(_exchange_id(source_union))
        + _union_local_index(source_union)
        + int(source_index)
    ) % len(ordered_exchanges)
    step = max(1, len(ordered_exchanges) // max(1, desired_count))

    chosen_exchanges: list[str] = []
    for offset in range(len(ordered_exchanges) * 2):
        exchange_id = ordered_exchanges[(start + (offset * step)) % len(ordered_exchanges)]
        if exchange_id not in chosen_exchanges:
            chosen_exchanges.append(exchange_id)
        if len(chosen_exchanges) == desired_count:
            break

    targets: list[str] = []
    for rank, exchange_id in enumerate(chosen_exchanges):
        ssus = exchange_map[exchange_id]
        slot = (int(source_index) + _union_local_index(source_union) + rank) % len(ssus)
        targets.append(ssus[slot])
    return sorted(_unique_ssus(targets), key=_ssu_sort_key)


def _pooled_target_unions(
    g: nx.Graph,
    rack_exchange_ids: Sequence[str],
    target_ssu: str,
) -> list[str]:
    pooled = [
        union_id
        for exchange_id in _ordered_rack_exchanges(g, rack_exchange_ids)
        for union_id in (f"{exchange_id}:union0", f"{exchange_id}:union1")
        if union_id in g
    ]
    return sorted(pooled, key=_union_sort_key)


def _unique_paths(paths: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    unique: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for path in paths:
        normalized = tuple(str(node_id) for node_id in path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return tuple(unique)


def _single_shortest_path(g: nx.Graph, src: str, dst: str) -> tuple[str, ...]:
    return tuple(str(node_id) for node_id in nx.shortest_path(g, src, dst))


def _data_plane_graph(g: nx.Graph) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(
        node_id
        for node_id, data in g.nodes(data=True)
        if data.get("node_role") not in {"dpu", "npu"}
    )
    for u, v, data in g.edges(data=True):
        if u not in graph or v not in graph:
            continue
        graph.add_edge(u, v, **data)
    return graph


def _union_data_plane_graph(g: nx.Graph) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(
        node_id
        for node_id, data in g.nodes(data=True)
        if data.get("node_role") not in {"dpu", "npu", "ssu"}
    )
    for u, v, data in g.edges(data=True):
        if u not in graph or v not in graph:
            continue
        graph.add_edge(u, v, **data)
    return graph


def _union_path_to_target_ssu(
    g: nx.Graph,
    union_graph: nx.Graph,
    start_union: str,
    target_ssu: str,
) -> tuple[str, ...]:
    home_unions = _home_unions_for_ssu(g, target_ssu)
    candidate_paths: list[tuple[str, ...]] = []
    for home_union in home_unions:
        try:
            backend_path = tuple(
                str(node_id) for node_id in nx.shortest_path(union_graph, start_union, home_union)
            )
        except nx.NetworkXNoPath:
            continue
        candidate_paths.append((*backend_path, str(target_ssu)))

    if not candidate_paths:
        raise nx.NetworkXNoPath(f"No Union-only path from {start_union} to home Union of {target_ssu}")

    return min(
        candidate_paths,
        key=lambda path: (len(path), _union_sort_key(path[-2])),
    )


def _path_from_npu_via_union_to_target(
    g: nx.Graph,
    union_graph: nx.Graph,
    npu_id: str,
    source_union: str,
    target_ssu: str,
) -> tuple[str, ...]:
    dpu_id = _attached_dpu_for_npu(g, npu_id)
    suffix = _union_path_to_target_ssu(g, union_graph, source_union, target_ssu)
    return (str(npu_id), str(dpu_id), str(source_union), *suffix[1:])


def _path_via_union(
    g: nx.Graph,
    union_graph: nx.Graph,
    source_union: str,
    via_union: str,
    target_ssu: str,
) -> tuple[str, ...]:
    first = tuple(str(node_id) for node_id in nx.shortest_path(union_graph, source_union, via_union))
    second = _union_path_to_target_ssu(g, union_graph, via_union, target_ssu)
    return (*first, *second[1:])


def _path_from_npu_via_two_unions_to_target(
    g: nx.Graph,
    union_graph: nx.Graph,
    npu_id: str,
    source_union: str,
    via_union: str,
    target_ssu: str,
) -> tuple[str, ...]:
    dpu_id = _attached_dpu_for_npu(g, npu_id)
    suffix = _path_via_union(g, union_graph, source_union, via_union, target_ssu)
    return (str(npu_id), str(dpu_id), *suffix)


def _paths_from_npu_to_target_via_source_unions(
    g: nx.Graph,
    union_graph: nx.Graph,
    npu_id: str,
    target_ssu: str,
) -> tuple[tuple[str, ...], ...]:
    paths = []
    for source_union in _union_neighbors_of_dpu(g, _attached_dpu_for_npu(g, npu_id)):
        try:
            paths.append(
                _path_from_npu_via_union_to_target(g, union_graph, npu_id, source_union, target_ssu)
            )
        except nx.NetworkXNoPath:
            continue
    return _unique_paths(paths)


def _paths_from_npu_to_target_via_rack_pool(
    g: nx.Graph,
    union_graph: nx.Graph,
    npu_id: str,
    rack_exchange_ids: Sequence[str],
    target_ssu: str,
) -> tuple[tuple[str, ...], ...]:
    source_unions = _union_neighbors_of_dpu(g, _attached_dpu_for_npu(g, npu_id))
    pooled_unions = _pooled_target_unions(g, rack_exchange_ids, target_ssu)
    paths = []
    for source_union in source_unions:
        for via_union in pooled_unions:
            try:
                paths.append(
                    _path_from_npu_via_two_unions_to_target(
                        g,
                        union_graph,
                        npu_id,
                        source_union,
                        via_union,
                        target_ssu,
                    )
                )
            except nx.NetworkXNoPath:
                continue
    return _unique_paths(paths)


def _local_exchange_target_ssu(g: nx.Graph, npu_id: str) -> str | None:
    exchange_id = _exchange_id(npu_id)
    exchange_map = _exchange_ssus(g)
    ssus = exchange_map.get(exchange_id, [])
    if not ssus:
        return None
    return ssus[_npu_local_index(npu_id) % len(ssus)]


def _best_source_union_for_target(
    g: nx.Graph,
    union_graph: nx.Graph,
    npu_id: str,
    target_ssu: str,
) -> tuple[str, tuple[str, ...]]:
    candidates = _union_neighbors_of_dpu(g, _attached_dpu_for_npu(g, npu_id))
    if not candidates:
        raise ValueError(f"NPU '{npu_id}' has no Union ingress candidates")

    best_union = candidates[0]
    best_path = _path_from_npu_via_union_to_target(g, union_graph, npu_id, best_union, target_ssu)
    for source_union in candidates[1:]:
        candidate_path = _path_from_npu_via_union_to_target(g, union_graph, npu_id, source_union, target_ssu)
        if len(candidate_path) < len(best_path) or (
            len(candidate_path) == len(best_path) and _union_sort_key(source_union) < _union_sort_key(best_union)
        ):
            best_union = source_union
            best_path = candidate_path
    return best_union, best_path


def _best_source_union_for_via_target(
    g: nx.Graph,
    union_graph: nx.Graph,
    npu_id: str,
    via_union: str,
    target_ssu: str,
) -> tuple[str, tuple[str, ...]]:
    candidates = _union_neighbors_of_dpu(g, _attached_dpu_for_npu(g, npu_id))
    if not candidates:
        raise ValueError(f"NPU '{npu_id}' has no Union ingress candidates")

    best_union = candidates[0]
    best_suffix = _path_via_union(g, union_graph, best_union, via_union, target_ssu)
    best_path = (str(npu_id), _attached_dpu_for_npu(g, npu_id), *best_suffix)
    for source_union in candidates[1:]:
        candidate_suffix = _path_via_union(g, union_graph, source_union, via_union, target_ssu)
        candidate_path = (str(npu_id), _attached_dpu_for_npu(g, npu_id), *candidate_suffix)
        if len(candidate_path) < len(best_path) or (
            len(candidate_path) == len(best_path) and _union_sort_key(source_union) < _union_sort_key(best_union)
        ):
            best_union = source_union
            best_path = candidate_path
    return best_union, tuple(best_path)


def _validated_rack_stripe_source_count(source_count: int) -> int:
    if type(source_count) is not int or source_count <= 0:
        raise ValueError("rack stripe source_count must be an integer > 0")
    return int(source_count)


def _validated_rack_stripe_target_count(cfg: AnalysisConfig) -> int:
    if type(cfg.rack_stripe_target_count) is not int or cfg.rack_stripe_target_count <= 0:
        raise ValueError("rack_stripe_target_count must be an integer > 0")
    return int(cfg.rack_stripe_target_count)


def _validated_npu_write_source_counts(cfg: AnalysisConfig) -> list[int]:
    counts: list[int] = []
    for value in cfg.npu_write_source_counts:
        count = _validated_rack_stripe_source_count(int(value))
        if count not in counts:
            counts.append(count)
    return counts


def _validated_npu_write_source_count(source_count: int) -> int:
    return _validated_rack_stripe_source_count(source_count)


def _npu_write_target_rack_exchange_ids(g: nx.Graph) -> list[str]:
    racks = _rack_exchange_groups(g)
    if not racks:
        return []
    return _ordered_rack_exchanges(g, racks[0])


def _npu_write_rack_exchange_ids_by_exchange(g: nx.Graph) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for rack_exchange_ids in _rack_exchange_groups(g):
        ordered = _ordered_rack_exchanges(g, rack_exchange_ids)
        for exchange_id in ordered:
            mapping[exchange_id] = ordered
    return mapping


def _npu_write_source_rack_exchange_ids(g: nx.Graph, npu_id: str) -> list[str]:
    return list(_npu_write_rack_exchange_ids_by_exchange(g).get(_exchange_id(npu_id), []))


def _npu_write_active_sources(g: nx.Graph, source_count: int) -> list[tuple[str, str]]:
    npu_ids = _npu_nodes(g)
    count = _validated_npu_write_source_count(source_count)
    if not npu_ids:
        return []

    target_rack = set(_npu_write_target_rack_exchange_ids(g))
    racks = _rack_exchange_groups(g)
    eligible = list(npu_ids)
    if len(racks) > 1 and target_rack:
        filtered = [npu_id for npu_id in npu_ids if _exchange_id(npu_id) not in target_rack]
        if len(filtered) >= count:
            eligible = filtered

    selected = (
        _spread_sample(eligible, count)
        if count <= len(eligible)
        else [eligible[index % len(eligible)] for index in range(count)]
    )
    return [(f"npu{index}", selected[index]) for index in range(len(selected))]


def _npu_write_target_set_ssus(g: nx.Graph, target_count: int = 4) -> list[str]:
    exchange_map = _exchange_ssus(g)
    target_exchanges = _npu_write_target_rack_exchange_ids(g)[: max(0, int(target_count))]
    targets: list[str] = []
    for exchange_id in target_exchanges:
        ssus = exchange_map.get(exchange_id, [])
        if ssus:
            targets.append(ssus[0])
    return sorted(targets, key=_ssu_sort_key)


def _single_ssu_hotspot_target(g: nx.Graph) -> str | None:
    targets = _npu_write_target_set_ssus(g, target_count=1)
    return targets[0] if targets else None


def _npu_write_local_rack_target_ssus(
    g: nx.Graph,
    npu_id: str,
    *,
    target_count: int = 4,
) -> list[str]:
    exchange_map = _exchange_ssus(g)
    rack_exchange_ids = _npu_write_source_rack_exchange_ids(g, npu_id)
    if not rack_exchange_ids:
        return []

    ordered_exchanges = _ordered_rack_exchanges(g, rack_exchange_ids)
    source_exchange = _exchange_id(npu_id)
    local_index = _npu_local_index(npu_id)
    desired_count = min(max(0, int(target_count)), len(ordered_exchanges))
    if desired_count <= 0:
        return []

    start = ordered_exchanges.index(source_exchange) if source_exchange in ordered_exchanges else 0
    step = max(1, len(ordered_exchanges) // max(1, desired_count))
    chosen_exchanges: list[str] = []
    for offset in range(len(ordered_exchanges) * 2):
        exchange_id = ordered_exchanges[(start + (offset * step)) % len(ordered_exchanges)]
        if exchange_id not in chosen_exchanges:
            chosen_exchanges.append(exchange_id)
        if len(chosen_exchanges) == desired_count:
            break

    targets = [
        _target_ssu(exchange_map, exchange_id, local_index)
        for exchange_id in chosen_exchanges
        if exchange_id in exchange_map
    ]
    return sorted(_unique_ssus(targets), key=_ssu_sort_key)


def _build_npu_write_local_1to1_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    total_bits = _validated_message_bits(cfg)
    reduced_graph = _union_data_plane_graph(g)
    demands: list[FlowDemand] = []
    for source_id, npu_id in _npu_write_active_sources(g, source_count):
        target_ssu = _local_exchange_target_ssu(g, npu_id)
        if target_ssu is None:
            continue
        dpu_id = _attached_dpu_for_npu(g, npu_id)
        explicit_paths = _paths_from_npu_to_target_via_source_unions(
            g,
            reduced_graph,
            npu_id,
            target_ssu,
        )
        demands.append(
            FlowDemand(
                src=dpu_id,
                dst=target_ssu,
                bits=total_bits,
                source_id=source_id,
                explicit_paths=explicit_paths,
            )
        )
    return demands


def _build_npu_write_local_1to1_pooling_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    total_bits = _validated_message_bits(cfg)
    reduced_graph = _union_data_plane_graph(g)
    demands: list[FlowDemand] = []
    for source_id, npu_id in _npu_write_active_sources(g, source_count):
        target_ssu = _local_exchange_target_ssu(g, npu_id)
        rack_exchange_ids = _npu_write_source_rack_exchange_ids(g, npu_id)
        if target_ssu is None or not rack_exchange_ids:
            continue
        explicit_paths = _paths_from_npu_to_target_via_rack_pool(
            g,
            reduced_graph,
            npu_id,
            rack_exchange_ids,
            target_ssu,
        )
        if not explicit_paths:
            continue
        demands.append(
            FlowDemand(
                src=_attached_dpu_for_npu(g, npu_id),
                dst=target_ssu,
                bits=total_bits,
                source_id=source_id,
                explicit_paths=explicit_paths,
            )
        )
    return demands


def _build_npu_write_local_1to1_sharding_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    total_bits = _validated_message_bits(cfg)
    reduced_graph = _union_data_plane_graph(g)
    demands: list[FlowDemand] = []
    for source_id, npu_id in _npu_write_active_sources(g, source_count):
        target_ssus = _npu_write_local_rack_target_ssus(g, npu_id, target_count=4)
        if not target_ssus:
            continue
        bits_per_target = total_bits / float(len(target_ssus))
        for target_ssu in target_ssus:
            explicit_paths = _paths_from_npu_to_target_via_source_unions(
                g,
                reduced_graph,
                npu_id,
                target_ssu,
            )
            demands.append(
                FlowDemand(
                    src=_attached_dpu_for_npu(g, npu_id),
                    dst=target_ssu,
                    bits=bits_per_target,
                    source_id=source_id,
                    explicit_paths=explicit_paths,
                )
            )
    return demands


def _build_npu_write_single_ssu_hotspot_direct_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    total_bits = _validated_message_bits(cfg)
    reduced_graph = _union_data_plane_graph(g)
    target_ssu = _single_ssu_hotspot_target(g)
    if target_ssu is None:
        return []

    demands: list[FlowDemand] = []
    for source_id, npu_id in _npu_write_active_sources(g, source_count):
        explicit_paths = _paths_from_npu_to_target_via_source_unions(
            g,
            reduced_graph,
            npu_id,
            target_ssu,
        )
        demands.append(
            FlowDemand(
                src=_attached_dpu_for_npu(g, npu_id),
                dst=target_ssu,
                bits=total_bits,
                source_id=source_id,
                explicit_paths=explicit_paths,
            )
        )
    return demands


def _build_npu_write_single_ssu_hotspot_pooling_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    total_bits = _validated_message_bits(cfg)
    reduced_graph = _union_data_plane_graph(g)
    target_ssu = _single_ssu_hotspot_target(g)
    rack_exchange_ids = _npu_write_target_rack_exchange_ids(g)
    if target_ssu is None or not rack_exchange_ids:
        return []

    demands: list[FlowDemand] = []
    for source_id, npu_id in _npu_write_active_sources(g, source_count):
        explicit_paths = _paths_from_npu_to_target_via_rack_pool(
            g,
            reduced_graph,
            npu_id,
            rack_exchange_ids,
            target_ssu,
        )
        if not explicit_paths:
            continue
        demands.append(
            FlowDemand(
                src=_attached_dpu_for_npu(g, npu_id),
                dst=target_ssu,
                bits=total_bits,
                source_id=source_id,
                explicit_paths=explicit_paths,
            )
        )
    return demands


def _build_npu_write_single_ssu_hotspot_sharding_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    total_bits = _validated_message_bits(cfg)
    reduced_graph = _union_data_plane_graph(g)
    target_ssus = _npu_write_target_set_ssus(g, target_count=4)
    if not target_ssus:
        return []

    bits_per_target = total_bits / float(len(target_ssus))
    demands: list[FlowDemand] = []
    for source_id, npu_id in _npu_write_active_sources(g, source_count):
        for target_ssu in target_ssus:
            explicit_paths = _paths_from_npu_to_target_via_source_unions(
                g,
                reduced_graph,
                npu_id,
                target_ssu,
            )
            demands.append(
                FlowDemand(
                    src=_attached_dpu_for_npu(g, npu_id),
                    dst=target_ssu,
                    bits=bits_per_target,
                    source_id=source_id,
                    explicit_paths=explicit_paths,
                )
            )
    return demands


def _build_npu_write_rack_target_set_direct_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    total_bits = _validated_message_bits(cfg)
    reduced_graph = _union_data_plane_graph(g)
    target_ssus = _npu_write_target_set_ssus(g, target_count=4)
    if not target_ssus:
        return []

    demands: list[FlowDemand] = []
    for source_rank, (source_id, npu_id) in enumerate(_npu_write_active_sources(g, source_count)):
        target_ssu = target_ssus[source_rank % len(target_ssus)]
        explicit_paths = _paths_from_npu_to_target_via_source_unions(
            g,
            reduced_graph,
            npu_id,
            target_ssu,
        )
        demands.append(
            FlowDemand(
                src=_attached_dpu_for_npu(g, npu_id),
                dst=target_ssu,
                bits=total_bits,
                source_id=source_id,
                explicit_paths=explicit_paths,
            )
        )
    return demands


def _build_npu_write_rack_target_set_pooling_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    total_bits = _validated_message_bits(cfg)
    reduced_graph = _union_data_plane_graph(g)
    target_ssus = _npu_write_target_set_ssus(g, target_count=4)
    rack_exchange_ids = _npu_write_target_rack_exchange_ids(g)
    if not target_ssus or not rack_exchange_ids:
        return []

    demands: list[FlowDemand] = []
    for source_rank, (source_id, npu_id) in enumerate(_npu_write_active_sources(g, source_count)):
        target_ssu = target_ssus[source_rank % len(target_ssus)]
        explicit_paths = _paths_from_npu_to_target_via_rack_pool(
            g,
            reduced_graph,
            npu_id,
            rack_exchange_ids,
            target_ssu,
        )
        if not explicit_paths:
            continue
        demands.append(
            FlowDemand(
                src=_attached_dpu_for_npu(g, npu_id),
                dst=target_ssu,
                bits=total_bits,
                source_id=source_id,
                explicit_paths=explicit_paths,
            )
        )
    return demands


def _build_npu_write_rack_target_set_sharding_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    total_bits = _validated_message_bits(cfg)
    reduced_graph = _union_data_plane_graph(g)
    target_ssus = _npu_write_target_set_ssus(g, target_count=4)
    if not target_ssus:
        return []

    bits_per_target = total_bits / float(len(target_ssus))
    demands: list[FlowDemand] = []
    for source_id, npu_id in _npu_write_active_sources(g, source_count):
        for target_ssu in target_ssus:
            explicit_paths = _paths_from_npu_to_target_via_source_unions(
                g,
                reduced_graph,
                npu_id,
                target_ssu,
            )
            demands.append(
                FlowDemand(
                    src=_attached_dpu_for_npu(g, npu_id),
                    dst=target_ssu,
                    bits=bits_per_target,
                    source_id=source_id,
                    explicit_paths=explicit_paths,
                )
            )
    return demands


def _build_rack_stripe_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
    aware: bool,
) -> list[FlowDemand]:
    total_bits_per_source = _validated_message_bits(cfg)
    target_count = _validated_rack_stripe_target_count(cfg)
    racks = _rack_exchange_groups(g)
    if not racks:
        return []

    rack_by_exchange = {
        exchange_id: rack_index
        for rack_index, exchange_ids in enumerate(racks)
        for exchange_id in exchange_ids
    }
    selected_sources = _spread_sample(
        _ssu_nodes(g),
        _validated_rack_stripe_source_count(source_count),
    )

    demands: list[FlowDemand] = []
    for source_index, src_ssu in enumerate(selected_sources):
        src_exchange = _exchange_id(src_ssu)
        source_rack_index = rack_by_exchange.get(src_exchange, 0)
        target_rack_index = _source_target_rack_index(src_ssu, source_rack_index, len(racks))
        rack_exchange_ids = racks[target_rack_index]
        if aware:
            target_ssus = _aware_rack_targets(g, src_ssu, rack_exchange_ids, target_count)
        else:
            target_ssus = _random_rack_targets(
                g,
                src_ssu,
                rack_exchange_ids,
                target_count,
                seed=(
                    int(cfg.random_seed) * 1_000_003
                    + (source_index * 97)
                    + (_exchange_sort_key(src_exchange) * 13)
                    + _local_ssu_index(src_ssu)
                ),
            )
        target_ssus = [target_ssu for target_ssu in target_ssus if target_ssu != src_ssu]
        if not target_ssus:
            continue
        bits_per_target = total_bits_per_source / float(len(target_ssus))
        for dst_ssu in target_ssus:
            demands.append(FlowDemand(src=src_ssu, dst=dst_ssu, bits=bits_per_target))

    return demands


def _unique_exchange_ids(exchange_ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for exchange_id in exchange_ids:
        if exchange_id in seen:
            continue
        seen.add(exchange_id)
        unique.append(exchange_id)
    return unique


def _rotated_pair(candidates: Sequence[str], seed: int, step: int | None = None) -> list[str]:
    unique_candidates = _unique_exchange_ids(candidates)
    if len(unique_candidates) <= 2:
        return unique_candidates[:2]

    count = len(unique_candidates)
    first_index = int(seed) % count
    second_step = max(1, int(step if step is not None else count // 2))
    indices = [first_index, (first_index + second_step) % count]
    picked: list[str] = []
    for index in [*indices, *range(count)]:
        candidate = unique_candidates[index % count]
        if candidate not in picked:
            picked.append(candidate)
        if len(picked) == 2:
            break
    return picked


def _balanced_modular_exchange_targets(
    src_ssu: str,
    candidate_exchanges: Sequence[str],
    *,
    salt: int = 0,
) -> list[str]:
    src_exchange = _exchange_id(src_ssu)
    ordered = sorted([src_exchange, *candidate_exchanges], key=_exchange_sort_key)
    ordered = _unique_exchange_ids(ordered)
    if len(ordered) < 3:
        return list(candidate_exchanges)[:2]

    src_position = ordered.index(src_exchange)
    local_index = _local_ssu_index(src_ssu)
    target_offsets = [
        1 + ((local_index + int(salt)) % (len(ordered) - 1)),
        1 + ((local_index + int(salt) + max(1, len(ordered) // 2)) % (len(ordered) - 1)),
    ]
    target_candidates = [
        ordered[(src_position + offset) % len(ordered)]
        for offset in target_offsets
    ]
    target_candidates.extend(candidate_exchanges)
    return [
        exchange_id
        for exchange_id in _unique_exchange_ids(target_candidates)
        if exchange_id != src_exchange and exchange_id in candidate_exchanges
    ][:2]


def _coord_to_exchange(g: nx.Graph) -> dict[tuple[int, ...], str]:
    mapping: dict[tuple[int, ...], str] = {}
    for ssu_id in _ssu_nodes(g):
        coord = g.nodes[ssu_id].get("exchange_grid_coord")
        if coord is None:
            continue
        mapping.setdefault(tuple(int(value) for value in coord), _exchange_id(ssu_id))
    return mapping


def _torus_replica_exchange_targets(
    g: nx.Graph,
    src_ssu: str,
    candidate_exchanges: Sequence[str],
) -> list[str]:
    coord = g.nodes[src_ssu].get("exchange_grid_coord")
    shape = g.graph.get("torus_exchange_grid_shape")
    if coord is None or shape is None:
        return []

    source_coord = tuple(int(value) for value in coord)
    torus_shape = tuple(int(value) for value in shape)
    coord_map = _coord_to_exchange(g)
    neighbors: list[str] = []
    for axis, axis_size in enumerate(torus_shape):
        if int(axis_size) <= 1:
            continue
        for delta in (1, -1):
            neighbor_coord = list(source_coord)
            neighbor_coord[axis] = (int(neighbor_coord[axis]) + delta) % int(axis_size)
            exchange_id = coord_map.get(tuple(neighbor_coord))
            if exchange_id is not None and exchange_id in candidate_exchanges:
                neighbors.append(exchange_id)

    # Keep the direction schedule translation-invariant across the torus.
    # Adding the coordinate to this seed can balance visual direction choices,
    # but it also creates uneven incoming replica pressure in 3D torus.
    seed = _local_ssu_index(src_ssu)
    return _rotated_pair(neighbors, seed=seed, step=max(1, len(_unique_exchange_ids(neighbors)) // 2))


def _fullmesh_replica_exchange_targets(
    g: nx.Graph,
    src_ssu: str,
    candidate_exchanges: Sequence[str],
) -> list[str]:
    coord = g.nodes[src_ssu].get("exchange_grid_coord")
    shape = g.graph.get("fullmesh_exchange_grid_shape")
    if coord is None or shape is None:
        return []

    row, col = (int(value) for value in coord)
    rows, cols = (int(value) for value in shape)
    coord_map = _coord_to_exchange(g)
    local_index = _local_ssu_index(src_ssu)
    targets: list[str] = []
    if cols > 1:
        col_offset = 1 + ((local_index + row) % (cols - 1))
        exchange_id = coord_map.get((row, (col + col_offset) % cols))
        if exchange_id is not None:
            targets.append(exchange_id)
    if rows > 1:
        row_offset = 1 + (((local_index // max(1, cols - 1)) + col) % (rows - 1))
        exchange_id = coord_map.get(((row + row_offset) % rows, col))
        if exchange_id is not None:
            targets.append(exchange_id)

    return [
        exchange_id
        for exchange_id in _unique_exchange_ids(targets)
        if exchange_id in candidate_exchanges
    ][:2]


def _df_exchange_by_server_slot(g: nx.Graph) -> dict[tuple[int, int], str]:
    mapping: dict[tuple[int, int], str] = {}
    for ssu_id in _ssu_nodes(g):
        node_data = g.nodes[ssu_id]
        server_id = node_data.get("server_id")
        slot = node_data.get("df_group_local_index")
        if server_id is None or slot is None:
            continue
        mapping.setdefault((int(server_id), int(slot)), _exchange_id(ssu_id))
    return mapping


def _df_replica_exchange_targets(
    g: nx.Graph,
    src_ssu: str,
    candidate_exchanges: Sequence[str],
) -> list[str]:
    node_data = g.nodes[src_ssu]
    server_id = node_data.get("server_id")
    source_slot = node_data.get("df_group_local_index")
    if server_id is None or source_slot is None:
        return []

    server = int(server_id)
    slot = int(source_slot)
    local_index = _local_ssu_index(src_ssu)
    exchange_nodes_per_server = int(g.graph.get("df_exchange_nodes_per_server", 0))
    server_count = int(g.graph.get("df_server_count", 0))
    global_links_per_union = int(g.graph.get("df_external_servers_per_union", 0))
    unions_per_server = int(g.graph.get("df_unions_per_server", exchange_nodes_per_server))
    global_pattern = str(g.graph.get("df_global_pattern", "contiguous"))
    server_slot_map = _df_exchange_by_server_slot(g)
    targets: list[str] = []

    if exchange_nodes_per_server > 1:
        local_offset = 1 + (local_index % (exchange_nodes_per_server - 1))
        local_target = server_slot_map.get(
            (server, (slot + local_offset) % exchange_nodes_per_server)
        )
        if local_target is not None:
            targets.append(local_target)

    if server_count > 1 and global_links_per_union > 0:
        global_choice = (local_index + server + slot) % global_links_per_union
        if global_pattern == "interleaved":
            relative_server_offset = slot + 1 + (global_choice * unions_per_server)
        else:
            relative_server_offset = (slot * global_links_per_union) + global_choice + 1
        target_server = (server + relative_server_offset) % server_count
        target_slot = unions_per_server - 1 - slot
        global_target = server_slot_map.get((target_server, target_slot))
        if global_target is not None:
            targets.append(global_target)

    return [
        exchange_id
        for exchange_id in _unique_exchange_ids(targets)
        if exchange_id in candidate_exchanges
    ][:2]


def _topology_aware_exchange_targets(
    g: nx.Graph,
    src_ssu: str,
    candidate_exchanges: Sequence[str],
) -> list[str]:
    family = str(g.graph.get("topology_family", "")).upper()
    if family == "TORUS":
        targets = _torus_replica_exchange_targets(g, src_ssu, candidate_exchanges)
    elif family == "FULLMESH":
        targets = _fullmesh_replica_exchange_targets(g, src_ssu, candidate_exchanges)
    elif family == "DF":
        targets = _df_replica_exchange_targets(g, src_ssu, candidate_exchanges)
    else:
        targets = []

    if len(targets) >= 2:
        return targets[:2]

    fallback = _balanced_modular_exchange_targets(
        src_ssu,
        candidate_exchanges,
        salt=_exchange_sort_key(_exchange_id(src_ssu)),
    )
    return _unique_exchange_ids([*targets, *fallback])[:2]


def build_replica3_random_demands(g: nx.Graph, cfg: AnalysisConfig) -> list[FlowDemand]:
    """Build the random baseline for 3-replica placement.

    Every SSU sends one message to two additional SSUs. The source plus two
    destinations are constrained to live in three distinct exchange nodes.
    """

    bits = _validated_message_bits(cfg)
    exchange_map = _exchange_ssus(g)
    component_exchange_map = _component_exchanges_by_exchange(g)
    rng = random.Random(cfg.random_seed)
    demands: list[FlowDemand] = []

    for src in _ssu_nodes(g):
        src_exchange = _exchange_id(src)
        candidate_exchanges = [
            exchange_id
            for exchange_id in component_exchange_map.get(src_exchange, [])
            if exchange_id != src_exchange
        ]
        if len(candidate_exchanges) < 2:
            continue
        for dst_exchange in rng.sample(candidate_exchanges, k=2):
            demands.append(
                FlowDemand(
                    src=src,
                    dst=rng.choice(exchange_map[dst_exchange]),
                    bits=bits,
                )
            )

    return demands


def build_replica3_topology_aware_demands(g: nx.Graph, cfg: AnalysisConfig) -> list[FlowDemand]:
    """Build a deterministic topology-aware 3-replica placement workload.

    Torus and FullMesh choose short-hop neighbor exchange nodes while rotating
    directions by coordinate and SSU index. Clos uses balanced modular offsets
    because all exchange pairs have the same Clos hop shape. Dragon-Fly sends
    one replica inside the local group and one over a direct global neighbor.
    """

    bits = _validated_message_bits(cfg)
    exchange_map = _exchange_ssus(g)
    component_exchange_map = _component_exchanges_by_exchange(g)
    demands: list[FlowDemand] = []

    for src in _ssu_nodes(g):
        src_exchange = _exchange_id(src)
        candidate_exchanges = [
            exchange_id
            for exchange_id in component_exchange_map.get(src_exchange, [])
            if exchange_id != src_exchange
        ]
        if len(candidate_exchanges) < 2:
            continue

        target_exchanges = _topology_aware_exchange_targets(g, src, candidate_exchanges)
        if len(target_exchanges) < 2:
            continue
        local_index = _local_ssu_index(src)
        for dst_exchange in target_exchanges[:2]:
            demands.append(
                FlowDemand(
                    src=src,
                    dst=_target_ssu(exchange_map, dst_exchange, local_index),
                    bits=bits,
                )
            )

    return demands


def build_rack_stripe_random_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    """Build a rack-local striped write workload with random target SSUs.

    Each active source writes one logical object whose total size is
    `message_size_mb`, then evenly stripes that object across multiple SSUs
    inside one target rack.
    """

    return _build_rack_stripe_demands(
        g,
        cfg,
        source_count=source_count,
        aware=False,
    )


def build_rack_stripe_topology_aware_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    """Build a rack-local striped write workload with balanced rack targets."""

    return _build_rack_stripe_demands(
        g,
        cfg,
        source_count=source_count,
        aware=True,
    )


def build_npu_write_single_direct_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    return _build_npu_write_single_ssu_hotspot_direct_demands(g, cfg, source_count=source_count)


def build_npu_write_single_pooled_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    return _build_npu_write_single_ssu_hotspot_pooling_demands(g, cfg, source_count=source_count)


def build_npu_write_four_target_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    return _build_npu_write_single_ssu_hotspot_sharding_demands(g, cfg, source_count=source_count)


def build_npu_write_local_1to1_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    return _build_npu_write_local_1to1_demands(g, cfg, source_count=source_count)


def build_npu_write_local_1to1_pooling_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    return _build_npu_write_local_1to1_pooling_demands(g, cfg, source_count=source_count)


def build_npu_write_local_1to1_sharding_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    return _build_npu_write_local_1to1_sharding_demands(g, cfg, source_count=source_count)


def build_npu_write_single_ssu_hotspot_direct_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    return _build_npu_write_single_ssu_hotspot_direct_demands(g, cfg, source_count=source_count)


def build_npu_write_single_ssu_hotspot_pooling_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    return _build_npu_write_single_ssu_hotspot_pooling_demands(g, cfg, source_count=source_count)


def build_npu_write_single_ssu_hotspot_sharding_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    return _build_npu_write_single_ssu_hotspot_sharding_demands(g, cfg, source_count=source_count)


def build_npu_write_rack_target_set_direct_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    return _build_npu_write_rack_target_set_direct_demands(g, cfg, source_count=source_count)


def build_npu_write_rack_target_set_pooling_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    return _build_npu_write_rack_target_set_pooling_demands(g, cfg, source_count=source_count)


def build_npu_write_rack_target_set_sharding_demands(
    g: nx.Graph,
    cfg: AnalysisConfig,
    *,
    source_count: int,
) -> list[FlowDemand]:
    return _build_npu_write_rack_target_set_sharding_demands(g, cfg, source_count=source_count)
