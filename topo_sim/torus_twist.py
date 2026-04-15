from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from .config import AnalysisConfig
from .metrics import evaluate_workload
from .topologies import build_twisted_torus, torus_shape
from .traffic import build_a2a_demands


_SUPPORTED_TORUS_TOPOLOGIES = (
    "2D-Torus",
    "3D-Torus",
    "3D-Torus-2x4x2",
    "3D-Torus-2x4x1",
)
_SEARCH_RESULT_HEADERS = [
    "rank",
    "topology",
    "candidate_label",
    "wrap_offsets",
    "is_baseline",
    "a2a_completion_time_s",
    "a2a_per_ssu_throughput_gbps",
    "a2a_average_hops",
    "a2a_max_link_utilization",
    "a2a_link_utilization_cv",
]


@dataclass(frozen=True, slots=True)
class TorusTwistSpec:
    topology_name: str
    shape: tuple[int, ...]
    wrap_offsets_by_axis: tuple[tuple[int, ...], ...]
    label: str
    is_baseline: bool


def _canonical_torus_topology_name(topology_name: str) -> str:
    normalized = str(topology_name).strip().lower()
    for candidate in _SUPPORTED_TORUS_TOPOLOGIES:
        if candidate.lower() == normalized:
            return candidate
    valid = ", ".join(_SUPPORTED_TORUS_TOPOLOGIES)
    raise ValueError(f"Unsupported torus topology '{topology_name}'. Valid: {valid}")


def _half_shift_choices(size: int) -> tuple[int, ...]:
    normalized_size = int(size)
    if normalized_size <= 1:
        return (0,)
    return tuple(sorted({0, normalized_size // 2}))


def _axis_wrap_vectors(shape: tuple[int, ...], axis: int) -> list[tuple[int, ...]]:
    if int(shape[axis]) <= 1:
        return [tuple(0 for _ in shape)]

    per_dim_choices: list[tuple[int, ...]] = []
    for dim, size in enumerate(shape):
        if dim == axis:
            per_dim_choices.append((0,))
            continue
        per_dim_choices.append(_half_shift_choices(int(size)))
    return [tuple(int(value) for value in values) for values in product(*per_dim_choices)]


def _wrap_offsets_label(wrap_offsets_by_axis: tuple[tuple[int, ...], ...]) -> str:
    if all(all(int(offset) == 0 for offset in axis_offsets) for axis_offsets in wrap_offsets_by_axis):
        return "baseline"
    return " | ".join(
        f"axis{axis}={list(int(offset) for offset in axis_offsets)}"
        for axis, axis_offsets in enumerate(wrap_offsets_by_axis)
    )


def generate_google_torus_twist_candidates(topology_name: str) -> list[TorusTwistSpec]:
    canonical_name = _canonical_torus_topology_name(topology_name)
    shape = torus_shape(canonical_name)
    per_axis_candidates = [
        _axis_wrap_vectors(shape, axis)
        for axis in range(len(shape))
    ]

    candidates: list[TorusTwistSpec] = []
    for axis_bundle in product(*per_axis_candidates):
        wrap_offsets_by_axis = tuple(
            tuple(int(value) for value in axis_offsets)
            for axis_offsets in axis_bundle
        )
        is_baseline = all(
            all(int(offset) == 0 for offset in axis_offsets)
            for axis_offsets in wrap_offsets_by_axis
        )
        candidates.append(
            TorusTwistSpec(
                topology_name=canonical_name,
                shape=shape,
                wrap_offsets_by_axis=wrap_offsets_by_axis,
                label=_wrap_offsets_label(wrap_offsets_by_axis),
                is_baseline=is_baseline,
            )
        )

    candidates.sort(
        key=lambda spec: (
            not spec.is_baseline,
            spec.wrap_offsets_by_axis,
        )
    )
    return candidates


def build_torus_twist_graph(
    cfg: AnalysisConfig,
    spec: TorusTwistSpec,
) -> nx.Graph:
    g = build_twisted_torus(
        cfg,
        spec.topology_name,
        wrap_offsets_by_axis=spec.wrap_offsets_by_axis,
    )
    g.graph["torus_twist_label"] = spec.label
    g.graph["torus_twist_spec"] = spec.wrap_offsets_by_axis
    return g


def evaluate_torus_twist_candidate(
    cfg: AnalysisConfig,
    spec: TorusTwistSpec,
) -> dict[str, Any]:
    g = build_torus_twist_graph(cfg, spec)
    workload = evaluate_workload(
        g,
        build_a2a_demands(g, cfg),
        routing_mode="SHORTEST_PATH",
        cfg=cfg,
    )
    return {
        "topology": spec.topology_name,
        "candidate_label": spec.label,
        "wrap_offsets": json.dumps(spec.wrap_offsets_by_axis),
        "is_baseline": spec.is_baseline,
        "a2a_completion_time_s": float(workload["completion_time_s"]),
        "a2a_per_ssu_throughput_gbps": float(workload["per_ssu_throughput_gbps"]),
        "a2a_average_hops": float(workload["average_hops"]),
        "a2a_max_link_utilization": float(workload["max_link_utilization"]),
        "a2a_link_utilization_cv": float(workload["link_utilization_cv"]),
    }


def _search_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["a2a_completion_time_s"]),
        float(row["a2a_max_link_utilization"]),
        float(row["a2a_link_utilization_cv"]),
        float(row["a2a_average_hops"]),
        str(row["candidate_label"]),
    )


def search_torus_twists(
    cfg: AnalysisConfig,
    topology_name: str,
) -> list[dict[str, Any]]:
    rows = [
        evaluate_torus_twist_candidate(cfg, spec)
        for spec in generate_google_torus_twist_candidates(topology_name)
    ]
    rows.sort(key=_search_sort_key)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def write_torus_twist_search_csv(
    output_path: str | Path,
    rows: Iterable[dict[str, Any]],
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SEARCH_RESULT_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def find_best_torus_twist(
    cfg: AnalysisConfig,
    topology_name: str,
) -> dict[str, Any]:
    rows = search_torus_twists(cfg, topology_name)
    if not rows:
        raise ValueError(f"No twist candidates generated for {topology_name}")
    return rows[0]


def _positive_float_arg(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _topology_arg(value: str) -> str:
    return _canonical_torus_topology_name(value)


def parse_args() -> argparse.Namespace:
    base_cfg = AnalysisConfig()
    parser = argparse.ArgumentParser(
        description="Search Google-style torus twist candidates with A2A shortest-path scoring."
    )
    parser.add_argument(
        "--topology",
        action="append",
        dest="topologies",
        type=_topology_arg,
        help="Repeat to search multiple torus topologies. Defaults to both 2D-Torus and 3D-Torus.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_torus_twist_search",
        help="Directory for per-topology CSV outputs.",
    )
    parser.add_argument(
        "--message-size-mb",
        type=_positive_float_arg,
        default=base_cfg.message_size_mb,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=base_cfg.random_seed,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_topologies = args.topologies or list(_SUPPORTED_TORUS_TOPOLOGIES)
    output_dir = Path(args.output_dir)
    cfg = AnalysisConfig(
        message_size_mb=args.message_size_mb,
        random_seed=args.seed,
    )

    for topology_name in selected_topologies:
        rows = search_torus_twists(cfg, topology_name)
        output_path = output_dir / f"{topology_name.lower().replace('-', '_')}_twist_search.csv"
        write_torus_twist_search_csv(output_path, rows)
        best = rows[0]
        print(
            f"{topology_name}: best={best['candidate_label']} "
            f"completion={best['a2a_completion_time_s']:.6f}s "
            f"throughput={best['a2a_per_ssu_throughput_gbps']:.2f}Gbps "
            f"cv={best['a2a_link_utilization_cv']:.6f}"
        )
        print(f"saved_csv={output_path}")


if __name__ == "__main__":
    main()
