from __future__ import annotations

import argparse
import math
from pathlib import Path

from topo_sim.config import AnalysisConfig
from topo_sim.pipeline import run_full_analysis
from topo_sim.topologies import available_topologies


_SUPPORTED_ROUTING_MODES = [
    "DOR",
    "SHORTEST_PATH",
    "FULL_PATH",
    "ECMP",
    "MIN_HOPS",
    "PORT_BALANCED",
]


def _sparse_ratio_arg(value: str) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return ratio


def _positive_float_arg(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _non_negative_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _topologies_arg(value: str, valid_names: set[str]) -> str:
    names = [x.strip() for x in value.split(",") if x.strip()]
    if not names:
        raise argparse.ArgumentTypeError("must include at least one topology")
    invalid = [name for name in names if name not in valid_names]
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown topology name(s): {', '.join(invalid)}")
    return value


def parse_args() -> argparse.Namespace:
    base_cfg = AnalysisConfig()
    valid_topologies = set(available_topologies())

    def _parse_topologies(value: str) -> str:
        return _topologies_arg(value, valid_topologies)

    parser = argparse.ArgumentParser(
        description="Network topology modeling and simulation toolkit"
    )
    parser.add_argument(
        "--topologies",
        type=_parse_topologies,
        default=",".join(base_cfg.topology_names),
        help="Comma-separated topology names. Example: 2D-FullMesh,2D-Torus,3D-Torus,Clos,DF",
    )
    parser.add_argument(
        "--routing-mode",
        type=str,
        default=base_cfg.routing_mode,
        choices=_SUPPORTED_ROUTING_MODES,
    )
    parser.add_argument(
        "--sparse-active-ratio",
        type=_sparse_ratio_arg,
        default=base_cfg.sparse_active_ratio,
    )
    parser.add_argument(
        "--sparse-target-count",
        type=_non_negative_int_arg,
        default=base_cfg.sparse_target_count,
    )
    parser.add_argument(
        "--port-balanced-max-detour-hops",
        type=_non_negative_int_arg,
        default=base_cfg.port_balanced_max_detour_hops,
    )
    parser.add_argument(
        "--clos-uplinks-per-exchange-node",
        type=_positive_int_arg,
        default=base_cfg.clos_uplinks_per_exchange_node,
    )
    parser.add_argument(
        "--df-unions-per-server",
        type=_positive_int_arg,
        default=base_cfg.df_unions_per_server,
    )
    parser.add_argument(
        "--df-external-servers-per-union",
        type=_positive_int_arg,
        default=base_cfg.df_external_servers_per_union,
    )
    parser.add_argument(
        "--message-size-mb", type=_positive_float_arg, default=base_cfg.message_size_mb
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=_positive_float_arg,
        default=base_cfg.link_bandwidth_gbps,
    )
    parser.add_argument(
        "--traffic-samples", type=_positive_int_arg, default=base_cfg.traffic_samples
    )
    parser.add_argument("--output-dir", type=str, default=str(base_cfg.output_dir))
    parser.add_argument("--seed", type=int, default=base_cfg.random_seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topology_list = [x.strip() for x in args.topologies.split(",") if x.strip()]

    cfg = AnalysisConfig(
        routing_mode=args.routing_mode,
        sparse_active_ratio=args.sparse_active_ratio,
        sparse_target_count=args.sparse_target_count,
        port_balanced_max_detour_hops=args.port_balanced_max_detour_hops,
        clos_uplinks_per_exchange_node=args.clos_uplinks_per_exchange_node,
        df_unions_per_server=args.df_unions_per_server,
        df_external_servers_per_union=args.df_external_servers_per_union,
        link_bandwidth_gbps=args.bandwidth_gbps,
        message_size_mb=args.message_size_mb,
        traffic_samples=args.traffic_samples,
        random_seed=args.seed,
        output_dir=Path(args.output_dir),
    )

    paths = run_full_analysis(cfg, topology_list)
    print("Analysis complete.")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
