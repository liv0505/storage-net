from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AnalysisConfig:
    """Configuration for topology modeling and simulation."""

    topology_names: list[str] = field(
        default_factory=lambda: [
            "2D-FullMesh",
            "2D-FullMesh-2x4",
            "2D-Torus",
            "2D-Torus-BestTwist",
            "3D-Torus",
            "3D-Torus-BestTwist",
            "3D-Torus-2x4x3",
            "3D-Torus-2x4x3-BestTwist",
            "3D-Torus-2x4x2",
            "3D-Torus-2x4x2-BestTwist",
            "3D-Torus-2x4x1",
            "3D-Torus-2x4x1-BestTwist",
            "Clos",
            "Clos-64",
            "Clos-128",
            "Clos-192",
            "Clos-256",
            "Clos-4P-FullMesh",
            "Clos-4P-Ring",
            "DF",
            "DF-3Local-2Global",
            "DF-3Local-1Global",
            "SparseMesh-Local",
            "SparseMesh-Global",
        ]
    )
    routing_mode: str = "SHORTEST_PATH"
    sparse_active_ratio: float = 0.25
    sparse_target_count: int = 2
    port_balanced_max_detour_hops: int = 1
    clos_uplinks_per_exchange_node: int = 4
    df_unions_per_server: int = 4
    df_external_servers_per_union: int = 3
    custom_traffic_file: str | None = None
    custom_traffic_name: str = "Custom M-to-N"
    enable_rack_stripe_workloads: bool = False
    rack_stripe_source_counts: tuple[int, ...] = (8, 64)
    rack_stripe_target_count: int = 4
    enable_npu_write_workloads: bool = False
    npu_write_source_counts: tuple[int, ...] = (64,)

    link_bandwidth_gbps: float = 100.0
    hop_latency_us: float = 2.0
    switch_latency_us: float = 0.5
    message_size_mb: float = 4.0

    host_cost: float = 300.0
    switch_cost: float = 1200.0
    link_cost: float = 200.0

    collective_startup_us: float = 8.0
    traffic_samples: int = 600
    simulation_window_s: float = 0.02
    random_seed: int = 42

    output_dir: Path = Path("outputs")
