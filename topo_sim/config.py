from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AnalysisConfig:
    """Configuration for topology modeling and simulation."""

    topology_names: list[str] = field(
        default_factory=lambda: ["2D-FullMesh", "2D-Torus", "3D-Torus", "Clos", "DF"]
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
