from __future__ import annotations


_WORKLOAD_DISPLAY_NAMES = {
    "A2A": "A2A",
    "Sparse 1-to-N": "Sparse M-to-N",
}

_TOPOLOGY_DISPLAY_NAMES = {
    "DF": "Dragon-Fly",
}


def display_workload_name(workload_name: str) -> str:
    return _WORKLOAD_DISPLAY_NAMES.get(workload_name, workload_name)


def display_topology_name(topology_name: str) -> str:
    return _TOPOLOGY_DISPLAY_NAMES.get(topology_name, topology_name)
