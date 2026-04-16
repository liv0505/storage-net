from __future__ import annotations


_WORKLOAD_DISPLAY_NAMES = {
    "A2A": "A2A",
    "Sparse 1-to-N": "Sparse M-to-N",
}

_TOPOLOGY_DISPLAY_NAMES = {
    "2D-FullMesh-2x4": "2D-FullMesh 2x4",
    "2D-Torus-BestTwist": "2D-Torus Best Twist",
    "3D-Torus-BestTwist": "3D-Torus Best Twist",
    "3D-Torus-2x4x3": "3D-Torus 2x4x3",
    "3D-Torus-2x4x3-BestTwist": "3D-Torus 2x4x3 Best Twist",
    "3D-Torus-2x4x2": "3D-Torus 2x4x2",
    "3D-Torus-2x4x2-BestTwist": "3D-Torus 2x4x2 Best Twist",
    "3D-Torus-2x4x1": "3D-Torus 2x4x1",
    "3D-Torus-2x4x1-BestTwist": "3D-Torus 2x4x1 Best Twist",
    "DF": "Dragon-Fly",
    "DF-Shuffled": "Dragon-Fly Shuffled",
    "DF-ScaleUp": "Dragon-Fly Scale-Up",
    "DF-2P-Double-4Global": "Dragon-Fly 2P Double + 4 Global",
    "DF-2P-Triple-3Global": "Dragon-Fly 2P Triple + 3 Global",
    "DF-2P-Double-Bridge-3Global": "Dragon-Fly 2P Double + Bridge + 3 Global",
    "SparseMesh-Local": "SparseMesh S=5;N=2",
    "SparseMesh-Global": "SparseMesh S=3;N=4",
}


def display_workload_name(workload_name: str) -> str:
    return _WORKLOAD_DISPLAY_NAMES.get(workload_name, workload_name)


def display_topology_name(topology_name: str) -> str:
    return _TOPOLOGY_DISPLAY_NAMES.get(topology_name, topology_name)
