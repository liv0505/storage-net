## Torus Topology Update Notes

### 2D-Torus

- Previous version: two independent `4x4` planes, for a total of `32 Union + 128 SSU`.
- Current version: two independent `2x4` planes, for a total of `16 Union + 64 SSU`.

This changes the topology size itself. Total bisection bandwidth drops from `6.4T` to `3.2T`, while per-SSU bisection bandwidth stays at `100G` because the SSU count also halves. For A2A traffic, completion time drops from `20.48 ms` to `10.24 ms`, which is close to a 2x reduction because each source now communicates with roughly half as many destinations. Per-SSU A2A throughput stays almost unchanged (`198.44 -> 196.88 Gbps`) because both the total offered volume per source and the completion time shrink together. The small throughput drop comes from the new `2x4` shape being less symmetric than `4x4`, which makes backend load distribution less even.

### 3D-Torus

- Previous version: one single-plane `4x4x4` torus, for a total of `64 Union + 256 SSU`.
- Current version: two independent `2x4x4` planes, still totaling `64 Union + 256 SSU`.

Here the total machine scale, total backend port budget, and total bisection bandwidth stay the same. As a result, the main A2A performance numbers stay essentially unchanged: A2A completion time remains `40.96 ms`, and per-SSU A2A throughput remains about `199.22 Gbps`. The main differences are in path organization and link-level accounting. In particular, the length-2 torus dimension is now treated explicitly as two physical `400G` links, so utilization-distribution metrics such as Link Utilization CV better match the physical implementation even when the global A2A bottleneck stays the same.
