# 64 NPU 写入热点实验规格

## 目的

这组实验只保留 64 个活跃逻辑 NPU 的版本，用来对比以下三类能力：

1. 近端写入的基线能力
2. 单热点目标下，单纯放开入口路径是否有收益
3. 小目标集合下，把数据拆分到多个 SSU 是否有收益


## 实验集合

每个拓扑统一跑 10 个 workload：

- `A2A`
- `64 NPUs | Local 1:1 | Direct`
- `64 NPUs | Local 1:1 | Rack Pooling`
- `64 NPUs | Local 1:1 | Rack Sharding`
- `64 NPUs | Single-SSU Hotspot | Direct`
- `64 NPUs | Single-SSU Hotspot | Rack Pooling`
- `64 NPUs | Single-SSU Hotspot | Rack Sharding`
- `64 NPUs | Rack Target-Set 4 | Direct`
- `64 NPUs | Rack Target-Set 4 | Rack Pooling`
- `64 NPUs | Rack Target-Set 4 | Rack Sharding`


## 公共规则

- 活跃源数固定为 `64`
- 每个活跃源对应 `1` 个逻辑 NPU
- 每个 NPU 的总写入量固定为 `M = message_size_mb`
- 路由统一使用 `Shortest Path`
- `DPU -> union0 / union1` 永远严格 `50% / 50%` 分流
- `Local 1:1`、`Single-SSU Hotspot`、`Rack Target-Set 4` 复用同一批 active NPUs
- `Rack Pooling` 只改变路径，不改变目标数量
- `Rack Sharding` 只改变目标数量，不叠加 pooling
- 对所有显式路径，`SSU` 只允许出现在最后一跳，不作为中转节点
- 也就是说，pooling 的路径池化只发生在 `Union / backend` 层，最后统一从目标 SSU 的 home Union 下沉到目标 SSU


## rack 定义

沿用当前代码里的 rack 划分逻辑：

- `2D-FullMesh`：每 8 个 exchange node 为一个 rack
- `2D-Torus`：每 8 个 exchange node 为一个 rack
- `3D-Torus`：每个 z-slice 为一个 rack，即 `2 x 4 = 8` 个 exchange node
- `Clos-4P-FullMesh`：每 8 个 exchange node 为一个 rack

统一定义：

- `target_rack = racks[0]`
- 热点类实验优先从 `非 target_rack` 中选 source NPU
- 如果拓扑只有一个 rack，则直接从这个 rack 内选


## source 选择

对每个 topology：

1. 列出所有 NPU，按 `(exchange_id, npu_local_index)` 排序
2. 若有多个 rack，则先排除 `target_rack` 内的 NPU
3. 用当前已有的均匀 spread 规则取 `64` 个 active NPUs
4. `Local 1:1` 复用这同一批 active NPUs


## Workload 0：Local 1:1

名字：

- `64 NPUs | Local 1:1 | Direct`
- `64 NPUs | Local 1:1 | Rack Pooling`
- `64 NPUs | Local 1:1 | Rack Sharding`

### Direct

- 每个 active NPU 只写本地 exchange group 内的 1 个 SSU
- 目标规则：
  - `target_ssu = source_exchange.ssu[npu_local_index mod 8]`
- 每个 NPU 使用 2 条显式路径
- 路径形式为：
  - `NPU -> DPU -> source_union -> target_ssu`
- 其中 `source_union` 取 source DPU 直连的两个 Union，各承担 `M/2`
- 如果 source Union 不是目标 SSU 的 home Union，则先在 Union/backend 图上走最短路到目标 SSU 的某个 home Union，再下沉到目标 SSU
- 两条路径均分流量，各 `M/2`

### Rack Pooling

- 目标 SSU 与 `Direct` 完全相同
- 允许通过 source 所在 rack 的全部 Union 入口汇聚到这个本地目标 SSU
- 路径形式为：
  - `NPU -> DPU -> source_union -> via_union -> target_home_union -> target_ssu`
- 其中：
  - `source_union` 仍是 source DPU 直连的两个 Union
  - `via_union` 从 source 所在 rack 的 Union 池中选择
  - `target_home_union` 是目标 SSU 直连的两个 home Union 之一
- 整条路径在到达 `target_ssu` 之前，只允许经过 Union 和 backend，不允许把目标 SSU 当作中转点
- 也就是只改入口路径，不改目标

### Rack Sharding

- 不再只写 1 个本地 SSU
- 每个 NPU 把总数据量拆成 `4` 份
- 目标是 source 所在 rack 内的 `4` 个 SSU
- 4 个目标保持同一个本地 SSU 槽位，只是在 rack 内分布到 4 个不同 exchange
- 每个目标写入 `M/4`
- 每个目标仍只走 source 侧两个本地 Union

意义：

- 这是近端写入基线


## Workload 1：Single-SSU Hotspot

名字：

- `64 NPUs | Single-SSU Hotspot | Direct`
- `64 NPUs | Single-SSU Hotspot | Rack Pooling`
- `64 NPUs | Single-SSU Hotspot | Rack Sharding`

固定目标：

- `E0 = target_rack` 中排序后的第一个 exchange
- `T = E0:ssu0`

### Direct

- 每个 NPU 的全部 `M` 都写到 `T`
- 只允许从 source DPU 的 2 个本地 Union 出发
- 路径仍是“source Union 最短到目标 SSU 的 home Union，再下沉到目标 SSU”

### Rack Pooling

- 每个 NPU 的全部 `M` 仍写到 `T`
- 允许通过 `target_rack` 的全部 `16` 个 Union 入口汇聚到 `T`
- 路径集合来自：
  - `source_union ∈ {union0, union1}`
  - `via_union ∈ target_rack` 的全部 `16` 个 Union
- 但实际可用路径只保留 Union plane 上可达、且去重后的那些路径
- 同样要求目标 `SSU` 只出现在最后一跳

### Rack Sharding

- 不再只打单个 `T`
- 定义：
  - `T4 = {E0:ssu0, E1:ssu0, E2:ssu0, E3:ssu0}`
  - `E0..E3` 为 `target_rack` 中排序后的前 `4` 个 exchange
- 每个 NPU 把总数据量拆成 `4` 份
- 每个目标写入 `M/4`
- 每个目标仍只走 source 侧两个本地 Union

意义：

- 这个实验专门看“单目标热点”下，入口池化和目标拆分各自能带来什么收益


## Workload 2：Rack Target-Set 4

名字：

- `64 NPUs | Rack Target-Set 4 | Direct`
- `64 NPUs | Rack Target-Set 4 | Rack Pooling`
- `64 NPUs | Rack Target-Set 4 | Rack Sharding`

固定目标集合：

- `A4 = {E0:ssu0, E1:ssu0, E2:ssu0, E3:ssu0}`
- `E0..E3` 仍为 `target_rack` 中排序后的前 `4` 个 exchange

### Direct

- 每个 NPU 只写 `A4` 中的 `1` 个目标
- 分配规则：
  - `target = A4[source_rank mod 4]`
- 每个写入仍是 source 侧 2 条 Union 路径，各 `M/2`
- 路径口径与前面一致：先走 Union/backend 最短路到目标 SSU 的 home Union，再下沉到目标 SSU

### Rack Pooling

- 每个 NPU 还是只写被分配到的那 `1` 个目标
- 目标不变，只把入口放开到 `target_rack` 的全部 Union
- 路径仍只在 Union/backend 层做 pooling，目标 SSU 只作为终点

### Rack Sharding

- 每个 NPU 把 `M` 均分给 `A4` 的全部 `4` 个目标
- 每个目标写入 `M/4`
- 每个目标仍只走 source 侧两个本地 Union

意义：

- 这个实验比单热点 SSU 更接近“小目标集合热点”
- 用来区分“只改入口路径”和“直接改目标放置”的收益差异
