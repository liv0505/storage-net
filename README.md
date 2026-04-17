# SSU-Centric Topology Modeling & Simulation Toolkit

一个面向数据交换节点组网分析的 Python 项目。当前版本以 `SSU -> SSU` 通信为中心，基于统一的 `8 SSU + 2 Union` 数据交换节点模型，对多类拓扑做结构与通信性能评估，并输出 CSV / HTML 报告。

## 当前建模范围

- 支持拓扑：`2D-FullMesh`、`2D-Torus`、`3D-Torus`、`Clos`、`Clos-64/128/192/256`、`Clos-4P-FullMesh`、`Dragon-Fly`、`SparseMesh`
- 基本单元：每个数据交换节点包含 `8` 个 SSU 和 `2` 个 Union
- 节点内互联：每个 SSU 通过 `2 x 200 Gbps` UB 端口分别连接到两个 Union
- 后端组网：Union 间后端链路默认按 `400 Gbps` 建模
- 前端 `1825` 网卡：作为 Union 端口预算约束处理，不单独建图

## 路由与通信语义

评估视角统一为 `SSU -> SSU`。

- 同一数据交换节点内的 SSU 通信：只经过节点内 Union 交换，不占用直连拓扑链路或 Clos 后端链路
- 跨数据交换节点的 SSU 通信：按 `源 SSU -> 源 Union -> 后端拓扑 -> 目的 Union -> 目的 SSU` 建模
- 直连拓扑可评估的路由模式：`DOR`、`SHORTEST_PATH`、`FULL_PATH`（兼容旧别名 `MIN_HOPS`、`PORT_BALANCED`）
- Clos 拓扑可评估的路由模式：`ECMP`

`SHORTEST_PATH`：源 SSU 先 1:1 分流到两个源端 Union，两个平面分别在所有最短路径间分摊，到达目的节点后再 1:1 下行到目标 SSU。\n\n`FULL_PATH`：源 SSU 同样先 1:1 分流到两个 Union；随后每个平面在所有可用出端口间均匀分流，并优先选择最短路径；若某个出端口上不存在可用最短路径，则退化为选择到目的节点跳数最少的非最短路径。兼容旧别名 `PORT_BALANCED`。

## 指标体系

### 结构指标

- `diameter`
- `average_hops`
- `bisection_bandwidth_gbps`

### 通信指标

针对两类 workload 分别输出：

- `A2A`：所有 SSU 参与全互发
- `Sparse 1-to-N`：仅部分 SSU 激活，默认激活比例 `25%`，每个激活 SSU 随机选择 `N=2` 个目标

每类 workload 输出：

- `completion_time_s`
- `completion_time_p50_s`
- `completion_time_p95_s`
- `per_ssu_throughput_gbps`
- `max_link_utilization`
- `link_utilization_cv`

## 安装

```bash
pip install -r requirements.txt
```

## 运行

默认运行全部 4 类拓扑：

```bash
python main.py
```

一个常用示例：

```bash
python main.py --topologies 2D-FullMesh,Clos --routing-mode FULL_PATH --sparse-active-ratio 0.25 --sparse-target-count 2 --port-balanced-max-detour-hops 1 --clos-uplinks-per-exchange-node 4 --output-dir outputs_example
```

## 输出产物

运行后会在输出目录生成：

- `summary.csv`
  - 仅包含新的结构指标与 workload 指标
- `run_config.json`
  - 包含 routing / workload 参数，以及本次实际分析的 `selected_topologies`
- `topology_dashboard.html`
  - 展示硬件假设、拓扑配置、路由说明、结构指标、A2A 与 Sparse 1-to-N 指标

## 配置项

`AnalysisConfig` 当前重点参数：

- `routing_mode`
- `sparse_active_ratio`
- `sparse_target_count`
- `port_balanced_max_detour_hops`
- `clos_uplinks_per_exchange_node`
- `message_size_mb`
- `random_seed`
- `output_dir`

## 开发说明

- CSV 是机器读结果的主出口
- HTML 与 CSV 使用同一批 SSU-centric 指标来源，避免旧指标兼容层造成语义漂移
- 当前实现优先保证结构正确性和可解释性，后续可以继续扩展更高保真链路模型、集合通信算法和离散事件仿真
