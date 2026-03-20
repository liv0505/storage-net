# Clos topology correction

## Summary
- The current Clos builder assigns each exchange node four shared uplinks, but the confirmed requirement is that each of the two union planes supplies four separate uplinks to plane-local L2 Union switches for a total of eight 400 Gbps uplinks per exchange.
- This change touches the builder, topology + routing tests, and the documentation so we document the plan here before implementing.

## Requirements
1. Two union planes per exchange, each with 4 upward 400 Gbps links into plane-local L2 Union switches.
2. Each plane-local L2 Union switch fans out to at most 18 exchange nodes in that plane and still carries 400 Gbps links.
3. ECMP routing between exchange nodes should expose `2 * clos_uplinks_per_exchange_node` shortest paths so path diversity reflects both planes.
4. Existing validation + config guardrails for Clos uplink counts remain in place and simply cap the per-plane uplinks.

## Proposed design
1. Build two disjoint spine pools (plane 0 and plane 1). Each pool has `clos_uplinks_per_exchange_node` L2 switches and is connected only to union nodes of that plane.
2. When wiring each exchange node, connect union 0 to all plane-0 spines and union 1 to all plane-1 spines so that each union contributes the configured number of uplinks and each exchange has `2 * clos_uplinks_per_exchange_node` uplinks in total.
3. Keep `_CLOS_EXCHANGE_NODE_COUNT = 18` and `_validate_clos_spine_fanout` so every spine still fans out to 18 exchanges in its plane.
4. Adjust `_validate_clos_uplink_budget` naming/message to reflect that the limit applies per plane, not per exchange, while still enforcing the 1-6 constraint.
5. Update tests to reflect the new counts (8 uplinks per exchange, `2 * clos_uplinks_per_exchange_node` spines, and dual-plane ECMP path counts) and add coverage ensuring union nodes each expose the per-plane uplinks.

## Tests to run
- `pytest tests/test_topologies.py -k clos`
- `pytest tests/test_routing.py::test_ecmp_returns_equal_cost_paths_for_clos`

## Follow-up
- After implementing, rerun the specified tests to confirm the builder + routing behavior align with the corrected counts.
