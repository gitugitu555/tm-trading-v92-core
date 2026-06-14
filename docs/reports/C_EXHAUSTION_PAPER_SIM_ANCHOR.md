# C_ExhaustionFade Paper-Sim Anchor

## Status

C_ExhaustionFade: PAPER_SIM_ANCHOR_REPRODUCED

## Configuration

| Metric | Value |
|---|---:|
| symbol | BTCUSDT |
| bar_size | 750 |
| horizon | 36 |
| starting_equity_usd | 100000.0 |
| exposure_fraction | 1.0 |
| fee_bps_per_side | 3 |
| slippage_bps_per_side | 3 |
| round_trip_cost_bps | 12 |

## Input Data

| Metric | Value |
|---|---:|
| bar_dir | /home/tokio/tm-trading-v92-phase1f/bars_750btc |
| production_path_touched | false |

## Anchor Result

| Metric | Value |
|---|---:|
| ending_equity_usd | 452380.0526755554 |
| total_return_pct | 352.380053 |
| trade_count | 221 |
| net_expectancy_bps | 72.212596 |
| win_rate | 0.615385 |
| profit_factor | 2.191757 |
| max_drawdown_pct | 18.673184 |
| positive_year_count | 6 |
| worst_year | 2025 |

## Interpretation

The paper-sim CLI reproduces the canonical 750 BTC / h36 / cost12-equivalent anchor.

This confirms:

- the CLI can load validated 750 BTC shards
- the pure paper-sim module is callable from the CLI
- paper-sim cost assumptions reproduce replay expectancy
- output remains report-only
- `production_path_touched=false`

This is not live trading readiness.

## Caveats

- Uses historical validated shards only.
- No live data feed.
- No exchange adapter.
- No order placement.
- No latency model beyond reserved fields.
- No partial-fill model.
- No order-book simulation.
- No portfolio allocator.
- 2025 remains the worst year and must stay monitored.

## Next Phase

Proceed to risk-overlay paper simulation design.

Do not proceed to live execution.
