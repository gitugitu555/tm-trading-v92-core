# C_ExhaustionFade Paper Simulation Design

## Status

C_ExhaustionFade: ROBUST_CANDIDATE

## Purpose

This paper simulator is the next validation layer after historical replay and the robustness grid.
It is intended to convert the reproduced C_ExhaustionFade replay into a realistic paper-trading event stream.
It is not live execution.

## Non-Goals

- No live trading
- No exchange keys
- No order placement
- No Jarvis or AI decision layer
- No new indicators
- No threshold optimization
- No portfolio allocator
- No Kelly sizing yet

## Canonical Strategy Contract

- Symbol: `BTCUSDT`
- Side: long-only
- Bar sizes supported: `500`, `750`, `1000` BTC
- Default production-candidate reference: `750 BTC / h36 / cost12`
- Signal:
  - `regime == EXHAUSTED`
  - `volume > vol_roll_95`
  - `close <= local_low`
- Entry:
  - next-bar open after the signal bar closes
- Exit:
  - time exit after `horizon` bars
- Position rule:
  - one position at a time
- Same-bar behavior:
  - no same-bar entry
- Data rule:
  - no future data

## Paper Event Model

The paper simulator should emit a deterministic event stream that mirrors the replay lifecycle and adds paper-execution state.

### Event Types

- `BAR_CLOSED`
- `SIGNAL_DETECTED`
- `PAPER_ORDER_CREATED`
- `PAPER_ORDER_FILLED`
- `PAPER_POSITION_OPENED`
- `PAPER_POSITION_CLOSED`
- `PAPER_EQUITY_UPDATED`
- `PAPER_RISK_EVENT`

### Required Event Fields

Each event should carry, as applicable:

- `event_type`
- `event_time`
- `symbol`
- `strategy_id`
- `bar_size`
- `horizon`
- `signal_index`
- `entry_index`
- `exit_index`
- `price_open`
- `price_high`
- `price_low`
- `price_close`
- `quantity`
- `notional_usd`
- `fee_bps`
- `slippage_bps`
- `gross_return_bps`
- `net_return_bps`
- `equity_after`

### Timestamp Source

- `BAR_CLOSED`: bar close timestamp
- `SIGNAL_DETECTED`: signal bar close timestamp
- `PAPER_ORDER_CREATED`: signal bar close timestamp
- `PAPER_ORDER_FILLED`: next bar open timestamp
- `PAPER_POSITION_OPENED`: next bar open timestamp
- `PAPER_POSITION_CLOSED`: exit bar open timestamp
- `PAPER_EQUITY_UPDATED`: exit bar open timestamp
- `PAPER_RISK_EVENT`: event timestamp of the triggering condition

## Paper Fill Model

The first paper-sim version should stay intentionally simple and deterministic.

- Default fill: next-bar open
- Configurable `slippage_bps` per side
- Configurable `fee_bps_per_side`
- Round-trip cost should reproduce the replay baseline when configured to `cost_bps=12`
- No partial fills in v1
- No order book simulation in v1
- No latency model in v1
- Reserve field: `latency_ms`

### Cost Baseline

The paper simulator should preserve parity with the replay cost model at the anchor configuration:

- `gross_return_bps - round_trip_cost_bps = net_return_bps`
- Default baseline should match the replay's `cost_bps=12` behavior

## Position Sizing v1

Keep sizing simple and explicit.

- Fixed `notional_usd`
- Fixed fraction of equity
- Maximum one open position
- No Kelly sizing
- No leverage by default
- Optional `max_exposure_fraction`

### Defaults

- `starting_equity_usd`: `100000`
- `exposure_fraction`: `1.0` for research reproduction
- `safer_paper_fraction`: `0.25` optional
- `max_open_positions`: `1`

## Risk Controls v1

Paper controls only. These are not strategy filters.

- `max_daily_loss_pct`
- `max_total_drawdown_pct`
- `max_consecutive_losses`
- `pause_after_loss_streak`
- `disable_if_missing_bar_data`
- `disable_if_signal_columns_missing`
- `disable_if_timestamp_order_invalid`
- `kill_switch` file or config flag

## Metrics

The paper simulator should output:

- `trade_count`
- `win_rate`
- `gross_expectancy_bps`
- `net_expectancy_bps`
- `profit_factor`
- `Sharpe`
- `Sortino` if easy later
- `max_drawdown_pct`
- `daily_equity_curve`
- `monthly_returns`
- `yearly_returns`
- `exposure_time`
- `average_holding_time`
- `worst_trade`
- `best_trade`
- `worst_day`
- `worst_month`
- `positive_year_count`

## Storage / Output Schema

All outputs should live under `reports/paper_sim/` and nowhere else.

### Files

- `reports/paper_sim/c_exhaustion_paper_trades.csv`
- `reports/paper_sim/c_exhaustion_paper_equity.csv`
- `reports/paper_sim/c_exhaustion_paper_events.jsonl`
- `reports/paper_sim/c_exhaustion_paper_summary.json`
- `reports/paper_sim/c_exhaustion_paper_summary.md`

### Trade Log Columns

- `trade_id`
- `signal_time`
- `entry_time`
- `exit_time`
- `signal_index`
- `entry_index`
- `exit_index`
- `entry_price`
- `exit_price`
- `gross_return_bps`
- `fees_bps`
- `slippage_bps`
- `net_return_bps`
- `holding_bars`
- `year`
- `equity_before`
- `equity_after`

### Equity Curve Columns

- `timestamp`
- `equity`
- `drawdown_pct`
- `open_position_count`
- `cash_usd`
- `exposure_usd`

### Event Log Columns

- `event_type`
- `event_time`
- `signal_time`
- `entry_time`
- `exit_time`
- `symbol`
- `strategy_id`
- `bar_size`
- `horizon`
- `signal_index`
- `entry_index`
- `exit_index`
- `price`
- `quantity`
- `notional_usd`
- `fee_bps`
- `slippage_bps`
- `gross_return_bps`
- `net_return_bps`
- `equity_after`
- `reason`

### Summary JSON Fields

- `strategy_id`
- `symbol`
- `bar_size`
- `horizon`
- `starting_equity_usd`
- `exposure_fraction`
- `fee_bps_per_side`
- `slippage_bps_per_side`
- `trade_count`
- `gross_expectancy_bps`
- `net_expectancy_bps`
- `win_rate`
- `profit_factor`
- `sharpe`
- `sortino`
- `max_drawdown_pct`
- `positive_year_count`
- `worst_year`
- `exposure_time`
- `average_holding_time`
- `production_path_touched`

## CLI Design

Future CLI:

```bash
python scripts/run_c_exhaustion_paper_sim.py \
  --bar-dir <bar_dir> \
  --bar-size 750 \
  --horizon 36 \
  --starting-equity 100000 \
  --exposure-fraction 1.0 \
  --fee-bps-per-side 3 \
  --slippage-bps-per-side 3 \
  --output-dir reports/paper_sim/c_exhaustion_750_h36 \
  --write-events \
  --write-trades \
  --write-equity \
  --write-summary
```

## Validation Plan

Paper sim v1 must pass:

- Anchor parity: with cost assumptions equal to `cost_bps=12`, paper sim reproduces `221` trades and approximately `72.212596` net bps
- No same-bar entry
- One-position rule
- No future data
- Equity curve timestamp order is monotonic
- Missing columns fail cleanly
- Production paths are refused
- Reports only, no cache writes

## Testing Plan

Required future tests:

- `test_paper_sim_reproduces_replay_trade_count`
- `test_paper_sim_next_bar_open_fill`
- `test_paper_sim_one_position_only`
- `test_paper_sim_cost_model_matches_replay_cost`
- `test_paper_sim_equity_curve_updates_after_exit`
- `test_paper_sim_refuses_production_output_path`
- `test_paper_sim_missing_required_columns_fails`
- `test_paper_sim_event_log_has_required_event_types`

## Implementation Phases

### Phase P1

Design doc only.

### Phase P2

Pure paper sim module:

- `paper/c_exhaustion_paper_sim.py`

### Phase P3

CLI:

- `scripts/run_c_exhaustion_paper_sim.py`

### Phase P4

Tests.

### Phase P5

Run anchor paper sim on `750/h36/cost12`-equivalent assumptions.

### Phase P6

Commit paper sim report only if anchor passes.

## Final Recommendation

Proceed to implementation only after this design doc is reviewed.
