from __future__ import annotations

import pandas as pd
import polars as pl

from replays.c_exhaustion_replay import (
    attach_c_exhaustion_signal,
    replay_c_exhaustionfade,
    summarize_trades,
    year_split_metrics,
)


def _bars(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _base_row(
    open_time: int,
    open_price: float,
    high: float,
    low: float,
    close: float,
    close_time: int,
    volume: float = 100.0,
    regime: str = "EXHAUSTED",
) -> dict:
    return {
        "open_time": open_time,
        "close_time": close_time,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "volume_delta": 10.0,
        "trade_count": 5,
        "regime": regime,
    }


def test_c_signal_requires_volume_above_vol_roll_95():
    base = _bars(
        [
            _base_row(1, 100, 110, 100, 104, 2, 100.0, "NOISE"),
            _base_row(2, 104, 110, 100, 104, 3, 100.0, "NOISE"),
            _base_row(3, 104, 110, 100, 104, 4, 100.0, "NOISE"),
            _base_row(4, 104, 110, 90, 90, 5, 100.0, "EXHAUSTED"),
        ]
    )
    signal = attach_c_exhaustion_signal(base, signal_lookback_bars=2, volume_lookback_bars=3)

    assert signal.select("c_signal").to_series().to_list() == [False, False, False, False]


def test_c_signal_fires_with_original_phase1g_conditions():
    df = _bars(
        [
            _base_row(1, 100, 110, 100, 104, 2, 100.0, "NOISE"),
            _base_row(2, 104, 110, 100, 104, 3, 100.0, "NOISE"),
            _base_row(3, 104, 110, 100, 104, 4, 100.0, "NOISE"),
            _base_row(4, 104, 110, 90, 90, 5, 105.0, "EXHAUSTED"),
        ]
    )

    signal = attach_c_exhaustion_signal(df, signal_lookback_bars=2, volume_lookback_bars=3)

    assert signal.select("c_signal").to_series().to_list() == [False, False, False, True]


def test_c_signal_does_not_use_future_volume_or_future_low():
    base = _bars(
        [
            _base_row(1, 100, 110, 100, 104, 2, 100.0, "NOISE"),
            _base_row(2, 104, 110, 100, 104, 3, 100.0, "NOISE"),
            _base_row(3, 104, 110, 100, 104, 4, 100.0, "NOISE"),
            _base_row(4, 104, 110, 90, 90, 5, 105.0, "EXHAUSTED"),
        ]
    )
    with_future_extreme = _bars(
        [
            _base_row(1, 100, 110, 100, 104, 2, 100.0, "NOISE"),
            _base_row(2, 104, 110, 100, 104, 3, 100.0, "NOISE"),
            _base_row(3, 104, 110, 100, 104, 4, 100.0, "NOISE"),
            _base_row(4, 104, 110, 90, 90, 5, 105.0, "EXHAUSTED"),
            _base_row(5, 90, 95, 1, 1, 6, 1000.0, "NOISE"),
        ]
    )

    signal_base = attach_c_exhaustion_signal(base, signal_lookback_bars=2, volume_lookback_bars=3)
    signal_future = attach_c_exhaustion_signal(with_future_extreme, signal_lookback_bars=2, volume_lookback_bars=3)

    assert signal_base.select("c_signal").to_series().to_list() == [False, False, False, True]
    assert signal_future.select("c_signal").to_series().to_list()[:4] == [False, False, False, True]


def test_next_bar_open_entry_is_used():
    df = _bars(
        [
            _base_row(1, 100, 110, 100, 104, 2, 100.0, "NOISE"),
            _base_row(2, 104, 110, 100, 104, 3, 100.0, "NOISE"),
            _base_row(3, 104, 110, 100, 104, 4, 100.0, "NOISE"),
            _base_row(4, 104, 110, 90, 90, 5, 105.0, "EXHAUSTED"),
            _base_row(5, 200, 205, 198, 202, 6, 100.0, "NOISE"),
            _base_row(6, 202, 210, 201, 205, 7, 100.0, "NOISE"),
        ]
    )

    result = replay_c_exhaustionfade(
        df,
        horizon_bars=1,
        round_trip_cost_bps=5.0,
        signal_lookback_bars=2,
        volume_lookback_bars=3,
    )
    assert result.summary["trade_count"] == 1
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == 200.0
    assert trade["signal_time"] == pd.Timestamp("1970-01-01 00:00:00.005000")
    assert trade["entry_time"] == pd.Timestamp("1970-01-01 00:00:00.005000")


def test_one_position_rule_skips_overlapping_signals():
    df = _bars(
        [
            _base_row(1, 100, 110, 100, 104, 2, 100.0, "NOISE"),
            _base_row(2, 104, 110, 100, 104, 3, 100.0, "NOISE"),
            _base_row(3, 104, 110, 100, 104, 4, 100.0, "NOISE"),
            _base_row(4, 104, 110, 90, 90, 5, 105.0, "EXHAUSTED"),
            _base_row(5, 90, 95, 80, 82, 6, 106.0, "EXHAUSTED"),
            _base_row(6, 82, 85, 70, 72, 7, 107.0, "EXHAUSTED"),
            _base_row(7, 300, 305, 295, 302, 8, 100.0, "NOISE"),
            _base_row(8, 302, 310, 301, 309, 9, 100.0, "NOISE"),
        ]
    )

    result = replay_c_exhaustionfade(
        df,
        horizon_bars=2,
        round_trip_cost_bps=5.0,
        signal_lookback_bars=2,
        volume_lookback_bars=3,
    )
    assert result.summary["trade_count"] == 1
    assert result.trades.iloc[0]["entry_index"] == 4
    assert result.trades.iloc[0]["exit_index"] == 6


def test_cost_accounting_is_explicit_and_linear():
    df = _bars(
        [
            _base_row(1, 100, 110, 100, 104, 2, 100.0, "NOISE"),
            _base_row(2, 104, 110, 100, 104, 3, 100.0, "NOISE"),
            _base_row(3, 104, 110, 100, 104, 4, 100.0, "NOISE"),
            _base_row(4, 104, 110, 90, 90, 5, 105.0, "EXHAUSTED"),
            _base_row(5, 210, 215, 205, 212, 6, 100.0, "NOISE"),
            _base_row(6, 220, 225, 219, 223, 7, 100.0, "NOISE"),
        ]
    )

    zero_cost = replay_c_exhaustionfade(
        df,
        horizon_bars=1,
        round_trip_cost_bps=0.0,
        signal_lookback_bars=2,
        volume_lookback_bars=3,
    )
    ten_cost = replay_c_exhaustionfade(
        df,
        horizon_bars=1,
        round_trip_cost_bps=10.0,
        signal_lookback_bars=2,
        volume_lookback_bars=3,
    )

    gross = zero_cost.trades.iloc[0]["gross_return_bps"]
    assert zero_cost.summary["net_expectancy_bps"] == gross
    assert ten_cost.summary["net_expectancy_bps"] == gross - 10.0
    assert ten_cost.trades.iloc[0]["net_return_bps"] == gross - 10.0


def test_year_split_metrics_and_trade_year_assignment():
    df = _bars(
        [
            _base_row(1704067200000, 100, 110, 100, 104, 1704067260000, 100.0, "NOISE"),   # 2024-01-01
            _base_row(1704067260000, 104, 110, 100, 104, 1704067320000, 100.0, "NOISE"),
            _base_row(1704067320000, 104, 110, 100, 104, 1704067380000, 100.0, "NOISE"),
            _base_row(1704067380000, 104, 110, 90, 90, 1704067440000, 105.0, "EXHAUSTED"), # trade 1
            _base_row(1704067440000, 200, 205, 198, 202, 1704067500000, 100.0, "NOISE"),
            _base_row(1704067500000, 202, 210, 201, 205, 1704067560000, 100.0, "NOISE"),
            _base_row(1735689600000, 300, 310, 299, 304, 1735689660000, 100.0, "NOISE"),   # 2025-01-01
            _base_row(1735689660000, 304, 315, 299, 304, 1735689720000, 100.0, "NOISE"),
            _base_row(1735689720000, 304, 315, 299, 304, 1735689780000, 100.0, "NOISE"),
            _base_row(1735689780000, 304, 315, 250, 255, 1735689840000, 105.0, "EXHAUSTED"), # trade 2
            _base_row(1735689840000, 400, 410, 399, 405, 1735689900000, 100.0, "NOISE"),
            _base_row(1735689900000, 405, 415, 404, 408, 1735689960000, 100.0, "NOISE"),
        ]
    )

    result = replay_c_exhaustionfade(
        df,
        horizon_bars=1,
        round_trip_cost_bps=5.0,
        signal_lookback_bars=2,
        volume_lookback_bars=3,
    )
    yearly = result.yearly.sort_values("year").reset_index(drop=True)

    assert result.summary["trade_count"] == 2
    assert yearly["year"].to_list() == [2024, 2025]
    assert yearly["trade_count"].to_list() == [1, 1]
    assert yearly["net_expectancy_bps"].iloc[0] == result.trades.iloc[0]["net_return_bps"]
    assert yearly["net_expectancy_bps"].iloc[1] == result.trades.iloc[1]["net_return_bps"]


def test_summarize_trades_reports_expected_cost_ladder_behavior():
    trades = pd.DataFrame(
        [
            {
                "gross_return_bps": 12.0,
                "net_return_bps": 7.0,
                "holding_bars": 36,
                "year": 2024,
            },
            {
                "gross_return_bps": -8.0,
                "net_return_bps": -13.0,
                "holding_bars": 36,
                "year": 2024,
            },
        ]
    )

    summary = summarize_trades(trades, cost_bps=5.0)
    assert summary["trade_count"] == 2
    assert summary["gross_expectancy_bps"] == 2.0
    assert summary["net_expectancy_bps"] == -3.0
    assert summary["profit_factor"] == 7.0 / 13.0
