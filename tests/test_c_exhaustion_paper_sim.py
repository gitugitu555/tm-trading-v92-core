from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from paper.c_exhaustion_paper_sim import PaperSimConfig, run_c_exhaustion_paper_sim
from replays.c_exhaustion_replay import replay_c_exhaustionfade


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


def _paper_frame_with_single_signal() -> pl.DataFrame:
    return _bars(
        [
            _base_row(1, 100, 110, 100, 104, 2, 100.0, "NOISE"),
            _base_row(2, 104, 110, 100, 104, 3, 100.0, "NOISE"),
            _base_row(3, 104, 110, 100, 104, 4, 100.0, "NOISE"),
            _base_row(4, 104, 110, 90, 90, 5, 200.0, "EXHAUSTED"),
            _base_row(5, 200, 205, 198, 202, 6, 100.0, "NOISE"),
            _base_row(6, 202, 210, 201, 205, 7, 100.0, "NOISE"),
        ]
    )


def test_paper_sim_reproduces_replay_trade_count():
    df = _paper_frame_with_single_signal()
    config = PaperSimConfig(horizon_bars=1)

    replay = replay_c_exhaustionfade(df, horizon_bars=1, round_trip_cost_bps=12.0)
    paper = run_c_exhaustion_paper_sim(df, config=config)

    assert paper.summary["trade_count"] == replay.summary["trade_count"] == 1


def test_paper_sim_cost_model_matches_replay_cost():
    df = _paper_frame_with_single_signal()
    config = PaperSimConfig(horizon_bars=1, fee_bps_per_side=3.0, slippage_bps_per_side=3.0)

    replay = replay_c_exhaustionfade(df, horizon_bars=1, round_trip_cost_bps=12.0)
    paper = run_c_exhaustion_paper_sim(df, config=config)

    assert paper.trades.iloc[0]["net_return_bps"] == pytest.approx(replay.trades.iloc[0]["net_return_bps"])
    assert paper.summary["net_expectancy_bps"] == pytest.approx(replay.summary["net_expectancy_bps"])


def test_paper_sim_next_bar_open_fill():
    df = _paper_frame_with_single_signal()
    config = PaperSimConfig(horizon_bars=1, fee_bps_per_side=3.0, slippage_bps_per_side=3.0)

    paper = run_c_exhaustion_paper_sim(df, config=config)
    trade = paper.trades.iloc[0]

    assert trade["entry_time"] == pd.Timestamp("1970-01-01 00:00:00.005000")
    assert trade["entry_price_raw"] == pytest.approx(200.0)
    assert trade["entry_fill_price"] == pytest.approx(200.0 * 1.0003)
    assert trade["exit_fill_price"] == pytest.approx(202.0 * 0.9997)


def test_paper_sim_one_position_only():
    df = _bars(
        [
            _base_row(1, 100, 110, 100, 104, 2, 100.0, "NOISE"),
            _base_row(2, 104, 110, 100, 104, 3, 100.0, "NOISE"),
            _base_row(3, 104, 110, 100, 104, 4, 100.0, "NOISE"),
            _base_row(4, 104, 110, 90, 90, 5, 200.0, "EXHAUSTED"),
            _base_row(5, 90, 95, 80, 82, 6, 210.0, "EXHAUSTED"),
            _base_row(6, 300, 305, 295, 302, 7, 100.0, "NOISE"),
            _base_row(7, 302, 310, 301, 309, 8, 100.0, "NOISE"),
        ]
    )
    config = PaperSimConfig(horizon_bars=2)

    paper = run_c_exhaustion_paper_sim(df, config=config)

    assert paper.summary["trade_count"] == 1
    assert paper.trades.iloc[0]["entry_index"] == 4
    assert paper.trades.iloc[0]["exit_index"] == 6


def test_paper_sim_equity_curve_updates_after_exit():
    df = _paper_frame_with_single_signal()
    config = PaperSimConfig(horizon_bars=1, fee_bps_per_side=3.0, slippage_bps_per_side=3.0)

    paper = run_c_exhaustion_paper_sim(df, config=config)
    trade = paper.trades.iloc[0]
    equity = paper.equity_curve

    assert list(equity["timestamp"]) == sorted(equity["timestamp"].tolist())
    assert trade["equity_after"] == pytest.approx(trade["equity_before"] + trade["pnl_usd"])
    assert equity.iloc[0]["equity"] == pytest.approx(trade["equity_after"])


def test_paper_sim_event_log_has_required_event_types():
    df = _paper_frame_with_single_signal()
    config = PaperSimConfig(horizon_bars=1)

    paper = run_c_exhaustion_paper_sim(df, config=config)
    event_types = set(paper.events["event_type"].tolist())

    assert {
        "SIGNAL_DETECTED",
        "PAPER_ORDER_CREATED",
        "PAPER_ORDER_FILLED",
        "PAPER_POSITION_OPENED",
        "PAPER_POSITION_CLOSED",
        "PAPER_EQUITY_UPDATED",
    }.issubset(event_types)


def test_paper_sim_rejects_max_open_positions_not_one():
    df = _paper_frame_with_single_signal()
    config = PaperSimConfig(horizon_bars=1, max_open_positions=2)

    with pytest.raises(ValueError, match="max_open_positions=1"):
        run_c_exhaustion_paper_sim(df, config=config)


def test_paper_sim_missing_required_columns_fails():
    df = pl.DataFrame(
        [
            {
                "open_time": 1,
                "close_time": 2,
                "open": 100.0,
                "high": 110.0,
                "low": 100.0,
                "close": 104.0,
                "volume_delta": 10.0,
                "trade_count": 5,
                "regime": "EXHAUSTED",
            }
        ]
    )
    config = PaperSimConfig(horizon_bars=1)

    with pytest.raises(ValueError, match="Missing required columns"):
        run_c_exhaustion_paper_sim(df, config=config)
