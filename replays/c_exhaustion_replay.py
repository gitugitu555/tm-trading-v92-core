"""Deterministic replay for the V9.2 C_ExhaustionFade family.

This module is intentionally pure:
- no file writes
- no cache updates
- no production path access

Replay semantics:
- 750 BTC bars are expected, but any chronologically sorted bar frame with the
  required columns is accepted.
- Signals are generated past-only from the EXHAUSTED regime and the prior
  rolling low.
- Entry occurs at the next bar open after the signal bar closes.
- Exit occurs at the next bar open after `horizon_bars` full bars have elapsed.
- Only one position may be open at a time.
- A flat round-trip cost is applied in basis points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import polars as pl

__all__ = [
    "CExhaustionReplayConfig",
    "CExhaustionReplayResult",
    "CExhaustionTrade",
    "add_v92_regime_labels",
    "attach_c_exhaustion_signal",
    "load_750btc_bars",
    "normalize_v92_bar_timestamps",
    "replay_c_exhaustionfade",
    "summarize_trades",
    "year_split_metrics",
]


@dataclass(frozen=True)
class CExhaustionReplayConfig:
    horizon_bars: int = 36
    round_trip_cost_bps: float = 5.0
    signal_lookback_bars: int = 50
    bar_size: int = 750


@dataclass(frozen=True)
class CExhaustionTrade:
    signal_index: int
    entry_index: int
    exit_index: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_return_bps: float
    net_return_bps: float
    holding_bars: int
    year: int


@dataclass(frozen=True)
class CExhaustionReplayResult:
    config: CExhaustionReplayConfig
    trades: pd.DataFrame
    summary: dict[str, Any]
    yearly: pd.DataFrame


def _ensure_polars_frame(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(f"Unsupported frame type: {type(df)!r}")


def _normalize_epoch_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column) > 1e14)
        .then(pl.from_epoch(column, time_unit="us"))
        .otherwise(pl.from_epoch(column, time_unit="ms"))
        .cast(pl.Datetime("ns"))
    )


def _normalize_bar_times(df: pl.DataFrame) -> pl.DataFrame:
    schema = df.schema
    if "open_time" not in schema or "close_time" not in schema:
        raise ValueError("Input frame must contain open_time and close_time columns.")
    if schema["open_time"] == pl.Datetime("ns") and schema["close_time"] == pl.Datetime("ns"):
        return df
    return df.with_columns(
        [
            _normalize_epoch_expr("open_time").alias("datetime_open"),
            _normalize_epoch_expr("close_time").alias("datetime_close"),
        ]
    )


def normalize_v92_bar_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    """Public wrapper used by CLI and tests."""
    return _normalize_bar_times(df)


def _add_regime_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Local copy of the V9.2 regime labels without importing the broken package init."""
    if "bar_range" not in df.columns:
        df = df.with_columns((pl.col("high") - pl.col("low")).alias("bar_range"))
    if "body_size" not in df.columns:
        df = df.with_columns((pl.col("close") - pl.col("open")).abs().alias("body_size"))

    df = df.with_columns(
        [
            ((pl.col("close") - pl.col("open")).abs() / pl.col("volume_delta").abs().clip(lower_bound=1e-9)).alias(
                "delta_efficiency"
            )
        ]
    )

    df = df.with_columns(
        [
            (pl.col("close") / pl.col("close").shift(1) - 1).rolling_std(window_size=500).shift(1).alias("rv_1d"),
            pl.col("volume").rolling_quantile(quantile=0.80, interpolation="nearest", window_size=3500).shift(1).alias("vol_80th_pct"),
            pl.col("delta_efficiency").rolling_median(window_size=200).shift(1).alias("de_median"),
            pl.col("low").rolling_min(window_size=500).shift(1).alias("rolling_min_low"),
            (pl.col("high") - pl.col("low")).rolling_mean(window_size=7000).shift(1).alias("rolling_mean_range"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("rv_1d").rolling_quantile(quantile=0.15, interpolation="nearest", window_size=15000).shift(1).alias(
                "rv_15th_pct"
            ),
            ((pl.col("close") - pl.col("rolling_min_low")) / pl.col("rolling_mean_range")).alias("adr_stretch"),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("rv_1d") < pl.col("rv_15th_pct"))
            .then(pl.lit("TREND_BUILDUP"))
            .when(
                (pl.col("volume") > pl.col("vol_80th_pct"))
                & (pl.col("body_size") < (0.25 * pl.col("bar_range")))
                & (pl.col("delta_efficiency") < pl.col("de_median"))
            )
            .then(pl.lit("ABSORPTION"))
            .when((pl.col("adr_stretch") > 0.85) | (pl.col("adr_stretch") < 0.15))
            .then(pl.lit("EXHAUSTED"))
            .otherwise(pl.lit("NOISE"))
            .alias("regime")
        ]
    )

    return df.drop(["rv_1d", "vol_80th_pct", "rolling_min_low", "rolling_mean_range", "rv_15th_pct", "adr_stretch"])


def add_v92_regime_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Public wrapper used by CLI and tests."""
    return _add_regime_labels(df)


def load_750btc_bars(bar_dir: Path) -> pl.DataFrame:
    """Load 750 BTC parquet shards from a directory and sort chronologically."""
    files = sorted(bar_dir.glob("BTCUSDT_tier2_750btc_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No 750 BTC parquet shards found in {bar_dir}")
    df = pl.concat([pl.scan_parquet(path) for path in files]).collect()
    return df.sort("open_time")


def attach_c_exhaustion_signal(
    df: pl.DataFrame | pd.DataFrame,
    *,
    signal_lookback_bars: int = 50,
) -> pl.DataFrame:
    """Attach the past-only C signal used by the deterministic replay.

    The signal fires when the bar is in EXHAUSTED regime and the close is at or
    below the prior rolling low. The lookback excludes the current bar.
    """
    frame = _ensure_polars_frame(df).sort("open_time")
    frame = _normalize_bar_times(frame)

    required = {"open", "high", "low", "close", "regime", "open_time", "close_time"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "regime" not in frame.columns:
        frame = _add_regime_labels(frame)

    frame = frame.with_columns(
        pl.col("low").rolling_min(window_size=signal_lookback_bars, min_samples=1).shift(1).alias("prior_rolling_low")
    )
    frame = frame.with_columns(
        (
            (pl.col("regime") == "EXHAUSTED")
            & pl.col("prior_rolling_low").is_not_null()
            & (pl.col("close") <= pl.col("prior_rolling_low"))
        ).alias("c_signal")
    )
    return frame


def _to_timestamp(value: Any) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        unit = "us" if float(value) > 1e14 else "ms"
        return pd.to_datetime(value, unit=unit)
    return pd.to_datetime(value)


def _trade_frame_to_records(trades: list[CExhaustionTrade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "signal_index",
                "entry_index",
                "exit_index",
                "signal_time",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "gross_return_bps",
                "net_return_bps",
                "holding_bars",
                "year",
            ]
        )
    return pd.DataFrame([trade.__dict__ for trade in trades])


def replay_c_exhaustionfade(
    df: pl.DataFrame | pd.DataFrame,
    *,
    horizon_bars: int = 36,
    round_trip_cost_bps: float = 5.0,
    signal_lookback_bars: int = 50,
    bar_size: int = 750,
) -> CExhaustionReplayResult:
    """Replay the C_ExhaustionFade long-only strategy with deterministic timing."""
    config = CExhaustionReplayConfig(
        horizon_bars=horizon_bars,
        round_trip_cost_bps=round_trip_cost_bps,
        signal_lookback_bars=signal_lookback_bars,
        bar_size=bar_size,
    )

    frame = attach_c_exhaustion_signal(df, signal_lookback_bars=signal_lookback_bars)
    pdf = frame.to_pandas()
    pdf = pdf.sort_values("open_time").reset_index(drop=True)

    trades: list[CExhaustionTrade] = []
    position_exit_index: int | None = None
    n_rows = len(pdf)

    for signal_index in range(n_rows):
        if position_exit_index is not None and signal_index >= position_exit_index:
            position_exit_index = None

        if position_exit_index is not None:
            continue

        if not bool(pdf.at[signal_index, "c_signal"]):
            continue

        entry_index = signal_index + 1
        exit_index = entry_index + horizon_bars

        if exit_index >= n_rows:
            continue

        entry_price = float(pdf.at[entry_index, "open"])
        exit_price = float(pdf.at[exit_index, "open"])
        gross_return_bps = (exit_price / entry_price - 1.0) * 10_000.0
        net_return_bps = gross_return_bps - round_trip_cost_bps

        signal_time = _to_timestamp(pdf.at[signal_index, "close_time"])
        entry_time = _to_timestamp(pdf.at[entry_index, "open_time"])
        exit_time = _to_timestamp(pdf.at[exit_index, "open_time"])
        year = int(entry_time.year)

        trades.append(
            CExhaustionTrade(
                signal_index=signal_index,
                entry_index=entry_index,
                exit_index=exit_index,
                signal_time=signal_time,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                gross_return_bps=gross_return_bps,
                net_return_bps=net_return_bps,
                holding_bars=horizon_bars,
                year=year,
            )
        )
        position_exit_index = exit_index

    trade_df = _trade_frame_to_records(trades)
    summary = summarize_trades(trade_df, cost_bps=round_trip_cost_bps)
    yearly = year_split_metrics(trade_df, cost_bps=round_trip_cost_bps)
    return CExhaustionReplayResult(config=config, trades=trade_df, summary=summary, yearly=yearly)


def summarize_trades(trades: pd.DataFrame, *, cost_bps: float) -> dict[str, Any]:
    """Summarize a trade log using the same cost model as replay."""
    if trades.empty:
        return {
            "trade_count": 0,
            "gross_expectancy_bps": 0.0,
            "net_expectancy_bps": 0.0,
            "win_rate": 0.0,
            "avg_win_bps": 0.0,
            "avg_loss_bps": 0.0,
            "median_trade_bps": 0.0,
            "p25_trade_bps": 0.0,
            "p75_trade_bps": 0.0,
            "max_win_bps": 0.0,
            "max_loss_bps": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_bps": 0.0,
            "avg_hold_bars": 0.0,
            "median_hold_bars": 0.0,
            "cost_bps": cost_bps,
        }

    net = trades["net_return_bps"].astype(float)
    gross = trades["gross_return_bps"].astype(float)
    wins = net[net > 0.0]
    losses = net[net < 0.0]
    equity_curve = net.cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max

    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and abs(losses.sum()) > 0 else float("inf") if len(wins) > 0 else 0.0

    return {
        "trade_count": int(len(trades)),
        "gross_expectancy_bps": float(gross.mean()),
        "net_expectancy_bps": float(net.mean()),
        "win_rate": float((net > 0.0).mean()),
        "avg_win_bps": float(wins.mean()) if len(wins) > 0 else 0.0,
        "avg_loss_bps": float(losses.mean()) if len(losses) > 0 else 0.0,
        "median_trade_bps": float(net.median()),
        "p25_trade_bps": float(net.quantile(0.25)),
        "p75_trade_bps": float(net.quantile(0.75)),
        "max_win_bps": float(net.max()),
        "max_loss_bps": float(net.min()),
        "profit_factor": profit_factor,
        "max_drawdown_bps": float(drawdown.min()),
        "avg_hold_bars": float(trades["holding_bars"].mean()),
        "median_hold_bars": float(trades["holding_bars"].median()),
        "cost_bps": cost_bps,
    }


def year_split_metrics(trades: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    """Compute per-year replay metrics using trade entry year."""
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "trade_count",
                "gross_expectancy_bps",
                "net_expectancy_bps",
                "win_rate",
                "profit_factor",
                "max_drawdown_bps",
                "cost_bps",
            ]
        )

    rows = []
    for year, group in trades.groupby("year", sort=True):
        net = group["net_return_bps"].astype(float)
        gross = group["gross_return_bps"].astype(float)
        wins = net[net > 0.0]
        losses = net[net < 0.0]
        equity_curve = net.cumsum()
        running_max = equity_curve.cummax()
        drawdown = equity_curve - running_max
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and abs(losses.sum()) > 0 else float("inf") if len(wins) > 0 else 0.0
        rows.append(
            {
                "year": int(year),
                "trade_count": int(len(group)),
                "gross_expectancy_bps": float(gross.mean()),
                "net_expectancy_bps": float(net.mean()),
                "win_rate": float((net > 0.0).mean()),
                "profit_factor": profit_factor,
                "max_drawdown_bps": float(drawdown.min()),
                "cost_bps": cost_bps,
            }
        )

    return pd.DataFrame(rows)
