"""Pure paper simulation for the V9.2 C_ExhaustionFade strategy.

This module is intentionally in-memory only:
- no file writes
- no cache writes
- no production path access
- no network calls
- no exchange calls
- no CLI parsing

It reuses the canonical replay trade stream and enriches it into a paper-trading
event log, paper trade log, equity curve, and summary metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import polars as pl

from replays.c_exhaustion_replay import replay_c_exhaustionfade

__all__ = [
    "PaperEvent",
    "PaperSimConfig",
    "PaperSimResult",
    "PaperTrade",
    "run_c_exhaustion_paper_sim",
]


@dataclass(frozen=True)
class PaperSimConfig:
    strategy_id: str = "C_ExhaustionFade"
    symbol: str = "BTCUSDT"
    bar_size: int = 750
    horizon_bars: int = 36
    starting_equity_usd: float = 100000.0
    exposure_fraction: float = 1.0
    fixed_notional_usd: float | None = None
    fee_bps_per_side: float = 3.0
    slippage_bps_per_side: float = 3.0
    max_open_positions: int = 1
    max_exposure_fraction: float = 1.0
    latency_ms: int = 0


@dataclass(frozen=True)
class PaperTrade:
    trade_id: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    signal_index: int
    entry_index: int
    exit_index: int
    entry_price_raw: float
    exit_price_raw: float
    entry_fill_price: float
    exit_fill_price: float
    quantity: float
    notional_usd: float
    gross_return_bps: float
    fees_bps: float
    slippage_bps: float
    net_return_bps: float
    pnl_usd: float
    equity_before: float
    equity_after: float
    holding_bars: int
    year: int


@dataclass(frozen=True)
class PaperEvent:
    event_type: str
    event_time: pd.Timestamp
    symbol: str
    strategy_id: str
    bar_size: int
    horizon: int
    signal_index: int
    entry_index: int
    exit_index: int
    price: float
    quantity: float
    notional_usd: float
    fee_bps: float
    slippage_bps: float
    gross_return_bps: float
    net_return_bps: float
    equity_after: float
    reason: str


@dataclass(frozen=True)
class PaperSimResult:
    config: PaperSimConfig
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    events: pd.DataFrame
    summary: dict[str, Any]


def _empty_trade_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trade_id",
            "signal_time",
            "entry_time",
            "exit_time",
            "signal_index",
            "entry_index",
            "exit_index",
            "entry_price_raw",
            "exit_price_raw",
            "entry_fill_price",
            "exit_fill_price",
            "quantity",
            "notional_usd",
            "gross_return_bps",
            "fees_bps",
            "slippage_bps",
            "net_return_bps",
            "pnl_usd",
            "equity_before",
            "equity_after",
            "holding_bars",
            "year",
        ]
    )


def _empty_equity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["timestamp", "equity", "drawdown_pct", "open_position_count", "cash_usd", "exposure_usd"]
    )


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_type",
            "event_time",
            "symbol",
            "strategy_id",
            "bar_size",
            "horizon",
            "signal_index",
            "entry_index",
            "exit_index",
            "price",
            "quantity",
            "notional_usd",
            "fee_bps",
            "slippage_bps",
            "gross_return_bps",
            "net_return_bps",
            "equity_after",
            "reason",
        ]
    )


def _trade_rows_to_frame(trades: list[PaperTrade]) -> pd.DataFrame:
    if not trades:
        return _empty_trade_frame()
    return pd.DataFrame([asdict(trade) for trade in trades])


def _event_rows_to_frame(events: list[PaperEvent]) -> pd.DataFrame:
    if not events:
        return _empty_event_frame()
    return pd.DataFrame([asdict(event) for event in events])


def _equity_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty_equity_frame()
    return pd.DataFrame(rows)


def _compute_notional(equity_before: float, config: PaperSimConfig) -> float:
    exposure_cap = equity_before * config.max_exposure_fraction
    if config.fixed_notional_usd is not None:
        return float(min(config.fixed_notional_usd, exposure_cap))
    nominal = equity_before * config.exposure_fraction
    return float(min(nominal, exposure_cap))


def _build_events_for_trade(
    *,
    config: PaperSimConfig,
    trade: PaperTrade,
) -> list[PaperEvent]:
    signal_row = PaperEvent(
        event_type="SIGNAL_DETECTED",
        event_time=trade.signal_time,
        symbol=config.symbol,
        strategy_id=config.strategy_id,
        bar_size=config.bar_size,
        horizon=config.horizon_bars,
        signal_index=trade.signal_index,
        entry_index=trade.entry_index,
        exit_index=trade.exit_index,
        price=trade.entry_price_raw,
        quantity=trade.quantity,
        notional_usd=trade.notional_usd,
        fee_bps=trade.fees_bps,
        slippage_bps=trade.slippage_bps,
        gross_return_bps=trade.gross_return_bps,
        net_return_bps=trade.net_return_bps,
        equity_after=trade.equity_before,
        reason="signal",
    )
    order_created = PaperEvent(
        event_type="PAPER_ORDER_CREATED",
        event_time=trade.signal_time,
        symbol=config.symbol,
        strategy_id=config.strategy_id,
        bar_size=config.bar_size,
        horizon=config.horizon_bars,
        signal_index=trade.signal_index,
        entry_index=trade.entry_index,
        exit_index=trade.exit_index,
        price=trade.entry_price_raw,
        quantity=trade.quantity,
        notional_usd=trade.notional_usd,
        fee_bps=trade.fees_bps,
        slippage_bps=trade.slippage_bps,
        gross_return_bps=trade.gross_return_bps,
        net_return_bps=trade.net_return_bps,
        equity_after=trade.equity_before,
        reason="order_created",
    )
    order_filled = PaperEvent(
        event_type="PAPER_ORDER_FILLED",
        event_time=trade.entry_time,
        symbol=config.symbol,
        strategy_id=config.strategy_id,
        bar_size=config.bar_size,
        horizon=config.horizon_bars,
        signal_index=trade.signal_index,
        entry_index=trade.entry_index,
        exit_index=trade.exit_index,
        price=trade.entry_fill_price,
        quantity=trade.quantity,
        notional_usd=trade.notional_usd,
        fee_bps=trade.fees_bps,
        slippage_bps=trade.slippage_bps,
        gross_return_bps=trade.gross_return_bps,
        net_return_bps=trade.net_return_bps,
        equity_after=trade.equity_before,
        reason="filled",
    )
    position_opened = PaperEvent(
        event_type="PAPER_POSITION_OPENED",
        event_time=trade.entry_time,
        symbol=config.symbol,
        strategy_id=config.strategy_id,
        bar_size=config.bar_size,
        horizon=config.horizon_bars,
        signal_index=trade.signal_index,
        entry_index=trade.entry_index,
        exit_index=trade.exit_index,
        price=trade.entry_fill_price,
        quantity=trade.quantity,
        notional_usd=trade.notional_usd,
        fee_bps=trade.fees_bps,
        slippage_bps=trade.slippage_bps,
        gross_return_bps=trade.gross_return_bps,
        net_return_bps=trade.net_return_bps,
        equity_after=trade.equity_before,
        reason="position_opened",
    )
    position_closed = PaperEvent(
        event_type="PAPER_POSITION_CLOSED",
        event_time=trade.exit_time,
        symbol=config.symbol,
        strategy_id=config.strategy_id,
        bar_size=config.bar_size,
        horizon=config.horizon_bars,
        signal_index=trade.signal_index,
        entry_index=trade.entry_index,
        exit_index=trade.exit_index,
        price=trade.exit_fill_price,
        quantity=trade.quantity,
        notional_usd=trade.notional_usd,
        fee_bps=trade.fees_bps,
        slippage_bps=trade.slippage_bps,
        gross_return_bps=trade.gross_return_bps,
        net_return_bps=trade.net_return_bps,
        equity_after=trade.equity_after,
        reason="position_closed",
    )
    equity_updated = PaperEvent(
        event_type="PAPER_EQUITY_UPDATED",
        event_time=trade.exit_time,
        symbol=config.symbol,
        strategy_id=config.strategy_id,
        bar_size=config.bar_size,
        horizon=config.horizon_bars,
        signal_index=trade.signal_index,
        entry_index=trade.entry_index,
        exit_index=trade.exit_index,
        price=trade.exit_fill_price,
        quantity=trade.quantity,
        notional_usd=trade.notional_usd,
        fee_bps=trade.fees_bps,
        slippage_bps=trade.slippage_bps,
        gross_return_bps=trade.gross_return_bps,
        net_return_bps=trade.net_return_bps,
        equity_after=trade.equity_after,
        reason="equity_updated",
    )
    return [signal_row, order_created, order_filled, position_opened, position_closed, equity_updated]


def _summary_from_result(
    *,
    config: PaperSimConfig,
    replay_summary: dict[str, Any],
    replay_yearly: pd.DataFrame,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
) -> dict[str, Any]:
    if equity_curve.empty:
        ending_equity = float(config.starting_equity_usd)
        total_return_pct = 0.0
        max_drawdown_pct = 0.0
        max_drawdown_bps = 0.0
    else:
        ending_equity = float(equity_curve.iloc[-1]["equity"])
        total_return_pct = float((ending_equity - config.starting_equity_usd) / config.starting_equity_usd * 100.0)
        max_drawdown_pct = float(equity_curve["drawdown_pct"].max())
        max_drawdown_bps = float(max_drawdown_pct * 100.0)

    positive_year_count = int((replay_yearly["net_expectancy_bps"] > 0).sum()) if not replay_yearly.empty else 0
    if replay_yearly.empty:
        worst_year: int | str = "n/a"
    else:
        worst_year = int(replay_yearly.sort_values(["net_expectancy_bps", "year"]).iloc[0]["year"])

    return {
        "trade_count": int(replay_summary["trade_count"]),
        "starting_equity_usd": float(config.starting_equity_usd),
        "ending_equity_usd": ending_equity,
        "total_return_pct": total_return_pct,
        "gross_expectancy_bps": float(replay_summary["gross_expectancy_bps"]),
        "net_expectancy_bps": float(replay_summary["net_expectancy_bps"]),
        "win_rate": float(replay_summary["win_rate"]),
        "profit_factor": float(replay_summary["profit_factor"]),
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_bps": max_drawdown_bps,
        "positive_year_count": positive_year_count,
        "worst_year": worst_year,
        "fee_bps_per_side": float(config.fee_bps_per_side),
        "slippage_bps_per_side": float(config.slippage_bps_per_side),
        "round_trip_cost_bps": float(2.0 * config.fee_bps_per_side + 2.0 * config.slippage_bps_per_side),
        "exposure_fraction": float(config.exposure_fraction),
        "fixed_notional_usd": None if config.fixed_notional_usd is None else float(config.fixed_notional_usd),
        "production_path_touched": False,
        "average_holding_time": float(trades["holding_bars"].mean()) if not trades.empty else 0.0,
        "worst_trade": float(trades["net_return_bps"].min()) if not trades.empty else 0.0,
        "best_trade": float(trades["net_return_bps"].max()) if not trades.empty else 0.0,
        "worst_day": "n/a",
        "worst_month": "n/a",
        "exposure_time": float(trades["holding_bars"].sum()) if not trades.empty else 0.0,
    }


def run_c_exhaustion_paper_sim(
    df: pl.DataFrame | pd.DataFrame,
    config: PaperSimConfig | None = None,
) -> PaperSimResult:
    """Run the pure paper simulation for C_ExhaustionFade.

    The module intentionally reuses the canonical replay timing and signal path.
    """
    config = config or PaperSimConfig()
    if config.max_open_positions != 1:
        raise ValueError("Phase P2 only supports max_open_positions=1")

    replay_result = replay_c_exhaustionfade(
        df,
        horizon_bars=config.horizon_bars,
        round_trip_cost_bps=2.0 * config.fee_bps_per_side + 2.0 * config.slippage_bps_per_side,
        bar_size=config.bar_size,
    )

    trades: list[PaperTrade] = []
    events: list[PaperEvent] = []
    equity_rows: list[dict[str, Any]] = []
    equity_before = float(config.starting_equity_usd)
    equity_peak = float(config.starting_equity_usd)

    for trade_id, row in enumerate(replay_result.trades.to_dict(orient="records"), start=1):
        entry_price_raw = float(row["entry_price"])
        exit_price_raw = float(row["exit_price"])
        entry_fill_price = entry_price_raw * (1.0 + config.slippage_bps_per_side / 10_000.0)
        exit_fill_price = exit_price_raw * (1.0 - config.slippage_bps_per_side / 10_000.0)
        gross_return_bps = float(row["gross_return_bps"])
        fees_bps = float(2.0 * config.fee_bps_per_side)
        slippage_bps = float(2.0 * config.slippage_bps_per_side)
        net_return_bps = float(gross_return_bps - fees_bps - slippage_bps)
        notional_usd = _compute_notional(equity_before, config)
        quantity = float(notional_usd / entry_fill_price) if entry_fill_price else 0.0
        pnl_usd = float(notional_usd * net_return_bps / 10_000.0)
        equity_after = float(equity_before + pnl_usd)

        paper_trade = PaperTrade(
            trade_id=trade_id,
            signal_time=row["signal_time"],
            entry_time=row["entry_time"],
            exit_time=row["exit_time"],
            signal_index=int(row["signal_index"]),
            entry_index=int(row["entry_index"]),
            exit_index=int(row["exit_index"]),
            entry_price_raw=entry_price_raw,
            exit_price_raw=exit_price_raw,
            entry_fill_price=float(entry_fill_price),
            exit_fill_price=float(exit_fill_price),
            quantity=quantity,
            notional_usd=float(notional_usd),
            gross_return_bps=gross_return_bps,
            fees_bps=fees_bps,
            slippage_bps=slippage_bps,
            net_return_bps=net_return_bps,
            pnl_usd=pnl_usd,
            equity_before=equity_before,
            equity_after=equity_after,
            holding_bars=int(row["holding_bars"]),
            year=int(row["year"]),
        )
        trades.append(paper_trade)
        events.extend(_build_events_for_trade(config=config, trade=paper_trade))
        drawdown_pct = 0.0 if equity_peak <= 0 else max(0.0, (equity_peak - equity_after) / equity_peak * 100.0)
        equity_rows.append(
            {
                "timestamp": paper_trade.exit_time,
                "equity": equity_after,
                "drawdown_pct": drawdown_pct,
                "open_position_count": 0,
                "cash_usd": equity_after,
                "exposure_usd": 0.0,
            }
        )
        equity_before = equity_after
        equity_peak = max(equity_peak, equity_after)

    trades_df = _trade_rows_to_frame(trades)
    events_df = _event_rows_to_frame(events)
    equity_df = _equity_rows_to_frame(equity_rows)
    summary = _summary_from_result(
        config=config,
        replay_summary=replay_result.summary,
        replay_yearly=replay_result.yearly,
        trades=trades_df,
        equity_curve=equity_df,
    )
    return PaperSimResult(
        config=config,
        trades=trades_df,
        equity_curve=equity_df,
        events=events_df,
        summary=summary,
    )
