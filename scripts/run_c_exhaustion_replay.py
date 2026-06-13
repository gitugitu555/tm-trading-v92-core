#!/usr/bin/env python3
"""Research-only CLI for the V9.2 C_ExhaustionFade replay."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replays.c_exhaustion_replay import (
    add_v92_regime_labels,
    normalize_v92_bar_timestamps,
    replay_c_exhaustionfade,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bar-dir", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--bar-size", type=int, default=750)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--cost-bps", type=float, default=12.0)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/c_exhaustion_replay"))
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--starting-equity", type=float, default=100000.0)
    parser.add_argument("--write-json", action="store_true")
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    return parser.parse_args(argv)


def _load_bar_dir(bar_dir: Path) -> pl.DataFrame:
    files = sorted(bar_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {bar_dir}")
    return pl.concat([pl.scan_parquet(path) for path in files]).collect().sort("open_time")


def _compute_equity_metrics(
    trades: pd.DataFrame,
    yearly: pd.DataFrame,
    *,
    exposure: float,
    starting_equity: float,
) -> dict[str, float | int | str | bool]:
    if trades.empty:
        return {
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "positive_year_count": 0,
            "worst_year": "n/a",
        }

    net_bps = trades["net_return_bps"].astype(float) * exposure
    if len(net_bps) > 1 and net_bps.std(ddof=1) > 0:
        sharpe = float((net_bps.mean() / net_bps.std(ddof=1)) * math.sqrt(len(net_bps)))
    else:
        sharpe = 0.0

    equity = starting_equity + starting_equity * net_bps.cumsum() / 10_000.0
    running_max = equity.cummax()
    drawdown_pct = ((equity - running_max) / running_max) * 100.0
    max_drawdown_pct = float(abs(drawdown_pct.min()))

    positive_year_count = int((yearly["net_expectancy_bps"] > 0).sum()) if not yearly.empty else 0
    if yearly.empty:
        worst_year = "n/a"
    else:
        worst_row = yearly.sort_values(["net_expectancy_bps", "year"]).iloc[0]
        worst_year = int(worst_row["year"])

    return {
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "positive_year_count": positive_year_count,
        "worst_year": worst_year,
    }


def _build_report(
    *,
    symbol: str,
    bar_size: int,
    horizon: int,
    cost_bps: float,
    exposure: float,
    starting_equity: float,
    result,
) -> dict:
    yearly_records = result.yearly.to_dict(orient="records")
    metrics = _compute_equity_metrics(
        result.trades,
        result.yearly,
        exposure=exposure,
        starting_equity=starting_equity,
    )
    return {
        "metadata": {
            "branch": "C_ExhaustionFade",
            "side": "long_only",
            "symbol": symbol,
            "bar_size": bar_size,
            "horizon": horizon,
            "cost_bps": cost_bps,
            "exposure": exposure,
            "starting_equity": starting_equity,
            "production_path_touched": False,
        },
        "summary": {
            "trade_count": int(result.summary["trade_count"]),
            "net_expectancy_bps": float(result.summary["net_expectancy_bps"]),
            "win_rate": float(result.summary["win_rate"]),
            "profit_factor": float(result.summary["profit_factor"]),
            "sharpe": float(metrics["sharpe"]),
            "max_drawdown_pct": float(metrics["max_drawdown_pct"]),
            "positive_year_count": int(metrics["positive_year_count"]),
            "worst_year": metrics["worst_year"],
            "production_path_touched": False,
        },
        "yearly": yearly_records,
        "trades": result.trades.to_dict(orient="records"),
    }


def _write_outputs(report: dict, output_dir: Path, args: argparse.Namespace, result) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if args.write_json:
        json_path = output_dir / "c_exhaustion_replay_report.json"
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        written.append(json_path)

    if args.write_csv:
        trades_path = output_dir / "c_exhaustion_replay_trades.csv"
        result.trades.to_csv(trades_path, index=False)
        written.append(trades_path)

    if args.write_summary:
        summary_path = output_dir / "c_exhaustion_replay_summary.txt"
        lines = [
            "branch=C_ExhaustionFade",
            "side=long_only",
            f"symbol={args.symbol}",
            f"bar_size={args.bar_size}",
            f"horizon={args.horizon}",
            f"cost_bps={args.cost_bps}",
            f"trade_count={report['summary']['trade_count']}",
            f"net_expectancy_bps={report['summary']['net_expectancy_bps']:.6f}",
            f"win_rate={report['summary']['win_rate']:.6f}",
            f"profit_factor={report['summary']['profit_factor']:.6f}",
            f"sharpe={report['summary']['sharpe']:.6f}",
            f"max_drawdown_pct={report['summary']['max_drawdown_pct']:.6f}",
            f"positive_year_count={report['summary']['positive_year_count']}",
            f"worst_year={report['summary']['worst_year']}",
            "production_path_touched=false",
        ]
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(summary_path)

    return written


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        bars = _load_bar_dir(args.bar_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    bars = normalize_v92_bar_timestamps(bars)
    bars = add_v92_regime_labels(bars)

    result = replay_c_exhaustionfade(
        bars,
        horizon_bars=args.horizon,
        round_trip_cost_bps=args.cost_bps,
        bar_size=args.bar_size,
    )

    report = _build_report(
        symbol=args.symbol,
        bar_size=args.bar_size,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        exposure=args.exposure,
        starting_equity=args.starting_equity,
        result=result,
    )
    _write_outputs(report, args.output_dir, args, result)

    summary = report["summary"]
    print(
        "branch=C_ExhaustionFade "
        f"side=long_only "
        f"bar_size={args.bar_size} "
        f"horizon={args.horizon} "
        f"cost_bps={args.cost_bps} "
        f"trade_count={summary['trade_count']} "
        f"net_expectancy_bps={summary['net_expectancy_bps']:.6f} "
        f"win_rate={summary['win_rate']:.6f} "
        f"profit_factor={summary['profit_factor']:.6f} "
        f"sharpe={summary['sharpe']:.6f} "
        f"max_drawdown_pct={summary['max_drawdown_pct']:.6f} "
        f"positive_year_count={summary['positive_year_count']} "
        f"worst_year={summary['worst_year']} "
        "production_path_touched=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

