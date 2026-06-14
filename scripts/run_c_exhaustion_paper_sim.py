#!/usr/bin/env python3
"""CLI wrapper for the pure V9.2 C_ExhaustionFade paper simulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.c_exhaustion_paper_sim import PaperSimConfig, run_c_exhaustion_paper_sim

_PRODUCTION_OUTPUT_BLOCKLIST = (
    "data",
    "data/hft",
    "data/hft/tier2",
    "data/raw",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bar-dir", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--bar-size", type=int, default=750, choices=[500, 750, 1000])
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--starting-equity", type=float, default=100000.0)
    parser.add_argument("--exposure-fraction", type=float, default=1.0)
    parser.add_argument("--fixed-notional-usd", type=float, default=None)
    parser.add_argument("--fee-bps-per-side", type=float, default=3.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=3.0)
    parser.add_argument("--max-exposure-fraction", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/paper_sim/c_exhaustion_750_h36"))
    parser.add_argument("--write-events", action="store_true")
    parser.add_argument("--write-trades", action="store_true")
    parser.add_argument("--write-equity", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    return parser.parse_args(argv)


def _is_under(candidate: Path, parent: Path) -> bool:
    candidate = candidate.resolve()
    parent = parent.resolve()
    return candidate == parent or candidate.is_relative_to(parent)


def _refuse_output_path(output_dir: Path) -> str | None:
    resolved_output = output_dir.expanduser().resolve()
    repo_root = ROOT.resolve()
    for relative in _PRODUCTION_OUTPUT_BLOCKLIST:
        blocked_root = (repo_root / relative).resolve()
        if _is_under(resolved_output, blocked_root):
            return f"Refused production/cache output path: {resolved_output}"
    return None


def _load_bar_dir(bar_dir: Path, symbol: str, bar_size: int) -> tuple[pl.DataFrame, list[Path]]:
    files = sorted(bar_dir.glob(f"{symbol}_tier2_{bar_size}btc_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for {symbol} {bar_size} BTC in {bar_dir}")
    frame = pl.concat([pl.scan_parquet(path) for path in files]).collect().sort("open_time")
    return frame, files


def _coverage_value(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _build_summary_json(
    *,
    args: argparse.Namespace,
    paper_result,
    bar_dir: Path,
    output_dir: Path,
    parquet_file_count: int,
    coverage_start: Any,
    coverage_end: Any,
) -> dict[str, Any]:
    summary = dict(paper_result.summary)
    summary.update(
        {
            "strategy_id": paper_result.config.strategy_id,
            "symbol": args.symbol,
            "bar_size": args.bar_size,
            "horizon": args.horizon,
            "starting_equity_usd": float(args.starting_equity),
            "exposure_fraction": float(args.exposure_fraction),
            "fixed_notional_usd": None if args.fixed_notional_usd is None else float(args.fixed_notional_usd),
            "fee_bps_per_side": float(args.fee_bps_per_side),
            "slippage_bps_per_side": float(args.slippage_bps_per_side),
            "round_trip_cost_bps": float(2.0 * args.fee_bps_per_side + 2.0 * args.slippage_bps_per_side),
            "bar_dir": str(bar_dir),
            "output_dir": str(output_dir),
            "parquet_file_count": int(parquet_file_count),
            "coverage_start": _coverage_value(coverage_start),
            "coverage_end": _coverage_value(coverage_end),
        }
    )
    return summary


def _build_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# C_ExhaustionFade Paper Simulation Summary",
        "",
        "## Configuration",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| strategy_id | {summary['strategy_id']} |",
        f"| symbol | {summary['symbol']} |",
        f"| bar_size | {summary['bar_size']} |",
        f"| horizon | {summary['horizon']} |",
        f"| starting_equity_usd | {summary['starting_equity_usd']} |",
        f"| exposure_fraction | {summary['exposure_fraction']} |",
        f"| fixed_notional_usd | {summary['fixed_notional_usd']} |",
        f"| fee_bps_per_side | {summary['fee_bps_per_side']} |",
        f"| slippage_bps_per_side | {summary['slippage_bps_per_side']} |",
        f"| round_trip_cost_bps | {summary['round_trip_cost_bps']} |",
        "",
        "## Input Data",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| bar_dir | {summary['bar_dir']} |",
        f"| output_dir | {summary['output_dir']} |",
        f"| parquet_file_count | {summary['parquet_file_count']} |",
        f"| coverage_start | {summary['coverage_start']} |",
        f"| coverage_end | {summary['coverage_end']} |",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| trade_count | {summary['trade_count']} |",
        f"| ending_equity_usd | {summary['ending_equity_usd']} |",
        f"| total_return_pct | {summary['total_return_pct']} |",
        f"| gross_expectancy_bps | {summary['gross_expectancy_bps']} |",
        f"| net_expectancy_bps | {summary['net_expectancy_bps']} |",
        f"| win_rate | {summary['win_rate']} |",
        f"| profit_factor | {summary['profit_factor']} |",
        f"| max_drawdown_pct | {summary['max_drawdown_pct']} |",
        f"| positive_year_count | {summary['positive_year_count']} |",
        f"| worst_year | {summary['worst_year']} |",
        "",
        "## Validation Notes",
        "",
        "- Uses the pure in-memory paper simulation module.",
        "- Reuses the canonical C replay signal and timing path.",
        "- Refuses production/cache output paths.",
        "- Writes only optional report files under `reports/paper_sim/`.",
        "",
        "## Non-Live Warning",
        "",
        "This is a paper simulation report only. It does not place orders and does not use exchange credentials.",
        "",
    ]
    return "\n".join(lines)


def _write_outputs(
    *,
    summary: dict[str, Any],
    paper_result,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    written: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_events:
        events_path = output_dir / "c_exhaustion_paper_events.jsonl"
        with events_path.open("w", encoding="utf-8") as fh:
            for record in paper_result.events.to_dict(orient="records"):
                fh.write(json.dumps(record, default=str))
                fh.write("\n")
        written.append(events_path)

    if args.write_trades:
        trades_path = output_dir / "c_exhaustion_paper_trades.csv"
        paper_result.trades.to_csv(trades_path, index=False)
        written.append(trades_path)

    if args.write_equity:
        equity_path = output_dir / "c_exhaustion_paper_equity.csv"
        paper_result.equity_curve.to_csv(equity_path, index=False)
        written.append(equity_path)

    if args.write_summary:
        summary_json_path = output_dir / "c_exhaustion_paper_summary.json"
        summary_md_path = output_dir / "c_exhaustion_paper_summary.md"
        summary_json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        summary_md_path.write_text(_build_summary_markdown(summary), encoding="utf-8")
        written.extend([summary_json_path, summary_md_path])

    return written


def _print_compact_summary(summary: dict[str, Any]) -> None:
    print(
        "strategy=C_ExhaustionFade "
        f"symbol={summary['symbol']} "
        f"bar_size={summary['bar_size']} "
        f"horizon={summary['horizon']} "
        f"starting_equity_usd={summary['starting_equity_usd']} "
        f"ending_equity_usd={summary['ending_equity_usd']} "
        f"total_return_pct={summary['total_return_pct']:.6f} "
        f"trade_count={summary['trade_count']} "
        f"net_expectancy_bps={summary['net_expectancy_bps']:.6f} "
        f"win_rate={summary['win_rate']:.6f} "
        f"profit_factor={summary['profit_factor']:.6f} "
        f"max_drawdown_pct={summary['max_drawdown_pct']:.6f} "
        f"positive_year_count={summary['positive_year_count']} "
        f"worst_year={summary['worst_year']} "
        f"production_path_touched={str(summary['production_path_touched']).lower()}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    refused = _refuse_output_path(args.output_dir)
    if refused is not None:
        print(refused, file=sys.stderr)
        return 2

    try:
        bars, files = _load_bar_dir(args.bar_dir, args.symbol, args.bar_size)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    config = PaperSimConfig(
        strategy_id="C_ExhaustionFade",
        symbol=args.symbol,
        bar_size=args.bar_size,
        horizon_bars=args.horizon,
        starting_equity_usd=args.starting_equity,
        exposure_fraction=args.exposure_fraction,
        fixed_notional_usd=args.fixed_notional_usd,
        fee_bps_per_side=args.fee_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
        max_exposure_fraction=args.max_exposure_fraction,
    )
    paper_result = run_c_exhaustion_paper_sim(bars, config=config)
    summary = _build_summary_json(
        args=args,
        paper_result=paper_result,
        bar_dir=args.bar_dir,
        output_dir=args.output_dir,
        parquet_file_count=len(files),
        coverage_start=bars["open_time"].min(),
        coverage_end=bars["open_time"].max(),
    )

    if args.write_events or args.write_trades or args.write_equity or args.write_summary:
        _write_outputs(summary=summary, paper_result=paper_result, output_dir=args.output_dir, args=args)

    _print_compact_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
