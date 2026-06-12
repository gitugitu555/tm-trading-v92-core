#!/usr/bin/env python3
"""CLI wrapper for the V9.2 L2 shadow backtest."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from research.v92_l2_backtest import (
    default_data_root,
    parse_date,
    run_l2_shadow_backtest,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--horizon-seconds", type=int, default=60)
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument(
        "--cost-bps",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 8, 12],
        help="Round-trip cost ladder in basis points.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/v92_alpha_rebuild/l2_backtest"))
    parser.add_argument("--strategy-label", default="l2_shadow_backtest")
    parser.add_argument("--repo-commit", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    commit = args.repo_commit
    if commit is None:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        )
    report = run_l2_shadow_backtest(
        data_root=args.data_root,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        horizon_seconds=args.horizon_seconds,
        cost_ladder=tuple(args.cost_bps),
        limit_days=args.limit_days,
        max_files=args.max_files,
        max_snapshots=args.max_snapshots,
        strategy_label=args.strategy_label,
        repo_commit=commit,
    )
    summary_path, markdown_path = write_report(
        report,
        output_dir=args.output_dir,
        cost_ladder=tuple(args.cost_bps),
    )
    print(summary_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
