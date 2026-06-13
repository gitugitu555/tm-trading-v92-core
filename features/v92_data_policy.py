"""
Centralized V9.2 data-selection and timestamp policy helpers.

These helpers are the single source of truth for archive discovery,
Tier-2 bar selection, timestamp normalization, and optional OFI enrichment.
Builders, dry-runs, evaluators, and tests must import these helpers rather
than reimplementing filename parsing, glob policy, or ms/us timestamp logic.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

AGGTRADE_MONTHLY_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)-aggTrades-(?P<year>\d{4})-(?P<month>\d{2})\.zip$")
AGGTRADE_DAILY_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)-aggTrades-(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\.zip$")
TIER2_ALL_RE_TEMPLATE = r"^{symbol}_tier2_500btc_ALL\.parquet$"
TIER2_SHARD_RE_TEMPLATE = r"^{symbol}_tier2_500btc_(?!ALL$).+\.parquet$"
MS_US_EPOCH_THRESHOLD = 100_000_000_000_000  # > 1e14 means microseconds for Binance-era data.


def _period_key(year: str, month: str) -> str:
    return f"{year}-{month}"


def discover_aggtrade_archives(
    search_dir: Path,
    symbol: str,
    overlap_policy: str = "monthly_wins",
) -> list[Path]:
    """
    Discover Binance aggTrade ZIP archives with an explicit overlap policy.

    Supported policies:
    - monthly_wins: if YYYY-MM monthly exists, exclude all YYYY-MM-DD daily files.
    - daily_wins: if any YYYY-MM-DD daily file exists, exclude the YYYY-MM monthly file.
    - reject_overlap: raise ValueError when monthly and daily files coexist for a month.

    Unsupported files matching the symbol aggTrades ZIP prefix raise ValueError so the
    builder cannot silently process ambiguous archives such as *_1m.zip variants.
    """
    search_dir = Path(search_dir)
    if overlap_policy not in {"monthly_wins", "daily_wins", "reject_overlap"}:
        raise ValueError(f"Unsupported overlap_policy={overlap_policy!r}")

    monthly: dict[str, Path] = {}
    daily: dict[str, list[Path]] = {}
    unsupported: list[Path] = []

    prefix = f"{symbol}-aggTrades-"
    for path in sorted(search_dir.glob(f"{prefix}*.zip")):
        name = path.name
        monthly_match = AGGTRADE_MONTHLY_RE.match(name)
        daily_match = AGGTRADE_DAILY_RE.match(name)

        if monthly_match and monthly_match.group("symbol") == symbol:
            period = _period_key(monthly_match.group("year"), monthly_match.group("month"))
            monthly[period] = path
        elif daily_match and daily_match.group("symbol") == symbol:
            period = _period_key(daily_match.group("year"), daily_match.group("month"))
            daily.setdefault(period, []).append(path)
        else:
            unsupported.append(path)

    if unsupported:
        names = ", ".join(p.name for p in unsupported)
        raise ValueError(f"Unsupported aggTrade archive filenames: {names}")

    overlapping = sorted(set(monthly).intersection(daily))
    if overlapping and overlap_policy == "reject_overlap":
        raise ValueError(f"Overlapping monthly/daily aggTrade archives: {', '.join(overlapping)}")

    selected: list[Path] = []
    all_periods = sorted(set(monthly).union(daily))
    for period in all_periods:
        has_monthly = period in monthly
        has_daily = period in daily
        if has_monthly and has_daily:
            if overlap_policy == "monthly_wins":
                selected.append(monthly[period])
            elif overlap_policy == "daily_wins":
                selected.extend(sorted(daily[period]))
            else:  # Defensive; reject_overlap handled above.
                raise ValueError(f"Unhandled overlap for {period}")
        elif has_monthly:
            selected.append(monthly[period])
        elif has_daily:
            selected.extend(sorted(daily[period]))

    return sorted(selected)


def discover_tier2_bar_files(
    tier2_dir: Path,
    symbol: str = "BTCUSDT",
    all_mode: str = "all_wins",
) -> list[Path]:
    """
    Discover V9.2 Tier-2 bar parquet files with explicit _ALL vs shard policy.

    Supported modes:
    - all_wins: if *_ALL.parquet exists, return only _ALL.
    - shards_only: ignore _ALL and return period shards only.
    - reject_mixed: raise ValueError when _ALL and shards coexist.
    """
    tier2_dir = Path(tier2_dir)
    if all_mode not in {"all_wins", "shards_only", "reject_mixed"}:
        raise ValueError(f"Unsupported all_mode={all_mode!r}")

    all_re = re.compile(TIER2_ALL_RE_TEMPLATE.format(symbol=re.escape(symbol)))
    shard_re = re.compile(TIER2_SHARD_RE_TEMPLATE.format(symbol=re.escape(symbol)))

    candidates = sorted(tier2_dir.glob(f"{symbol}_tier2_500btc_*.parquet"))
    all_files = [p for p in candidates if all_re.match(p.name)]
    shard_files = [p for p in candidates if shard_re.match(p.name) and not all_re.match(p.name)]

    if len(all_files) > 1:
        raise ValueError(f"Multiple _ALL parquet files found for {symbol}")

    has_all = bool(all_files)
    has_shards = bool(shard_files)

    if has_all and has_shards and all_mode == "reject_mixed":
        raise ValueError(f"Mixed _ALL and shard Tier-2 files found for {symbol}")
    if has_all and all_mode == "all_wins":
        return all_files
    if all_mode == "shards_only":
        return shard_files
    return all_files if has_all else shard_files


def epoch_to_datetime_expr(column: str) -> pl.Expr:
    """
    Convert Binance-era integer epochs to Polars Datetime(ns).

    Values above MS_US_EPOCH_THRESHOLD are interpreted as microseconds;
    smaller values are interpreted as milliseconds.
    """
    epoch_col = pl.col(column).cast(pl.Int64)
    return (
        pl.when(epoch_col > MS_US_EPOCH_THRESHOLD)
        .then(pl.from_epoch(epoch_col, time_unit="us"))
        .otherwise(pl.from_epoch(epoch_col, time_unit="ms"))
        .cast(pl.Datetime("ns"))
    )


def epoch_to_ns_value(value: int | float) -> int:
    """Normalize one Binance-era ms/us epoch scalar into nanoseconds."""
    value_int = int(value)
    if value_int > MS_US_EPOCH_THRESHOLD:
        return value_int * 1_000
    return value_int * 1_000_000


def _ensure_bar_datetimes(bars: pl.DataFrame) -> pl.DataFrame:
    out = bars
    if "datetime_open" not in out.columns:
        out = out.with_columns(epoch_to_datetime_expr("open_time").alias("datetime_open"))
    if "datetime_close" not in out.columns:
        out = out.with_columns(epoch_to_datetime_expr("close_time").alias("datetime_close"))
    return out


def join_ofi_to_bars_preserve_coverage(
    bars: pl.DataFrame,
    ofi: pl.DataFrame | None,
) -> pl.DataFrame:
    """
    Join optional OFI coverage to bars without dropping any bar rows.

    Output guarantees:
    - row count equals input bars row count;
    - bar_ofi exists and is null outside complete OFI coverage;
    - has_ofi_coverage marks bars fully contained inside OFI coverage.
    """
    original_height = bars.height
    bars = _ensure_bar_datetimes(bars).sort("datetime_open")

    if ofi is None or ofi.height == 0:
        result = bars.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("bar_ofi"),
            pl.lit(False).alias("has_ofi_coverage"),
        )
        assert result.height == original_height
        return result

    ofi = ofi.sort("datetime").with_columns(pl.col("datetime").cast(pl.Datetime("ns")))
    ofi_min = ofi.select(pl.col("datetime").min()).item()
    ofi_max = ofi.select(pl.col("datetime").max()).item()

    ofi_cumulative = ofi.with_columns(pl.col("ofi").cum_sum().alias("cumulative_ofi"))

    joined_open = bars.join_asof(
        ofi_cumulative.select(["datetime", "cumulative_ofi"]),
        left_on="datetime_open",
        right_on="datetime",
        strategy="backward",
    ).rename({"cumulative_ofi": "ofi_at_open"}).drop("datetime")

    joined_open = joined_open.sort("datetime_close")
    joined = joined_open.join_asof(
        ofi_cumulative.select(["datetime", "cumulative_ofi"]),
        left_on="datetime_close",
        right_on="datetime",
        strategy="backward",
    ).rename({"cumulative_ofi": "ofi_at_close"}).drop("datetime")

    coverage_expr = (
        (pl.col("datetime_open") >= pl.lit(ofi_min))
        & (pl.col("datetime_close") <= pl.lit(ofi_max))
        & pl.col("ofi_at_open").is_not_null()
        & pl.col("ofi_at_close").is_not_null()
    )

    result = joined.with_columns(coverage_expr.alias("has_ofi_coverage")).with_columns(
        pl.when(pl.col("has_ofi_coverage"))
        .then(pl.col("ofi_at_close") - pl.col("ofi_at_open"))
        .otherwise(None)
        .alias("bar_ofi")
    )

    assert result.height == original_height
    return result.sort("datetime_open")
