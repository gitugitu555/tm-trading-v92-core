"""V9.2 Canonical Regime Classifier — single source of truth.

Safety contract
---------------
* All rolling statistics are shifted by one bar (.shift(1)) so that the
  current bar's own values never influence its own rolling thresholds. The
  current bar's completed OHLCV may still be used in the close-of-bar regime
  decision, which is safe because entries occur on the next bar.
* Classification order is fixed: TREND_BUILDUP → ABSORPTION → EXHAUSTED → NOISE.
  This matches the replay fallback order (which was more correct) and is now
  the only order used everywhere.
* ADR stretch uses rolling_min of low prices (not rolling_min of close), which
  is a tighter, more correct measure of range location.
* The `delta_efficiency` condition inside ABSORPTION gates out bars where price
  movement is disproportionate to order-flow, reducing noise classification.

This module is imported directly by replays, scripts, and tests.
There is no fallback copy. If this import fails, the caller must fix the
import path — not duplicate the logic.
"""

from __future__ import annotations

import polars as pl

from features.v92_data_policy import epoch_to_datetime_expr


def add_regime_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Attach a past-only 'regime' string column to every volume bar.

    Expected input columns: open_time, close, high, low, open, volume
    Optional input columns: volume_delta (used for delta_efficiency gate)

    Returns the DataFrame with a new 'regime' column added.
    Intermediate columns used for computation are dropped before returning.
    """
    # ── Timestamp normalization ──────────────────────────────────────────────
    if "datetime" not in df.columns:
        if "open_time" in df.columns and df["open_time"].dtype in (
            pl.Int64, pl.Float64, pl.UInt64, pl.UInt32, pl.Int32
        ):
            df = df.with_columns(epoch_to_datetime_expr("open_time").alias("datetime"))
        elif "start_ts_ns" in df.columns:
            df = df.with_columns(
                pl.from_epoch("start_ts_ns", time_unit="ns").alias("datetime")
            )

    df = df.sort("datetime")

    # ── Per-bar primitives (no leakage — only current-bar OHLCV) ────────────
    df = df.with_columns(
        [
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("_return"),
            (pl.col("high") - pl.col("low")).alias("bar_range"),
            (pl.col("close") - pl.col("open")).abs().alias("body_size"),
        ]
    )

    # Delta efficiency: price move per unit of signed order flow.
    # Falls back to a constant if volume_delta is absent.
    if "volume_delta" in df.columns:
        df = df.with_columns(
            (
                pl.col("body_size")
                / pl.col("volume_delta").abs().clip(lower_bound=1e-9)
            ).alias("_delta_efficiency")
        )
    else:
        df = df.with_columns(pl.lit(1.0).alias("_delta_efficiency"))

    # ── Rolling statistics — ALL shifted by 1 to avoid same-bar leakage ─────
    df = df.with_columns(
        [
            # Realized volatility: ~1 day of 750-BTC bars
            pl.col("_return")
            .rolling_std(window_size=500)
            .shift(1)
            .alias("rv_1d"),

            # Volume 80th percentile: ~7 days of bars
            pl.col("volume")
            .rolling_quantile(quantile=0.80, interpolation="nearest", window_size=3500)
            .shift(1)
            .alias("vol_80th_pct"),

            # Delta efficiency median: ~200-bar lookback
            pl.col("_delta_efficiency")
            .rolling_median(window_size=200)
            .shift(1)
            .alias("_de_median"),

            # Range mean for ADR denominator: ~14 days of bars
            (pl.col("high") - pl.col("low"))
            .rolling_mean(window_size=7000)
            .shift(1)
            .alias("_rolling_mean_range"),

            # Rolling min of low prices for ADR numerator: ~1 day of bars
            pl.col("low")
            .rolling_min(window_size=500)
            .shift(1)
            .alias("_rolling_min_low"),
        ]
    )

    # RV 15th percentile: ~30 days of bars — computed after rv_1d so it can
    # reference the already-computed (and shifted) rv_1d column.
    df = df.with_columns(
        pl.col("rv_1d")
        .rolling_quantile(quantile=0.15, interpolation="nearest", window_size=15000)
        .shift(1)
        .alias("rv_15th_pct")
    )

    # ADR stretch: where is current close relative to recent range?
    df = df.with_columns(
        (
            (pl.col("close") - pl.col("_rolling_min_low"))
            / pl.col("_rolling_mean_range").clip(lower_bound=1e-9)
        ).alias("adr_stretch")
    )

    # ── Regime classification — fixed priority order ─────────────────────────
    # Order: TREND_BUILDUP → ABSORPTION → EXHAUSTED → NOISE
    # Rationale: ABSORPTION and EXHAUSTED can overlap; a bar with very high
    # volume, small body AND near a daily extreme should be labelled ABSORPTION
    # (the dominant microstructure signal) rather than EXHAUSTED (a location
    # signal). This matches the replay fallback order that was already in use.
    df = df.with_columns(
        pl.when(pl.col("rv_1d") < pl.col("rv_15th_pct"))
        .then(pl.lit("TREND_BUILDUP"))
        .when(
            (pl.col("volume") > pl.col("vol_80th_pct"))
            & (pl.col("body_size") < (0.25 * pl.col("bar_range")))
            & (pl.col("_delta_efficiency") < pl.col("_de_median"))
        )
        .then(pl.lit("ABSORPTION"))
        .when(
            (pl.col("adr_stretch") > 0.85) | (pl.col("adr_stretch") < 0.15)
        )
        .then(pl.lit("EXHAUSTED"))
        .otherwise(pl.lit("NOISE"))
        .alias("regime")
    )

    # ── Drop internal columns ────────────────────────────────────────────────
    _internal = [
        "_return",
        "_delta_efficiency",
        "_de_median",
        "_rolling_mean_range",
        "_rolling_min_low",
    ]
    existing_internal = [c for c in _internal if c in df.columns]
    if existing_internal:
        df = df.drop(existing_internal)

    return df
