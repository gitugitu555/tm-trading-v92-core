#!/usr/bin/env python3
"""
V9.2 Alpha OFI Strategy Tester
Tests Branch A/B/C signal diagnostics gated/reported by the Regime Classifier.
Includes native rigorous statistical diagnostics (t-stat, CI, out-of-sample).

Safety policy:
- Tier-2 bar discovery is centralized in discover_tier2_bar_files().
- Timestamp normalization is centralized in epoch_to_datetime_expr().
- Optional OFI enrichment must preserve bar coverage.
- Branch B canonical behavior is cross-regime collection, then regime reporting.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.regime_classifier import add_regime_labels
from features.v92_data_policy import (
    discover_tier2_bar_files,
    epoch_to_datetime_expr,
    join_ofi_to_bars_preserve_coverage,
)

COSTS_BPS = 5.0
BOOTSTRAP_SAMPLES = 1000


@njit
def bootstrap_mean_ci_tstat(signed_returns: np.ndarray, samples: int = 1000):
    n = len(signed_returns)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0

    mean_val = np.mean(signed_returns)
    std_val = np.std(signed_returns)
    t_stat = mean_val / (std_val / math.sqrt(n)) if std_val > 0 else 0.0

    bs_means = np.zeros(samples)
    for i in range(samples):
        indices = np.random.randint(0, n, size=n)
        bs_means[i] = np.mean(signed_returns[indices])

    bs_means.sort()
    ci_low = bs_means[int(0.025 * samples)]
    ci_high = bs_means[int(0.975 * samples)]

    return mean_val, t_stat, ci_low, ci_high


def calculate_stats(df: pd.DataFrame, prefix: str = ""):
    """Calculates robust statistical metrics for a set of events."""
    n = len(df)
    if n == 0:
        return f"{prefix} Count: 0"

    warning = " [WARNING: INSUFFICIENT SAMPLE < 100]" if n < 100 else ""

    signed_ret = df["signed_return_bps"].values
    median = np.median(signed_ret) - COSTS_BPS

    mean_raw, t_stat, ci_low_raw, ci_high_raw = bootstrap_mean_ci_tstat(signed_ret, BOOTSTRAP_SAMPLES)
    mean_net = mean_raw - COSTS_BPS
    ci_low = ci_low_raw - COSTS_BPS
    ci_high = ci_high_raw - COSTS_BPS

    return f"{prefix} Count: {n:<4} | Mean Net: {mean_net:>6.2f} bps | Median Net: {median:>6.2f} bps | t-stat: {t_stat:>5.2f} | 95% CI: [{ci_low:>6.2f}, {ci_high:>6.2f}]{warning}"


def evaluate_branch_events(
    df_pd: pd.DataFrame,
    events_list: list,
    threshold_window: int = 1000,
    structure_window: int = 50,
    horizon: int = 24,
):
    """
    Evaluate signals for Branches A, B, and C.

    Branch B default is canonical cross-regime collection: it can fire in any
    regime when OFI absorption conditions are met, then reporting groups by
    regime after collection.
    """
    min_periods = max(1, min(3, threshold_window))
    structure_min_periods = max(1, min(2, structure_window))

    df_pd = df_pd.copy()
    df_pd["vol_delta_roll_90"] = df_pd["volume_delta"].rolling(threshold_window, min_periods=min_periods).quantile(0.90)
    df_pd["vol_delta_roll_10"] = df_pd["volume_delta"].rolling(threshold_window, min_periods=min_periods).quantile(0.10)

    df_pd["vol_roll_95"] = df_pd["volume"].rolling(threshold_window, min_periods=min_periods).quantile(0.95)

    # Previous completed structure only. Do not include the current bar in the
    # local high/low thresholds used for breakout/exhaustion decisions.
    df_pd["local_high"] = df_pd["high"].rolling(structure_window, min_periods=structure_min_periods).max().shift(1)
    df_pd["local_low"] = df_pd["low"].rolling(structure_window, min_periods=structure_min_periods).min().shift(1)

    if "bar_ofi" in df_pd.columns:
        df_pd["ofi_roll_90"] = df_pd["bar_ofi"].rolling(threshold_window, min_periods=min_periods).quantile(0.90)
        df_pd["ofi_roll_10"] = df_pd["bar_ofi"].rolling(threshold_window, min_periods=min_periods).quantile(0.10)

    df_pd["fwd_close"] = df_pd["close"].shift(-horizon)
    df_pd["raw_return"] = (df_pd["fwd_close"] - df_pd["close"]) / df_pd["close"]

    df_pd = df_pd.dropna(subset=["fwd_close", "vol_delta_roll_90", "local_high", "local_low"])

    for _, row in df_pd.iterrows():
        regime = row["regime"]

        # --- BRANCH A (Breakout / Trend Follow) ---
        if regime == "TREND_BUILDUP":
            side_a = 0
            if row["volume_delta"] > row["vol_delta_roll_90"] and row["close"] >= row["local_high"]:
                side_a = 1
            elif row["volume_delta"] < row["vol_delta_roll_10"] and row["close"] <= row["local_low"]:
                side_a = -1
            if side_a != 0:
                events_list.append({"branch": "A_Breakout", "datetime": row["datetime_close"], "regime": regime, "side": side_a, "raw_return": row["raw_return"], "signed_return_bps": side_a * row["raw_return"] * 10_000})

        # --- BRANCH B (OFI Absorption) ---
        has_ofi_coverage = bool(row.get("has_ofi_coverage", True))
        if has_ofi_coverage and "bar_ofi" in row and not pd.isna(row["bar_ofi"]):
            side_b = 0
            if row["bar_ofi"] < row["ofi_roll_10"] and row["close"] > row["local_low"]:
                side_b = 1
            elif row["bar_ofi"] > row["ofi_roll_90"] and row["close"] < row["local_high"]:
                side_b = -1
            if side_b != 0:
                events_list.append({"branch": "B_Absorption", "datetime": row["datetime_close"], "regime": regime, "side": side_b, "raw_return": row["raw_return"], "signed_return_bps": side_b * row["raw_return"] * 10_000})

        # --- BRANCH C (Exhaustion Fade) ---
        if regime == "EXHAUSTED":
            side_c = 0
            if row["volume"] > row["vol_roll_95"] and row["close"] >= row["local_high"]:
                side_c = -1
            elif row["volume"] > row["vol_roll_95"] and row["close"] <= row["local_low"]:
                side_c = 1
            if side_c != 0:
                events_list.append({"branch": "C_ExhaustionFade", "datetime": row["datetime_close"], "regime": regime, "side": side_c, "raw_return": row["raw_return"], "signed_return_bps": side_c * row["raw_return"] * 10_000})


def main():
    print("1. Loading 500-BTC Base Bars & Applying Regime Classifier...")
    tier2_dir = ROOT / "data/hft/tier2"
    bar_files = discover_tier2_bar_files(tier2_dir, symbol="BTCUSDT", all_mode="all_wins")

    if not bar_files:
        print("No Volume Bar files found.")
        return

    df_bars_pl = pl.concat([pl.scan_parquet(f) for f in bar_files]).collect()
    df_bars_pl = df_bars_pl.sort("open_time")

    df_bars_pl = add_regime_labels(df_bars_pl)
    df_bars_pl = df_bars_pl.with_columns([
        epoch_to_datetime_expr("open_time").alias("datetime_open"),
        epoch_to_datetime_expr("close_time").alias("datetime_close"),
    ])

    print("2. Loading Available 1-Second OFI Cache...")
    ofi_dir = ROOT / "data/hft/tier2/ofi"
    ofi_files = sorted(ofi_dir.glob("BTCUSDT_ofi_1s_*.parquet"))

    if ofi_files:
        print(f"   -> Found {len(ofi_files)} OFI hour-files. Loading and assembling...")
        df_ofi_pl = pl.concat([pl.read_parquet(f) for f in ofi_files])
        df_ofi_pl = df_ofi_pl.sort("datetime").with_columns(pl.col("datetime").cast(pl.Datetime("ns")))
        print("3. AsOf Joining OFI to Volume Bars while preserving non-OFI coverage...")
        df_full = join_ofi_to_bars_preserve_coverage(df_bars_pl, df_ofi_pl)
    else:
        print("   -> No OFI files found. Branch B will be disabled, but Branch A/C remain eligible.")
        df_full = join_ofi_to_bars_preserve_coverage(df_bars_pl, None)

    print(f"   -> Executing strategy logic across {len(df_full)} bars...")
    df_pd = df_full.to_pandas()

    events = []
    evaluate_branch_events(df_pd, events)

    df_events = pd.DataFrame(events)
    if len(df_events) == 0:
        print("No events fired based on the thresholds.")
        return

    df_events["year"] = df_events["datetime"].dt.year
    df_events["period"] = np.where(df_events["year"] >= 2024, "2024-2026", "2020-2023")

    print("\n" + "=" * 80)
    print("V9.2 ALPHA STRATEGY EVALUATION - RIGOROUS DIAGNOSTICS")
    print("=" * 80)

    for branch in ["A_Breakout", "B_Absorption", "C_ExhaustionFade"]:
        branch_df = df_events[df_events["branch"] == branch]
        if len(branch_df) == 0:
            continue

        print(f"\n--- BRANCH: {branch} ---")
        for period in ["2020-2023", "2024-2026"]:
            print(f"  Period: {period}")
            sub_df = branch_df[branch_df["period"] == period]
            if len(sub_df) == 0:
                print("    No events.")
                continue

            for regime in ["TREND_BUILDUP", "ABSORPTION", "EXHAUSTED", "NOISE"]:
                regime_df = sub_df[sub_df["regime"] == regime]
                if len(regime_df) == 0:
                    continue
                print("    " + calculate_stats(regime_df, prefix=f"[{regime}]"))

        target_regime = "TREND_BUILDUP" if branch == "A_Breakout" else "ABSORPTION" if branch == "B_Absorption" else "EXHAUSTED"
        print(f"  [Out-of-Sample Split ({target_regime} Regime Only)]")
        oos_df = branch_df[branch_df["regime"] == target_regime].sort_values("datetime").reset_index(drop=True)
        if len(oos_df) >= 100:
            half_idx = len(oos_df) // 2
            print("    " + calculate_stats(oos_df.iloc[:half_idx], prefix="[First Half] "))
            print("    " + calculate_stats(oos_df.iloc[half_idx:], prefix="[Second Half]"))
        else:
            print(f"    Not enough events for a meaningful OOS split (Count: {len(oos_df)})")


if __name__ == "__main__":
    main()
