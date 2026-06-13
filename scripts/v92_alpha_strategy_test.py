#!/usr/bin/env python3
"""
V9.2 Alpha OFI Strategy Tester
Tests the OFI Absorption signal gated by the Regime Classifier.
Includes native rigorous statistical diagnostics (t-stat, CI, out-of-sample).
Updated to use lazy chunked streaming, VPIN gating, anti-patterns, and session filters.
"""

import sys
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from numba import njit
import math
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.regime_classifier import add_regime_labels
from features.session_filter import SessionFilter
from features.vpin_gate import VPINGate
from features.anti_pattern_gate import check_branch_a_anti_patterns, check_branch_c_anti_patterns
from features.close_quality import BRANCH_A_MIN_CLOSE_QUALITY, BRANCH_B_MIN_CLOSE_QUALITY, BRANCH_C_MAX_CLOSE_QUALITY

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
    
    # Numba bootstrap
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
        
    signed_ret = df['signed_return_bps'].values
    median = np.median(signed_ret) - COSTS_BPS
    
    mean_raw, t_stat, ci_low_raw, ci_high_raw = bootstrap_mean_ci_tstat(signed_ret, BOOTSTRAP_SAMPLES)
    mean_net = mean_raw - COSTS_BPS
    ci_low = ci_low_raw - COSTS_BPS
    ci_high = ci_high_raw - COSTS_BPS
    
    return f"{prefix} Count: {n:<4} | Mean Net: {mean_net:>6.2f} bps | Median Net: {median:>6.2f} bps | t-stat: {t_stat:>5.2f} | 95% CI: [{ci_low:>6.2f}, {ci_high:>6.2f}]{warning}"


def main():
    print("1. Identifying 500-BTC Base Bars...")
    tier2_dir = ROOT / "data/hft/tier2"
    bar_files = sorted(list(tier2_dir.glob("BTCUSDT_tier2_500btc_*.parquet")))
    
    if not bar_files:
        print("No Volume Bar files found.")
        return
        
    lazy_df = pl.concat([pl.scan_parquet(f) for f in bar_files])
    
    # Check OFI Cache
    ofi_dir = ROOT / "data/hft/tier2/ofi"
    ofi_files = sorted(list(ofi_dir.glob("BTCUSDT_ofi_1s_*.parquet")))
    df_ofi_pl = None
    has_ofi = False
    
    if ofi_files:
        print(f"2. Found {len(ofi_files)} OFI hour-files. Loading and assembling...")
        df_ofi_pl = pl.concat([pl.read_parquet(f) for f in ofi_files])
        df_ofi_pl = df_ofi_pl.sort("datetime").with_columns(
            pl.col("ofi").cum_sum().alias("cumulative_ofi"),
            pl.col("datetime").cast(pl.Datetime("ns"))
        )
        has_ofi = True
    else:
        print("2. No OFI files found. Skipping Branch B (Absorption) generation.")
        
    print("3. Executing chunked strategy pipeline (Memory Optimized)...")
    
    CHUNK_SIZE = 250_000
    LOOKBACK = 15_500
    HORIZON_BARS = 24
    
    events = []
    
    session_filter = SessionFilter()
    vpin_gate = VPINGate()
    
    blocked_stats = {
        "session_filter": 0,
        "vpin_gate": 0,
        "anti_pattern": 0,
        "close_quality": 0,
        "blocked_reasons": {}
    }
    
    i = 0
    while True:
        start_idx = max(0, i - LOOKBACK)
        request_len = i + CHUNK_SIZE + HORIZON_BARS - start_idx
        
        chunk_df = lazy_df.slice(start_idx, request_len).collect()
        if chunk_df.height == 0:
            break
            
        chunk_df = add_regime_labels(chunk_df)
        
        # Align timestamps identically to original logic
        chunk_df = chunk_df.with_columns([
            pl.when(pl.col("open_time") > 1e14)
              .then(pl.from_epoch("open_time", time_unit="us"))
              .otherwise(pl.from_epoch("open_time", time_unit="ms")).cast(pl.Datetime("ns")).alias("datetime_open"),
            pl.when(pl.col("close_time") > 1e14)
              .then(pl.from_epoch("close_time", time_unit="us"))
              .otherwise(pl.from_epoch("close_time", time_unit="ms")).cast(pl.Datetime("ns")).alias("datetime_close")
        ])
        
        if has_ofi:
            chunk_df = chunk_df.join_asof(df_ofi_pl.select(["datetime", "cumulative_ofi"]), left_on="datetime_open", right_on="datetime", strategy="backward")
            chunk_df = chunk_df.rename({"cumulative_ofi": "ofi_at_open"}).drop("datetime")
            
            chunk_df = chunk_df.join_asof(df_ofi_pl.select(["datetime", "cumulative_ofi"]), left_on="datetime_close", right_on="datetime", strategy="backward")
            chunk_df = chunk_df.rename({"cumulative_ofi": "ofi_at_close"}).drop("datetime")
            
            chunk_df = chunk_df.with_columns(
                (pl.col("ofi_at_close") - pl.col("ofi_at_open")).alias("bar_ofi")
            )
            
        rolling_cols = [
            pl.col("high").rolling_max(window_size=50).shift(1).alias("local_high"),
            pl.col("low").rolling_min(window_size=50).shift(1).alias("local_low"),
            pl.col("volume_delta").rolling_quantile(0.90, interpolation="nearest", window_size=1000).shift(1).alias("vol_delta_roll_90"),
            pl.col("volume_delta").rolling_quantile(0.10, interpolation="nearest", window_size=1000).shift(1).alias("vol_delta_roll_10"),
            pl.col("volume").rolling_quantile(0.95, interpolation="nearest", window_size=1000).shift(1).alias("vol_roll_95"),
            pl.col("close").shift(-HORIZON_BARS).alias("fwd_close")
        ]
        
        if has_ofi:
            rolling_cols.extend([
                pl.col("bar_ofi").rolling_quantile(0.90, interpolation="nearest", window_size=1000).shift(1).alias("ofi_roll_90"),
                pl.col("bar_ofi").rolling_quantile(0.10, interpolation="nearest", window_size=1000).shift(1).alias("ofi_roll_10")
            ])
            
        chunk_df = chunk_df.with_columns(rolling_cols)
        
        chunk_df = chunk_df.with_columns([
            ((pl.col("fwd_close") - pl.col("close")) / pl.col("close")).alias("raw_return")
        ])
        
        primary_start = i - start_idx
        process_df = chunk_df.slice(primary_start, CHUNK_SIZE)
        if process_df.height == 0:
            break
            
        process_df = process_df.drop_nulls()
        
        for row in process_df.iter_rows(named=True):
            buy_volume = (row["volume"] + row["volume_delta"]) / 2.0
            sell_volume = (row["volume"] - row["volume_delta"]) / 2.0
            vpin_info = vpin_gate.update(buy_volume, sell_volume)
            
            # Reconstruct datetime for session filter and analysis
            timestamp_ms = row["open_time"]
            dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
            utc_hour = dt.hour
            regime = row["regime"]
            
            is_branch_a_setup = regime == "TREND_BUILDUP" and \
                                ((row["volume_delta"] > row["vol_delta_roll_90"] and row["close"] >= row["local_high"]) or \
                                 (row["volume_delta"] < row["vol_delta_roll_10"] and row["close"] <= row["local_low"]))
                                 
            is_branch_b_setup = False
            if has_ofi:
                is_branch_b_setup = regime == "ABSORPTION" and \
                                    ((row["bar_ofi"] < row["ofi_roll_10"] and row["close"] > row["local_low"]) or \
                                     (row["bar_ofi"] > row["ofi_roll_90"] and row["close"] < row["local_high"]))
                                     
            is_branch_c_setup = regime == "EXHAUSTED" and \
                                ((row["volume"] > row["vol_roll_95"] and row["close"] <= row["local_low"]) or \
                                 (row["volume"] > row["vol_roll_95"] and row["close"] >= row["local_high"]))

            branch = None
            side = None
            
            if is_branch_a_setup:
                branch = "A_Breakout"
                side = "LONG" if row["volume_delta"] > 0 else "SHORT"
                if not session_filter.is_eligible(utc_hour, branch):
                    blocked_stats["session_filter"] += 1
                    continue
                if vpin_info["should_block_branch_a"]:
                    blocked_stats["vpin_gate"] += 1
                    continue
                    
                close_quality = row.get("close_quality", 0.5)
                if side == "LONG" and close_quality < BRANCH_A_MIN_CLOSE_QUALITY:
                    blocked_stats["close_quality"] += 1
                    continue
                elif side == "SHORT" and close_quality > (1.0 - BRANCH_A_MIN_CLOSE_QUALITY):
                    blocked_stats["close_quality"] += 1
                    continue
                    
                block, reasons = check_branch_a_anti_patterns(row, side)
                if block:
                    blocked_stats["anti_pattern"] += 1
                    for r in reasons:
                        blocked_stats["blocked_reasons"][r] = blocked_stats["blocked_reasons"].get(r, 0) + 1
                    continue
                    
            elif is_branch_b_setup:
                branch = "B_Absorption"
                side = "LONG" if row["bar_ofi"] < 0 else "SHORT"
                if not session_filter.is_eligible(utc_hour, branch):
                    blocked_stats["session_filter"] += 1
                    continue
                if vpin_info["should_block_branch_b"]:
                    blocked_stats["vpin_gate"] += 1
                    continue
                    
                close_quality = row.get("close_quality", 0.5)
                if side == "LONG" and close_quality < BRANCH_B_MIN_CLOSE_QUALITY:
                    blocked_stats["close_quality"] += 1
                    continue
                elif side == "SHORT" and close_quality > (1.0 - BRANCH_B_MIN_CLOSE_QUALITY):
                    blocked_stats["close_quality"] += 1
                    continue
                    
            elif is_branch_c_setup:
                branch = "C_ExhaustionFade"
                side = "LONG" if row["close"] <= row["local_low"] else "SHORT"
                if not session_filter.is_eligible(utc_hour, branch):
                    blocked_stats["session_filter"] += 1
                    continue
                if row.get("close_quality", 1) > BRANCH_C_MAX_CLOSE_QUALITY:
                    blocked_stats["close_quality"] += 1
                    continue
                    
                block, reasons = check_branch_c_anti_patterns(row)
                if block:
                    blocked_stats["anti_pattern"] += 1
                    for r in reasons:
                        blocked_stats["blocked_reasons"][r] = blocked_stats["blocked_reasons"].get(r, 0) + 1
                    continue
                    
            if branch:
                sign = 1 if side == "LONG" else -1
                events.append({
                    "branch": branch,
                    "side": side,
                    "datetime": dt, # Use native datetime for Pandas
                    "regime": row["regime"],
                    "raw_return": row["raw_return"],
                    "signed_return_bps": (sign * row["raw_return"] * 10000)
                })
                
        if chunk_df.height < request_len:
            break
            
        i += CHUNK_SIZE

    df_events = pd.DataFrame(events)
    if len(df_events) == 0:
        print("No events fired based on the thresholds.")
        return
        
    df_events['year'] = df_events['datetime'].dt.year
    df_events['period'] = np.where(df_events['year'] >= 2024, "2024-2026", "2020-2023")
    
    print("\n" + "="*80)
    print("V9.2 ALPHA STRATEGY EVALUATION - RIGOROUS DIAGNOSTICS")
    print("="*80)
    
    for branch in ["A_Breakout", "B_Absorption", "C_ExhaustionFade"]:
        branch_df = df_events[df_events['branch'] == branch]
        if len(branch_df) == 0: continue
        
        print(f"\n--- BRANCH: {branch} ---")
        for period in ["2020-2023", "2024-2026"]:
            print(f"  Period: {period}")
            sub_df = branch_df[branch_df['period'] == period]
            if len(sub_df) == 0:
                print("    No events.")
                continue
                
            for regime in ["TREND_BUILDUP", "ABSORPTION", "EXHAUSTED", "NOISE"]:
                regime_df = sub_df[sub_df['regime'] == regime]
                if len(regime_df) == 0: continue
                print("    " + calculate_stats(regime_df, prefix=f"[{regime}]"))
                
        # OOS Split for the primary target regime of the branch
        target_regime = "TREND_BUILDUP" if branch == "A_Breakout" else "ABSORPTION" if branch == "B_Absorption" else "EXHAUSTED"
        print(f"  [Out-of-Sample Split ({target_regime} Regime Only)]")
        oos_df = branch_df[branch_df['regime'] == target_regime].sort_values("datetime").reset_index(drop=True)
        if len(oos_df) >= 100:
            half_idx = len(oos_df) // 2
            print("    " + calculate_stats(oos_df.iloc[:half_idx], prefix="[First Half] "))
            print("    " + calculate_stats(oos_df.iloc[half_idx:], prefix="[Second Half]"))
        else:
            print(f"    Not enough events for a meaningful OOS split (Count: {len(oos_df)})")
            
    print(f"\nBlocked Stats: {blocked_stats}")

if __name__ == "__main__":
    main()
