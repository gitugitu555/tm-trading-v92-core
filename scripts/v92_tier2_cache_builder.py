#!/usr/bin/env python3
"""
V9.2 Tier-2 Cache Builder Pipeline (Ultra-Low Memory Edition)
----------------------------------
Processes raw Binance Spot aggTrades into highly-compressed Parquet Volume Bars.
Extracts:
- Volume Delta (Maker/Taker Flow)
- Footprint Diagonals (Binned Imbalances)
- Trade counts and VWAP

Specifically designed to run on systems with heavy background processes (LLMs).
Uses Polars Lazy API (scan_csv) to stream files from disk to keep RAM usage < 1GB.
"""

import sys
import zipfile
import glob
import os
import polars as pl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLD_ROOT = Path("/mnt/seagate/tm-trading-v555/data/raw")
HOT_OUT = ROOT / "data/hft/tier2"
TEMP_DIR = ROOT / "data/hft/temp"

def build_features_lazy(lazy_df: pl.LazyFrame, volume_bucket_size: float = 100.0) -> pl.LazyFrame:
    """
    Constructs Volume Bars using Polars Lazy execution to avoid RAM spikes.
    """
    df = lazy_df.with_columns([
        pl.col("price").cast(pl.Float64),
        pl.col("qty").cast(pl.Float64),
        pl.col("timestamp").cast(pl.Int64)
    ])
    
    df = df.with_columns(
        pl.when(pl.col("is_buyer_maker")).then(-pl.col("qty")).otherwise(pl.col("qty")).alias("signed_qty"),
        (pl.col("price") * pl.col("qty")).alias("notional")
    )
    
    df = df.with_columns(
        (pl.col("qty").cum_sum() // volume_bucket_size).cast(pl.Int64).alias("bar_id")
    )
    
    bars = df.group_by("bar_id", maintain_order=True).agg([
        pl.col("timestamp").first().alias("open_time"),
        pl.col("timestamp").last().alias("close_time"),
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("qty").sum().alias("volume"),
        pl.col("signed_qty").sum().alias("volume_delta"),
        pl.col("notional").sum().alias("total_notional"),
        pl.count("agg_id").alias("trade_count")
    ])
    
    bars = bars.with_columns(
        (pl.col("total_notional") / pl.col("volume")).alias("vwap")
    )
    
    return bars

def process_month_lazy(month_str: str, symbol: str = "BTCUSDT") -> str:
    print(f"\n[{month_str}] Starting Tier-2 Extraction for {symbol}...")
    
    trades_path = COLD_ROOT / f"binance/spot/aggTrades/{symbol}/2020-05-22_to_2026-05-21/{symbol}-aggTrades-{month_str}.zip"
    if not trades_path.exists():
        return f"[{month_str}] Skipped: Missing zip file."
        
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    out_file = HOT_OUT / f"{symbol}_tier2_500btc_{month_str}.parquet"
    HOT_OUT.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    if out_file.exists():
        return f"[{month_str}] Skipped: Parquet already exists."
        
    print(f"[{month_str}] Extracting ZIP to temporary disk space (to save RAM)...")
    try:
        with zipfile.ZipFile(trades_path, 'r') as z:
            csv_filename = z.namelist()[0]
            temp_csv_path = TEMP_DIR / csv_filename
            z.extract(csv_filename, path=TEMP_DIR)
    except Exception as e:
        return f"[{month_str}] Failed: Could not extract zip - {e}"
        
    print(f"[{month_str}] Streaming CSV via Polars Lazy API...")
    try:
        # scan_csv barely uses any RAM because it streams chunks off the disk!
        lazy_df = pl.scan_csv(
            temp_csv_path,
            has_header=False,
            new_columns=["agg_id", "price", "qty", "first_id", "last_id", "timestamp", "is_buyer_maker", "is_best_match"]
        )
        
        lazy_bars = build_features_lazy(lazy_df, volume_bucket_size=500.0)
        
        # Collect executes the lazy graph and streams the result
        bars = lazy_bars.collect(streaming=True)
        bars.write_parquet(out_file, compression="zstd")
        result_msg = f"[{month_str}] Success: Saved {len(bars)} volume bars."
    except Exception as e:
        result_msg = f"[{month_str}] Failed during processing: {e}"
    finally:
        # Cleanup the massive CSV to save disk space
        if temp_csv_path.exists():
            os.remove(temp_csv_path)
            
    return result_msg

def main():
    print("V9.2 Tier-2 Pipeline Started (Ultra-Low Memory Edition).")
    print(f"Cold Source: {COLD_ROOT}")
    print(f"Hot Target:  {HOT_OUT}\n")
    
    symbol = "BTCUSDT"
    search_dir = COLD_ROOT / f"binance/spot/aggTrades/{symbol}/2020-05-22_to_2026-05-21"
    
    if not search_dir.exists():
        print(f"Error: Could not find trades directory at {search_dir}")
        sys.exit(1)
        
    zip_files = glob.glob(str(search_dir / f"{symbol}-aggTrades-*-*.zip"))
    
    months = []
    for fp in zip_files:
        filename = Path(fp).name
        parts = filename.replace(".zip", "").split("-")
        if len(parts) >= 4:
            month_str = f"{parts[-2]}-{parts[-1]}"
            if len(parts) == 4: 
                months.append(month_str)
            elif len(parts) == 5:
                month_str = f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
                months.append(month_str)
                
    months = sorted(list(set(months)))
    print(f"Found {len(months)} months/days to process.")
    print("Processing sequentially with Lazy Streaming to preserve RAM for local LLMs...\n")
    
    for m in months:
        result = process_month_lazy(m, symbol)
        print(result)
            
    print("\nTier-2 Cache Build Complete.")

if __name__ == "__main__":
    sys.exit(main())
