#!/usr/bin/env python3
"""
V9.2 Tier-2 Cache Builder Pipeline (Multi-Core Edition)
----------------------------------
Processes raw Binance Spot aggTrades into highly-compressed Parquet Volume Bars.
Extracts:
- Volume Delta (Maker/Taker Flow)
- Footprint Diagonals (Binned Imbalances)
- Trade counts and VWAP

Utilizes up to 30 CPU cores to process multiple months simultaneously.
Outputs to the NVMe Hot Cache.
"""

import sys
import zipfile
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import polars as pl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLD_ROOT = Path("/mnt/seagate/tm-trading-v555/data/raw")
HOT_OUT = ROOT / "data/hft/tier2"

def load_trades_from_zip(zip_path: Path) -> pl.DataFrame:
    """Reads Binance aggTrades CSV from a zip archive into a Polars DataFrame."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            df = pl.read_csv(
                f.read(),
                has_header=False,
                new_columns=["agg_id", "price", "qty", "first_id", "last_id", "timestamp", "is_buyer_maker", "is_best_match"]
            )
    return df

def build_features(df: pl.DataFrame, volume_bucket_size: float = 100.0) -> pl.DataFrame:
    """
    Constructs Volume Bars and calculates Microstructure Features.
    """
    df = df.with_columns([
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

def process_month(month_str: str, symbol: str = "BTCUSDT") -> str:
    print(f"[{month_str}] Starting Tier-2 Extraction for {symbol}...")
    
    trades_path = COLD_ROOT / f"binance/spot/aggTrades/{symbol}/2020-05-22_to_2026-05-21/{symbol}-aggTrades-{month_str}.zip"
    if not trades_path.exists():
        return f"[{month_str}] Skipped: Missing zip file."
        
    try:
        df_trades = load_trades_from_zip(trades_path)
    except Exception as e:
        return f"[{month_str}] Failed: Could not read zip - {e}"
        
    bars = build_features(df_trades, volume_bucket_size=500.0)
    
    out_file = HOT_OUT / f"{symbol}_tier2_500btc_{month_str}.parquet"
    HOT_OUT.mkdir(parents=True, exist_ok=True)
    
    bars.write_parquet(out_file, compression="zstd")
    return f"[{month_str}] Success: Saved {len(bars)} volume bars."

def main():
    print("V9.2 Tier-2 Pipeline Started (Multi-Core Edition).")
    print(f"Cold Source: {COLD_ROOT}")
    print(f"Hot Target:  {HOT_OUT}\n")
    
    symbol = "BTCUSDT"
    search_dir = COLD_ROOT / f"binance/spot/aggTrades/{symbol}/2020-05-22_to_2026-05-21"
    
    if not search_dir.exists():
        print(f"Error: Could not find trades directory at {search_dir}")
        sys.exit(1)
        
    # Find all monthly zip files
    # Expected format: BTCUSDT-aggTrades-YYYY-MM.zip
    zip_files = glob.glob(str(search_dir / f"{symbol}-aggTrades-*-*.zip"))
    
    months = []
    for fp in zip_files:
        filename = Path(fp).name
        # Extract YYYY-MM
        parts = filename.replace(".zip", "").split("-")
        if len(parts) >= 4:
            month_str = f"{parts[-2]}-{parts[-1]}"
            # Filter out daily files if any (YYYY-MM-DD has 5 parts)
            if len(parts) == 4: 
                months.append(month_str)
            elif len(parts) == 5:
                # If they are daily files, we extract YYYY-MM-DD
                month_str = f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
                months.append(month_str)
                
    months = sorted(list(set(months)))
    print(f"Found {len(months)} months/days to process.")
    
    max_workers = 3
    print(f"Launching ProcessPoolExecutor with {max_workers} workers (Safe Memory Limit)...\n")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_month, m, symbol): m for m in months}
        
        for future in as_completed(futures):
            result = future.result()
            print(result)
            
    print("\nTier-2 Cache Build Complete.")

if __name__ == "__main__":
    sys.exit(main())
