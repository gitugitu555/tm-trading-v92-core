#!/usr/bin/env python3
import sys
import zipfile
import glob
import os
import shutil
import tempfile
import polars as pl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLD_ROOT = Path("/mnt/seagate/tm-trading-v555/data/raw")
HOT_OUT = ROOT / "data/hft/tier2"

def process_archive_batched(temp_csv_path: Path, current_cumulative_qty: float, volume_bucket_size: float = 500.0) -> tuple[pl.DataFrame, float]:
    reader = pl.read_csv_batched(
        temp_csv_path, 
        has_header=False,
        new_columns=["agg_id", "price", "qty", "first_id", "last_id", "timestamp", "is_buyer_maker", "is_best_match"],
        batch_size=500_000
    )
    
    all_batch_bars = []
    
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
            
        batch_df = batches[0]
        batch_df = batch_df.with_columns([
            pl.col("price").cast(pl.Float64),
            pl.col("qty").cast(pl.Float64),
            pl.col("timestamp").cast(pl.Int64)
        ])
        
        batch_df = batch_df.with_columns([
            pl.when(pl.col("is_buyer_maker")).then(-pl.col("qty")).otherwise(pl.col("qty")).alias("signed_qty"),
            (pl.col("price") * pl.col("qty")).alias("notional")
        ])
        
        batch_df = batch_df.with_columns(
            ((pl.col("qty").cum_sum() + current_cumulative_qty) // volume_bucket_size).cast(pl.Int64).alias("bar_id")
        )
        
        current_cumulative_qty += batch_df["qty"].sum()
        
        bars = batch_df.group_by("bar_id", maintain_order=True).agg([
            pl.col("timestamp").first().alias("open_time"),
            pl.col("timestamp").last().alias("close_time"),
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("qty").sum().alias("volume"),
            pl.col("signed_qty").sum().alias("volume_delta"),
            pl.col("notional").sum().alias("total_notional"),
            pl.len().alias("trade_count")
        ])
        
        all_batch_bars.append(bars)
        
    return pl.concat(all_batch_bars), current_cumulative_qty

def main():
    print("V9.2 Tier-2 Pipeline Started (Ultra-Low Memory Batched Edition).")
    print(f"Cold Source: {COLD_ROOT}")
    print(f"Hot Target:  {HOT_OUT}\n")
    
    symbol = "BTCUSDT"
    search_dir = COLD_ROOT / f"binance/spot/aggTrades/{symbol}/2020-05-22_to_2026-05-21"
    
    if not search_dir.exists():
        print(f"Error: Could not find trades directory at {search_dir}")
        sys.exit(1)
        
    zip_files = sorted(list(search_dir.glob(f"{symbol}-aggTrades-*-*.zip")))
    
    if not zip_files:
        print("No zip files found.")
        sys.exit(1)
        
    HOT_OUT.mkdir(parents=True, exist_ok=True)
    out_file = HOT_OUT / f"{symbol}_tier2_500btc_ALL.parquet"
    
    temp_dir = Path(tempfile.mkdtemp())
    try:
        current_cumulative_qty = 0.0
        all_archive_bars = []
        
        for zip_path in zip_files:
            print(f"Processing {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                csv_filename = z.namelist()[0]
                temp_csv_path = temp_dir / csv_filename
                z.extract(csv_filename, path=temp_dir)
                
            bars_df, current_cumulative_qty = process_archive_batched(temp_csv_path, current_cumulative_qty)
            all_archive_bars.append(bars_df)
            
            os.remove(temp_csv_path)
            
        full_bars = pl.concat(all_archive_bars).sort("open_time")
        
        print("Merging split bars across batch boundaries...")
        final_bars = full_bars.group_by("bar_id", maintain_order=True).agg([
            pl.col("open_time").min(),
            pl.col("close_time").max(),
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("volume_delta").sum(),
            (pl.col("total_notional").sum() / pl.col("volume").sum()).alias("vwap"),
            pl.col("trade_count").sum()
        ])
        
        final_bars = final_bars.with_columns([
            (pl.col("close") - pl.col("open")).abs().alias("body_size"),
            (pl.col("high") - pl.col("low")).alias("bar_range")
        ])
        
        final_bars = final_bars.with_columns([
            ((pl.col("close") - pl.col("low")) / pl.col("bar_range").clip(lower_bound=1e-9)).alias("close_quality")
        ]).sort("open_time")
        
        final_bars.write_parquet(out_file, compression="zstd")
        print(f"Success: Wrote {len(final_bars)} highly contiguous volume bars to {out_file}")
        
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
