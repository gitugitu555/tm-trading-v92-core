#!/usr/bin/env python3
"""
V9.2 L2 Tier-2 Cache Builder (OFI Extraction)
---------------------------------------------
Processes raw highly-compressed L2 Orderbook diffs (.parquet.zst) into 
aggregated 1-second Tier-2 Parquet files containing Order Flow Imbalance.

Uses sequential processing to remain Ultra-Low Memory compatible alongside LLMs.
"""

import sys
import io
import os
import glob
import pandas as pd
import polars as pl
import zstandard as zstd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.microstructure_numba_ofi import process_chunk_fast as process_chunk

COLD_ROOT = Path("/mnt/seagate/tm-trading-v555/data/raw/cryptohftdata/orderbook/binance_futures/BTCUSDT")
HOT_OUT = ROOT / "data/hft/tier2/ofi"

def load_zst_parquet(filepath: Path) -> pd.DataFrame:
    """Decompresses a .parquet.zst file into a Pandas DataFrame."""
    with open(filepath, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            decompressed_data = reader.read()
            
    df = pd.read_parquet(io.BytesIO(decompressed_data), engine='pyarrow')
    return df

def process_hour(zst_path: Path) -> str:
    """Processes a single hour of L2 tick data into 1-second OFI bars."""
    # Extract YYYY-MM-DD and HH from path
    parts = zst_path.parts
    date_str = parts[-3]
    hour_str = parts[-2]
    
    out_file = HOT_OUT / f"BTCUSDT_ofi_1s_{date_str}_{hour_str}.parquet"
    if out_file.exists():
        return f"[{date_str} {hour_str}:00] Skipped: Parquet already exists."
        
    print(f"[{date_str} {hour_str}:00] Decompressing and loading L2 Orderbook...")
    try:
        df = load_zst_parquet(zst_path)
    except Exception as e:
        return f"[{date_str} {hour_str}:00] Failed to load ZST: {e}"
        
    if df.empty:
        return f"[{date_str} {hour_str}:00] Skipped: Empty file."
        
    print(f"[{date_str} {hour_str}:00] Calculating tick-level OFI (Numba Accelerated)...")
    df = df.sort_values(by=['transaction_time', 'event_time', 'first_update_id']).reset_index(drop=True)
    df = process_chunk(df)
    
    print(f"[{date_str} {hour_str}:00] Aggregating into 1-second bars...")
    # Convert timestamp (Unix MS) to datetime to aggregate
    df['datetime'] = pd.to_datetime(df['transaction_time'], unit='ms')
    df.set_index('datetime', inplace=True)
    
    # Resample to 1-second intervals, summing the tick OFI
    # We also count total updates to measure activity
    ofi_bars = df.resample('1S').agg({
        'ofi': 'sum',
        'event_time': 'count' # Using event_time as a count column
    }).rename(columns={'event_time': 'update_count'})
    
    # Drop empty seconds to save massive space
    ofi_bars = ofi_bars[ofi_bars['update_count'] > 0]
    ofi_bars.reset_index(inplace=True)
    
    # Convert to Polars for efficient highly-compressed Parquet saving
    pl_bars = pl.from_pandas(ofi_bars)
    pl_bars.write_parquet(out_file, compression="zstd")
    
    return f"[{date_str} {hour_str}:00] Success: Saved {len(pl_bars)} 1-second OFI bars."

from concurrent.futures import ProcessPoolExecutor, as_completed

def main():
    print("V9.2 L2 OFI Cache Builder Started (Parallelized).")
    print(f"Cold Source: {COLD_ROOT}")
    print(f"Hot Target:  {HOT_OUT}\n")
    
    HOT_OUT.mkdir(parents=True, exist_ok=True)
    
    # Find all .zst files
    search_pattern = str(COLD_ROOT / "*" / "*" / "BTCUSDT_orderbook.parquet.zst")
    files = sorted(glob.glob(search_pattern))
    
    print(f"Found {len(files)} total hours of L2 data to process.")
    
    # We are reverting to pure sequential processing because the 2026 bull market 
    # ticks are so insanely dense that even 4 workers trigger the OOM-killer.
    print(f"Using sequential processing for maximum memory safety...")
    
    for fp in files:
        try:
            result = process_hour(Path(fp))
            print(result)
        except Exception as exc:
            print(f"File generated an exception: {exc}")
        
    print("L2 OFI Cache Build Complete.")

if __name__ == "__main__":
    sys.exit(main())
