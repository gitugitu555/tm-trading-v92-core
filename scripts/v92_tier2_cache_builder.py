#!/usr/bin/env python3
"""
V9.2 Tier-2 Cache Builder Pipeline.

This version keeps raw archive discovery centralized in
discover_aggtrade_archives() and supports a dry-run inventory mode that shares
the exact same selection helper as the extraction path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import polars as pl

from features.v92_data_policy import discover_aggtrade_archives, discover_tier2_bar_files

ROOT = Path(__file__).resolve().parents[1]
COLD_ROOT = Path("/mnt/seagate/tm-trading-v555/data/raw")
HOT_OUT = ROOT / "data/hft/tier2"
TEMP_DIR = ROOT / "data/hft/temp"


def build_features_lazy(lazy_df: pl.LazyFrame, volume_bucket_size: float = 100.0) -> pl.LazyFrame:
    """Construct fixed-size volume bars using Polars lazy execution."""
    df = lazy_df.with_columns(
        [
            pl.col("price").cast(pl.Float64),
            pl.col("qty").cast(pl.Float64),
            pl.col("timestamp").cast(pl.Int64),
        ]
    )

    df = df.with_columns(
        pl.when(pl.col("is_buyer_maker"))
        .then(-pl.col("qty"))
        .otherwise(pl.col("qty"))
        .alias("signed_qty"),
        (pl.col("price") * pl.col("qty")).alias("notional"),
    )

    df = df.with_columns((pl.col("qty").cum_sum() // volume_bucket_size).cast(pl.Int64).alias("bar_id"))

    bars = df.group_by("bar_id", maintain_order=True).agg(
        [
            pl.col("timestamp").first().alias("open_time"),
            pl.col("timestamp").last().alias("close_time"),
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("qty").sum().alias("volume"),
            pl.col("signed_qty").sum().alias("volume_delta"),
            pl.col("notional").sum().alias("total_notional"),
            pl.len().alias("trade_count"),
        ]
    )

    return bars.with_columns((pl.col("total_notional") / pl.col("volume")).alias("vwap"))


def _archive_suffix(archive_path: Path, symbol: str) -> str:
    prefix = f"{symbol}-aggTrades-"
    name = archive_path.name
    if not name.startswith(prefix) or not name.endswith(".zip"):
        raise ValueError(f"Unexpected archive filename: {name}")
    return name[len(prefix) : -4]


def _extract_first_csv(archive_path: Path, temp_dir: Path) -> Path:
    with zipfile.ZipFile(archive_path, "r") as zf:
        csv_candidates = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_candidates:
            raise ValueError(f"No CSV payload found in {archive_path.name}")
        member = csv_candidates[0]
        zf.extract(member, path=temp_dir)
        return temp_dir / member


def process_archive_lazy(archive_path: Path, symbol: str = "BTCUSDT", output_dir: Path = HOT_OUT) -> str:
    """Extract one archive, build bars, and persist a parquet shard."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    suffix = _archive_suffix(archive_path, symbol)
    out_file = output_dir / f"{symbol}_tier2_500btc_{suffix}.parquet"

    if not archive_path.exists():
        return f"[{suffix}] Skipped: Missing zip file."
    if out_file.exists():
        return f"[{suffix}] Skipped: Parquet already exists."

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=TEMP_DIR) as tmpdir:
        temp_dir = Path(tmpdir)
        try:
            temp_csv_path = _extract_first_csv(archive_path, temp_dir)
        except Exception as exc:
            return f"[{suffix}] Failed: Could not extract zip - {exc}"

        try:
            lazy_df = pl.scan_csv(
                temp_csv_path,
                has_header=False,
                new_columns=[
                    "agg_id",
                    "price",
                    "qty",
                    "first_id",
                    "last_id",
                    "timestamp",
                    "is_buyer_maker",
                    "is_best_match",
                ],
            )
            lazy_bars = build_features_lazy(lazy_df, volume_bucket_size=500.0)
            bars = lazy_bars.collect(streaming=True)
            bars.write_parquet(out_file, compression="zstd")
        except Exception as exc:
            return f"[{suffix}] Failed during processing: {exc}"

    return f"[{suffix}] Success: Saved {out_file.name}."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V9.2 Tier-2 cache builder")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--search-dir",
        type=Path,
        default=COLD_ROOT / "binance/spot/aggTrades/BTCUSDT/2020-05-22_to_2026-05-21",
    )
    parser.add_argument("--tier2-dir", type=Path, default=HOT_OUT)
    parser.add_argument("--output-dir", type=Path, default=HOT_OUT)
    parser.add_argument(
        "--overlap-policy",
        choices=("monthly_wins", "daily_wins", "reject_overlap"),
        default="monthly_wins",
    )
    parser.add_argument("--all-mode", choices=("all_wins", "shards_only", "reject_mixed"), default="all_wins")
    parser.add_argument("--dry-run", action="store_true", help="List selected archives and exit.")
    parser.add_argument("--manifest-json", action="store_true", help="Emit a structured dry-run manifest as JSON.")
    return parser


def _build_manifest(args: argparse.Namespace, archive_inventory: dict, tier2_inventory: dict) -> dict:
    errors = list(archive_inventory.get("errors", [])) + list(tier2_inventory.get("errors", []))
    return {
        "symbol": args.symbol,
        "archive_search_dir": str(args.search_dir),
        "tier2_dir": str(args.tier2_dir),
        "overlap_policy": args.overlap_policy,
        "all_mode": args.all_mode,
        "selected_archives": [str(path) for path in archive_inventory.get("selected_archives", [])],
        "selected_archive_count": archive_inventory.get("selected_archive_count", 0),
        "selected_tier2_files": [str(path) for path in tier2_inventory.get("selected_tier2_files", [])],
        "selected_tier2_count": tier2_inventory.get("selected_tier2_count", 0),
        "rejected_overlaps": archive_inventory.get("rejected_overlaps", []),
        "unsupported_filenames": archive_inventory.get("unsupported_filenames", []),
        "errors": errors,
        "side_effects_disabled": True,
    }


def _report_inventory(manifest: dict) -> None:
    print("V9.2 Tier-2 Archive Inventory")
    print(f"Policy mode: {manifest['overlap_policy']}")
    print(f"Selected archive count: {manifest['selected_archive_count']}")
    for archive in manifest["selected_archives"]:
        print(f"  - {archive}")
    print(f"Tier-2 mode: {manifest['all_mode']}")
    print(f"Selected Tier-2 file count: {manifest['selected_tier2_count']}")
    for bar_file in manifest["selected_tier2_files"]:
        print(f"  - {bar_file}")
    if manifest["rejected_overlaps"]:
        print("Rejected overlaps:")
        for overlap in manifest["rejected_overlaps"]:
            print(f"  - {overlap}")
    if manifest["unsupported_filenames"]:
        print("Unsupported filenames:")
        for filename in manifest["unsupported_filenames"]:
            print(f"  - {filename}")
    if manifest["errors"]:
        print("Errors:")
        for error in manifest["errors"]:
            print(f"  - {error}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.dry_run:
        archive_inventory = discover_aggtrade_archives(
            args.search_dir,
            args.symbol,
            overlap_policy=args.overlap_policy,
            return_manifest=True,
        )
        tier2_inventory = discover_tier2_bar_files(
            args.tier2_dir,
            symbol=args.symbol,
            all_mode=args.all_mode,
            return_manifest=True,
        )
        manifest = _build_manifest(args, archive_inventory, tier2_inventory)
        if args.manifest_json:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        else:
            _report_inventory(manifest)
        return 1 if manifest["errors"] else 0

    try:
        archives = discover_aggtrade_archives(args.search_dir, args.symbol, overlap_policy=args.overlap_policy)
    except ValueError as exc:
        print(f"Archive discovery failed: {exc}")
        return 1

    print("V9.2 Tier-2 Pipeline Started.")
    print(f"Source: {args.search_dir}")
    print(f"Target: {args.output_dir}")
    print(f"Policy: {args.overlap_policy}")
    print(f"Selected archives: {len(archives)}")

    for archive in archives:
        result = process_archive_lazy(archive, args.symbol, args.output_dir)
        print(result)

    print("\nTier-2 Cache Build Complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
