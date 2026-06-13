from pathlib import Path

import polars as pl
import pytest

from features.v92_data_policy import (
    discover_aggtrade_archives,
    discover_tier2_bar_files,
    epoch_to_datetime_expr,
    join_ofi_to_bars_preserve_coverage,
)


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder")
    return path


def test_aggtrade_archive_policy_monthly_wins(tmp_path: Path):
    monthly = touch(tmp_path / "BTCUSDT-aggTrades-2021-05.zip")
    touch(tmp_path / "BTCUSDT-aggTrades-2021-05-01.zip")
    touch(tmp_path / "BTCUSDT-aggTrades-2021-05-31.zip")

    selected = discover_aggtrade_archives(tmp_path, "BTCUSDT", overlap_policy="monthly_wins")

    assert selected == [monthly]


def test_aggtrade_archive_policy_daily_wins(tmp_path: Path):
    touch(tmp_path / "BTCUSDT-aggTrades-2021-05.zip")
    day_1 = touch(tmp_path / "BTCUSDT-aggTrades-2021-05-01.zip")
    day_31 = touch(tmp_path / "BTCUSDT-aggTrades-2021-05-31.zip")

    selected = discover_aggtrade_archives(tmp_path, "BTCUSDT", overlap_policy="daily_wins")

    assert selected == [day_1, day_31]


def test_aggtrade_archive_policy_reject_overlap(tmp_path: Path):
    touch(tmp_path / "BTCUSDT-aggTrades-2021-05.zip")
    touch(tmp_path / "BTCUSDT-aggTrades-2021-05-01.zip")

    with pytest.raises(ValueError, match="Overlapping"):
        discover_aggtrade_archives(tmp_path, "BTCUSDT", overlap_policy="reject_overlap")


def test_aggtrade_archive_policy_rejects_unsupported_name(tmp_path: Path):
    touch(tmp_path / "BTCUSDT-aggTrades-2022-05-09_1m.zip")

    with pytest.raises(ValueError, match="Unsupported"):
        discover_aggtrade_archives(tmp_path, "BTCUSDT")


def test_tier2_bar_policy_all_wins(tmp_path: Path):
    all_file = touch(tmp_path / "BTCUSDT_tier2_500btc_ALL.parquet")
    touch(tmp_path / "BTCUSDT_tier2_500btc_2021-05.parquet")
    touch(tmp_path / "BTCUSDT_tier2_500btc_2021-06.parquet")

    selected = discover_tier2_bar_files(tmp_path, symbol="BTCUSDT", all_mode="all_wins")

    assert selected == [all_file]


def test_tier2_bar_policy_shards_only(tmp_path: Path):
    touch(tmp_path / "BTCUSDT_tier2_500btc_ALL.parquet")
    shard_1 = touch(tmp_path / "BTCUSDT_tier2_500btc_2021-05.parquet")
    shard_2 = touch(tmp_path / "BTCUSDT_tier2_500btc_2021-06.parquet")

    selected = discover_tier2_bar_files(tmp_path, symbol="BTCUSDT", all_mode="shards_only")

    assert selected == [shard_1, shard_2]


def test_tier2_bar_policy_reject_mixed(tmp_path: Path):
    touch(tmp_path / "BTCUSDT_tier2_500btc_ALL.parquet")
    touch(tmp_path / "BTCUSDT_tier2_500btc_2021-05.parquet")

    with pytest.raises(ValueError, match="Mixed"):
        discover_tier2_bar_files(tmp_path, symbol="BTCUSDT", all_mode="reject_mixed")


def test_epoch_to_datetime_expr_handles_ms_and_us_epochs():
    df = pl.DataFrame(
        {
            "epoch": [
                1_704_067_200_000,       # 2024-01-01 in ms
                1_764_547_200_000_000,   # Binance-era us timestamp
            ]
        }
    )

    out = df.with_columns(epoch_to_datetime_expr("epoch").alias("dt"))
    years = out.select(pl.col("dt").dt.year()).to_series().to_list()

    assert all(2020 <= year <= 2030 for year in years)


def test_join_ofi_to_bars_preserves_rows_and_marks_coverage():
    bars = pl.DataFrame(
        {
            "open_time": [1_704_067_200_000, 1_704_067_201_000, 1_704_067_202_000, 1_704_067_203_000],
            "close_time": [1_704_067_200_500, 1_704_067_201_500, 1_704_067_202_500, 1_704_067_203_500],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [500.0, 500.0, 500.0, 500.0],
            "volume_delta": [10.0, -10.0, 20.0, -20.0],
        }
    )
    ofi = pl.DataFrame(
        {
            "datetime": [
                "2024-01-01T00:00:01.000",
                "2024-01-01T00:00:01.500",
                "2024-01-01T00:00:02.500",
            ],
            "ofi": [1.0, 2.0, -1.0],
        }
    ).with_columns(pl.col("datetime").str.strptime(pl.Datetime("ns")))

    out = join_ofi_to_bars_preserve_coverage(bars, ofi)

    assert out.height == bars.height
    assert "bar_ofi" in out.columns
    assert "has_ofi_coverage" in out.columns
    assert out.select(pl.col("has_ofi_coverage").sum()).item() == 2
    assert out.filter(~pl.col("has_ofi_coverage")).select(pl.col("bar_ofi").is_null().all()).item()


def test_join_ofi_none_preserves_rows_and_disables_coverage():
    bars = pl.DataFrame(
        {
            "open_time": [1_704_067_200_000, 1_704_067_201_000],
            "close_time": [1_704_067_200_500, 1_704_067_201_500],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [500.0, 500.0],
            "volume_delta": [10.0, -10.0],
        }
    )

    out = join_ofi_to_bars_preserve_coverage(bars, None)

    assert out.height == bars.height
    assert out.select(pl.col("has_ofi_coverage").sum()).item() == 0
    assert out.select(pl.col("bar_ofi").is_null().all()).item()
