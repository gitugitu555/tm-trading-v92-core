from __future__ import annotations

import json
from pathlib import Path

import polars as pl

import scripts.run_c_exhaustion_paper_sim as cli


def _write_parquet_shard(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_parquet(path)


def _make_rows() -> list[dict]:
    rows: list[dict] = []
    base_open = 1000.0
    base_low = 995.0
    start_ts = 1704067200000  # 2024-01-01 UTC in ms

    for idx in range(100):
        open_time = start_ts + idx * 60_000
        close_time = open_time + 59_000
        if idx == 60:
            low = 900.0
            close = 900.0
            regime = "EXHAUSTED"
            volume = 2000.0
        else:
            low = base_low + idx * 2.0
            close = low + 4.0
            regime = "NOISE"
            volume = 1000.0 + idx
        rows.append(
            {
                "open_time": open_time,
                "close_time": close_time,
                "open": base_open + idx,
                "high": base_open + idx + 8.0,
                "low": low,
                "close": close,
                "volume": volume,
                "volume_delta": 25.0,
                "trade_count": 50,
                "regime": regime,
            }
        )
    return rows


def test_cli_prints_summary_without_writes(tmp_path, capsys):
    bar_dir = tmp_path / "bars"
    bar_dir.mkdir()
    rows = _make_rows()
    _write_parquet_shard(bar_dir / "BTCUSDT_tier2_750btc_0.parquet", rows[:50])
    _write_parquet_shard(bar_dir / "BTCUSDT_tier2_750btc_1.parquet", rows[50:])

    output_dir = tmp_path / "reports"
    rc = cli.main(
        [
            "--bar-dir",
            str(bar_dir),
            "--output-dir",
            str(output_dir),
            "--bar-size",
            "750",
            "--horizon",
            "36",
            "--fee-bps-per-side",
            "3",
            "--slippage-bps-per-side",
            "3",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "strategy=C_ExhaustionFade" in captured.out
    assert "symbol=BTCUSDT" in captured.out
    assert "production_path_touched=false" in captured.out
    assert not output_dir.exists()


def test_cli_writes_requested_reports(tmp_path, capsys):
    bar_dir = tmp_path / "bars"
    output_dir = tmp_path / "reports"
    bar_dir.mkdir()
    rows = _make_rows()
    _write_parquet_shard(bar_dir / "BTCUSDT_tier2_750btc_0.parquet", rows[:50])
    _write_parquet_shard(bar_dir / "BTCUSDT_tier2_750btc_1.parquet", rows[50:])

    rc = cli.main(
        [
            "--bar-dir",
            str(bar_dir),
            "--output-dir",
            str(output_dir),
            "--bar-size",
            "750",
            "--horizon",
            "36",
            "--fee-bps-per-side",
            "3",
            "--slippage-bps-per-side",
            "3",
            "--write-events",
            "--write-trades",
            "--write-equity",
            "--write-summary",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "strategy=C_ExhaustionFade" in captured.out

    events_path = output_dir / "c_exhaustion_paper_events.jsonl"
    trades_path = output_dir / "c_exhaustion_paper_trades.csv"
    equity_path = output_dir / "c_exhaustion_paper_equity.csv"
    summary_json_path = output_dir / "c_exhaustion_paper_summary.json"
    summary_md_path = output_dir / "c_exhaustion_paper_summary.md"

    assert events_path.exists()
    assert trades_path.exists()
    assert equity_path.exists()
    assert summary_json_path.exists()
    assert summary_md_path.exists()

    summary = json.loads(summary_json_path.read_text())
    assert summary["production_path_touched"] is False
    assert summary["round_trip_cost_bps"] == 12.0
    assert summary["trade_count"] == 1

    assert len(trades_path.read_text().strip().splitlines()) >= 2
    assert len(equity_path.read_text().strip().splitlines()) >= 2

    event_types = {
        json.loads(line)["event_type"]
        for line in events_path.read_text().strip().splitlines()
        if line.strip()
    }
    assert {
        "SIGNAL_DETECTED",
        "PAPER_ORDER_CREATED",
        "PAPER_ORDER_FILLED",
        "PAPER_POSITION_OPENED",
        "PAPER_POSITION_CLOSED",
        "PAPER_EQUITY_UPDATED",
    }.issubset(event_types)


def test_cli_refuses_production_output_path(tmp_path, monkeypatch, capsys):
    bar_dir = tmp_path / "bars"
    bar_dir.mkdir()
    rows = _make_rows()
    _write_parquet_shard(bar_dir / "BTCUSDT_tier2_750btc_0.parquet", rows[:20])

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    blocked_output = tmp_path / "data" / "hft" / "tier2"

    rc = cli.main(
        [
            "--bar-dir",
            str(bar_dir),
            "--output-dir",
            str(blocked_output),
            "--bar-size",
            "750",
            "--horizon",
            "36",
            "--write-summary",
        ]
    )
    captured = capsys.readouterr()

    assert rc != 0
    assert "Refused production/cache output path" in captured.err
    assert not blocked_output.exists()


def test_cli_fails_cleanly_on_empty_bar_dir(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    rc = cli.main(["--bar-dir", str(empty_dir), "--output-dir", str(tmp_path / "reports")])
    captured = capsys.readouterr()

    assert rc == 1
    assert "No parquet files found" in captured.err


def test_cli_anchor_cost_equivalent_summary(tmp_path, capsys):
    bar_dir = tmp_path / "bars"
    bar_dir.mkdir()
    rows = _make_rows()
    _write_parquet_shard(bar_dir / "BTCUSDT_tier2_750btc_0.parquet", rows[:50])
    _write_parquet_shard(bar_dir / "BTCUSDT_tier2_750btc_1.parquet", rows[50:])

    output_dir = tmp_path / "reports"
    rc = cli.main(
        [
            "--bar-dir",
            str(bar_dir),
            "--output-dir",
            str(output_dir),
            "--bar-size",
            "750",
            "--horizon",
            "36",
            "--fee-bps-per-side",
            "3",
            "--slippage-bps-per-side",
            "3",
            "--write-summary",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    summary = json.loads((output_dir / "c_exhaustion_paper_summary.json").read_text())
    assert summary["round_trip_cost_bps"] == 12.0
    assert "round_trip_cost_bps" not in captured.out
