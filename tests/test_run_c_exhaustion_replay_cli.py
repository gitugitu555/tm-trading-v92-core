from __future__ import annotations

import json
from pathlib import Path

import polars as pl

import scripts.run_c_exhaustion_replay as cli


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
        else:
            low = base_low + idx * 2.0
            close = low + 4.0
            regime = "NOISE"
        rows.append(
            {
                "open_time": open_time,
                "close_time": close_time,
                "open": base_open + idx,
                "high": base_open + idx + 8.0,
                "low": low,
                "close": close,
                "volume": 1000.0 + idx,
                "volume_delta": 25.0,
                "trade_count": 50,
                "regime": regime,
            }
        )
    return rows


def test_cli_runs_and_writes_local_reports(tmp_path, monkeypatch, capsys):
    bar_dir = tmp_path / "bars"
    output_dir = tmp_path / "reports"
    bar_dir.mkdir()

    rows = _make_rows()
    _write_parquet_shard(bar_dir / "chunk_0.parquet", rows[:50])
    _write_parquet_shard(bar_dir / "chunk_1.parquet", rows[50:])

    monkeypatch.setattr(cli, "add_v92_regime_labels", lambda df: df)

    rc = cli.main(
        [
            "--bar-dir",
            str(bar_dir),
            "--output-dir",
            str(output_dir),
            "--write-json",
            "--write-csv",
            "--write-summary",
            "--bar-size",
            "750",
            "--horizon",
            "36",
            "--cost-bps",
            "12",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "branch=C_ExhaustionFade" in captured.out
    assert "side=long_only" in captured.out
    assert "bar_size=750" in captured.out
    assert "horizon=36" in captured.out
    assert "cost_bps=12.0" in captured.out
    assert "production_path_touched=false" in captured.out

    report_json = output_dir / "c_exhaustion_replay_report.json"
    report_csv = output_dir / "c_exhaustion_replay_trades.csv"
    report_txt = output_dir / "c_exhaustion_replay_summary.txt"

    assert report_json.exists()
    assert report_csv.exists()
    assert report_txt.exists()

    report = json.loads(report_json.read_text())
    assert report["metadata"]["branch"] == "C_ExhaustionFade"
    assert report["metadata"]["side"] == "long_only"
    assert report["metadata"]["bar_size"] == 750
    assert report["metadata"]["horizon"] == 36
    assert report["metadata"]["cost_bps"] == 12.0
    assert report["summary"]["production_path_touched"] is False
    assert report["summary"]["trade_count"] == 1
    assert report["trades"][0]["entry_price"] == 1000.0 + 61
    assert report["trades"][0]["signal_index"] == 60

    summary_text = report_txt.read_text()
    assert "branch=C_ExhaustionFade" in summary_text
    assert "side=long_only" in summary_text
    assert "production_path_touched=false" in summary_text

    assert not (Path(cli.ROOT) / "data" / "hft" / "tier2" / "c_exhaustion_replay_report.json").exists()
    assert not (Path(cli.ROOT) / "data" / "hft" / "tier2" / "c_exhaustion_replay_trades.csv").exists()
    assert not (Path(cli.ROOT) / "data" / "hft" / "tier2" / "c_exhaustion_replay_summary.txt").exists()


def test_cli_fails_cleanly_on_empty_bar_dir(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    rc = cli.main(["--bar-dir", str(empty_dir), "--output-dir", str(tmp_path / "reports")])
    captured = capsys.readouterr()

    assert rc == 1
    assert "No parquet files found" in captured.err

