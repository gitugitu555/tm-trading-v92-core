import json
from pathlib import Path

import pytest

import scripts.v92_tier2_cache_builder as tier2_builder


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder")
    return path


def _run_manifest_output(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]) -> dict:
    called = {"process_archive_lazy": False}

    def _fail_if_called(*_args, **_kwargs):
        called["process_archive_lazy"] = True
        raise AssertionError("process_archive_lazy must not run in dry-run mode")

    monkeypatch.setattr(tier2_builder, "process_archive_lazy", _fail_if_called)

    exit_code = tier2_builder.main(
        [
            "--dry-run",
            "--manifest-json",
            "--search-dir",
            str(tmp_path / "archives"),
            "--tier2-dir",
            str(tmp_path / "tier2"),
            *args,
        ]
    )
    assert exit_code in {0, 1}
    assert called["process_archive_lazy"] is False
    return json.loads(capsys.readouterr().out)


def test_manifest_monthly_wins_counts_selected_archives(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archives = tmp_path / "archives"
    tier2 = tmp_path / "tier2"
    touch(archives / "BTCUSDT-aggTrades-2021-05.zip")
    touch(archives / "BTCUSDT-aggTrades-2021-05-01.zip")
    touch(archives / "BTCUSDT-aggTrades-2021-05-31.zip")
    touch(tier2 / "BTCUSDT_tier2_500btc_ALL.parquet")

    manifest = _run_manifest_output(capsys, tmp_path, monkeypatch, [])

    assert manifest["overlap_policy"] == "monthly_wins"
    assert manifest["selected_archive_count"] == 1
    assert manifest["selected_archives"] == [str(archives / "BTCUSDT-aggTrades-2021-05.zip")]
    assert manifest["side_effects_disabled"] is True


def test_manifest_daily_wins_counts_selected_archives(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archives = tmp_path / "archives"
    tier2 = tmp_path / "tier2"
    touch(archives / "BTCUSDT-aggTrades-2021-05.zip")
    touch(archives / "BTCUSDT-aggTrades-2021-05-01.zip")
    touch(archives / "BTCUSDT-aggTrades-2021-05-31.zip")
    touch(tier2 / "BTCUSDT_tier2_500btc_ALL.parquet")

    manifest = _run_manifest_output(capsys, tmp_path, monkeypatch, ["--overlap-policy", "daily_wins"])

    assert manifest["overlap_policy"] == "daily_wins"
    assert manifest["selected_archive_count"] == 2
    assert manifest["selected_archives"] == [
        str(archives / "BTCUSDT-aggTrades-2021-05-01.zip"),
        str(archives / "BTCUSDT-aggTrades-2021-05-31.zip"),
    ]


def test_manifest_reject_overlap_reports_errors(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archives = tmp_path / "archives"
    tier2 = tmp_path / "tier2"
    touch(archives / "BTCUSDT-aggTrades-2021-05.zip")
    touch(archives / "BTCUSDT-aggTrades-2021-05-01.zip")
    touch(tier2 / "BTCUSDT_tier2_500btc_ALL.parquet")

    manifest = _run_manifest_output(capsys, tmp_path, monkeypatch, ["--overlap-policy", "reject_overlap"])

    assert manifest["rejected_overlaps"] == ["2021-05"]
    assert manifest["errors"]
    assert manifest["selected_archive_count"] == 0


def test_manifest_reports_unsupported_filenames_separately(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archives = tmp_path / "archives"
    tier2 = tmp_path / "tier2"
    touch(archives / "BTCUSDT-aggTrades-2021-05.zip")
    touch(archives / "BTCUSDT-aggTrades-2022-05-09_1m.zip")
    touch(tier2 / "BTCUSDT_tier2_500btc_ALL.parquet")

    manifest = _run_manifest_output(capsys, tmp_path, monkeypatch, [])

    assert manifest["unsupported_filenames"] == ["BTCUSDT-aggTrades-2022-05-09_1m.zip"]
    assert manifest["selected_archive_count"] == 1


def test_manifest_never_returns_all_and_shards_together(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archives = tmp_path / "archives"
    tier2 = tmp_path / "tier2"
    touch(archives / "BTCUSDT-aggTrades-2021-05.zip")
    touch(tier2 / "BTCUSDT_tier2_500btc_ALL.parquet")
    touch(tier2 / "BTCUSDT_tier2_500btc_2021-05.parquet")
    touch(tier2 / "BTCUSDT_tier2_500btc_2021-06.parquet")

    manifest = _run_manifest_output(capsys, tmp_path, monkeypatch, [])

    assert manifest["selected_tier2_files"] == [str(tier2 / "BTCUSDT_tier2_500btc_ALL.parquet")]
    assert manifest["selected_tier2_count"] == 1
    assert manifest["side_effects_disabled"] is True
