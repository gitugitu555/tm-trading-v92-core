import pandas as pd

from scripts.v92_alpha_strategy_test import evaluate_branch_events


def test_branch_b_default_collects_across_all_regimes():
    regimes = ["TREND_BUILDUP", "ABSORPTION", "EXHAUSTED", "NOISE"]
    rows = []
    ts = pd.Timestamp("2024-01-01T00:00:00")

    for regime_idx, regime in enumerate(regimes):
        base = 100.0 + regime_idx * 10
        # Warm-up rows establish rolling OFI thresholds and previous structure.
        rows.extend([
            {
                "datetime_close": ts + pd.Timedelta(seconds=len(rows)),
                "regime": regime,
                "open": base,
                "high": base + 2.0,
                "low": base - 2.0,
                "close": base,
                "volume": 500.0,
                "volume_delta": 0.0,
                "bar_ofi": 0.0,
                "has_ofi_coverage": True,
            },
            {
                "datetime_close": ts + pd.Timedelta(seconds=len(rows) + 1),
                "regime": regime,
                "open": base,
                "high": base + 2.0,
                "low": base - 2.0,
                "close": base,
                "volume": 500.0,
                "volume_delta": 0.0,
                "bar_ofi": 0.0,
                "has_ofi_coverage": True,
            },
            # Extreme sell OFI while price holds above prior local low -> Branch B long.
            {
                "datetime_close": ts + pd.Timedelta(seconds=len(rows) + 2),
                "regime": regime,
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base - 0.5,
                "volume": 600.0,
                "volume_delta": -10.0,
                "bar_ofi": -100.0,
                "has_ofi_coverage": True,
            },
            # Forward close row for horizon=1.
            {
                "datetime_close": ts + pd.Timedelta(seconds=len(rows) + 3),
                "regime": regime,
                "open": base,
                "high": base + 2.0,
                "low": base - 2.0,
                "close": base + 1.0,
                "volume": 500.0,
                "volume_delta": 0.0,
                "bar_ofi": 0.0,
                "has_ofi_coverage": True,
            },
        ])

    df = pd.DataFrame(rows)
    events = []
    evaluate_branch_events(df, events, threshold_window=3, structure_window=2, horizon=1)

    branch_b_regimes = {event["regime"] for event in events if event["branch"] == "B_Absorption"}

    assert branch_b_regimes == set(regimes)


def test_branch_b_requires_ofi_coverage():
    df = pd.DataFrame(
        [
            {
                "datetime_close": pd.Timestamp("2024-01-01T00:00:00"),
                "regime": "NOISE",
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 500.0,
                "volume_delta": 0.0,
                "bar_ofi": 0.0,
                "has_ofi_coverage": False,
            },
            {
                "datetime_close": pd.Timestamp("2024-01-01T00:00:01"),
                "regime": "NOISE",
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 500.0,
                "volume_delta": 0.0,
                "bar_ofi": 0.0,
                "has_ofi_coverage": False,
            },
            {
                "datetime_close": pd.Timestamp("2024-01-01T00:00:02"),
                "regime": "NOISE",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 99.5,
                "volume": 600.0,
                "volume_delta": -10.0,
                "bar_ofi": -100.0,
                "has_ofi_coverage": False,
            },
            {
                "datetime_close": pd.Timestamp("2024-01-01T00:00:03"),
                "regime": "NOISE",
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 101.0,
                "volume": 500.0,
                "volume_delta": 0.0,
                "bar_ofi": 0.0,
                "has_ofi_coverage": False,
            },
        ]
    )

    events = []
    evaluate_branch_events(df, events, threshold_window=3, structure_window=2, horizon=1)

    assert [event for event in events if event["branch"] == "B_Absorption"] == []
