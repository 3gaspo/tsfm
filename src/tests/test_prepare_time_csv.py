from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_panel
from data.time import DEFAULT_SETTINGS_BY_FREQUENCY
from scripts.prepare_time_csv import prepare_time_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_series(
    path: Path,
    timestamps: pd.DatetimeIndex,
    columns: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": timestamps, **columns}).to_csv(path, index=False)


def test_hourly_alignment_length_filter_and_sample_catalog() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
        root = Path(directory)
        source = root / "source"
        output = root / "prepared"
        aligned = pd.date_range("2025-01-01", periods=10, freq="h")
        shifted = pd.date_range("2025-02-01", periods=10, freq="h")

        _write_series(source / "A" / "H" / "item0.csv", aligned, {"load": np.arange(10)})
        _write_series(
            source / "A" / "H" / "item1.csv",
            aligned,
            {"load": np.arange(10) * 2, "missing": np.full(10, np.nan)},
        )
        _write_series(source / "A" / "H" / "item2.csv", shifted, {"load": np.arange(10)})
        _write_series(
            source / "B" / "H" / "item0.csv", aligned[:5], {"load": np.arange(5)}
        )
        _write_series(source / "C" / "D" / "item0.csv", aligned, {"load": np.arange(10)})

        catalog = prepare_time_csv(
            output_root=output,
            source_root=source,
            settings=[(4, 2)],
            stride=2,
            frequencies=["H"],
        )

        assert catalog["selected_source_files"] == 4
        assert catalog["num_datasets"] == 2
        assert catalog["num_series"] == 3
        assert catalog["num_timestamps"] == 20
        assert catalog["num_values"] == 30
        assert [entry["name"] for entry in catalog["datasets"]] == [
            "a_h_part01",
            "a_h_part02",
        ]

        first_config = json.loads(
            (output / "a_h_part01" / "config.json").read_text(encoding="utf-8")
        )
        assert first_config["target_cols"] == ["item0__load", "item1__load"]
        metadata = first_config["time"]
        assert metadata["num_samples"] == metadata["num_timestamps"] == 10
        assert metadata["num_series"] == 2
        assert metadata["num_values"] == 20
        assert metadata["dropped_series"] == ["A/H/item1.csv:missing"]
        assert metadata["evaluation_samples"]["4:2"] == {
            "eligible": True,
            "query_dates_stride_1": 5,
            "series_windows_stride_1": 10,
            "query_dates_stride_2": 3,
            "series_windows_stride_2": 6,
        }

        panel = load_panel(
            {
                "path": str(output / "a_h_part01"),
                "name": "a_h_part01",
                "covariate_mode": "none",
            }
        )
        assert panel.values.shape == (2, 1, 10)
        assert panel.user_names == ["item0__load", "item1__load"]
        assert any(item["reason"].startswith("too short") for item in catalog["skipped"])
        assert not (output / "c_d").exists()
        assert json.loads((output / "catalog.json").read_text(encoding="utf-8")) == catalog


def test_task_size_limits_apply_before_alignment_split() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
        root = Path(directory)
        source = root / "source"
        output = root / "prepared"
        dates = pd.date_range("2025-01-01", periods=10, freq="h")
        shifted = pd.date_range("2025-02-01", periods=10, freq="h")
        _write_series(source / "TooWide" / "H" / "item0.csv", dates, {"x": np.arange(10)})
        _write_series(
            source / "TooWide" / "H" / "item1.csv", shifted, {"x": np.arange(10)}
        )
        _write_series(
            source / "TooLong" / "H" / "item0.csv",
            pd.date_range("2025-01-01", periods=11, freq="h"),
            {"x": np.arange(11)},
        )
        _write_series(source / "Keep" / "H" / "item0.csv", dates, {"x": np.arange(10)})

        catalog = prepare_time_csv(
            output_root=output,
            source_root=source,
            settings=[(4, 2)],
            frequencies=["H"],
            max_series=1,
            max_dates_per_series=10,
        )

        assert [entry["name"] for entry in catalog["datasets"]] == ["keep_h"]
        reasons = [item["reason"] for item in catalog["skipped"]]
        assert "too many series: 2 > 1" in reasons
        assert "series too long: 11 > 10 dates" in reasons
        assert catalog["max_series"] == 1
        assert catalog["max_dates_per_series"] == 10


def test_default_settings_follow_each_source_cadence() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
        root = Path(directory)
        source = root / "source"
        output = root / "prepared"
        cases = (
            ("Hourly", "H", "h", 800, "hourly"),
            ("Daily", "D", "D", 50, "daily"),
            ("QuarterHour", "15T", "15min", 800, "15min"),
        )
        for dataset, source_frequency, pandas_frequency, length, _ in cases:
            _write_series(
                source / dataset / source_frequency / "item0.csv",
                pd.date_range("2025-01-01", periods=length, freq=pandas_frequency),
                {"x": np.arange(length)},
            )

        catalog = prepare_time_csv(output_root=output, source_root=source)
        entries = {entry["source_frequency"]: entry for entry in catalog["datasets"]}
        for _, source_frequency, _, _, normalized in cases:
            expected = {
                f"{lookback}:{horizon}"
                for lookback, horizon in DEFAULT_SETTINGS_BY_FREQUENCY[normalized]
            }
            assert set(entries[source_frequency]["evaluation_samples"]) == expected


if __name__ == "__main__":
    test_hourly_alignment_length_filter_and_sample_catalog()
    test_task_size_limits_apply_before_alignment_split()
    test_default_settings_follow_each_source_cadence()
