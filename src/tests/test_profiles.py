from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pipeline.profiles import RANGE_SETTINGS, tasks_for_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _catalog(path: Path) -> Path:
    catalog = {
        "datasets": [
            {
                "name": "hourly_panel",
                "configured_frequency": "H",
                "num_timestamps": 1_000,
            },
            {
                "name": "daily_panel",
                "configured_frequency": "D",
                "num_timestamps": 100,
            },
            {
                "name": "quarter_hour_panel",
                "configured_frequency": "15T",
                "num_timestamps": 1_000,
            },
            {
                "name": "short_quarter_hour_panel",
                "configured_frequency": "15T",
                "num_timestamps": 200,
            },
        ]
    }
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


def _settings(tasks: list[dict[str, object]], dataset: str) -> set[str]:
    return {str(task["setting"]) for task in tasks if task["dataset"] == dataset}


def test_test_profile_uses_hourly_long_range_without_catalog() -> None:
    assert tasks_for_profile("standard", "test") == [
        {"dataset": "electricity", "setting": "504:168", "period": 168}
    ]


def test_full_profile_maps_known_and_time_dataset_cadences() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
        catalog = _catalog(Path(directory) / "catalog.json")
        tasks = tasks_for_profile("standard", "full", catalog_path=catalog)

    assert _settings(tasks, "electricity") == {
        f"{lags}:{horizon}" for lags, horizon in RANGE_SETTINGS["hourly"].values()
    }
    assert _settings(tasks, "exchange_rate") == {
        f"{lags}:{horizon}" for lags, horizon in RANGE_SETTINGS["daily"].values()
    }
    assert _settings(tasks, "time/hourly_panel") == {
        "168:24",
        "336:48",
        "504:168",
    }
    assert _settings(tasks, "time/daily_panel") == {"7:1", "14:2", "30:7"}
    assert _settings(tasks, "time/quarter_hour_panel") == {
        "96:4",
        "192:8",
        "672:96",
    }
    assert _settings(tasks, "time/short_quarter_hour_panel") == {"96:4", "192:8"}


def test_foundation_full_uses_mid_and_long_ranges() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
        catalog = _catalog(Path(directory) / "catalog.json")
        tasks = tasks_for_profile("foundation", "full", catalog_path=catalog)

    assert _settings(tasks, "electricity") == {"336:48", "504:168"}
    assert _settings(tasks, "exchange_rate") == {"14:2", "30:7"}
    assert _settings(tasks, "time/quarter_hour_panel") == {"192:8", "672:96"}
    assert _settings(tasks, "time/short_quarter_hour_panel") == {"192:8"}


def test_explicit_overrides_replace_automatic_catalog_grid() -> None:
    tasks = tasks_for_profile(
        "standard",
        "full",
        datasets_override="electricity exchange_rate",
        settings_override="12:3",
    )
    assert tasks == [
        {"dataset": "electricity", "setting": "12:3", "period": 168},
        {"dataset": "exchange_rate", "setting": "12:3", "period": 7},
    ]


if __name__ == "__main__":
    test_test_profile_uses_hourly_long_range_without_catalog()
    test_full_profile_maps_known_and_time_dataset_cadences()
    test_foundation_full_uses_mid_and_long_ranges()
    test_explicit_overrides_replace_automatic_catalog_grid()
