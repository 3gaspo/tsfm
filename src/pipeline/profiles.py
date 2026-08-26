"""Cadence-aware dataset and setting profiles for TSFM workflows."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from data.cadence import (
    DATASET_FREQUENCIES,
    LOOKBACK_PERIOD_BY_FREQUENCY,
    RANGE_NAMES,
    RANGE_SETTINGS,
    normalize_frequency,
)

LOGGER = logging.getLogger(__name__)

STANDARD_DATASETS = ("electricity", "traffic", "solar", "exchange_rate")
FOUNDATION_DATASETS = (
    "electricity",
    "traffic",
    "solar",
    "weather",
    "exchange_rate",
)


def load_time_metadata(catalog_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if catalog_path is None:
        return {}
    path = Path(catalog_path).expanduser().resolve()
    catalog = json.loads(path.read_text(encoding="utf-8"))
    return {f"time/{item['name']}": item for item in catalog["datasets"]}


def dataset_frequency(dataset: str, metadata: dict[str, dict[str, Any]]) -> str:
    known = DATASET_FREQUENCIES.get(str(dataset).casefold())
    if known is not None:
        return known
    values = metadata.get(str(dataset))
    if values is None:
        raise KeyError(f"dataset {dataset!r} has no cadence metadata")
    frequency = values.get("configured_frequency", values.get("source_frequency"))
    if frequency is None:
        raise KeyError(f"TIME dataset {dataset!r} has no configured frequency")
    return normalize_frequency(str(frequency))


def _split_override(value: str | None) -> list[str]:
    return [item for item in (value or "").replace(",", " ").split() if item]


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _profile_datasets(
    profile: str,
    mode: str,
    metadata: dict[str, dict[str, Any]],
    override: str | None,
) -> list[str]:
    selected = _split_override(override)
    if selected:
        return _unique(selected)
    if mode == "test":
        return ["electricity"]
    base = list(FOUNDATION_DATASETS if profile == "foundation" else STANDARD_DATASETS)
    if mode == "ultra":
        base.extend(["weather", "etth1"])
    if not metadata:
        raise FileNotFoundError(
            f"{mode} mode requires datasets/time/catalog.json for automatic TIME discovery"
        )
    return _unique([*base, *metadata])


def _range_names(profile: str, mode: str) -> tuple[str, ...]:
    if mode == "test":
        return ("long",)
    if profile == "foundation" and mode == "full":
        return ("mid", "long")
    return RANGE_NAMES


def _parse_setting(value: str) -> tuple[int, int]:
    left, separator, right = value.partition(":")
    if not separator:
        raise ValueError(f"setting must be L:H, got {value!r}")
    lookback, horizon = int(left), int(right)
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")
    return lookback, horizon


def tasks_for_profile(
    profile: str,
    mode: str,
    *,
    catalog_path: str | Path | None = None,
    datasets_override: str | None = None,
    settings_override: str | None = None,
) -> list[dict[str, Any]]:
    if profile not in {"standard", "foundation"}:
        raise ValueError("profile must be standard or foundation")
    if mode not in {"test", "full", "ultra"}:
        raise ValueError("mode must be test, full, or ultra")
    metadata = load_time_metadata(catalog_path)
    datasets = _profile_datasets(profile, mode, metadata, datasets_override)
    explicit_settings = [
        _parse_setting(value) for value in _split_override(settings_override)
    ]
    tasks: list[dict[str, Any]] = []
    for dataset in datasets:
        frequency = dataset_frequency(dataset, metadata)
        settings = explicit_settings or [
            RANGE_SETTINGS[frequency][name] for name in _range_names(profile, mode)
        ]
        for lookback, horizon in settings:
            values = metadata.get(dataset)
            if values is not None and int(values["num_timestamps"]) < lookback + horizon:
                LOGGER.info(
                    "skip ineligible TIME task dataset=%s L=%s H=%s dates=%s required=%s",
                    dataset,
                    lookback,
                    horizon,
                    values["num_timestamps"],
                    lookback + horizon,
                )
                continue
            tasks.append(
                {
                    "dataset": dataset,
                    "setting": f"{lookback}:{horizon}",
                    "period": LOOKBACK_PERIOD_BY_FREQUENCY[frequency],
                }
            )
    if not tasks:
        raise ValueError("profile produced no eligible dataset-setting tasks")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("standard", "foundation"), required=True)
    parser.add_argument("--mode", choices=("test", "full", "ultra"), required=True)
    parser.add_argument("--catalog")
    parser.add_argument("--datasets-override")
    parser.add_argument("--settings-override")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for task in tasks_for_profile(
        args.profile,
        args.mode,
        catalog_path=args.catalog,
        datasets_override=args.datasets_override,
        settings_override=args.settings_override,
    ):
        print(f"{task['dataset']}\t{task['setting']}\t{task['period']}")


if __name__ == "__main__":
    main()
