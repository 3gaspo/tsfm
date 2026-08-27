"""Concise panel loading and deterministic strided evaluation windows.

The project evaluates univariate targets. A wide CSV therefore represents one
target series per non-date column. Optional known covariates may be global
columns in the same CSV or user-aligned wide panels supplied as separate paths.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)
PROJECT_SCOPE = "tsfm_evaluation"
PORTABLE_KEYS = {
    "date_col",
    "target_cols",
    "covariate_cols",
    "covariate_paths",
    "drop_users",
    "aggr",
    "aggr_period",
}
def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    return list(value)


def _unique(values: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _resolve_source(path: str | Path, name: str | None = None) -> Path:
    source = Path(path).expanduser().resolve()
    if source.is_file():
        return source
    if not source.exists():
        raise FileNotFoundError(source)
    if name and (source / f"{name}.csv").exists():
        return source / f"{name}.csv"
    if (source / f"{source.name}.csv").exists():
        return source / f"{source.name}.csv"
    csv_files = sorted(source.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    raise FileNotFoundError(f"could not identify one dataset file under {source}")


def _resolve_config_path(source: Path, explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        return path / "config.json" if path.is_dir() else path
    parent = source if source.is_dir() else source.parent
    candidate = parent / "config.json"
    return candidate if candidate.exists() else None


def _merge_dataset_options(
    source: Path,
    run_options: Mapping[str, Any],
) -> tuple[dict[str, Any], Path | None, list[str]]:
    config_path = _resolve_config_path(source, run_options.get("config_path"))
    raw: dict[str, Any] = {}
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        raw = json.loads(config_path.read_text(encoding="utf-8"))

    shared = {
        key: value
        for key, value in raw.items()
        if key in PORTABLE_KEYS and not key.startswith("_")
    }
    scoped = raw.get(PROJECT_SCOPE) or {}
    if not isinstance(scoped, Mapping):
        raise ValueError(f"{PROJECT_SCOPE!r} dataset config must be a mapping")

    effective = dict(shared)
    for key, value in scoped.items():
        if key in PORTABLE_KEYS and value is not None:
            effective[key] = value
    for key in PORTABLE_KEYS:
        value = run_options.get(key)
        if value is not None:
            effective[key] = value
    effective["drop_users"] = _unique(_as_list(effective.get("drop_users")))
    applied = sorted(key for key, value in effective.items() if value is not None)
    LOGGER.info(
        "dataset config path=%s applied_keys=%s",
        config_path if config_path is not None else "none",
        applied,
    )
    return effective, config_path, applied


def _read_csv(source: Path, options: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(source)
    date_col = options.get("date_col")
    if date_col is None:
        first = str(frame.columns[0])
        if first.lower() in {"date", "datetime", "timestamp", "time"}:
            date_col = first
    if date_col is not None:
        if date_col not in frame.columns:
            raise KeyError(f"date column {date_col!r} is absent from {source}")
        frame.index = pd.to_datetime(frame.pop(date_col))
    else:
        frame.index = pd.RangeIndex(len(frame))

    aggr = options.get("aggr")
    if aggr:
        period = options.get("aggr_period") or "h"
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("resampling requires a datetime index")
        if aggr == "sum":
            frame = frame.resample(period).sum()
        elif aggr == "mean":
            frame = frame.resample(period).mean()
        else:
            raise ValueError(f"unsupported aggregation {aggr!r}")
    return frame


def _drop_columns(columns: Sequence[str], drop_users: Sequence[Any]) -> list[str]:
    drop_names: list[str] = []
    for value in drop_users:
        if isinstance(value, str) and not value.lstrip("-").isdigit():
            drop_names.append(value)
        else:
            index = int(value)
            if index < 0 or index >= len(columns):
                raise IndexError(f"drop_users index {index} is outside {len(columns)} targets")
            drop_names.append(str(columns[index]))
    return [str(column) for column in columns if str(column) not in set(drop_names)]


def _load_csv_panel(
    source: Path,
    options: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    frame = _read_csv(source, options)
    covariate_cols = [str(value) for value in _as_list(options.get("covariate_cols"))]
    missing_covariates = [column for column in covariate_cols if column not in frame]
    if missing_covariates:
        raise KeyError(f"missing covariate columns: {missing_covariates}")

    configured_targets = options.get("target_cols")
    target_cols = (
        [str(value) for value in _as_list(configured_targets)]
        if configured_targets is not None
        else [str(column) for column in frame.columns if str(column) not in covariate_cols]
    )
    missing_targets = [column for column in target_cols if column not in frame]
    if missing_targets:
        raise KeyError(f"missing target columns: {missing_targets}")
    target_cols = _drop_columns(target_cols, _as_list(options.get("drop_users")))
    if not target_cols:
        raise ValueError("dataset has no target columns after drop_users")
    targets = frame[target_cols].astype(np.float32)
    covariates = frame[covariate_cols].astype(np.float32) if covariate_cols else None
    return targets, covariates


def _source_record(source: Path) -> dict[str, Any]:
    stat = source.stat()
    return {
        "source": str(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def _external_covariate(
    path: str | Path,
    target: pd.DataFrame,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source = _resolve_source(path)
    options, config_path, applied_keys = _merge_dataset_options(source, {})
    frame, _ = _load_csv_panel(source, options)
    if not frame.index.equals(target.index):
        raise ValueError(f"covariate timeline does not match targets: {source}")
    if set(target.columns).issubset(frame.columns):
        frame = frame[list(target.columns)]
    elif frame.shape[1] == target.shape[1]:
        frame.columns = target.columns
    elif frame.shape[1] == 1:
        values = np.repeat(frame.to_numpy(), target.shape[1], axis=1)
        frame = pd.DataFrame(values, index=target.index, columns=target.columns)
    else:
        raise ValueError(f"covariate users do not align with targets: {source}")
    metadata = {
        **_source_record(source),
        "config_path": None if config_path is None else str(config_path),
        "applied_config_keys": applied_keys,
        "effective_options": options,
    }
    tensor = torch.as_tensor(frame.to_numpy(copy=True).T.copy(), dtype=torch.float32).unsqueeze(1)
    return tensor, metadata


@dataclass
class PanelData:
    """A user panel with optional future-known, user-aligned covariates."""

    name: str
    values: torch.Tensor  # (users, 1, dates)
    datetimes: np.ndarray
    user_names: list[str]
    covariates: torch.Tensor | None = None  # (users, channels, dates)
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.values = torch.as_tensor(self.values, dtype=torch.float32)
        if self.values.ndim != 3 or self.values.shape[1] != 1:
            raise ValueError("values must have shape (users, 1, dates)")
        self.datetimes = np.asarray(self.datetimes)
        if len(self.datetimes) != self.values.shape[-1]:
            raise ValueError("datetime and value lengths differ")
        if len(self.user_names) != self.values.shape[0]:
            raise ValueError("user_names must contain one entry per user")
        if self.covariates is not None:
            self.covariates = torch.as_tensor(self.covariates, dtype=torch.float32)
            if self.covariates.shape[0] != self.values.shape[0]:
                raise ValueError("covariates and targets have different user counts")
            if self.covariates.shape[-1] != self.values.shape[-1]:
                raise ValueError("covariates and targets have different date counts")


def load_panel(config: Mapping[str, Any]) -> PanelData:
    """Load one panel and apply the shared/project/run dataset config contract."""
    source = _resolve_source(config["path"], config.get("name"))
    options, config_path, applied_keys = _merge_dataset_options(source, config)
    targets, global_covariates = _load_csv_panel(source, options)
    covariates = None
    if global_covariates is not None:
        global_values = torch.as_tensor(
            global_covariates.to_numpy(copy=True).T.copy(),
            dtype=torch.float32,
        )
        covariates = global_values.unsqueeze(0).expand(targets.shape[1], -1, -1).clone()

    external = [_external_covariate(path, targets) for path in _as_list(options.get("covariate_paths"))]
    if external:
        external_tensors = [tensor for tensor, _ in external]
        covariates = torch.cat(
            ([covariates] if covariates is not None else []) + external_tensors,
            dim=1,
        )

    values = torch.as_tensor(
        targets.to_numpy(copy=True).T.copy(),
        dtype=torch.float32,
    ).unsqueeze(1)
    mode = str(config.get("covariate_mode", "none")).lower()
    if mode == "none":
        covariates = None
    elif mode == "identity":
        covariates = values.clone()
    elif mode == "known":
        if covariates is None:
            raise ValueError("covariate_mode='known' requires covariate_cols, covariate_paths, or a context tensor")
    else:
        raise ValueError(f"unknown covariate_mode={mode!r}")

    metadata = {
        **_source_record(source),
        "config_path": None if config_path is None else str(config_path),
        "applied_config_keys": applied_keys,
        "effective_options": options,
        "covariate_mode": mode,
        "external_covariates": [item for _, item in external],
    }
    return PanelData(
        name=str(config.get("name") or (source.name if source.is_dir() else source.stem)),
        values=values,
        datetimes=targets.index.to_numpy(),
        user_names=[str(column) for column in targets.columns],
        covariates=covariates,
        metadata=metadata,
    )


class StridedWindowDataset(Dataset):
    """Date-major, all-user deterministic windows with optional pair filtering."""

    def __init__(
        self,
        panel: PanelData,
        lags: int,
        horizon: int,
        *,
        stride: int = 512,
        remove_constant: bool = False,
        constant_eps: float = 1e-8,
        start_fraction: float = 0.0,
        end_fraction: float = 1.0,
    ):
        self.panel = panel
        self.lags = int(lags)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.remove_constant = bool(remove_constant)
        self.constant_eps = float(constant_eps)
        dates = panel.values.shape[-1]
        if self.lags < 1 or self.horizon < 1 or self.stride < 1:
            raise ValueError("lags, horizon, and stride must be positive")
        if not 0.0 <= start_fraction < end_fraction <= 1.0:
            raise ValueError("evaluation fractions must satisfy 0 <= start < end <= 1")

        first_target = int(math.floor(start_fraction * dates))
        target_stop = int(math.ceil(end_fraction * dates))
        first_query = max(self.lags - 1, first_target - 1)
        last_query = min(dates - self.horizon - 1, target_stop - self.horizon - 1)
        if first_query > last_query:
            raise ValueError("no complete window exists for the selected interval")
        self.query_indices = list(range(first_query, last_query + 1, self.stride))
        self.pairs: list[tuple[int, int]] = []
        self.removed_constant_pairs = 0
        for query in self.query_indices:
            start = query - self.lags + 1
            lookbacks = panel.values[:, :, start : query + 1]
            non_constant = (
                torch.isfinite(lookbacks).all(dim=-1)
                & (lookbacks.std(dim=-1, unbiased=False) > self.constant_eps)
            ).any(dim=1)
            if not torch.isfinite(lookbacks).all():
                raise ValueError(f"non-finite lookback encountered at query index {query}")
            for user in range(panel.values.shape[0]):
                if self.remove_constant and not bool(non_constant[user]):
                    self.removed_constant_pairs += 1
                    continue
                self.pairs.append((user, query))
        if not self.pairs:
            raise ValueError("constant filtering removed every evaluation window")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        user, query = self.pairs[int(index)]
        start = query - self.lags + 1
        stop = query + self.horizon + 1
        values = self.panel.values[user, :, start:stop]
        if not torch.isfinite(values).all():
            raise ValueError(f"non-finite target window for user={user} query={query}")
        if self.panel.covariates is None:
            past_covariates = torch.empty((0, self.lags), dtype=torch.float32)
            future_covariates = torch.empty((0, self.horizon), dtype=torch.float32)
        else:
            context = self.panel.covariates[user, :, start:stop]
            if not torch.isfinite(context).all():
                raise ValueError(f"non-finite covariate window for user={user} query={query}")
            past_covariates = context[:, : self.lags]
            future_covariates = context[:, self.lags :]
        return {
            "inputs": values[:, : self.lags],
            "targets": values[:, self.lags :],
            "past_covariates": past_covariates,
            "future_covariates": future_covariates,
            "user_id": user,
            "user_name": self.panel.user_names[user],
            "query_index": query,
            "query_datetime": str(self.panel.datetimes[query]),
        }
