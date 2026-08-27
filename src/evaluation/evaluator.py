"""Deterministic inference, metrics, provenance, and completion artifacts."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import StridedWindowDataset, load_panel
from pipeline.runs import (
    allocate_run,
    identity_path,
    mark_ready,
    mark_status,
    validate_completed,
)
from model_loading import build_forecaster


LOGGER = logging.getLogger(__name__)
METRICS = ("mse", "mae", "nmse", "nmae", "mase")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _weekly_period_steps(datetimes: np.ndarray) -> int:
    """Infer the integer number of observations in seven days."""
    values = np.asarray(datetimes)
    if np.issubdtype(values.dtype, np.number):
        raise ValueError(
            "lookback baseline needs dated inputs or an explicit model.lookback_period"
        )
    try:
        timestamps = pd.to_datetime(values).to_numpy(dtype="datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "lookback baseline needs dated inputs or an explicit model.lookback_period"
        ) from exc
    differences = np.diff(timestamps.astype(np.int64))
    differences = differences[differences > 0]
    if not len(differences):
        raise ValueError("cannot infer lookback period from fewer than two distinct dates")
    cadence_ns = float(np.median(differences))
    steps = float(pd.Timedelta(days=7).value) / cadence_ns
    rounded = int(round(steps))
    if rounded < 1 or not np.isclose(steps, rounded, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "dataset cadence does not divide one week; set model.lookback_period explicitly"
        )
    return rounded


def _resolve_lookback_period(config: dict[str, Any], panel: Any) -> None:
    model = config["model"]
    if str(model.get("name", "")).casefold() != "lookback":
        return
    period = model.get("lookback_period")
    period = _weekly_period_steps(panel.datetimes) if period is None else int(period)
    if period < 1:
        raise ValueError("model.lookback_period must be positive")
    model["lookback_period"] = period
    LOGGER.info("lookback weekly period=%d observations", period)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(value), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _run_allocation(config: Mapping[str, Any], panel) -> tuple[Path, str]:
    output = config["output"]
    data = config["data"]
    task = config["task"]
    model = config["model"]
    preprocessing = config["preprocessing"]
    evaluation = config["evaluation"]
    model_config = {
        "covariate_mode": str(data.get("covariate_mode", "none")),
        "normalization": "instance" if preprocessing.get("instance_normalize", True) else "raw",
        "constant_policy": "remove" if evaluation.get("remove_constant", False) else "keep",
    }
    order = list(output.get("model_config_order") or model_config)
    root = identity_path(
        output.get("dir", "outputs/univariate"),
        panel.name,
        int(task["lags"]),
        int(task["horizon"]),
        str(model["name"]),
        order,
        model_config,
    )
    pipeline = {
        "data.input_contract": "csv_only_drop_replacement",
        "data.date_col": data.get("date_col"),
        "data.target_cols": data.get("target_cols"),
        "data.covariate_cols": data.get("covariate_cols"),
        "data.covariate_paths": data.get("covariate_paths"),
        "data.drop_users": (panel.metadata or {}).get("effective_options", {}).get(
            "drop_users", []
        ),
        "data.aggr": data.get("aggr"),
        "data.aggr_period": data.get("aggr_period"),
        "model.local_files_only": model.get("local_files_only", True),
        "model.cross_learning": model.get("cross_learning", False),
        "model.quantile_index": model.get("quantile_index"),
        "model.quantile_level": model.get("quantile_level", 0.5),
        "model.seasonal_periods": model.get("seasonal_periods"),
        "model.lookback_period": model.get("lookback_period"),
        "model.kwargs": model.get("kwargs", {}),
        "preprocessing.eps": preprocessing.get("eps", 1e-8),
        "evaluation.stride": evaluation.get("stride", 512),
        "evaluation.start_fraction": evaluation.get("start_fraction", 0.0),
        "evaluation.end_fraction": evaluation.get("end_fraction", 1.0),
        "evaluation.save_window_metrics": evaluation.get("save_window_metrics", True),
        "evaluation.mase_seasonality": evaluation.get("mase_seasonality", 1),
    }
    runtime = {
        "evaluation.batch_size": int(evaluation.get("batch_size", 64)),
        "model.device": str(model.get("device", "cuda")),
    }
    inputs = {"dataset": _plain(panel.metadata or {})}
    weights = model.get("weights_path")
    if weights:
        inputs["weights"] = str(Path(weights).expanduser())
    allocation = allocate_run(
        root,
        project="tsfm_evaluation",
        workflow=str(output.get("workflow", "univariate")),
        dataset=panel.name,
        lookback=int(task["lags"]),
        horizon=int(task["horizon"]),
        backbone=str(model["name"]),
        model_config_order=order,
        model_config=model_config,
        pipeline_config=pipeline,
        runtime_config=runtime,
        seeds=[int(evaluation.get("seed", 1))],
        purpose=str(output.get("purpose", "development")),
        mode=output.get("mode"),
        display_name=str(model["name"]),
        row_config=list(output.get("row_config") or []),
        column_config=list(output.get("column_config") or []),
        inputs=inputs,
        policy=str(output.get("conflict_policy", "overwrite_exact")),
        skip_completed=bool(output.get("skip_completed", True)),
        force=bool(output.get("force", False)),
        run_index=output.get("run_index"),
        launch_id=output.get("launch_id"),
    )
    return allocation.run_dir, allocation.action


def summarize_window_metrics(frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Aggregate sample- and equal-user-weighted error summaries."""
    if frame.empty:
        raise ValueError("cannot summarize an empty evaluation")
    metric_names = [name for name in METRICS if name in frame]
    grouped = frame.groupby(["user_id", "user_name"], sort=True)
    per_user = grouped.size().rename("windows").to_frame()
    for metric in metric_names:
        per_user[metric] = grouped[metric].mean()
        per_user[f"std_{metric}"] = grouped[metric].std(ddof=0)
    per_user = per_user.reset_index().sort_values("user_id")
    summary: dict[str, float] = {}
    for metric in metric_names:
        user_values = per_user[metric].to_numpy(dtype=float)
        tail = max(1, math.ceil(0.1 * len(user_values)))
        summary[metric] = float(frame[metric].mean())
        summary[f"sample_std_{metric}"] = float(frame[metric].std(ddof=0))
        summary[f"user_mean_{metric}"] = float(user_values.mean())
        summary[f"user_std_{metric}"] = float(user_values.std(ddof=0))
        summary[f"w10_{metric}"] = float(np.sort(user_values)[-tail:].mean())
    return summary, per_user


def _metric_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lookback: torch.Tensor,
    eps: float,
    mase_seasonality: int,
) -> dict[str, torch.Tensor]:
    error = prediction - target
    absolute_error = error.abs()
    scale = lookback.std(dim=-1, keepdim=True, unbiased=False) + float(eps)
    seasonality = int(mase_seasonality)
    if seasonality < 1 or lookback.shape[-1] <= seasonality:
        raise ValueError("MASE seasonality must be positive and smaller than the lookback")
    mase_scale = (
        lookback[..., seasonality:] - lookback[..., :-seasonality]
    ).abs().mean(dim=-1, keepdim=True).clamp_min(float(eps))
    return {
        "mse": error.square(),
        "mae": absolute_error,
        "nmse": (error / scale).square(),
        "nmae": absolute_error / scale,
        "mase": absolute_error / mase_scale,
    }


def _window_rows(
    metric_tensors: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
) -> list[dict[str, Any]]:
    window_values = {
        metric: values.mean(dim=(1, 2)).detach().cpu()
        for metric, values in metric_tensors.items()
    }
    return [
        {
            "user_id": int(batch["user_id"][index]),
            "user_name": str(batch["user_name"][index]),
            "query_index": int(batch["query_index"][index]),
            "query_datetime": str(batch["query_datetime"][index]),
            **{
                metric: float(values[index])
                for metric, values in window_values.items()
            },
        }
        for index in range(next(iter(metric_tensors.values())).shape[0])
    ]


def _horizon_frame(
    sums: Mapping[str, np.ndarray],
    sums_of_squares: Mapping[str, np.ndarray],
    count: int,
) -> pd.DataFrame:
    if count < 1:
        raise ValueError("cannot summarize empty horizon metrics")
    result: dict[str, Any] = {"horizon": np.arange(1, len(next(iter(sums.values()))) + 1)}
    for metric in METRICS:
        mean = sums[metric] / count
        variance = np.maximum(sums_of_squares[metric] / count - np.square(mean), 0.0)
        result[metric] = mean
        result[f"std_{metric}"] = np.sqrt(variance)
    return pd.DataFrame(result)


def evaluate(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run one fully resolved inference configuration and save its artifacts."""
    config = _plain(config)
    evaluation_config = config["evaluation"]
    seed = int(evaluation_config.get("seed", 1))
    _seed_everything(seed)

    panel = load_panel(config["data"])
    _resolve_lookback_period(config, panel)
    lags = int(config["task"]["lags"])
    horizon = int(config["task"]["horizon"])
    eps = float(config["preprocessing"].get("eps", 1e-8))
    mase_seasonality = int(evaluation_config.get("mase_seasonality", 1))
    windows = StridedWindowDataset(
        panel,
        lags,
        horizon,
        stride=int(evaluation_config.get("stride", 512)),
        remove_constant=bool(evaluation_config.get("remove_constant", False)),
        constant_eps=eps,
        start_fraction=float(evaluation_config.get("start_fraction", 0.0)),
        end_fraction=float(evaluation_config.get("end_fraction", 1.0)),
    )
    run_dir, run_action = _run_allocation(config, panel)
    save_windows = bool(evaluation_config.get("save_window_metrics", True))
    if run_action == "skip":
        validate_completed(run_dir)
        LOGGER.info("skip complete run=%s", run_dir)
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    mark_status(run_dir, "running")

    LOGGER.info(
        "evaluate dataset=%s users=%d dates=%d windows=%d removed_constant=%d L=%d H=%d stride=%d",
        panel.name,
        panel.values.shape[0],
        panel.values.shape[-1],
        len(windows),
        windows.removed_constant_pairs,
        lags,
        horizon,
        windows.stride,
    )
    loader = DataLoader(
        windows,
        batch_size=int(evaluation_config.get("batch_size", 64)),
        shuffle=False,
        num_workers=0,
    )
    requested_device = str(config["model"].get("device", "cuda"))
    device = _device(requested_device)
    model_config = dict(config["model"])
    model_config["device"] = str(device)
    load_start = perf_counter()
    model = build_forecaster(
        model_config,
        config["preprocessing"],
        lags=lags,
        horizon=horizon,
    ).to(device)
    model.eval()
    model_load_seconds = perf_counter() - load_start

    rows: list[dict[str, Any]] = []
    horizon_sums = {name: np.zeros(horizon, dtype=np.float64) for name in METRICS}
    horizon_sums_of_squares = {
        name: np.zeros(horizon, dtype=np.float64) for name in METRICS
    }
    horizon_count = 0
    inference_seconds = 0.0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            past = batch["past_covariates"].to(device)
            future = batch["future_covariates"].to(device)
            if past.shape[1] == 0:
                past = future = None
            _sync(device)
            started = perf_counter()
            prediction = model(
                inputs,
                past_covariates=past,
                future_covariates=future,
            )
            _sync(device)
            inference_seconds += perf_counter() - started
            if prediction.shape != targets.shape:
                raise ValueError(
                    f"prediction shape {tuple(prediction.shape)} != target shape {tuple(targets.shape)}"
                )
            metric_tensors = _metric_tensors(
                prediction,
                targets,
                inputs,
                eps,
                mase_seasonality,
            )
            rows.extend(_window_rows(metric_tensors, batch))
            for metric, values in metric_tensors.items():
                per_sample_horizon = values.mean(dim=1).detach().cpu().double().numpy()
                horizon_sums[metric] += per_sample_horizon.sum(axis=0)
                horizon_sums_of_squares[metric] += np.square(per_sample_horizon).sum(axis=0)
            horizon_count += int(prediction.shape[0])

    window_frame = pd.DataFrame(rows)
    metrics, per_user = summarize_window_metrics(window_frame)
    horizon_frame = _horizon_frame(
        horizon_sums,
        horizon_sums_of_squares,
        horizon_count,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    per_user.to_csv(run_dir / "per_user_metrics.csv", index=False)
    horizon_frame.to_csv(run_dir / "horizon_metrics.csv", index=False)
    if save_windows:
        window_frame.to_csv(run_dir / "window_metrics.csv", index=False)

    summary = {
        "dataset": panel.name,
        "model": str(config["model"]["name"]),
        "lookback_period": config["model"].get("lookback_period"),
        "covariate_mode": str(config["data"].get("covariate_mode", "none")),
        "instance_normalize": bool(config["preprocessing"].get("instance_normalize", True)),
        "remove_constant": bool(evaluation_config.get("remove_constant", False)),
        "lags": lags,
        "horizon": horizon,
        "stride": windows.stride,
        "seed": seed,
        "users": int(per_user.shape[0]),
        "series_windows": int(window_frame.shape[0]),
        "query_dates": len(windows.query_indices),
        "removed_constant_windows": windows.removed_constant_pairs,
        "mase_seasonality": mase_seasonality,
        "metrics": metrics,
        "inference": {
            "seconds": inference_seconds,
            "seconds_per_user": inference_seconds / len(per_user),
            "seconds_per_series_window": inference_seconds / len(window_frame),
            "model_load_seconds": model_load_seconds,
            "device": str(device),
            "batch_size": int(evaluation_config.get("batch_size", 64)),
        },
        "data": panel.metadata,
        "run_dir": str(run_dir),
    }
    _write_json(config, run_dir / "resolved_config.json")
    _write_json(summary, run_dir / "summary.json")
    required = [
        "summary.json",
        "per_user_metrics.csv",
        "horizon_metrics.csv",
        "resolved_config.json",
    ]
    if save_windows:
        required.append("window_metrics.csv")
    if os.environ.get("DEFER_MANIFEST_COMPLETION") == "1":
        mark_ready(run_dir, required_artifacts=required)
    else:
        mark_status(run_dir, "completed", required_artifacts=required)
    LOGGER.info(
        "artifacts ready run=%s mse=%.6g nmse=%.6g inference_seconds=%.3f",
        run_dir,
        metrics["mse"],
        metrics["nmse"],
        inference_seconds,
    )
    return summary
