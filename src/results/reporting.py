"""Build tables, comparisons, and plots from completed TSFM evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.cadence import forecast_range_name
from pipeline.runs import (
    SelectedRun,
    load_manifest,
    manifest_is_selectable,
    select_identity_runs,
    write_report_manifest,
)
from visualization.reporting import build_plots



ERROR_METRICS = ("mse", "mae", "nmse", "nmae", "mase")
TIMING_METRICS = (
    "inference_seconds",
    "inference_seconds_per_user",
    "inference_seconds_per_series_window",
)
TABLE_METRICS = (*ERROR_METRICS, *TIMING_METRICS)
BASELINE_MODELS = {"persistence", "expected", "repeat", "lookback"}
AVERAGED_ARTIFACTS = {
    "window_metrics.csv": ("user_id", "user_name", "query_index", "query_datetime"),
    "per_user_metrics.csv": ("user_id", "user_name"),
    "horizon_metrics.csv": ("horizon",),
}


def _flatten(prefix: str, value: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            out.update(_flatten(name, item))
        elif not isinstance(item, (list, dict)):
            out[name] = item
    return out


def _names(value: str | None) -> set[str]:
    return {item.strip() for item in (value or "").replace(";", ",").split(",") if item.strip()}


def _pipeline_pairs(values: list[str] | None) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"pipeline config must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        lowered = value.casefold()
        if lowered in {"true", "false"}:
            selected[key] = lowered == "true"
        else:
            try:
                selected[key] = int(value)
            except ValueError:
                try:
                    selected[key] = float(value)
                except ValueError:
                    selected[key] = value
    return selected


def _task_pairs(values: list[str] | None) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    for item in values or []:
        dataset, separator, setting = item.rpartition("=")
        if not separator or not dataset or not re.fullmatch(r"[1-9]\d*:[1-9]\d*", setting):
            raise ValueError(f"task must be DATASET=L:H, got {item!r}")
        selected.add((dataset, setting))
    return selected


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "value"


def _is_baseline(model: Any) -> bool:
    name = str(model).casefold()
    return name in BASELINE_MODELS


def _is_chronos(model: Any) -> bool:
    return str(model) == "chronos2"


def _latex_escape(value: Any) -> str:
    return str(value).replace("_", r"\_")


def _group_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "dataset",
        "lags",
        "horizon",
        "covariate_mode",
        "instance_normalize",
        "remove_constant",
        "stride",
        "seed",
    ]
    return [name for name in candidates if name in frame]


def _read_csv(run_dir: Path, name: str) -> pd.DataFrame | None:
    path = run_dir / name
    return pd.read_csv(path) if path.is_file() else None


def _average_artifact(
    run_dirs: list[Path],
    name: str,
    keys: tuple[str, ...],
    output: Path,
) -> None:
    frames = [_read_csv(run_dir, name) for run_dir in run_dirs]
    present = [frame is not None for frame in frames]
    if not any(present):
        return
    if not all(present):
        missing = [str(path / name) for path, exists in zip(run_dirs, present) if not exists]
        raise FileNotFoundError(
            f"cannot average {name}; selected repeats are missing {missing}"
        )
    loaded = [frame for frame in frames if frame is not None]
    first_columns = list(loaded[0].columns)
    indexed: list[pd.DataFrame] = []
    for frame in loaded:
        missing_keys = [key for key in keys if key not in frame]
        if missing_keys:
            raise ValueError(f"{name} is missing alignment keys {missing_keys}")
        if set(frame.columns) != set(first_columns):
            raise ValueError(f"selected {name} artifacts have different columns")
        if frame.duplicated(list(keys)).any():
            raise ValueError(f"{name} contains duplicate alignment keys {keys}")
        indexed.append(frame.set_index(list(keys)).sort_index())
    reference = indexed[0]
    if any(not frame.index.equals(reference.index) for frame in indexed[1:]):
        raise ValueError(f"selected {name} artifacts do not contain the same aligned rows")

    averaged = reference.copy()
    for column in reference.columns:
        values = [frame[column] for frame in indexed]
        if all(pd.api.types.is_numeric_dtype(value.dtype) for value in values):
            averaged[column] = np.mean(
                np.stack([value.to_numpy(dtype=float) for value in values]),
                axis=0,
            )
        elif any(not value.equals(values[0]) for value in values[1:]):
            raise ValueError(
                f"selected {name} artifacts disagree on non-numeric column {column!r}"
            )
    output.mkdir(parents=True, exist_ok=True)
    averaged.reset_index()[first_columns].to_csv(output / name, index=False)


def _average_analysis_frame(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Average selected runs and their aligned analysis artifacts."""
    group_columns = ["identity_signature", "run_label"]
    rows: list[dict[str, Any]] = []
    for _, group in frame.groupby(group_columns, dropna=False, sort=False):
        run_dirs = [Path(value) for value in group["_run_dir"]]
        source_key = "|".join(str(path.resolve()) for path in run_dirs)
        source_hash = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:12]
        first = group.iloc[0]
        averaged_dir = (
            output
            / "averaged_inputs"
            / _safe_name(first["identity_signature"])
            / f"{_safe_name(first['run_label'])}_{source_hash}"
        )
        for name, keys in AVERAGED_ARTIFACTS.items():
            _average_artifact(run_dirs, name, keys, averaged_dir)

        row = first.to_dict()
        numeric = group.select_dtypes(include="number").columns
        for column in numeric:
            if column != "seed":
                row[column] = float(group[column].mean())
        row.pop("seed", None)
        row["averaged_seeds"] = ",".join(
            str(value) for value in sorted(set(group.get("seed", pd.Series(dtype=int))))
        )
        row["averaged_runs"] = int(len(group))
        row["summary_path"] = ";".join(group["summary_path"].astype(str))
        row["run_path"] = ";".join(group["run_path"].astype(str))
        row["manifest_id"] = ";".join(group["manifest_id"].astype(str))
        row["_run_dir"] = str(averaged_dir)
        rows.append(row)
    return pd.DataFrame(rows)


def _comparison_frame(
    frame: pd.DataFrame,
    metric: str,
    reference_model: str,
) -> pd.DataFrame:
    metric_column = f"metrics_{metric}" if metric in ERROR_METRICS else metric
    if metric_column not in frame:
        raise KeyError(f"report does not contain {metric_column}")
    rows: list[dict[str, Any]] = []
    group_columns = _group_columns(frame)
    for key, group in frame.groupby(group_columns, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, key_values))
        candidates = group[group["model"].map(_is_baseline)]
        if reference_model == "best_baseline":
            if candidates.empty:
                continue
            reference = candidates.loc[candidates[metric_column].astype(float).idxmin()]
        else:
            matches = group[group["model"].astype(str).str.casefold() == reference_model.casefold()]
            if matches.empty:
                continue
            reference = matches.iloc[0]
        reference_value = float(reference[metric_column])
        for _, model_row in group.iterrows():
            value = float(model_row[metric_column])
            improvement = (
                100.0 * (reference_value - value) / reference_value
                if abs(reference_value) > 1e-12
                else float("nan")
            )
            rows.append(
                {
                    **identity,
                    "setting": f"{int(model_row['lags'])}:{int(model_row['horizon'])}",
                    "range": forecast_range_name(
                        int(model_row["lags"]), int(model_row["horizon"])
                    ),
                    "model": str(model_row["model"]),
                    "metric": metric,
                    "value": value,
                    "reference_model": str(reference["model"]),
                    "reference_value": reference_value,
                    "improvement_pct": improvement,
                }
            )
    return pd.DataFrame(rows)


def _marginal_frame(comparison: pd.DataFrame, axis: str) -> pd.DataFrame:
    columns = [
        axis,
        "model",
        "metric",
        "mean_value",
        "mean_reference_value",
        "mean_improvement_pct",
        "configurations",
        "reference_models",
    ]
    if comparison.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (label, model, metric), group in comparison.groupby(
        [axis, "model", "metric"], dropna=False, sort=False
    ):
        rows.append(
            {
                axis: label,
                "model": model,
                "metric": metric,
                "mean_value": float(group["value"].mean()),
                "mean_reference_value": float(group["reference_value"].mean()),
                "mean_improvement_pct": float(group["improvement_pct"].mean()),
                "configurations": int(len(group)),
                "reference_models": ",".join(sorted(set(group["reference_model"].astype(str)))),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _write_average_latex(
    by_dataset: pd.DataFrame,
    by_range: pd.DataFrame,
    metric: str,
    reference_model: str,
    path: Path,
) -> None:
    models = list(
        dict.fromkeys(
            [
                *by_dataset.get("model", pd.Series(dtype=str)).astype(str),
                *by_range.get("model", pd.Series(dtype=str)).astype(str),
            ]
        )
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            f"\\caption{{Equal-configuration mean {_latex_escape(metric)}. "
            f"Parentheses report mean percentage improvement relative to "
            f"{_latex_escape(reference_model)}.}}"
        ),
        r"\resizebox{\textwidth}{!}{%",
        fr"\begin{{tabular}}{{l{'c' * len(models)}}}",
        r"\toprule",
        "Scope & " + " & ".join(_latex_escape(model) for model in models) + r" \\",
        r"\midrule",
    ]

    def add_panel(title: str, frame: pd.DataFrame, axis: str) -> None:
        lines.append(fr"\multicolumn{{{len(models) + 1}}}{{l}}{{\textit{{{title}}}}} \\")
        for label in dict.fromkeys(frame.get(axis, pd.Series(dtype=str)).astype(str)):
            selected = frame[frame[axis].astype(str) == label].set_index("model")
            cells = []
            for model in models:
                if model not in selected.index:
                    cells.append("--")
                    continue
                row = selected.loc[model]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                cells.append(
                    f"{float(row['mean_value']):.4f} "
                    f"({float(row['mean_improvement_pct']):+.2f}\\%)"
                )
            lines.append(_latex_escape(label) + " & " + " & ".join(cells) + r" \\")

    add_panel("Average over cadence ranges", by_dataset, "dataset")
    lines.append(r"\midrule")
    add_panel("Average over datasets", by_range, "range")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            fr"\label{{tab:tsfm-{_safe_name(metric)}-marginals}}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _chronos_win_rates(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "lags",
        "horizon",
        "setting",
        "metric",
        "best_baseline",
        "best_baseline_value",
        "chronos_value",
        "chronos_win_pct",
        "tie_pct",
        "windows",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, group in frame.groupby(_group_columns(frame), dropna=False, sort=False):
        chronos_rows = group[group["model"].map(_is_chronos)]
        baseline_rows = group[group["model"].map(_is_baseline)]
        if chronos_rows.empty or baseline_rows.empty:
            continue
        chronos = chronos_rows.iloc[0]
        chronos_windows = _read_csv(Path(chronos["_run_dir"]), "window_metrics.csv")
        if chronos_windows is None:
            continue
        for metric in ERROR_METRICS:
            metric_column = f"metrics_{metric}"
            if metric_column not in baseline_rows or metric not in chronos_windows:
                continue
            best = baseline_rows.loc[baseline_rows[metric_column].astype(float).idxmin()]
            baseline_windows = _read_csv(Path(best["_run_dir"]), "window_metrics.csv")
            if baseline_windows is None or metric not in baseline_windows:
                continue
            keys = ["user_id", "query_index"]
            paired = chronos_windows[keys + [metric]].merge(
                baseline_windows[keys + [metric]],
                on=keys,
                how="inner",
                validate="one_to_one",
                suffixes=("_chronos", "_baseline"),
            )
            if paired.empty:
                continue
            chronos_values = paired[f"{metric}_chronos"].to_numpy(dtype=float)
            baseline_values = paired[f"{metric}_baseline"].to_numpy(dtype=float)
            rows.append(
                {
                    "dataset": chronos["dataset"],
                    "lags": int(chronos["lags"]),
                    "horizon": int(chronos["horizon"]),
                    "setting": f"{int(chronos['lags'])}:{int(chronos['horizon'])}",
                    "metric": metric,
                    "best_baseline": best["model"],
                    "best_baseline_value": float(best[metric_column]),
                    "chronos_value": float(chronos[metric_column]),
                    "chronos_win_pct": float(100.0 * np.mean(chronos_values < baseline_values)),
                    "tie_pct": float(100.0 * np.mean(chronos_values == baseline_values)),
                    "windows": int(len(paired)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_report(
    root: str | Path,
    output: str | Path,
    *,
    datasets: set[str] | None = None,
    settings: set[str] | None = None,
    tasks: set[tuple[str, str]] | None = None,
    models: set[str] | None = None,
    pipeline_config: Mapping[str, Any] | None = None,
    config_policy: str = "distinct",
    repeat_policy: str = "selected",
    purposes: set[str] | None = None,
    metric: str = "nmse",
    reference_model: str = "best_baseline",
    make_plots: bool = True,
) -> Path:
    metric = str(metric).casefold()
    if metric not in TABLE_METRICS:
        raise ValueError(f"metric must be one of {TABLE_METRICS}, got {metric!r}")
    root = Path(root).expanduser().resolve()
    active_launch = os.environ.get("EXPERIMENT_LAUNCH_ID")
    output = Path(output).expanduser().resolve()
    identity_roots = sorted(
        {path.parent.parent for path in root.rglob("manifest.json") if path.parent.name.startswith("run_")}
    )
    selected_runs: list[SelectedRun] = []
    for identity_root in identity_roots:
        sample_path = next(identity_root.glob("run_*/manifest.json"), None)
        if sample_path is None:
            continue
        manifests = [load_manifest(path) for path in identity_root.glob("run_*/manifest.json")]
        if not any(
            manifest_is_selectable(manifest, allow_ready_launch_id=active_launch)
            for manifest in manifests
        ):
            continue
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        identity = sample["identity"]
        setting = f"{identity['lookback']}:{identity['horizon']}"
        if tasks and (identity["dataset"], setting) not in tasks:
            continue
        if datasets and identity["dataset"] not in datasets:
            continue
        if settings and setting not in settings:
            continue
        if models and identity["backbone"] not in models:
            continue
        selected_runs.extend(
            select_identity_runs(
                identity_root,
                requested_pipeline=pipeline_config,
                config_policy=config_policy,
                repeat_policy=repeat_policy,
                purposes=purposes,
                allow_ready_launch_id=active_launch,
            )
        )

    rows = []
    for selected in selected_runs:
        path = selected.run_dir / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        row = _flatten("", summary)
        row["summary_path"] = str(path)
        row["run_path"] = selected.run_dir.relative_to(root).as_posix()
        row["run_label"] = selected.label
        row["manifest_id"] = selected.manifest["manifest_id"]
        row["identity_signature"] = selected.manifest["signatures"]["path"]
        row["_run_dir"] = str(selected.run_dir)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no completed run summaries under {root}")

    output.mkdir(parents=True, exist_ok=True)
    raw_frame = pd.DataFrame(rows).sort_values(
        ["dataset", "lags", "horizon", "model", "covariate_mode", "instance_normalize", "remove_constant"]
    )
    average_policy = config_policy == "average" or repeat_policy == "average"
    analysis_frame = (
        _average_analysis_frame(raw_frame, output) if average_policy else raw_frame
    )
    comparison = _comparison_frame(analysis_frame, metric, reference_model)
    by_dataset = _marginal_frame(comparison, "dataset")
    by_setting = _marginal_frame(comparison, "setting")
    by_range = _marginal_frame(comparison, "range")
    win_rates = _chronos_win_rates(analysis_frame)
    plot_index = build_plots(analysis_frame, output) if make_plots else pd.DataFrame()

    frame = analysis_frame.drop(columns=["_run_dir"])

    result_path = output / "results.csv"
    frame.to_csv(result_path, index=False)
    comparison.to_csv(output / f"{metric}_configurations.csv", index=False)
    by_dataset.to_csv(output / f"{metric}_average_by_dataset.csv", index=False)
    by_setting.to_csv(output / f"{metric}_average_by_setting.csv", index=False)
    by_range.to_csv(output / f"{metric}_average_by_range.csv", index=False)
    win_rates.to_csv(output / "chronos_win_rates.csv", index=False)
    plot_index.to_csv(output / "plot_index.csv", index=False)
    _write_average_latex(
        by_dataset,
        by_range,
        metric,
        reference_model,
        output / f"{metric}_average_table.tex",
    )
    write_report_manifest(
        output / "report_manifest.json",
        inputs=selected_runs,
        config_policy=config_policy,
        repeat_policy=repeat_policy,
        filters={
            "datasets": sorted(datasets or []),
            "settings": sorted(settings or []),
            "tasks": [f"{dataset}={setting}" for dataset, setting in sorted(tasks or [])],
            "models": sorted(models or []),
            "pipeline": dict(pipeline_config or {}),
            "purposes": sorted(purposes or []),
            "metric": metric,
            "reference_model": reference_model,
            "make_plots": bool(make_plots),
        },
    )
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="evaluation output root")
    parser.add_argument("--output", required=True, help="report directory")
    parser.add_argument("--datasets")
    parser.add_argument("--settings")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--models")
    parser.add_argument("--pipeline-config", action="append", default=[])
    parser.add_argument("--config-policy", choices=["distinct", "latest", "average"], default="distinct")
    parser.add_argument("--repeat-policy", default="selected")
    parser.add_argument("--purpose", action="append", default=[])
    parser.add_argument("--metric", choices=TABLE_METRICS, default="nmse")
    parser.add_argument("--reference-model", default="best_baseline")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(
        build_report(
            args.root,
            args.output,
            datasets=_names(args.datasets),
            settings=_names(args.settings),
            tasks=_task_pairs(args.task),
            models=_names(args.models),
            pipeline_config=_pipeline_pairs(args.pipeline_config),
            config_policy=args.config_policy,
            repeat_policy=args.repeat_policy,
            purposes=set(args.purpose),
            metric=args.metric,
            reference_model=args.reference_model,
            make_plots=not args.no_plots,
        )
    )


if __name__ == "__main__":
    main()
