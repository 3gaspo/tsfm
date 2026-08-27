"""Plot generation for TSFM evaluation reports."""

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


ERROR_METRICS = ("mse", "mae", "nmse", "nmae", "mase")


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "value"


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


def _set_error_scale(ax: Any, values: list[np.ndarray], axis: str = "y") -> None:
    finite = np.concatenate([np.asarray(value, dtype=float).reshape(-1) for value in values])
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if not len(finite):
        return
    setter = ax.set_yscale if axis == "y" else ax.set_xscale
    setter("symlog", linthresh=max(float(np.percentile(finite, 5)), 1e-12))


def _save_figure(fig: Any, base_path: Path) -> tuple[Path, Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png = base_path.with_suffix(".png")
    pdf = base_path.with_suffix(".pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def build_plots(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Render the per-user, per-window, and horizon report figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index_rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    plot_group_columns = [name for name in _group_columns(frame) if name != "seed"]
    for _, group in frame.groupby(plot_group_columns, dropna=False, sort=False):
        first = group.iloc[0]
        dataset = str(first["dataset"])
        setting = f"{int(first['lags'])}_{int(first['horizon'])}"
        config = "_".join(
            [
                str(first.get("covariate_mode", "none")),
                "instance" if bool(first.get("instance_normalize", False)) else "raw",
                "remove_constants" if bool(first.get("remove_constant", False)) else "keep_constants",
            ]
        )
        artifact_frames: dict[str, dict[str, pd.DataFrame]] = {}
        for _, row in group.iterrows():
            label = str(row["model"])
            run_dir = Path(row["_run_dir"])
            per_user = _read_csv(run_dir, "per_user_metrics.csv")
            windows = _read_csv(run_dir, "window_metrics.csv")
            horizon = _read_csv(run_dir, "horizon_metrics.csv")
            if per_user is not None and windows is not None and horizon is not None:
                artifact_frames[label] = {
                    "per_user": per_user,
                    "windows": windows,
                    "horizon": horizon,
                }
        if not artifact_frames:
            continue
        for metric in ERROR_METRICS:
            plot_dir = output / "plots" / _safe_name(dataset) / setting / _safe_name(config) / metric
            series = [
                artifacts["per_user"][[metric, f"std_{metric}"]].to_numpy(dtype=float)
                for artifacts in artifact_frames.values()
                if metric in artifacts["per_user"] and f"std_{metric}" in artifacts["per_user"]
            ]
            if series:
                fig, ax = plt.subplots(figsize=(7.2, 5.0))
                for label, artifacts in artifact_frames.items():
                    users = artifacts["per_user"]
                    ax.scatter(
                        users[metric],
                        users[f"std_{metric}"],
                        s=18,
                        alpha=0.55,
                        linewidths=0,
                        label=label,
                    )
                _set_error_scale(ax, [value[:, 0] for value in series], axis="x")
                _set_error_scale(ax, [value[:, 1] for value in series], axis="y")
                ax.set_xlabel(f"Per-user mean {metric}")
                ax.set_ylabel(f"Per-user window std {metric}")
                ax.set_title(f"{dataset} {setting.replace('_', ':')} — per-user errors")
                ax.grid(True, alpha=0.2)
                ax.legend(frameon=False, fontsize=8, ncol=2)
                png, pdf = _save_figure(fig, plot_dir / "per_user_scatter")
                plt.close(fig)
                index_rows.append(
                    {
                        "dataset": dataset,
                        "setting": setting,
                        "metric": metric,
                        "kind": "per_user_scatter",
                        "png": png.relative_to(output).as_posix(),
                        "pdf": pdf.relative_to(output).as_posix(),
                    }
                )

            window_values = {
                label: artifacts["windows"][metric].to_numpy(dtype=float)
                for label, artifacts in artifact_frames.items()
                if metric in artifacts["windows"]
            }
            if window_values:
                fig, ax = plt.subplots(figsize=(7.2, 4.4))
                finite_values = [
                    values[np.isfinite(values) & (values >= 0)] for values in window_values.values()
                ]
                all_values = np.concatenate([values for values in finite_values if len(values)])
                positive = all_values[all_values > 0]
                floor = max(float(np.min(positive)) * 0.5, 1e-15) if len(positive) else 1e-15
                transformed = {
                    label: np.log10(np.maximum(values[np.isfinite(values)], floor))
                    for label, values in window_values.items()
                }
                combined = np.concatenate(list(transformed.values()))
                bins = np.linspace(float(combined.min()), float(combined.max()), 55)
                if np.allclose(bins[0], bins[-1]):
                    bins = np.linspace(bins[0] - 0.5, bins[-1] + 0.5, 55)
                for label, values in transformed.items():
                    ax.hist(
                        values,
                        bins=bins,
                        density=True,
                        histtype="step",
                        linewidth=1.5,
                        label=label,
                    )
                ax.set_xlabel(f"log10(window {metric})")
                ax.set_ylabel("Density")
                ax.set_title(f"{dataset} {setting.replace('_', ':')} — window errors")
                ax.grid(True, alpha=0.2)
                ax.legend(frameon=False, fontsize=8, ncol=2)
                png, pdf = _save_figure(fig, plot_dir / "window_histogram")
                plt.close(fig)
                index_rows.append(
                    {
                        "dataset": dataset,
                        "setting": setting,
                        "metric": metric,
                        "kind": "window_histogram",
                        "png": png.relative_to(output).as_posix(),
                        "pdf": pdf.relative_to(output).as_posix(),
                    }
                )

            horizon_values = {
                label: artifacts["horizon"][metric].to_numpy(dtype=float)
                for label, artifacts in artifact_frames.items()
                if metric in artifacts["horizon"]
            }
            if horizon_values:
                fig, ax = plt.subplots(figsize=(9.0, 4.4))
                for label, values in horizon_values.items():
                    ax.plot(np.arange(1, len(values) + 1), values, linewidth=1.4, label=label)
                _set_error_scale(ax, list(horizon_values.values()), axis="y")
                ax.set_xlabel("Forecast horizon step")
                ax.set_ylabel(f"Mean {metric}")
                ax.set_title(f"{dataset} {setting.replace('_', ':')} — horizon errors")
                ax.grid(True, alpha=0.2)
                ax.legend(frameon=False, fontsize=8, ncol=2)
                png, pdf = _save_figure(fig, plot_dir / "horizon_errors")
                plt.close(fig)
                index_rows.append(
                    {
                        "dataset": dataset,
                        "setting": setting,
                        "metric": metric,
                        "kind": "horizon_errors",
                        "png": png.relative_to(output).as_posix(),
                        "pdf": pdf.relative_to(output).as_posix(),
                    }
                )
    return pd.DataFrame(index_rows)
