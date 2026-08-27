from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import evaluate
from results.reporting import build_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_repeat_end_to_end() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
        root = Path(directory)
        data_root = root / "data"
        data_root.mkdir()
        values = np.arange(14, dtype=np.float32)
        pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=14, freq="h"),
                "a": values,
                "b": 2 * values,
            }
        ).to_csv(data_root / "toy.csv", index=False)
        (data_root / "config.json").write_text(
            json.dumps({"date_col": "date", "drop_users": []}),
            encoding="utf-8",
        )
        config = {
            "data": {
                "path": str(data_root),
                "name": "toy",
                "config_path": None,
                "date_col": None,
                "target_cols": None,
                "covariate_cols": None,
                "covariate_paths": [],
                "covariate_mode": "none",
                "drop_users": [],
                "aggr": None,
                "aggr_period": None,
            },
            "task": {"lags": 4, "horizon": 2},
            "model": {"name": "repeat", "device": "cpu"},
            "preprocessing": {"instance_normalize": True, "eps": 1e-8},
            "evaluation": {
                "stride": 2,
                "batch_size": 4,
                "remove_constant": False,
                "start_fraction": 0.0,
                "end_fraction": 1.0,
                "seed": 1,
                "save_window_metrics": True,
                "mase_seasonality": 1,
            },
            "output": {
                "dir": str(root / "runs"),
                "workflow": "univariate",
                "model_config_order": ["covariate_mode", "normalization", "constant_policy"],
                "row_config": ["covariate_mode"],
                "column_config": ["normalization", "constant_policy"],
                "purpose": "smoke",
                "mode": "test",
                "conflict_policy": "overwrite_exact",
                "force": False,
                "run_index": None,
                "launch_id": "test-repeat",
                "skip_completed": True,
            },
            "misc": {"log_level": "INFO"},
        }
        summary = evaluate(config)
        assert abs(summary["metrics"]["mse"] - 10.0) < 1e-6
        assert abs(summary["metrics"]["nmse"] - 3.2) < 1e-5
        assert abs(summary["metrics"]["mae"] - 3.0) < 1e-6
        assert abs(summary["metrics"]["nmae"] - 3.2 ** 0.5) < 1e-5
        assert abs(summary["metrics"]["mase"] - 2.0) < 1e-6
        assert summary["inference"]["seconds_per_user"] > 0.0
        assert summary["inference"]["seconds_per_series_window"] > 0.0
        for metric in ("mse", "mae", "nmse", "nmae", "mase"):
            assert f"sample_std_{metric}" in summary["metrics"]
            assert f"user_mean_{metric}" in summary["metrics"]
            assert f"user_std_{metric}" in summary["metrics"]
            assert f"w10_{metric}" in summary["metrics"]
        run_dir = Path(summary["run_dir"])
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        assert manifest["status"] == "completed"
        assert manifest["launch"]["mode"] == "test"
        assert (run_dir / "window_metrics.csv").exists()
        assert (run_dir / "per_user_metrics.csv").exists()
        assert (run_dir / "horizon_metrics.csv").exists()
        assert evaluate(config)["run_dir"] == summary["run_dir"]
        report = build_report(root / "runs", root / "report")
        assert report.exists()
        assert (root / "report" / "report_manifest.json").exists()
        assert (root / "report" / "nmse_average_by_dataset.csv").exists()
        assert (root / "report" / "nmse_average_by_setting.csv").exists()
        assert (root / "report" / "nmse_average_by_range.csv").exists()
        assert (root / "report" / "nmse_average_table.tex").exists()
        assert (root / "report" / "plot_index.csv").exists()
        assert build_report(root / "runs", root / "report") == report
        assert build_report(
            root / "runs",
            root / "task_report",
            tasks={("toy", "4:2")},
        ).exists()
        try:
            build_report(
                root / "runs",
                root / "stale_task_report",
                tasks={("toy", "168:24")},
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("exact task filtering admitted a stale setting")
        timing_report = build_report(
            root / "runs",
            root / "timing_report",
            metric="inference_seconds_per_series_window",
            make_plots=False,
        )
        assert timing_report.exists()
        assert (
            root
            / "timing_report"
            / "inference_seconds_per_series_window_average_by_dataset.csv"
        ).exists()


if __name__ == "__main__":
    test_repeat_end_to_end()
