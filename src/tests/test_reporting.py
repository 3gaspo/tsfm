from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from results.reporting import (
    _average_analysis_frame,
    _chronos_win_rates,
    _comparison_frame,
    _marginal_frame,
    _task_file_values,
    _task_pairs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_best_baseline_marginals_and_chronos_wins() -> None:
    assert _task_pairs(["electricity=168:24", "time/a_h=336:48"]) == {
        ("electricity", "168:24"),
        ("time/a_h", "336:48"),
    }
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
        root = Path(directory)
        tasks_file = root / "tasks.txt"
        tasks_file.write_text(
            "electricity=168:24\ntime/a_h=336:48\n", encoding="utf-8"
        )
        assert _task_pairs(_task_file_values(tasks_file)) == {
            ("electricity", "168:24"),
            ("time/a_h", "336:48"),
        }
        common = {
            "dataset": "toy",
            "lags": 4,
            "horizon": 2,
            "covariate_mode": "none",
            "instance_normalize": True,
            "remove_constant": False,
            "stride": 2,
            "seed": 1,
        }
        rows = []
        for model, nmse, values in (
            ("persistence", 3.0, [4.0, 2.0, 3.0]),
            ("repeat", 2.0, [2.0, 2.0, 2.0]),
            ("chronos2", 1.5, [1.0, 3.0, 1.0]),
        ):
            run_dir = root / model
            run_dir.mkdir()
            pd.DataFrame(
                {
                    "user_id": [0, 0, 1],
                    "query_index": [4, 6, 4],
                    "nmse": values,
                }
            ).to_csv(run_dir / "window_metrics.csv", index=False)
            rows.append(
                {
                    **common,
                    "model": model,
                    "metrics_nmse": nmse,
                    "_run_dir": str(run_dir),
                }
            )
        frame = pd.DataFrame(rows)
        comparison = _comparison_frame(frame, "nmse", "best_baseline")
        chronos = comparison[comparison["model"] == "chronos2"].iloc[0]
        assert chronos["reference_model"] == "repeat"
        assert chronos["improvement_pct"] == 25.0
        assert chronos["range"] == "custom"
        by_dataset = _marginal_frame(comparison, "dataset")
        assert by_dataset[by_dataset["model"] == "chronos2"].iloc[0][
            "mean_improvement_pct"
        ] == 25.0
        wins = _chronos_win_rates(frame)
        selected = wins[wins["metric"] == "nmse"].iloc[0]
        assert selected["best_baseline"] == "repeat"
        assert abs(selected["chronos_win_pct"] - 200.0 / 3.0) < 1e-9


def test_average_policy_precedes_comparisons_and_artifact_analysis() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
        root = Path(directory)
        rows = []
        for model, identity, summaries, windows in (
            ("repeat", "repeat-id", [2.0, 4.0], [[2.0, 4.0], [4.0, 2.0]]),
            ("chronos2", "chronos-id", [1.0, 3.0], [[1.0, 3.0], [3.0, 1.0]]),
        ):
            for repeat, (summary, values) in enumerate(zip(summaries, windows), start=1):
                run_dir = root / f"{model}_{repeat}"
                run_dir.mkdir()
                pd.DataFrame(
                    {
                        "user_id": [0, 0],
                        "user_name": ["a", "a"],
                        "query_index": [4, 6],
                        "query_datetime": ["2024-01-01", "2024-01-02"],
                        "nmse": values,
                    }
                ).to_csv(run_dir / "window_metrics.csv", index=False)
                pd.DataFrame(
                    {
                        "user_id": [0],
                        "user_name": ["a"],
                        "windows": [2],
                        "nmse": [sum(values) / 2],
                        "std_nmse": [1.0],
                    }
                ).to_csv(run_dir / "per_user_metrics.csv", index=False)
                pd.DataFrame(
                    {"horizon": [1, 2], "nmse": values, "std_nmse": [0.5, 0.5]}
                ).to_csv(run_dir / "horizon_metrics.csv", index=False)
                rows.append(
                    {
                        "dataset": "toy",
                        "lags": 4,
                        "horizon": 2,
                        "covariate_mode": "none",
                        "instance_normalize": True,
                        "remove_constant": False,
                        "stride": 2,
                        "seed": repeat,
                        "model": model,
                        "metrics_nmse": summary,
                        "identity_signature": identity,
                        "run_label": model,
                        "summary_path": str(run_dir / "summary.json"),
                        "run_path": run_dir.name,
                        "manifest_id": f"{model}-{repeat}",
                        "_run_dir": str(run_dir),
                    }
                )

        averaged = _average_analysis_frame(pd.DataFrame(rows), root / "report")
        assert len(averaged) == 2
        assert "seed" not in averaged
        chronos = averaged[averaged["model"] == "chronos2"].iloc[0]
        assert chronos["metrics_nmse"] == 2.0
        assert chronos["averaged_runs"] == 2
        averaged_windows = pd.read_csv(Path(chronos["_run_dir"]) / "window_metrics.csv")
        assert averaged_windows["nmse"].tolist() == [2.0, 2.0]

        comparison = _comparison_frame(averaged, "nmse", "best_baseline")
        selected = comparison[comparison["model"] == "chronos2"].iloc[0]
        assert abs(selected["improvement_pct"] - 100.0 / 3.0) < 1e-9
        wins = _chronos_win_rates(averaged)
        assert wins[wins["metric"] == "nmse"].iloc[0]["chronos_win_pct"] == 100.0


if __name__ == "__main__":
    test_best_baseline_marginals_and_chronos_wins()
    test_average_policy_precedes_comparisons_and_artifact_analysis()
