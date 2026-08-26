from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from results.reporting import (
    _chronos_win_rates,
    _comparison_frame,
    _marginal_frame,
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
        common = {
            "dataset": "toy",
            "lags": 4,
            "horizon": 2,
            "covariate_mode": "none",
            "instance_normalize": True,
            "remove_constant": False,
            "use_time_features": True,
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


if __name__ == "__main__":
    test_best_baseline_marginals_and_chronos_wins()
