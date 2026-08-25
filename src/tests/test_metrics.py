from __future__ import annotations

import pandas as pd

from evaluation import summarize_window_metrics


def test_equal_user_and_worst_tail_metrics() -> None:
    frame = pd.DataFrame(
        [
            {"user_id": 0, "user_name": "a", "query_index": 1, "mse": 1.0, "nmse": 2.0},
            {"user_id": 0, "user_name": "a", "query_index": 2, "mse": 3.0, "nmse": 4.0},
            {"user_id": 1, "user_name": "b", "query_index": 1, "mse": 8.0, "nmse": 10.0},
        ]
    )
    summary, per_user = summarize_window_metrics(frame)
    assert summary["mse"] == 4.0
    assert summary["user_mean_mse"] == 5.0
    assert summary["user_std_mse"] == 3.0
    assert summary["w10_mse"] == 8.0
    assert abs(summary["sample_std_mse"] - 2.9439202888) < 1e-9
    assert per_user.set_index("user_id").loc[0, "mse"] == 2.0
    assert per_user.set_index("user_id").loc[0, "std_mse"] == 1.0


if __name__ == "__main__":
    test_equal_user_and_worst_tail_metrics()
