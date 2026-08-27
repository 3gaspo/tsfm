from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import PanelData, StridedWindowDataset, load_panel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_config_merge_and_identity_covariate() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
        root = Path(directory)
        dates = pd.date_range("2025-01-01", periods=12, freq="h")
        frame = pd.DataFrame(
            {
                "date": dates,
                "a": np.arange(12),
                "b": np.ones(12),
                "c": np.arange(12) * 2,
                "d": np.arange(12) * 3,
            }
        )
        frame.to_csv(root / "toy.csv", index=False)
        (root / "config.json").write_text(
            json.dumps(
                {
                    "date_col": "date",
                    "drop_users": [1],
                    "tsfm_evaluation": {"drop_users": [2]},
                }
            ),
            encoding="utf-8",
        )
        panel = load_panel(
            {
                "path": str(root),
                "name": "toy",
                "drop_users": [3],
                "covariate_mode": "identity",
            }
        )
        assert panel.user_names == ["a", "b", "c"]
        assert torch.equal(panel.values, panel.covariates)
        dataset = StridedWindowDataset(panel, 4, 2, stride=3)
        assert dataset.query_indices == [3, 6, 9]
        assert [dataset[index]["query_index"] for index in range(len(dataset))] == [
            3,
            3,
            3,
            6,
            6,
            6,
            9,
            9,
            9,
        ]


def test_constant_filter_is_pairwise_and_deterministic() -> None:
    values = torch.tensor(
        [
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            [[1, 1, 1, 1, 1, 1, 1, 1]],
        ],
        dtype=torch.float32,
    )
    panel = PanelData("toy", values, np.arange(8), ["varying", "constant"])
    kept = StridedWindowDataset(panel, 3, 2, stride=2, remove_constant=False)
    filtered = StridedWindowDataset(panel, 3, 2, stride=2, remove_constant=True)
    assert kept.query_indices == filtered.query_indices == [2, 4]
    assert len(kept) == 4
    assert len(filtered) == 2
    assert filtered.removed_constant_pairs == 2
    assert [pair[0] for pair in filtered.pairs] == [0, 0]


def test_tensor_only_directory_is_rejected() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
        root = Path(directory)
        (root / "values.pt").write_bytes(b"not a CSV")
        try:
            load_panel({"path": str(root), "covariate_mode": "none"})
        except FileNotFoundError as error:
            assert "could not identify one dataset file" in str(error)
        else:
            raise AssertionError("TSFM must not consume tensor-only datasets")


if __name__ == "__main__":
    test_config_merge_and_identity_covariate()
    test_constant_filter_is_pairwise_and_deterministic()
    test_tensor_only_directory_is_rejected()
