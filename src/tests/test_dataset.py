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
        assert panel.user_names == ["a"]
        assert torch.equal(panel.values, panel.covariates)
        dataset = StridedWindowDataset(panel, 4, 2, stride=3)
        assert dataset.query_indices == [3, 6, 9]
        assert [dataset[index]["query_index"] for index in range(len(dataset))] == [3, 6, 9]


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


def test_timetensors_tensor_names_and_context_shapes() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
        root = Path(directory)
        values = torch.arange(54, dtype=torch.float32).reshape(3, 1, 18)
        global_context = torch.stack(
            [torch.arange(18, dtype=torch.float32), torch.arange(18, dtype=torch.float32) * 2]
        )
        torch.save(values, root / "values.pt")
        torch.save(np.arange(18), root / "datetimes.pt")
        torch.save(torch.tensor([10, 20, 30]), root / "individual_ids.pt")
        torch.save(torch.tensor([1.0, 2.0, 3.0]), root / "individual_context.pt")
        torch.save(global_context, root / "global_context.pt")
        (root / "dataset_metadata.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "individual_names": {"10": "alice", "20": "bob", "30": "carol"},
                }
            ),
            encoding="utf-8",
        )

        panel = load_panel(
            {
                "path": str(root),
                "drop_users": ["bob"],
                "covariate_mode": "known",
            }
        )
        assert panel.user_names == ["alice", "carol"]
        assert panel.covariates is not None
        assert panel.covariates.shape == (2, 3, 18)
        assert torch.equal(panel.covariates[:, 0, 0], torch.tensor([1.0, 3.0]))
        assert torch.equal(panel.covariates[0, 1:], global_context)
        assert set(panel.metadata["source_files"]) == {
            "values.pt",
            "datetimes.pt",
            "individual_ids.pt",
            "individual_context.pt",
            "global_context.pt",
            "dataset_metadata.json",
        }


if __name__ == "__main__":
    test_config_merge_and_identity_covariate()
    test_constant_filter_is_pairwise_and_deterministic()
    test_timetensors_tensor_names_and_context_shapes()
