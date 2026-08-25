from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_root_fronts_and_workflows() -> None:
    fronts = sorted(PROJECT_ROOT.glob("[0-9][0-9]_*.slurm"))
    assert (PROJECT_ROOT / "publish_job.sh").is_file()
    assert [path.name for path in fronts] == [
        "00_prepare_time.slurm",
        "01_univariate.slurm",
        "02_controls.slurm",
        "03_covariates.slurm",
        "04_foundation_models.slurm",
    ]
    for front in fronts:
        text = front.read_text(encoding="utf-8")
        assert 'PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"' in text
        assert "#SBATCH --ntasks=1" in text
        assert "logs/%x_%j.out" in text
        assert "BASH_SOURCE" not in text
        assert "--array" not in text
    preparation = (PROJECT_ROOT / "src/slurm/prepare_time.sh").read_text(
        encoding="utf-8"
    )
    assert "srun --ntasks=1" in preparation
    assert "scripts.prepare_time_csv" in preparation
    assert 'TIME_OUTPUT_ROOT:-$PROJECT_ROOT/datasets/time' in preparation
    assert 'TIME_FREQUENCIES:-15T H D' in preparation
    assert 'TIME_SETTINGS:-336:48 504:168' in preparation
    assert "TIME_OVERWRITE:-false" in preparation
    assert "TIME preparation completed successfully" in preparation
    common = (PROJECT_ROOT / "src/slurm/benchmark_common.sh").read_text(encoding="utf-8")
    assert "STAGES:-evaluate,report" in common
    assert "srun --ntasks=1" in common
    assert 'DEFER_MANIFEST_COMPLETION=1 srun --ntasks=1' in common
    assert "pipeline.runs complete-launch" in common
    assert common.count("pipeline.runs complete-launch") == 2
    assert "SETTINGS=(504:168)" in common
    assert "DATASETS=(electricity traffic solar exchange_rate)" in common
    assert "SETTINGS=(168:24 336:48 504:168)" in common
    assert "model_valid_for_setting" in common
    assert 'local period="${LOOKBACK_PERIOD_STEPS:-168}"' in common
    assert '"model.lookback_period=$LOOKBACK_PERIOD_STEPS"' in common
    assert 'evaluation.mase_seasonality=${MASE_SEASONALITY:-1}' in common
    assert '--metric "${TABLE_METRIC:-nmse}"' in common
    assert 'TSICL_WEIGHTS:-$WEIGHTS_ROOT/tsicl/tsicl-v1.ckpt' in common
    assert 'TIREX2_WEIGHTS:-$WEIGHTS_ROOT/tirex2' in common
    assert 'CHRONOS_BOLT_WEIGHTS:-$WEIGHTS_ROOT/chronos-bolt-base' in common
    assert "ts_icl) batch_size=32" in common
    assert "tirex2) batch_size=64" in common
    assert "chronos_bolt) batch_size=128" in common
    univariate = (PROJECT_ROOT / "src/slurm/benchmark_univariate.sh").read_text(
        encoding="utf-8"
    )
    assert "MODELS=(persistence expected repeat lookback chronos2)" in univariate
    assert "lookback0" not in univariate
    assert "lookback168" not in univariate
    foundation = (
        PROJECT_ROOT / "src/slurm/benchmark_foundation_models.sh"
    ).read_text(encoding="utf-8")
    assert "DATASETS=(electricity traffic solar weather exchange_rate)" in foundation
    assert "SETTINGS=(336:48 504:168)" in foundation
    assert "MODELS=(chronos2 chronos_bolt ts_icl tirex2 tabpfn_ts)" in foundation
    assert 'run_evaluation "$dataset" "$setting" "$model" none false false true' in foundation
    assert 'TABLE_REFERENCE_MODEL="${TABLE_REFERENCE_MODEL:-chronos2}"' in foundation


if __name__ == "__main__":
    test_root_fronts_and_workflows()
