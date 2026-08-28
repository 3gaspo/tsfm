from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_root_fronts_and_workflows() -> None:
    slurm_root = PROJECT_ROOT / "slurm"
    dgx_fronts = sorted((slurm_root / "dgx/main").glob("*.slurm"))
    selena_fronts = sorted((slurm_root / "selena/main").glob("*.slurm"))
    fronts = dgx_fronts + selena_fronts
    assert (PROJECT_ROOT / "publish_job.sh").is_file()
    assert not list(PROJECT_ROOT.glob("*.slurm"))
    assert [path.name for path in dgx_fronts] == [
        "01_univariate.slurm",
        "02_controls.slurm",
        "03_covariates.slurm",
        "04_foundation_models.slurm",
    ]
    assert [path.name for path in selena_fronts] == [
        "01_univariate_selena.slurm",
        "02_controls_selena.slurm",
        "03_covariates_selena.slurm",
        "04_foundation_models_selena.slurm",
    ]
    for front in fronts:
        text = front.read_text(encoding="utf-8")
        assert 'PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"' in text
        assert "#SBATCH --ntasks=1" in text
        assert "BASH_SOURCE" not in text
        assert "--array" not in text
    assert len(dgx_fronts) == len(selena_fronts) == 4
    for front in dgx_fronts:
        text = front.read_text(encoding="utf-8")
        assert "#SBATCH --partition=h100" in text
        assert "#SBATCH --output=logs/%x_%j.out" in text
        assert "#SBATCH --error=logs/%x_%j.err" in text
        assert "#SBATCH --wckey=" not in text
    for front in selena_fronts:
        text = front.read_text(encoding="utf-8")
        assert "#SBATCH --partition=an" in text
        assert "#SBATCH --qos=an_preemptable" in text
        assert "#SBATCH --output=/scratch/users/%u/codes/tsfm/logs_selena/%x_%j.out" in text
        assert "#SBATCH --error=/scratch/users/%u/codes/tsfm/logs_selena/%x_%j.err" in text
        assert "#SBATCH --exclusive" in text
        assert "#SBATCH --no-requeue" not in text
        assert "#SBATCH --wckey=P12CU:DATASCIENCE" in text
        assert 'source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"' in text
        assert 'EXPERIMENT_LAUNCH_ID="selena_${SLURM_JOB_ID' in text
    selena_runtime = (PROJECT_ROOT / "src/slurm/selena_runtime.sh").read_text(
        encoding="utf-8"
    )
    assert '${NNI_FILE:-$HOME/codes/.secrets/nni}' in selena_runtime
    assert (
        'SELENA_SCRATCH_PROJECT_ROOT="/scratch/users/$selena_nni/codes/$PROJECT_NAME"'
        in selena_runtime
    )
    assert (
        'OUTPUTS_ROOT="${OUTPUTS_ROOT:-$SELENA_SCRATCH_PROJECT_ROOT/outputs_selena}"'
        in selena_runtime
    )
    assert (
        'LOGS_ROOT="${LOGS_ROOT:-$SELENA_SCRATCH_PROJECT_ROOT/logs_selena}"'
        in selena_runtime
    )
    common = (PROJECT_ROOT / "src/slurm/benchmark_common.sh").read_text(encoding="utf-8")
    assert "STAGES:-evaluate,report" in common
    assert 'LOGS_ROOT="${LOGS_ROOT:-$PROJECT_ROOT/logs}"' in common
    assert 'OUTPUTS_ROOT="${OUTPUTS_ROOT:-$PROJECT_ROOT/outputs}"' in common
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUTS_ROOT/$EXPERIMENT_FAMILY}"' in common
    assert '--output "$OUTPUTS_ROOT/reports/$family/${EXPERIMENT_MODE:-test}"' in common
    assert "srun --ntasks=1" in common
    assert 'DEFER_MANIFEST_COMPLETION=1 srun --ntasks=1' in common
    assert "pipeline.runs complete-launch" in common
    assert common.count("pipeline.runs complete-launch") == 2
    assert "python -m pipeline.profiles" in common
    assert '--profile "$profile"' in common
    assert '--datasets-override "$DATASETS_OVERRIDE"' in common
    assert '--settings-override "$SETTINGS_OVERRIDE"' in common
    assert 'catalog="$(find_time_catalog || true)"' in common
    assert 'TASK_DATASETS+=("$dataset")' in common
    assert 'TASK_SETTINGS+=("$setting")' in common
    assert 'TASK_PERIODS+=("$period")' in common
    assert 'report_args+=(--task "${TASK_DATASETS[$task_index]}=${TASK_SETTINGS[$task_index]}")' in common
    assert "model_valid_for_setting" in common
    assert 'local period="${LOOKBACK_PERIOD_STEPS:-$cadence_period}"' in common
    assert '"model.lookback_period=$LOOKBACK_PERIOD_STEPS"' in common
    assert 'evaluation.mase_seasonality=${MASE_SEASONALITY:-1}' in common
    assert '--metric "${TABLE_METRIC:-nmse}"' in common
    assert '"$PROJECT_ROOT/../$kind"' in common
    assert '"$PROJECT_ROOT/../../../$kind"' in common
    assert 'mapfile -t roots < <(resource_candidates datasets)' in common
    assert 'mapfile -t roots < <(resource_candidates weights)' in common
    assert 'find_weight_path tsicl/tsicl-v1.ckpt' in common
    assert 'find_weight_path chronos-bolt-base' in common
    assert 'find_weight_path chronos-t5-base' in common
    for adapter in ("chronos2.py", "chronos_bolt.py", "chronos_t5.py", "ts_icl.py"):
        source = (PROJECT_ROOT / "src/external_models" / adapter).read_text(
            encoding="utf-8"
        )
        assert 'project.parent / "weights"' in source
    assert "ts_icl) batch_size=32" in common
    assert "chronos_t5) batch_size=32" in common
    assert "chronos_bolt) batch_size=128" in common
    assert (PROJECT_ROOT / "src/external_models/tabpfn.py").is_file()
    assert not (PROJECT_ROOT / "src/external_models/tirex2.py").exists()
    assert (PROJECT_ROOT / "archive/retired_external_models/tirex2.py").is_file()
    univariate = (PROJECT_ROOT / "src/slurm/benchmark_univariate.sh").read_text(
        encoding="utf-8"
    )
    assert "MODELS=(persistence expected repeat lookback chronos2)" in univariate
    assert 'for task_index in "${!TASK_DATASETS[@]}"' in univariate
    assert 'model_valid_for_setting "$model" "$setting" "$period"' in univariate
    assert "lookback0" not in univariate
    assert "lookback168" not in univariate
    covariates = (PROJECT_ROOT / "src/slurm/benchmark_covariates.sh").read_text(
        encoding="utf-8"
    )
    assert "MODELS=(chronos2 ts_icl)" in covariates
    assert "TIME_FEATURES" not in covariates
    foundation = (
        PROJECT_ROOT / "src/slurm/benchmark_foundation_models.sh"
    ).read_text(encoding="utf-8")
    assert "set_profile_axes foundation" in foundation
    assert 'for task_index in "${!TASK_DATASETS[@]}"' in foundation
    assert "MODELS=(chronos2 chronos_bolt chronos_t5 ts_icl)" in foundation
    assert 'run_evaluation "$dataset" "$setting" "$model" none false false "$seed"' in foundation
    assert 'TABLE_REFERENCE_MODEL="${TABLE_REFERENCE_MODEL:-chronos2}"' in foundation
    code_sync = (PROJECT_ROOT / "sync_code_to_selena.sh").read_text(
        encoding="utf-8"
    )
    result_sync = (PROJECT_ROOT / "sync_results_to_dgx.sh").read_text(
        encoding="utf-8"
    )
    for script in (code_sync, result_sync):
        assert 'PROJECT_NAME="$(basename "$PROJECT_ROOT")"' in script
        assert "sed -n '1p'" in script
        assert 'NNI_FILE="$HOME/codes/.secrets/nni"' in script
    for excluded in (
        ".git/",
        ".venv/",
        ".secrets/",
        "pyproject.toml",
        "uv.lock",
        "datasets/",
        "weights/",
        "outputs/",
        "logs/",
    ):
        assert f"--exclude='{excluded}'" in code_sync
    assert "selena.hpc.edf.fr" in code_sync
    assert "--delete" in code_sync
    assert "dgx-front.retd.edf.fr" not in result_sync
    assert (
        'SOURCE_ROOT="$nni@selena.hpc.edf.fr:/scratch/users/$nni/codes/$PROJECT_NAME"'
        in result_sync
    )
    assert (
        'SCRATCH_PROJECT_ROOT="/scratch/users/$nni/codes/$PROJECT_NAME"'
        in code_sync
    )
    assert '"mkdir -p \'$SCRATCH_PROJECT_ROOT/outputs_selena\'' in code_sync
    assert 'DESTINATION_ROOT="$PROJECT_ROOT"' in result_sync
    assert 'mkdir -p "$DESTINATION_ROOT/outputs_selena"' in result_sync
    assert "--include='outputs_selena/.gitkeep'" in code_sync
    assert "--exclude='outputs_selena/***'" in code_sync
    assert "--include='logs_selena/.gitkeep'" in code_sync
    assert "--exclude='logs_selena/***'" in code_sync
    assert '"$SOURCE_ROOT/outputs_selena/"' in result_sync
    assert '"$SOURCE_ROOT/logs_selena/"' in result_sync
    assert "pulled from Selena to DGX" in result_sync
    assert "--delete" not in result_sync


if __name__ == "__main__":
    test_root_fronts_and_workflows()
