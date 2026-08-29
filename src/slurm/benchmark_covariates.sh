#!/bin/bash

EXPERIMENT_FAMILY=covariates
source "$PROJECT_ROOT/src/slurm/benchmark_common.sh"
resolve_roots
set_profile_axes
MODELS=(chronos2 ts_icl)
COVARIATE_MODES=(none identity)
TABLE_REFERENCE_MODEL="${TABLE_REFERENCE_MODEL:-no_covariates}"
if [ -n "${MODELS_OVERRIDE:-}" ]; then read -r -a MODELS <<< "$MODELS_OVERRIDE"; fi
if [ -n "${COVARIATE_MODES_OVERRIDE:-}" ]; then read -r -a COVARIATE_MODES <<< "$COVARIATE_MODES_OVERRIDE"; fi

if stage_enabled evaluate; then
    stage_start evaluate
    for task_index in "${!TASK_DATASETS[@]}"; do
        dataset="${TASK_DATASETS[$task_index]}"
        setting="${TASK_SETTINGS[$task_index]}"
        for model in "${MODELS[@]}"; do
            for covariate_mode in "${COVARIATE_MODES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    run_evaluation "$dataset" "$setting" "$model" "$covariate_mode" true false "$seed"
                done
            done
        done
    done
    stage_complete
fi
if stage_enabled report; then
    stage_start report
    build_report covariates
    stage_complete
fi
