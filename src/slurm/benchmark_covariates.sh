#!/bin/bash

EXPERIMENT_FAMILY=covariates
source "$PROJECT_ROOT/src/slurm/benchmark_common.sh"
resolve_roots
set_profile_axes
MODELS=(chronos2 tabpfn_ts)
COVARIATE_MODES=(none identity)
if [ -n "${MODELS_OVERRIDE:-}" ]; then read -r -a MODELS <<< "$MODELS_OVERRIDE"; fi
if [ -n "${COVARIATE_MODES_OVERRIDE:-}" ]; then read -r -a COVARIATE_MODES <<< "$COVARIATE_MODES_OVERRIDE"; fi

if stage_enabled evaluate; then
    for task_index in "${!TASK_DATASETS[@]}"; do
        dataset="${TASK_DATASETS[$task_index]}"
        setting="${TASK_SETTINGS[$task_index]}"
        for model in "${MODELS[@]}"; do
            for covariate_mode in "${COVARIATE_MODES[@]}"; do
                TIME_FEATURES=(true)
                if [ "$model" = "tabpfn_ts" ] && [ "$covariate_mode" != "none" ] && [ "${EXPERIMENT_MODE:-test}" = "ultra" ]; then
                    TIME_FEATURES=(true false)
                fi
                if [ -n "${TIME_FEATURES_OVERRIDE:-}" ]; then read -r -a TIME_FEATURES <<< "$TIME_FEATURES_OVERRIDE"; fi
                if [ "$covariate_mode" = "none" ]; then TIME_FEATURES=(true); fi
                for use_time_features in "${TIME_FEATURES[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        run_evaluation "$dataset" "$setting" "$model" "$covariate_mode" true false "$use_time_features" "$seed"
                    done
                done
            done
        done
    done
fi
if stage_enabled report; then build_report covariates; fi
