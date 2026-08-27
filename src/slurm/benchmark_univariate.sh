#!/bin/bash

EXPERIMENT_FAMILY=univariate
source "$PROJECT_ROOT/src/slurm/benchmark_common.sh"
resolve_roots
set_profile_axes
MODELS=(persistence expected repeat lookback chronos2)
if [ -n "${MODELS_OVERRIDE:-}" ]; then read -r -a MODELS <<< "$MODELS_OVERRIDE"; fi

if stage_enabled evaluate; then
    for task_index in "${!TASK_DATASETS[@]}"; do
        dataset="${TASK_DATASETS[$task_index]}"
        setting="${TASK_SETTINGS[$task_index]}"
        period="${TASK_PERIODS[$task_index]}"
        for model in "${MODELS[@]}"; do
            if ! model_valid_for_setting "$model" "$setting" "$period"; then
                log "skip invalid baseline model=$model dataset=$dataset setting=$setting period=$period"
                continue
            fi
            for seed in "${SEEDS[@]}"; do
                run_evaluation "$dataset" "$setting" "$model" none true false "$seed"
            done
        done
    done
fi
if stage_enabled report; then build_report univariate; fi
