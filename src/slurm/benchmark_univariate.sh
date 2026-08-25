#!/bin/bash

EXPERIMENT_FAMILY=univariate
source "$PROJECT_ROOT/src/slurm/benchmark_common.sh"
resolve_roots
set_profile_axes
MODELS=(persistence expected repeat lookback chronos2)
if [ -n "${MODELS_OVERRIDE:-}" ]; then read -r -a MODELS <<< "$MODELS_OVERRIDE"; fi

if stage_enabled evaluate; then
    for dataset in "${DATASETS[@]}"; do
        for setting in "${SETTINGS[@]}"; do
            for model in "${MODELS[@]}"; do
                if ! model_valid_for_setting "$model" "$setting"; then
                    log "skip invalid baseline model=$model setting=$setting"
                    continue
                fi
                for seed in "${SEEDS[@]}"; do
                    run_evaluation "$dataset" "$setting" "$model" none true false true "$seed"
                done
            done
        done
    done
fi
if stage_enabled report; then build_report univariate; fi
