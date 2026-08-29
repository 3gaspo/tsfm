#!/bin/bash

EXPERIMENT_FAMILY=foundation_models
source "$PROJECT_ROOT/src/slurm/benchmark_common.sh"
resolve_roots
set_profile_axes foundation
MODELS=(chronos2 chronos_bolt chronos_t5 ts_icl)
if [ -n "${MODELS_OVERRIDE:-}" ]; then read -r -a MODELS <<< "$MODELS_OVERRIDE"; fi
log "foundation models=${MODELS[*]}"

if stage_enabled evaluate; then
    stage_start evaluate
    for task_index in "${!TASK_DATASETS[@]}"; do
        dataset="${TASK_DATASETS[$task_index]}"
        setting="${TASK_SETTINGS[$task_index]}"
        for model in "${MODELS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_evaluation "$dataset" "$setting" "$model" none false false "$seed"
            done
        done
    done
    stage_complete
fi
if stage_enabled report; then
    stage_start report
    TABLE_REFERENCE_MODEL="${TABLE_REFERENCE_MODEL:-chronos2}"
    build_report foundation_models
    stage_complete
fi
