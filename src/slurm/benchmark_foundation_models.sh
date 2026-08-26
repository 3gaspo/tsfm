#!/bin/bash

EXPERIMENT_FAMILY=foundation_models
source "$PROJECT_ROOT/src/slurm/benchmark_common.sh"
resolve_roots
set_profile_axes foundation
MODELS=(chronos2 chronos_bolt ts_icl tabpfn_ts)
# tirex2 remains adapter-supported but is excluded from foundation launches for now.
if [ -n "${MODELS_OVERRIDE:-}" ]; then read -r -a MODELS <<< "$MODELS_OVERRIDE"; fi
log "foundation models=${MODELS[*]}"

if stage_enabled evaluate; then
    for task_index in "${!TASK_DATASETS[@]}"; do
        dataset="${TASK_DATASETS[$task_index]}"
        setting="${TASK_SETTINGS[$task_index]}"
        for model in "${MODELS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_evaluation "$dataset" "$setting" "$model" none false false true "$seed"
            done
        done
    done
fi
if stage_enabled report; then
    TABLE_REFERENCE_MODEL="${TABLE_REFERENCE_MODEL:-chronos2}"
    build_report foundation_models
fi
