#!/bin/bash

EXPERIMENT_FAMILY=controls
source "$PROJECT_ROOT/src/slurm/benchmark_common.sh"
resolve_roots
set_profile_axes
MODELS=(repeat chronos2 tabpfn_ts)
NORMS=(true false)
CONSTANT_POLICIES=(false true)
if [ -n "${MODELS_OVERRIDE:-}" ]; then read -r -a MODELS <<< "$MODELS_OVERRIDE"; fi
if [ -n "${INSTANCE_NORMS_OVERRIDE:-}" ]; then read -r -a NORMS <<< "$INSTANCE_NORMS_OVERRIDE"; fi
if [ -n "${REMOVE_CONSTANT_OVERRIDE:-}" ]; then read -r -a CONSTANT_POLICIES <<< "$REMOVE_CONSTANT_OVERRIDE"; fi

if stage_enabled evaluate; then
    for dataset in "${DATASETS[@]}"; do
        for setting in "${SETTINGS[@]}"; do
            for model in "${MODELS[@]}"; do
                for instance_norm in "${NORMS[@]}"; do
                    for remove_constant in "${CONSTANT_POLICIES[@]}"; do
                        for seed in "${SEEDS[@]}"; do
                            run_evaluation "$dataset" "$setting" "$model" none "$instance_norm" "$remove_constant" true "$seed"
                        done
                    done
                done
            done
        done
    done
fi
if stage_enabled report; then build_report controls; fi
