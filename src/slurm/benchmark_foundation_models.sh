#!/bin/bash

EXPERIMENT_FAMILY=foundation_models
source "$PROJECT_ROOT/src/slurm/benchmark_common.sh"
resolve_roots

case "${EXPERIMENT_MODE:-test}" in
    test)
        DATASETS=(electricity)
        SETTINGS=(504:168)
        SEEDS=(1)
        ;;
    full)
        DATASETS=(electricity traffic solar weather exchange_rate)
        SETTINGS=(336:48 504:168)
        SEEDS=(1)
        ;;
    ultra)
        DATASETS=(electricity traffic solar weather exchange_rate ETTh1)
        SETTINGS=(168:24 336:48 504:168)
        SEEDS=(1)
        ;;
    *)
        log "unknown EXPERIMENT_MODE=${EXPERIMENT_MODE:-}"
        exit 2
        ;;
esac
if [ -n "${DATASETS_OVERRIDE:-}" ]; then read -r -a DATASETS <<< "$DATASETS_OVERRIDE"; fi
if [ -n "${SETTINGS_OVERRIDE:-}" ]; then read -r -a SETTINGS <<< "$SETTINGS_OVERRIDE"; fi
if [ -n "${SEEDS_OVERRIDE:-}" ]; then read -r -a SEEDS <<< "$SEEDS_OVERRIDE"; fi
MODELS=(chronos2 chronos_bolt ts_icl tirex2 tabpfn_ts)
if [ -n "${MODELS_OVERRIDE:-}" ]; then read -r -a MODELS <<< "$MODELS_OVERRIDE"; fi
log "profile mode=${EXPERIMENT_MODE:-test} datasets=${DATASETS[*]} settings=${SETTINGS[*]} models=${MODELS[*]} seeds=${SEEDS[*]}"

if stage_enabled evaluate; then
    for dataset in "${DATASETS[@]}"; do
        for setting in "${SETTINGS[@]}"; do
            for model in "${MODELS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    run_evaluation "$dataset" "$setting" "$model" none false false true "$seed"
                done
            done
        done
    done
fi
if stage_enabled report; then
    TABLE_REFERENCE_MODEL="${TABLE_REFERENCE_MODEL:-chronos2}"
    build_report foundation_models
fi
