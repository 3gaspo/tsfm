#!/bin/bash

set -euo pipefail

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "$(timestamp) | $*"
}

stage_enabled() {
    case ",${STAGES:-evaluate,report}," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

nonempty_directory() {
    [ -d "$1" ] && find "$1" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -print -quit | grep -q .
}

resolve_roots() {
    local shared_root
    shared_root="$(cd "$PROJECT_ROOT/../../.." && pwd)"
    if [ -n "${DATA_ROOT:-}" ]; then
        DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
    elif nonempty_directory "$PROJECT_ROOT/datasets"; then
        DATA_ROOT="$PROJECT_ROOT/datasets"
    else
        DATA_ROOT="$shared_root/datasets"
    fi
    if [ -n "${WEIGHTS_ROOT:-}" ]; then
        WEIGHTS_ROOT="$(cd "$WEIGHTS_ROOT" && pwd)"
    elif nonempty_directory "$PROJECT_ROOT/weights"; then
        WEIGHTS_ROOT="$PROJECT_ROOT/weights"
    else
        WEIGHTS_ROOT="$shared_root/weights"
    fi
    : "${EXPERIMENT_FAMILY:?set EXPERIMENT_FAMILY before sourcing benchmark_common.sh}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/$EXPERIMENT_FAMILY}"
    export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    EXPERIMENT_LAUNCH_ID="${EXPERIMENT_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
    export EXPERIMENT_LAUNCH_ID
    trap tsfm_on_exit EXIT
    log "resources data_root=$DATA_ROOT weights_root=$WEIGHTS_ROOT output_root=$OUTPUT_ROOT"
}

tsfm_on_exit() {
    local status=$?
    trap - EXIT
    if [ "$status" -ne 0 ]; then
        uv run python -m pipeline.runs interrupt-launch --root "$OUTPUT_ROOT" --launch-id "$EXPERIMENT_LAUNCH_ID" || true
    elif uv run python -m pipeline.runs complete-launch --root "$OUTPUT_ROOT" --launch-id "$EXPERIMENT_LAUNCH_ID" >/dev/null; then
        :
    else
        status=$?
    fi
    exit "$status"
}

set_profile_axes() {
    case "${EXPERIMENT_MODE:-test}" in
        test)
            DATASETS=(electricity)
            SETTINGS=(504:168)
            SEEDS=(1)
            ;;
        full)
            DATASETS=(electricity traffic solar exchange_rate)
            SETTINGS=(168:24 336:48 504:168)
            SEEDS=(1)
            ;;
        ultra)
            DATASETS=(electricity traffic solar exchange_rate weather ETTh1)
            SETTINGS=(24:24 168:24 336:48 504:168 672:168 1344:336)
            SEEDS=(1)
            ;;
        *)
            log "unknown EXPERIMENT_MODE=${EXPERIMENT_MODE:-}"
            return 2
            ;;
    esac
    if [ -n "${DATASETS_OVERRIDE:-}" ]; then read -r -a DATASETS <<< "$DATASETS_OVERRIDE"; fi
    if [ -n "${SETTINGS_OVERRIDE:-}" ]; then read -r -a SETTINGS <<< "$SETTINGS_OVERRIDE"; fi
    if [ -n "${SEEDS_OVERRIDE:-}" ]; then read -r -a SEEDS <<< "$SEEDS_OVERRIDE"; fi
    log "profile mode=${EXPERIMENT_MODE:-test} datasets=${DATASETS[*]} settings=${SETTINGS[*]} seeds=${SEEDS[*]}"
}

model_weight_argument() {
    local model="$1"
    if [ "$model" = "chronos2" ]; then
        if [ -n "${CHRONOS_WEIGHTS:-}" ]; then
            echo "model.weights_path=$CHRONOS_WEIGHTS"
        elif [ -d "$WEIGHTS_ROOT/chronos2" ]; then
            echo "model.weights_path=$WEIGHTS_ROOT/chronos2"
        fi
    elif [ "$model" = "chronos_bolt" ]; then
        local checkpoint="${CHRONOS_BOLT_WEIGHTS:-$WEIGHTS_ROOT/chronos-bolt-base}"
        if [ -d "$checkpoint" ]; then echo "model.weights_path=$checkpoint"; fi
    elif [ "$model" = "tabpfn_ts" ]; then
        local checkpoint="${TABPFN_WEIGHTS:-$WEIGHTS_ROOT/tabpfnts/tabpfn-v2.5-regressor-v2.5_default.ckpt}"
        if [ -f "$checkpoint" ]; then echo "model.weights_path=$checkpoint"; fi
    elif [ "$model" = "ts_icl" ]; then
        local checkpoint="${TSICL_WEIGHTS:-$WEIGHTS_ROOT/tsicl/tsicl-v1.ckpt}"
        if [ -f "$checkpoint" ]; then echo "model.weights_path=$checkpoint"; fi
    elif [ "$model" = "tirex2" ]; then
        local checkpoint="${TIREX2_WEIGHTS:-$WEIGHTS_ROOT/tirex2}"
        if [ -d "$checkpoint" ]; then echo "model.weights_path=$checkpoint"; fi
    fi
}

model_valid_for_setting() {
    local model="$1"
    local setting="$2"
    local lags="${setting%%:*}"
    local horizon="${setting##*:}"
    if [ "$model" = lookback ]; then
        local period="${LOOKBACK_PERIOD_STEPS:-168}"
        local periods_back=$(((horizon + period - 1) / period))
        [ $((periods_back * period)) -le "$lags" ]
        return
    fi
    [ "$horizon" -le "$lags" ]
}

run_evaluation() {
    local dataset="$1"
    local setting="$2"
    local model="$3"
    local covariate_mode="$4"
    local instance_normalize="$5"
    local remove_constant="$6"
    local use_time_features="$7"
    local seed="$8"
    local lags="${setting%%:*}"
    local horizon="${setting##*:}"
    local batch_size
    local purpose
    case "$model" in
        persistence|expected|repeat|lookback) batch_size=512 ;;
        chronos2) batch_size=64 ;;
        chronos_bolt) batch_size=128 ;;
        ts_icl) batch_size=32 ;;
        tirex2) batch_size=64 ;;
        tabpfn_ts) batch_size=1 ;;
        *) log "unknown model=$model"; return 2 ;;
    esac
    batch_size="${BATCH_SIZE_OVERRIDE:-$batch_size}"
    if [ "${EXPERIMENT_MODE:-test}" = test ]; then purpose=smoke; else purpose=publication; fi
    local command=(
        uv run python -m scripts.evaluate
        "data.path=$DATA_ROOT/$dataset"
        "data.name=$dataset"
        "data.covariate_mode=$covariate_mode"
        "task.lags=$lags"
        "task.horizon=$horizon"
        "model.name=$model"
        "model.device=${MODEL_DEVICE:-cuda}"
        "model.use_time_features=$use_time_features"
        "preprocessing.instance_normalize=$instance_normalize"
        "evaluation.stride=${EVAL_STRIDE:-512}"
        "evaluation.batch_size=$batch_size"
        "evaluation.remove_constant=$remove_constant"
        "evaluation.start_fraction=${TARGET_START_FRACTION:-0.0}"
        "evaluation.end_fraction=${TARGET_END_FRACTION:-1.0}"
        "evaluation.seed=$seed"
        "evaluation.mase_seasonality=${MASE_SEASONALITY:-1}"
        "output.dir=$OUTPUT_ROOT"
        "output.workflow=$EXPERIMENT_FAMILY"
        "output.mode=${EXPERIMENT_MODE:-test}"
        "output.purpose=$purpose"
        "output.conflict_policy=${RUN_CONFLICT_POLICY:-overwrite_exact}"
        "output.force=${FORCE_RUN:-false}"
        "output.skip_completed=${SKIP_COMPLETED:-true}"
        "output.launch_id=$EXPERIMENT_LAUNCH_ID"
    )
    local weight_argument
    weight_argument="$(model_weight_argument "$model")"
    if [ -n "$weight_argument" ]; then command+=("$weight_argument"); fi
    if [ "$model" = lookback ] && [ -n "${LOOKBACK_PERIOD_STEPS:-}" ]; then
        command+=("model.lookback_period=$LOOKBACK_PERIOD_STEPS")
    fi
    if [ -n "${COVARIATE_PATHS_OVERRIDE:-}" ]; then
        command+=("data.covariate_paths=$COVARIATE_PATHS_OVERRIDE")
    fi
    if [ -n "${COVARIATE_COLS_OVERRIDE:-}" ]; then
        command+=("data.covariate_cols=$COVARIATE_COLS_OVERRIDE")
    fi
    if [ "$covariate_mode" = "known" ] && [ -z "${COVARIATE_PATHS_OVERRIDE:-}" ] && [ -z "${COVARIATE_COLS_OVERRIDE:-}" ]; then
        log "known covariates require COVARIATE_PATHS_OVERRIDE or COVARIATE_COLS_OVERRIDE"
        return 2
    fi
    log "configuration dataset=$dataset L=$lags H=$horizon model=$model covariates=$covariate_mode instance_norm=$instance_normalize remove_constant=$remove_constant time_features=$use_time_features seed=$seed batch_size=$batch_size"
    DEFER_MANIFEST_COMPLETION=1 srun --ntasks=1 "${command[@]}"
    uv run python -m pipeline.runs complete-launch --root "$OUTPUT_ROOT" --launch-id "$EXPERIMENT_LAUNCH_ID" >/dev/null
}

build_report() {
    local family="$1"
    local -a report_args
    log "report family=$family"
    report_args=(
        "$OUTPUT_ROOT"
        --output "$PROJECT_ROOT/outputs/reports/$family/${EXPERIMENT_MODE:-test}"
        --datasets "$(IFS=,; echo "${DATASETS[*]}")"
        --settings "$(IFS=,; echo "${SETTINGS[*]}")"
        --models "$(IFS=,; echo "${MODELS[*]}")"
        --config-policy "${TABLE_CONFIG_POLICY:-distinct}"
        --repeat-policy "${TABLE_REPEAT_POLICY:-selected}"
        --metric "${TABLE_METRIC:-nmse}"
        --reference-model "${TABLE_REFERENCE_MODEL:-best_baseline}"
    )
    if [ "${TABLE_PLOTS:-true}" = false ]; then report_args+=(--no-plots); fi
    if [ -n "${TABLE_PIPELINE_CONFIGS:-}" ]; then
        local item
        for item in ${TABLE_PIPELINE_CONFIGS}; do report_args+=(--pipeline-config "$item"); done
    fi
    if [ -n "${TABLE_PURPOSES:-}" ]; then
        local purpose
        for purpose in ${TABLE_PURPOSES}; do report_args+=(--purpose "$purpose"); done
    elif [ "${EXPERIMENT_MODE:-test}" = test ]; then
        report_args+=(--purpose smoke)
    else
        report_args+=(--purpose publication)
    fi
    srun --ntasks=1 uv run python -m scripts.report "${report_args[@]}"
}
