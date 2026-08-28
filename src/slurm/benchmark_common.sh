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

resource_candidates() {
    local kind="$1"
    printf '%s\n' \
        "$PROJECT_ROOT/$kind" \
        "$PROJECT_ROOT/../$kind" \
        "$PROJECT_ROOT/../../../$kind"
}

resolve_roots() {
    DATA_ROOT_EXPLICIT="${DATA_ROOT:-}"
    WEIGHTS_ROOT_EXPLICIT="${WEIGHTS_ROOT:-}"
    if [ -n "${DATA_ROOT:-}" ]; then
        DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
    else
        DATA_ROOT=""
    fi
    if [ -n "${WEIGHTS_ROOT:-}" ]; then
        WEIGHTS_ROOT="$(cd "$WEIGHTS_ROOT" && pwd)"
    else
        WEIGHTS_ROOT=""
    fi
    : "${EXPERIMENT_FAMILY:?set EXPERIMENT_FAMILY before sourcing benchmark_common.sh}"
    LOGS_ROOT="${LOGS_ROOT:-$PROJECT_ROOT/logs}"
    OUTPUTS_ROOT="${OUTPUTS_ROOT:-$PROJECT_ROOT/outputs}"
    mkdir -p "$LOGS_ROOT" "$OUTPUTS_ROOT"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUTS_ROOT/$EXPERIMENT_FAMILY}"
    export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    EXPERIMENT_LAUNCH_ID="${EXPERIMENT_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
    export EXPERIMENT_LAUNCH_ID
    trap tsfm_on_exit EXIT
    log "resources data_root=${DATA_ROOT:-auto} weights_root=${WEIGHTS_ROOT:-auto} output_root=$OUTPUT_ROOT"
}

find_dataset_path() {
    local dataset="$1"
    local roots=()
    local root candidate match
    if [ -n "$DATA_ROOT_EXPLICIT" ]; then
        roots=("$DATA_ROOT")
    else
        mapfile -t roots < <(resource_candidates datasets)
    fi
    for root in "${roots[@]}"; do
        candidate="$root/$dataset"
        if [ -f "$candidate" ]; then
            printf '%s/%s\n' "$(cd "$(dirname "$candidate")" && pwd)" "$(basename "$candidate")"
            return 0
        fi
        if [ -d "$candidate" ]; then
            match="$(find "$candidate" -maxdepth 1 -type f -iname '*.csv' -print -quit)"
            if [ -n "$match" ]; then
                (cd "$candidate" && pwd)
                return 0
            fi
        fi
    done
    log "missing dataset=$dataset searched=${roots[*]}" >&2
    return 1
}

find_weight_path() {
    local relative="$1"
    local roots=()
    local root candidate
    if [ -n "$WEIGHTS_ROOT_EXPLICIT" ]; then
        roots=("$WEIGHTS_ROOT")
    else
        mapfile -t roots < <(resource_candidates weights)
    fi
    for root in "${roots[@]}"; do
        candidate="$root/$relative"
        if [ -e "$candidate" ]; then
            if [ -d "$candidate" ]; then
                (cd "$candidate" && pwd)
            else
                printf '%s/%s\n' "$(cd "$(dirname "$candidate")" && pwd)" "$(basename "$candidate")"
            fi
            return 0
        fi
    done
    log "missing weight=$relative searched=${roots[*]}" >&2
    return 1
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

find_time_catalog() {
    local roots=()
    local root
    if [ -n "$DATA_ROOT_EXPLICIT" ]; then
        roots=("$DATA_ROOT")
    else
        mapfile -t roots < <(resource_candidates datasets)
    fi
    for root in "${roots[@]}"; do
        if [ -f "$root/time/catalog.json" ]; then
            printf '%s/%s\n' "$(cd "$root/time" && pwd)" "catalog.json"
            return 0
        fi
    done
    return 1
}

append_unique() {
    local value="$1"
    shift
    local item
    for item in "$@"; do
        if [ "$item" = "$value" ]; then return; fi
    done
    PROFILE_UNIQUE+=("$value")
}

set_profile_axes() {
    local profile="${1:-standard}"
    local catalog=""
    local task_output
    local dataset setting period
    local -a command=(
        uv run python -m pipeline.profiles
        --profile "$profile"
        --mode "${EXPERIMENT_MODE:-test}"
    )
    catalog="$(find_time_catalog || true)"
    if [ -n "$catalog" ]; then command+=(--catalog "$catalog"); fi
    if [ -n "${DATASETS_OVERRIDE:-}" ]; then
        command+=(--datasets-override "$DATASETS_OVERRIDE")
    fi
    if [ -n "${SETTINGS_OVERRIDE:-}" ]; then
        command+=(--settings-override "$SETTINGS_OVERRIDE")
    fi
    task_output="$("${command[@]}")"
    TASK_DATASETS=()
    TASK_SETTINGS=()
    TASK_PERIODS=()
    while IFS=$'\t' read -r dataset setting period; do
        [ -n "$dataset" ] || continue
        TASK_DATASETS+=("$dataset")
        TASK_SETTINGS+=("$setting")
        TASK_PERIODS+=("$period")
    done <<< "$task_output"
    [ "${#TASK_DATASETS[@]}" -gt 0 ] || {
        log "profile produced no tasks"
        return 2
    }
    DATASETS=()
    SETTINGS=()
    PROFILE_UNIQUE=()
    for dataset in "${TASK_DATASETS[@]}"; do
        append_unique "$dataset" "${PROFILE_UNIQUE[@]}"
    done
    DATASETS=("${PROFILE_UNIQUE[@]}")
    PROFILE_UNIQUE=()
    for setting in "${TASK_SETTINGS[@]}"; do
        append_unique "$setting" "${PROFILE_UNIQUE[@]}"
    done
    SETTINGS=("${PROFILE_UNIQUE[@]}")
    unset PROFILE_UNIQUE
    SEEDS=(1)
    if [ -n "${SEEDS_OVERRIDE:-}" ]; then read -r -a SEEDS <<< "$SEEDS_OVERRIDE"; fi
    log "profile mode=${EXPERIMENT_MODE:-test} datasets=${DATASETS[*]} settings=${SETTINGS[*]} tasks=${#TASK_DATASETS[@]} seeds=${SEEDS[*]}"
}

model_weight_argument() {
    local model="$1"
    local checkpoint
    if [ "$model" = "chronos2" ]; then
        if [ -n "${CHRONOS_WEIGHTS:-}" ]; then
            echo "model.weights_path=$CHRONOS_WEIGHTS"
        else
            checkpoint="$(find_weight_path chronos2)" || return
            echo "model.weights_path=$checkpoint"
        fi
    elif [ "$model" = "chronos_bolt" ]; then
        checkpoint="${CHRONOS_BOLT_WEIGHTS:-}"
        if [ -z "$checkpoint" ]; then checkpoint="$(find_weight_path chronos-bolt-base)" || return; fi
        echo "model.weights_path=$checkpoint"
    elif [ "$model" = "chronos_t5" ]; then
        checkpoint="${CHRONOS_T5_WEIGHTS:-}"
        if [ -z "$checkpoint" ]; then checkpoint="$(find_weight_path chronos-t5-base)" || return; fi
        echo "model.weights_path=$checkpoint"
    elif [ "$model" = "ts_icl" ]; then
        checkpoint="${TSICL_WEIGHTS:-}"
        if [ -z "$checkpoint" ]; then checkpoint="$(find_weight_path tsicl/tsicl-v1.ckpt)" || return; fi
        echo "model.weights_path=$checkpoint"
    fi
}

model_valid_for_setting() {
    local model="$1"
    local setting="$2"
    local cadence_period="$3"
    local lags="${setting%%:*}"
    local horizon="${setting##*:}"
    if [ "$model" = lookback ]; then
        local period="${LOOKBACK_PERIOD_STEPS:-$cadence_period}"
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
    local seed="$7"
    local lags="${setting%%:*}"
    local horizon="${setting##*:}"
    local batch_size
    local purpose
    local dataset_path
    case "$model" in
        persistence|expected|repeat|lookback) batch_size=512 ;;
        chronos2) batch_size=64 ;;
        chronos_bolt) batch_size=128 ;;
        chronos_t5) batch_size=32 ;;
        ts_icl) batch_size=32 ;;
        *) log "unknown model=$model"; return 2 ;;
    esac
    batch_size="${BATCH_SIZE_OVERRIDE:-$batch_size}"
    dataset_path="$(find_dataset_path "$dataset")"
    if [ "${EXPERIMENT_MODE:-test}" = test ]; then purpose=smoke; else purpose=publication; fi
    local command=(
        uv run python -m scripts.evaluate
        "data.path=$dataset_path"
        "data.name=$dataset"
        "data.covariate_mode=$covariate_mode"
        "task.lags=$lags"
        "task.horizon=$horizon"
        "model.name=$model"
        "model.device=${MODEL_DEVICE:-cuda}"
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
    if [ "${DROP_USERS_OVERRIDE+x}" = x ]; then
        command+=("data.drop_users=$DROP_USERS_OVERRIDE")
    fi
    if [ "$covariate_mode" = "known" ] && [ -z "${COVARIATE_PATHS_OVERRIDE:-}" ] && [ -z "${COVARIATE_COLS_OVERRIDE:-}" ]; then
        log "known covariates require COVARIATE_PATHS_OVERRIDE or COVARIATE_COLS_OVERRIDE"
        return 2
    fi
    log "configuration dataset=$dataset L=$lags H=$horizon model=$model covariates=$covariate_mode instance_norm=$instance_normalize remove_constant=$remove_constant seed=$seed batch_size=$batch_size"
    DEFER_MANIFEST_COMPLETION=1 srun --ntasks=1 "${command[@]}"
    uv run python -m pipeline.runs complete-launch --root "$OUTPUT_ROOT" --launch-id "$EXPERIMENT_LAUNCH_ID" >/dev/null
}

build_report() {
    local family="$1"
    local report_status
    local task_file
    local -a report_args
    log "report family=$family"
    task_file="$(mktemp "$OUTPUTS_ROOT/.report_tasks.XXXXXX")"
    local task_index
    for task_index in "${!TASK_DATASETS[@]}"; do
        printf '%s=%s\n' "${TASK_DATASETS[$task_index]}" "${TASK_SETTINGS[$task_index]}" >> "$task_file"
    done
    report_args=(
        "$OUTPUT_ROOT"
        --output "$OUTPUTS_ROOT/reports/$family/${EXPERIMENT_MODE:-test}"
        --diagnostics-output "$OUTPUTS_ROOT/diagnostics/$family/${EXPERIMENT_MODE:-test}"
        --tasks-file "$task_file"
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
    if srun --ntasks=1 uv run python -m scripts.report "${report_args[@]}"; then
        report_status=0
    else
        report_status=$?
    fi
    rm -f -- "$task_file"
    return "$report_status"
}
