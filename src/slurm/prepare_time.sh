#!/bin/bash

set -euo pipefail

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "$(timestamp) | $*"
}

OUTPUT_ROOT="${TIME_OUTPUT_ROOT:-$PROJECT_ROOT/datasets/time}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

command=(
    uv run python -m scripts.prepare_time_csv
    --output-root "$OUTPUT_ROOT"
    --repo-id "${TIME_REPO_ID:-Real-TSF/TIME-ProcessedCSV}"
    --revision "${TIME_REVISION:-main}"
    --frequencies ${TIME_FREQUENCIES:-15T H D}
    --settings ${TIME_SETTINGS:-336:48 504:168}
    --stride "${TIME_STRIDE:-512}"
    --max-series "${TIME_MAX_SERIES:-500}"
    --max-dates-per-series "${TIME_MAX_DATES_PER_SERIES:-10000}"
)

if [ -n "${TIME_DATASETS:-}" ]; then
    command+=(--datasets ${TIME_DATASETS})
fi
if [ -n "${TIME_SOURCE_ROOT:-}" ]; then
    command+=(--source-root "$TIME_SOURCE_ROOT")
fi
if [ -n "${TIME_CACHE_DIR:-}" ]; then
    command+=(--cache-dir "$TIME_CACHE_DIR")
fi
if [ "${TIME_OVERWRITE:-false}" = true ]; then
    command+=(--overwrite)
fi

log "TIME preparation output_root=$OUTPUT_ROOT frequencies=${TIME_FREQUENCIES:-15T H D} settings=${TIME_SETTINGS:-336:48 504:168} stride=${TIME_STRIDE:-512}"
if [ -n "${TIME_DATASETS:-}" ]; then
    log "TIME source datasets=$TIME_DATASETS"
else
    log "TIME source datasets=all"
fi

srun --ntasks=1 "${command[@]}"

log "validating TIME catalog=$OUTPUT_ROOT/catalog.json"
srun --ntasks=1 uv run python - "$OUTPUT_ROOT/catalog.json" <<'PY'
import json
import sys
from pathlib import Path

catalog_path = Path(sys.argv[1])
catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
datasets = catalog["datasets"]
if not datasets:
    raise RuntimeError(f"TIME preparation produced no datasets: {catalog_path}")
missing = [
    entry["name"]
    for entry in datasets
    if not (catalog_path.parent / entry["csv"]).is_file()
    or not (catalog_path.parent / entry["config"]).is_file()
]
if missing:
    raise RuntimeError(f"TIME catalog references missing outputs: {missing}")
print(
    f"Validated {len(datasets)} TIME datasets, {catalog['num_series']} series, "
    f"and {len(catalog['skipped'])} skipped inputs."
)
print("Prepared dataset names: " + " ".join(entry["name"] for entry in datasets))
PY

log "TIME preparation completed successfully"
