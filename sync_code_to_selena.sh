#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SECRET_FILE="$HOME/codes/.secrets/nni"

if [ ! -f "$SECRET_FILE" ]; then
    echo "ERROR: missing $SECRET_FILE" >&2
    exit 1
fi

NNI="$(tr -d '[:space:]' < "$SECRET_FILE")"
nni="${NNI,,}"
if [[ ! "$nni" =~ ^[a-z][a-z0-9_-]*$ ]]; then
    echo "ERROR: $SECRET_FILE must contain only the NNI" >&2
    exit 1
fi

SOURCE_ROOT="/home/$nni/codes/tsfm/"
DESTINATION="$nni@selena.hpc.edf.fr:~/codes/tsfm/"

if [ ! -d "$SOURCE_ROOT/src" ] || [ ! -f "$SOURCE_ROOT/pyproject.toml" ]; then
    echo "ERROR: TSFM project not found at $SOURCE_ROOT" >&2
    exit 1
fi

echo "Synchronizing TSFM code from DGX to Selena..."
rsync -rlptz --delete --partial --info=progress2 \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='.secrets/' \
    --exclude='datasets/' \
    --exclude='weights/' \
    --exclude='outputs/' \
    --exclude='logs/' \
    "$SOURCE_ROOT" \
    "$DESTINATION"

echo "SUCCESS: Selena's TSFM code matches DGX."
echo "Preserved on Selena: .venv, .secrets, datasets, weights, outputs, and logs."
