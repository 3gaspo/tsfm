#!/bin/bash

set -euo pipefail

SOURCE_ROOT="/home/h61084/codes/tsfm/"
DESTINATION="h61084@selena.hpc.edf.fr:~/codes/tsfm/"

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
