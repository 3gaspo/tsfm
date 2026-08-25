#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SECRET_FILE="$HOME/codes/.secrets/proxy.credentials"

if [ ! -f "$SECRET_FILE" ]; then
    echo "ERROR: missing $SECRET_FILE" >&2
    exit 1
fi

NNI="$(sed -n '1p' "$SECRET_FILE" | tr -d '[:space:]')"
nni="${NNI,,}"
if [[ ! "$nni" =~ ^[a-z][a-z0-9_-]*$ ]]; then
    echo "ERROR: the first line of $SECRET_FILE must contain the NNI" >&2
    exit 1
fi

SOURCE_ROOT="$PROJECT_ROOT"
DESTINATION_ROOT="$nni@dgx-front.retd.edf.fr:/home/$nni/codes/tsfm"

if [ ! -d "$SOURCE_ROOT/outputs" ] || [ ! -d "$SOURCE_ROOT/logs" ]; then
    echo "ERROR: expected $SOURCE_ROOT/outputs and $SOURCE_ROOT/logs on Selena" >&2
    exit 1
fi

echo "Synchronizing TSFM outputs from Selena to DGX..."
rsync -rlptz --partial --info=progress2 \
    "$SOURCE_ROOT/outputs/" \
    "$DESTINATION_ROOT/outputs/"

echo "Synchronizing TSFM logs from Selena to DGX..."
rsync -rlptz --partial --info=progress2 \
    "$SOURCE_ROOT/logs/" \
    "$DESTINATION_ROOT/logs/"

echo "SUCCESS: outputs and logs were transferred to DGX."
