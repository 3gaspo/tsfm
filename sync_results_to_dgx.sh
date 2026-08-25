#!/bin/bash

set -euo pipefail

SOURCE_ROOT="$HOME/codes/tsfm"
DESTINATION_ROOT="h61084@dgx-front.retd.edf.fr:/home/h61084/codes/tsfm"

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
