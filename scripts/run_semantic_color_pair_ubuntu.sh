#!/usr/bin/env bash
# Activate the previously working semantic CUDA environment before running.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
LEGACY_DIR="runs/checkpoints/flir_v2_struct_sem512_protected_seed42_ubuntu_aligned_40_40"
SEMANTIC_NAME="flir_v2_semonly_from80_seed42_20261020_preview5"
COLOR_NAME="flir_v2_semcolor_from80_seed42_20261020_preview5"

# Check both destinations before starting the first five-epoch run.
for name in "$SEMANTIC_NAME" "$COLOR_NAME"; do
    for directory in runs/checkpoints runs/results; do
        if [[ -e "$directory/$name" ]]; then
            printf 'Experiment already exists: %s/%s. Preserve it and use new names.\n' "$directory" "$name" >&2
            exit 1
        fi
    done
done
"$PYTHON_BIN" -u scripts/package_semantic_color_ubuntu.py --verify-installed
"$PYTHON_BIN" -u scripts/run_semantic_color.py --check-only

printf '\nStarting semantic only: %s\n' "$SEMANTIC_NAME"
"$PYTHON_BIN" -u scripts/run_semantic_color.py --train \
    --name "$SEMANTIC_NAME" --color-condition semantic --color-weight 0 \
    --init-semantic-dir "$LEGACY_DIR" --init-semantic-epoch 80 \
    --stop-after-epoch 5 --lr 0.00002

printf '\nStarting semantic plus manual color prior: %s\n' "$COLOR_NAME"
"$PYTHON_BIN" -u scripts/run_semantic_color.py --train \
    --name "$COLOR_NAME" --color-condition semantic_color --color-weight 1 \
    --init-semantic-dir "$LEGACY_DIR" --init-semantic-epoch 80 \
    --stop-after-epoch 5 --lr 0.00002

printf '\nBoth runs and their test_5 exports completed.\n'
