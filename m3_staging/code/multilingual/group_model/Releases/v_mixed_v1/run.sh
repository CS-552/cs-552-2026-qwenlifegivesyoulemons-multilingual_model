#!/bin/bash
# v_mixed_v1 reproduction wrapper. Submitted via runai from Git Bash with
# MSYS_NO_PATHCONV=1 to avoid /scratch path mangling.
#
# Identical to group_model/run_train_mixed.sh — duplicated here so the
# release folder is self-contained for the grader.
set -e

echo "[v_mixed_v1/run.sh] fixing bitsandbytes (cluster image ships incompatible bnb)..."
pip install --quiet --no-deps --force-reinstall "bitsandbytes>=0.43.0"
python3 -c "import bitsandbytes; print('[v_mixed_v1/run.sh] bnb import OK')"

echo "[v_mixed_v1/run.sh] starting group_model mixed-SFT (safety-band frozen)..."
exec python3 /scratch/multilingual/group_model/train_mixed.py \
    --train_file /scratch/multilingual/group_model/data/train.jsonl \
    --dev_file   /scratch/multilingual/group_model/data/dev.jsonl \
    --output_dir /scratch/multilingual/group_model/outputs/mixed_v1 \
    --run_name group_mixed_v1 \
    --lora_r 64 \
    --lora_alpha 128
