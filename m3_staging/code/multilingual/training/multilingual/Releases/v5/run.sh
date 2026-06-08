#!/bin/bash
# v5 reproduction wrapper. Submitted via runai from Git Bash with
# MSYS_NO_PATHCONV=1 to avoid /scratch path mangling.
#
# Wraps the shared training/multilingual/run_train.sh with v5-specific
# output_dir and lora hyperparameters. The chat-template override is applied
# at PUSH time (push_to_hub.py reads chat_template.jinja), not here.
set -e

echo "[v5/run.sh] fixing bitsandbytes (cluster image ships incompatible bnb)..."
pip install --quiet --no-deps --force-reinstall "bitsandbytes>=0.43.0"
python3 -c "import bitsandbytes; print('[v5/run.sh] bnb import OK')"

echo "[v5/run.sh] starting v5 multilingual LoRA SFT..."
exec python3 /scratch/multilingual/training/multilingual/train_lora.py \
    --train_file /scratch/multilingual/training/multilingual/data/train.jsonl \
    --dev_file   /scratch/multilingual/training/multilingual/data/dev.jsonl \
    --output_dir /scratch/multilingual/training/multilingual/outputs/lora_v5 \
    --run_name multilingual_v5 \
    --lora_r 64 \
    --lora_alpha 128
