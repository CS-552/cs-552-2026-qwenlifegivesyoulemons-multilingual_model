#!/bin/bash
# Wrapper around train_lora.py for the EPFL RCP cluster.
#
# The course image ships an old bitsandbytes without a CUDA 12.8 binary,
# but PEFT imports bnb at LoRA-creation time even when we never call into it.
# Force-reinstall a recent wheel that has the cu128 binary, then train.
#
# Invoked from runai submit:
#   --command -- bash /scratch/multilingual/training/multilingual/run_train.sh
set -e

echo "[run_train] fixing bitsandbytes (cluster image ships an incompatible version)..."
pip install --quiet --upgrade --force-reinstall "bitsandbytes>=0.44.0"

echo "[run_train] starting training..."
exec python3 /scratch/multilingual/training/multilingual/train_lora.py \
    --train_file /scratch/multilingual/datasets/multilingual/train.jsonl \
    --dev_file /scratch/multilingual/datasets/multilingual/dev.jsonl \
    --output_dir /scratch/multilingual/training/multilingual/outputs/lora_v1 \
    --run_name lora_v1
