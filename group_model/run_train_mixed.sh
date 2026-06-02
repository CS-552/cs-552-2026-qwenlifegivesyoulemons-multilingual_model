#!/bin/bash
# Group_model mixed-SFT cluster wrapper.
#
# Runs train_mixed.py on the combined math + gk + multilingual data,
# with safety-band layers frozen, per-domain weighted sampling, and Qwen3's
# default chat template (thinking-on by default).
#
# Invoked from runai submit:
#   --command -- bash /scratch/multilingual/group_model/run_train_mixed.sh
set -e

echo "[run_train_mixed] fixing bitsandbytes (cluster image ships incompatible bnb)..."
pip install --quiet --no-deps --force-reinstall "bitsandbytes>=0.43.0"
python3 -c "import bitsandbytes; print('[run_train_mixed] bnb import OK')"

echo "[run_train_mixed] starting group_model mixed-SFT..."
exec python3 /scratch/multilingual/group_model/train_mixed.py \
    --train_file /scratch/multilingual/group_model/data/train.jsonl \
    --dev_file   /scratch/multilingual/group_model/data/dev.jsonl \
    --output_dir /scratch/multilingual/group_model/outputs/mixed_v1 \
    --run_name group_mixed_v1 \
    --lora_r 64 \
    --lora_alpha 128
