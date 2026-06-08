#!/bin/bash
# Move 2, stage 2: SFT-LoRA on top of the CPT'd checkpoint.
#
# Identical to run_train.sh EXCEPT --base_model points at the stage-1 CPT
# output instead of vanilla Qwen3-1.7B. The bnb workaround IS needed here
# (this stage uses PEFT, which imports bitsandbytes at LoRA-creation time).
#
# Prereq: stage 1 (run_cpt.sh) must have finished and produced
#   /scratch/multilingual/training/multilingual/outputs/cpt_v1/final/
#
# Invoked from runai submit:
#   --command -- bash /scratch/multilingual/training/multilingual/run_sft_on_cpt.sh
set -e

CPT_CKPT=/scratch/multilingual/training/multilingual/outputs/cpt_v1/final
if [ ! -d "$CPT_CKPT" ]; then
    echo "[run_sft_on_cpt] ERROR: CPT checkpoint not found at $CPT_CKPT"
    echo "[run_sft_on_cpt] Run stage 1 (run_cpt.sh) first."
    exit 1
fi

echo "[run_sft_on_cpt] fixing bitsandbytes (cluster image ships incompatible bnb)..."
pip install --quiet --no-deps --force-reinstall "bitsandbytes>=0.43.0"
python3 -c "import bitsandbytes; print('[run_sft_on_cpt] bnb import OK')"

echo "[run_sft_on_cpt] SFT-LoRA on CPT checkpoint: $CPT_CKPT"
exec python3 /scratch/multilingual/training/multilingual/train_lora.py \
    --base_model "$CPT_CKPT" \
    --train_file /scratch/multilingual/datasets/multilingual/train.jsonl \
    --dev_file /scratch/multilingual/datasets/multilingual/dev.jsonl \
    --output_dir /scratch/multilingual/training/multilingual/outputs/lora_v4 \
    --run_name lora_v4 \
    --lora_r 64 \
    --lora_alpha 128
