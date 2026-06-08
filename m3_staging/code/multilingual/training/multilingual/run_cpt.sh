#!/bin/bash
# Move 2, stage 1: full-FT continued pretraining on the regional Wikipedia
# corpus. No bitsandbytes workaround needed here — CPT is full-FT and never
# imports peft, so the bnb-import crash that hits LoRA training doesn't apply.
#
# Invoked from runai submit:
#   --command -- bash /scratch/multilingual/training/multilingual/run_cpt.sh
set -e

echo "[run_cpt] starting continued pretraining (full-FT)..."
exec python3 /scratch/multilingual/training/multilingual/train_cpt.py \
    --corpus_dir /scratch/multilingual/datasets/multilingual/cpt_corpus \
    --output_dir /scratch/multilingual/training/multilingual/outputs/cpt_v1 \
    --run_name cpt_v1
