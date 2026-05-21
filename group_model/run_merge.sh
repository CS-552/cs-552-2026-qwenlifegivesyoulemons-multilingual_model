#!/bin/bash
# Install mergekit (if missing) and run a merge YAML.
#
# Usage:
#   bash run_merge.sh merge_linear.yaml outputs/linear_v1
#   bash run_merge.sh merge_ties.yaml   outputs/ties_v1
#
# Output directory will contain config.json, model.safetensors[.index.json],
# tokenizer files, etc. — a complete Qwen3-1.7B-shaped checkpoint ready to
# be finalized by push_group.py and uploaded to HF.
set -e

YAML="$1"
OUT="$2"
if [ -z "$YAML" ] || [ -z "$OUT" ]; then
    echo "Usage: bash run_merge.sh <yaml> <output_dir>"
    exit 1
fi

echo "[run_merge] installing mergekit..."
pip install --quiet mergekit

# --cuda: use the A100 (1.7B merge is fast on GPU, slow on CPU)
# --lazy-unpickle: load tensors lazily to keep memory peak low
# --allow-crimes: tolerate minor config drift between specialties (they all
#                 share architecture; this just avoids spurious assert failures)
echo "[run_merge] running merge: $YAML -> $OUT"
mergekit-yaml "$YAML" "$OUT" \
    --cuda \
    --lazy-unpickle \
    --allow-crimes

echo "[run_merge] merge complete. Next:"
echo "    python3 push_group.py --merged_dir $OUT --push"
