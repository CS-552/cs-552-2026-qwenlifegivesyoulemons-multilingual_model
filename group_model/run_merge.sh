#!/bin/bash
# Run a merge YAML using our custom merge.py (replaces mergekit because
# mergekit 0.1.x crashes on the cluster image's pydantic — see merge.py
# docstring for details).
#
# Usage:
#   bash run_merge.sh merge_linear.yaml outputs/linear_v1
#   bash run_merge.sh merge_ties.yaml   outputs/ties_v1
set -e

YAML="$1"
OUT="$2"
if [ -z "$YAML" ] || [ -z "$OUT" ]; then
    echo "Usage: bash run_merge.sh <yaml> <output_dir>"
    exit 1
fi

# pyyaml is standard but the image might lack it; quiet install if missing.
python3 -c "import yaml" 2>/dev/null || pip install --quiet pyyaml

echo "[run_merge] $YAML -> $OUT"
exec python3 /scratch/multilingual/group_model/merge.py "$YAML" "$OUT"
