# group_model — weight-averaging fusion

First strategy of three under comparison: merge the 4 specialty checkpoints
into a single Qwen3-1.7B-shaped model that the course CI evaluates on all 4
benchmarks.

## Team specialty mix

Three teammates trained **full-FT** specialties; one (multilingual) is
LoRA-merged-to-full. All four end up on HuggingFace as full Qwen3-1.7B
safetensors checkpoints, so the merge math is uniform at the artifact level.
But the underlying deltas-from-base are very different in magnitude:
**full-FT specialties drift much farther from Qwen3-1.7B than LoRA-merged
ones do.** That asymmetry drives the choice of merge method.

## Strategies tried, in order

| Config | Method | Why try it |
|---|---|---|
| `merge_linear.yaml` | Plain equal-weight linear average | Baseline / floor. Likely weak — full-FT contributions will dominate the LoRA-merged one. |
| `merge_ties.yaml` | TIES (density-pruned task vectors) | The principled fix: subtracts the common base to get per-specialty *task vectors*, prunes low-magnitude noise, resolves sign conflicts, then merges. Standard tool for merging full-FT models in the same family. |

If neither beats the strongest individual specialty (e.g., your multilingual
v3 at 75%), we either tune TIES (`density`, per-specialty `weight`) or move
to the next fusion strategy (mixed-SFT or KD).

## Workflow

```bash
# On the cluster, in an interactive pod:

cd /scratch/multilingual/group_model    # or wherever this repo is cloned

# 1) Plain linear average (baseline, ~5 min)
bash run_merge.sh merge_linear.yaml outputs/linear_v1
python3 push_group.py \
    --merged_dir outputs/linear_v1 \
    --branch linear-baseline \
    --commit_msg "Linear merge baseline" \
    --push                  # pushes to branch, not main, for archival

# 2) TIES merge (the real attempt, ~15 min)
bash run_merge.sh merge_ties.yaml outputs/ties_v1
python3 push_group.py \
    --merged_dir outputs/ties_v1 \
    --commit_msg "TIES merge v1 (density=0.5, equal weights)" \
    --push                  # default branch=main, CI evaluates this
```

Each merge downloads all 4 specialty checkpoints (~14 GB total to
`/scratch/hf_cache`, cached across pods). After the first run the
download is free for the team.

## What `push_group.py` does on top of mergekit

- Forces `no_think` on the tokenizer mergekit copied from a source model
  (whichever specialty was first in the YAML — its chat template might
  have thinking-mode on; we want pass@1 + 1800s budget => no_think).
- Writes `generation_config.json` with the tightened settings
  (`max_new_tokens=32`, `eos_token_id=<|im_end|>`) — same as the
  specialty pushes.
- Writes `.push_metadata.json` to guarantee HF's `lastModified` advances
  on every push (so the CI never silently skips a re-eval).
- Optionally pushes to a non-main branch (for archival comparison
  between linear and TIES, etc.).

## Why the chat template matters here

The mergekit output's tokenizer is copied from one source model. The math
or general_knowledge specialties might use thinking-mode for their reasoning
traces. For the group_model evaluated on **pass@1** with a **1800s** wall-
clock cap across 4 benchmarks, thinking-mode is a tax: long reasoning
strings eat the budget. Forcing `no_think` (via the same one-line Jinja
override as `training/multilingual/chat_template.jinja`) makes the merged
model emit `\boxed{X}` directly. If a teammate's specialty was trained
*expecting* thinking output, the merged model may struggle on that
domain — that's diagnostic feedback for the next iteration.

## When to re-weight

If the first TIES result is dominated by one specialty (e.g., the merged
model is great at math, weak elsewhere), drop that specialty's `weight`
in the YAML (e.g., `0.15`) and bump the under-represented ones (`0.30`).
Re-run, push to a new branch, compare.
