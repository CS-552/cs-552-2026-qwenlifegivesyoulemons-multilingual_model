# v_ties — TIES merge of four specialists (rejected)

**Status:** rejected; kept for ablation reporting
**CI 4-domain average:** 0.295 (vs 0.525 for v_mixed_v1; -23pp)

## Idea
Yadav et al.'s TIES merging: trim each specialist's parameter delta vs the
base to its top-`density` magnitudes per tensor, elect a sign per parameter
across the four specialists, and merge with equal weights.

We picked TIES over plain linear averaging because:
- Plain averaging is known to suffer from sign-conflict collapse.
- TIES's sign election handles heterogeneous specialty signals in theory.

## Why it failed at 1.7B
1. **Heterogeneous fine-tune scales.** Math/GK/safety specialties are
   full-FT (large delta magnitudes); multilingual is LoRA-merged (small
   delta after `merge_and_unload`). The LoRA delta gets pruned away by the
   density filter or out-voted in sign election.
2. **Small base model.** At 1.7B, each parameter carries more functional
   load per neuron than at 7B+, so destructive interference between
   specialty deltas is harsher.
3. **Equal weights penalize the smaller delta.** Setting weight 0.25 for
   the multilingual LoRA-derived specialty further dilutes its contribution.

## Reproducing
Merge config: `merge_ties.yaml` (also in `group_model/`):
```yaml
merge_method: ties
base_model: Qwen/Qwen3-1.7B
density: 0.5
models:
  - model: <math specialist HF id>
    parameters: { weight: 0.25 }
  - model: <gk specialist HF id>
    parameters: { weight: 0.25 }
  - model: <safety specialist HF id>
    parameters: { weight: 0.25 }
  - model: <multilingual specialist HF id>
    parameters: { weight: 0.25 }
```
Custom merger (mergekit's pydantic 2.10.6 dependency breaks on the cluster image):
```bash
python3 merge.py --config merge_ties.yaml --output_dir outputs/ties
```

## What we learned
Naive merge methods need to be benchmarked against mixed-SFT at small scale.
The +23pp gap is the headline empirical finding of the joint-model effort.

## Files in this folder
- `README.md` — this file
- `merge_ties.yaml` — the merge config used
