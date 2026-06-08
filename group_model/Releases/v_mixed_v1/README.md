# v_mixed_v1 — Mixed-SFT + SPPFT safety-band freeze

**Status:** current release on `cs-552-2026-qwenlifegivesyoulemons/group_model`
**CI 4-domain average:** 0.525 (math + gk + safety + multilingual)

## Idea
TIES merging of four specialists collapsed to 0.295 at this scale: full-FT
specialties dominate, the LoRA-merged multilingual delta gets washed out.
Instead, train **one shared LoRA adapter** on the union of three specialty
datasets via mixed-SFT, with two design choices:

1. **Per-domain WeightedRandomSampler** — equalizes batch composition so
   the larger math pool doesn't drown out gk/multilingual.
2. **Safety-band layer freezing (SPPFT)** — transformer blocks 15-19 of
   Qwen3-1.7B form the safety-discrimination band (Li et al. 2024,
   arXiv:2408.17003). LoRA target_modules **exclude** those layers, so
   the base model's safety alignment is preserved during multi-domain SFT
   without needing safety data in the mix.

## Data
| Source                                | Domain         | Items   | Target format                     |
|---------------------------------------|----------------|---------|-----------------------------------|
| `build_math.py` (NuminaMath-CoT, 50k)| math           | 46,556  | filled `<think>...</think>` + `\boxed{answer}` |
| `build_gk.py` (MMLU-Pro + GPQA + MedMCQA) | gk        | 22,230  | empty `<think>\n\n</think>` + `\boxed{LETTER}` |
| `build_multilingual_v1.py`            | multilingual   | 182,224 | empty `<think>\n\n</think>` + `\boxed{LETTER}` |

Combine + 200/domain dev holdout:
```bash
python3 build_math.py
python3 build_gk.py
python3 build_multilingual_v1.py
python3 build_mixed_sft.py --dev_per_domain 200
```
Output: `data/train.jsonl` (250,410 rows) + `data/dev.jsonl` (600 rows).

## Training
LoRA: `r=64`, `α=128`, dropout 0.05, target modules = {q,k,v,o, gate_proj, up_proj, down_proj}.
Layers: **23 of 28** (blocks 0-14 and 20-27); blocks 15-19 untouched.
Trainable parameter share: ~3.2% (vs ~3.9% without the safety-band exclusion).

```bash
bash run.sh   # see this folder's run.sh — wraps run_train_mixed.sh
```

## Push (keeps Qwen3 default template)
```bash
python3 push_group_mixed.py \
    --adapter_dir outputs/mixed_v1/final \
    --hf_repo cs-552-2026-qwenlifegivesyoulemons/group_model \
    --push
```
**Critical**: do NOT apply the multilingual specialty's bilko-style template
here. The group model needs `enable_thinking=true` so math can reason.
`push_group_mixed.py` deliberately omits the template override.

## Decoding config baked into the push
```json
{
  "max_new_tokens": 2048,
  "do_sample": true,
  "temperature": 0.2,
  "top_p": 0.9,
  "top_k": 50,
  "eos_token_id": "<|im_end|>"
}
```
`max_new_tokens=2048` so math reasoning chains fit; MC items ignore most.

## Why this beat TIES
TIES averages weight deltas across heterogeneous fine-tunes (full-FT vs LoRA,
different domains), and at 1.7B the deltas collapse destructively. Mixed-SFT
sidesteps the merge problem by training the joint signal from scratch on the
union of data. SPPFT-style freezing handles safety as a separate concern:
since the safety discrimination band is intact, the base model's RLHF
alignment carries through unchanged.

## Files in this folder
- `README.md` — this file
- `run.sh` — pinned cluster command
