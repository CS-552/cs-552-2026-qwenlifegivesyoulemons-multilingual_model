# v5 — bilko-style chat template on v3 data

**Status:** current release on `cs-552-2026-qwenlifegivesyoulemons/multilingual_model` (commit `da69392`, `main` branch)
**CI score:** 0.75 (2026-06-05 EVAL_REPORT, scored against `main`/v5). The 2026-06-08 EVAL_REPORT shows 0.69 but pulled from the `v3` branch, not `main` — v5 files were never re-evaluated. See `Releases/README.md` for the open CI-resolution issue.

## Idea
v1→v2→v3 plateaued at ~0.75 across three data interventions; v4_cpt regressed (-6pp).
v5 holds the v3 data fixed and perturbs the **prompt framing instead of the data**:
the tokenizer's `chat_template` is replaced with a Bilko-style override that
sets `enable_thinking = false`, injects a `default_system_message` framing the
model as a "multilingual multiple-choice classifier", and appends a per-turn
`user_instruction_suffix` reinforcing the output format. The override is
applied at **push time only** — training itself runs against the standard
Qwen3 template so the LoRA weights are template-agnostic.

## Data (same as v3)
- **Global-MMLU** (it, es, zh, ru, hi) — translated MMLU baseline (no English bleed).
- **INCLUDE** — native MCQ across regions (small, high-quality, upweighted).
- **MMLU-en CS-only subset** — English representation, kept narrow so the model
  doesn't English-bias non-English prompts.
- **Native MCQ**: CEval (zh), CMMLU (zh), Kaleidoscope (multi), EXAMS (multi),
  HEAD-QA (es), MILU (hi).

Build (from `datasets/multilingual/scripts/`):
```bash
python3 build_global_mmlu.py
python3 build_include.py
python3 build_mmlu_en.py
python3 build_ceval.py
python3 build_cmmlu.py
python3 build_kaleidoscope.py
python3 build_exams.py
python3 build_headqa.py
python3 build_milu.py
python3 build_all.py --copies 2 --cs_weight 5 --include_weight 4 --bulk_cap 4000
```
Output: `data/train.jsonl` and `data/dev.jsonl` under `training/multilingual/`.

## Training
LoRA: `r=64`, `α=128`, dropout 0.05, target modules = {q,k,v,o, gate_proj, up_proj, down_proj}.
Optimizer: AdamW, lr 2e-4, cosine schedule, warmup 3%.
Precision: bf16 + gradient checkpointing.
Batch: per-language WeightedRandomSampler (cross-bucket balance independent of raw counts).

```bash
bash run.sh   # see this folder's run.sh — wraps run_train.sh with v5 paths
```

## Push (where the template change actually lands)
```bash
python3 push_to_hub.py \
    --adapter_dir outputs/lora_v5/final \
    --hf_repo cs-552-2026-qwenlifegivesyoulemons/multilingual_model \
    --push
```
`push_to_hub.py` reads `chat_template.jinja` and **SETs** (not prepends) it
as the merged model's `tokenizer.chat_template`. The Jinja file is included
in this release folder for archival.

## Why v5 over v3
Data ablation hit a ceiling (~0.75) and CPT regressed. The next axis to
perturb was prompt framing without retraining. v5 ships the same weights
as v3 with the bilko-style template. The first CI EVAL_REPORT against
`main` scored 0.75; a later EVAL_REPORT reading 0.69 turned out to be
evaluating the `v3` branch (not `main`), so it does not measure v5 — only
v3 under current CI rules. The actionable item is fixing the CI's branch
resolution, not the training recipe.

## Files in this folder
- `README.md` — this file
- `run.sh` — pinned cluster command
- `chat_template.jinja` — the exact template baked into the v5 push
