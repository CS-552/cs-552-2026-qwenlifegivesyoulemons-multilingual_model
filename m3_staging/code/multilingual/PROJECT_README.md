# qwenlifegivesyoulemons — Project Code

Code for the EPFL CS-552 MNLP Spring 2026 standard project. Post-trains
`Qwen/Qwen3-1.7B` into four single-domain specialists and a single joint
"group" model, evaluated by the course CI on a hidden test set.

This README is the **how-to-run-this-repo** document required by M3.
The starter README from M2 (CI rules, evaluation contract, validation
samples) is preserved at `README.md` for reference.

---

## Team layout

| Repo path (HF)                                       | Owner              | Status   |
|------------------------------------------------------|--------------------|----------|
| `cs-552-2026-qwenlifegivesyoulemons/group_model`     | whole team         | shipped  |
| `cs-552-2026-qwenlifegivesyoulemons/math_model`      | teammate (math)    | shipped  |
| `cs-552-2026-qwenlifegivesyoulemons/general_knowledge_model` | teammate (gk) | shipped  |
| `cs-552-2026-qwenlifegivesyoulemons/safety_model`    | teammate (safety)  | shipped  |
| `cs-552-2026-qwenlifegivesyoulemons/multilingual_model` | (me)            | shipped  |

This repo contains everything for the **multilingual specialty** and the
**group model**. The other three specialty pipelines live in teammates'
trees but produce CI-compatible artefacts that get merged here for the
TIES baseline (rejected) and are referenced by HF id only for the
mixed-SFT group model (current).

---

## Directory layout

```
.
├── README.md                       course starter (unchanged from M2)
├── PROJECT_README.md               THIS FILE — team how-to-run guide
├── datasets/multilingual/          dataset builders (raw -> intermediate -> mixed)
│   ├── scripts/
│   │   ├── common.py               LANGS, schemas, JSONL helpers
│   │   ├── augment_variable_k.py   2..20 option-count augmentation
│   │   ├── build_*.py              one builder per source (10 sources)
│   │   └── build_all.py            orchestrator: clean -> split -> augment -> jsonl
│   ├── cleaned_datasets/           intermediate JSONL (per source / per lang)
│   ├── train.jsonl, dev.jsonl      produced by build_all.py
│   └── cpt_corpus/                 regional Wikipedia for v4 CPT (rejected)
├── training/multilingual/
│   ├── train_lora.py               LoRA SFT (current production path)
│   ├── train_cpt.py                full-FT continued pretraining (v4, rejected)
│   ├── push_to_hub.py              merge LoRA + apply bilko template + push
│   ├── chat_template.jinja         the bilko-style MC-classifier template
│   ├── run_train.sh                cluster wrapper (calls train_lora.py)
│   ├── run_cpt.sh                  cluster wrapper (calls train_cpt.py)
│   └── Releases/                   per-version reproduction (currently: v5)
├── group_model/
│   ├── build_math.py               NuminaMath -> filled-<think> CoT targets
│   ├── build_gk.py                 MMLU-Pro/GPQA/MedMCQA -> empty-<think> MC
│   ├── build_multilingual_v1.py    v1-era multilingual subset for the mix
│   ├── build_mixed_sft.py          stitch + train/dev split (200/domain)
│   ├── train_mixed.py              mixed-SFT LoRA with SPPFT safety-band freeze
│   ├── push_group_mixed.py         merge + push (KEEPS Qwen3 default template)
│   ├── merge.py                    self-contained TIES/linear (mergekit replacement)
│   ├── merge_ties.yaml             TIES config (rejected baseline)
│   ├── eval_dev.py                 local per-domain pass@1 on dev.jsonl
│   ├── eval_base.py                local pass@1 of UN-fine-tuned Qwen3-1.7B (baseline)
│   ├── run_train_mixed.sh          cluster wrapper for mixed-SFT
│   └── Releases/                   per-version reproduction (v_mixed_v1, v_ties)
├── validation_samples/             frozen 10-item sample per domain (course-provided)
└── evaluate/                       course-provided sample eval code (read-only)
```

---

## Setup

### Local (Windows / Linux / macOS for dataset prep)
```bash
conda create -n mnlp python=3.11 -y
conda activate mnlp
pip install -r requirements.txt
hf auth login          # paste a write-scope HF token (do NOT echo it)
```

### Cluster (EPFL RCP / Run:AI, A100 40GB)
Every cluster job re-installs `bitsandbytes>=0.43.0 --no-deps` because the
shipped image's bnb has no CUDA 12.8 binary and the pydantic resolver rejects
newer versions. The `run_*.sh` wrappers handle this automatically — do not
strip those lines.

From Git Bash, prepend `MSYS_NO_PATHCONV=1` to every `runai submit` to
prevent `/scratch` from being rewritten to `C:/Program Files/Git/scratch`:
```bash
MSYS_NO_PATHCONV=1 runai submit ...
```

Detailed cluster onboarding: see `TEAM_CLUSTER_GUIDE.md` at the repo root.

---

## Reproducibility — seeds

All randomness in this codebase is seeded so the grader's reproduction
should match the report numbers within sampling noise.

| Component                                  | Seed source                                  |
|--------------------------------------------|----------------------------------------------|
| Dataset assembly (split + augmentation)    | `build_all.py --seed 42` (default)            |
| Mixed-SFT data assembly                    | `build_mixed_sft.py --seed 42` (default)      |
| LoRA training (multilingual + group)       | `--seed 42` flag in train_lora / train_mixed  |
| WeightedRandomSampler                      | seeded from `args.seed` in both trainers      |
| CPT training                               | `train_cpt.py --seed 42`                      |
| Local evaluators                           | `--seed 42` in eval_base / eval_dev           |
| CI inference (out of our control)          | n=8 sampling at temp 0.2 / top_p 0.9 / top_k 50 |

Two things outside our seeding contract:
- vLLM sampling at CI time uses its own RNG state — pass@k metrics may
  vary by ~0.5pp between runs even with the same checkpoint.
- The CI changed from 4k context (until 2026-05-31) to 16k context (after).
  Some of our scores were measured under the 4k regime and may differ
  under 16k; see the per-release READMEs for which window applies.

---

## End-to-end: multilingual specialty

### 1. Build the data (one-time per version)
```bash
cd datasets/multilingual/scripts
python build_global_mmlu.py        # translated MMLU, 5 target langs
python build_include.py            # native regional MCQ, small
python build_mmlu_en.py            # English CS-only subset
python build_ceval.py              # zh native
python build_cmmlu.py              # zh native (hub raw-file workaround)
python build_kaleidoscope.py       # multi native
python build_exams.py              # multi native
python build_headqa.py             # es native (hub raw-file workaround)
python build_milu.py               # hi native
python build_all.py \
    --copies 2 --cs_weight 5 --include_weight 4 --bulk_cap 4000
```
Outputs `datasets/multilingual/train.jsonl` + `dev.jsonl`.

### 2. Train (cluster)
```bash
MSYS_NO_PATHCONV=1 runai submit \
    multilingual-lora-v5 \
    --image ic-registry.epfl.ch/cs-552/cs-552-image:latest \
    --gpu 1 --cpu 4 --memory 32G \
    --command -- bash /scratch/multilingual/training/multilingual/run_train.sh
```

### 3. Push to HF (where the bilko template gets applied)
```bash
cd training/multilingual
python push_to_hub.py \
    --adapter_dir outputs/lora_v5/final \
    --hf_repo cs-552-2026-qwenlifegivesyoulemons/multilingual_model \
    --push
```
`push_to_hub.py` reads `chat_template.jinja` and SETs it as the merged
tokenizer's `chat_template`. The `.push_metadata.json` it also writes
guarantees HF's `lastModified` advances so the next nightly CI run picks
up the new revision even when only the template changed.

### 4. Local sanity check
```bash
cd group_model
python eval_dev.py --model_dir ../training/multilingual/outputs/lora_v5/merged \
    --dev_file ../training/multilingual/dev.jsonl --do_sample
```

---

## End-to-end: group model (current = mixed-SFT)

### 1. Build the per-domain data
```bash
cd group_model
python build_math.py                # NuminaMath-CoT sample, 46556 rows
python build_gk.py                  # MMLU-Pro + GPQA + MedMCQA, 22230 rows
python build_multilingual_v1.py     # v1-era multilingual subset, 182224 rows
python build_mixed_sft.py --dev_per_domain 200
```
Outputs `group_model/data/train.jsonl` (~250k) + `dev.jsonl` (600).

### 2. Train (cluster)
```bash
MSYS_NO_PATHCONV=1 runai submit \
    group-mixed-v1 \
    --image ic-registry.epfl.ch/cs-552/cs-552-image:latest \
    --gpu 1 --cpu 4 --memory 32G \
    --command -- bash /scratch/multilingual/group_model/run_train_mixed.sh
```
LoRA `r=64`, `α=128`, 23 of 28 transformer blocks targeted — blocks 15-19
(the SPPFT safety-discrimination band) are EXCLUDED so the base model's
safety alignment is preserved without retraining on safety data.

### 3. Push (KEEPS Qwen3 default chat template)
```bash
python push_group_mixed.py \
    --adapter_dir outputs/mixed_v1/final \
    --hf_repo cs-552-2026-qwenlifegivesyoulemons/group_model \
    --push
```
**Do not** apply the multilingual specialty's bilko template here — the
group model needs `enable_thinking=true` so math reasoning chains fit.
`push_group_mixed.py` deliberately omits the override.

### 4. Local sanity check
```bash
python eval_dev.py --model_dir outputs/mixed_v1/merged --dev_file data/dev.jsonl --do_sample
```

---

## Baseline (un-fine-tuned Qwen3-1.7B)

The "+18-24pp" claim for multilingual and the "+23pp" claim for group
model both reference this baseline:
```bash
cd group_model
# Joint dev (3 domains, 600 items)
python eval_base.py --dev_file data/dev.jsonl

# Multilingual-only dev
python eval_base.py --dev_file ../datasets/multilingual/dev.jsonl
```

---

## Multilingual ablation history

These versions were trained and either kept or rejected. Full per-version
reproducibility is in `training/multilingual/Releases/`.

| Version  | Data + recipe                                                              | Pass@1 (CI) | Status               |
|----------|----------------------------------------------------------------------------|-------------|----------------------|
| v1       | Global-MMLU + INCLUDE + MMLU-en CS-only; copies=2                          | 0.74        | superseded           |
| v2       | v1 + CS upsampling (`--cs_weight 5`)                                       | 0.75        | superseded           |
| v3       | v2 + native MCQ (CEval, CMMLU, Kaleidoscope, EXAMS, HEAD-QA, MILU)         | 0.75        | superseded by v5     |
| v4_cpt   | v3 base + 1-epoch CPT on regional Wikipedia (5 langs, 200k articles)       | 0.69        | regression, rejected |
| v5       | v3 data + bilko-style chat template (enable_thinking=false + MC system msg)| 0.75 (main) | **current**          |

Notes:
- **v4_cpt regression (-6pp)**: full-FT CPT without rehearsal mix produced
  catastrophic forgetting. Reverted to v3 weights on HF via a re-push.
- **v5 score (0.75)**: measured by the 2026-06-05 EVAL_REPORT against
  commit `da69392` on `main`. A later 0.69 EVAL_REPORT turned out to be
  pulling from the `v3` branch, not `main`, so it measures v3 under current
  CI rules — not v5. Open item: fix the CI's branch resolution (HF default
  branch / refs check).

## Group-model ablation

Full per-version reproducibility in `group_model/Releases/`.

| Version       | Strategy                                                          | 4-domain avg | Status   |
|---------------|-------------------------------------------------------------------|--------------|----------|
| v_ties        | TIES merge of 4 specialists (density 0.5, equal weights)          | 0.295        | rejected |
| v_mixed_v1    | Mixed-SFT (math+gk+multilingual) + SPPFT safety-band layer freeze | 0.525        | current  |

Headline: **mixed-SFT + safety-band freezing beats TIES by 23pp at the
1.7B scale**, while keeping safety at the base-model level (0.74).

---

## Course CI — what to expect

- **Frequency**: nightly until 2026-05-31; every 48h after.
- **Context length**: 4k until 2026-05-31; 16k after.
- **Sampling**: n=8 completions per problem at temp 0.2 / top_p 0.9 / top_k 50.
- **Hard cap**: 1800s wall-clock per model on 1× A100 40GB.
- **Re-eval trigger**: HF `lastModified` advancing. Our pushes write a
  `.push_metadata.json` with a fresh ISO timestamp so weight-only or
  template-only diffs still trigger a fresh eval.

Common failure modes and their fixes are listed in the M2 README; the
ones we hit and resolved during the project:

| Symptom                                  | Cause                                                  | Fix                                                       |
|------------------------------------------|--------------------------------------------------------|-----------------------------------------------------------|
| CI didn't trigger after weight push      | `lastModified` unchanged                               | `.push_metadata.json` with current timestamp (in push_to_hub.py) |
| `pass@1 ≈ 0` on multilingual after CPT   | Catastrophic forgetting of the boxed-format contract   | Reverted to pre-CPT LoRA weights                          |
| Score dropped 5pp after switching greedy | Calibration-bound model — modal completion often wrong | Reverted to sampling at temp 0.2                          |
| 0.75 → 0.69 on apparently-same commit    | CI resolved to `v3` branch instead of `main`           | Verify HF default branch == `main`; consider explicit `revision="main"` on upload_folder |

---

## Decoding configs (baked into each push)

### Multilingual specialty
```json
{
  "max_new_tokens": 32,
  "do_sample": true,
  "temperature": 0.2,
  "top_p": 0.9,
  "top_k": 50,
  "eos_token_id": "<|im_end|>"
}
```
32 tokens is enough for `\boxed{X}` plus the closing `<|im_end|>` — the
template forces `enable_thinking=false` so there's no reasoning chain.

### Group model
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
2048 tokens because math reasoning chains need the headroom. MC items
ignore most of the budget.

---

## Key references

- **SPPFT** (safety-band layer freezing): Li et al. 2024,
  *Safety Layers in Aligned Large Language Models: The Key to LLM Security*,
  arXiv:2408.17003. Used to identify blocks 15-19 of Qwen3-1.7B as the
  safety-discrimination band; LoRA target_modules excludes those blocks
  during mixed-SFT.
- **TIES**: Yadav et al. 2023, *TIES-Merging: Resolving Interference When
  Merging Models*. Custom implementation in `group_model/merge.py` (mergekit
  was unusable on the cluster image due to a pydantic resolver conflict).
- **Variable-k MC augmentation**: addresses the M2-flagged "2-20 option
  count" failure mode. Implementation in
  `datasets/multilingual/scripts/augment_variable_k.py`.

---

## Submission notes for M3

This tree is laid out as the M2 working repo. For the M3 submission:
- `code/` should be this tree (or a copy/subset of it).
- `final_report/` lives in a sibling directory; LaTeX source +
  compiled PDF go there.
- HF model repos stay at the same slugs (`cs-552-2026-qwenlifegivesyoulemons/*`).
- This `PROJECT_README.md` is what the grader should read first; rename
  to `README.md` in the M3 submission so it lands at the canonical path.
