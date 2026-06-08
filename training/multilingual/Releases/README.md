# Multilingual Specialty Releases

End-to-end reproducibility index for the multilingual specialist. Each
subdirectory captures one version that was trained, evaluated, and either
kept or rejected. Given a release folder, anyone should be able to rebuild
the exact dataset, re-run the exact training command, and push the exact
checkpoint that produced the reported score.

## Versions

| Version  | Data + recipe                                                                 | Multilingual pass@1     | Status                |
|----------|-------------------------------------------------------------------------------|-------------------------|-----------------------|
| v1       | Global-MMLU + INCLUDE + MMLU-en CS-only; LoRA r=64, α=128, copies=2           | 0.74                    | superseded            |
| v2       | v1 + CS upsampling (`--cs_weight 5`)                                          | 0.75                    | superseded            |
| v3       | v2 + native MCQ (CEval, CMMLU, Kaleidoscope, EXAMS, HEAD-QA, MILU)            | 0.75                    | superseded by v5      |
| v4_cpt   | v3 base + full-FT continued pretraining on regional Wikipedia (5 langs)       | 0.69                    | regression, rejected  |
| v5       | v3 data + bilko-style chat template override (no_think + MC system message)   | 0.75 / 0.69 (see below) | current               |

### v5 score split — CI is resolving to the wrong branch
The 2026-06-05 EVAL_REPORT PR was scored against the v5 commit (`da69392`)
on `main` and returned **0.75**. The 2026-06-08 EVAL_REPORT PR returned
**0.69**, but inspection of the PR shows the CI pulled from the **v3
branch**, not `main`. Our v5 files (commit `da69392` on `main`) were never
actually re-evaluated. The 0.69 is whatever v3's files score under current
CI rules — it is not a measurement of v5.

**Open item**: check HF's default-branch setting and any tag/ref called
`latest` on the `multilingual_model` repo; the CI's `repo_info` call is
resolving to a stale revision. A `huggingface_hub` push that explicitly
sets `revision="main"` on `upload_folder` (or a default-branch reset on
the HF side) should fix it.

## Baseline
Base Qwen3-1.7B on the same multilingual dev set:
```bash
cd group_model
python3 eval_base.py --dev_file ../training/multilingual/data/dev.jsonl
```
The multilingual pass@1 from this command is the comparison anchor for
the "+18-24pp gain" reported in the abstract.

## Reproducing a release
1. `cd` into the version's folder.
2. Read `README.md` for the exact data recipe, hyperparameters, and decoding config.
3. Follow `run.sh` (cluster command) — pre-pinned with the version's args.

## Conventions
- All training is LoRA on Qwen3-1.7B in bf16 with gradient checkpointing.
- All scores are CI pass@1 on the hidden multilingual test set unless otherwise noted.
- All pushes write a `.push_metadata.json` so the HF `lastModified` advances
  even when only the chat template changed (required for the CI to pick up
  template-only diffs).
