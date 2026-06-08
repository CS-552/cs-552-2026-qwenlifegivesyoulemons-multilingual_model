# Multilingual training data

Pipeline for the `multilingual_model` specialty checkpoint.
Languages: Italian (`it`), Spanish (`es`), Chinese (`zh`), Russian (`ru`), Hindi (`hi`).

## Layout

```
multilingual/
├── raw/              # HF dataset cache (gitignored, several GB)
├── _intermediate/    # per-source intermediate JSONL (one file per source+lang)
├── processed/        # variable-k-augmented intermediate JSONL
├── train.jsonl       # final eval-shape SFT data
├── dev.jsonl         # held-out, per-language stratified
└── scripts/
    ├── common.py             schema, letter/option helpers, JSONL IO
    ├── build_mmmlu.py        openai/MMMLU
    ├── build_global_mmlu.py  CohereForAI/Global-MMLU
    ├── build_include.py      CohereForAI/include-base-44
    ├── augment_variable_k.py 2..20 option-count resampler
    └── build_all.py          orchestrator
```

## Schemas

**Intermediate** (`_intermediate/`, `processed/`):

```json
{"question": "...", "options": ["...", "..."], "answer_idx": 0,
 "source": "mmmlu", "lang": "it"}
```

**Eval-shape** (`train.jsonl`, `dev.jsonl`, matches `validation_samples/multilingual.jsonl`):

```json
{"prompt": "...\n\nA) ...\nB) ...", "answer": "B"}
```

## Run

Full pipeline:

```bash
cd scripts
python build_all.py --out_dir ..
```

Iterate on one source while debugging:

```bash
python build_mmmlu.py --out_dir ../_intermediate
```

Skip the download/build stage if `_intermediate/` is already populated:

```bash
python build_all.py --out_dir .. --skip_build
```

## Notes

- The `build_global_mmlu.py` and `build_include.py` builders are stubs:
  verify the live HF dataset schemas before relying on them.
- `augment_variable_k.py` is the highest-leverage piece — the course README
  flags 4-only training as a top failure mode. Don't skip it.
