"""Build intermediate JSONL from Cohere's Global-MMLU.

HF dataset: https://huggingface.co/datasets/CohereForAI/Global-MMLU
Useful column: `cultural_sensitivity_label` ('CA' = cultural_agnostic,
'CS' = cultural_sensitive). The CS subset is the one closest to the
"regional knowledge / civics / culture" axis the course eval cares about.

VERIFY before first run: config codes and column names by listing configs:
    from datasets import get_dataset_config_names
    print(get_dataset_config_names("CohereForAI/Global-MMLU"))

This is currently a STUB: confirm the schema, then fill in `build_split`.
"""

import argparse
from pathlib import Path

from datasets import load_dataset

from common import LANGS, make_intermediate, write_jsonl

LANG_TO_CONFIG = {
    "it": "it",
    "es": "es",
    "zh": "zh",
    "ru": "ru",
    "hi": "hi",
}

ANSWER_LETTERS = ("A", "B", "C", "D")


def build_split(lang, cache_dir, cs_only=False):
    ds = load_dataset("CohereForAI/Global-MMLU", LANG_TO_CONFIG[lang],
                      split="test", cache_dir=cache_dir)
    for ex in ds:
        if cs_only and ex.get("cultural_sensitivity_label") != "CS":
            continue
        answer = ex.get("answer")
        if answer not in ANSWER_LETTERS:
            continue
        options = [ex.get("option_a"), ex.get("option_b"),
                   ex.get("option_c"), ex.get("option_d")]
        if any(o is None or str(o).strip() == "" for o in options):
            continue
        yield make_intermediate(
            question=ex["question"],
            options=options,
            answer_idx=ANSWER_LETTERS.index(answer),
            source="global_mmlu",
            lang=lang,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--cs_only", action="store_true",
                    help="keep only culturally-sensitive items")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for lang in args.langs:
        rows = list(build_split(lang, args.cache_dir, args.cs_only))
        n = write_jsonl(rows, out_dir / f"global_mmlu_{lang}.jsonl")
        print(f"global_mmlu/{lang}: wrote {n} items")


if __name__ == "__main__":
    main()
