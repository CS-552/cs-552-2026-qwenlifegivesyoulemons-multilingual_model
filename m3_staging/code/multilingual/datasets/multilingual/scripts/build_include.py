"""Build intermediate JSONL from CohereForAI's INCLUDE-base-44.

HF dataset: https://huggingface.co/datasets/CohereForAI/include-base-44
Per-language regional exams (incl. professional-licensing material) -> the
best match for the course's "professional licensing" axis.

Verified schema:
    Configs: per-language, full English names ("Italian", "Russian", ...).
    Splits: 'test' and 'validation' (we combine both — neither is the
            course eval set, so no leakage risk).
    Columns: language, country, domain, subject, regional_feature, level,
             question, option_a, option_b, option_c, option_d, answer
    answer:  0-based integer index in [0, 3].

Volume is small (~500 items per language); the value-add is regional /
professional-licensing content, not raw count.
"""

import argparse
from pathlib import Path

from datasets import load_dataset

from common import LANGS, make_intermediate, write_jsonl

LANG_TO_CONFIG = {
    "it": "Italian",
    "es": "Spanish",
    "zh": "Chinese",
    "ru": "Russian",
    "hi": "Hindi",
}


def build_split(lang, cache_dir):
    ds = load_dataset("CohereForAI/include-base-44", LANG_TO_CONFIG[lang],
                      cache_dir=cache_dir)
    for split in ds.values():
        for ex in split:
            answer = ex.get("answer")
            if not isinstance(answer, int) or not 0 <= answer <= 3:
                continue
            options = [ex.get("option_a"), ex.get("option_b"),
                       ex.get("option_c"), ex.get("option_d")]
            if any(o is None or str(o).strip() == "" for o in options):
                continue
            yield make_intermediate(
                question=ex["question"],
                options=options,
                answer_idx=answer,
                source="include",
                lang=lang,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for lang in args.langs:
        rows = list(build_split(lang, args.cache_dir))
        n = write_jsonl(rows, out_dir / f"include_{lang}.jsonl")
        print(f"include/{lang}: wrote {n} items")


if __name__ == "__main__":
    main()
