"""Build intermediate JSONL from Kaleidoscope (native national exam papers).

HF: manzar/kaleidoscope-bench-text-only-subset
Native exam questions from real national exams across many languages. We
pull the es / hi / ru configs — Kaleidoscope's coverage of our targets
(it has no it or zh config; C-Eval/CMMLU cover zh, EXAMS covers it).

Schema (verified): columns include question, options (list), answer (int
index, 0-based), language, country, plus image_* fields. This is the
text-only subset, but we still defensively skip any row referencing an
image. License: Apache-2.0.

Per-language yield is small (Kaleidoscope is a few hundred items per lang)
but it is genuine native regional content — valuable for ru/hi, our
thinnest languages.
"""

import argparse
from pathlib import Path

from datasets import load_dataset

from common import make_intermediate, write_jsonl

LANG_CONFIGS = ("es", "hi", "ru")


def build_lang(lang, cache_dir):
    ds = load_dataset("manzar/kaleidoscope-bench-text-only-subset", lang,
                      cache_dir=cache_dir)
    for split in ds.values():
        for ex in split:
            # Defensive: skip any image-dependent row even in the text subset.
            if ex.get("image") is not None or ex.get("image_png") is not None:
                continue
            options = ex.get("options") or []
            if not (2 <= len(options) <= 20):
                continue
            if any(o is None or str(o).strip() == "" for o in options):
                continue
            answer = ex.get("answer")
            if not isinstance(answer, int) or not 0 <= answer < len(options):
                continue
            yield make_intermediate(
                question=ex["question"],
                options=options,
                answer_idx=answer,
                source="kaleidoscope",
                lang=lang,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--langs", nargs="+", default=list(LANG_CONFIGS))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    for lang in args.langs:
        rows = list(build_lang(lang, args.cache_dir))
        n = write_jsonl(rows, out_dir / f"kaleidoscope_{lang}.jsonl")
        print(f"kaleidoscope/{lang}: wrote {n} items")


if __name__ == "__main__":
    main()
