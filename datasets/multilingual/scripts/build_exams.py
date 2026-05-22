"""Build intermediate JSONL from EXAMS (native multilingual exam questions).

HF: mhardalov/exams, config 'multilingual'
Native high-school graduation / state exam questions across 16 languages.
We keep Spanish and Italian (EXAMS' coverage of our target set).

Schema (verified):
  columns: id, question, answerKey, info
  question = {'stem': str,
              'choices': {'text': [...], 'label': [...], 'para': [...]}}
  answerKey = a label string matching one entry of choices.label
  info     = {'grade': int, 'subject': str, 'language': <full English name>}
License: CC-BY-SA-4.0.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from common import make_intermediate, write_jsonl

# EXAMS tags language by full English name; map to our ISO codes.
LANG_NAME_TO_CODE = {
    "Spanish": "es",
    "Italian": "it",
}


def build(cache_dir):
    ds = load_dataset("mhardalov/exams", "multilingual", cache_dir=cache_dir)
    for split in ds.values():
        for ex in split:
            info = ex.get("info") or {}
            lang = LANG_NAME_TO_CODE.get(info.get("language"))
            if lang is None:
                continue
            q = ex.get("question") or {}
            stem = q.get("stem")
            choices = q.get("choices") or {}
            texts = choices.get("text") or []
            labels = choices.get("label") or []
            answer_key = ex.get("answerKey")
            if not stem or not (2 <= len(texts) <= 20):
                continue
            if len(texts) != len(labels) or answer_key not in labels:
                continue
            if any(t is None or str(t).strip() == "" for t in texts):
                continue
            yield make_intermediate(
                question=stem,
                options=texts,
                answer_idx=labels.index(answer_key),
                source="exams",
                lang=lang,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    by_lang = defaultdict(list)
    for row in build(args.cache_dir):
        by_lang[row["lang"]].append(row)

    for lang, items in by_lang.items():
        n = write_jsonl(items, out_dir / f"exams_{lang}.jsonl")
        print(f"exams/{lang}: wrote {n} items")


if __name__ == "__main__":
    main()
