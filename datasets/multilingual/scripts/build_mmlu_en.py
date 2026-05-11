"""Build intermediate JSONL from the original English MMLU.

HF dataset: https://huggingface.co/datasets/cais/mmlu
Used as anchor data to keep the model's English performance intact across
the team's group_model fusion (weight averaging especially). Capped to ~17%
of the training mix via weighted sampling at training time — NOT by
shrinking this file. Pull everything; the dataloader is the budget.

Schema (verified): question, choices (list[4]), answer (0-3 int), subject.
"""

import argparse
from pathlib import Path

from datasets import load_dataset

from common import make_intermediate, write_jsonl


def build_split(cache_dir):
    ds = load_dataset("cais/mmlu", "all", split="test", cache_dir=cache_dir)
    for ex in ds:
        choices = ex.get("choices") or []
        if len(choices) != 4 or any(c is None or str(c).strip() == "" for c in choices):
            continue
        answer = ex.get("answer")
        if not isinstance(answer, int) or not 0 <= answer < len(choices):
            continue
        yield make_intermediate(
            question=ex["question"],
            options=choices,
            answer_idx=answer,
            source="mmlu_en",
            lang="en",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    rows = list(build_split(args.cache_dir))
    n = write_jsonl(rows, Path(args.out_dir) / "mmlu_en.jsonl")
    print(f"mmlu_en: wrote {n} items")


if __name__ == "__main__":
    main()
