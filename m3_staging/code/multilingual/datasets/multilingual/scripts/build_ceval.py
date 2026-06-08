"""Build intermediate JSONL from C-Eval (Chinese exam benchmark).

HF: ceval/ceval-exam
Native Chinese professional / civics / academic exams across 52 subjects
(accountant, civil_servant, law, legal_professional, tax_accountant,
teacher_qualification, fire_engineer, ...). NOT translated MMLU — genuine
China-specific content, the kind the eval's "regional / professional
licensing" axis tests.

Schema (verified): columns id, question, A, B, C, D, answer, explanation.
answer is a letter A-D. Splits: test, val, dev (this HF mirror carries
answers in all three; rows with a missing/invalid answer are skipped).
License: CC-BY-NC-SA-4.0 (non-commercial — fine for coursework).

All 52 subjects are kept: the STEM ones overlap with Global-MMLU but still
give the model Chinese-language MC practice; the mix weighting in
build_all.py controls overall balance.
"""

import argparse
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

from common import make_intermediate, write_jsonl

ANSWER_LETTERS = ("A", "B", "C", "D")


def build(cache_dir):
    configs = get_dataset_config_names("ceval/ceval-exam")
    for config in configs:
        ds = load_dataset("ceval/ceval-exam", config, cache_dir=cache_dir)
        for split in ds.values():
            for ex in split:
                answer = ex.get("answer")
                if answer not in ANSWER_LETTERS:
                    continue
                options = [ex.get("A"), ex.get("B"), ex.get("C"), ex.get("D")]
                if any(o is None or str(o).strip() == "" for o in options):
                    continue
                yield make_intermediate(
                    question=ex["question"],
                    options=options,
                    answer_idx=ANSWER_LETTERS.index(answer),
                    source="ceval",
                    lang="zh",
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    rows = list(build(args.cache_dir))
    n = write_jsonl(rows, Path(args.out_dir) / "ceval_zh.jsonl")
    print(f"ceval/zh: wrote {n} items")


if __name__ == "__main__":
    main()
