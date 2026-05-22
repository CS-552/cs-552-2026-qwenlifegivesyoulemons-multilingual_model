"""Build intermediate JSONL from CMMLU (Chinese-native MC knowledge).

HF: haonan-li/cmmlu
The repo ships a loader script (cmmlu.py) that modern `datasets` refuses to
run, plus the data archive `cmmlu_v1_0_1.zip`. We bypass the script entirely:
download the zip via huggingface_hub and parse its CSVs directly.

CMMLU CSV schema: header row with columns Question, A, B, C, D, Answer
(Answer is a letter A-D; the leading unnamed column is a row index).

Native China-specific content (civics, law, driving rules, history, ...)
across 67 subjects. License: CC-BY-NC-4.0 (non-commercial — fine for
coursework).
"""

import argparse
import csv
import io
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from common import make_intermediate, write_jsonl

ANSWER_LETTERS = ("A", "B", "C", "D")
ZIP_NAME = "cmmlu_v1_0_1.zip"


def build(cache_dir):
    zip_path = hf_hub_download("haonan-li/cmmlu", ZIP_NAME,
                               repo_type="dataset", cache_dir=cache_dir)
    with zipfile.ZipFile(zip_path) as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        print(f"  cmmlu: {len(csv_names)} CSV files in {ZIP_NAME}")
        for name in csv_names:
            with z.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                for row in reader:
                    answer = (row.get("Answer") or "").strip().upper()
                    if answer not in ANSWER_LETTERS:
                        continue
                    options = [row.get("A"), row.get("B"),
                               row.get("C"), row.get("D")]
                    if any(o is None or str(o).strip() == "" for o in options):
                        continue
                    question = row.get("Question")
                    if not question or not str(question).strip():
                        continue
                    yield make_intermediate(
                        question=question,
                        options=options,
                        answer_idx=ANSWER_LETTERS.index(answer),
                        source="cmmlu",
                        lang="zh",
                    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    rows = list(build(args.cache_dir))
    n = write_jsonl(rows, Path(args.out_dir) / "cmmlu_zh.jsonl")
    print(f"cmmlu/zh: wrote {n} items")
    if n == 0:
        print("  WARNING: 0 items — CSV column names may differ; "
              "inspect a CSV header inside cmmlu_v1_0_1.zip")


if __name__ == "__main__":
    main()
