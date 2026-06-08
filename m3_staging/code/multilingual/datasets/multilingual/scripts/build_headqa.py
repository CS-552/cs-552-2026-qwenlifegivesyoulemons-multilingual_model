"""Build intermediate JSONL from HEAD-QA (Spanish healthcare licensing exams).

HF: dvilares/head_qa
Ships a loader script (head_qa.py) that modern `datasets` refuses to run.
We bypass it: download data/head-qa-es-en-pdfs.zip and parse the Spanish
JSON (HEAD/HEAD.json) directly.

JSON schema (verified):
  {"version", "language": "es",
   "exams": {<exam_name>: {"name", "data": [question, ...]}}}
  question = {"qid", "qtext",
              "ra": "<correct answer id, as a string>",
              "answers": [{"aid": <int>, "atext": str}, ...],
              "image": "<empty string when text-only>"}

The correct option is the answer whose `aid` equals `ra` (matched by id,
not list position, to be robust to ordering).

Native Spanish official healthcare professional exams (medicine, nursing,
pharmacy, psychology, biology, chemistry). License: MIT.
"""

import argparse
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from common import make_intermediate, write_jsonl

ZIP_NAME = "data/head-qa-es-en-pdfs.zip"
SPANISH_JSON = "HEAD/HEAD.json"


def build(cache_dir):
    zip_path = hf_hub_download("dvilares/head_qa", ZIP_NAME,
                               repo_type="dataset", cache_dir=cache_dir)
    with zipfile.ZipFile(zip_path) as z:
        with z.open(SPANISH_JSON) as f:
            data = json.load(f)

    for exam in data.get("exams", {}).values():
        for q in exam.get("data", []):
            if q.get("image"):
                continue  # skip image-dependent questions
            answers = q.get("answers") or []
            options = [a.get("atext") for a in answers]
            if not (2 <= len(options) <= 20):
                continue
            if any(o is None or str(o).strip() == "" for o in options):
                continue
            try:
                ra = int(q["ra"])
            except (KeyError, ValueError, TypeError):
                continue
            # Correct option = the one whose aid == ra (match by id).
            answer_idx = next(
                (i for i, a in enumerate(answers) if a.get("aid") == ra), None)
            if answer_idx is None:
                continue
            question = q.get("qtext")
            if not question or not str(question).strip():
                continue
            yield make_intermediate(
                question=question,
                options=options,
                answer_idx=answer_idx,
                source="headqa",
                lang="es",
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    rows = list(build(args.cache_dir))
    n = write_jsonl(rows, Path(args.out_dir) / "headqa_es.jsonl")
    print(f"headqa/es: wrote {n} items")


if __name__ == "__main__":
    main()
