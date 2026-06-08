"""Build intermediate JSONL from MILU (native Indian-exam MC knowledge, Hindi).

HF: ai4bharat/MILU  (GATED — accept the access agreement on the dataset page
and be HF-authenticated, e.g. `hf auth login`, before running.)

India-centric exam questions across 41 subjects / 8 domains, drawn from
regional and state-level Indian exams. The strongest native Hindi source by
far (Kaleidoscope-hi alone is only ~900 items) — closes the Hindi gap.

Schema (verified, config 'Hindi'):
  splits : validation, test  (~15.6k items combined)
  columns: question, option1..option4, target, is_translated, language,
           domain, subject
  target : a string like 'option3' naming the correct option.

We keep ALL items (both is_translated True/False). MILU is India-centric
throughout — even its translated items carry Indian-exam content, unlike
Global-MMLU bulk which is generic English MMLU. License: CC-BY-4.0.
"""

import argparse
from pathlib import Path

from datasets import load_dataset

from common import make_intermediate, write_jsonl


def build(cache_dir):
    ds = load_dataset("ai4bharat/MILU", "Hindi", cache_dir=cache_dir)
    for split in ds.values():
        for ex in split:
            options = [ex.get("option1"), ex.get("option2"),
                       ex.get("option3"), ex.get("option4")]
            if any(o is None or str(o).strip() == "" for o in options):
                continue
            # target is e.g. 'option3' -> 0-based index 2
            try:
                answer_idx = int(str(ex.get("target")).replace("option", "").strip()) - 1
            except (ValueError, AttributeError, TypeError):
                continue
            if not 0 <= answer_idx < len(options):
                continue
            question = ex.get("question")
            if not question or not str(question).strip():
                continue
            yield make_intermediate(
                question=question,
                options=options,
                answer_idx=answer_idx,
                source="milu",
                lang="hi",
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    rows = list(build(args.cache_dir))
    n = write_jsonl(rows, Path(args.out_dir) / "milu_hi.jsonl")
    print(f"milu/hi: wrote {n} items")


if __name__ == "__main__":
    main()
