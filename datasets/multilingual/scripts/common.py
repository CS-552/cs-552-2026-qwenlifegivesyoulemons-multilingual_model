"""Shared helpers for multilingual MC dataset builders.

Pipeline:
    raw HF datasets
      -> per-source builder scripts produce *intermediate* JSONL:
          {"question": str, "options": [str, ...], "answer_idx": int,
           "source": str, "lang": str}
      -> augment_variable_k.py rewrites option counts to span 2..20
      -> build_all.py converts intermediate -> eval-shape, splits train/dev:
          {"prompt": str, "answer": str}     # answer is a letter A..T
"""

import json
import string
import unicodedata
from pathlib import Path

MAX_OPTIONS = 20
LETTERS = string.ascii_uppercase[:MAX_OPTIONS]
LANGS = ("it", "es", "zh", "ru", "hi")  # target eval languages
LANGS_WITH_EN = LANGS + ("en",)         # plus English used as anchor data


def letter(idx):
    if not 0 <= idx < MAX_OPTIONS:
        raise ValueError(f"option index {idx} out of range [0, {MAX_OPTIONS})")
    return LETTERS[idx]


def normalize(text):
    return unicodedata.normalize("NFKC", str(text)).strip()


def make_intermediate(question, options, answer_idx, source, lang):
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"expected 2..{MAX_OPTIONS} options, got {len(options)}")
    if not 0 <= answer_idx < len(options):
        raise ValueError(f"answer_idx {answer_idx} not in [0, {len(options)})")
    if lang not in LANGS_WITH_EN:
        raise ValueError(f"lang must be one of {LANGS_WITH_EN}, got {lang!r}")
    return {
        "question": normalize(question),
        "options": [normalize(o) for o in options],
        "answer_idx": int(answer_idx),
        "source": source,
        "lang": lang,
    }


def to_eval_row(item):
    """Intermediate -> eval-shape row (matches validation_samples schema)."""
    lines = [item["question"], ""]
    for i, opt in enumerate(item["options"]):
        lines.append(f"{letter(i)}) {opt}")
    return {"prompt": "\n".join(lines), "answer": letter(item["answer_idx"])}


def to_train_row(item):
    """Eval-shape + lang/source metadata for training-time sampling.
    Train/dev files are local-only — not pushed to HF — so extra keys are fine.
    """
    return {**to_eval_row(item), "lang": item["lang"], "source": item["source"]}


def validate_eval_row(row):
    if set(row.keys()) != {"prompt", "answer"}:
        raise ValueError(f"row has wrong keys: {sorted(row.keys())}")
    if not isinstance(row["prompt"], str) or not row["prompt"].strip():
        raise ValueError("prompt must be a non-empty string")
    if row["answer"] not in LETTERS:
        raise ValueError(f"answer must be in A..T, got {row['answer']!r}")


def write_jsonl(rows, out_path, validate=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if validate is not None:
                validate(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
