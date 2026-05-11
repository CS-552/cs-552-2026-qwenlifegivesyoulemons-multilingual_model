"""Mix cleaned_datasets/ -> train.jsonl + dev.jsonl.

Usage (from this scripts/ dir):
    python build_all.py

Pipeline:
    cleaned_datasets/**/*.jsonl  (intermediate, includes cs_only/ subfolder)
      -> variable-k augmentation in-process (k spans 2..20)
      -> per-language stratified split (TARGET_LANGS only; en goes to train)
      -> eval-shape jsonl

Outputs:
    ../train.jsonl
    ../dev.jsonl

Builders (build_global_mmlu.py / build_include.py / build_mmlu_en.py) are
run separately — this script only assembles what's already in
cleaned_datasets/.
"""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    LANGS, LANGS_WITH_EN,
    read_jsonl, to_train_row, write_jsonl,
)
from augment_variable_k import augment_one, build_distractor_pool


def collect_intermediate(cleaned_dir):
    """Yield every intermediate item from cleaned_dir, recursively."""
    for p in sorted(cleaned_dir.rglob("*.jsonl")):
        yield from read_jsonl(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..",
                    help="datasets/multilingual root (parent of cleaned_datasets/)")
    ap.add_argument("--dev_per_lang", type=int, default=200,
                    help="dev items per target language (English not in dev)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copies", type=int, default=2,
                    help="number of variable-k augmented copies per source item")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cleaned = root / "cleaned_datasets"
    if not cleaned.exists():
        raise SystemExit(f"missing {cleaned}; run the builders first")

    rng = random.Random(args.seed)

    items = list(collect_intermediate(cleaned))
    print(f"Loaded {len(items)} intermediate items from {cleaned}")
    print("  Lang counts:  ", dict(Counter(x["lang"] for x in items).most_common()))
    print("  Source counts:", dict(Counter(x["source"] for x in items).most_common()))

    pool = build_distractor_pool(items)

    augmented = []
    for item in items:
        for _ in range(args.copies):
            augmented.append(augment_one(item, pool, rng))
    print(f"Augmented to {len(augmented)} items (copies={args.copies})")
    k_hist = Counter(len(x["options"]) for x in augmented)
    print(f"  k distribution (min={min(k_hist)} max={max(k_hist)}):")
    for k in sorted(k_hist):
        print(f"    k={k:>2}: {k_hist[k]}")

    by_lang = defaultdict(list)
    for item in augmented:
        by_lang[item["lang"]].append(item)

    train, dev = [], []
    for lang in LANGS_WITH_EN:
        bucket = by_lang.get(lang, [])
        if not bucket:
            continue
        rng.shuffle(bucket)
        if lang in LANGS:
            dev_n = min(args.dev_per_lang, len(bucket))
            dev.extend(bucket[:dev_n])
            train.extend(bucket[dev_n:])
            print(f"  {lang}: train={len(bucket) - dev_n} dev={dev_n}")
        else:
            train.extend(bucket)
            print(f"  {lang}: train={len(bucket)} dev=0 (English; not evaluated)")

    rng.shuffle(train)
    rng.shuffle(dev)
    n_train = write_jsonl((to_train_row(x) for x in train), root / "train.jsonl")
    n_dev = write_jsonl((to_train_row(x) for x in dev), root / "dev.jsonl")
    print(f"TOTAL: train={n_train} dev={n_dev}")


if __name__ == "__main__":
    main()
