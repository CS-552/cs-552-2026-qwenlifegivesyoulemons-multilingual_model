"""Mix cleaned_datasets/ -> train.jsonl + dev.jsonl.

Usage (from this scripts/ dir):
    python build_all.py                # default cs_weight=3
    python build_all.py --cs_weight 5  # heavier CS upsampling

Pipeline:
    cleaned_datasets/**/*.jsonl     (intermediate; cs_only/ subfolder tagged)
      -> per-language stratified split BY SOURCE ITEM   (no train<->dev leakage)
      -> variable-k augmentation:
           - bulk items:  --copies augmented variants per source
           - CS items:    --copies * --cs_weight variants per source
      -> dev: one augmented copy per held-out source item
      -> eval-shape JSONL

Why split by source item, not by augmented item?
    Each source MC question yields N augmented copies (different k values).
    If we shuffle then split the augmented pool, two copies of the same
    question can land one in train and one in dev — direct leakage. Splitting
    source items first guarantees train and dev are disjoint at the question
    level.

Why CS upsampling?
    The course eval explicitly tests "regional knowledge, civics, culture,
    professional licensing". The CS (cultural-sensitive) subset of Global-MMLU
    is hand-annotated for exactly that axis. Bumping its weight in training
    points the model at the eval distribution.

Outputs:
    ../train.jsonl
    ../dev.jsonl
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
    """Yield intermediate items, tagging cs_only/ files with _is_cs=True."""
    for p in sorted(cleaned_dir.rglob("*.jsonl")):
        is_cs = "cs_only" in p.parts
        for item in read_jsonl(p):
            yield {**item, "_is_cs": is_cs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..",
                    help="datasets/multilingual root (parent of cleaned_datasets/)")
    ap.add_argument("--dev_per_lang", type=int, default=200,
                    help="dev source items per target language (English not in dev)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copies", type=int, default=2,
                    help="variable-k augmented copies per train source item (bulk)")
    ap.add_argument("--cs_weight", type=int, default=3,
                    help="multiplier on --copies for CS items only. "
                         "cs_weight=3 means CS source items produce 3x the "
                         "augmented variants of bulk source items.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cleaned = root / "cleaned_datasets"
    if not cleaned.exists():
        raise SystemExit(f"missing {cleaned}; run the builders first")

    rng = random.Random(args.seed)

    items = list(collect_intermediate(cleaned))
    n_cs = sum(1 for x in items if x.get("_is_cs"))
    print(f"Loaded {len(items)} intermediate items from {cleaned}")
    print(f"  Lang counts:   {dict(Counter(x['lang'] for x in items).most_common())}")
    print(f"  Source counts: {dict(Counter(x['source'] for x in items).most_common())}")
    print(f"  CS items:      {n_cs} ({n_cs/len(items):.1%} of pool)")

    # Build distractor pool from ALL items (richer pool than train-only).
    pool = build_distractor_pool(items)

    # Split source items per-language BEFORE augmentation (prevents leakage).
    by_lang = defaultdict(list)
    for item in items:
        by_lang[item["lang"]].append(item)

    train_src, dev_src = [], []
    for lang in LANGS_WITH_EN:
        bucket = by_lang.get(lang, [])
        if not bucket:
            continue
        rng.shuffle(bucket)
        if lang in LANGS:
            dev_n = min(args.dev_per_lang, len(bucket))
            dev_src.extend(bucket[:dev_n])
            train_src.extend(bucket[dev_n:])
            print(f"  {lang}: train_src={len(bucket) - dev_n} dev_src={dev_n}")
        else:
            train_src.extend(bucket)
            print(f"  {lang}: train_src={len(bucket)} dev_src=0 (English; not evaluated)")

    # Augment train. CS items get cs_weight extra copies.
    train_aug = []
    cs_train_src = 0
    for item in train_src:
        if item.get("_is_cs"):
            n_copies = args.copies * args.cs_weight
            cs_train_src += 1
        else:
            n_copies = args.copies
        for _ in range(n_copies):
            train_aug.append(augment_one(item, pool, rng))
    print(f"Train augmented: {len(train_src)} src -> {len(train_aug)} aug "
          f"(cs_src={cs_train_src} x {args.copies * args.cs_weight} copies)")

    # Augment dev: exactly one copy per held-out source item.
    dev_aug = [augment_one(item, pool, rng) for item in dev_src]
    print(f"Dev augmented:   {len(dev_src)} src -> {len(dev_aug)} aug")

    k_hist = Counter(len(x["options"]) for x in train_aug)
    print(f"  Train k distribution (min={min(k_hist)} max={max(k_hist)}):")
    for k in sorted(k_hist):
        print(f"    k={k:>2}: {k_hist[k]}")

    rng.shuffle(train_aug)
    rng.shuffle(dev_aug)
    n_train = write_jsonl((to_train_row(x) for x in train_aug), root / "train.jsonl")
    n_dev = write_jsonl((to_train_row(x) for x in dev_aug), root / "dev.jsonl")
    print(f"TOTAL: train={n_train} dev={n_dev}")


if __name__ == "__main__":
    main()
