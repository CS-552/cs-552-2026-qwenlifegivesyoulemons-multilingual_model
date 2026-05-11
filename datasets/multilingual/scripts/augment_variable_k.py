"""Resample MC items so option counts span 2..20.

The course README flags 4-only training as a top failure mode: models trained
on only 4-way MC collapse on long-tail items. We keep the gold option, then
fill k-1 distractor slots from the item's own distractors first, then from a
same-(source, lang) pool to keep them in-distribution.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from common import MAX_OPTIONS, read_jsonl, write_jsonl


def build_distractor_pool(items):
    """Deduplicated distractor pool keyed by (source, lang).

    Dedup matters: many MC datasets repeat boilerplate options ('True', 'False',
    'None of the above', etc.). Keeping duplicates inflates the pool 10-100x
    without adding variety, and breaks random.sample's uniqueness guarantee
    in augment_one.
    """
    pool = defaultdict(set)
    for item in items:
        key = (item["source"], item["lang"])
        for i, opt in enumerate(item["options"]):
            if i != item["answer_idx"]:
                pool[key].add(opt)
    return {k: list(v) for k, v in pool.items()}


def sample_k(rng):
    # Flat distribution over 2..MAX_OPTIONS so every option count gets coverage.
    return rng.randint(2, MAX_OPTIONS)


def augment_one(item, pool, rng, target_k=None):
    options = item["options"]
    answer_idx = item["answer_idx"]
    gold = options[answer_idx]

    # Own distractors, deduplicated against gold and each other.
    own = []
    seen = {gold}
    for i, opt in enumerate(options):
        if i != answer_idx and opt not in seen:
            own.append(opt)
            seen.add(opt)

    k = target_k if target_k is not None else sample_k(rng)
    needed = k - 1

    chosen = own[:needed]
    seen = {gold, *chosen}

    if len(chosen) < needed:
        bag = pool.get((item["source"], item["lang"]), [])
        # Oversample by a small slack to absorb dedup collisions in one shot,
        # avoiding a full shuffle of the (potentially huge) pool.
        oversample = min(len(bag), (needed - len(chosen)) + 5)
        if oversample > 0:
            for d in rng.sample(bag, oversample):
                if d in seen:
                    continue
                chosen.append(d)
                seen.add(d)
                if len(chosen) >= needed:
                    break

    # Shrink k if pool + own_distractors couldn't fill the request.
    if len(chosen) < needed:
        k = len(chosen) + 1

    out_options = list(chosen)
    gold_pos = rng.randrange(k)
    out_options.insert(gold_pos, gold)

    return {**item, "options": out_options, "answer_idx": gold_pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="dir of intermediate *.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copies", type=int, default=1,
                    help="how many augmented copies per source item")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    rng = random.Random(args.seed)

    all_items = []
    for p in sorted(in_dir.glob("*.jsonl")):
        all_items.extend(list(read_jsonl(p)))
    pool = build_distractor_pool(all_items)

    for p in sorted(in_dir.glob("*.jsonl")):
        items = list(read_jsonl(p))
        out_items = []
        for item in items:
            for _ in range(args.copies):
                out_items.append(augment_one(item, pool, rng))
        write_jsonl(out_items, out_dir / p.name)
        print(f"{p.name}: {len(items)} -> {len(out_items)} items")


if __name__ == "__main__":
    main()
