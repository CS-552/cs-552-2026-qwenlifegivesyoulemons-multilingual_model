"""Build multilingual training data for the group_model mixed-SFT.

Reuses the v1-era multilingual sources from
`datasets/multilingual/cleaned_datasets/`:
  - global_mmlu_{it,es,zh,ru,hi}.jsonl    (translated MMLU, 5 langs)
  - include_{it,es,zh,ru,hi}.jsonl        (regional INCLUDE-base-44)
  - mmlu_en.jsonl                         (English anchor)
  - cs_only/global_mmlu_*.jsonl           (cultural-sensitive subset)

EXPLICITLY EXCLUDES v5 native-harvest additions (ceval, cmmlu, kaleidoscope,
exams, headqa, milu) — user's call: simpler, proven v1 baseline, and v5's
additions did not improve the standalone score.

Applies variable-k augmentation (k in [2, 20]) identical to v1's pipeline,
so the model stays robust to the eval's variable option-count requirement.
This is a multilingual-specific requirement (the course README flags
4-only training as a top failure mode).

Output schema (unified, matches build_math.py / build_gk.py):
  prompt   : str  -- question + lettered options
  target   : str  -- '<think>\\n\\n</think>\\n\\n\\\\boxed{LETTER}'
  domain   : 'multilingual'
  source   : 'global_mmlu' | 'include' | 'mmlu_en'
  lang     : 'it' | 'es' | 'zh' | 'ru' | 'hi' | 'en'
"""

import argparse
import json
import random
import string
from collections import Counter, defaultdict
from pathlib import Path

LETTERS = string.ascii_uppercase  # A..T covers the 2..20 option range
MIN_OPTIONS = 2
MAX_OPTIONS = 20

# v1-era source files in cleaned_datasets/ (flat) — anything not matching is v5.
V1_PREFIXES = ("global_mmlu_", "include_", "mmlu_en")


def is_v1_top_level(filename):
    return (
        filename.endswith(".jsonl")
        and any(filename.startswith(p) for p in V1_PREFIXES)
    )


def collect_v1_items(cleaned_dir):
    """Yield intermediate items from v1 sources only.

    Picks up cleaned_datasets/{global_mmlu_*, include_*, mmlu_en}.jsonl plus
    cleaned_datasets/cs_only/global_mmlu_*.jsonl (the cultural subset).
    Skips v5 additions (ceval, cmmlu, kaleidoscope, exams, headqa, milu).
    """
    cleaned = Path(cleaned_dir)
    for p in sorted(cleaned.glob("*.jsonl")):
        if not is_v1_top_level(p.name):
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item["_is_cs"] = False
                yield item
    cs_dir = cleaned / "cs_only"
    if cs_dir.exists():
        for p in sorted(cs_dir.glob("global_mmlu_*.jsonl")):
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    item["_is_cs"] = True
                    yield item


def build_distractor_pool(items):
    """Deduplicated distractor pool keyed by (source, lang)."""
    pool = defaultdict(set)
    for item in items:
        key = (item["source"], item["lang"])
        for i, opt in enumerate(item["options"]):
            if i != item["answer_idx"]:
                pool[key].add(opt)
    return {k: list(v) for k, v in pool.items()}


def augment_one(item, pool, rng):
    """Variable-k augmentation: produce one item with k in [2, 20] options."""
    options = item["options"]
    answer_idx = item["answer_idx"]
    gold = options[answer_idx]

    own = []
    seen = {gold}
    for i, opt in enumerate(options):
        if i != answer_idx and opt not in seen:
            own.append(opt)
            seen.add(opt)

    k = rng.randint(MIN_OPTIONS, MAX_OPTIONS)
    needed = k - 1
    chosen = own[:needed]
    seen = {gold, *chosen}

    if len(chosen) < needed:
        bag = pool.get((item["source"], item["lang"]), [])
        oversample = min(len(bag), (needed - len(chosen)) + 5)
        if oversample > 0:
            for d in rng.sample(bag, oversample):
                if d in seen:
                    continue
                chosen.append(d)
                seen.add(d)
                if len(chosen) >= needed:
                    break
    if len(chosen) < needed:
        k = len(chosen) + 1

    out_options = list(chosen)
    gold_pos = rng.randrange(k)
    out_options.insert(gold_pos, gold)
    return {**item, "options": out_options, "answer_idx": gold_pos}


def format_mc_prompt(question, options):
    lines = [str(question).strip(), ""]
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}) {str(opt).strip()}")
    return "\n".join(lines)


def to_unified_row(item):
    return {
        "prompt": format_mc_prompt(item["question"], item["options"]),
        "target": f"<think>\n\n</think>\n\n\\boxed{{{LETTERS[item['answer_idx']]}}}",
        "domain": "multilingual",
        "source": item["source"],
        "lang": item["lang"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned_dir", required=True,
                    help="path to datasets/multilingual/cleaned_datasets/")
    ap.add_argument("--out", required=True,
                    help="output jsonl, e.g. data/multilingual.jsonl")
    ap.add_argument("--copies", type=int, default=2,
                    help="variable-k augmented copies per source item "
                         "(v1 default = 2; per-domain sampler equalizes at train time)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    items = list(collect_v1_items(args.cleaned_dir))
    print(f"Loaded {len(items)} v1 source items from {args.cleaned_dir}")
    print(f"  lang counts:   {dict(Counter(x['lang'] for x in items).most_common())}")
    print(f"  source counts: {dict(Counter(x['source'] for x in items).most_common())}")
    print(f"  CS items:      {sum(1 for x in items if x.get('_is_cs'))}")

    pool = build_distractor_pool(items)

    augmented = []
    for item in items:
        for _ in range(args.copies):
            augmented.append(augment_one(item, pool, rng))
    print(f"Augmented: {len(items)} src -> {len(augmented)} aug "
          f"(copies={args.copies})")
    k_hist = Counter(len(x["options"]) for x in augmented)
    print(f"  k distribution range: {min(k_hist)}..{max(k_hist)} "
          f"(flatness check: avg per-k = {sum(k_hist.values()) // len(k_hist)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng.shuffle(augmented)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in augmented:
            row = to_unified_row(item)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"multilingual: wrote {n} items -> {out_path}")


if __name__ == "__main__":
    main()
