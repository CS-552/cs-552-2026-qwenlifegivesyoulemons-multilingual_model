"""Combine math + gk + multilingual jsonls into the final group_model
train/dev splits for mixed-SFT.

Inputs (must exist before running):
  data/math.jsonl          (from build_math.py)
  data/gk.jsonl            (from build_gk.py)
  data/multilingual.jsonl  (from build_multilingual_v1.py)

Outputs:
  data/train.jsonl  -- all domains, shuffled
  data/dev.jsonl    -- N items per domain held out (for eval_loss tracking)

Per-domain balancing happens at TRAIN TIME (train_mixed.py reads each row's
`domain` field and uses a WeightedRandomSampler so each domain gets ~equal
probability per batch, regardless of pool size). This script just stitches
the three pre-built jsonls together and carves out a dev split.

The unified row schema (set by each per-domain builder):
  prompt   : str
  target   : str  -- filled <think> for math, empty <think> for MC (gk/multilingual)
  domain   : 'math' | 'gk' | 'multilingual'
  source   : str  (per-dataset)
  lang     : str  ('en' for math/gk; one of it/es/zh/ru/hi/en for multilingual)
  [subsource: str  optional metadata, kept as-is]
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_FILES = ("math.jsonl", "gk.jsonl", "multilingual.jsonl")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data",
                    help="dir containing per-domain jsonls; train/dev written here too")
    ap.add_argument("--dev_per_domain", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)

    by_domain = defaultdict(list)
    for name in SOURCE_FILES:
        p = data_dir / name
        if not p.exists():
            raise SystemExit(f"missing {p} — run the corresponding build_*.py first")
        for row in read_jsonl(p):
            by_domain[row["domain"]].append(row)

    print("Loaded per-domain items:")
    for d in sorted(by_domain):
        print(f"  {d:14s}: {len(by_domain[d]):>7d}")

    train, dev = [], []
    for domain in sorted(by_domain):
        rows = by_domain[domain]
        rng.shuffle(rows)
        n_dev = min(args.dev_per_domain, len(rows) // 4)  # cap dev at 25% of domain
        dev.extend(rows[:n_dev])
        train.extend(rows[n_dev:])
        print(f"  {domain:14s}: train={len(rows) - n_dev}, dev={n_dev}")

    rng.shuffle(train)
    rng.shuffle(dev)

    n_train = write_jsonl(train, data_dir / "train.jsonl")
    n_dev = write_jsonl(dev, data_dir / "dev.jsonl")
    print(f"TOTAL: train={n_train} dev={n_dev}")

    # Per-domain breakdown in final train mix
    train_domains = Counter(r["domain"] for r in train)
    print(f"\nTrain domain breakdown: {dict(train_domains)}")

    # Multilingual lang breakdown
    mlt_langs = Counter(r.get("lang", "?") for r in train if r["domain"] == "multilingual")
    if mlt_langs:
        print(f"Multilingual lang breakdown: {dict(mlt_langs.most_common())}")

    # Source breakdown
    train_sources = Counter(r.get("source", "?") for r in train)
    print(f"Train source breakdown: {dict(train_sources.most_common())}")


if __name__ == "__main__":
    main()
