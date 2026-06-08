"""Stream Wikipedia (IT/ES/ZH/RU/HI) into a plain-text corpus for CPT.

Move 2, stage 1 input. Regional MC data is scarce (confirmed: full INCLUDE
unavailable, v1/v2/v3 plateaued at ~75%). But raw encyclopedic text covering
civics / geography / culture / history / law is abundant. We continue-pretrain
Qwen3-1.7B on this so the later SFT-LoRA has knowledge to draw on.

HF dataset: wikimedia/wikipedia  (per-language configs like '20231101.it')
Schema: {id, url, title, text}

Output: one JSONL per language at <out_dir>/<lang>.jsonl, rows {"text","lang"}.
Streaming + per-language cap so we never download full Wikipedia (millions
of articles per language).

VERIFY the snapshot config exists before a long run:
    from datasets import get_dataset_config_names
    cfgs = get_dataset_config_names("wikimedia/wikipedia")
    print([c for c in cfgs if c.endswith(('.it', '.es', '.zh', '.ru', '.hi'))])
"""

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset

LANGS = ("it", "es", "zh", "ru", "hi")
MIN_CHARS = 600       # drop stubs / disambiguation / list pages
MAX_CHARS = 40_000    # truncate pathologically long articles


def stream_lang(lang, snapshot, max_articles, cache_dir):
    config = f"{snapshot}.{lang}"
    ds = load_dataset("wikimedia/wikipedia", config,
                      split="train", streaming=True, cache_dir=cache_dir)
    n = 0
    for ex in ds:
        text = (ex.get("text") or "").strip()
        if len(text) < MIN_CHARS:
            continue
        yield {"text": text[:MAX_CHARS], "lang": lang}
        n += 1
        if n >= max_articles:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True,
                    help="e.g. ../cpt_corpus")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--snapshot", default="20231101",
                    help="Wikipedia dump date prefix; verify it exists first")
    ap.add_argument("--max_articles", type=int, default=40000,
                    help="per-language article cap (40k x 5 langs x ~1k tok "
                         "~= 200M tokens, ~1 epoch is a few hours on an A100)")
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang in args.langs:
        out_path = out_dir / f"{lang}.jsonl"
        n = 0
        with out_path.open("w", encoding="utf-8") as f:
            for row in stream_lang(lang, args.snapshot, args.max_articles,
                                   args.cache_dir):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        print(f"wikipedia/{lang}: wrote {n} articles -> {out_path}")


if __name__ == "__main__":
    main()
    # HF streaming leaves a background prefetch thread mid-request when we
    # return early on the article cap. Normal interpreter shutdown then
    # segfaults in the aiohttp/threading teardown (cosmetic — all data is
    # already written and flushed by this point). Hard-exit to skip the
    # broken finalization path.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
