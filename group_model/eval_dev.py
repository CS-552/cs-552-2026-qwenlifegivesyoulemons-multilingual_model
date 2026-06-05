"""Local per-domain (and per-language) pass@1 on dev.jsonl.

The course CI only exposes one aggregate score per domain. This script
re-creates the same metric locally on data/dev.jsonl, plus a multilingual
per-language breakdown the CI hides. Useful for the report's results table
and for spot-checking checkpoints without burning a CI cycle (24h round-trip).

Usage:
    python3 eval_dev.py \\
        --model_dir outputs/mixed_v1/merged \\
        --dev_file  data/dev.jsonl
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same regex the course CI uses for \boxed{...} extraction (handles 1 level of nesting)
BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_boxed(text):
    # Take the LAST \boxed{...} — for math the answer trails the reasoning
    matches = list(BOXED_RE.finditer(text))
    return matches[-1].group(1).strip() if matches else None


def normalize(s):
    # Light normalization for comparing extracted vs gold
    return None if s is None else str(s).strip().lower().replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="merged model dir")
    ap.add_argument("--dev_file", default="data/dev.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--do_sample", action="store_true",
                    help="match course CI (sampling at temp 0.2); off by default for reproducibility here")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all dev items")
    args = ap.parse_args()

    # Load model + tokenizer
    print(f"[load] model: {args.model_dir}")
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load dev set
    rows = [json.loads(l) for l in open(args.dev_file, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[data] {len(rows)} dev items")

    # Extract the gold answer for each row. Two dev schemas in this repo:
    #  - group_model dev:  {"prompt", "target", "domain", "lang", ...}  -> gold is the \boxed{X} in target
    #  - multilingual dev: {"prompt", "answer", "lang", "source", ...}  -> gold is the answer field directly
    for r in rows:
        if "target" in r:
            r["gold"] = extract_boxed(r["target"])
        else:
            r["gold"] = str(r.get("answer", "")).strip() or None
        # If no domain in row (multilingual dev), tag everything as "multilingual"
        r.setdefault("domain", "multilingual")

    # Generation loop (simple batching; for big runs use vLLM)
    results = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for r in batch
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else 1.0,
                pad_token_id=tok.pad_token_id,
            )
        # Decode only the newly generated tokens
        for j, r in enumerate(batch):
            in_len = inputs.input_ids[j].ne(tok.pad_token_id).sum().item()
            gen = tok.decode(out[j][in_len:], skip_special_tokens=False)
            predicted = extract_boxed(gen)
            r["pred"] = predicted
            r["correct"] = (
                predicted is not None
                and normalize(predicted) == normalize(r["gold"])
            )
            results.append(r)
        print(f"  scored {len(results)}/{len(rows)}", end="\r", flush=True)
    print()

    # Aggregate per domain (and per-language inside multilingual)
    per_domain = defaultdict(lambda: {"n": 0, "correct": 0})
    per_lang = defaultdict(lambda: {"n": 0, "correct": 0})
    fmt_ok = 0  # how many produced ANY \boxed{...}
    for r in results:
        per_domain[r["domain"]]["n"] += 1
        per_domain[r["domain"]]["correct"] += int(r["correct"])
        if r["domain"] == "multilingual":
            lg = r.get("lang", "?")
            per_lang[lg]["n"] += 1
            per_lang[lg]["correct"] += int(r["correct"])
        if r["pred"] is not None:
            fmt_ok += 1

    # Pretty print
    print("\n=== Per-domain pass@1 ===")
    total_n, total_c = 0, 0
    for d in sorted(per_domain):
        n, c = per_domain[d]["n"], per_domain[d]["correct"]
        total_n += n
        total_c += c
        print(f"  {d:14s}  {c:>4d}/{n:<4d} = {c/n:.4f}")
    print(f"  {'OVERALL':14s}  {total_c:>4d}/{total_n:<4d} = {total_c/total_n:.4f}")

    if per_lang:
        print("\n=== Multilingual per-language pass@1 ===")
        for lg in sorted(per_lang):
            n, c = per_lang[lg]["n"], per_lang[lg]["correct"]
            print(f"  {lg:6s}  {c:>4d}/{n:<4d} = {c/n:.4f}")

    print(f"\n=== Format compliance ===")
    print(f"  {fmt_ok}/{len(results)} generations contain at least one \\boxed{{...}} "
          f"({fmt_ok/len(results):.2%})")


if __name__ == "__main__":
    main()
