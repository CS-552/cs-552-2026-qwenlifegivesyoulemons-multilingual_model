"""Evaluate the un-fine-tuned Qwen3-1.7B baseline on our local dev set.

Why this script exists:
  Every "improvement" we claim (multilingual +18-24pp, group-model 0.525 vs
  TIES 0.295) is measured AGAINST the base model's score on the SAME dev
  set. The course CI doesn't expose a base-model number directly, so we
  compute it locally with the same metric (pass@1 over \\boxed{...}
  extraction) and the same generation defaults.

This is a thin sibling of eval_dev.py:
  - hard-codes the base model id (no merged adapter to point at)
  - uses Qwen3-1.7B's native chat template UNMODIFIED (no bilko override),
    so the baseline measures the truly out-of-the-box model
  - keeps the same boxed regex / normalization as eval_dev.py so the
    baseline number is comparable to every fine-tuned eval we run

Usage (from group_model/):
    python3 eval_base.py --dev_file data/dev.jsonl
    python3 eval_base.py --dev_file ../training/multilingual/data/dev.jsonl
"""

import argparse
import json
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Locked course base — do NOT change this. The grading contract requires the
# architecture to stay at Qwen3-1.7B; the baseline must match that exactly.
BASE_MODEL = "Qwen/Qwen3-1.7B"

# Same \boxed{...} regex used by eval_dev.py and the course CI; one level
# of brace nesting (e.g. \boxed{\frac{1}{2}}) is supported.
BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")

# Soft-extraction patterns for the base model, which doesn't naturally emit
# \boxed{...}. Tried in order; the LAST positional match across all patterns
# wins (the answer usually comes after the reasoning). Used only when
# --soft_match is passed.
SOFT_PATTERNS = [
    re.compile(r"(?:final\s+)?answer\s*(?:is)?\s*:?\s*\**\s*\(?([A-T])\)?\b", re.IGNORECASE),
    re.compile(r"the\s+correct\s+(?:answer|option|choice)\s+is\s*\**\s*\(?([A-T])\)?\b", re.IGNORECASE),
    re.compile(r"\b(?:option|choice)\s+\(?([A-T])\)?\b", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*\**\s*\(?([A-T])\)?\s*\**\s*$", re.MULTILINE),
    re.compile(r"\b([A-T])\b"),  # lowest confidence: any isolated capital letter
]


def extract_boxed(text):
    # Take the LAST match — for math chains the answer comes after the reasoning.
    matches = list(BOXED_RE.finditer(text))
    return matches[-1].group(1).strip() if matches else None


def soft_extract(text):
    # Strict boxed wins if present
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    # Otherwise try soft patterns; return the LAST positional match
    # (Answer is usually emitted after the reasoning chain)
    candidates = []
    for pat in SOFT_PATTERNS:
        for m in pat.finditer(text):
            candidates.append((m.start(), m.group(1).upper()))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def normalize(s):
    # Light normalization for comparing extracted vs gold (strip + lower + de-space).
    return None if s is None else str(s).strip().lower().replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_file", default="data/dev.jsonl",
                    help="path to dev jsonl; works with group_model or "
                         "multilingual dev schemas (auto-detected)")
    ap.add_argument("--base_model", default=BASE_MODEL,
                    help="HF model id; default is the locked course base.")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--do_sample", action="store_true",
                    help="match CI sampling (temp 0.2); off by default here "
                         "for deterministic baseline numbers.")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all dev items")
    ap.add_argument("--seed", type=int, default=42,
                    help="seeds torch + cuda RNG so do_sample=True is reproducible")
    ap.add_argument("--soft_match", action="store_true",
                    help="fall back to letter-pattern extraction when no \\boxed{} "
                         "is produced. Required for the base model, which wasn't "
                         "trained on the boxed-output contract — without this every "
                         "prediction is None and pass@1 is artificially 0%%.")
    args = ap.parse_args()
    extractor = soft_extract if args.soft_match else extract_boxed
    print(f"[extract] using {'soft_extract (boxed -> answer-is -> letter)' if args.soft_match else 'strict extract_boxed'}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Direct HF load — no PEFT, no merge, no template override. This is the
    # exact state of the model BEFORE any post-training we did.
    print(f"[load] base model: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    rows = [json.loads(l) for l in open(args.dev_file, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[data] {len(rows)} dev items from {args.dev_file}")

    # Two dev schemas in this repo (auto-detected, same as eval_dev.py):
    #  - group_model dev:  {"prompt", "target", "domain", "lang"}  -> gold is the \boxed in target
    #  - multilingual dev: {"prompt", "answer", "lang", "source"} -> gold is the answer field
    for r in rows:
        # Gold extraction always uses STRICT boxed parsing — the gold targets
        # are authored with \boxed{} and we don't want soft matching to confuse
        # the ground truth.
        if "target" in r:
            r["gold"] = extract_boxed(r["target"])
        else:
            r["gold"] = str(r.get("answer", "")).strip() or None
        # multilingual dev has no domain field; tag the whole set as multilingual.
        r.setdefault("domain", "multilingual")

    results = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        # Base Qwen3-1.7B ships its OWN chat template (defaults to enable_thinking=true
        # when add_generation_prompt=True and no kwarg is passed). We keep that
        # unchanged here so the baseline is the genuine out-of-the-box model.
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
        # Decode only the newly-generated tokens (skip the prompt portion).
        for j, r in enumerate(batch):
            in_len = inputs.input_ids[j].ne(tok.pad_token_id).sum().item()
            gen = tok.decode(out[j][in_len:], skip_special_tokens=False)
            predicted = extractor(gen)  # extract_boxed or soft_extract per --soft_match
            r["pred"] = predicted
            r["correct"] = (
                predicted is not None
                and normalize(predicted) == normalize(r["gold"])
            )
            results.append(r)
        print(f"  scored {len(results)}/{len(rows)}", end="\r", flush=True)
    print()

    # Aggregate per domain (+ per language inside the multilingual domain).
    per_domain = defaultdict(lambda: {"n": 0, "correct": 0})
    per_lang = defaultdict(lambda: {"n": 0, "correct": 0})
    fmt_ok = 0  # how many generations contained ANY \boxed{...} — format compliance
    for r in results:
        per_domain[r["domain"]]["n"] += 1
        per_domain[r["domain"]]["correct"] += int(r["correct"])
        if r["domain"] == "multilingual":
            lg = r.get("lang", "?")
            per_lang[lg]["n"] += 1
            per_lang[lg]["correct"] += int(r["correct"])
        if r["pred"] is not None:
            fmt_ok += 1

    print(f"\n=== Base Qwen3-1.7B pass@1 on {args.dev_file} ===")
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

    # Format compliance is a useful sanity check: if the base model rarely
    # emits \boxed{...} that's a separate failure mode from "wrong answer".
    print(f"\n=== Format compliance ===")
    print(f"  {fmt_ok}/{len(results)} generations contain at least one \\boxed{{...}} "
          f"({fmt_ok/len(results):.2%})")


if __name__ == "__main__":
    main()
