"""Build math training data from NuminaMath-CoT for the group_model mixed-SFT.

HF: AI-MO/NuminaMath-CoT  (~860k items in train)
We sample 50k so math doesn't dominate the mix by volume — the per-domain
weighted sampler equalizes per-batch exposure anyway, so volume only affects
*diversity* of items the model sees in its 1/N domain share.

Schema (unified across all group_model/build_*.py builders):
  prompt   : str  -- what the user asks
  target   : str  -- what the assistant should produce, including any <think> block
  domain   : "math"
  source   : "NuminaMath-CoT"
  subsource: original NuminaMath sub-dataset (e.g. orca-math, math, ...)
  lang     : "en"

For math, the target wraps the chain-of-thought reasoning in <think>...</think>
and ends with \\boxed{answer}. This matches the course math eval (free-form,
pass@8, \\boxed{} extraction) and trains the model to actually reason on
free-form problems while still emitting a final boxed answer.

The chat template (enable_thinking=true at the group_model level) lets the
model learn to FILL the <think> block on math examples and LEAVE IT EMPTY
on MC examples (which the GK / multilingual builders will produce with
empty-think targets).
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

# Capture the LAST \boxed{...} in a string, tolerating one level of nested braces
# (e.g. \boxed{\frac{1}{2}}). Solutions in NuminaMath-CoT consistently end in
# such a boxed final answer.
LAST_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_final_boxed(solution):
    """Return (reasoning_before_box, final_answer) or None if no \\boxed found."""
    matches = list(LAST_BOXED_RE.finditer(solution))
    if not matches:
        return None
    last = matches[-1]
    reasoning = solution[: last.start()].rstrip()
    answer = last.group(1).strip()
    return reasoning, answer


def build(n_sample, seed, cache_dir):
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", cache_dir=cache_dir)
    print(f"  NuminaMath-CoT train: {len(ds)} items total")

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(ds)), min(n_sample, len(ds))))
    sampled = ds.select(indices)
    print(f"  sampled {len(sampled)} items (seed={seed})")

    n_kept = 0
    n_no_box = 0
    n_malformed = 0
    for ex in sampled:
        problem = ex.get("problem")
        solution = ex.get("solution")
        if not isinstance(problem, str) or not isinstance(solution, str):
            n_malformed += 1
            continue
        if not problem.strip() or not solution.strip():
            n_malformed += 1
            continue
        extracted = extract_final_boxed(solution)
        if extracted is None:
            n_no_box += 1
            continue
        reasoning, answer = extracted
        if not reasoning or not answer:
            n_malformed += 1
            continue
        target = f"<think>\n{reasoning}\n</think>\n\n\\boxed{{{answer}}}"
        yield {
            "prompt": problem.strip(),
            "target": target,
            "domain": "math",
            "source": "NuminaMath-CoT",
            "subsource": ex.get("source", "unknown"),
            "lang": "en",
        }
        n_kept += 1
    print(f"  kept {n_kept} | no-\\boxed {n_no_box} | malformed {n_malformed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="output jsonl, e.g. data/math.jsonl")
    ap.add_argument("--n_sample", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default=None,
                    help="HF datasets cache; use /scratch/hf_cache on the cluster")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in build(args.n_sample, args.seed, args.cache_dir):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"math: wrote {n} items -> {out_path}")


if __name__ == "__main__":
    main()
