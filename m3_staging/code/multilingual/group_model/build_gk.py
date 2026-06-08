"""Build GK training data for the group_model mixed-SFT (sft2/grpo2 recipe).

Three sources, all 4-10 option MC, all reformatted with empty-think targets:
  - MMLU-Pro     (TIGER-Lab/MMLU-Pro)       : ~12k test items, 4-10 options
  - GPQA Diamond (Idavidrein/gpqa)          : ~198 items, 4 options, GATED
  - MedMCQA      (openlifescienceai/medmcqa): sample 10k from train, 4 options

Output schema (unified, matches build_math.py):
  prompt   : str  -- formatted question + lettered options
  target   : str  -- '<think>\\n\\n</think>\\n\\n\\\\boxed{LETTER}'
  domain   : 'gk'
  source   : 'mmlu_pro' | 'gpqa_diamond' | 'medmcqa'
  subsource: per-dataset metadata (MMLU-Pro category, GPQA domain, etc.)
  lang     : 'en'

EMPTY-think target is deliberate: MC questions teach the model to emit
\\boxed{LETTER} directly, with no reasoning chain. Math builder uses FILLED
think blocks. The model learns to switch based on prompt shape (lettered
options vs free-form problem).

PREREQ: GPQA Diamond is GATED. Before running:
  1. Accept terms at https://huggingface.co/datasets/Idavidrein/gpqa
  2. Be HF-authenticated in the shell (`hf auth login`).
"""

import argparse
import json
import random
import string
from pathlib import Path

from datasets import load_dataset

LETTERS = string.ascii_uppercase  # A..Z; MC options here cap at 10


def format_mc_prompt(question, options):
    lines = [question.strip(), ""]
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}) {str(opt).strip()}")
    return "\n".join(lines)


def make_row(question, options, answer_idx, source, **extras):
    return {
        "prompt": format_mc_prompt(question, options),
        "target": f"<think>\n\n</think>\n\n\\boxed{{{LETTERS[answer_idx]}}}",
        "domain": "gk",
        "source": source,
        "lang": "en",
        **extras,
    }


def build_mmlu_pro(cache_dir):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", cache_dir=cache_dir)
    print(f"  MMLU-Pro test: {len(ds)} items")
    n_kept = 0
    for ex in ds:
        options = ex.get("options") or []
        if not (2 <= len(options) <= 20):
            continue
        if any(o is None or str(o).strip() == "" for o in options):
            continue
        ai = ex.get("answer_index")
        if not isinstance(ai, int) or not 0 <= ai < len(options):
            continue
        question = ex.get("question")
        if not question or not str(question).strip():
            continue
        yield make_row(
            question, options, ai, "mmlu_pro",
            subsource=ex.get("category", "unknown"),
        )
        n_kept += 1
    print(f"  mmlu_pro: kept {n_kept}")


def build_gpqa_diamond(cache_dir, seed):
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", cache_dir=cache_dir)
    sp = list(ds.keys())[0]
    print(f"  GPQA Diamond {sp}: {len(ds[sp])} items")
    rng = random.Random(seed)
    n_kept = 0
    for ex in ds[sp]:
        correct = ex.get("Correct Answer")
        incorrects = [
            ex.get("Incorrect Answer 1"),
            ex.get("Incorrect Answer 2"),
            ex.get("Incorrect Answer 3"),
        ]
        if not correct or any(x is None or str(x).strip() == "" for x in incorrects):
            continue
        options = [correct] + incorrects
        rng.shuffle(options)
        answer_idx = options.index(correct)
        question = ex.get("Question") or ex.get("question")
        if not question or not str(question).strip():
            continue
        yield make_row(
            question, options, answer_idx, "gpqa_diamond",
            subsource=ex.get("High-level domain", "unknown"),
        )
        n_kept += 1
    print(f"  gpqa_diamond: kept {n_kept}")


def build_medmcqa(cache_dir, n_sample, seed):
    ds = load_dataset("openlifescienceai/medmcqa", split="train", cache_dir=cache_dir)
    print(f"  MedMCQA train: {len(ds)} items")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(ds)), min(n_sample, len(ds))))
    sampled = ds.select(indices)
    print(f"  sampled {len(sampled)} (seed={seed})")
    n_kept = 0
    for ex in sampled:
        options = [ex.get("opa"), ex.get("opb"), ex.get("opc"), ex.get("opd")]
        if any(o is None or str(o).strip() == "" for o in options):
            continue
        cop = ex.get("cop")
        # MedMCQA's cop is 0-3 (0-indexed) on most public mirrors; some legacy
        # versions are 1-indexed. Handle both defensively.
        if not isinstance(cop, int):
            continue
        if 0 <= cop <= 3:
            answer_idx = cop
        elif 1 <= cop <= 4:
            answer_idx = cop - 1
        else:
            continue
        question = ex.get("question")
        if not question or not str(question).strip():
            continue
        yield make_row(
            question, options, answer_idx, "medmcqa",
            subsource=ex.get("subject_name", "unknown"),
        )
        n_kept += 1
    print(f"  medmcqa: kept {n_kept}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="output jsonl, e.g. data/gk.jsonl")
    ap.add_argument("--cache_dir", default=None,
                    help="HF cache (use /scratch/hf_cache on the cluster)")
    ap.add_argument("--medmcqa_sample", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in build_mmlu_pro(args.cache_dir):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
        for row in build_gpqa_diamond(args.cache_dir, args.seed):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
        for row in build_medmcqa(args.cache_dir, args.medmcqa_sample, args.seed):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"gk: wrote {n} items -> {out_path}")


if __name__ == "__main__":
    main()
