"""Finalize a mergekit output and push it to the team's group_model HF repo.

mergekit produces a complete Qwen3-1.7B-shaped checkpoint, but two things
still need attention before the course CI will be happy with it:

  1. Chat template: mergekit copies the tokenizer from one of the source
     models (usually the first listed). If that specialty uses a different
     chat template than what we want for the group_model — e.g., the math
     specialty might keep thinking-mode on while we want no_think for
     pass@1 speed across all 4 domains — the merged tokenizer is wrong.
     We force no_think here, consistent with the multilingual specialty's
     setup.

  2. generation_config.json: same tightened settings as the specialty
     pushes (max_new_tokens=32, eos=<|im_end|>) so the group_model doesn't
     waste budget on rambling generations.

Usage:
    python3 push_group.py \
        --merged_dir outputs/ties_v1 \
        --hf_repo    cs-552-2026-qwenlifegivesyoulemons/group_model \
        --commit_msg "group_model TIES merge (math + gk + safety + multilingual)" \
        --push

Without --push: writes the finalized files into <merged_dir> and exits.
"""

import argparse
import datetime
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

NO_THINK_OVERRIDE = "{%- set enable_thinking = false %}\n"


def force_no_think(tokenizer):
    base = tokenizer.chat_template or ""
    if NO_THINK_OVERRIDE.strip() and NO_THINK_OVERRIDE.strip() not in base:
        tokenizer.chat_template = NO_THINK_OVERRIDE + base
    return tokenizer


def build_generation_config(tokenizer):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        im_end_id = tokenizer.eos_token_id
    return {
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 32,
        "repetition_penalty": 1.0,
        "eos_token_id": im_end_id,
    }


def verify_template(tokenizer):
    sample = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False, add_generation_prompt=True,
    )
    print("[verify] chat-template output (last 300 chars):")
    print("---")
    print(sample[-300:])
    print("---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", required=True,
                    help="mergekit output dir (e.g. outputs/ties_v1)")
    ap.add_argument("--hf_repo",
                    default="cs-552-2026-qwenlifegivesyoulemons/group_model")
    ap.add_argument("--branch", default="main",
                    help="HF branch to push to (use a side branch like "
                         "'linear-baseline' for archival comparisons)")
    ap.add_argument("--commit_msg", default=None)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    merged_dir = Path(args.merged_dir).resolve()
    if not merged_dir.exists():
        raise SystemExit(f"missing {merged_dir}")

    # Verify the mergekit output has the right shape
    safetensors = list(merged_dir.glob("*.safetensors"))
    if not safetensors:
        raise SystemExit(f"no .safetensors in {merged_dir} — mergekit run may have failed")
    total_gb = sum(f.stat().st_size for f in safetensors) / 1e9
    print(f"[verify] {len(safetensors)} safetensors file(s), {total_gb:.2f} GB total")
    if not (merged_dir / "config.json").exists():
        raise SystemExit("config.json missing from merged_dir")

    # Force no_think on the tokenizer mergekit copied
    print("[fix] forcing no_think on tokenizer + saving back to merged_dir")
    tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
    force_no_think(tokenizer)
    tokenizer.save_pretrained(merged_dir)

    # Write generation_config.json (tightened: max_new_tokens=32, eos=<|im_end|>)
    gen_config = build_generation_config(tokenizer)
    (merged_dir / "generation_config.json").write_text(
        json.dumps(gen_config, indent=2), encoding="utf-8")
    print(f"[save] generation_config -> {merged_dir / 'generation_config.json'} "
          f"(eos_token_id={gen_config['eos_token_id']}, "
          f"max_new_tokens={gen_config['max_new_tokens']})")

    verify_template(tokenizer)

    # Commit message: include run name + timestamp so versions are distinguishable
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = merged_dir.name  # e.g. 'ties_v1'
    commit_msg = args.commit_msg or f"group_model merge: {run_name} @ {timestamp}"

    # Metadata file — same trick as push_to_hub.py to guarantee lastModified
    # advances on every push, so the CI re-evaluates.
    meta = {
        "push_timestamp": datetime.datetime.now().isoformat(),
        "run_name": run_name,
        "merged_dir": str(merged_dir),
        "branch": args.branch,
        "commit_message": commit_msg,
    }
    (merged_dir / ".push_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    if not args.push:
        print(f"[dry-run] finalized at {merged_dir}; re-run with --push to upload")
        return

    print(f"[push] {merged_dir} -> https://huggingface.co/{args.hf_repo} "
          f"(branch: {args.branch})")
    print(f"[push] commit message: {commit_msg!r}")
    if not os.environ.get("HF_TOKEN"):
        print("[warn] HF_TOKEN not set; relying on cached `huggingface-cli login` creds")

    from huggingface_hub import create_repo, create_branch, upload_folder
    create_repo(args.hf_repo, exist_ok=True, repo_type="model")
    if args.branch != "main":
        try:
            create_branch(args.hf_repo, branch=args.branch, repo_type="model")
        except Exception as e:
            print(f"[branch] already exists or skipped: {e}")
    upload_folder(
        folder_path=str(merged_dir),
        repo_id=args.hf_repo,
        revision=args.branch,
        commit_message=commit_msg,
    )
    print(f"[push] done — branch '{args.branch}' should appear in the next "
          f"nightly CI run (CI evaluates main).")


if __name__ == "__main__":
    main()
