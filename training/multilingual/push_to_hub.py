"""Merge LoRA adapter into Qwen3-1.7B and push a vLLM-loadable checkpoint to HF.

Output checkpoint structure (matches the course CI contract):
    config.json
    model.safetensors[.index.json + shards]
    generation_config.json
    tokenizer.json / tokenizer_config.json / chat_template.jinja
    (anything else the tokenizer saves alongside)

Usage:
    python push_to_hub.py \
        --adapter_dir /scratch/.../outputs/lora_v1/final \
        --hf_repo cs-552-2026-<org>/multilingual_model \
        --push

Without --push the script does a local merge + save only (dry-run);
inspect the merged_dir before re-running with --push.

Auth: set HF_TOKEN in the environment, or `huggingface-cli login` first.
"""

import argparse
import datetime
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent

def build_generation_config(tokenizer, greedy=False):
    """Build generation_config.json contents, resolving stop tokens against the
    tokenizer at push time (so we don't hardcode model-version-specific IDs).

    With --greedy: do_sample=False, no temperature/top_p/top_k. Useful for
    testing whether pass@1 leaks points to sampling variance — at temp=0.2
    we still occasionally sample a non-most-likely letter, which can hurt
    when the most-likely answer is the right one.
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        im_end_id = tokenizer.eos_token_id
    cfg = {
        "max_new_tokens": 32,
        "repetition_penalty": 1.0,
        "eos_token_id": im_end_id,
    }
    if greedy:
        cfg["do_sample"] = False
    else:
        cfg.update({
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 50,
        })
    return cfg


def force_no_think(tokenizer):
    override = (HERE / "chat_template.jinja").read_text(encoding="utf-8")
    base = tokenizer.chat_template or ""
    if override.strip() and override.strip() not in base:
        tokenizer.chat_template = override + base
    return tokenizer


def verify_template(tokenizer):
    sample = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False, add_generation_prompt=True,
    )
    print("[verify] chat-template output (last 300 chars):")
    print("---")
    print(sample[-300:])
    print("---")
    if "<think>" in sample and "</think>" not in sample[sample.find("<think>"):]:
        print("[verify] WARNING: open <think> with no closing tag — no_think likely not applied")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True, help="LoRA final/ dir from train_lora.py")
    ap.add_argument("--base_model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--merged_dir", default=None,
                    help="local merged dir (default: <adapter_dir>/../merged)")
    ap.add_argument("--hf_repo", required=True,
                    help="e.g. cs-552-2026-<org>/multilingual_model")
    ap.add_argument("--push", action="store_true",
                    help="actually push to HF (otherwise local merge only)")
    ap.add_argument("--commit_msg", default=None,
                    help="commit message prefix. If omitted, auto-generates "
                         "'LoRA SFT: <run_name> @ <timestamp>' so v1/v2/v3 "
                         "are visually distinguishable on HF's commit history.")
    ap.add_argument("--greedy", action="store_true",
                    help="write a greedy generation_config (do_sample=False, "
                         "no temp/top_p/top_k). Use to test whether sampling "
                         "variance is leaking pass@1 points at temp=0.2.")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir).resolve()
    merged_dir = Path(args.merged_dir or adapter_dir.parent / "merged").resolve()
    merged_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = adapter_dir.parent.name  # e.g. 'lora_v2' from .../lora_v2/final
    commit_msg = args.commit_msg or f"LoRA SFT: {run_name} @ {timestamp}"

    print(f"[load] tokenizer from {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    force_no_think(tokenizer)

    print(f"[load] base {args.base_model} (bf16)")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    print(f"[load] adapter {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)

    print("[merge] merging LoRA into base weights")
    model = model.merge_and_unload()

    print(f"[save] merged model -> {merged_dir}")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    gen_config = build_generation_config(tokenizer, greedy=args.greedy)
    gen_config_path = merged_dir / "generation_config.json"
    with gen_config_path.open("w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=2)
    print(f"[save] generation_config -> {gen_config_path} "
          f"(do_sample={gen_config.get('do_sample', True)}, "
          f"eos_token_id={gen_config['eos_token_id']}, "
          f"max_new_tokens={gen_config['max_new_tokens']})")

    # Write a metadata file with the push timestamp. This guarantees the file
    # tree changes between pushes even when model weights happen to hash
    # identically, forcing HF's lastModified to advance and the course CI to
    # pick up the new revision.
    metadata = {
        "push_timestamp": datetime.datetime.now().isoformat(),
        "run_name": run_name,
        "adapter_dir": str(adapter_dir),
        "commit_message": commit_msg,
    }
    meta_path = merged_dir / ".push_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[save] push metadata -> {meta_path}")

    verify_template(tokenizer)

    if not args.push:
        print(f"[dry-run] merged checkpoint ready at {merged_dir}")
        print("         re-run with --push to upload to HF")
        return

    print(f"[push] {merged_dir} -> https://huggingface.co/{args.hf_repo}")
    print(f"[push] commit message: {commit_msg!r}")
    if not os.environ.get("HF_TOKEN"):
        print("[warn] HF_TOKEN not set; relying on cached `huggingface-cli login` creds")

    from huggingface_hub import create_repo, upload_folder
    create_repo(args.hf_repo, exist_ok=True, repo_type="model")
    upload_folder(
        folder_path=str(merged_dir),
        repo_id=args.hf_repo,
        commit_message=commit_msg,
    )
    print(f"[push] done — {args.hf_repo} should appear in the next nightly CI run")


if __name__ == "__main__":
    main()
