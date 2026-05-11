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
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent

DEFAULT_GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 50,
    "max_new_tokens": 256,
    "repetition_penalty": 1.0,
}


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
    ap.add_argument("--commit_msg", default="LoRA SFT checkpoint")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir).resolve()
    merged_dir = Path(args.merged_dir or adapter_dir.parent / "merged").resolve()
    merged_dir.mkdir(parents=True, exist_ok=True)

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

    gen_config_path = merged_dir / "generation_config.json"
    with gen_config_path.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_GENERATION_CONFIG, f, indent=2)
    print(f"[save] generation_config -> {gen_config_path}")

    verify_template(tokenizer)

    if not args.push:
        print(f"[dry-run] merged checkpoint ready at {merged_dir}")
        print("         re-run with --push to upload to HF")
        return

    print(f"[push] {merged_dir} -> https://huggingface.co/{args.hf_repo}")
    if not os.environ.get("HF_TOKEN"):
        print("[warn] HF_TOKEN not set; relying on cached `huggingface-cli login` creds")

    from huggingface_hub import create_repo, upload_folder
    create_repo(args.hf_repo, exist_ok=True, repo_type="model")
    upload_folder(
        folder_path=str(merged_dir),
        repo_id=args.hf_repo,
        commit_message=args.commit_msg,
    )
    print(f"[push] done — {args.hf_repo} should appear in the next nightly CI run")


if __name__ == "__main__":
    main()
