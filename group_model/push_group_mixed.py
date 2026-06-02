"""Merge the group_model mixed-SFT LoRA adapter into Qwen3-1.7B and push to HF.

Differs from training/multilingual/push_to_hub.py in TWO load-bearing ways:

1. **NO custom chat template applied.** The multilingual specialty uses the
   bilko-style MC-classifier override (default system_message + user
   instruction suffix + enable_thinking=false). For the group_model that
   would BREAK math: with thinking forced off, the model gets an empty
   <think></think> in its prompt and is expected to generate the answer
   directly — no room to reason. We keep Qwen3's default chat template,
   whose enable_thinking defaults to TRUE when the CI calls
   apply_chat_template without that kwarg (per the README, the CI doesn't
   pass it).

2. **max_new_tokens bumped to 2048** so math reasoning chains fit. MC items
   naturally emit ~10 tokens (empty think + boxed letter) and ignore the
   rest of the budget.

Usage:
    python3 push_group_mixed.py \\
        --adapter_dir /scratch/.../outputs/mixed_v1/final \\
        --hf_repo cs-552-2026-qwenlifegivesyoulemons/group_model \\
        --push
"""

import argparse
import datetime
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_generation_config(tokenizer, greedy=False, max_new_tokens=2048):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        im_end_id = tokenizer.eos_token_id
    cfg = {
        "max_new_tokens": max_new_tokens,
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


def verify_template(tokenizer):
    """Render a sample prompt; thinking-on means NO empty <think></think> in prefix."""
    sample = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False, add_generation_prompt=True,
    )
    print("[verify] sample chat-template render (last 300 chars):")
    print("---")
    print(sample[-300:])
    print("---")
    if "<think>\n\n</think>" in sample[-200:]:
        print("[verify] WARNING: empty <think></think> injected into the prefix — "
              "thinking is OFF, which would block math reasoning. Check that the "
              "tokenizer's chat_template hasn't been overridden with a no_think variant.")
    else:
        print("[verify] thinking-on: no <think></think> injected in prefix. "
              "Model emits think tags itself based on prompt shape.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True,
                    help="group_model LoRA final/ dir from train_mixed.py")
    ap.add_argument("--base_model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--merged_dir", default=None,
                    help="local merged dir (default: <adapter_dir>/../merged)")
    ap.add_argument("--hf_repo",
                    default="cs-552-2026-qwenlifegivesyoulemons/group_model")
    ap.add_argument("--push", action="store_true",
                    help="actually push to HF (otherwise local merge only)")
    ap.add_argument("--commit_msg", default=None)
    ap.add_argument("--greedy", action="store_true",
                    help="greedy decoding (NOT recommended; for the multilingual "
                         "specialty it cost 5pp, and math also benefits from sampling "
                         "diversity in its reasoning chains)")
    ap.add_argument("--max_new_tokens", type=int, default=2048,
                    help="big enough for math reasoning chains; MC ignores most")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir).resolve()
    merged_dir = Path(args.merged_dir or adapter_dir.parent / "merged").resolve()
    merged_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = adapter_dir.parent.name
    commit_msg = args.commit_msg or f"group_mixed_sft: {run_name} @ {timestamp}"

    print(f"[load] tokenizer from {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    # CRITICAL: do NOT modify chat_template here. Qwen3's default template
    # defaults to enable_thinking=true when no kwarg is passed (which is the
    # CI contract per the course README). Modifying it would break math.

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

    gen_config = build_generation_config(
        tokenizer, greedy=args.greedy, max_new_tokens=args.max_new_tokens)
    gen_config_path = merged_dir / "generation_config.json"
    gen_config_path.write_text(
        json.dumps(gen_config, indent=2), encoding="utf-8")
    print(f"[save] generation_config -> {gen_config_path} "
          f"(do_sample={gen_config.get('do_sample', True)}, "
          f"max_new_tokens={gen_config['max_new_tokens']}, "
          f"eos_token_id={gen_config['eos_token_id']})")

    metadata = {
        "push_timestamp": datetime.datetime.now().isoformat(),
        "run_name": run_name,
        "adapter_dir": str(adapter_dir),
        "approach": "mixed_sft+safety_band_frozen+default_qwen3_template",
        "commit_message": commit_msg,
    }
    (merged_dir / ".push_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    verify_template(tokenizer)

    if not args.push:
        print(f"[dry-run] merged checkpoint ready at {merged_dir}")
        print("         re-run with --push to upload to HF")
        return

    print(f"[push] {merged_dir} -> https://huggingface.co/{args.hf_repo}")
    print(f"[push] commit message: {commit_msg!r}")
    if not os.environ.get("HF_TOKEN"):
        print("[warn] HF_TOKEN not set; relying on cached `hf auth login` creds")

    from huggingface_hub import create_repo, upload_folder
    create_repo(args.hf_repo, exist_ok=True, repo_type="model")
    upload_folder(
        folder_path=str(merged_dir),
        repo_id=args.hf_repo,
        commit_message=commit_msg,
    )
    print(f"[push] done — CI evaluates on the next nightly run")


if __name__ == "__main__":
    main()
