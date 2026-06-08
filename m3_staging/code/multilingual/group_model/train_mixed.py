"""Mixed-SFT LoRA training for the group_model on math + GK + multilingual.

Key design choices baked in:

1. PER-DOMAIN WEIGHTED SAMPLING. With ~46k math, ~22k gk, ~182k multilingual
   items, multilingual would naturally dominate by ~8x. WeightedRandomSampler
   assigns weight = 1/N_domain to each row so every batch is ~1/3 from each
   domain regardless of pool size.

2. SAFETY-BAND FROZEN. Per the SPPFT doc, blocks 15-19 of Qwen3-1.7B carry
   the safety-discrimination signal. LoRA target_modules EXCLUDES those
   blocks so they're frozen during training, preserving base safety behavior
   while the other 23 blocks absorb math/GK/multilingual.

3. QWEN3 DEFAULT CHAT TEMPLATE. We do NOT apply a custom template (unlike
   the multilingual specialty). Qwen3's default template enables thinking
   when the caller doesn't pass enable_thinking — which is exactly what the
   CI does. The model self-selects: filled <think> for math (per training
   targets), empty <think> for MC (per training targets).

4. MIXED-FORMAT TARGETS (set by the per-domain builders):
     math:           '<think>\\n{cot}\\n</think>\\n\\n\\boxed{answer}'
     gk/multilingual: '<think>\\n\\n</think>\\n\\n\\boxed{letter}'

5. Same LoRA hyperparams as v5 multilingual (r=64, alpha=128, attn+MLP).
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

HERE = Path(__file__).parent
DEFAULT_BASE = "Qwen/Qwen3-1.7B"

# Safety band — these 5 blocks get LEFT ALONE during training (see SPPFT doc)
SAFETY_BLOCKS = {15, 16, 17, 18, 19}
N_LAYERS = 28  # Qwen3-1.7B layer count

ATTN_PROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_PROJ = ["gate_proj", "up_proj", "down_proj"]


def build_target_modules():
    # Build a list of every projection module in every layer EXCEPT 15-19
    # so LoRA never touches the safety-discrimination layers
    targets = []
    for i in range(N_LAYERS):
        if i in SAFETY_BLOCKS:
            continue  # skip safety band
        for p in ATTN_PROJ:
            targets.append(f"model.layers.{i}.self_attn.{p}")
        for p in MLP_PROJ:
            targets.append(f"model.layers.{i}.mlp.{p}")
    return targets


def load_jsonl(path):
    # Read jsonl into a list (we have ~250k rows — fits in memory comfortably)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_example(row, tokenizer, max_length):
    # Tokenize one (prompt, target) pair and build the loss mask
    # The trick: tokenize WITHOUT the assistant content to get the prefix length,
    # then mask labels[:prefix_len] = -100 so loss only counts the target
    messages = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["target"]},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    prefix_text = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True,
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]

    # Truncate to max_length (some NuminaMath solutions are very long — accepted)
    full_ids = full_ids[:max_length]
    prefix_len = min(len(prefix_ids), len(full_ids))
    labels = [-100] * prefix_len + full_ids[prefix_len:]  # -100 = ignored by CE loss
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def build_domain_weights(rows):
    # Per-row sampler weight = 1/N_domain so each domain ends up ~equal per batch
    # (multilingual has 8x the items but we don't want it to dominate training)
    counts = Counter(r.get("domain", "?") for r in rows)
    return [1.0 / counts[r.get("domain", "?")] for r in rows]


class DomainBalancedTrainer(Trainer):
    """HF Trainer with a per-domain WeightedRandomSampler."""

    def __init__(self, *args, sampler_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampler_weights = sampler_weights

    def _get_train_sampler(self, *args, **kwargs):
        if self._sampler_weights is None:
            return super()._get_train_sampler(*args, **kwargs)
        generator = torch.Generator().manual_seed(self.args.seed)
        return WeightedRandomSampler(
            weights=self._sampler_weights,
            num_samples=len(self._sampler_weights),
            replacement=True,
            generator=generator,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=DEFAULT_BASE)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--dev_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_length", type=int, default=2048,
                    help="some NuminaMath solutions exceed this; truncation "
                         "is acceptable — the model still learns the format")
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", default="group_mixed_v1")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"[load] tokenizer {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # IMPORTANT: do NOT touch chat_template. We want Qwen3's default which
    # defaults enable_thinking=true when the CI calls apply_chat_template
    # without that kwarg (per the course README contract).

    # Load base model in bf16 — gradient checkpointing on, KV cache off (we'd recompute anyway during training)
    print(f"[load] base {args.base_model} (bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    # Print model config — sanity check we got the right Qwen3-1.7B
    cfg = model.config
    total = sum(p.numel() for p in model.parameters())
    print(f"[load] model_type={cfg.model_type} layers={cfg.num_hidden_layers} "
          f"params={total:,}")

    # Build the LoRA target list with safety blocks excluded
    target_modules = build_target_modules()
    expected = (N_LAYERS - len(SAFETY_BLOCKS)) * (len(ATTN_PROJ) + len(MLP_PROJ))
    print(f"[lora] {len(target_modules)} target modules (expected {expected}); "
          f"safety blocks {sorted(SAFETY_BLOCKS)} excluded")

    # Attach LoRA adapters — only the listed target_modules get trainable params
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # ~3.2% trainable (vs ~3.9% if we trained all layers)

    # Load + report data
    train_rows = load_jsonl(args.train_file)
    dev_rows = load_jsonl(args.dev_file)
    print(f"[data] train={len(train_rows)} dev={len(dev_rows)}")
    print(f"[data] train domains: {dict(Counter(r['domain'] for r in train_rows))}")
    print(f"[data] dev   domains: {dict(Counter(r['domain'] for r in dev_rows))}")

    # Tokenize everything up front (HF datasets caches the result on disk so
    # re-runs are fast). ~7-8 min for 250k items, dominated by the multilingual rows.
    train_ds = Dataset.from_list(train_rows).map(
        lambda r: encode_example(r, tokenizer, args.max_length),
        remove_columns=Dataset.from_list(train_rows).column_names,
        desc="tokenize train",
    )
    dev_ds = Dataset.from_list(dev_rows).map(
        lambda r: encode_example(r, tokenizer, args.max_length),
        remove_columns=Dataset.from_list(dev_rows).column_names,
        desc="tokenize dev",
    )

    # Spot-check the first sample — easy way to catch chat-template or
    # masking bugs BEFORE we sink hours of A100 time
    sample = train_ds[0]
    n_supervised = sum(1 for t in sample["labels"] if t != -100)
    first_supervised = next((i for i, t in enumerate(sample["labels"]) if t != -100), None)
    print(f"[data] sample 0: total_tokens={len(sample['input_ids'])} "
          f"supervised_tokens={n_supervised} first_sup_idx={first_supervised}")
    if first_supervised is not None:
        # Print the few tokens RIGHT BEFORE the supervised region (should end in "assistant\n")
        # and the START of the target (should be the <think> tag for math, or the empty think for MC)
        before = tokenizer.decode(sample["input_ids"][max(0, first_supervised - 30):first_supervised])
        target_preview = tokenizer.decode([t for t in sample["labels"][first_supervised:first_supervised + 80] if t != -100])
        print(f"[data] context tail before mask: ...{before!r}")
        print(f"[data] supervised target preview: {target_preview!r}")

    # Pre-compute the per-row weights for the domain-balanced sampler
    sampler_weights = build_domain_weights(train_rows)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        optim="adamw_torch",
        seed=args.seed,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=args.run_name,
        remove_unused_columns=False,
    )

    # Seq2Seq collator handles padding labels with -100 so masked positions
    # stay masked after padding to the batch's max length
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100,
    )

    # Our custom Trainer that swaps in the WeightedRandomSampler for per-domain balance
    trainer = DomainBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        sampler_weights=sampler_weights,
    )

    # Dry-run bails out before the actual training loop — useful for verifying
    # setup without burning hours of GPU
    if args.dry_run:
        print("[dry_run] setup OK; exiting before trainer.train()")
        return

    # Go
    trainer.train()

    # Save the final LoRA adapter + tokenizer to a clean subdirectory
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[done] LoRA adapter -> {final_dir}")


if __name__ == "__main__":
    main()
