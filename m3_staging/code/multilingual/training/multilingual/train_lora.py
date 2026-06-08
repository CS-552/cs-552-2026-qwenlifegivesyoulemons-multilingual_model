"""LoRA SFT of Qwen3-1.7B on multilingual MC data.

Inputs:
    train.jsonl / dev.jsonl  produced by datasets/multilingual/scripts/build_all.py.
    Rows: {"prompt": ..., "answer": "<LETTER>", "lang": "..."}.

Output:
    <output_dir>/final/  LoRA adapter ready for push_to_hub.py.

Defaults are tuned for 1x A100 40GB on the EPFL RCP cluster (bf16 +
gradient checkpointing). Override via CLI if your run is constrained.

Usage (typical, from the cluster):
    python train_lora.py \
        --train_file /scratch/.../train.jsonl \
        --dev_file   /scratch/.../dev.jsonl \
        --output_dir /scratch/.../outputs/lora_v1
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


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def force_no_think(tokenizer):
    """Prepend the no_think override (chat_template.jinja) to the tokenizer's
    chat template. Idempotent: re-running won't stack the override."""
    override = (HERE / "chat_template.jinja").read_text(encoding="utf-8")
    base = tokenizer.chat_template or ""
    if override.strip() and override.strip() not in base:
        tokenizer.chat_template = override + base
    return tokenizer


def encode_example(row, tokenizer, max_length):
    """Tokenize one row, mask labels on everything but the assistant answer."""
    messages = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": f"\\boxed{{{row['answer']}}}"},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    prefix_text = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True,
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]

    full_ids = full_ids[:max_length]
    prefix_len = min(len(prefix_ids), len(full_ids))
    labels = [-100] * prefix_len + full_ids[prefix_len:]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def build_lang_weights(rows):
    """Equal probability per language regardless of pool size."""
    counts = Counter(r.get("lang", "?") for r in rows)
    return [1.0 / counts[r.get("lang", "?")] for r in rows]


class LangBalancedTrainer(Trainer):
    """HF Trainer with a per-language weighted sampler (one slot per lang).

    This is the 'English penalty' mechanism: regardless of how big the English
    MMLU pool is, English is drawn with the same probability per batch as
    Russian or Hindi.
    """

    def __init__(self, *args, sampler_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampler_weights = sampler_weights

    def _get_train_sampler(self, train_dataset=None):
        if self._sampler_weights is None:
            return super()._get_train_sampler(train_dataset)
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
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--dry_run", action="store_true",
                    help="run all setup (load model, apply LoRA, tokenize dataset, "
                         "print diagnostics) and exit BEFORE training. Use this to "
                         "verify the model is Qwen3-1.7B, LoRA is wrapping the right "
                         "modules, and the label-mask boundary lands where expected.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] tokenizer {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    force_no_think(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[load] tokenizer vocab_size={tokenizer.vocab_size} "
          f"has_chat_template={tokenizer.chat_template is not None}")

    print(f"[load] base model {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    cfg = model.config
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[load] model_type={cfg.model_type} "
          f"hidden_size={cfg.hidden_size} "
          f"layers={cfg.num_hidden_layers} "
          f"heads={cfg.num_attention_heads} "
          f"kv_heads={getattr(cfg, 'num_key_value_heads', 'n/a')} "
          f"vocab={cfg.vocab_size}")
    print(f"[load] total params: {total_params:,}  "
          f"(Qwen3-1.7B should report ~1,720,000,000)")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    wrapped = [n for n, m in model.named_modules() if hasattr(m, "lora_A")]
    print(f"[lora] {len(wrapped)} modules got LoRA adapters")
    if wrapped:
        print(f"[lora] first 3 wrapped: {wrapped[:3]}")
        if len(wrapped) > 3:
            print(f"[lora] last  3 wrapped: {wrapped[-3:]}")

    print(f"[data] reading {args.train_file} / {args.dev_file}")
    train_rows = load_jsonl(args.train_file)
    dev_rows = load_jsonl(args.dev_file)
    print(f"[data] train={len(train_rows)} dev={len(dev_rows)}")
    print(f"[data] train langs: {dict(Counter(r.get('lang', '?') for r in train_rows))}")

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

    sample = train_ds[0]
    n_supervised = sum(1 for t in sample["labels"] if t != -100)
    first_supervised = next((i for i, t in enumerate(sample["labels"]) if t != -100), None)
    print(f"[data] sample 0: total_tokens={len(sample['input_ids'])} "
          f"supervised_tokens={n_supervised} "
          f"first_supervised_idx={first_supervised}")
    if first_supervised is not None:
        before = tokenizer.decode(sample["input_ids"][max(0, first_supervised - 20):first_supervised])
        target_ids = [t for t in sample["labels"][first_supervised:first_supervised + 50] if t != -100]
        target = tokenizer.decode(target_ids)
        print(f"[data] context tail before mask ends: ...{before!r}")
        print(f"[data] supervised target           : {target!r}")
        print(f"[data] expect target to contain \\boxed{{X}} for some letter X")

    sampler_weights = build_lang_weights(train_rows)

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

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100,
    )

    trainer = LangBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        sampler_weights=sampler_weights,
    )

    if args.dry_run:
        print("[dry_run] setup complete; exiting before trainer.train()")
        return

    trainer.train()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[done] LoRA adapter -> {final_dir}")


if __name__ == "__main__":
    main()
