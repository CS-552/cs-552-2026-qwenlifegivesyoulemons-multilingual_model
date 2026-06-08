"""Full-FT continued pretraining of Qwen3-1.7B on the regional Wikipedia corpus.

Move 2, stage 1. Full fine-tune (NOT LoRA) — the whole point of CPT is
absorbing new knowledge, and a low-rank adapter can't hold broad encyclopedic
facts. Stage 2 is then plain SFT-LoRA on top of this checkpoint:

    python train_lora.py --base_model <this output_dir>/final  [...]

Catastrophic-forgetting controls:
  - low LR (default 1e-5, ~10x below the SFT LoRA lr)
  - 1 epoch over a capped corpus
  - v3 LoRA stays the HF fallback if CPT regresses pass@1 (do not overwrite
    the HF repo until the CPT->SFT model is validated)

Memory (1.7B full-FT, A100 40GB): bf16 weights+grads ~7GB, AdamW fp32 states
~14GB, rest activations. Fits with gradient checkpointing + small per-device
batch + grad accumulation. If OOM: lower --block_size or --batch_size.
"""

import argparse
import json
import os
from collections import Counter
from itertools import chain
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

DEFAULT_BASE = "Qwen/Qwen3-1.7B"


def load_corpus(corpus_dir):
    rows = []
    for p in sorted(Path(corpus_dir).glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=DEFAULT_BASE)
    ap.add_argument("--corpus_dir", required=True,
                    help="dir of <lang>.jsonl from build_cpt_corpus.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.02)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", default="cpt_v1")
    ap.add_argument("--dry_run", action="store_true",
                    help="load model, tokenize+pack corpus, print stats, exit")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] tokenizer {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[load] base model {args.base_model} (FULL fine-tune, bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    total = sum(p.numel() for p in model.parameters())
    print(f"[load] model_type={model.config.model_type} "
          f"params={total:,} (ALL trainable — full FT)")

    rows = load_corpus(args.corpus_dir)
    if not rows:
        raise SystemExit(f"no corpus found in {args.corpus_dir}")
    print(f"[data] {len(rows)} articles; "
          f"langs={dict(Counter(r.get('lang', '?') for r in rows))}")

    block = args.block_size
    packed_cache = Path(args.corpus_dir) / f".packed_{block}"

    if packed_cache.exists():
        from datasets import load_from_disk
        ds = load_from_disk(str(packed_cache))
        print(f"[data] loaded pre-packed dataset from {packed_cache} "
              f"({len(ds)} blocks)")
    else:
        ds = Dataset.from_list(rows)
        ds = ds.map(lambda b: tok(b["text"], add_special_tokens=True),
                    batched=True, remove_columns=ds.column_names,
                    desc="tokenize")

        def group(batch):
            concat = list(chain(*batch["input_ids"]))
            n = (len(concat) // block) * block
            ids = [concat[i:i + block] for i in range(0, n, block)]
            return {"input_ids": ids, "labels": [x[:] for x in ids]}

        # remove_columns drops the leftover attention_mask (old row count)
        # so the row-count change from packing doesn't collide.
        ds = ds.map(group, batched=True, remove_columns=ds.column_names,
                    desc=f"pack into {block}-token blocks")
        ds.save_to_disk(str(packed_cache))
        print(f"[data] saved pre-packed dataset to {packed_cache}")

    n_tok = len(ds) * block
    print(f"[data] {len(ds)} packed blocks of {block} "
          f"(~{n_tok / 1e6:.1f}M training tokens)")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        max_grad_norm=1.0,
        optim="adamw_torch",
        seed=args.seed,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=args.run_name,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds, data_collator=default_data_collator,
    )

    if args.dry_run:
        print("[dry_run] setup OK; exiting before training")
        return

    trainer.train()
    final = out_dir / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    print(f"[done] CPT checkpoint -> {final}")
    print(f"[next] python train_lora.py --base_model {final} ...")


if __name__ == "__main__":
    main()
