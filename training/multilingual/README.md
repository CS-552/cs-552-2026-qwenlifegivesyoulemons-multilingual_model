# Multilingual specialty training

LoRA SFT of Qwen3-1.7B on the multilingual MC data produced under
`datasets/multilingual/`.

## Files

- `train_lora.py` — main training script. LoRA, bf16, per-language weighted sampling.
- `push_to_hub.py` — merge adapter into base, save vLLM-loadable checkpoint, push to HF.
- `chat_template.jinja` — one-line override prepended to Qwen3's chat template to force `no_think`. Both scripts read it.
- `outputs/` — checkpoints, gitignored.

## End-to-end flow

```bash
# 1. Build the data once (from datasets/multilingual/scripts/)
python build_global_mmlu.py --out_dir ../cleaned_datasets --cache_dir /scratch/hf_cache
python build_include.py     --out_dir ../cleaned_datasets --cache_dir /scratch/hf_cache
python build_mmlu_en.py     --out_dir ../cleaned_datasets --cache_dir /scratch/hf_cache
python build_global_mmlu.py --out_dir ../cleaned_datasets/cs_only --cs_only --cache_dir /scratch/hf_cache
python build_all.py   # produces train.jsonl + dev.jsonl at ../

# 2. Train (on the cluster)
python train_lora.py \
    --train_file /scratch/<repo>/datasets/multilingual/train.jsonl \
    --dev_file   /scratch/<repo>/datasets/multilingual/dev.jsonl \
    --output_dir /scratch/<repo>/training/multilingual/outputs/lora_v1

# 3. Merge + push when satisfied
python push_to_hub.py \
    --adapter_dir /scratch/<repo>/training/multilingual/outputs/lora_v1/final \
    --hf_repo cs-552-2026-<org>/multilingual_model \
    --push
```

## Key decisions baked in

- **LoRA r=32, alpha=64** on attn + MLP — merge-friendly for the team's fusion stage.
- **bf16 + gradient checkpointing** — fits the A100 40GB cap on the RCP cluster.
- **`no_think` mode** — pass@1 metric + 1800s wall-clock budget makes long reasoning a tax.
- **Per-language weighted sampling** — English is drawn at 1/6 probability regardless of MMLU's larger pool size (the "English penalty" from the design discussion).
- **Variable-k augmentation upstream** — handled by `augment_variable_k.py`; this script just trains on whatever distribution it gets.
- **Output: `\boxed{LETTER}`** — assistant target format matches the course CI contract.

## Tunable hyperparameters

All in `train_lora.py --help`:

| Flag | Default | When to change |
|---|---|---|
| `--epochs` | 2.0 | Bump to 3 if dev loss still falling; cut to 1 if overfit early |
| `--batch_size` | 4 | Try 8 if VRAM headroom; reduce to 2 with grad_accum=16 if OOM |
| `--grad_accum` | 8 | Keep effective batch ~32 |
| `--lr` | 1e-4 | Try 5e-5 if loss spikes; 2e-4 if convergence too slow |
| `--lora_r` | 32 | 16 cheaper, 64 if r=32 underfits |
| `--max_length` | 2048 | Drop to 1024 if VRAM tight; long 20-option items push 2k |
