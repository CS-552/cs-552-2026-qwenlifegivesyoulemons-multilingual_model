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

## Iteration history

| Version | Change | pass@1 |
|---|---|---|
| v1 | LoRA r=32 SFT on Global-MMLU + INCLUDE + CS + English | 0.74 |
| v2 | + CS upsampling (cs_weight=3), split-by-source leak fix | 0.75 |
| v3 | + bulk cap, INCLUDE/CS reweight, LoRA r=64 | 0.75 (plateau) |
| v4 | **Move 2: CPT → SFT** (see below) | TBD |

v1–v3 plateaued at ~75%. Reweighting and capacity changes moved nothing →
the bottleneck is **knowledge the base model lacks**, not training
distribution. That's CPT territory.

## Move 2: continued pretraining (two-stage)

Stage 1 — full-FT continued pretraining on a regional Wikipedia corpus to
inject knowledge. Stage 2 — the usual SFT-LoRA, but on the CPT'd checkpoint
instead of vanilla Qwen3-1.7B.

```bash
# Stage 0: build the CPT corpus (from datasets/multilingual/scripts/)
python build_cpt_corpus.py --out_dir ../cpt_corpus --cache_dir /scratch/hf_cache

# Stage 1: full-FT CPT (cluster, ~several hours)
#   run_cpt.sh wraps train_cpt.py; submit it non-interactively like run_train.sh
#   output: outputs/cpt_v1/final/

# Stage 2: SFT-LoRA on the CPT checkpoint
python train_lora.py \
    --base_model /scratch/multilingual/training/multilingual/outputs/cpt_v1/final \
    --train_file /scratch/multilingual/datasets/multilingual/train.jsonl \
    --dev_file   /scratch/multilingual/datasets/multilingual/dev.jsonl \
    --output_dir /scratch/multilingual/training/multilingual/outputs/lora_v4 \
    --lora_r 64 --lora_alpha 128 --run_name lora_v4

# Stage 3: merge + push (push_to_hub merges the v4 LoRA onto the CPT base,
# because --adapter_dir's adapter_config.json records the CPT base path)
python push_to_hub.py \
    --adapter_dir .../outputs/lora_v4/final \
    --base_model  .../outputs/cpt_v1/final \
    --hf_repo cs-552-2026-<org>/multilingual_model \
    --commit_msg "v4: CPT (regional Wikipedia) -> SFT-LoRA" --push
```

**Fallback**: v3 LoRA stays on HF until v4 is validated. If CPT regresses
pass@1, simply don't push v4 — v3 remains the graded model. Keep
`outputs/lora_v3/` on `/scratch` untouched.

**Why full-FT CPT, not LoRA-CPT**: a low-rank adapter can't hold broad
encyclopedic knowledge — that defeats CPT's purpose. The merge-compatibility
cost (CPT'd base drifts from Qwen3, breaking the team's *weight-averaging*
fusion) is acceptable: the *mixed-SFT* and *KD* fusion strategies don't use
weights, and standalone leaderboard grading doesn't care.

### CPT files

- `datasets/multilingual/scripts/build_cpt_corpus.py` — stream Wikipedia → text shards
- `train_cpt.py` — full-FT causal-LM pretraining (NOT LoRA)
- `run_cpt.sh` — cluster wrapper for stage 1
