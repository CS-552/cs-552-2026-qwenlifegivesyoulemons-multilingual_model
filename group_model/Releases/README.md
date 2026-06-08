# Group Model Releases

End-to-end reproducibility index for the joint four-domain model. Each
subdirectory captures one fusion strategy we tried.

## Versions

| Version       | Strategy                                                                              | 4-domain avg pass@1 | Status              |
|---------------|---------------------------------------------------------------------------------------|---------------------|---------------------|
| v_ties        | TIES merge of four specialists (math, gk, safety, multilingual), density 0.5         | 0.295               | rejected            |
| v_mixed_v1    | Mixed-SFT with LoRA over math+gk+multilingual data + SPPFT safety-band layer freeze   | 0.525               | current             |

## Headline result
Mixed-SFT + safety-band freezing **outperforms TIES by 23 percentage points
at the 1.7B scale** on the four-domain joint benchmark, while keeping safety
at the base-model level (0.74) without retraining the safety domain.

## Baseline
Base Qwen3-1.7B on the joint dev set:
```bash
python3 eval_base.py --dev_file data/dev.jsonl
```

## Reproducing a release
1. `cd` into the version's folder.
2. Read `README.md` for the exact recipe.
3. Follow `run.sh` for the cluster command.

## Conventions
- All experiments operate on Qwen3-1.7B unmodified (no architecture changes).
- Mixed-SFT uses LoRA r=64, α=128 with safety blocks 15-19 excluded from `target_modules`.
- Group-model pushes keep Qwen3's **default** chat template (thinking-on by default)
  so math reasoning chains fit — contrast with the multilingual specialty, which
  overrides the template to thinking-off.
