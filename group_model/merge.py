"""Merge N Qwen3-1.7B-shaped HF checkpoints into one safetensors file.

Self-contained replacement for mergekit. mergekit 0.1.x has a class-definition
bug that hits with the cluster image's pydantic, and pinning to its required
pydantic 2.10.6 doesn't fix it. Re-implementing the two methods we need from
first principles is ~150 lines and removes the dependency entirely.

Two methods, same YAML config format as mergekit:

  linear:   W_merged = sum_i (w_i * W_i)
            (with weights normalized to sum to 1)

  ties:     T_i      = W_i - W_base                       # task vector
            TRIM:    zero all but top-K% magnitude per T_i (K = density)
            ELECT:   per-param majority sign across pruned T_i
            MERGE:   average only T_i values whose sign matches the elect
            W_merged = W_base + merged_task_vector

Reads the same YAML as the original mergekit configs (merge_linear.yaml /
merge_ties.yaml) for drop-in compatibility — no config changes needed.
"""

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

MODEL_FILE_PATTERNS = [
    "*.safetensors", "*.json", "*.txt", "*.jinja",
    "tokenizer*", "vocab*", "merges*",
]


def download_model(repo_id, cache_dir):
    print(f"  [dl] {repo_id}", flush=True)
    return snapshot_download(
        repo_id, cache_dir=cache_dir,
        allow_patterns=MODEL_FILE_PATTERNS,
    )


def load_state_dict(model_dir, label=""):
    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors in {model_dir}")
    print(f"  [load] {label} from {model_dir} ({len(shards)} shard(s))", flush=True)
    sd = {}
    for shard in shards:
        sd.update(load_file(str(shard), device="cpu"))
    total_gb = sum(t.numel() * t.element_size() for t in sd.values()) / 1e9
    print(f"  [load] {label} done: {len(sd)} tensors, {total_gb:.2f} GB", flush=True)
    return sd


def is_float_tensor(t):
    return t.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _log_progress(method, i, n_total, key, t_start):
    import time
    elapsed = time.time() - t_start
    rate = (i + 1) / max(elapsed, 1e-6)
    eta = (n_total - i - 1) / max(rate, 1e-6)
    print(f"  [{method}] {i+1}/{n_total} {key} "
          f"({elapsed:.1f}s elapsed, {eta:.0f}s ETA)", flush=True)


def linear_merge(state_dicts, weights):
    """Weighted element-wise average across N state dicts."""
    import time
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()
    keys = list(state_dicts[0].keys())
    out = {}
    t_start = time.time()
    for i, key in enumerate(keys):
        ref = state_dicts[0][key]
        if not is_float_tensor(ref):
            out[key] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            acc.add_(sd[key].to(torch.float32), alpha=w.item())
        out[key] = acc.to(ref.dtype)
        # Log every tensor for the first 5, then every 20
        if i < 5 or i % 20 == 0:
            _log_progress("linear", i, len(keys), key, t_start)
    return out


def ties_merge(state_dicts, base_sd, weights, density):
    """TIES: trim + sign-elect + disjoint-average task vectors."""
    import time
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()
    keys = list(state_dicts[0].keys())
    out = {}
    t_start = time.time()
    for i, key in enumerate(keys):
        ref = state_dicts[0][key]
        if not is_float_tensor(ref):
            out[key] = ref.clone()
            continue
        base = base_sd[key].to(torch.float32)
        tvs = [sd[key].to(torch.float32) - base for sd in state_dicts]

        # TRIM: per task vector, keep top-density% magnitude, zero the rest
        for j, tv in enumerate(tvs):
            flat = tv.abs().flatten()
            if flat.numel() == 0:
                continue
            k = max(int(flat.numel() * density), 1)
            threshold = torch.topk(flat, k, largest=True).values[-1]
            tvs[j] = torch.where(tv.abs() >= threshold, tv, torch.zeros_like(tv))

        stacked = torch.stack(tvs, dim=0)               # (N, *shape)
        elected = torch.sign(stacked.sum(dim=0))        # (*shape)

        agree = (torch.sign(stacked) == elected.unsqueeze(0)).float()
        contrib_count = agree.sum(dim=0).clamp(min=1.0)
        w_expand = weights.view(-1, *([1] * (stacked.dim() - 1)))
        merged_tv = (stacked * agree * w_expand).sum(dim=0) / contrib_count

        out[key] = (base + merged_tv).to(ref.dtype)
        if i < 5 or i % 20 == 0:
            _log_progress("ties", i, len(keys), key, t_start)
    return out


def save_merged(state_dict, output_dir, source_dir):
    """Save state_dict + copy tokenizer / config / etc. from source_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(output_dir / "model.safetensors"))
    src = Path(source_dir)
    for f in src.iterdir():
        if f.suffix == ".safetensors":
            continue
        if f.name.endswith(".safetensors.index.json"):
            continue  # output is single-file
        if f.is_file():
            shutil.copy2(f, output_dir / f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="mergekit-style YAML")
    ap.add_argument("output_dir")
    ap.add_argument("--cache_dir", default="/scratch/hf_cache")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    method = cfg["merge_method"]
    models = cfg["models"]
    weights = [m["parameters"].get("weight", 1.0) for m in models]
    print(f"[merge] method={method}  n_models={len(models)}  weights={weights}", flush=True)

    print("[merge] downloading specialty models...", flush=True)
    model_dirs = [download_model(m["model"], args.cache_dir) for m in models]

    print("[merge] loading specialty state dicts (CPU)...", flush=True)
    sds = [load_state_dict(d, label=f"specialty[{i}]")
           for i, d in enumerate(model_dirs)]

    if method == "linear":
        print(f"[merge] starting linear merge over {len(sds[0])} tensors", flush=True)
        merged = linear_merge(sds, weights)
    elif method == "ties":
        base_repo = cfg["base_model"]
        density = models[0]["parameters"].get("density", 0.5)
        print(f"[merge] TIES density={density}, base={base_repo}", flush=True)
        base_dir = download_model(base_repo, args.cache_dir)
        base_sd = load_state_dict(base_dir, label="base")
        print(f"[merge] starting TIES merge over {len(sds[0])} tensors "
              f"(heads up: the first tensor is the embedding ~311M params, "
              f"expect ~1-3 min before the first '[ties] 1/N' print)", flush=True)
        merged = ties_merge(sds, base_sd, weights, density)
    else:
        raise SystemExit(f"unknown merge_method: {method}")

    print(f"[merge] saving merged checkpoint -> {args.output_dir}", flush=True)
    save_merged(merged, args.output_dir, model_dirs[0])
    print(f"[done] {args.output_dir}", flush=True)
    print(f"[next] python3 push_group.py --merged_dir {args.output_dir} --push",
          flush=True)


if __name__ == "__main__":
    main()
