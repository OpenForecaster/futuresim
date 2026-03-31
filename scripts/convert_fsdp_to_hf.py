#!/usr/bin/env python3
"""Convert SkyRL FSDP sharded checkpoints to HuggingFace safetensors format.

Loads FSDP-sharded model weights (model_world_size_N_rank_*.pt), merges them
into a single state dict, and saves as safetensors using the HF model's
save_pretrained(). The output can be directly loaded by vLLM.

Usage:
    python scripts/convert_fsdp_to_hf.py --ckpt_dir /path/to/ckpt_dir
    python scripts/convert_fsdp_to_hf.py --ckpt_dir /path/to/ckpt_dir --step global_step_750
    python scripts/convert_fsdp_to_hf.py --ckpt_dir /path/to/ckpt_dir --step latest --output_dir /path/to/output
"""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_latest_step(ckpt_dir: Path) -> str:
    """Find the latest checkpoint step directory."""
    latest_file = ckpt_dir / "latest_ckpt_global_step.txt"
    if latest_file.exists():
        step_num = latest_file.read_text().strip()
        step_dir = f"global_step_{step_num}"
        if (ckpt_dir / step_dir).exists():
            return step_dir

    # Fallback: find highest numbered global_step_* directory
    step_dirs = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("global_step_")],
        key=lambda d: int(re.search(r"(\d+)", d.name).group(1)),
    )
    if not step_dirs:
        raise FileNotFoundError(f"No global_step_* directories found in {ckpt_dir}")
    return step_dirs[-1].name


def load_fsdp_shards(policy_dir: Path) -> dict:
    """Load and merge FSDP sharded model weights into a single state dict."""
    shard_files = sorted(policy_dir.glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No model shards found in {policy_dir}")

    world_size = len(shard_files)
    print(f"Found {world_size} FSDP shards")

    merged = {}
    for shard_file in shard_files:
        rank = int(re.search(r"rank_(\d+)", shard_file.name).group(1))
        print(f"  Loading rank {rank}: {shard_file.name}")
        shard = torch.load(shard_file, map_location="cpu", weights_only=False)

        for key, tensor in shard.items():
            if key in merged:
                # FSDP shards the first dimension — concatenate
                merged[key] = torch.cat([merged[key], tensor], dim=0)
            else:
                merged[key] = tensor

    print(f"Merged state dict: {len(merged)} parameters")
    return merged


def convert_checkpoint(ckpt_dir: str, step: str = "latest", output_dir: str = None):
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Resolve step
    if step == "latest":
        step = find_latest_step(ckpt_path)
    step_dir = ckpt_path / step
    if not step_dir.exists():
        raise FileNotFoundError(f"Step directory not found: {step_dir}")

    policy_dir = step_dir / "policy"
    hf_dir = policy_dir / "huggingface"

    if not policy_dir.exists():
        raise FileNotFoundError(f"Policy directory not found: {policy_dir}")

    # Check if already exported as HF safetensors
    if list(policy_dir.glob("*.safetensors")) or list(hf_dir.glob("*.safetensors")):
        print(f"Safetensors already exist in {policy_dir}, skipping conversion")
        return

    # Determine output path
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = step_dir / "hf_model"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Converting: {step_dir}")
    print(f"Output: {out_path}")

    # Load the base model config from huggingface subdir
    if not hf_dir.exists():
        raise FileNotFoundError(
            f"HuggingFace config not found at {hf_dir}. "
            "Need config.json and tokenizer files."
        )

    print("Loading base model from config...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Load and merge FSDP shards
    print("Loading FSDP shards...")
    merged_state_dict = load_fsdp_shards(policy_dir)

    # Load into model
    print("Loading merged weights into model...")
    missing, unexpected = model.load_state_dict(merged_state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    # Save as safetensors
    print(f"Saving to {out_path}...")
    model.save_pretrained(out_path, safe_serialization=True)

    # Copy tokenizer files
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    tokenizer.save_pretrained(out_path)

    print(f"Done! Model saved to {out_path}")
    print(f"You can now run it with vLLM:")
    print(f"  python -m vllm.entrypoints.openai.api_server --model {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SkyRL FSDP checkpoints to HuggingFace safetensors"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/lustre/scratch/nchandak/forecast-sim/skyrl/forecast_grpo_vs_opsd/opsd_Qwen3-4B_v5_freeform61k_lr5e-6",
        help="Path to the checkpoint directory containing global_step_* subdirectories",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="latest",
        help="Which step to convert (e.g. 'global_step_750' or 'latest')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for HF model (default: <step_dir>/hf_model)",
    )
    args = parser.parse_args()
    convert_checkpoint(args.ckpt_dir, args.step, args.output_dir)


if __name__ == "__main__":
    main()
