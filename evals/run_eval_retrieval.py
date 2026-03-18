"""
Wrapper to run eval_retrieval.py from forecast-sim env with Qwen3.5-27B-text.

Usage:
    python run_eval_retrieval.py [--model_dir /path/to/model] [other eval_retrieval args...]

Changes from original:
    - Caps tensor_parallel_size to 4 (Qwen3.5-27B has num_key_value_heads=4)
    - Removes unsupported 'language_model_only' vLLM kwarg for this env
    - Uses model dir with patched config (Qwen3.5-27B-text-only)
"""
import sys
import os

# Add the custom_eval_scripts directory to path for utils import
sys.path.insert(0, "/home/nchandak/forecasting/custom_eval_scripts")

# Patch load_model_and_tokenizer to work with this vLLM version
import utils as eval_utils
import torch
import logging
from vllm import LLM
from transformers import AutoTokenizer

def patched_load_model_and_tokenizer(
    model_path, model_name=None, gpu_memory_utilization=0.85,
    dtype="auto", language_model_only=False
):
    logger = logging.getLogger(__name__)
    if model_name is None:
        model_name = model_path.rstrip("/").split("/")[-1]
    logger.info(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tc = config.text_config if hasattr(config, 'text_config') else config
    num_kv_heads = getattr(tc, 'num_key_value_heads', 8)
    gpu_count = torch.cuda.device_count()

    # TP must divide num_key_value_heads evenly
    tp_size = min(gpu_count, num_kv_heads)

    # For large MoE models that don't fit in TP GPUs, use pipeline parallelism
    # 228GB model needs more than 2x80GB GPUs
    pp_size = 1
    if tp_size * 80 < 230:  # rough check: model won't fit in TP GPUs alone
        pp_size = min(gpu_count // tp_size, 4)
    logger.info(f"Using TP={tp_size}, PP={pp_size} (num_kv_heads={num_kv_heads})")

    model = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )
    return model, tokenizer

eval_utils.load_model_and_tokenizer = patched_load_model_and_tokenizer

if __name__ == "__main__":
    # Set default model_dir to text-only if not specified
    if "--model_dir" not in sys.argv:
        sys.argv.extend(["--model_dir", "/fast/nchandak/models/Qwen3.5-27B-text-only"])

    # Run eval_retrieval's main block by exec'ing it as __main__
    eval_script = "/home/nchandak/forecasting/custom_eval_scripts/eval_retrieval.py"
    with open(eval_script) as f:
        code = f.read()

    # Execute with __name__ == "__main__" so the main block runs
    exec(compile(code, eval_script, "exec"), {"__name__": "__main__", "__file__": eval_script})
