# vLLM 0.13 Embedding Hang: Debug Summary

## The Problem

`model.embed()` in vLLM 0.13.0 **hangs on the very last prompt** of every call and never returns. The function processes all N-1 items successfully (visible in stderr progress bar as `N-1/N` at full speed), but the Nth (final) item never completes. The call blocks indefinitely.

This is **100% reproducible** — it happens every time regardless of:
- Batch size (tested 32, 4096, 41940)
- Whether the last item is real data or a dummy padding string
- Whether prefix caching is enabled or disabled
- GPU type (A40 44GB or A100 80GB)
- Number of workers

### Exact stderr pattern observed
```
Processed prompts: 100%|█████████▉| 4095/4096 [01:49<00:00, 44.48it/s] ← hangs here forever
Processed prompts: 100%|█████████▉| 41939/41940 [35:03<00:00, 18.96it/s] ← same pattern
```

The last item is always stuck. The progress bar never reaches `N/N`.

## Environment

- **vLLM**: 0.13.0
- **torch**: 2.9.0+cu126
- **transformers**: 4.57.4
- **Python**: 3.12.10
- **Model**: Qwen3-Embedding-8B (local, `/mnt/nfs/datasets_ac/models/Qwen3-Embedding-8B`)
- **Architecture**: `Qwen3ForCausalLM` wrapped as SentenceTransformer (Transformer → Pooling → Normalize)
- **GPUs tested**: NVIDIA A40 (44GB), NVIDIA A100 (80GB)
- **flash-attn**: NOT installed (standalone package), but vLLM uses its built-in FLASH_ATTN backend
- **Cluster**: AISA cluster with SLURM

## What Was Tried

### 1. SentenceTransformers backend (before vLLM upgrade)
- **Approach**: Used `sentence-transformers` 5.2.2 directly instead of vLLM
- **Result**: Worked but extremely slow. OOM at batch=64 on A40, fell back to batch=8 → ~1.5 emb/s. Would take ~8.5 days with 4 A40s.
- **Why abandoned**: Too slow for ~23.7M chunks

### 2. vLLM 0.13 with default max_model_len=40960
- **Approach**: Collaborator updated env to vLLM 0.13 which supports Qwen3 natively
- **Result**: `model.embed()` hung on the very last prompt (41939/41940). Ran 15+ hours with no output files saved.
- **Details**: KV cache concurrency was only 4.46x at max_model_len=40960. vLLM processed ~20 it/s in progress bar but never returned.

### 3. Reduced max_model_len=2048
- **Approach**: Set `max_model_len=2048` to improve KV cache concurrency
- **Result**: Crashed with `ValueError: The decoder prompt (length 2811) is longer than the maximum model length of 2048`
- **Why**: Some chunks exceed 2048 tokens after tokenization despite being chunked at 512 tokens (title prepending + tokenizer differences)

### 4. max_model_len=8192 + truncate_prompt_tokens
- **Approach**: Set max_model_len=8192, added `truncate_prompt_tokens=max_model_len` to handle oversized prompts
- **Result**: Same hang at last prompt. Concurrency improved to 22.3x (A40) / 50x (A100) but still hangs.

### 5. Batched embed() calls (batch_size=4096)
- **Approach**: Instead of one huge `model.embed(all_41940_texts)`, process in batches of 4096
- **Result**: Each batch hangs at last item. First batch: `4095/4096` then stuck forever.

### 6. Batched with smaller batch_size=32
- **Approach**: Used shell script's batch_size=32
- **Result**: Same hang. `31/32` processed, stuck on item 32. ~70 emb/s throughput before hang.

### 7. Dummy padding text appended
- **Approach**: Append a dummy text `"padding"` to each batch so the hang affects only the dummy
- **Result**: vLLM processes all real items (4096/4097) but hangs on the dummy. `model.embed()` never returns because it waits for ALL items including the hung dummy.

### 8. Disabled prefix caching
- **Approach**: `enable_prefix_caching=False` in LLM constructor
- **Result**: Same hang at `4095/4096`

### 9. model.encode() instead of model.embed()
- **Approach**: Used the older `model.encode()` API
- **Result**: Crashed with `ValueError: pooling_task required for LLM.encode` — incompatible with `convert="embed"` mode

## What Was NOT Tried (Potential Next Steps)

### A. Thread-based timeout with dummy padding (most promising)
The dummy approach *does* compute all real embeddings — it just blocks waiting for the hung dummy. Running `model.embed()` in a thread with a timeout would allow collecting the already-computed results:
```python
import concurrent.futures
# Run embed(batch + [dummy]) in a thread
# After timeout, the real N results are already in memory
# Kill thread, extract results
```
**Complication**: vLLM's internal state may be corrupted after forcefully abandoning a call.

### B. Async vLLM API
Use `AsyncLLMEngine` or `model.beam_search`-style async calls to collect embeddings as they complete rather than waiting for the synchronous return.

### C. Downgrade vLLM
Try vLLM 0.8.5 (minimum for Qwen3 embedding) or 0.10.x — the hang may be a regression in 0.13.

### D. Use SentenceTransformers with optimizations
Go back to SentenceTransformers but with:
- flash_attention_2 properly installed (`pip install flash-attn --no-build-isolation`)
- Larger batch sizes on A100s (batch=64-128 should fit in 80GB)
- This was working at ~1.1 it/s with batch=8 on A40; with batch=64 on A100 + flash_attn could be ~10-20x faster

### E. File a vLLM bug report
This appears to be a vLLM 0.13 bug in the V1 pooling engine. The hang is in the EngineCore subprocess which processes the last request but never signals completion back to the client. Relevant search:
- https://github.com/vllm-project/vllm/issues/17972 (server hangs after initial requests)
- https://github.com/vllm-project/vllm/issues/22731 (0 tokens/s throughput with running requests)

### F. Use V0 engine
vLLM 0.13 defaults to V1 engine. Forcing `--engine-mode v0` or equivalent Python flag may avoid the hang if it's V1-specific.

## Current State of the Code

### Key files
- `scripts/embed_articles.py` — main embedding script with vLLM backend
- `aisa_scripts/embed/run_embed_aisa.sh` — SLURM shell wrapper
- `aisa_scripts/embed/submit_job_aisa.py` — SLURM job submitter

### Current configuration
- `max_model_len=8192` (env: `EMBED_MAX_MODEL_LEN`)
- `enable_prefix_caching=False`
- `batch_size=4096` (from shell script)
- `truncate_prompt_tokens=max_model_len`
- `convert="embed"` mode
- `model.encode()` is used (but crashes — needs to be reverted to `model.embed()`)

### Embedding data
- **Total scope**: 2023-01-01 to 2026-01-31, ~1026 days with data, ~11.5M articles, ~23.7M chunks
- **Completed**: 20 days (from earlier SentenceTransformers run)
- **Remaining**: ~1006 days

### Performance benchmarks (when not hanging)
| Backend | GPU | Batch size | Throughput | Est. total time (4 workers) |
|---------|-----|-----------|------------|---------------------------|
| SentenceTransformers | A40 | 8 | ~8 emb/s/worker | ~8.5 days |
| vLLM 0.13 (before hang) | A40 | 32 | ~70 emb/s/worker | ~23 hours |
| vLLM 0.13 (before hang) | A100 | 4096 | ~45 it/s/worker | ~3-6 hours (estimated) |
