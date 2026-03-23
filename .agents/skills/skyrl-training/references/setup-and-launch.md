# Setup And Launch

## One-Time Environment Setup

Use the **repository `.venv`** (same as `uv sync` at the repo root), not a separate `.skyrl-venv`, unless you intentionally want an isolated SkyRL-only environment.

```bash
# From the forecast-sim repo root (after `uv sync` created `.venv`)
source .venv/bin/activate

cd third_party/SkyRL
uv sync --active --extra fsdp
cd ../..
```

Install FlashAttention in the same env with the cluster CUDA toolchain:

```bash
source /etc/profile.d/modules.sh || true
module load cuda/12.9
source .venv/bin/activate
export FLASH_ATTN_CUDA_ARCHS='90;100'
python -m pip install --no-deps --no-build-isolation flash-attn==2.8.3
```

## Shared `PYTHONPATH`

```bash
export PYTHONPATH="$(pwd):$(pwd)/third_party/SkyRL:$(pwd)/third_party/SkyRL/skyrl-gym"
```

## Optional Manual Data Prep

```bash
python scripts/prepare_skyrl_openforesight_search_data.py \
  --dataset_path "${FSIM_DATASET_PATH}" \
  --prepared-data-dir "${FSIM_SKYRL_PREPARED_DATA_DIR}" \
  --search_db "${FSIM_SEARCH_DB}"
```

`scripts/run_skyrl_openforesight_search.py` can also build the data automatically if it is missing.

Set **`FSIM_SKYRL_PREPARED_DATA_DIR`** (and `data.prepared_data_dir: ${FSIM_SKYRL_PREPARED_DATA_DIR}` in YAML) to **one** directory reused by all runs. To consolidate from older per-experiment folders, copy the latest `train.parquet` / `validation.parquet` into that directory (or run prep once with `--force-rebuild-data`). Per-run SkyRL outputs use **`training.log_path`** / **`FSIM_SKYRL_LOG_BASE`**, not `FSIM_SKYRL_PREPARED_DATA_DIR`.

When the SkyRL config sets `matching: "openrouter"`, the warmup env reuses
forecast-sim's `AnswerMatcher`, so `OPENROUTER_API_KEY` must be available in
`.env` or the shell environment on the worker.

The shared SkyRL config expects `FSIM_SKYRL_MODEL_PATH` to point at the
training checkpoint, for example `Qwen3.5-4B-text`.

The tracked warmup config now carries eval-aligned agent knobs under `agent:`,
including `max_search_results`, `max_outcomes_per_question`, and the
`warmup_*token*` budget fields used by the Qwen tokenbudget runs.

For the tracked Qwen3.5 warmup config, keep `training.ref_fsdp_cpu_offload`
disabled. The validated smoke reached real search, training, and eval only
with the ref model kept on GPU.

Search embeddings are served through forecast-sim's `VLLMInference` wrapper.
The shared config supports:

- `search.embedding_gpu_mem`: vLLM server GPU memory fraction
- `search.aux_cuda_visible_devices`: optional GPU pinning for the embedding server

Leave `aux_cuda_visible_devices: null` to let vLLM use the process-visible GPUs,
or set it when you reserve an extra GPU for search embeddings.

**Chat template (training):** `scripts/run_skyrl_openforesight_search.py` sets
`generator.chat_template.name_or_path` to **null** when `training.chat_template_path`
is unset, so SkyRL uses the **tokenizer’s default HF `chat_template`** for the
policy checkpoint (aligns with vLLM eval when the same model is used). To use the
older Hermes-style Jinja instead, set `training.chat_template_path` to
`skyrl_integration/templates/qwen3_tools_without_thinking.jinja2` (or an absolute path).
That file can wrap assistant spans in `{% generation %}` / `{% endgeneration %}`
when you need `return_assistant_tokens_mask=True` with a **custom** template.

**Thinking:** `training.enable_thinking` is passed into `chat_template_kwargs` and
defaults to **true** (Qwen3.5 HF templates assume thinking blocks). Set
`enable_thinking: false` in YAML to match disable-thinking eval runs.

## Local Launch

```bash
python scripts/run_skyrl_openforesight_search.py \
  --config configs/skyrl_openforesight_search_warmup_qwen3.5_4b.yaml
```

## HTCondor Launch

```bash
python mpi_scripts/skyrl_search/submit_skyrl_search_train.py \
  --config configs/skyrl_openforesight_search_warmup_qwen3.5_4b.yaml \
  --runs 1
```

Override resources or settings at submit time:

```bash
python mpi_scripts/skyrl_search/submit_skyrl_search_train.py \
  --config configs/skyrl_openforesight_search_warmup_qwen3.5_4b.yaml \
  --set resources.gpus=8 \
  --set resources.bid=80 \
  --set training.inference_num_engines=8
```
