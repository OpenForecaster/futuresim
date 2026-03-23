# Runtime Notes

- `mpi_scripts/skyrl_search/run_skyrl_search_train.sh` wires `PYTHONPATH` so repo-local `sitecustomize.py` shims are active in workers.
- OpenForesight warmup **prepared parquets** (`train.parquet` / `validation.parquet`) use **`FSIM_SKYRL_PREPARED_DATA_DIR`** / `data.prepared_data_dir` as a **single shared cache** across runs. Per-run SkyRL logs and infra use **`FSIM_SKYRL_LOG_BASE`** and `training.log_path`, not the prepared-data path.
- The wrapper exports `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` to avoid duplicate-GPU NCCL initialization issues on packed nodes.
- The wrapper also sets `SOFTFILELOCK=1` and cluster-safe Hugging Face cache paths.
- Real FlashAttention is expected to be installed in the env; there is no repo-local fake `flash_attn` shim anymore.
- The OpenForesight warmup env can use forecast-sim's OpenRouter-backed
  `AnswerMatcher` via `matching: "openrouter"` and logs matcher count/time/cost
  metrics through the usual SkyRL env metadata stream.
- The warmup env now reuses forecast-sim's `VLLMInference` wrapper for LanceDB
  embeddings instead of lazy-loading `vllm.LLM(...)` in-process. This avoids
  CUDA bad-fork failures when search first runs inside SkyRL's Ray actor.
- The warmup env also reuses the shared `BudgetTracker` semantics from the
  Qwen eval loop. The tracked config defaults to token-budget feedback
  (`warmup_max_total_tokens`, reserve threshold, force-submit threshold) rather
  than action-count instructions.
- `search.embedding_gpu_mem` and `search.aux_cuda_visible_devices` control the
  embedding server footprint and optional GPU pinning.
- The tracked Qwen3.5 warmup config keeps `trainer.ref.fsdp_config.cpu_offload`
  disabled. In the validated smoke, enabling ref CPU offload caused a Qwen3.5
  rotary-embedding device mismatch during the ref forward pass.
- `scripts/run_skyrl_openforesight_search.py` defaults to the tokenizer’s HF
  `chat_template` when `training.chat_template_path` is unset (Hermes-style
  `skyrl_integration/templates/qwen3_tools_without_thinking.jinja2` is opt-in).
  `training.enable_thinking` defaults to **true** (Qwen3.5 HF chat templates use
  thinking blocks by default); set `enable_thinking: false` in YAML to match
  disable-thinking eval runs.
- Warmup env string ingress uses [`skyrl_integration/vllm_qwen3_coder_text.py`](../../../../skyrl_integration/vllm_qwen3_coder_text.py)
  to mirror vLLM’s `Qwen3CoderToolParser` XML (`<tool_call>` / `<function=` /
  `<parameter=`). If you use a custom Jinja template for SkyRL loss masks, wrap assistant spans
  in `{% generation %}` / `{% endgeneration %}` when needed for
  `return_assistant_tokens_mask=True`.
- Qwen boundary rule: keep model-facing prompt text and tool schemas on the
  native `agents/qwenAgent` path. The SkyRL integration adds chat templating and
  env-side **vLLM `qwen3_coder` XML** parsing of assistant text (no alternate
  JSON-in-`<tool_call>` path). Do not patch `agents/qwenAgent` for SkyRL-only
  needs unless the shared eval path itself is changing.
- Warmup env `search_news` / submit selection use `qwen_execute_news_search`,
  `qwen_optional_search_dates_from_parsed`, and `qwen_parse_warmup_submit_outcomes`
  from `agents/qwenAgent/agent.py` (same path as `QwenBasicAgent._qwen_handle_search`
  and `_qwen_handle_submit` forecast filtering + `_forecasts_within_probability_bounds`).
- Current `sitecustomize.py` shims (SkyRL **v0.1.0**; no backward-compat with older SkyRL):
  - torch **2.10+** process-group kwarg selection (`backend_options` vs `pg_options`)
  - Qwen3.5 FSDP wrap policy when `_no_split_modules` references absent vision layers
  - Qwen3.5 FSDP→vLLM weight-name mapping in `FSDPWeightExtractor.extract_weights`
  - Qwen `apply_chat_template(..., tokenize=True)` → plain token-id lists (SkyRL gym generator)
  - Transformers `RotaryEmbeddingConfigMixin.convert_rope_params_to_dict`: coerce `ignore_keys_at_rope_validation` list/tuple → set (vLLM Qwen3.5 text config vs Transformers RoPE `|=` merge)

## Logs

SkyRL logs are written under `FSIM_SKYRL_LOG_BASE/<sim_name>/<sim_name>_rXX/`.

Each run directory contains:

- `config.yaml`
- `<cluster_id>.out`
- `<cluster_id>.err`
- `<cluster_id>.log`
