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
- `scripts/run_skyrl_openforesight_search.py` leaves SkyRL’s default chat template
  config when `training.chat_template_path` is unset (HF tokenizer template;
  matches eval_r00 / eval_bs160-style configs). Custom Jinja is opt-in via
  `training.chat_template_path` only.
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
- **vLLM + Transformers RoPE:** stock vLLM’s `site-packages/vllm/transformers_utils/configs/qwen3_5.py` used to set `ignore_keys_at_rope_validation` as a **list**, which breaks current Transformers (set-style `|=` merge). If you hit `TypeError: unsupported operand type(s) for |: 'list' and 'set'`, change that value to a **set** in your venv (or upgrade vLLM once upstream ships the fix). Re-running `uv sync` can overwrite site-packages, so re-apply after upgrades if needed.

## Ray + HTCondor (cluster jobs)

`mpi_scripts/skyrl_search/run_skyrl_search_train.sh` sets up Ray for MPI/Condor execute nodes:

- **`RAY_TMPDIR` length:** Ray’s plasma/Unix sockets live under `${RAY_TMPDIR}/ray/session_.../sockets/...`. Paths must stay within the **AF_UNIX length limit (~107 bytes)**. Avoid long `mktemp` prefixes under Condor scratch (e.g. `.../ray-sess-XXXXXX/...` plus Ray’s session suffix can overflow) or you get `OSError: validate_socket_filename failed`. The wrapper prefers **`_CONDOR_SCRATCH_DIR` / `CONDOR_SCRATCH_DIR`** (node-local) when present, otherwise falls back to **`TMPDIR`**, and uses a **short** fixed subdir **`${RAY_BASE}/r`** (only set `RAY_TMPDIR` yourself if you know the path stays short).
- **`ulimit -n`:** Low defaults on some nodes caused `Failed to register worker to Raylet: ... End of file`. The wrapper raises the open-file limit (**65536**, fallback **8192**).
- **Defaults (override as needed):** `RAY_USE_MULTIPROCESSING_CPU_COUNT=1`, `RAY_DISABLE_DOCKER_CPU_WARNING=1`.
- **Debug-only Ray in `.out`:** export **`SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1`** before launch (noisy; default is off in SkyRL).

## Logs

SkyRL logs are written under `FSIM_SKYRL_LOG_BASE/<sim_name>/<sim_name>_rXX/`.

Each run directory contains:

- `config.yaml`
- `<cluster_id>.out`
- `<cluster_id>.err`
- `<cluster_id>.log`
