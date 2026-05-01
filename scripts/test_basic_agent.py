#!/usr/bin/env python
"""
Test script for BasicAgent forecasting.

Usage (Search agent with OpenRouter + vLLM matcher):
    python scripts/test_basic_agent.py \\
        --sim_name search_agent_run \\
        --provider openrouter \\
        --openrouter_model xiaomi/mimo-v2-flash:free \\
        --matching vllm \\
        --matcher /fast/rolmedo/models/qwen3-4b-it-2507 \\
        --search_db /is/cluster/fast/sgoel/forecasting/news/deduped_articles/lance/Qwen3-Embedding-8B \\
        --embedding_model /is/cluster/fast/sgoel/models/Qwen3-Embedding-8B \\
        --embedding_gpu_mem 0.4 \\
        --matcher_gpu_mem 0.3 \\
        --start_date 2024-12-25 --end_date 2024-12-27

Usage (No-search baseline):
    python scripts/test_basic_agent.py \\
        --sim_name no_search_baseline \\
        --provider openrouter \\
        --openrouter_model xiaomi/mimo-v2-flash:free \\
        --matching vllm \\
        --matcher /fast/rolmedo/models/qwen3-4b-it-2507 \\
        --matcher_gpu_mem 0.5 \\
        --start_date 2024-12-25 --end_date 2024-12-27

Usage (Multi-agent - config file):
    python scripts/test_basic_agent.py --agents_config configs/shared/agents_example.yaml --sim_name multi_agent_run
"""

import argparse
import os
import sys
import json
from datetime import datetime, date

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathing import REPO_ROOT, expand_env_tree, load_repo_env, raise_for_unresolved_env_vars

load_repo_env(REPO_ROOT)

from environment.env import SimulationEnvironment, SimForecastInterface
from environment.matcher_cache import resolve_sim_matcher_cache_path
from agents.basicAgent import BasicAgent, AgentConfig
from agents.allQAgent import AllQAgent, AllQDailyAgent
from agents.ogAgent import OgAgent
from agents.gptossAgent import GPTOSSBasicAgent, GPTOSSAllQAgent
from agents.miroAgent import MiroBasicAgent, MiroAllQAgent
from agents.qwenAgent import QwenBasicAgent, QwenAllQAgent


# Default paths
DATASET_PATH = os.getenv("FSIM_DATASET_PATH", "/is/cluster/fast/sgoel/forecasting/qs/OpenForesight/data/")
DATASET_CACHE = os.getenv("FSIM_DATASET_CACHE", "/is/cluster/fast/sgoel/forecasting/qs/cache")
MODEL_PATH = os.getenv("FSIM_DEFAULT_MODEL_PATH", "/is/cluster/fast/rolmedo/models/qwen3-4b-it-2507")
CURRENT_SIM_DIR = os.getenv("FSIM_OUTPUT_BASE", "/is/cluster/fast/sgoel/forecasting/current_sim")
MATCHER_PATH = os.getenv("FSIM_MATCHER_MODEL", "/fast/rolmedo/models/qwen3-4b-it-2507")
EMBEDDING_MODEL_PATH = os.getenv("FSIM_EMBEDDING_MODEL", "/is/cluster/fast/sgoel/models/Qwen3-Embedding-8B")


class ThreadSafeLLM:
    """Deprecated: embedding is now served via vLLM OpenAI server (see below)."""
    pass


def create_output_dir(sim_name: str, base_dir: str = CURRENT_SIM_DIR) -> str:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(base_dir, sim_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_config(output_dir: str, args: argparse.Namespace, extra: dict = None):
    """Save run configuration to config.json."""
    config = vars(args).copy()
    config['timestamp'] = datetime.now().isoformat()
    if extra:
        config.update(extra)
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Config saved to {config_path}")


def _optional_int(value):
    if value is None:
        return None
    return int(value)


def prepare_restart_directory(source_dir: str, restart_day: str, output_dir: str) -> None:
    """
    Prepare a new output directory for restarting a simulation from a specific day.
    
    Copies:
    - actions.jsonl entries with sim_date < restart_day
    - Memory snapshots with date < restart_day
    - Matcher cache (for efficiency)
    
    After this, you can use --resume on the new directory to continue.
    """
    from datetime import date
    import shutil
    
    if isinstance(restart_day, datetime):
        restart_date = restart_day.date()
    elif isinstance(restart_day, date):
        restart_date = restart_day
    else:
        restart_date = datetime.strptime(str(restart_day), "%Y-%m-%d").date()
    
    # 1. Copy actions.jsonl entries before restart_day
    src_actions = os.path.join(source_dir, "actions.jsonl")
    dst_actions = os.path.join(output_dir, "actions.jsonl")
    
    if not os.path.exists(src_actions):
        raise FileNotFoundError(f"Source actions.jsonl not found: {src_actions}")
    
    copied_count = 0
    with open(src_actions, 'r') as src, open(dst_actions, 'w') as dst:
        for line in src:
            try:
                record = json.loads(line)
                record_date_str = record.get("sim_date")
                if record_date_str:
                    record_date = date.fromisoformat(record_date_str)
                    if record_date < restart_date:
                        dst.write(line)
                        copied_count += 1
            except (json.JSONDecodeError, ValueError):
                continue
    
    print(f"  Copied {copied_count} action records (before {restart_day})")
    
    # 2. Copy memory snapshots before restart_day for each agent
    src_agents = os.path.join(source_dir, "agents")
    if os.path.exists(src_agents):
        for agent_name in os.listdir(src_agents):
            src_agent_dir = os.path.join(src_agents, agent_name)
            src_memory_dir = os.path.join(src_agent_dir, "memory")
            
            if os.path.isdir(src_memory_dir):
                dst_agent_dir = os.path.join(output_dir, "agents", agent_name)
                dst_memory_dir = os.path.join(dst_agent_dir, "memory")
                os.makedirs(dst_memory_dir, exist_ok=True)
                
                for entry in os.listdir(src_memory_dir):
                    entry_path = os.path.join(src_memory_dir, entry)
                    # Support all memory formats:
                    #   plain:      YYYY-MM-DD.txt
                    #   structured: YYYY-MM-DD.yaml
                    #   active (old): YYYY-MM-DD.yaml + memo_YYYY-MM-DD.csv
                    #   active (new): YYYY-MM-DD/ directory with mem.csv + meta.yaml
                    if os.path.isdir(entry_path):
                        # New directory format: memory/YYYY-MM-DD/
                        try:
                            dir_date = date.fromisoformat(entry)
                            if dir_date < restart_date:
                                shutil.copytree(
                                    entry_path,
                                    os.path.join(dst_memory_dir, entry)
                                )
                        except ValueError:
                            continue
                    elif entry.endswith(".txt") or entry.endswith(".yaml"):
                        try:
                            file_date = date.fromisoformat(entry.rsplit(".", 1)[0])
                            if file_date < restart_date:
                                shutil.copy(entry_path, os.path.join(dst_memory_dir, entry))
                        except ValueError:
                            continue
                    elif entry.startswith("memo_") and entry.endswith(".csv"):
                        try:
                            file_date = date.fromisoformat(entry[len("memo_"):-len(".csv")])
                            if file_date < restart_date:
                                shutil.copy(entry_path, os.path.join(dst_memory_dir, entry))
                        except ValueError:
                            continue
                            
                print(f"  Copied memory for agent: {agent_name}")

            # Also copy timing stats (and other per-day lightweight logs) so the restarted
            # run directory has a continuous history.
            src_timing = os.path.join(src_agent_dir, "timing_stats.jsonl")
            if os.path.exists(src_timing):
                dst_agent_dir = os.path.join(output_dir, "agents", agent_name)
                os.makedirs(dst_agent_dir, exist_ok=True)
                dst_timing = os.path.join(dst_agent_dir, "timing_stats.jsonl")
                copied = 0
                with open(src_timing, "r") as src, open(dst_timing, "w") as dst:
                    for line in src:
                        try:
                            rec = json.loads(line)
                            d = date.fromisoformat(str(rec.get("date")))
                            if d < restart_date:
                                dst.write(line)
                                copied += 1
                        except Exception:
                            continue
                if copied:
                    print(f"  Copied timing stats for agent: {agent_name} ({copied} lines)")
    
    # 2.5 Copy metrics history (before restart_day) if present.
    # Note: some runs have no CSV header; we filter by the first comma-delimited field.
    for metrics_name in ("daily_metrics.csv", "test_daily_metrics.csv"):
        src_metrics = os.path.join(source_dir, metrics_name)
        if not os.path.exists(src_metrics):
            continue
        dst_metrics = os.path.join(output_dir, metrics_name)
        copied = 0
        with open(src_metrics, "r") as src, open(dst_metrics, "w") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                first = line.split(",", 1)[0]
                # Optional header support
                if first.lower() == "date":
                    dst.write(line + "\n")
                    continue
                try:
                    d = date.fromisoformat(first)
                except ValueError:
                    continue
                if d < restart_date:
                    dst.write(line + "\n")
                    copied += 1
        if copied:
            print(f"  Copied {copied} rows from {metrics_name} (before {restart_day})")

    # 3. Copy matcher cache if exists
    src_cache = os.path.join(source_dir, "matcher_cache.json")
    if os.path.exists(src_cache):
        shutil.copy(src_cache, output_dir)
        print(f"  Copied matcher cache")
    
    # 4. Copy config.json as reference
    src_config = os.path.join(source_dir, "config.json")
    if os.path.exists(src_config):
        dst_config = os.path.join(output_dir, "source_config.json")
        shutil.copy(src_config, dst_config)


def load_agents_config(config_path: str) -> dict:
    """Load agents configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for config files. Install with: pip install pyyaml")
    
    with open(config_path, 'r') as f:
        config = expand_env_tree(yaml.safe_load(f))
    raise_for_unresolved_env_vars(config, f"agents config {config_path}")
    return config


def create_inference_provider(provider: str, model: str, args, openrouter_provider_order=None):
    """Create an inference provider instance."""
    if provider == "vllm":
        from inference.vllm import VLLMInference
        rope_scaling = getattr(args, "rope_scaling", None)
        agent_max_model_len = getattr(args, "agent_max_model_len", None)
        if agent_max_model_len is None:
            agent_max_model_len = args.max_model_len
        tool_call_parser = _resolve_vllm_tool_call_parser(model, args)
        return VLLMInference(
            model,
            max_model_len=agent_max_model_len,
            gpu_memory_utilization=getattr(args, "vllm_gpu_mem", 0.3),
            timeout=getattr(args, "vllm_request_timeout", 120.0),
            max_num_seqs=getattr(args, "vllm_max_num_seqs", 8),
            tensor_parallel_size=getattr(args, "vllm_tensor_parallel_size", 1),
            data_parallel_size=getattr(args, "vllm_data_parallel_size", 1),
            pipeline_parallel_size=getattr(args, "vllm_pipeline_parallel_size", 1),
            enable_expert_parallel=getattr(args, "vllm_enable_expert_parallel", False),
            all2all_backend=getattr(args, "vllm_all2all_backend", None),
            startup_timeout=getattr(args, "vllm_startup_timeout", 300.0),
            rope_scaling=rope_scaling,
            enable_tools=getattr(args, "vllm_enable_tools", False),
            tool_call_parser=tool_call_parser,
            tool_parser_plugin=getattr(args, "vllm_tool_parser_plugin", None),
            enable_prefix_caching=getattr(args, "vllm_enable_prefix_caching", True),
            cuda_visible_devices=getattr(args, "agent_cuda_visible_devices", None),
            language_model_only=getattr(args, "language_model_only", False),
        )
    elif provider == "openrouter":
        from inference.openrouter import OpenRouterInference
        kwargs = {}
        if openrouter_provider_order:
            kwargs["provider"] = {"order": openrouter_provider_order, "allow_fallbacks": True}
        return OpenRouterInference(model, **kwargs)
    elif provider == "azure":
        from inference.azure_openai import AzureOpenAIInference
        return AzureOpenAIInference(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_model_short_name(model: str) -> str:
    """Extract short model name from full model path/ID."""
    # Handle paths like /path/to/model-name
    name = os.path.basename(model)
    # Handle OpenRouter IDs like vendor/model-name:variant
    if '/' in model:
        name = model.split('/')[-1]
    # Remove version tags
    name = name.split(':')[0]
    return name


def _is_qwen_model(model: str) -> bool:
    model_l = str(model or "").lower()
    return ("qwen" in model_l) and ("gpt-oss" not in model_l)


def _is_qwen35_model(model: str) -> bool:
    model_l = str(model or "").lower()
    return ("qwen3.5" in model_l) or ("qwen3_5" in model_l)


def _is_miro_model(model: str) -> bool:
    return "mirothinker" in str(model or "").lower()


def _resolve_vllm_tool_call_parser(model: str, args) -> str | None:
    parser = getattr(args, "vllm_tool_call_parser", None)
    if isinstance(parser, str):
        parser = parser.strip()
    if parser:
        return parser

    if not bool(getattr(args, "vllm_enable_tools", False)):
        return None

    if _is_qwen35_model(model):
        return "qwen3_coder"

    if _is_qwen_model(model):
        return "hermes"

    if _is_miro_model(model):
        return None

    return "openai"


def create_agents_from_config(config: dict, args, output_dir: str, search_tool=None) -> list:
    """Create agent instances from config dict."""
    from inference.openrouter import GlobalRateLimiter
    def _as_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("1", "true", "yes", "on"):
                return True
            if s in ("0", "false", "no", "off"):
                return False
        return bool(v)
    
    defaults = config.get('defaults', {})
    agents_list = config.get('agents', [])

    if not agents_list:
        raise ValueError("No agents defined in config file")

    print(f"  Found {len(agents_list)} agents in config", flush=True)
    
    # If any agent uses OpenRouter, size the shared HTTP connection pool to match
    # the run's parallelism (max warmup_parallelism across agents).
    if any((a.get('provider', defaults.get('provider', 'openrouter')) == 'openrouter') for a in agents_list):
        try:
            from inference.openrouter import configure_http_pool
            desired_pool = int(getattr(args, 'warmup_parallelism', 20))
            desired_pool = max(desired_pool, int(defaults.get('warmup_parallelism', desired_pool) or desired_pool))
            for a in agents_list:
                if a.get('provider', defaults.get('provider', 'openrouter')) != 'openrouter':
                    continue
                wp = a.get('warmup_parallelism', defaults.get('warmup_parallelism', desired_pool))
                if wp is not None:
                    desired_pool = max(desired_pool, int(wp))
            configure_http_pool(pool_maxsize=desired_pool, pool_connections=desired_pool)
            print(f"  OpenRouter HTTP pool: connections=maxsize={desired_pool}", flush=True)
        except Exception as e:
            print(f"  Warning: failed to configure OpenRouter HTTP pool: {e}", flush=True)

    # Configure global rate limiter
    GlobalRateLimiter.configure(args.rate_limit)
    print(f"  Rate limiter configured: {args.rate_limit} req/s", flush=True)
    
    agents = []
    agent_counts = {}  # Track counts per scaffold+model combo for unique IDs
    
    for i, agent_def in enumerate(agents_list):
        # Merge defaults with agent-specific settings
        provider = agent_def.get('provider', defaults.get('provider', 'openrouter'))
        scaffold = agent_def.get('scaffold', defaults.get('scaffold', 'basic'))
        model = agent_def.get('model')
        
        if not model:
            raise ValueError("Each agent must specify a 'model'")
        
        print(f"  [{i+1}/{len(agents_list)}] Creating {scaffold}:{provider}:{model}...", flush=True)

        # Claude Code agent doesn't need an inference provider — skip the LLM setup
        # and jump straight to agent creation.
        if scaffold == 'minimalHarness':
            from agents.minimalHarnessAgent.agent import MinimalHarnessAgent, MinimalHarnessConfig
            agent_id = f"minimalHarness_{model.replace('/', '_').replace('.', '')}_{i+1:03d}"
            cc_search_cutoff = agent_def.get('search_cutoff_days', defaults.get('search_cutoff_days', getattr(args, 'search_cutoff_days', 0)))
            harness_backend = agent_def.get('harness_backend', defaults.get('harness_backend', 'claude_code'))
            openrouter_api_key = ''
            if harness_backend == 'opencode':
                try:
                    from configs.openrouter_api_key import OPENROUTER_API_KEY as _OR_KEY
                    openrouter_api_key = _OR_KEY
                except ImportError:
                    openrouter_api_key = os.environ.get('OPENROUTER_API_KEY', '')

            anthropic_base_url = agent_def.get('anthropic_base_url', defaults.get('anthropic_base_url', ''))
            anthropic_auth_token = agent_def.get('anthropic_auth_token', defaults.get('anthropic_auth_token', ''))
            if anthropic_base_url and not anthropic_auth_token:
                if 'z.ai' in anthropic_base_url:
                    try:
                        from configs.glm_api_key import GLM_API_KEY as _GLM_KEY
                        anthropic_auth_token = _GLM_KEY
                    except ImportError:
                        anthropic_auth_token = os.environ.get('GLM_API_KEY', '') or os.environ.get('ANTHROPIC_AUTH_TOKEN', '')
                elif 'deepseek.com' in anthropic_base_url:
                    try:
                        from configs.deepseek_api_key import DEEPSEEK_API_KEY as _DS_KEY
                        anthropic_auth_token = _DS_KEY
                    except ImportError:
                        anthropic_auth_token = os.environ.get('DEEPSEEK_API_KEY', '') or os.environ.get('ANTHROPIC_AUTH_TOKEN', '')
                else:
                    anthropic_auth_token = os.environ.get('ANTHROPIC_AUTH_TOKEN', '')
            cc_config = MinimalHarnessConfig(
                model=model,
                search_db=getattr(args, 'search_db', '') or '',
                embedding_model=getattr(args, 'embedding_model', '') or '',
                search_type=agent_def.get('search_type', defaults.get('search_type', 'hybrid')),
                search_cutoff_days=cc_search_cutoff,
                articles_base=config.get('articles_base', os.environ.get('FSIM_ARTICLES_BASE', '')),
                start_date=args.sim_start_date,
                end_date=getattr(args, 'end_date', None),
                timeout_seconds=int(agent_def.get('timeout_seconds', defaults.get('timeout_seconds', 7200))),
                max_budget_usd=agent_def.get('max_budget_usd', defaults.get('max_budget_usd')),
                claude_code_path=agent_def.get('claude_code_path', defaults.get('claude_code_path', 'claude')),
                harness_backend=harness_backend,
                opencode_path=agent_def.get('opencode_path', defaults.get('opencode_path', 'opencode')),
                codex_path=agent_def.get('codex_path', defaults.get('codex_path', 'codex')),
                reasoning_effort=agent_def.get('reasoning_effort', defaults.get('reasoning_effort', 'high')),
                codex_resume=bool(agent_def.get('codex_resume', defaults.get('codex_resume', False))),
                prompt_mode=str(agent_def.get('prompt_mode', defaults.get('prompt_mode', 'default'))),
                max_outcomes_per_question=int(agent_def.get(
                    'max_outcomes_per_question',
                    defaults.get('max_outcomes_per_question', config.get('max_outcomes_per_question', 5)),
                )),
                timegap_days=int(agent_def.get(
                    'timegap_days',
                    defaults.get('timegap_days', config.get('timegap_days', 1)),
                )),
                claude_code_resume=bool(
                    agent_def.get('claude_code_resume', defaults.get('claude_code_resume', False))
                    or (args.resume and str(agent_def.get('prompt_mode', defaults.get('prompt_mode', 'default'))) != 'active_memory')
                ),
                openrouter_api_key=openrouter_api_key,
                anthropic_base_url=anthropic_base_url,
                anthropic_auth_token=anthropic_auth_token,
                extra_flags=list(agent_def.get('extra_flags', defaults.get('extra_flags', []))),
                sandbox=bool(agent_def.get('sandbox', defaults.get('sandbox', False))),
                network_isolation=bool(agent_def.get('network_isolation', defaults.get('network_isolation', False))),
                egress_allowlist=list(agent_def.get('egress_allowlist', defaults.get('egress_allowlist', []))),
            )
            agent = MinimalHarnessAgent(
                agent_id=agent_id,
                config=cc_config,
                search_tool=search_tool,
                agent_dir=os.path.join(output_dir, 'agents', agent_id),
                articles_base=config.get('articles_base', os.environ.get('FSIM_ARTICLES_BASE', '')),
            )
            agents.append(agent)
            print(f"  Created agent: {agent_id} (Minimal Harness) [Scaffold: {scaffold}]")
            continue

        # Create inference provider
        openrouter_provider_order = agent_def.get('openrouter_provider_order', defaults.get('openrouter_provider_order', None))
        inference_provider = create_inference_provider(provider, model, args, openrouter_provider_order=openrouter_provider_order)
        
        # Build agent config with merged settings
        max_actions = _optional_int(agent_def.get('max_actions', defaults.get('max_actions', args.max_actions)))
        warmup_max_actions = _optional_int(
            agent_def.get('warmup_max_actions', defaults.get('warmup_max_actions', getattr(args, 'warmup_max_actions', None)))
        )
        max_total_tokens = _optional_int(
            agent_def.get('max_total_tokens', defaults.get('max_total_tokens', getattr(args, 'max_total_tokens', None)))
        )
        warmup_max_total_tokens = _optional_int(
            agent_def.get(
                'warmup_max_total_tokens',
                defaults.get('warmup_max_total_tokens', getattr(args, 'warmup_max_total_tokens', None))
            )
        )
        submit_reserve_tokens = _optional_int(
            agent_def.get(
                'submit_reserve_tokens',
                defaults.get('submit_reserve_tokens', getattr(args, 'submit_reserve_tokens', 8192))
            )
        )
        warmup_submit_reserve_tokens = _optional_int(
            agent_def.get(
                'warmup_submit_reserve_tokens',
                defaults.get('warmup_submit_reserve_tokens', getattr(args, 'warmup_submit_reserve_tokens', None))
            )
        )
        force_submit_threshold_tokens = _optional_int(
            agent_def.get(
                'force_submit_threshold_tokens',
                defaults.get(
                    'force_submit_threshold_tokens',
                    getattr(args, 'force_submit_threshold_tokens', 16384),
                )
            )
        )
        warmup_force_submit_threshold_tokens = _optional_int(
            agent_def.get(
                'warmup_force_submit_threshold_tokens',
                defaults.get(
                    'warmup_force_submit_threshold_tokens',
                    getattr(args, 'warmup_force_submit_threshold_tokens', None),
                )
            )
        )
        warmup_parallelism = agent_def.get('warmup_parallelism', defaults.get('warmup_parallelism', getattr(args, 'warmup_parallelism', 20)))
        max_outcomes_per_question = int(
            agent_def.get(
                'max_outcomes_per_question',
                defaults.get('max_outcomes_per_question', 5),
            )
        )
        max_retries = agent_def.get('max_retries', defaults.get('max_retries', args.max_retries))
        temperature = agent_def.get('temperature', defaults.get('temperature', args.temperature))
        top_p = agent_def.get('top_p', defaults.get('top_p', getattr(args, 'top_p', None)))
        top_k = agent_def.get('top_k', defaults.get('top_k', getattr(args, 'top_k', None)))
        repetition_penalty = agent_def.get(
            'repetition_penalty',
            defaults.get('repetition_penalty', getattr(args, 'repetition_penalty', None))
        )
        max_tokens = agent_def.get('max_tokens', defaults.get('max_tokens', args.max_tokens))
        reasoning = agent_def.get('reasoning', defaults.get('reasoning', None))
        search_cutoff_days = agent_def.get('search_cutoff_days', defaults.get('search_cutoff_days', getattr(args, 'search_cutoff_days', 0)))
        resolution_guard = _optional_int(
            agent_def.get('resolution_guard', defaults.get('resolution_guard', getattr(args, 'resolution_guard', None)))
        )
        enable_memory = agent_def.get('enable_memory', defaults.get('enable_memory', True))
        memory_format = agent_def.get('memory_format', defaults.get('memory_format', 'structured'))
        memory_max_entries = int(agent_def.get('memory_max_entries', defaults.get('memory_max_entries', 500)))
        memory_update_max_total_tokens = int(
            agent_def.get('memory_update_max_total_tokens', defaults.get('memory_update_max_total_tokens', 50000))
        )
        singleans = agent_def.get('singleans', defaults.get('singleans', False))
        tool_result_keep_last = _optional_int(
            agent_def.get(
                'tool_result_keep_last',
                defaults.get('tool_result_keep_last', getattr(args, 'tool_result_keep_last', -1))
            )
        )
        gptoss_prompt_mode = agent_def.get(
            'gptoss_prompt_mode',
            defaults.get('gptoss_prompt_mode', getattr(args, 'gptoss_prompt_mode', 'instructions'))
        )
        gptoss_reasoning_effort = str(agent_def.get(
            'gptoss_reasoning_effort',
            defaults.get('gptoss_reasoning_effort', getattr(args, 'gptoss_reasoning_effort', 'medium'))
        )).lower()
        gptoss_include_reasoning = _as_bool(agent_def.get(
            'gptoss_include_reasoning',
            defaults.get('gptoss_include_reasoning', getattr(args, 'gptoss_include_reasoning', True))
        ))
        gptoss_responses_max_retries = int(agent_def.get(
            'gptoss_responses_max_retries',
            defaults.get('gptoss_responses_max_retries', getattr(args, 'gptoss_responses_max_retries', 3))
        ))
        gptoss_retry_backoff_base_s = float(agent_def.get(
            'gptoss_retry_backoff_base_s',
            defaults.get('gptoss_retry_backoff_base_s', getattr(args, 'gptoss_retry_backoff_base_s', 1.0))
        ))
        gptoss_retry_backoff_max_s = float(agent_def.get(
            'gptoss_retry_backoff_max_s',
            defaults.get('gptoss_retry_backoff_max_s', getattr(args, 'gptoss_retry_backoff_max_s', 16.0))
        ))
        
        # Generate unique agent ID: scaffold_modelname_NNN
        model_short = get_model_short_name(model)
        base_id = f"{scaffold}_{model_short}"
        count = agent_counts.get(base_id, 0) + 1
        agent_counts[base_id] = count
        agent_id = f"{base_id}_{count:03d}"
        
        # Create agent directory for logging
        agent_dir = os.path.join(output_dir, "agents", agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        
        agent_config = AgentConfig(
            max_actions=max_actions,
            warmup_max_actions=warmup_max_actions,
            max_total_tokens=max_total_tokens,
            warmup_max_total_tokens=warmup_max_total_tokens,
            submit_reserve_tokens=submit_reserve_tokens,
            warmup_submit_reserve_tokens=warmup_submit_reserve_tokens,
            force_submit_threshold_tokens=force_submit_threshold_tokens,
            warmup_force_submit_threshold_tokens=warmup_force_submit_threshold_tokens,
            warmup_parallelism=warmup_parallelism,
            max_submit_retries=max_retries,
            max_outcomes_per_question=max_outcomes_per_question,
            memory_dir=agent_dir,  # Per-agent memory directory
            enable_memory=enable_memory,
            memory_format=memory_format,
            memory_max_entries=memory_max_entries,
            memory_update_max_total_tokens=memory_update_max_total_tokens,
            append_model_output_logs=bool(getattr(args, "resume", None)),
            singleans=singleans,
            tool_result_keep_last=tool_result_keep_last,
            sampling_params={
                'temperature': temperature,
                'max_tokens': max_tokens,
                **({'top_p': float(top_p)} if top_p is not None else {}),
                **({'top_k': int(top_k)} if top_k is not None else {}),
                **({'repetition_penalty': float(repetition_penalty)} if repetition_penalty is not None else {}),
                **({'reasoning': reasoning} if reasoning is not None else {}),
            },
            search_cutoff_days=search_cutoff_days,
            resolution_guard=resolution_guard,
            timegap_days=getattr(args, 'timegap_days', 1),
            single_agent_mode=(len(agents_list) == 1),  # Adjust prompt for single-agent runs
            gptoss_prompt_mode=gptoss_prompt_mode,
            gptoss_reasoning_effort=gptoss_reasoning_effort,
            gptoss_include_reasoning=gptoss_include_reasoning,
            gptoss_responses_max_retries=gptoss_responses_max_retries,
            gptoss_retry_backoff_base_s=gptoss_retry_backoff_base_s,
            gptoss_retry_backoff_max_s=gptoss_retry_backoff_max_s,
        )

        # GPT-OSS Harmony tool calling: only when using vLLM + enable_tools + gpt-oss weights.
        if scaffold == 'basic':
            agent_cls = BasicAgent
            agent = agent_cls(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model,
                search_tool=search_tool
            )
        elif scaffold in ('allQ', 'allq'):
            agent_cls = AllQAgent
            agent = agent_cls(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model,
                search_tool=search_tool,
                start_date=args.sim_start_date  # Passed from create_agents call
            )
        elif scaffold == 'qwenbasic':
            agent = QwenBasicAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model,
                search_tool=search_tool
            )
        elif scaffold == 'qwenallq':
            agent = QwenAllQAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model,
                search_tool=search_tool,
                start_date=args.sim_start_date
            )
        elif scaffold == 'mirobasic':
            agent = MiroBasicAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model,
                search_tool=search_tool
            )
        elif scaffold == 'miroallq':
            agent = MiroAllQAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model,
                search_tool=search_tool,
                start_date=args.sim_start_date
            )
        elif scaffold == 'gptossbasic':
            agent = GPTOSSBasicAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model,
                search_tool=search_tool
            )
        elif scaffold == 'gptossallq':
            agent = GPTOSSAllQAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model,
                search_tool=search_tool,
                start_date=args.sim_start_date
            )
        elif scaffold == 'allqd':
            agent = AllQDailyAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model,
                search_tool=search_tool,
                start_date=args.sim_start_date
            )
        elif scaffold in ('og', 'ogagent', 'ogAgent'):
            agent = OgAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model,
                search_tool=search_tool,
                start_date=args.sim_start_date
            )
        else:
            raise ValueError(
                f"Unknown scaffold: {scaffold}. Only 'basic', 'allQ', 'allqd', 'og', "
                "'qwenbasic', 'qwenallq', 'mirobasic', 'miroallq', 'gptossbasic', "
                "'gptossallq', and 'minimalHarness' are supported."
            )
        
        agents.append(agent)
        print(f"  Created agent: {agent_id} ({provider}:{model}) [Scaffold: {scaffold}]")
    
    return agents


def main():
    parser = argparse.ArgumentParser(description="Test BasicAgent Forecasting")
    
    # Simulation settings
    parser.add_argument("--sim_name", default="debug_sim",
                       help="Name for this simulation run")
    parser.add_argument("--start_date", default="2024-12-25",
                       help="First resolution date (YYYY-MM-DD) - questions resolving from this date")
    parser.add_argument("--end_date", default="2024-12-27",
                       help="Last resolution date (YYYY-MM-DD) - questions resolving until this date")
    # Optional: decouple question-resolution filtering window from the simulation window.
    # This is useful when you want to run only Day 0 (start_date=end_date) but still
    # warm up over a broader set of questions.
    parser.add_argument("--resolution_start", default=None,
                       help="Resolution window start (YYYY-MM-DD). Defaults to --start_date.")
    parser.add_argument("--resolution_end", default=None,
                       help="Resolution window end (YYYY-MM-DD). Defaults to --end_date.")
    parser.add_argument("--lookback_days", type=int, default=7,
                       help="Days before first resolution to start simulation (default 7)")
    parser.add_argument("--resolution_guard", type=int, default=None,
                       help="Warmup-only per-question current/search date: resolution_date - resolution_guard days; replaces the shared warmup sim-day date when set")
    
    # Data paths
    # Data paths
    parser.add_argument("--dataset", default="openforesight",
                       choices=["openforesight", "metaculus_binary", "metaculus_mcq"],
                       help="Dataset source to use")
    parser.add_argument("--dataset_path", default=DATASET_PATH,
                       help="Path to local dataset (for openforesight)")
    parser.add_argument("--dataset_cache", 
                       default=DATASET_CACHE,
                       help="Cache directory for fetched datasets")
    parser.add_argument("--output_base", default=CURRENT_SIM_DIR,
                       help="Base directory for simulation outputs")
    parser.add_argument("--split", choices=["train", "test", "validation"], default="train",
                       help="Dataset split to use (default: train)")
    parser.add_argument("--prepend_train_resolution_start", default=None,
                       help="Optional OpenForesight train prelude start date (YYYY-MM-DD).")
    parser.add_argument("--prepend_train_resolution_end", default=None,
                       help="Optional OpenForesight train prelude end date (YYYY-MM-DD).")
    parser.add_argument("--subsample_per_month", type=int, default=1000,
                       help="Per-month question cap for the optional prepended train window.")
    parser.add_argument("--timegap_days", type=int, default=1,
                       help="Wake the model every N days instead of daily (default 1).")
    
    # Multi-agent config
    parser.add_argument("--agents_config", default=None,
                       help="Path to YAML config file for multi-agent runs")
    parser.add_argument("--rate_limit", type=float, default=32.0,
                       help="OpenRouter rate limit (requests/second, default 32)")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Run agents in parallel (default: True)")
    parser.add_argument("--no_parallel", action="store_true",
                       help="Disable parallel agent execution")
    # Single-agent settings (used when no config file)
    parser.add_argument("--scaffold", choices=["basic", "allQ", "allq", "allqd", "og", "qwenbasic", "qwenallq", "mirobasic", "miroallq", "gptossbasic", "gptossallq"], default="basic",
                       help="Agent scaffold to use (default: basic). Native Qwen/Miro/GPT-OSS variants must be selected explicitly.")
    parser.add_argument("--provider", choices=["vllm", "openrouter"], default="openrouter",
                       help="Inference provider: 'vllm' (local) or 'openrouter' (API)")
    parser.add_argument("--model_path", default=MODEL_PATH,
                       help="Path to model for VLLM inference")
    parser.add_argument("--openrouter_model", default=None,
                       help="Model ID for OpenRouter (e.g., 'xiaomi/mimo-v2-flash:free'). "
                            "Ignored when agents are configured via YAML.")
    parser.add_argument("--max_model_len", type=int, default=32768,
                       help="Max context length for VLLM (default 32768)")
    parser.add_argument("--agent_max_model_len", type=int, default=None,
                       help="Max context length for agent vLLM only (defaults to --max_model_len)")
    parser.add_argument("--matcher_max_model_len", type=int, default=None,
                       help="Max context length for matcher vLLM only (defaults to --max_model_len)")
    parser.add_argument("--embedding_max_model_len", type=int, default=None,
                       help="Max context length for embedding vLLM only (defaults to --max_model_len)")
    parser.add_argument("--no_inference", action="store_true",
                       help="Run without LLM (for testing setup)")
    
    # Agent settings
    parser.add_argument("--max_actions", type=int, default=None,
                       help="Optional action budget per day (queries + submissions)")
    parser.add_argument("--warmup_max_actions", type=int, default=None,
                       help="Optional action budget per question during warmup phase (AllQAgent only)")
    parser.add_argument("--max_total_tokens", type=int, default=None,
                       help="Optional context-window budget per day/question loop (tracks current prompt occupancy, not cumulative spend)")
    parser.add_argument("--warmup_max_total_tokens", type=int, default=None,
                       help="Optional context-window budget per warmup question loop")
    parser.add_argument("--submit_reserve_tokens", type=int, default=8192,
                       help="Reserved context-window headroom to keep available for a final submit (default 8192)")
    parser.add_argument("--warmup_submit_reserve_tokens", type=int, default=None,
                       help="Optional warmup-specific submit reserve tokens")
    parser.add_argument("--force_submit_threshold_tokens", type=int, default=16384,
                       help="Force-submit once remaining context budget is at or below this threshold (default 16384)")
    parser.add_argument("--warmup_force_submit_threshold_tokens", type=int, default=None,
                       help="Optional warmup-specific force-submit threshold tokens")
    parser.add_argument("--warmup_parallelism", type=int, default=20,
                       help="Number of concurrent threads for warmup phase (default 20)")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Max retry attempts for forecast parsing")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Max output tokens to generate per LLM call")
    parser.add_argument("--gptoss_prompt_mode", choices=["instructions", "first_user"], default="instructions",
                       help="GPT-OSS Responses prompt placement mode")
    parser.add_argument("--gptoss_reasoning_effort", choices=["low", "medium", "high"], default="medium",
                       help="GPT-OSS reasoning effort for Responses API")
    parser.add_argument("--gptoss_include_reasoning", action="store_true", default=True,
                       help="Request reasoning content in GPT-OSS Responses output")
    parser.add_argument("--no_gptoss_include_reasoning", action="store_true",
                       help="Disable reasoning content in GPT-OSS Responses output")
    parser.add_argument("--gptoss_responses_max_retries", type=int, default=3,
                       help="Max retries for GPT-OSS /v1/responses calls (default 3)")
    parser.add_argument("--gptoss_retry_backoff_base_s", type=float, default=1.0,
                       help="Base backoff seconds for GPT-OSS responses retries (default 1.0)")
    parser.add_argument("--gptoss_retry_backoff_max_s", type=float, default=16.0,
                       help="Max backoff seconds for GPT-OSS responses retries (default 16.0)")

    # vLLM runtime settings (when provider == vllm in config)
    parser.add_argument("--vllm_gpu_mem", type=float, default=0.3,
                       help="GPU memory fraction for vLLM agent models (0.0-1.0, default 0.3)")
    parser.add_argument("--vllm_max_num_seqs", type=int, default=8,
                       help="vLLM max concurrent sequences (default 8)")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size for vLLM agent models (default 1)")
    parser.add_argument("--vllm_data_parallel_size", type=int, default=1,
                       help="Data parallel size for vLLM agent models (default 1)")
    parser.add_argument("--vllm_pipeline_parallel_size", type=int, default=1,
                       help="Pipeline parallel size for vLLM agent models (default 1)")
    parser.add_argument("--vllm_enable_expert_parallel", action="store_true", default=False,
                       help="Enable vLLM expert parallel mode for MoE models")
    parser.add_argument("--vllm_all2all_backend", default=None,
                       help="Optional vLLM all-to-all backend (e.g. allgather_reducescatter, deepep_high_throughput)")
    parser.add_argument("--vllm_startup_timeout", type=float, default=300.0,
                       help="vLLM server startup timeout in seconds (default 300)")
    parser.add_argument("--vllm_request_timeout", type=float, default=120.0,
                       help="vLLM request timeout in seconds for chat/embeddings calls (default 120)")
    parser.add_argument("--vllm_enable_tools", action="store_true", default=False,
                       help="Start vLLM servers with tool-calling enabled (required for native Qwen/Miro tool calls)")
    parser.add_argument("--vllm_enable_prefix_caching", action=argparse.BooleanOptionalAction, default=True,
                       help="Enable vLLM automatic prefix caching for chat/matcher servers (default: on)")
    parser.add_argument("--vllm_tool_call_parser", default=None,
                       help="vLLM tool parser name (e.g. qwen3_coder, openai). Auto-detected when omitted.")
    parser.add_argument("--vllm_tool_parser_plugin", default=None,
                       help="Optional --tool-parser-plugin value for custom parsers")
    parser.add_argument("--language_model_only", action="store_true", default=False,
                       help="Pass --language-model-only to vLLM (skip vision encoder for multimodal checkpoints)")

    # GPU pinning for local multi-GPU runs.
    # If you set aux_cuda_visible_devices=1 and agent_cuda_visible_devices=0, then:
    # - matcher vLLM subprocess + in-process embedder will use GPU 1 (parent process visibility)
    # - agent vLLM subprocesses will be pinned to GPU 0 (per-subprocess override)
    parser.add_argument("--aux_cuda_visible_devices", default=None,
                       help="CUDA_VISIBLE_DEVICES for matcher + embedder (parent process). E.g. '1'")
    parser.add_argument("--agent_cuda_visible_devices", default=None,
                       help="CUDA_VISIBLE_DEVICES for agent vLLM subprocesses. E.g. '0'")
    
    # Answer matching settings
    parser.add_argument("--matching", choices=["exact", "openrouter", "vllm"], default="vllm",
                       help="Answer matching mode: 'exact', 'openrouter', or 'vllm'")
    parser.add_argument("--matcher", default=MATCHER_PATH,
                       help="Matcher model: OpenRouter model ID or VLLM model path")
    
    # Search settings
    parser.add_argument("--search_db", default="",
                       help="Path to LanceDB directory for article search (optional)")
    parser.add_argument("--embedding_model", default=EMBEDDING_MODEL_PATH,
                       help="Path to embedding model for semantic search")
    parser.add_argument("--embedding_gpu_mem", type=float, default=0.4,
                       help="GPU memory fraction for embedding model (0.0-1.0, default 0.4)")
    parser.add_argument("--matcher_gpu_mem", type=float, default=0.3,
                       help="GPU memory fraction for matcher model (0.0-1.0, default 0.3)")
    parser.add_argument("--search_cutoff_days", type=int, default=0,
                       help="Days before current date to cutoff search results (default 0)")
    
    # Config file
    parser.add_argument("--config", help="Path to YAML configuration file to load arguments from")
    
    # Resume
    parser.add_argument("--resume", help="Directory of a previous run to resume")
    parser.add_argument("--rescore", action="store_true", help="Recalculate metrics from history before resuming")
    
    # Restart from specific day
    parser.add_argument("--restart_from", help="Directory of a previous run to restart from")
    parser.add_argument("--restart_from_day", help="Day to restart from (YYYY-MM-DD). Simulation re-runs from this day.")
    
    # Parse CLI args first to get config path if provided
    args, remaining_argv = parser.parse_known_args()
    config = None
    # Keys explicitly set by the YAML config file (used during restart to avoid
    # source-config leakage — e.g. the new YAML intentionally omitting max_actions
    # should NOT be backfilled from the source run's config).
    _yaml_config_keys: set = set()

    # Load config if provided
    if args.config:
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            import yaml
            config = expand_env_tree(yaml.safe_load(f))
        raise_for_unresolved_env_vars(config, f"run config {args.config}")
        _yaml_config_keys = set(config.keys())
        # Budget/agent settings live under 'defaults' in the YAML but are flattened
        # to top-level keys in config.json. Include them so the source config from a
        # restart doesn't overwrite what the new YAML intentionally sets.
        if isinstance(config.get('defaults'), dict):
            _yaml_config_keys |= set(config['defaults'].keys())

        # Set defaults from config
        parser.set_defaults(**config)

        # Override with any explicitly provided CLI args
        # This requires re-parsing to ensure CLI args take precedence over config defaults
        args = parser.parse_args(args=remaining_argv + sys.argv[1:])
    elif args.resume:
        # Resume mode: Load config from the resume directory
        config_path = os.path.join(args.resume, 'config.json')
        if not os.path.exists(config_path):
            print(f"Error: Config file not found in resume directory: {config_path}")
            sys.exit(1)
            
        print(f"Resuming simulation from: {args.resume}")
        print(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            config = expand_env_tree(json.load(f))
        raise_for_unresolved_env_vars(config, f"resume config {config_path}")
            
        # Set defaults from config
        parser.set_defaults(**config)
        
        # Override with any explicitly provided CLI args
        args = parser.parse_args(args=remaining_argv + sys.argv[1:])
    else:
        args = parser.parse_args()

    # YAML parsing can materialize date-like values as datetime/date objects when
    # overrides are passed via submit_sim.py --set. Normalize back to ISO strings.
    def _normalize_date_like(v):
        if isinstance(v, datetime):
            return v.date().isoformat()
        if isinstance(v, date):
            return v.isoformat()
        return v

    for key in (
        "start_date",
        "end_date",
        "resolution_start",
        "resolution_end",
        "restart_from_day",
        "prepend_train_resolution_start",
        "prepend_train_resolution_end",
    ):
        if hasattr(args, key):
            setattr(args, key, _normalize_date_like(getattr(args, key)))

    def _config_uses_vllm(cfg: dict | None, parsed_args: argparse.Namespace) -> bool:
        # Legacy single-agent mode.
        if getattr(parsed_args, "provider", None) == "vllm":
            return True
        if not cfg:
            return False

        defaults = cfg.get("defaults", {}) or {}
        default_provider = defaults.get("provider")
        for a in (cfg.get("agents") or []):
            p = a.get("provider", default_provider)
            if p == "vllm":
                return True
        return default_provider == "vllm"

    # Apply GPU pinning BEFORE starting matcher server or loading embedding model.
    #
    # Important: on clusters, HTCondor/Slurm often set CUDA_VISIBLE_DEVICES to an
    # allocated subset (sometimes UUIDs). We interpret numeric specs like "0"/"1"
    # as indices into that *current* visible list, and rewrite them to the actual
    # entry (index or UUID) to avoid pinning to non-allocated GPUs.
    orig_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    base_visible = None
    if orig_visible:
        base_visible = [x.strip() for x in orig_visible.split(",") if x.strip()]

    def _map_cuda_spec(spec: str | None) -> str | None:
        if spec is None:
            return None
        s = str(spec).strip()
        if not s:
            return None
        if base_visible is not None:
            # Support comma-separated index lists like "0,1" for tensor-parallel runs.
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if parts and all(p.isdigit() for p in parts):
                mapped = []
                for p in parts:
                    idx = int(p)
                    if not (0 <= idx < len(base_visible)):
                        return None
                    mapped.append(base_visible[idx])
                return ",".join(mapped)
        return s

    aux_mapped = _map_cuda_spec(getattr(args, "aux_cuda_visible_devices", None))
    agent_mapped = _map_cuda_spec(getattr(args, "agent_cuda_visible_devices", None))

    if getattr(args, "aux_cuda_visible_devices", None) is not None and aux_mapped is None:
        print(
            f"Warning: aux_cuda_visible_devices={args.aux_cuda_visible_devices!r} not compatible with "
            f"CUDA_VISIBLE_DEVICES={orig_visible!r}; ignoring aux pinning."
        )
    if getattr(args, "agent_cuda_visible_devices", None) is not None and agent_mapped is None:
        print(
            f"Warning: agent_cuda_visible_devices={args.agent_cuda_visible_devices!r} not compatible with "
            f"CUDA_VISIBLE_DEVICES={orig_visible!r}; ignoring agent pinning."
        )

    args.aux_cuda_visible_devices = aux_mapped
    args.agent_cuda_visible_devices = agent_mapped

    # Only apply agent pinning if we will actually start vLLM agent servers.
    if args.agent_cuda_visible_devices and not _config_uses_vllm(config, args):
        print(
            "Warning: agent_cuda_visible_devices was set but no agent is configured with provider=vllm; "
            "ignoring agent pinning."
        )
        args.agent_cuda_visible_devices = None

    # This ensures the in-process embedding model lands on aux GPU deterministically.
    if args.aux_cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.aux_cuda_visible_devices)
    
    # Handle parallel flag
    if args.no_parallel:
        args.parallel = False
    if args.no_gptoss_include_reasoning:
        args.gptoss_include_reasoning = False
    
    # Parse dates
    # start_date/end_date define the SIMULATION window (via sim_start + sim_end).
    # resolution_start/resolution_end define which questions are loaded (by resolution date).
    sim_resolution_start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    sim_resolution_end = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    resolution_start_str = args.resolution_start or args.start_date
    resolution_end_str = args.resolution_end or args.end_date
    resolution_start = datetime.strptime(resolution_start_str, "%Y-%m-%d").date()
    resolution_end = datetime.strptime(resolution_end_str, "%Y-%m-%d").date()
    
    # Simulation starts lookback_days before first resolution
    from datetime import timedelta
    sim_start = sim_resolution_start - timedelta(days=args.lookback_days)
    sim_end = sim_resolution_end
    
    print(f"Resolution window: {resolution_start} to {resolution_end}")
    print(f"Simulation window: {sim_start} to {sim_end} (lookback={args.lookback_days} days)")
    
    # Handle restart-from-day (creates new dir, copies truncated logs, then uses resume)
    is_restart = False
    if args.restart_from:
        if not args.restart_from_day:
            print("Error: --restart_from_day required when using --restart_from")
            sys.exit(1)
        
        # Validate source exists
        if not os.path.exists(args.restart_from):
            print(f"Error: Restart source not found: {args.restart_from}")
            sys.exit(1)
        
        # Load config from source run
        src_config_path = os.path.join(args.restart_from, 'config.json')
        if os.path.exists(src_config_path):
            print(f"Loading configuration from source run: {src_config_path}")
            with open(src_config_path, 'r') as f:
                src_config = json.load(f)
            # Update args with source config (preserve restart-specific flags).
            # Keys explicitly set by the new YAML config are authoritative and must
            # NOT be overwritten by the source run's config — even when the YAML
            # intentionally omits a key (e.g. max_actions) that the source had.
            restart_from = args.restart_from
            restart_from_day = args.restart_from_day
            _skip_keys = {'restart_from', 'restart_from_day', 'resume', 'timestamp'} | _yaml_config_keys
            for key, val in src_config.items():
                if key not in _skip_keys:
                    if not hasattr(args, key) or getattr(args, key) is None:
                        setattr(args, key, val)
            args.restart_from = restart_from
            args.restart_from_day = restart_from_day
        
        # Create new output directory
        output_dir = create_output_dir(args.sim_name + "_restart", args.output_base)
        print(f"Restart output directory: {output_dir}")
        
        # Create agents directory
        agents_dir = os.path.join(output_dir, "agents")
        os.makedirs(agents_dir, exist_ok=True)
        
        # Prepare restart: copy truncated logs and memory
        print(f"Preparing restart from {args.restart_from} @ day {args.restart_from_day}...")
        prepare_restart_directory(args.restart_from, args.restart_from_day, output_dir)
        
        # Save new config with restart metadata
        save_config(output_dir, args, {
            'restart_source': args.restart_from,
            'restart_from_day': args.restart_from_day
        })
        
        # Set resume flag to use existing _restore_state logic
        args.resume = output_dir
        is_restart = True
        print(f"Restart prepared. Will resume from truncated state.")
    
    # Create output directory
    if args.resume:
        output_dir = args.resume
        print(f"Resuming in directory: {output_dir}")
        # Validate agents dir exists
        agents_dir = os.path.join(output_dir, "agents")
        if not os.path.exists(agents_dir):
             print(f"Warning: agents directory not found in {output_dir}")
             os.makedirs(agents_dir, exist_ok=True)
    elif not is_restart:
        output_dir = create_output_dir(args.sim_name, args.output_base)
        print(f"Output directory: {output_dir}")
        
        # Create agents directory
        agents_dir = os.path.join(output_dir, "agents")
        os.makedirs(agents_dir, exist_ok=True)
        
        # Save config
        save_config(output_dir, args)
    
    # IMPORTANT: Initialize matcher server BEFORE embedding model
    # The matcher runs as a subprocess and needs GPU allocation first.
    # If embedding model loads first (in-process), the subprocess can't share GPU.
    # Set output dir for matcher logs
    # Set output dir for matcher logs
    os.environ['SIM_OUTPUT_DIR'] = output_dir

    matcher_cache_path = str(
        resolve_sim_matcher_cache_path(
            output_dir=output_dir,
            matching=args.matching,
            matcher=args.matcher,
            split=args.split,
            matcher_cache=getattr(args, "matcher_cache", None),
        )
    )
    local_matcher_cache_path = os.path.join(output_dir, "matcher_cache.json")
    if os.path.normpath(matcher_cache_path) == os.path.normpath(os.path.abspath(local_matcher_cache_path)):
        print(f"Matcher cache: per-run JSON at {matcher_cache_path}")
    else:
        print(f"Matcher cache: shared JSON at {matcher_cache_path}")
    
    matcher_provider = None
    
    # Optimization: Skip matcher GPU for Metaculus (uses exact string matching)
    if args.dataset.startswith("metaculus"):
        print(f"\nDataset '{args.dataset}' uses exact matching. Skipping matcher GPU allocation.")
        matcher_provider = None # Passed as None to env, will set self.matcher = None
    elif args.matching == "vllm":
        print(f"\nInitializing VLLM matcher server (must start before embedding model)...")
        from inference.vllm import VLLMInference
        matcher_rope_scaling = getattr(args, "matcher_rope_scaling", None)
        if matcher_rope_scaling is None:
            matcher_rope_scaling = getattr(args, "rope_scaling", None)
        matcher_max_model_len = getattr(args, "matcher_max_model_len", None)
        if matcher_max_model_len is None:
            matcher_max_model_len = args.max_model_len
        matcher_provider = VLLMInference(args.matcher, max_model_len=matcher_max_model_len, 
                                          gpu_memory_utilization=args.matcher_gpu_mem,
                                          timeout=getattr(args, "vllm_request_timeout", 120.0),
                                          max_num_seqs=getattr(args, "vllm_max_num_seqs", 8),
                                          startup_timeout=getattr(args, "vllm_startup_timeout", 300.0),
                                          rope_scaling=matcher_rope_scaling,
                                          enable_prefix_caching=getattr(args, "vllm_enable_prefix_caching", True),
                                          cuda_visible_devices=getattr(args, "aux_cuda_visible_devices", None))
        # Force server startup by making a warmup call - fail job if this fails
        print(f"  Warming up matcher ({args.matcher}, GPU: {args.matcher_gpu_mem:.0%})...")
        matcher_provider.chat([{"role": "user", "content": "test"}], {"temperature": 0, "max_tokens": 1})
        print(f"  Matcher server ready!")
    elif args.matching == "openrouter":
        from inference.openrouter import OpenRouterInference
        matcher_kwargs = {}
        matcher_prov_order = getattr(args, "matcher_openrouter_provider_order", None)
        if matcher_prov_order:
            matcher_kwargs["provider"] = {"order": matcher_prov_order, "allow_fallbacks": True}
        matcher_provider = OpenRouterInference(args.matcher, **matcher_kwargs)
        print(f"Answer matching: OpenRouter with {args.matcher}")
    elif args.matching == "exact":
        print(f"Answer matching: exact (normalized string comparison)")
    else:
        raise ValueError(f"Unknown matching mode: {args.matching}")
    
    # Setup search tool if specified (preload embedding model)
    # Note: This comes AFTER matcher server is initialized
    search_tool = None
    if args.search_db:
        print(f"\nSetting up search tool...", flush=True)
        from agents.search_tools.lancedb import LanceDBSearchTool
        
        # Preload embedding model for zero-latency searches
        embedding_model = None
        if args.embedding_model and os.path.exists(args.embedding_model):
            print(f"  Loading embedding model: {args.embedding_model} (GPU: {args.embedding_gpu_mem:.0%})")
            try:
                from inference.vllm import VLLMInference
                embedding_rope_scaling = getattr(args, "embedding_rope_scaling", None)
                if embedding_rope_scaling is None:
                    embedding_rope_scaling = getattr(args, "rope_scaling", None)
                embedding_max_model_len = getattr(args, "embedding_max_model_len", None)
                if embedding_max_model_len is None:
                    embedding_max_model_len = args.max_model_len
                embedding_model = VLLMInference(
                    args.embedding_model,
                    max_model_len=embedding_max_model_len,
                    gpu_memory_utilization=args.embedding_gpu_mem,
                    timeout=getattr(args, "vllm_request_timeout", 120.0),
                    max_num_seqs=getattr(args, "vllm_max_num_seqs", 8),
                    startup_timeout=getattr(args, "vllm_startup_timeout", 300.0),
                    rope_scaling=embedding_rope_scaling,
                    enable_prefix_caching=False,
                    cuda_visible_devices=getattr(args, "aux_cuda_visible_devices", None),
                )
                # Force server startup and verify /v1/embeddings works.
                warm = embedding_model.embed(["test"], use_tqdm=False)
                if not warm:
                    raise RuntimeError("Embedding warmup returned empty result")
                # Publish the actual URL so child agents (esp. parallel sims with
                # multiple vLLMs on the host) don't have to scan port ranges and
                # can deterministically pick the right embedding server.
                if getattr(embedding_model, "_port", None):
                    os.environ["FSIM_EMBEDDING_URL"] = f"http://127.0.0.1:{embedding_model._port}"
                print("  Embedding server ready (queries will use semantic/hybrid search)")
            except Exception as e:
                print(f"  Warning: Failed to start embedding server: {e}")
                print("  Falling back to keyword-only search")
        
        search_tool = LanceDBSearchTool(args.search_db, embedding_model=embedding_model)
        if search_tool.is_available:
            print(f"  LanceDB connected: {args.search_db}")
        else:
            print(f"  Warning: LanceDB not available at {args.search_db}")
            search_tool = None
    
    # Determine agents to create
    agents = []
    
    if args.agents_config or (args.config and 'agents' in config):
        # Multi-agent mode from config file
        if args.agents_config:
            print(f"\nLoading agents from config: {args.agents_config}", flush=True)
            config_agents = load_agents_config(args.agents_config)
        else:
            # Use the main config which already has agents
            config_agents = config
            print(f"\nUsing agents defined in main config", flush=True)
            
        print(f"  Config loaded successfully", flush=True)
        print(f"  Config loaded successfully", flush=True)
        # We need to pass sim_start_date to create_agents_from_config
        # It's computed later in main, but we need it here.
        # So we'll need to move date parsing up or just pass args and let it find it.
        # Actually args.sim_start_date doesn't exist yet, we calculated sim_start local var.
        # Let's attach it to args object temporarily
        args.sim_start_date = sim_start
        agents = create_agents_from_config(config_agents, args, output_dir, search_tool)
        
        # Save agent config copy if it was a separate file
        if args.agents_config:
            import shutil
            config_copy = os.path.join(output_dir, "agents_config.yaml")
            shutil.copy(args.agents_config, config_copy)
        
    elif not args.no_inference:
        # Single agent mode (legacy)
        from inference.openrouter import GlobalRateLimiter
        GlobalRateLimiter.configure(args.rate_limit)
        
        if args.provider == "vllm":
            from inference.vllm import VLLMInference
            agent_max_model_len = getattr(args, "agent_max_model_len", None)
            if agent_max_model_len is None:
                agent_max_model_len = args.max_model_len
            print(f"Loading VLLM model: {args.model_path} (max_model_len={agent_max_model_len})")
            inference_provider = VLLMInference(
                args.model_path,
                max_model_len=agent_max_model_len,
                gpu_memory_utilization=getattr(args, "vllm_gpu_mem", 0.3),
                timeout=getattr(args, "vllm_request_timeout", 120.0),
                max_num_seqs=getattr(args, "vllm_max_num_seqs", 8),
                tensor_parallel_size=getattr(args, "vllm_tensor_parallel_size", 1),
                data_parallel_size=getattr(args, "vllm_data_parallel_size", 1),
                pipeline_parallel_size=getattr(args, "vllm_pipeline_parallel_size", 1),
                enable_expert_parallel=getattr(args, "vllm_enable_expert_parallel", False),
                all2all_backend=getattr(args, "vllm_all2all_backend", None),
                startup_timeout=getattr(args, "vllm_startup_timeout", 300.0),
                rope_scaling=getattr(args, "rope_scaling", None),
                enable_tools=getattr(args, "vllm_enable_tools", False),
                tool_call_parser=_resolve_vllm_tool_call_parser(args.model_path, args),
                tool_parser_plugin=getattr(args, "vllm_tool_parser_plugin", None),
                enable_prefix_caching=getattr(args, "vllm_enable_prefix_caching", True),
                cuda_visible_devices=getattr(args, "agent_cuda_visible_devices", None),
                language_model_only=getattr(args, "language_model_only", False),
            )
            model_name = os.path.basename(args.model_path)
            
        elif args.provider == "openrouter":
            from inference.openrouter import OpenRouterInference
            if not args.openrouter_model:
                print("Error: --openrouter_model is required in single-agent mode with --provider openrouter")
                sys.exit(1)
            try:
                from inference.openrouter import configure_http_pool
                desired_pool = max(20, int(getattr(args, 'warmup_parallelism', 20)))
                configure_http_pool(pool_maxsize=desired_pool, pool_connections=desired_pool)
                print(f"OpenRouter HTTP pool: connections=maxsize={desired_pool}")
            except Exception as e:
                print(f"Warning: failed to configure OpenRouter HTTP pool: {e}")
            print(f"Using OpenRouter model: {args.openrouter_model}")
            inference_provider = OpenRouterInference(args.openrouter_model)
            model_name = args.openrouter_model
        
        # Create agent
        model_short = get_model_short_name(model_name)
        agent_id = f"basic_{model_short}_001"
        
        # Create agent directory
        agent_dir = os.path.join(output_dir, "agents", agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        
        agent_config = AgentConfig(
            max_actions=args.max_actions,
            warmup_max_actions=args.warmup_max_actions,
            max_total_tokens=args.max_total_tokens,
            warmup_max_total_tokens=args.warmup_max_total_tokens,
            submit_reserve_tokens=args.submit_reserve_tokens,
            warmup_submit_reserve_tokens=args.warmup_submit_reserve_tokens,
            force_submit_threshold_tokens=args.force_submit_threshold_tokens,
            warmup_force_submit_threshold_tokens=args.warmup_force_submit_threshold_tokens,
            warmup_parallelism=args.warmup_parallelism,
            max_submit_retries=args.max_retries,
            max_outcomes_per_question=int(getattr(args, 'max_outcomes_per_question', 5) or 5),
            memory_dir=agent_dir,
            enable_memory=True,
            append_model_output_logs=bool(args.resume),
            tool_result_keep_last=int(getattr(args, 'tool_result_keep_last', -1)),
            sampling_params={
                'temperature': args.temperature,
                'max_tokens': args.max_tokens,
                **({'top_p': float(args.top_p)} if getattr(args, 'top_p', None) is not None else {}),
                **({'top_k': int(args.top_k)} if getattr(args, 'top_k', None) is not None else {}),
                **({'repetition_penalty': float(args.repetition_penalty)} if getattr(args, 'repetition_penalty', None) is not None else {}),
            },
            search_cutoff_days=args.search_cutoff_days,
            resolution_guard=args.resolution_guard,
            timegap_days=args.timegap_days,
            single_agent_mode=True,  # Legacy CLI mode is always single-agent
            gptoss_prompt_mode=args.gptoss_prompt_mode,
            gptoss_reasoning_effort=args.gptoss_reasoning_effort,
            gptoss_include_reasoning=args.gptoss_include_reasoning,
            gptoss_responses_max_retries=args.gptoss_responses_max_retries,
            gptoss_retry_backoff_base_s=args.gptoss_retry_backoff_base_s,
            gptoss_retry_backoff_max_s=args.gptoss_retry_backoff_max_s,
        )
        if args.scaffold == 'basic':
            agent_cls = BasicAgent
            agent = agent_cls(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model_name,
                search_tool=search_tool
            )
        elif args.scaffold in ['allQ', 'allq']:
            agent_cls = AllQAgent
            agent = agent_cls(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model_name,
                search_tool=search_tool,
                start_date=sim_start
            )
        elif args.scaffold == 'qwenbasic':
            agent = QwenBasicAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model_name,
                search_tool=search_tool
            )
        elif args.scaffold == 'qwenallq':
            agent = QwenAllQAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model_name,
                search_tool=search_tool,
                start_date=sim_start
            )
        elif args.scaffold == 'mirobasic':
            agent = MiroBasicAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model_name,
                search_tool=search_tool
            )
        elif args.scaffold == 'miroallq':
            agent = MiroAllQAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model_name,
                search_tool=search_tool,
                start_date=sim_start
            )
        elif args.scaffold == 'gptossbasic':
            agent = GPTOSSBasicAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model_name,
                search_tool=search_tool
            )
        elif args.scaffold == 'gptossallq':
            agent = GPTOSSAllQAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model_name,
                search_tool=search_tool,
                start_date=sim_start
            )
        elif args.scaffold == 'allqd':
            agent = AllQDailyAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model_name,
                search_tool=search_tool,
                start_date=sim_start
            )
        elif args.scaffold == 'og':
            agent = OgAgent(
                agent_id=agent_id,
                inference_provider=inference_provider,
                config=agent_config,
                model_name=model_name,
                search_tool=search_tool,
                start_date=sim_start
            )
        elif args.scaffold == 'minimalHarness':
            from agents.minimalHarnessAgent.agent import MinimalHarnessAgent, MinimalHarnessConfig
            cc_config = MinimalHarnessConfig(
                model=model_name,
                search_db=getattr(args, 'search_db', '') or '',
                embedding_model=getattr(args, 'embedding_model', '') or '',
                search_cutoff_days=getattr(args, 'search_cutoff_days', 0),
                articles_base=getattr(args, 'articles_base', os.environ.get('FSIM_ARTICLES_BASE', '')),
                start_date=sim_start,
                end_date=getattr(args, 'end_date', None),
            )
            agent = MinimalHarnessAgent(
                agent_id=agent_id,
                config=cc_config,
                search_tool=search_tool,
                agent_dir=os.path.join(output_dir, 'agents', agent_id),
                articles_base=getattr(args, 'articles_base', os.environ.get('FSIM_ARTICLES_BASE', '')),
            )
        else:
            raise ValueError(
                f"Unknown scaffold: {args.scaffold}. Only 'basic', 'allQ', 'allqd', 'og', "
                "'qwenbasic', 'qwenallq', 'mirobasic', 'miroallq', 'gptossbasic', "
                "'gptossallq', and 'minimalHarness' are supported."
            )
        agents.append(agent)
        print(f"  Created agent: {agent_id}")
    else:
        print("Running without LLM inference (--no_inference)")
    
    # Note: matcher_provider was already initialized before embedding model (see above)
    
    env_max_outcomes_per_question = max(
        (
            int(getattr(getattr(agent, "config", None), "max_outcomes_per_question", 5) or 5)
            for agent in agents
        ),
        default=5,
    )

    # Initialize environment with resolution date filter
    print(f"\nLoading dataset: {args.dataset}")
    env = SimulationEnvironment(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        dataset_cache=args.dataset_cache,
        start_date=sim_start,
        end_date=sim_end,
        inference_provider=matcher_provider,
        output_dir=output_dir,
        resolution_start=resolution_start,
        resolution_end=resolution_end,
        parallel=args.parallel,
        split=args.split,
        prepend_train_resolution_start=args.prepend_train_resolution_start,
        prepend_train_resolution_end=args.prepend_train_resolution_end,
        subsample_per_month=args.subsample_per_month,
        timegap_days=args.timegap_days,
        resume_dir=args.resume if args.resume else None,
        matcher_cache_path=matcher_cache_path,
        max_outcomes_per_question=env_max_outcomes_per_question,
    )
    
    # Add all agents
    for agent in agents:
        env.add_agent(agent)
        
    # Optional rescore
    if args.resume and args.rescore:
        env.rescore()
    
    # Run WARMUP phase for agents that support it (Day 0)
    # SKIP warmup if resuming - predictions already exist in actions.jsonl
    if args.resume:
        print("\n" + "="*60)
        print("Skipping Warmup (resuming from previous run)")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Checking for Agent Warmup (Day 0 predictions)...")
        print("="*60)
        
        # Get active questions
        active_questions = env.q_pool.get_active()
        
        if active_questions:
            # CRITICAL: Initialize prediction histories BEFORE warmup
            # This ensures warmup predictions are properly tracked (not just logged).
            # Normally, histories are created in env.step(), but warmup runs before that.
            from environment.scoring import PredictionHistory
            
            for q in active_questions:
                if q.qid not in env.prediction_histories:
                    env.prediction_histories[q.qid] = PredictionHistory(
                        question_id=q.qid,
                        start_date=env.current_date,
                        resolution_date=q.resolution_date
                    )
            
            print(f"  Initialized {len(active_questions)} prediction histories for warmup.")
            
            # Create a ForecastInterface for warmup
            from threading import Lock
            warmup_interface = SimForecastInterface(
                active_questions,
                env.current_aggregates,
                env.prediction_histories,
                env.current_date,
                env.logger,
                resolved_questions=env.resolved_questions,
                resolved_agent_predictions=env.resolved_agent_predictions,
                histories_lock=env._histories_lock,
                market_csv_path=None,  # No market CSV yet
                timegap_days=env.timegap_days,
                last_active_date=env._get_last_active_date(),
                next_active_date=env._get_next_active_date(),
                simulation_end_date=env.end_date,
                num_agents=len(agents),
                max_outcomes_per_question=env.max_outcomes_per_question,
            )
            warmup_interface.source_name = getattr(env, 'source_name', 'openforesight')
            warmup_interface.source_context = getattr(env, 'source_context', '')
            
            # Only AllQAgent uses the explicit "warmup" (day-0 all-questions sweep).
            # AllQDailyAgent ("allqd") should behave identically on every day, so we do not
            # run a separate phase 0 for it.
            for agent in agents:
                if isinstance(agent, AllQAgent) and not isinstance(agent, AllQDailyAgent):
                    agent.warmup(warmup_interface, env.current_date)
        else:
            print("  No active questions for warmup.")

    
    if not agents:
        print("No agents added")
        return
    
    # Run simulation
    print("\n" + "="*60)
    print(f"Starting simulation with {len(agents)} agent(s)...")
    if args.parallel:
        print(f"Parallel execution: ENABLED")
    print("="*60 + "\n")
    
    env.run()
    
    print("\n" + "="*60)
    print(f"Simulation complete. Logs at: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
