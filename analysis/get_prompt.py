#!/usr/bin/env python3
"""Render start-of-day agent prompt from current codebase.

This reconstructs run state from actions/history and then calls the *current*
prompt-building code for the run's configured scaffold/agent, instead of reading
stored model_raw prompts.

Example:
  python analysis/get_prompt.py \
    /fast/sgoel/forecasting/current_sim/allq_sim_oss120b_128k_a10_50_med_r00/26-02-20-23-57-26 \
    2025-06-24

Writes:
  <run_dir>/analysis/prompt_check/<day>.txt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.basicAgent import AgentConfig, BasicAgent
from agents.allQAgent import AllQAgent
from agents.allQAgent.agent import AllQDailyAgent
from agents.gptossAgent import GPTOSSBasicAgent, GPTOSSAllQAgent
from agents.ogAgent import OgAgent
from agents.qwenAgent import QwenBasicAgent, QwenAllQAgent
from agents.basicAgent.tools import build_action_tools
from environment.env import SimulationEnvironment, SimForecastInterface


class _PromptOnlySearchTool:
    def __init__(self, available: bool):
        self.is_available = bool(available)

    def search(self, *args: Any, **kwargs: Any) -> list:
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render start-of-day prompt using current code + run history (no model_raw parsing)."
        )
    )
    parser.add_argument("run_dir", help="Timestamped run folder")
    parser.add_argument("day", help="Day to render (YYYY-MM-DD)")
    parser.add_argument(
        "--agent-id",
        default=None,
        help="Agent folder name under run_dir/agents (required if multiple agents)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )
    return parser.parse_args()


def _parse_iso_day(s: str) -> date:
    try:
        return date.fromisoformat(str(s))
    except Exception as exc:
        raise ValueError(f"Invalid day (expected YYYY-MM-DD): {s}") from exc


def _load_config(run_dir: Path) -> Dict[str, Any]:
    cfg_json = run_dir / "config.json"
    cfg_yaml = run_dir / "config.yaml"

    if cfg_json.exists():
        return json.loads(cfg_json.read_text(encoding="utf-8"))

    if cfg_yaml.exists():
        if yaml is None:
            raise RuntimeError("PyYAML not available and config.json missing")
        data = yaml.safe_load(cfg_yaml.read_text(encoding="utf-8"))
        return data or {}

    raise FileNotFoundError(f"No config.json or config.yaml in {run_dir}")


def _model_short_name(model: str) -> str:
    name = os.path.basename(model)
    if "/" in model:
        name = model.split("/")[-1]
    return name.split(":")[0]


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _agent_defs_from_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    defaults = cfg.get("defaults") or {}
    agents = cfg.get("agents") or []

    if not agents:
        # Legacy single-agent fallback
        provider = cfg.get("provider", defaults.get("provider", "openrouter"))
        model = (
            cfg.get("model")
            or cfg.get("model_path")
            or cfg.get("openrouter_model")
            or defaults.get("model")
        )
        scaffold = cfg.get("scaffold", defaults.get("scaffold", "basic"))
        if not model:
            raise ValueError("Could not determine model from config")
        agents = [{"provider": provider, "model": model, "scaffold": scaffold}]

    expanded: List[Dict[str, Any]] = []
    for a in agents:
        provider = a.get("provider", defaults.get("provider", cfg.get("provider", "openrouter")))
        scaffold = a.get("scaffold", defaults.get("scaffold", cfg.get("scaffold", "basic")))
        model = a.get("model") or cfg.get("model") or cfg.get("model_path") or cfg.get("openrouter_model")
        if not model:
            raise ValueError(f"Missing model for agent entry: {a}")
        merged = dict(a)
        merged["provider"] = provider
        merged["scaffold"] = scaffold
        merged["model"] = model
        expanded.append(merged)
    return expanded


def _build_expected_agent_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    defaults = cfg.get("defaults") or {}
    defs = _agent_defs_from_config(cfg)
    counts: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []

    for a in defs:
        scaffold = str(a.get("scaffold", "basic"))
        model = str(a.get("model"))
        provider = str(a.get("provider", "openrouter"))

        base = f"{scaffold}_{_model_short_name(model)}"
        counts[base] = counts.get(base, 0) + 1
        agent_id = f"{base}_{counts[base]:03d}"

        max_actions = _optional_int(a.get("max_actions", defaults.get("max_actions", cfg.get("max_actions", None))))
        warmup_max_actions = _optional_int(
            a.get("warmup_max_actions", defaults.get("warmup_max_actions", cfg.get("warmup_max_actions", None)))
        )
        max_total_tokens = _optional_int(
            a.get("max_total_tokens", defaults.get("max_total_tokens", cfg.get("max_total_tokens", None)))
        )
        warmup_max_total_tokens = _optional_int(
            a.get(
                "warmup_max_total_tokens",
                defaults.get("warmup_max_total_tokens", cfg.get("warmup_max_total_tokens", None)),
            )
        )
        submit_reserve_tokens = _optional_int(
            a.get(
                "submit_reserve_tokens",
                defaults.get("submit_reserve_tokens", cfg.get("submit_reserve_tokens", 8192)),
            )
        )
        warmup_submit_reserve_tokens = _optional_int(
            a.get(
                "warmup_submit_reserve_tokens",
                defaults.get("warmup_submit_reserve_tokens", cfg.get("warmup_submit_reserve_tokens", None)),
            )
        )
        force_submit_threshold_tokens = _optional_int(
            a.get(
                "force_submit_threshold_tokens",
                defaults.get(
                    "force_submit_threshold_tokens",
                    cfg.get("force_submit_threshold_tokens", 16384),
                ),
            )
        )
        warmup_force_submit_threshold_tokens = _optional_int(
            a.get(
                "warmup_force_submit_threshold_tokens",
                defaults.get(
                    "warmup_force_submit_threshold_tokens",
                    cfg.get("warmup_force_submit_threshold_tokens", None),
                ),
            )
        )
        warmup_parallelism = int(
            a.get("warmup_parallelism", defaults.get("warmup_parallelism", cfg.get("warmup_parallelism", 20)))
        )
        max_retries = int(a.get("max_retries", defaults.get("max_retries", cfg.get("max_retries", 3))))
        temperature = float(a.get("temperature", defaults.get("temperature", cfg.get("temperature", 0.7))))
        max_tokens = int(a.get("max_tokens", defaults.get("max_tokens", cfg.get("max_tokens", 2048))))
        reasoning = a.get("reasoning", defaults.get("reasoning", cfg.get("reasoning", None)))
        search_cutoff_days = int(
            a.get("search_cutoff_days", defaults.get("search_cutoff_days", cfg.get("search_cutoff_days", 0)))
        )
        enable_memory = bool(a.get("enable_memory", defaults.get("enable_memory", True)))
        singleans = bool(a.get("singleans", defaults.get("singleans", False)))

        gptoss_prompt_mode = str(
            a.get("gptoss_prompt_mode", defaults.get("gptoss_prompt_mode", cfg.get("gptoss_prompt_mode", "instructions")))
        )
        gptoss_reasoning_effort = str(
            a.get(
                "gptoss_reasoning_effort",
                defaults.get("gptoss_reasoning_effort", cfg.get("gptoss_reasoning_effort", "medium")),
            )
        ).lower()
        gptoss_include_reasoning = bool(
            a.get(
                "gptoss_include_reasoning",
                defaults.get("gptoss_include_reasoning", cfg.get("gptoss_include_reasoning", True)),
            )
        )
        gptoss_responses_max_retries = int(
            a.get(
                "gptoss_responses_max_retries",
                defaults.get("gptoss_responses_max_retries", cfg.get("gptoss_responses_max_retries", 3)),
            )
        )
        gptoss_retry_backoff_base_s = float(
            a.get(
                "gptoss_retry_backoff_base_s",
                defaults.get("gptoss_retry_backoff_base_s", cfg.get("gptoss_retry_backoff_base_s", 1.0)),
            )
        )
        gptoss_retry_backoff_max_s = float(
            a.get(
                "gptoss_retry_backoff_max_s",
                defaults.get("gptoss_retry_backoff_max_s", cfg.get("gptoss_retry_backoff_max_s", 16.0)),
            )
        )

        spec = {
            "agent_id": agent_id,
            "provider": provider,
            "scaffold": scaffold,
            "model": model,
            "config": AgentConfig(
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
                memory_dir="",  # patched later
                enable_memory=enable_memory,
                singleans=singleans,
                sampling_params={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **({"reasoning": reasoning} if reasoning is not None else {}),
                },
                search_cutoff_days=search_cutoff_days,
                timegap_days=int(cfg.get("timegap_days", 1)),
                single_agent_mode=(len(defs) == 1),
                gptoss_prompt_mode=gptoss_prompt_mode,
                gptoss_reasoning_effort=gptoss_reasoning_effort,
                gptoss_include_reasoning=gptoss_include_reasoning,
                gptoss_responses_max_retries=gptoss_responses_max_retries,
                gptoss_retry_backoff_base_s=gptoss_retry_backoff_base_s,
                gptoss_retry_backoff_max_s=gptoss_retry_backoff_max_s,
            ),
        }
        out.append(spec)

    return out


def _pick_agent_spec(cfg: Dict[str, Any], run_dir: Path, agent_id_arg: Optional[str]) -> Dict[str, Any]:
    specs = _build_expected_agent_specs(cfg)
    agents_dir = run_dir / "agents"
    if not agents_dir.is_dir():
        raise ValueError(f"Missing agents dir: {agents_dir}")

    on_disk = sorted([p.name for p in agents_dir.iterdir() if p.is_dir()])
    spec_by_id = {s["agent_id"]: s for s in specs}

    if agent_id_arg:
        if agent_id_arg not in spec_by_id:
            known = ", ".join(sorted(spec_by_id.keys()))
            raise ValueError(f"Unknown --agent-id {agent_id_arg}. Known: {known}")
        return spec_by_id[agent_id_arg]

    candidates = [sid for sid in on_disk if sid in spec_by_id]
    if len(candidates) == 1:
        return spec_by_id[candidates[0]]

    if len(candidates) == 0:
        known = ", ".join(sorted(spec_by_id.keys()))
        raise ValueError(f"No matching configured agent dirs found. Config IDs: {known}")

    raise ValueError(f"Multiple agents found ({', '.join(candidates)}). Pass --agent-id.")


def _filter_actions_for_day_start(src_actions: Path, dst_actions: Path, day: date) -> Tuple[int, int]:
    kept = 0
    total = 0
    with src_actions.open("r", encoding="utf-8") as fin, dst_actions.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            sim_date_raw = rec.get("sim_date")
            if not sim_date_raw:
                continue
            try:
                rec_day = date.fromisoformat(str(sim_date_raw))
            except Exception:
                continue

            keep = False
            if rec_day < day:
                keep = True
            elif rec_day == day and rec.get("type") == "resolution":
                # Include same-day resolutions (these happen before agent prompting).
                keep = True

            if keep:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    return kept, total


def _build_environment_for_day_start(
    run_dir: Path, cfg: Dict[str, Any], day: date
) -> Tuple[SimulationEnvironment, Path]:
    src_actions = run_dir / "actions.jsonl"
    if not src_actions.exists():
        raise FileNotFoundError(f"Missing actions.jsonl: {src_actions}")

    sim_resolution_start = _parse_iso_day(str(cfg.get("start_date")))
    sim_resolution_end = _parse_iso_day(str(cfg.get("end_date")))
    lookback_days = int(cfg.get("lookback_days", 7))

    resolution_start = _parse_iso_day(str(cfg.get("resolution_start") or cfg.get("start_date")))
    resolution_end = _parse_iso_day(str(cfg.get("resolution_end") or cfg.get("end_date")))

    sim_start = sim_resolution_start - timedelta(days=lookback_days)
    sim_end = sim_resolution_end

    if day < sim_start or day > sim_end:
        raise ValueError(f"Requested day {day} outside simulation window [{sim_start}, {sim_end}]")

    tmpdir = Path(tempfile.mkdtemp(prefix="prompt_replay_"))
    truncated_actions = tmpdir / "actions.jsonl"
    _filter_actions_for_day_start(src_actions, truncated_actions, day)

    env = SimulationEnvironment(
        dataset=str(cfg.get("dataset", "openforesight")),
        dataset_path=cfg.get("dataset_path"),
        dataset_cache=cfg.get("dataset_cache"),
        start_date=sim_start,
        end_date=sim_end,
        inference_provider=None,
        output_dir=str(tmpdir),
        resolution_start=resolution_start,
        resolution_end=resolution_end,
        parallel=bool(cfg.get("parallel", False)),
        split=str(cfg.get("split", "test")),
        prepend_train_resolution_start=cfg.get("prepend_train_resolution_start"),
        prepend_train_resolution_end=cfg.get("prepend_train_resolution_end"),
        subsample_per_month=cfg.get("subsample_per_month"),
        timegap_days=int(cfg.get("timegap_days", 1)),
        resume_dir=str(tmpdir),
        min_forecasters=int(cfg.get("min_forecasters", 0)),
        resolved_only=bool(cfg.get("resolved_only", False)),
    )

    # Restore advances by the configured cadence; force back to requested day-start.
    env.current_date = day

    # Recompute aggregates and market snapshot for this day-start state.
    active_questions = env.q_pool.get_active()
    env._update_aggregates(active_questions)
    env.market_csv_path = env.market_writer.write(
        active_questions,
        env.resolved_questions,
        env.current_aggregates,
        env.prediction_histories,
    )

    return env, tmpdir


def _instantiate_agent(spec: Dict[str, Any], run_dir: Path, cfg: Dict[str, Any], sim_start_day: date):
    agent_id = spec["agent_id"]
    provider = str(spec["provider"])
    scaffold = str(spec["scaffold"])
    model = str(spec["model"])
    agent_cfg: AgentConfig = spec["config"]

    agent_dir = run_dir / "agents" / agent_id
    if not agent_dir.is_dir():
        raise ValueError(f"Agent dir missing for {agent_id}: {agent_dir}")

    agent_cfg.memory_dir = str(agent_dir)

    search_available = bool(cfg.get("search_db"))
    search_tool = _PromptOnlySearchTool(search_available)

    use_gptoss_harmony = (
        provider == "vllm"
        and bool(cfg.get("vllm_enable_tools", False))
        and ("gpt-oss" in model.lower())
    )

    if scaffold in ("basic",):
        cls = GPTOSSBasicAgent if use_gptoss_harmony else BasicAgent
        return cls(agent_id=agent_id, inference_provider=None, config=agent_cfg, model_name=model, search_tool=search_tool)

    if scaffold in ("allQ", "allq"):
        cls = GPTOSSAllQAgent if use_gptoss_harmony else AllQAgent
        return cls(
            agent_id=agent_id,
            inference_provider=None,
            config=agent_cfg,
            model_name=model,
            search_tool=search_tool,
            start_date=sim_start_day,
        )

    if scaffold == "qwenbasic":
        return QwenBasicAgent(
            agent_id=agent_id,
            inference_provider=None,
            config=agent_cfg,
            model_name=model,
            search_tool=search_tool,
        )

    if scaffold == "qwenallq":
        return QwenAllQAgent(
            agent_id=agent_id,
            inference_provider=None,
            config=agent_cfg,
            model_name=model,
            search_tool=search_tool,
            start_date=sim_start_day,
        )

    if scaffold == "allqd":
        # allqd prompts are per-question, not a single day-level prompt.
        raise NotImplementedError(
            "Scaffold 'allqd' has per-question prompts, not a single start-of-day prompt."
        )

    if scaffold in ("og", "ogagent", "ogAgent"):
        return OgAgent(
            agent_id=agent_id,
            inference_provider=None,
            config=agent_cfg,
            model_name=model,
            search_tool=search_tool,
            start_date=sim_start_day,
        )

    raise ValueError(f"Unsupported scaffold: {scaffold}")


def _seed_feedback_state(agent: Any, forecast_interface: SimForecastInterface, day: date) -> None:
    """Preload FeedbackHandler cumulative stats with events before `day`."""
    fh = getattr(agent, "_feedback_handler", None)
    if fh is None:
        return

    for event in getattr(forecast_interface, "resolution_events", []) or []:
        sim_date_raw = event.get("sim_date")
        if not sim_date_raw:
            continue
        try:
            ev_day = date.fromisoformat(str(sim_date_raw))
        except Exception:
            continue
        if ev_day >= day:
            continue

        qid = str(event.get("qid")) if event.get("qid") is not None else None
        if not qid or qid in fh.processed_qids:
            continue

        per_agent = event.get("agents") or {}
        my_stats = per_agent.get(agent.agent_id)
        if not isinstance(my_stats, dict):
            continue
        brier_raw = my_stats.get("brier")
        if brier_raw is None:
            continue

        fh.processed_qids.add(qid)
        fh.total_brier_sum += float(brier_raw)
        fh.total_tw_peer_sum += float(my_stats.get("tw_peer", 0.0) or 0.0)
        fh.total_resolved_count += 1
        if bool(my_stats.get("is_accurate", False)):
            fh.total_accuracy_count += 1


def _build_forecast_interface_for_day(env: SimulationEnvironment, agent_id: str) -> SimForecastInterface:
    active_questions = env.q_pool.get_active()
    safe_questions = env._get_safe_active_questions(active_questions)

    fi = SimForecastInterface(
        safe_questions,
        env.current_aggregates,
        env.prediction_histories,
        env.current_date,
        env.logger,
        resolved_questions=env.resolved_questions,
        resolution_events=env.resolution_events,
        resolved_agent_predictions=env.resolved_agent_predictions,
        histories_lock=env._histories_lock,
        market_csv_path=env.market_csv_path,
    )
    fi.source_name = getattr(env, "source_name", "openforesight")
    fi.source_context = getattr(env, "source_context", "")
    fi.set_agent_context(agent_id)
    return fi


def _render_prompt_payload(agent: Any, day: date) -> Dict[str, Any]:
    # Keep memory behavior aligned with runtime.
    if getattr(agent, "_memory", None) is not None:
        agent._memory.set_date(day)

    if isinstance(agent, QwenBasicAgent):
        instructions = agent._build_qwen_instructions(day)
        tools = build_action_tools(
            enable_query=True,
            enable_search=bool(agent._search_handler.is_available),
            max_outcomes_per_question=agent.config.max_outcomes_per_question,
            max_search_results=agent.config.max_search_results,
            search_chunk_tokens=agent._search_handler.chunk_tokens,
        )
        return {
            "prompt_mode": "qwen_chat_completions_tools",
            "messages": [{"role": "user", "content": instructions}],
            "tools": tools,
        }

    if isinstance(agent, GPTOSSBasicAgent):
        instructions = agent._build_harmony_instructions(day)
        instructions_for_api, conversation = agent._seed_harmony_conversation(
            instructions,
            day,
            budget_status_text=agent._build_start_budget_status(),
        )
        raw_payload = agent._build_model_input_for_logging(
            instructions=instructions_for_api,
            conversation=conversation,
        )
        payload = json.loads(raw_payload)
        payload["prompt_mode"] = str(getattr(agent.config, "gptoss_prompt_mode", "instructions"))
        return payload

    # Classic XML/scaffold prompt path.
    instructions = agent._build_instructions(day)
    return {
        "prompt_mode": "legacy_user_message",
        "messages": [{"role": "user", "content": instructions}],
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"run_dir is not a directory: {run_dir}")

    day = _parse_iso_day(args.day)
    cfg = _load_config(run_dir)

    spec = _pick_agent_spec(cfg, run_dir, args.agent_id)

    # Determine simulation start day (same as scripts/test_basic_agent.py logic).
    sim_resolution_start = _parse_iso_day(str(cfg.get("start_date")))
    lookback_days = int(cfg.get("lookback_days", 7))
    sim_start_day = sim_resolution_start - timedelta(days=lookback_days)

    env, temp_state_dir = _build_environment_for_day_start(run_dir, cfg, day)
    try:
        agent = _instantiate_agent(spec, run_dir, cfg, sim_start_day)
        fi = _build_forecast_interface_for_day(env, spec["agent_id"])

        # Match runtime wiring before prompt build.
        agent._setup_day(fi, day)
        _seed_feedback_state(agent, fi, day)

        payload = _render_prompt_payload(agent, day)

        out_dir = run_dir / "analysis" / "prompt_check"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{day.isoformat()}.txt"

        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists (use --overwrite): {out_path}")

        header = {
            "run_dir": str(run_dir),
            "day": day.isoformat(),
            "agent_id": spec["agent_id"],
            "scaffold": spec["scaffold"],
            "provider": spec["provider"],
            "model": spec["model"],
            "notes": "Rendered from current codebase using reconstructed day-start state; no model_raw prompt reuse.",
        }

        text = (
            json.dumps(header, ensure_ascii=False, indent=2)
            + "\n"
            + ("=" * 80)
            + "\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n"
        )
        out_path.write_text(text, encoding="utf-8")
        print(out_path)
        return 0
    finally:
        # Close temp logger files from reconstructed env.
        try:
            env.logger.close()
        except Exception:
            pass
        shutil.rmtree(temp_state_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
