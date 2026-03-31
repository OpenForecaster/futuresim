"""Build SkyRL training data from OpenForesight parquet splits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import glob
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from agents.basicAgent.config import AgentConfig
from agents.qwenAgent.agent import QwenAllQAgent
from datasets import Dataset
import pandas as pd

from skyrl_integration.envs import OPENFORESIGHT_SEARCH_WARMUP_ENV_ID

_REQUIRED_COLUMNS = [
    "qid",
    "question_title",
    "background",
    "resolution_criteria",
    "answer_type",
    "answer",
    "resolution_date",
]


@dataclass
class SplitBuildResult:
    path: str
    rows: int


@dataclass
class DatasetBuildResult:
    train: SplitBuildResult
    validation: SplitBuildResult


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()

    text = str(value).strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _load_split_dataframe(dataset_path: str, split: str) -> pd.DataFrame:
    pattern = str(Path(dataset_path) / f"{split}-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"No parquet files found for split='{split}' under {dataset_path}")

    dfs = []
    for path in files:
        df = pd.read_parquet(path, columns=_REQUIRED_COLUMNS)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def read_search_chunk_tokens(search_db: str) -> Optional[int]:
    if not search_db:
        return None

    config_path = Path(search_db) / "config.json"
    if not config_path.exists():
        return None

    try:
        raw = json.loads(config_path.read_text())
    except Exception:
        return None

    value = raw.get("chunk_tokens")
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _build_row(
    row: pd.Series,
    *,
    lookback_days: int,
    global_sim_date: Optional[date],
    search_max_date: Optional[date],
    search_db: str,
    embedding_model: str,
    embedding_gpu_mem: float,
    embedding_max_num_seqs: int,
    aux_cuda_visible_devices: str,
    search_type: str,
    search_topk: int,
    search_cutoff_days: int,
    search_min_days: int,
    max_snippet_chars: int,
    allow_substring_match: bool,
    matching: str,
    matcher: str,
    warmup_max_actions: Optional[int],
    warmup_max_total_tokens: Optional[int],
    warmup_submit_reserve_tokens: int,
    warmup_force_submit_threshold_tokens: int,
    budget_model_path: str,
    max_outcomes_per_question: int,
    split_name: str,
    matcher_cache_path: str = "",
) -> Optional[Dict[str, Any]]:
    resolution_date = _parse_date(row.get("resolution_date"))
    if resolution_date is None:
        return None

    answer = str(row.get("answer") or "").strip()
    if not answer:
        return None

    sim_date = global_sim_date or (resolution_date - timedelta(days=lookback_days))
    if sim_date >= resolution_date:
        return None

    search_chunk_tokens = read_search_chunk_tokens(search_db)
    question_id = str(row.get("qid") or "")
    question_title = str(row.get("question_title") or "").strip()
    background = str(row.get("background") or "").strip()
    resolution_criteria = str(row.get("resolution_criteria") or "").strip()
    answer_type = str(row.get("answer_type") or "").strip()
    prompt_agent = QwenAllQAgent(
        agent_id="skyrl_prompt_builder",
        inference_provider=None,
        config=AgentConfig(
            enable_memory=False,
            max_outcomes_per_question=int(max_outcomes_per_question),
            max_search_results=int(search_topk),
            search_cutoff_days=int(search_cutoff_days),
            warmup_max_actions=warmup_max_actions,
            warmup_max_total_tokens=warmup_max_total_tokens,
            warmup_submit_reserve_tokens=int(warmup_submit_reserve_tokens),
            warmup_force_submit_threshold_tokens=int(warmup_force_submit_threshold_tokens),
        ),
        model_name=str(budget_model_path or ""),
        search_tool=SimpleNamespace(
            is_available=bool(search_db),
            chunk_tokens=search_chunk_tokens,
        ),
    )
    prompt_question = SimpleNamespace(
        qid=question_id,
        title=question_title,
        background=background,
        resolution_criteria=resolution_criteria,
        answer_type=answer_type,
    )
    user_prompt = prompt_agent._build_warmup_system_prompt(sim_date, prompt_question, forecast_interface=None)

    return {
        "data_source": f"openforesight-{split_name}",
        "prompt": [
            {"role": "user", "content": user_prompt},
        ],
        "env_class": OPENFORESIGHT_SEARCH_WARMUP_ENV_ID,
        "reward_spec": {
            "method": "openforesight_multiclass_brier_skill",
            "ground_truth": answer,
        },
        "question_id": str(row.get("qid") or ""),
        "question_title": str(row.get("question_title") or ""),
        "resolution_date": resolution_date.isoformat(),
        "sim_date": sim_date.isoformat(),
        "source_split": split_name,
        "global_sim_date": global_sim_date.isoformat() if global_sim_date is not None else "",
        "lookback_days": int(lookback_days),
        "max_outcomes_per_question": int(max_outcomes_per_question),
        "search_db": search_db,
        "embedding_model": embedding_model,
        "embedding_gpu_mem": float(embedding_gpu_mem),
        "embedding_max_num_seqs": int(embedding_max_num_seqs),
        "aux_cuda_visible_devices": str(aux_cuda_visible_devices or ""),
        "search_type": search_type,
        "search_topk": int(search_topk),
        "max_search_results": int(search_topk),
        "search_cutoff_days": int(search_cutoff_days),
        "search_min_days": int(search_min_days),
        "search_max_date": search_max_date.isoformat() if search_max_date is not None else "",
        "max_snippet_chars": int(max_snippet_chars),
        "allow_substring_match": bool(allow_substring_match),
        "matching": str(matching),
        "matcher": str(matcher),
        "warmup_max_actions": int(warmup_max_actions) if warmup_max_actions is not None else None,
        "warmup_max_total_tokens": int(warmup_max_total_tokens) if warmup_max_total_tokens is not None else None,
        "warmup_submit_reserve_tokens": int(warmup_submit_reserve_tokens),
        "warmup_force_submit_threshold_tokens": int(warmup_force_submit_threshold_tokens),
        "budget_model_path": str(budget_model_path or ""),
        "matcher_cache_path": str(matcher_cache_path or ""),
        "extra_info": {
            "split": split_name,
            "answer_type": str(row.get("answer_type") or ""),
        },
    }


def _build_split_dataset(
    *,
    dataset_path: str,
    split: str,
    output_path: Path,
    lookback_days: int,
    global_sim_date: Optional[date],
    search_max_date: Optional[date],
    search_db: str,
    embedding_model: str,
    embedding_gpu_mem: float,
    embedding_max_num_seqs: int,
    aux_cuda_visible_devices: str,
    search_type: str,
    search_topk: int,
    search_cutoff_days: int,
    search_min_days: int,
    resolution_start: Optional[date],
    resolution_end: Optional[date],
    max_questions: Optional[int],
    seed: int,
    max_snippet_chars: int,
    allow_substring_match: bool,
    matching: str,
    matcher: str,
    warmup_max_actions: Optional[int],
    warmup_max_total_tokens: Optional[int],
    warmup_submit_reserve_tokens: int,
    warmup_force_submit_threshold_tokens: int,
    budget_model_path: str,
    max_outcomes_per_question: int,
    matcher_cache_path: str = "",
) -> SplitBuildResult:
    df = _load_split_dataframe(dataset_path=dataset_path, split=split)
    df["_resolution_date"] = df["resolution_date"].apply(_parse_date)
    df = df.dropna(subset=["_resolution_date"])

    if resolution_start is not None:
        df = df[df["_resolution_date"] >= resolution_start]
    if resolution_end is not None:
        df = df[df["_resolution_date"] <= resolution_end]

    if max_questions is not None and len(df) > max_questions:
        df = df.sample(n=max_questions, random_state=seed)

    records = []
    for _, row in df.iterrows():
        built = _build_row(
            row,
            lookback_days=lookback_days,
            global_sim_date=global_sim_date,
            search_max_date=search_max_date,
            search_db=search_db,
            embedding_model=embedding_model,
            embedding_gpu_mem=embedding_gpu_mem,
            embedding_max_num_seqs=embedding_max_num_seqs,
            aux_cuda_visible_devices=aux_cuda_visible_devices,
            search_type=search_type,
            search_topk=search_topk,
            search_cutoff_days=search_cutoff_days,
            search_min_days=search_min_days,
            max_snippet_chars=max_snippet_chars,
            allow_substring_match=allow_substring_match,
            matching=matching,
            matcher=matcher,
            warmup_max_actions=warmup_max_actions,
            warmup_max_total_tokens=warmup_max_total_tokens,
            warmup_submit_reserve_tokens=warmup_submit_reserve_tokens,
            warmup_force_submit_threshold_tokens=warmup_force_submit_threshold_tokens,
            budget_model_path=budget_model_path,
            max_outcomes_per_question=max_outcomes_per_question,
            split_name=split,
            matcher_cache_path=matcher_cache_path,
        )
        if built is not None:
            records.append(built)

    if not records:
        raise ValueError(f"No usable rows found for split='{split}' after filtering")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(records)
    ds.to_parquet(str(output_path))

    return SplitBuildResult(path=str(output_path), rows=len(records))


def prepare_openforesight_search_dataset(
    *,
    dataset_path: str,
    prepared_data_dir: str,
    search_db: str,
    embedding_model: str,
    embedding_gpu_mem: float = 0.3,
    embedding_max_num_seqs: int = 16,
    aux_cuda_visible_devices: str = "",
    train_split: str = "train",
    val_split: str = "validation",
    lookback_days: int = 7,
    global_sim_date: Optional[str] = None,
    search_max_date: Optional[str] = None,
    search_type: str = "hybrid",
    search_topk: int = 5,
    search_cutoff_days: int = 0,
    search_min_days: int = 0,
    resolution_start: Optional[str] = None,
    resolution_end: Optional[str] = None,
    max_train_questions: Optional[int] = None,
    max_val_questions: Optional[int] = None,
    seed: int = 42,
    max_snippet_chars: int = 1200,
    allow_substring_match: bool = True,
    matching: str = "exact",
    matcher: str = "",
    warmup_max_actions: Optional[int] = None,
    warmup_max_total_tokens: Optional[int] = None,
    warmup_submit_reserve_tokens: int = 8192,
    warmup_force_submit_threshold_tokens: int = 16384,
    budget_model_path: str = "",
    max_outcomes_per_question: int = 5,
    matcher_cache_path: str = "",
) -> DatasetBuildResult:
    resolution_start_d = _parse_date(resolution_start)
    resolution_end_d = _parse_date(resolution_end)
    global_sim_date_d = _parse_date(global_sim_date)
    search_max_date_d = _parse_date(search_max_date)

    output_root = Path(prepared_data_dir)
    train_path = output_root / "train.parquet"
    val_path = output_root / "validation.parquet"

    train_result = _build_split_dataset(
        dataset_path=dataset_path,
        split=train_split,
        output_path=train_path,
        lookback_days=lookback_days,
        global_sim_date=global_sim_date_d,
        search_max_date=search_max_date_d,
        search_db=search_db,
        embedding_model=embedding_model,
        embedding_gpu_mem=float(embedding_gpu_mem),
        embedding_max_num_seqs=int(embedding_max_num_seqs),
        aux_cuda_visible_devices=str(aux_cuda_visible_devices or ""),
        search_type=search_type,
        search_topk=search_topk,
        search_cutoff_days=search_cutoff_days,
        search_min_days=search_min_days,
        resolution_start=resolution_start_d,
        resolution_end=resolution_end_d,
        max_questions=max_train_questions,
        seed=seed,
        max_snippet_chars=max_snippet_chars,
        allow_substring_match=allow_substring_match,
        matching=matching,
        matcher=matcher,
        warmup_max_actions=warmup_max_actions,
        warmup_max_total_tokens=warmup_max_total_tokens,
        warmup_submit_reserve_tokens=warmup_submit_reserve_tokens,
        warmup_force_submit_threshold_tokens=warmup_force_submit_threshold_tokens,
        budget_model_path=str(budget_model_path or ""),
        max_outcomes_per_question=max_outcomes_per_question,
        matcher_cache_path=str(matcher_cache_path or ""),
    )

    val_result = _build_split_dataset(
        dataset_path=dataset_path,
        split=val_split,
        output_path=val_path,
        lookback_days=lookback_days,
        global_sim_date=global_sim_date_d,
        search_max_date=search_max_date_d,
        search_db=search_db,
        embedding_model=embedding_model,
        embedding_gpu_mem=float(embedding_gpu_mem),
        embedding_max_num_seqs=int(embedding_max_num_seqs),
        aux_cuda_visible_devices=str(aux_cuda_visible_devices or ""),
        search_type=search_type,
        search_topk=search_topk,
        search_cutoff_days=search_cutoff_days,
        search_min_days=search_min_days,
        resolution_start=resolution_start_d,
        resolution_end=resolution_end_d,
        max_questions=max_val_questions,
        seed=seed + 1,
        max_snippet_chars=max_snippet_chars,
        allow_substring_match=allow_substring_match,
        matching=matching,
        matcher=matcher,
        warmup_max_actions=warmup_max_actions,
        warmup_max_total_tokens=warmup_max_total_tokens,
        warmup_submit_reserve_tokens=warmup_submit_reserve_tokens,
        warmup_force_submit_threshold_tokens=warmup_force_submit_threshold_tokens,
        budget_model_path=str(budget_model_path or ""),
        max_outcomes_per_question=max_outcomes_per_question,
        matcher_cache_path=str(matcher_cache_path or ""),
    )

    return DatasetBuildResult(train=train_result, validation=val_result)
