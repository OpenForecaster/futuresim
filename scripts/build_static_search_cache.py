#!/usr/bin/env python
"""Build one static title-search evidence file per question."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment.data_loader import QuestionPool
from pathing import expand_env_tree, load_repo_env, raise_for_unresolved_env_vars


def _parse_date(value: Optional[str]):
    if not value:
        return None
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _load_config(path: str) -> Dict[str, Any]:
    load_repo_env()
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    config = expand_env_tree(config)
    raise_for_unresolved_env_vars(config, path)
    return config


def _cfg(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    defaults = config.get("defaults", {}) or {}
    return defaults.get(key, config.get(key, default))


class _EmbeddingOutput:
    def __init__(self, embedding):
        self.outputs = type("obj", (), {"embedding": embedding})()


class _EmbeddingServerClient:
    def __init__(self, server_url: str, model_name: str = ""):
        self._url = server_url.rstrip("/")
        self._model = model_name

    def embed(self, texts: list, use_tqdm: bool = False) -> list:
        import json as _json
        import urllib.request

        del use_tqdm
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        payload = _json.dumps({"input": texts, "model": self._model}).encode()
        req = urllib.request.Request(
            f"{self._url}/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        return [_EmbeddingOutput(item["embedding"]) for item in data["data"]]


def _build_search_tool(args, config: Dict[str, Any]):
    from agents.search_tools.lancedb import LanceDBSearchTool

    search_db = args.search_db or config.get("search_db") or os.environ.get("FSIM_SEARCH_DB", "")
    embedding_model_path = (
        args.embedding_model
        or config.get("embedding_model")
        or os.environ.get("FSIM_EMBEDDING_MODEL", "")
    )
    search_type = str(_cfg(config, "search_type", "hybrid")).lower()

    embedding_model = None
    if search_type != "keyword":
        if args.embedding_server_url:
            embedding_model = _EmbeddingServerClient(args.embedding_server_url, embedding_model_path)
            warm = embedding_model.embed(["test"], use_tqdm=False)
            if not warm:
                raise RuntimeError("Embedding server warmup returned empty result.")
        else:
            from inference.vllm import VLLMInference

            embedding_model = VLLMInference(
                embedding_model_path,
                max_model_len=args.embedding_max_model_len,
                gpu_memory_utilization=args.embedding_gpu_mem,
                timeout=args.vllm_request_timeout,
                max_num_seqs=args.vllm_max_num_seqs,
                startup_timeout=args.vllm_startup_timeout,
                enable_prefix_caching=False,
                enforce_eager=args.vllm_enforce_eager,
            )
            warm = embedding_model.embed(["test"], use_tqdm=False)
            if not warm:
                raise RuntimeError("Embedding warmup returned empty result.")

    tool = LanceDBSearchTool(search_db, embedding_model=embedding_model)
    if not tool.is_available:
        raise RuntimeError(f"LanceDB search is not available at {search_db}")
    return tool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Simulation config to mirror.")
    parser.add_argument("--output-dir", default="", help="Override static_search_dir from config.")
    parser.add_argument("--search-db", default="", help="Override search_db.")
    parser.add_argument("--embedding-model", default="", help="Override embedding_model.")
    parser.add_argument("--embedding-server-url", default="", help="Reuse an existing /v1/embeddings server.")
    parser.add_argument("--embedding_gpu_mem", "--embedding-gpu-mem", dest="embedding_gpu_mem", type=float, default=0.4)
    parser.add_argument("--embedding_max_model_len", "--embedding-max-model-len", dest="embedding_max_model_len", type=int, default=8192)
    parser.add_argument("--vllm_request_timeout", "--vllm-request-timeout", dest="vllm_request_timeout", type=float, default=120.0)
    parser.add_argument("--vllm_startup_timeout", "--vllm-startup-timeout", dest="vllm_startup_timeout", type=float, default=300.0)
    parser.add_argument("--vllm_max_num_seqs", "--vllm-max-num-seqs", dest="vllm_max_num_seqs", type=int, default=8)
    parser.add_argument("--vllm_enforce_eager", "--vllm-enforce-eager", dest="vllm_enforce_eager", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N question limit for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing per-question files.")
    args = parser.parse_args()

    config = _load_config(args.config)
    output_dir_value = args.output_dir or _cfg(config, "static_search_dir", "")
    if not output_dir_value:
        raise ValueError("Set static_search_dir in the config or pass --output-dir.")
    output_dir = Path(output_dir_value)

    resolution_guard = int(_cfg(config, "resolution_guard", 1))
    search_type = str(_cfg(config, "search_type", "hybrid"))
    max_results = int(_cfg(config, "static_search_max_results", 5))
    search_cutoff_days = int(_cfg(config, "search_cutoff_days", 0))

    pool = QuestionPool(
        dataset=config.get("dataset", "openforesight"),
        dataset_path=config.get("dataset_path"),
        dataset_cache=config.get("dataset_cache"),
        split=config.get("split", "train"),
        prepend_train_resolution_start=_parse_date(config.get("prepend_train_resolution_start")),
        prepend_train_resolution_end=_parse_date(config.get("prepend_train_resolution_end")),
        subsample_per_month=config.get("subsample_per_month"),
        resolution_start=_parse_date(config.get("resolution_start")),
        resolution_end=_parse_date(config.get("resolution_end")),
        min_forecasters=int(config.get("min_forecasters", 0) or 0),
        resolved_only=bool(config.get("resolved_only", False)),
    )
    questions = list(pool.get_active())
    questions.sort(key=lambda q: (q.resolution_date, str(q.qid)))
    if args.limit:
        questions = questions[: args.limit]

    search_tool = _build_search_tool(args, config)

    from agents.minimalHarnessAgent.static_search import ensure_static_search_file

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.jsonl"
    index_path.unlink(missing_ok=True)

    records = []
    with open(index_path, "a") as index_f:
        for i, q in enumerate(questions, start=1):
            search_date = q.resolution_date - timedelta(days=resolution_guard)
            path, _ = ensure_static_search_file(
                output_dir=output_dir,
                search_tool=search_tool,
                q=q,
                search_date=search_date,
                search_type=search_type,
                search_cutoff_days=search_cutoff_days,
                max_results=max_results,
                overwrite=args.overwrite,
            )
            record = {
                "qid": str(q.qid),
                "title": str(q.title),
                "resolution_date": str(q.resolution_date),
                "search_date": search_date.isoformat(),
                "query": str(q.title),
                "path": str(path),
            }
            records.append(record)
            index_f.write(json.dumps(record) + "\n")
            if i % 10 == 0 or i == len(questions):
                print(f"Static search progress: {i}/{len(questions)}", flush=True)

    summary = {
        "config": args.config,
        "output_dir": str(output_dir),
        "num_questions": len(records),
        "resolution_guard": resolution_guard,
        "search_type": search_type,
        "max_results": max_results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(records)} static search file(s) to {output_dir}")


if __name__ == "__main__":
    main()
