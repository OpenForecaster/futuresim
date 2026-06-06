"""Optional hybrid LanceDB search service for hosted adapters."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from .handler import SearchHandler


class _EmbeddingOutput:
    """Mimic vLLM's embedding output shape for LanceDBSearchTool."""

    def __init__(self, embedding):
        self.outputs = type("EmbeddingOutputs", (), {"embedding": embedding})()


class VLLMEmbeddingClient:
    """Lightweight client for an already-running OpenAI-compatible embedding server."""

    def __init__(self, server_url: str, model_name: str = ""):
        self._url = server_url.rstrip("/")
        if self._url.endswith("/v1"):
            self._url = self._url[:-3].rstrip("/")
        self._model = model_name

    def embed(self, texts: list[str], use_tqdm: bool = False) -> list[_EmbeddingOutput]:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        payload = json.dumps({"input": texts, "model": self._model}).encode()
        req = urllib.request.Request(
            f"{self._url}/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return [_EmbeddingOutput(item["embedding"]) for item in data["data"]]


@dataclass
class HybridSearchConfig:
    search_db: str = ""
    embedding_model: str = ""
    embedding_server_url: str = ""
    search_type: str = "hybrid"
    search_cutoff_days: int = 0
    max_results: int = 5


class HybridSearchService:
    """Date-capped hybrid search wrapper used by ORS/Verifiers/hosted tools."""

    def __init__(self, config: HybridSearchConfig):
        if config.search_type != "hybrid":
            raise ValueError("Hosted search supports only search_type='hybrid'.")
        if not config.search_db:
            raise ValueError("Hybrid search requires search_db.")
        if not Path(config.search_db).exists():
            raise FileNotFoundError(f"Hybrid search_db not found: {config.search_db}")

        from .lancedb.store import LanceDBSearchTool

        embedding_model = None
        if config.embedding_server_url:
            embedding_model = VLLMEmbeddingClient(
                config.embedding_server_url,
                model_name=config.embedding_model,
            )

        search_tool = LanceDBSearchTool(
            config.search_db,
            embedding_model=embedding_model,
            model_path=config.embedding_model if not embedding_model else None,
        )
        if not search_tool.is_available:
            raise RuntimeError(f"LanceDB search table is not available at {config.search_db}")

        self.config = config
        self.handler = SearchHandler(
            search_tool=search_tool,
            search_cutoff_days=max(0, int(config.search_cutoff_days or 0)),
        )

    def set_date(self, current_date: date) -> None:
        self.handler.set_date(current_date)

    def search_news(
        self,
        query: str,
        *,
        max_results: Optional[int] = None,
        min_date: Optional[date] = None,
        max_date: Optional[date] = None,
    ) -> str:
        result_text, error = self.handler.search(
            query,
            max_results=max_results or self.config.max_results,
            search_type="hybrid",
            min_date=min_date,
            max_date=max_date,
        )
        if error:
            return f"Search error: {error}"
        return result_text or "No articles found matching your query."
