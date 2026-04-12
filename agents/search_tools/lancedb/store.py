"""LanceDB search implementation."""

import os
import json
import re
from datetime import date
from pathlib import Path
from typing import List, Optional, Dict, Any

import lancedb

from ..base import BaseSearchTool, SearchResult, Article


class LanceDBSearchTool(BaseSearchTool):
    """LanceDB-based search with hybrid search and date filtering."""
    
    TABLE_NAME = "articles"
    _hybrid_warned = True
    
    def __init__(self, db_path: str, embedding_model=None, model_path: str = None):
        """
        Args:
            db_path: Path to LanceDB directory
            embedding_model: Pre-loaded embedding model (optional)
            model_path: Path to model for lazy loading (optional)
        """
        self._db_path = db_path
        self._embedding_model = embedding_model
        self._model_path = model_path
        self._model_loaded = embedding_model is not None
        self._db = None
        self._table = None
        self._available = False
        self._config: Dict[str, Any] = {}
        self._chunk_tokens = 512
        self._connect()
    
    @property
    def chunk_tokens(self) -> int:
        return self._chunk_tokens
    
    @property
    def config(self) -> Dict[str, Any]:
        return self._config
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def _load_embedding_model(self):
        """Lazy load embedding model on first use."""
        if self._model_loaded:
            return
        
        if not self._model_path and self._config.get("model"):
            # Use model name from LanceDB config (no hardcoded path prefix)
            self._model_path = self._config["model"]
        
        if self._model_path:
            try:
                from vllm import LLM
                print(f"[LanceDB] Loading embedding model from {self._model_path}...")
                self._embedding_model = LLM(model=self._model_path, convert="embed")
                self._model_loaded = True
                print("[LanceDB] Embedding model loaded")
            except Exception as e:
                print(f"[LanceDB] Failed to load embedding model: {e}")
    
    def _connect(self) -> None:
        if not os.path.exists(self._db_path):
            return
        
        config_path = Path(self._db_path) / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    self._config = json.load(f)
                self._chunk_tokens = self._config.get("chunk_tokens", 512)
            except Exception:
                pass
        
        try:
            self._db = lancedb.connect(self._db_path)
            if self.TABLE_NAME in self._db.table_names():
                self._table = self._db.open_table(self.TABLE_NAME)
                self._available = True
        except Exception as e:
            print(f"[LanceDB] Failed to connect: {e}")
    
    def search(self, query: str, max_results: int = 10, max_date: Optional[date] = None, 
               search_type: str = "hybrid", min_date: Optional[date] = None) -> List[SearchResult]:
        if not self._available:
            return []
        
        # Build date filter - max_date prevents future leakage, min_date narrows scope
        # Use timestamp literals (not date) to work with hybrid/FTS search (LanceDB bug #1636)
        where_clauses = []
        if max_date:
            where_clauses.append(f"date <= timestamp '{max_date.isoformat()}T23:59:59'")
        if min_date:
            where_clauses.append(f"date >= timestamp '{min_date.isoformat()}T00:00:00'")
        where = " AND ".join(where_clauses) if where_clauses else None
        
        def _execute(results_builder):
            if where:
                # Use prefilter=False consistently across search modes for reproducibility.
                # In current LanceDB hybrid, prefilter=True is wired incorrectly.
                results_builder = results_builder.where(where, prefilter=False)
            return results_builder.limit(max_results).to_list()

        def _sanitize_fts_query(q: str, *, keep_quotes: bool = True) -> str:
            # Tantivy parser treats some punctuation as operators and can throw
            # Syntax Error or unknown-field errors for strings like "St. Peter's ..."
            # or field-style terms such as "site:example.com".
            allowed = r'[^\w\s"]+' if keep_quotes else r"[^\w\s]+"
            cleaned = re.sub(allowed, " ", q or "")
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned

        def _is_fts_parser_error(err: Exception) -> bool:
            low = str(err).lower()
            return (
                "syntax error:" in low
                or "field does not exist" in low
                or "unknown field" in low
            )

        def _build_fts_retry_candidates(q: str) -> List[str]:
            """Build progressively safer fallback queries for strict FTS parsers."""
            candidates: List[str] = []
            seen = set()

            def _add(s: str) -> None:
                s2 = (s or "").strip()
                if not s2 or s2 in seen:
                    return
                seen.add(s2)
                candidates.append(s2)

            # Pass 1: punctuation-cleaned, keep phrase quotes.
            cleaned_keep_quotes = _sanitize_fts_query(q, keep_quotes=True)
            _add(cleaned_keep_quotes)

            # Pass 2: if quotes are unbalanced, drop all quotes.
            if cleaned_keep_quotes.count('"') % 2 != 0:
                _add(cleaned_keep_quotes.replace('"', " "))

            # Pass 3: fully quote-free cleaned query.
            _add(_sanitize_fts_query(q, keep_quotes=False))

            # Pass 4: strict token fallback (most parser-safe).
            tokens = re.findall(r"\w+", q or "")
            _add(" ".join(tokens))

            return candidates

        try:
            if search_type == "keyword":
                try:
                    results = self._table.search(query, query_type="fts")
                    rows = _execute(results)
                except Exception as e:
                    if _is_fts_parser_error(e):
                        retry_err: Optional[Exception] = None
                        for q2 in _build_fts_retry_candidates(query):
                            if q2 == (query or "").strip():
                                continue
                            print(f"[LanceDB] Retrying FTS with sanitized query: {q2}")
                            try:
                                results = self._table.search(q2, query_type="fts")
                                rows = _execute(results)
                                break
                            except Exception as e2:
                                retry_err = e2
                                if _is_fts_parser_error(e2):
                                    continue
                                raise
                        else:
                            if retry_err:
                                raise retry_err
                            raise
                    else:
                        raise
            else:
                # Lazy load embedding model if needed
                self._load_embedding_model()
                
                if not self._embedding_model:
                    print("[LanceDB] No embedding model available for semantic/hybrid search")
                    return []
                else:
                    # Encode query with instruction prefix (per Qwen3-Embedding docs)
                    query_embedding = self._encode_query(query)
                    if not query_embedding:
                        print("[LanceDB] Empty query embedding returned by embedding model")
                        return []
                    if search_type == "semantic":
                        results = self._table.search(query_embedding)
                        rows = _execute(results)
                    else:  # hybrid - use separate vector() and text()
                        try:
                            results = self._table.search(query_type="hybrid").vector(query_embedding).text(query)
                            rows = _execute(results)
                        except Exception as e:
                            if _is_fts_parser_error(e):
                                retry_err: Optional[Exception] = None
                                for q2 in _build_fts_retry_candidates(query):
                                    if q2 == (query or "").strip():
                                        continue
                                    print(f"[LanceDB] Retrying hybrid with sanitized query: {q2}")
                                    try:
                                        results = self._table.search(query_type="hybrid").vector(query_embedding).text(q2)
                                        rows = _execute(results)
                                        return self._to_results(rows)
                                    except Exception as e2:
                                        retry_err = e2
                                        if _is_fts_parser_error(e2):
                                            continue
                                        break
                                # All retries exhausted or non-FTS error — fall through to semantic
                            if not LanceDBSearchTool._hybrid_warned:
                                print(f"[LanceDB] Hybrid search unavailable, falling back to semantic: {e}")
                                LanceDBSearchTool._hybrid_warned = True
                            results = self._table.search(query_embedding)
                            rows = _execute(results)
            return self._to_results(rows)
        except Exception as e:
            print(f"[LanceDB] Search failed ({search_type}): {e}")
            return []
    
    def _encode_query(self, query: str) -> list:
        """Encode query with instruction prefix for retrieval."""
        # Per Qwen3-Embedding docs: queries need instruction, documents don't
        task = "Given a web search query, retrieve relevant passages that answer the query"
        instruct_query = f"Instruct: {task}\nQuery:{query}"
        
        # Check if it's a vLLM model or sentence-transformers
        if hasattr(self._embedding_model, 'embed'):
            # vLLM model - suppress progress bar
            outputs = self._embedding_model.embed([instruct_query], use_tqdm=False)
            if not outputs:
                print("[LanceDB] Embedding API returned no outputs")
                return []
            if not outputs[0].outputs.embedding:
                print("[LanceDB] Embedding API returned an empty embedding vector")
                return []
            return outputs[0].outputs.embedding
        else:
            # sentence-transformers or similar
            vec = self._embedding_model.encode(instruct_query).tolist()
            if not vec:
                print("[LanceDB] Embedding model returned an empty embedding vector")
                return []
            return vec
    
    def get_article(self, article_id: str) -> Optional[Article]:
        if not self._available:
            return None
        try:
            rows = self._table.search().where(f"id = '{article_id}'").limit(1).to_list()
            if not rows:
                return None
            r = rows[0]
            return Article(
                id=r.get("id", ""), title=r.get("title", ""), source=r.get("source", ""),
                date=r.get("date"), content=r.get("content", ""), url=r.get("url", ""),
                description=r.get("description", "")
            )
        except Exception:
            return None
    
    def count_articles(self, min_date: Optional[date] = None, max_date: Optional[date] = None) -> Optional[int]:
        if not self._available:
            return None
        try:
            where_clauses = []
            if min_date:
                where_clauses.append(f"date >= timestamp '{min_date.isoformat()}T00:00:00'")
            if max_date:
                where_clauses.append(f"date <= timestamp '{max_date.isoformat()}T23:59:59'")
            where = " AND ".join(where_clauses) if where_clauses else None
            if where:
                return self._table.count_rows(where)
            return self._table.count_rows()
        except Exception as e:
            print(f"[LanceDB] count_articles failed: {e}")
            return None

    def _to_results(self, rows: list) -> List[SearchResult]:
        return [SearchResult(
            article_id=r.get("article_id", r.get("id", "")),
            title=r.get("title", ""),
            source=r.get("source", ""),
            date=r.get("date"),
            date_publish=r.get("date_publish"),
            snippet=r.get("content", ""),
            score=r.get("_score", r.get("_distance", 0.0)),
            url=r.get("url", "")
        ) for r in rows]
