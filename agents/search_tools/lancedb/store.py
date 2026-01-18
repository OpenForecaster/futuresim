"""LanceDB search implementation."""

import os
import json
from datetime import date
from pathlib import Path
from typing import List, Optional, Dict, Any

import lancedb

from ..base import BaseSearchTool, SearchResult, Article


class LanceDBSearchTool(BaseSearchTool):
    """LanceDB-based search with hybrid search and date filtering."""
    
    TABLE_NAME = "articles"
    
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
            # Try default path with model name from config
            self._model_path = f"/is/cluster/fast/sgoel/models/{self._config['model']}"
        
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
        where_clauses = []
        if max_date:
            where_clauses.append(f"date <= date '{max_date.isoformat()}'")
        if min_date:
            where_clauses.append(f"date >= date '{min_date.isoformat()}'")
        where = " AND ".join(where_clauses) if where_clauses else None
        
        try:
            if search_type == "keyword":
                results = self._table.search(query, query_type="fts")
            else:
                # Lazy load embedding model if needed
                self._load_embedding_model()
                
                if not self._embedding_model:
                    # Fall back to keyword search if no model
                    results = self._table.search(query, query_type="fts")
                else:
                    # Encode query with instruction prefix (per Qwen3-Embedding docs)
                    query_embedding = self._encode_query(query)
                    if search_type == "semantic":
                        results = self._table.search(query_embedding)
                    else:  # hybrid - use separate vector() and text()
                        results = self._table.search(query_type="hybrid").vector(query_embedding).text(query)
            
            if where:
                results = results.where(where)
            return self._to_results(results.limit(max_results).to_list())
        except Exception as e:
            print(f"[LanceDB] Search error: {e}")
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
            return outputs[0].outputs.embedding
        else:
            # sentence-transformers or similar
            return self._embedding_model.encode(instruct_query).tolist()
    
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
