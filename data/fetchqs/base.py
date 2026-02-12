from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict
import pandas as pd
import os
import json

@dataclass
class CachedQuestion:
    """Standardized question format for all platforms."""
    qid: str
    title: str
    background: str
    resolution_criteria: str
    answer_type: str  # 'binary', 'mcq', 'numeric', 'date'
    resolution_date: date
    ground_truth_answer: str = ""
    options: Optional[List[str]] = None
    source: str = ""  # metaculus, kalshi, etc.
    metadata: Optional[Dict] = None
    # Optional raw prompt from the upstream dataset (e.g. OpenForesight HF field "prompt").
    # Kept separate from background/resolution_criteria so agents can choose to use it verbatim.
    prompt: str = ""

    def to_dict(self):
        d = asdict(self)
        d['resolution_date'] = str(d['resolution_date'])
        return d

    @staticmethod
    def from_dict(d):
        d['resolution_date'] = datetime.strptime(d['resolution_date'], "%Y-%m-%d").date()
        return CachedQuestion(**d)


class DataFetcher(ABC):
    """Base class for platform-specific data fetchers."""
    
    def __init__(self, dataset_path: str = None, split: str = "train"):
        self.dataset_path = dataset_path
        self.split = split
        
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the data source (e.g. 'metaculus_binary')."""
        pass
        
    @abstractmethod
    def fetch_new(self, start_date: date, end_date: date) -> List[CachedQuestion]:
        """Fetch new data from API (or source) for the date range."""
        pass
        
    def get_prompt_context(self) -> str:
        """Return source-specific context for the agent prompt."""
        return ""
        
    def load_from_cache(self, cache_dir: str, resolution_start: date = None, 
                       resolution_end: date = None, min_forecasters: int = 0,
                       resolved_only: bool = False) -> List[CachedQuestion]:
        """Load from local parquet/arrow cache with optional filtering."""
        cache_path = os.path.join(cache_dir, f"{self.source_name}.parquet")
        
        if not os.path.exists(cache_path):
            print(f"Cache not found at {cache_path}. Returning empty list.")
            return []
            
        df = pd.read_parquet(cache_path)
        
        if df.empty or 'resolution_date' not in df.columns:
            print(f"Cache at {cache_path} is empty or invalid. Returning empty list.")
            return []
        
        # Filter by date
        # Use errors='coerce' to handle out-of-bound dates (e.g. 2300)
        df['resolution_date'] = pd.to_datetime(df['resolution_date'], errors='coerce').dt.date
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['resolution_date'])
        
        if resolution_start:
            df = df[df['resolution_date'] >= resolution_start]
        if resolution_end:
            df = df[df['resolution_date'] <= resolution_end]
            
        questions = []
        for _, row in df.iterrows():
            # Handle options JSON
            options = None
            opt_val = row.get('options')
            if opt_val is not None:
                if isinstance(opt_val, str):
                    try:
                        options = json.loads(opt_val)
                    except:
                        options = None
                elif hasattr(opt_val, 'tolist'):
                    options = opt_val.tolist()
                elif isinstance(opt_val, list):
                    options = opt_val

            # Handle metadata
            metadata = {}
            meta_val = row.get('metadata')
            if meta_val is not None:
                 if isinstance(meta_val, str):
                     try:
                         metadata = json.loads(meta_val)
                     except:
                         pass
                 elif isinstance(meta_val, dict):
                     metadata = meta_val
            
            # Runtime Filtering
            if resolved_only:
                status = metadata.get('status', 'resolved') 
                if status != 'resolved' and metadata.get('resolved') is not True:
                     continue
            
            if min_forecasters > 0:
                nf = metadata.get('nr_forecasters', 0)
                if nf < min_forecasters:
                    continue

            q = CachedQuestion(
                qid=str(row['qid']),
                title=row['title'],
                background=row['background'],
                resolution_criteria=row['resolution_criteria'],
                answer_type=row['answer_type'],
                resolution_date=row['resolution_date'],
                ground_truth_answer=row['ground_truth_answer'],
                options=options,
                source=row.get('source', self.source_name),
                metadata=metadata,
                prompt=row.get('prompt', '') or ""
            )
            questions.append(q)
            
        print(f"Loaded {len(questions)} questions from {cache_path}")
        return questions
        
    def save_to_cache(self, questions: List[CachedQuestion], cache_dir: str):
        """Save questions to local parquet cache (append/merge)."""
        cache_path = os.path.join(cache_dir, f"{self.source_name}.parquet")
        os.makedirs(cache_dir, exist_ok=True)
        
        new_df = pd.DataFrame([q.to_dict() for q in questions])
        
        if os.path.exists(cache_path):
            existing_df = pd.read_parquet(cache_path)
            # Merge logic: update existing QIDs, add new ones
            combined = pd.concat([existing_df, new_df])
            combined = combined.drop_duplicates(subset=['qid'], keep='last')
            combined.to_parquet(cache_path, index=False)
            print(f"Updated cache at {cache_path} (Total: {len(combined)})")
        else:
            new_df.to_parquet(cache_path, index=False)
            print(f"Created cache at {cache_path} (Total: {len(new_df)})")
