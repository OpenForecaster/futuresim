from .base import DataFetcher, CachedQuestion
from datasets import load_dataset
from datetime import datetime, date
import os

class OpenForesightFetcher(DataFetcher):
    """Fetcher for the standard OpenForesight dataset (HuggingFace/disk)."""
    
    @property
    def source_name(self) -> str:
        return "openforesight"
    
    def fetch_new(self, start_date: date, end_date: date):
        raise NotImplementedError("OpenForesight data is static/pre-downloaded.")
        
    def get_prompt_context(self) -> str:
        return "" # No special context, standard rules apply
        
    def load_from_cache(self, cache_dir: str, resolution_start: date = None, 
                       resolution_end: date = None, min_forecasters: int = 0,
                       resolved_only: bool = False) -> list[CachedQuestion]:
        """Load from the dataset_path (HF dataset) instead of parquet cache."""
        
        # Determine path or dataset name
        path = self.dataset_path or "openforesight" 
        print(f"Loading OpenForesight data from: {path} (split={self.split})")
        
        ds = load_dataset(path, split=self.split)
        
        questions = []
        for item in ds:
            # Parse date
            r_date = self._parse_date(item.get('resolution_date'))
            if not r_date:
                continue
                
            # Filter
            if resolution_start and r_date < resolution_start:
                continue
            if resolution_end and r_date > resolution_end:
                continue
                
            questions.append(CachedQuestion(
                qid=str(item.get('qid')),
                title=item.get('question_title', ''),
                background=item.get('background', ''),
                resolution_criteria=item.get('resolution_criteria', ''),
                answer_type=item.get('answer_type', ''),
                resolution_date=r_date,
                ground_truth_answer=item.get('answer', ''),
                options=None, # OpenForesight (current) doesn't use options column
                source="openforesight",
                prompt=item.get('prompt', '') or ""
            ))
            
        return questions

    def _parse_date(self, date_val) -> date:
        if date_val is None:
            return None
        if isinstance(date_val, str):
            try:
                dt = datetime.strptime(date_val, "%Y-%m-%d %H:%M:%S")
                return dt.date()
            except ValueError:
                try:
                    return datetime.strptime(date_val, "%Y-%m-%d").date()
                except ValueError:
                    return None
        elif isinstance(date_val, datetime):
            return date_val.date()
        elif isinstance(date_val, date):
            return date_val
        return None
