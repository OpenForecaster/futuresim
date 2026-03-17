from datetime import datetime, date
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass
from collections import defaultdict
import heapq
from data.fetchqs import get_fetcher

@dataclass
class Question:
    qid: str
    title: str
    background: str
    resolution_criteria: str
    answer_type: str
    resolution_date: date
    ground_truth_answer: str = ""
    options: Optional[List[str]] = None
    prompt: str = ""
    source_split: str = ""


class QuestionPool:
    """
    Manages forecasting questions loaded via DataFetcher.
    Uses heap for O(log N) resolution lookups.
    
    Args:
        dataset: Dataset name (openforesight, metaculus_binary, etc)
        dataset_path: Path for openforesight dataset
        dataset_cache: Path for cached datasets directory
        split: Dataset split to use
        resolution_start: Only include questions resolving on/after this date
        resolution_end: Only include questions resolving on/before this date
    """
    def __init__(self, 
                 dataset: str = "openforesight",
                 dataset_path: str = None,
                 dataset_cache: str = None,
                 split: str = "train",
                 prepend_train_resolution_start: date = None,
                 prepend_train_resolution_end: date = None,
                 subsample_per_month: Optional[int] = None,
                 resolution_start: date = None, 
                 resolution_end: date = None,
                 min_forecasters: int = 0,
                 resolved_only: bool = False):
        
        self.fetcher = get_fetcher(
            dataset,
            dataset_path,
            dataset_cache,
            split,
            prepend_train_resolution_start=prepend_train_resolution_start,
            prepend_train_resolution_end=prepend_train_resolution_end,
            subsample_per_month=subsample_per_month,
        )
        self.resolution_start = resolution_start
        self.resolution_end = resolution_end
        self.min_forecasters = min_forecasters
        self.resolved_only = resolved_only
        
        self._all_questions: Dict[str, Question] = {}
        # Min-heap by (resolution_date, qid) for efficient resolution lookup
        self._heap: List[tuple] = []
        # Track resolved question IDs
        self._resolved: set = set()
        
        self._load_questions(dataset_cache)
        
    def _load_questions(self, cache_dir):
        """Load questions from fetcher and build index."""
        # Note: OpenForesight fetcher ignores cache_dir
        # Metaculus fetcher needs it
        questions = self.fetcher.load_from_cache(
            cache_dir=cache_dir,
            resolution_start=self.resolution_start,
            resolution_end=self.resolution_end,
            min_forecasters=self.min_forecasters,
            resolved_only=self.resolved_only
        )
        
        for q_data in questions:
            q = Question(
                qid=str(q_data.qid),
                title=q_data.title,
                background=q_data.background,
                resolution_criteria=q_data.resolution_criteria,
                answer_type=q_data.answer_type,
                resolution_date=q_data.resolution_date,
                ground_truth_answer=q_data.ground_truth_answer,
                options=q_data.options,
                prompt=getattr(q_data, "prompt", "") or "",
                source_split=getattr(q_data, "source_split", "") or "",
            )
            
            self._all_questions[q.qid] = q
            heapq.heappush(self._heap, (q.resolution_date, q.qid))

    def pop_resolving(self, current_date: date) -> List[Question]:
        """
        Pop and return all questions that resolve on current_date.
        O(K log N) where K = number of resolving questions.
        """
        resolving = []
        while self._heap and self._heap[0][0] <= current_date:
            res_date, qid = heapq.heappop(self._heap)
            if qid in self._resolved:
                continue  # Already resolved (shouldn't happen, but safety)
            if res_date == current_date:
                self._resolved.add(qid)
                resolving.append(self._all_questions[qid])
            # Questions with res_date < current_date are past due, also resolve
            elif res_date < current_date:
                self._resolved.add(qid)
                resolving.append(self._all_questions[qid])
        return resolving
    
    def get_active(self) -> List[Question]:
        """
        Get all active (not yet resolved) questions.
        O(N) but returns list for iteration.
        """
        return [
            self._all_questions[qid] 
            for _, qid in self._heap 
            if qid not in self._resolved
        ]
    
    def get_active_ids(self) -> set:
        """Get set of active question IDs. O(N)."""
        return {qid for _, qid in self._heap if qid not in self._resolved}
    
    def get_question(self, qid: str) -> Optional[Question]:
        """Get a specific question by ID."""
        return self._all_questions.get(qid)
    
    def get_date_range(self) -> tuple:
        """Return (min_date, max_date) of all questions."""
        if not self._all_questions:
            return None, None
        dates = [q.resolution_date for q in self._all_questions.values()]
        return min(dates), max(dates)
    
    @property
    def total_count(self) -> int:
        return len(self._all_questions)
    
    @property
    def resolved_count(self) -> int:
        return len(self._resolved)
    
    @property
    def active_count(self) -> int:
        return self.total_count - self.resolved_count

    def reset(self):
        """Reset the pool to its initial state (all questions unresolved)."""
        self._resolved.clear()
        self._heap = []
        for q in self._all_questions.values():
            heapq.heappush(self._heap, (q.resolution_date, q.qid))
