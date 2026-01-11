from datasets import load_dataset
from datetime import datetime, date
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass
from collections import defaultdict
import heapq

@dataclass
class Question:
    qid: str
    title: str
    background: str
    resolution_criteria: str
    answer_type: str
    resolution_date: date
    ground_truth_answer: str = ""


class QuestionPool:
    """
    Manages forecasting questions loaded from HuggingFace.
    Uses heap for O(log N) resolution lookups.
    """
    def __init__(self, dataset_name: str, split: str = "train"):
        print(f"Loading dataset {dataset_name}...")
        self.ds = load_dataset(dataset_name, split=split)
        
        self._all_questions: Dict[str, Question] = {}
        # Min-heap by (resolution_date, qid) for efficient resolution lookup
        self._heap: List[tuple] = []
        # Track resolved question IDs
        self._resolved: set = set()
        
        print("Indexing questions...")
        self._build_index()
        print(f"Indexed {len(self._all_questions)} questions.")
        
    def _parse_date(self, date_val) -> Optional[date]:
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

    def _build_index(self):
        """Build heap and question dictionary."""
        for item in self.ds:
            r_date = self._parse_date(item.get('resolution_date'))
            
            if not r_date:
                continue
                
            q = Question(
                qid=str(item.get('qid')),
                title=item.get('question_title', ''),
                background=item.get('background', ''),
                resolution_criteria=item.get('resolution_criteria', ''),
                answer_type=item.get('answer_type', ''),
                resolution_date=r_date,
                ground_truth_answer=item.get('answer', '')
            )
            
            self._all_questions[q.qid] = q
            heapq.heappush(self._heap, (r_date, q.qid))

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
