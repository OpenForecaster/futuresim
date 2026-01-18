"""
Timing utilities for agent performance analysis.

Tracks timing metrics for:
- LLM calls
- Search actions  
- DataFrame queries
- Daily totals

Usage:
    timer = AgentTimer()
    
    with timer.track("llm"):
        response = llm.generate(...)
    
    with timer.track("search"):
        results = search(...)
    
    with timer.track("df_query"):
        df.query(...)
    
    # At end of day
    stats = timer.get_stats()
    timer.save_day_stats(output_dir, date)
    timer.reset()  # For next day
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TimingStats:
    """Statistics for a single category of operations."""
    count: int = 0
    total_seconds: float = 0.0
    times: List[float] = field(default_factory=list)
    
    @property
    def avg_seconds(self) -> float:
        return self.total_seconds / self.count if self.count > 0 else 0.0
    
    def add(self, duration: float) -> None:
        self.count += 1
        self.total_seconds += duration
        self.times.append(duration)
    
    def reset(self) -> None:
        self.count = 0
        self.total_seconds = 0.0
        self.times = []
    
    def to_dict(self) -> Dict:
        return {
            "count": self.count,
            "total_seconds": round(self.total_seconds, 3),
            "avg_seconds": round(self.avg_seconds, 3),
            "times": [round(t, 3) for t in self.times]
        }


class AgentTimer:
    """
    Tracks timing metrics for agent operations.
    
    Categories tracked:
    - llm: LLM inference calls
    - search: Search operations
    - df_query: DataFrame query execution
    """
    
    CATEGORIES = ["llm", "search", "df_query"]
    
    def __init__(self):
        self._stats: Dict[str, TimingStats] = {
            cat: TimingStats() for cat in self.CATEGORIES
        }
        self._day_start: Optional[float] = None
        self._day_total: float = 0.0
    
    def start_day(self) -> None:
        """Mark the start of a day for total time tracking."""
        self._day_start = time.time()
    
    def end_day(self) -> None:
        """Mark the end of a day."""
        if self._day_start is not None:
            self._day_total = time.time() - self._day_start
    
    @contextmanager
    def track(self, category: str):
        """
        Context manager to track time for an operation.
        
        Usage:
            with timer.track("llm"):
                response = llm.generate(...)
        """
        if category not in self._stats:
            self._stats[category] = TimingStats()
        
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self._stats[category].add(duration)
    
    def record(self, category: str, duration: float) -> None:
        """Manually record a timing for an operation."""
        if category not in self._stats:
            self._stats[category] = TimingStats()
        self._stats[category].add(duration)
    
    def get_stats(self) -> Dict:
        """Get all timing statistics."""
        stats = {
            "day_total_seconds": round(self._day_total, 3),
        }
        for cat, timing in self._stats.items():
            stats[cat] = timing.to_dict()
        return stats
    
    def get_summary(self) -> Dict:
        """Get summary statistics (counts and averages only, no individual times)."""
        summary = {
            "day_total_seconds": round(self._day_total, 3),
        }
        for cat, timing in self._stats.items():
            summary[f"{cat}_count"] = timing.count
            summary[f"{cat}_total_seconds"] = round(timing.total_seconds, 3)
            summary[f"{cat}_avg_seconds"] = round(timing.avg_seconds, 3)
        return summary
    
    def save_day_stats(self, output_dir: str, current_date: date) -> None:
        """Save timing stats for the day to a JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stats_file = output_path / "timing_stats.jsonl"
        
        stats = self.get_summary()
        stats["date"] = str(current_date)
        
        with open(stats_file, "a") as f:
            f.write(json.dumps(stats) + "\n")
    
    def reset(self) -> None:
        """Reset all stats for next day."""
        for timing in self._stats.values():
            timing.reset()
        self._day_start = None
        self._day_total = 0.0
    
    def __repr__(self) -> str:
        summary = self.get_summary()
        lines = [f"AgentTimer (day_total: {summary['day_total_seconds']:.2f}s)"]
        for cat in self.CATEGORIES:
            count = summary.get(f"{cat}_count", 0)
            avg = summary.get(f"{cat}_avg_seconds", 0)
            total = summary.get(f"{cat}_total_seconds", 0)
            if count > 0:
                lines.append(f"  {cat}: {count} calls, {total:.2f}s total, {avg:.3f}s avg")
        return "\n".join(lines)
