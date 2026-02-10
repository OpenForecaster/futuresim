"""
BasicMemory: Persistent memory for forecasting agents.

Handles loading, saving, and updating agent memory between simulation days.

Memory is stored per-day to support restarting simulations from a specific day.
Structure: {memory_dir}/memory/{YYYY-MM-DD}.txt
"""

from pathlib import Path
from datetime import date
from typing import Optional


class BasicMemory:
    """
    Persistent text memory for forecasting agents.
    
    Memory is the only context retained between simulation days.
    Stores per-day snapshots: {memory_dir}/memory/{date}.txt
    
    When loading, retrieves the most recent memory file before the current date.
    When saving, creates a new file for the current date.
    """
    
    def __init__(self, agent_id: str, memory_dir: Optional[str] = None):
        """
        Initialize memory handler.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_dir: Directory to store memory files. If None, memory is ephemeral.
        """
        self.agent_id = agent_id
        self._memory_dir: Optional[Path] = None
        self._content: str = ""
        self._current_date: Optional[date] = None
        
        if memory_dir:
            self._memory_dir = Path(memory_dir) / "memory"
    
    def set_date(self, current_date: date) -> None:
        """
        Set the current simulation date and load the appropriate memory.
        
        Loads the most recent memory file with a date < current_date.
        This should be called at the start of each simulation day.
        """
        self._current_date = current_date
        self._content = ""  # Reset
        
        if not self._memory_dir:
            return
            
        if not self._memory_dir.exists():
            return
        
        # Find most recent memory file before current date
        memory_files = sorted(self._memory_dir.glob("*.txt"))
        most_recent = None
        
        for mem_file in memory_files:
            try:
                # Parse date from filename (YYYY-MM-DD.txt)
                file_date = date.fromisoformat(mem_file.stem)
                if file_date < current_date:
                    most_recent = mem_file
            except ValueError:
                # Skip files that don't match the date format
                continue
        
        if most_recent:
            self._content = most_recent.read_text().strip()
    
    def get(self) -> str:
        """Get current memory content."""
        return self._content
    
    def update(self, new_memory: str, save_date: Optional[date] = None) -> None:
        """
        Update memory with new content.
        
        This replaces the entire memory (not a diff).
        Saves to a file named {save_date}.txt (defaults to current date).
        """
        self._content = new_memory.strip()
        self._save(save_date or self._current_date)
    
    def _save(self, save_date: Optional[date]) -> None:
        """Persist memory to disk if configured."""
        if not self._memory_dir or not save_date:
            return
            
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        memory_path = self._memory_dir / f"{save_date}.txt"
        memory_path.write_text(self._content)
    
    def __bool__(self) -> bool:
        """Check if memory has content."""
        return bool(self._content)
    
    def __len__(self) -> int:
        """Get memory length in characters."""
        return len(self._content)
