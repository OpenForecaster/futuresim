"""
BasicMemory: Persistent memory for forecasting agents.

Handles loading, saving, and updating agent memory between simulation days.
"""

from pathlib import Path
from typing import Optional


class BasicMemory:
    """
    Persistent text memory for forecasting agents.
    
    Memory is the only context retained between simulation days.
    Stores as a simple text file: {agent_id}_memory.txt
    """
    
    def __init__(self, agent_id: str, memory_dir: Optional[str] = None):
        """
        Initialize memory handler.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_dir: Directory to store memory files. If None, memory is ephemeral.
        """
        self.agent_id = agent_id
        self._memory_path: Optional[Path] = None
        self._content: str = ""
        
        if memory_dir:
            self._memory_path = Path(memory_dir) / f"{agent_id}_memory.txt"
            if self._memory_path.exists():
                self._content = self._memory_path.read_text().strip()
    
    def get(self) -> str:
        """Get current memory content."""
        return self._content
    
    def update(self, new_memory: str) -> None:
        """
        Update memory with new content.
        
        This replaces the entire memory (not a diff).
        """
        self._content = new_memory.strip()
        self._save()
    
    def _save(self) -> None:
        """Persist memory to disk if configured."""
        if self._memory_path:
            self._memory_path.parent.mkdir(parents=True, exist_ok=True)
            self._memory_path.write_text(self._content)
    
    def __bool__(self) -> bool:
        """Check if memory has content."""
        return bool(self._content)
    
    def __len__(self) -> int:
        """Get memory length in characters."""
        return len(self._content)
