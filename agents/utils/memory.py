"""
Memory systems for forecasting agents.

Handles loading, saving, and updating agent memory between simulation days.

BasicMemory: Plain text per-day snapshots ({memory_dir}/memory/{YYYY-MM-DD}.txt)
StructuredMemory: YAML-based entries with metadata ({memory_dir}/memory/{YYYY-MM-DD}.yaml)
"""

from pathlib import Path
from datetime import date
from dataclasses import dataclass, asdict
from typing import Optional, List
import hashlib
import time

import yaml


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


# --- Structured Memory ---

VALID_ENTRY_TYPES = {"reasoning", "calibration", "insight", "fact"}
FIELD_LIMITS = {"name": 150, "qids": 100, "content": 800}
MAX_ENTRIES = 30


@dataclass
class MemoryEntry:
    """A single structured memory entry."""
    id: str
    name: str
    type: str
    qids: str
    content: str
    added: str  # ISO date string


class StructuredMemory:
    """
    YAML-based structured memory for forecasting agents.

    Each memory entry has metadata (id, name, type, qids) and content.
    Agents add new entries and delete stale ones instead of rewriting everything.

    Storage: {memory_dir}/memory/{YYYY-MM-DD}.yaml
    Backward compatible: falls back to loading .txt files from BasicMemory.
    """

    def __init__(self, agent_id: str, memory_dir: Optional[str] = None):
        self.agent_id = agent_id
        self._memory_dir: Optional[Path] = None
        self._entries: List[MemoryEntry] = []
        self._current_date: Optional[date] = None

        if memory_dir:
            self._memory_dir = Path(memory_dir) / "memory"

    def set_date(self, current_date: date) -> None:
        """Load the most recent memory snapshot before current_date."""
        self._current_date = current_date
        self._entries = []

        if not self._memory_dir or not self._memory_dir.exists():
            return

        # Try .yaml files first, then fall back to .txt
        most_recent_yaml = self._find_most_recent(current_date, "*.yaml")
        if most_recent_yaml:
            self._entries = self._load_yaml(most_recent_yaml)
            return

        most_recent_txt = self._find_most_recent(current_date, "*.txt")
        if most_recent_txt:
            txt_content = most_recent_txt.read_text().strip()
            file_date = date.fromisoformat(most_recent_txt.stem)
            self._entries = self._migrate_txt(txt_content, str(file_date))

    def _find_most_recent(self, current_date: date, glob_pattern: str) -> Optional[Path]:
        """Find the most recent file matching glob_pattern with date < current_date."""
        files = sorted(self._memory_dir.glob(glob_pattern))
        most_recent = None
        for f in files:
            try:
                file_date = date.fromisoformat(f.stem)
                if file_date < current_date:
                    most_recent = f
            except ValueError:
                continue
        return most_recent

    _TYPE_PRIORITY = {"reasoning": 0, "insight": 1, "fact": 2, "calibration": 3}

    def get(self) -> str:
        """Render entries into compact text for prompt injection.

        Entries are grouped by type priority (reasoning > insight > fact > calibration)
        so evidence for active predictions appears first in the context window.
        """
        if not self._entries:
            return ""
        sorted_entries = sorted(
            self._entries,
            key=lambda e: (self._TYPE_PRIORITY.get(e.type, 9), e.added),
        )
        lines = []
        for e in sorted_entries:
            qids_part = f", {e.qids}" if e.qids else ""
            lines.append(f"[{e.id}] ({e.type}{qids_part}) {e.name}")
            lines.append(f"  {e.content} (added: {e.added})")
            lines.append("")
        return "\n".join(lines).strip()

    def add_entry(self, name: str, entry_type: str, qids: str, content: str) -> str:
        """
        Add a new memory entry. Returns the generated entry ID.

        Fields are truncated to their character limits.
        If entry count exceeds MAX_ENTRIES, the oldest entry is dropped.
        """
        # Validate type
        if entry_type not in VALID_ENTRY_TYPES:
            entry_type = "insight"  # Default fallback

        # Truncate fields
        name = name.strip()[:FIELD_LIMITS["name"]]
        qids = qids.strip()[:FIELD_LIMITS["qids"]]
        content = content.strip()[:FIELD_LIMITS["content"]]

        entry_id = self._generate_id()
        entry = MemoryEntry(
            id=entry_id,
            name=name,
            type=entry_type,
            qids=qids,
            content=content,
            added=str(self._current_date) if self._current_date else "",
        )
        self._entries.append(entry)

        # Enforce max entries (drop oldest first)
        while len(self._entries) > MAX_ENTRIES:
            self._entries.pop(0)

        self._save(self._current_date)
        return entry_id

    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry by ID. Returns True if found and deleted."""
        entry_id = entry_id.strip()
        for i, e in enumerate(self._entries):
            if e.id == entry_id:
                self._entries.pop(i)
                self._save(self._current_date)
                return True
        return False

    def update(self, new_memory: str, save_date: Optional[date] = None) -> None:
        """
        Backward-compat full replacement.

        If new_memory looks like YAML (list of dicts), parse it.
        Otherwise treat as plain text and create a single migrated entry.
        """
        new_memory = new_memory.strip()
        if not new_memory:
            self._entries = []
            self._save(save_date or self._current_date)
            return

        # Try YAML parse first
        try:
            data = yaml.safe_load(new_memory)
            if isinstance(data, list) and all(isinstance(d, dict) for d in data):
                self._entries = self._parse_entry_list(data)
                self._save(save_date or self._current_date)
                return
        except yaml.YAMLError:
            pass

        # Fallback: treat as plain text migration
        added = str(save_date or self._current_date or "")
        self._entries = self._migrate_txt(new_memory, added)
        self._save(save_date or self._current_date)

    def _save(self, save_date: Optional[date]) -> None:
        """Persist entries as YAML to disk."""
        if not self._memory_dir or not save_date:
            return
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        path = self._memory_dir / f"{save_date}.yaml"
        data = [asdict(e) for e in self._entries]
        path.write_text(yaml.safe_dump(data, default_flow_style=False, allow_unicode=True))

    def _load_yaml(self, path: Path) -> List[MemoryEntry]:
        """Load entries from a YAML file."""
        try:
            data = yaml.safe_load(path.read_text())
            if not isinstance(data, list):
                return []
            return self._parse_entry_list(data)
        except (yaml.YAMLError, Exception):
            return []

    def _parse_entry_list(self, data: list) -> List[MemoryEntry]:
        """Parse a list of dicts into MemoryEntry objects."""
        entries = []
        for d in data:
            if not isinstance(d, dict):
                continue
            try:
                entries.append(MemoryEntry(
                    id=str(d.get("id", self._generate_id())),
                    name=str(d.get("name", ""))[:FIELD_LIMITS["name"]],
                    type=str(d.get("type", "insight")) if str(d.get("type", "insight")) in VALID_ENTRY_TYPES else "insight",
                    qids=str(d.get("qids", ""))[:FIELD_LIMITS["qids"]],
                    content=str(d.get("content", ""))[:FIELD_LIMITS["content"]],
                    added=str(d.get("added", "")),
                ))
            except Exception:
                continue
        return entries

    def _migrate_txt(self, txt_content: str, added_date: str) -> List[MemoryEntry]:
        """Convert plain text memory into structured entries (one per paragraph)."""
        if not txt_content:
            return []

        paragraphs = [p.strip() for p in txt_content.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [txt_content.strip()]

        entries = []
        for i, para in enumerate(paragraphs):
            # Use first line (up to limit) as name, rest as content
            lines = para.split("\n", 1)
            name = lines[0].strip()[:FIELD_LIMITS["name"]]
            content = (lines[1].strip() if len(lines) > 1 else para.strip())[:FIELD_LIMITS["content"]]
            if not name:
                name = f"Migrated entry {i+1}"

            entries.append(MemoryEntry(
                id=self._generate_id(),
                name=name,
                type="insight",
                qids="",
                content=content,
                added=added_date,
            ))

        return entries[:MAX_ENTRIES]

    def _generate_id(self) -> str:
        """Generate an 8-char hex ID."""
        raw = f"{self.agent_id}:{time.time_ns()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:8]

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __len__(self) -> int:
        """Total rendered character count."""
        return len(self.get())

    @property
    def entry_count(self) -> int:
        return len(self._entries)
