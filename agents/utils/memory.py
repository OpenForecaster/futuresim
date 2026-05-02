"""
Memory systems for forecasting agents.

Handles loading, saving, and updating agent memory between simulation days.

BasicMemory: Plain text per-day snapshots ({memory_dir}/memory/{YYYY-MM-DD}.txt)
StructuredMemory: YAML-based entries with metadata ({memory_dir}/memory/{YYYY-MM-DD}.yaml)
ActiveMemory: Question-specific DataFrame (mem_df) + reduced StructuredMemory for meta-insights
"""

from pathlib import Path
from datetime import date
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import csv
import re

import pandas as pd
import yaml


def _strip_xml_tags(text: str) -> str:
    """Remove XML/HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text)


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

FIELD_LIMITS = {"name": 64, "description": 256, "content": 1024}
MAX_ENTRIES = 500
RESERVED_WORDS = {"anthropic", "claude"}


@dataclass
class MemoryEntry:
    """A single structured memory entry (skills-style, name is primary key)."""
    name: str          # primary key: lowercase, hyphens, numbers only
    description: str
    content: str
    added: str  # ISO date string


class StructuredMemory:
    """
    YAML-based structured memory for forecasting agents.

    Each memory entry uses name as primary key (skills-style).
    Agents add new entries and delete stale ones instead of rewriting everything.

    Storage: {memory_dir}/memory/{YYYY-MM-DD}.yaml
    Backward compatible: falls back to loading .txt files from BasicMemory.
    """

    def __init__(self, agent_id: str, memory_dir: Optional[str] = None,
                 max_entries: int = None, field_limits: dict = None):
        self.agent_id = agent_id
        self._memory_dir: Optional[Path] = None
        self._entries: List[MemoryEntry] = []
        self._current_date: Optional[date] = None
        self._max_entries = max_entries if max_entries is not None else MAX_ENTRIES
        self._field_limits = field_limits if field_limits is not None else FIELD_LIMITS

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


    def get(self) -> str:
        """Render entries into compact text for prompt injection."""
        if not self._entries:
            return ""
        lines = []
        for e in self._entries:
            lines.append(f"[{e.name}] {e.description} ({e.added})")
            if e.content:
                lines.append(f"  {e.content}")
            lines.append("")
        return "\n".join(lines).strip()

    def get_index(self) -> str:
        """Render entry index (name + description) for prompt injection. No content."""
        if not self._entries:
            return ""
        lines = []
        for e in self._entries:
            desc = e.description or "(no description)"
            lines.append(f"[{e.name}] {desc} ({e.added})")
        return "\n".join(lines)

    def retrieve(self, entry_name: str) -> Optional[str]:
        """Retrieve full content of a single entry by name. Returns None if not found."""
        entry_name = self._normalize_lookup_name(entry_name)
        for e in self._entries:
            if e.name == entry_name:
                lines = [
                    f"[{e.name}]",
                    f"Description: {e.description}",
                    f"Content: {e.content}",
                    f"Added: {e.added}",
                ]
                return "\n".join(lines)
        return None

    def _normalize_lookup_name(self, raw: str) -> str:
        """Normalize a name for lookup (strip, lowercase, collapse whitespace to hyphens)."""
        return raw.strip().lower().replace(" ", "-")

    def add_entry(self, name: str, description: str, content: str) -> str:
        """
        Add a new memory entry. Returns the validated name on success,
        or raises ValueError on duplicate/invalid name.

        Name is auto-normalized (lowercase, hyphens). Fields are truncated.
        If entry count exceeds max_entries, the oldest entry is dropped.
        """
        name = self._validate_name(name)
        description = _strip_xml_tags(description).strip()[:self._field_limits["description"]]
        content = _strip_xml_tags(content).strip()[:self._field_limits["content"]]

        if not description:
            description = name  # Fallback: use name as description

        # Reject duplicates
        if any(e.name == name for e in self._entries):
            raise ValueError(f"Entry '{name}' already exists — use update_entry to modify it")

        entry = MemoryEntry(
            name=name,
            description=description,
            content=content,
            added=str(self._current_date) if self._current_date else "",
        )
        self._entries.append(entry)

        # Enforce max entries (drop oldest first)
        while len(self._entries) > self._max_entries:
            self._entries.pop(0)

        self._save(self._current_date)
        return name

    def update_entry(self, entry_name: str, *,
                     description: Optional[str] = None,
                     content: Optional[str] = None) -> bool:
        """Partial update of an existing entry by name. Returns True if found."""
        entry_name = self._normalize_lookup_name(entry_name)
        for e in self._entries:
            if e.name == entry_name:
                if description is not None:
                    e.description = _strip_xml_tags(description).strip()[:self._field_limits["description"]]
                if content is not None:
                    e.content = _strip_xml_tags(content).strip()[:self._field_limits["content"]]
                self._save(self._current_date)
                return True
        return False

    def delete_entry(self, entry_name: str) -> bool:
        """Delete an entry by name. Returns True if found and deleted."""
        entry_name = self._normalize_lookup_name(entry_name)
        for i, e in enumerate(self._entries):
            if e.name == entry_name:
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
        """Parse a list of dicts into MemoryEntry objects.

        Backward compatible: old YAML files with ``id``/``type``/``qids`` fields
        are handled (id is ignored, type/qids folded into description).
        """
        entries = []
        seen_names: set = set()
        for i, d in enumerate(data):
            if not isinstance(d, dict):
                continue
            try:
                raw_name = str(d.get("name", ""))
                try:
                    name = self._validate_name(raw_name) if raw_name else f"migrated-entry-{i}"
                except ValueError:
                    name = f"migrated-entry-{i}"

                # Dedup: append suffix if name collision
                base_name = name
                suffix = 2
                while name in seen_names:
                    name = f"{base_name}-{suffix}"[:self._field_limits["name"]]
                    suffix += 1
                seen_names.add(name)

                # Backward compat: generate description from old type/qids if missing
                description = d.get("description")
                if description is None:
                    parts = []
                    old_type = d.get("type", "")
                    old_qids = d.get("qids", "")
                    if old_type:
                        parts.append(f"[{old_type}]")
                    if old_qids:
                        parts.append(f"({old_qids})")
                    content_preview = str(d.get("content", ""))[:150]
                    if content_preview:
                        parts.append(content_preview)
                    description = " ".join(parts) if parts else name
                description = str(description)[:self._field_limits["description"]]

                entries.append(MemoryEntry(
                    name=name,
                    description=description,
                    content=str(d.get("content", ""))[:self._field_limits["content"]],
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
        seen_names: set = set()
        for i, para in enumerate(paragraphs):
            # Use first line (up to limit) as name, rest as content
            lines = para.split("\n", 1)
            raw_name = lines[0].strip()
            content = (lines[1].strip() if len(lines) > 1 else para.strip())[:self._field_limits["content"]]
            try:
                name = self._validate_name(raw_name) if raw_name else f"migrated-entry-{i+1}"
            except ValueError:
                name = f"migrated-entry-{i+1}"

            # Dedup
            base_name = name
            suffix = 2
            while name in seen_names:
                name = f"{base_name}-{suffix}"[:self._field_limits["name"]]
                suffix += 1
            seen_names.add(name)

            description = content[:self._field_limits["description"]]

            entries.append(MemoryEntry(
                name=name,
                description=description,
                content=content,
                added=added_date,
            ))

        return entries[:self._max_entries]

    def _validate_name(self, raw: str) -> str:
        """Normalize and validate a memory entry name (skills-style).

        Auto-normalizes: strip XML tags, lowercase, replace spaces/underscores
        with hyphens, remove invalid chars, collapse multiple hyphens,
        strip leading/trailing hyphens, truncate to limit.

        Raises ValueError if result is empty or contains reserved words.
        """
        name = _strip_xml_tags(raw).strip().lower()
        name = re.sub(r'[\s_]+', '-', name)           # spaces/underscores -> hyphens
        name = re.sub(r'[^a-z0-9-]', '', name)        # strip invalid chars
        name = re.sub(r'-{2,}', '-', name)            # collapse multiple hyphens
        name = name.strip('-')                         # no leading/trailing hyphens
        name = name[:self._field_limits["name"]]

        if not name:
            raise ValueError("Name is empty after normalization")
        for word in RESERVED_WORDS:
            if word in name:
                raise ValueError(f"Name cannot contain reserved word '{word}'")
        return name

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __len__(self) -> int:
        """Total rendered character count."""
        return len(self.get())

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# --- Active Memory ---

ACTIVE_META_MAX_ENTRIES = 15
ACTIVE_META_FIELD_LIMITS = {"name": 64, "description": 256, "content": 400}
ACTIVE_MEM_CHAR_LIMIT = 1000
MEM_DF_COLUMNS = ["qid", "question", "last_updated", "memory", "category"]


class ActiveMemory:
    """
    Combined memory: question-specific DataFrame (mem_df) + reduced StructuredMemory.

    mem_df: One row per question, keyed by qid. Stored as CSV.
    meta-insights: StructuredMemory with reduced limits (15 entries, 400-char content).

    Storage (new directory format):
      {memory_dir}/memory/{YYYY-MM-DD}/mem.csv    (question-specific notes)
      {memory_dir}/memory/{YYYY-MM-DD}/meta.yaml  (meta-insights)

    Backward compat loading from flat format:
      {memory_dir}/memory/memo_{YYYY-MM-DD}.csv
      {memory_dir}/memory/{YYYY-MM-DD}.yaml
    """

    def __init__(self, agent_id: str, memory_dir: Optional[str] = None,
                 max_entries: int = ACTIVE_META_MAX_ENTRIES):
        self.agent_id = agent_id
        self._memory_dir: Optional[Path] = None
        self._mem_df: pd.DataFrame = pd.DataFrame(columns=MEM_DF_COLUMNS)
        self._current_date: Optional[date] = None

        # Meta-insight layer — ephemeral (no independent disk writes).
        # ActiveMemory manages all persistence via save().
        self._meta = StructuredMemory(
            agent_id, memory_dir=None,
            max_entries=max_entries,
            field_limits=ACTIVE_META_FIELD_LIMITS,
        )

        if memory_dir:
            self._memory_dir = Path(memory_dir) / "memory"

    def set_date(self, current_date: date) -> None:
        """Load most recent mem CSV and meta-insight YAML before current_date."""
        self._current_date = current_date
        self._mem_df = pd.DataFrame(columns=MEM_DF_COLUMNS)
        self._meta._current_date = current_date
        self._meta._entries = []

        if not self._memory_dir or not self._memory_dir.exists():
            return

        # Find most recent date directory or flat files before current_date
        most_recent_csv = None
        most_recent_yaml = None

        # Try new directory format: memory/YYYY-MM-DD/
        for d in sorted(self._memory_dir.iterdir()):
            if d.is_dir():
                try:
                    dir_date = date.fromisoformat(d.name)
                    if dir_date < current_date:
                        csv_path = d / "mem.csv"
                        yaml_path = d / "meta.yaml"
                        if csv_path.exists():
                            most_recent_csv = csv_path
                        if yaml_path.exists():
                            most_recent_yaml = yaml_path
                except ValueError:
                    continue

        # Fall back to old flat format
        if most_recent_csv is None:
            most_recent_csv = self._find_most_recent_flat(current_date, "memo_*.csv")
        if most_recent_yaml is None:
            most_recent_yaml = self._find_most_recent_flat(current_date, "*.yaml")

        if most_recent_csv:
            self._mem_df = self._load_csv(most_recent_csv)
        if most_recent_yaml:
            self._meta._entries = self._meta._load_yaml(most_recent_yaml)

    def _find_most_recent_flat(self, current_date: date, glob_pattern: str) -> Optional[Path]:
        """Find the most recent flat file matching glob_pattern with date < current_date."""
        files = sorted(self._memory_dir.glob(glob_pattern))
        most_recent = None
        for f in files:
            if f.is_dir():
                continue
            try:
                stem = f.stem
                date_str = stem.replace("memo_", "") if stem.startswith("memo_") else stem
                file_date = date.fromisoformat(date_str)
                if file_date < current_date:
                    most_recent = f
            except ValueError:
                continue
        return most_recent

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load mem_df from a CSV file. Drops confidence column from old format."""
        try:
            df = pd.read_csv(path, dtype={"qid": str})
            # Drop old confidence column if present
            if "confidence" in df.columns:
                df = df.drop(columns=["confidence"])
            # Ensure all expected columns exist
            for col in MEM_DF_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df[MEM_DF_COLUMNS].copy()
        except Exception:
            return pd.DataFrame(columns=MEM_DF_COLUMNS)

    # --- Retrieval ---

    def get(self) -> str:
        """Render meta-insights for prompt injection (delegates to StructuredMemory)."""
        return self._meta.get()

    def get_mem_df(self) -> pd.DataFrame:
        """Return a copy of the question-specific memory DataFrame."""
        return self._mem_df.copy()

    def mem_summary(self, expanded_qids: set = None) -> str:
        """Compact summary of mem_df. Full memory text shown for expanded_qids."""
        if self._mem_df.empty:
            return "(empty)"

        expanded_qids = {str(q) for q in expanded_qids} if expanded_qids else set()
        compact_lines = []
        expanded_lines = []

        for _, row in self._mem_df.iterrows():
            qid = str(row['qid'])
            q_trunc = str(row["question"])[:50]
            if len(str(row["question"])) > 50:
                q_trunc += "..."
            cat = row.get("category", "") or ""

            if qid in expanded_qids:
                expanded_lines.append(
                    f"  [{qid}] {q_trunc} (updated: {row['last_updated']})\n"
                    f"    {row.get('memory', '')}"
                )
            else:
                compact_lines.append(
                    f"  {qid} | {q_trunc} | {row['last_updated']} | {cat}"
                )

        sections = []
        if expanded_lines:
            sections.append(
                "Entries you interacted with today (showing stored memory):\n"
                + "\n".join(expanded_lines)
            )
        if compact_lines:
            header = "  QID | Question | Last Updated | Category"
            sections.append(
                "Other entries:\n" + header + "\n" + "\n".join(compact_lines)
            )

        return "\n\n".join(sections) if sections else "(empty)"

    # --- Mutation ---

    def mem_add(self, qid: str, question: str, memory: str,
                category: str = "") -> None:
        """Add or upsert a question-specific memory entry."""
        qid = str(qid).strip()
        memory = memory.strip()[:ACTIVE_MEM_CHAR_LIMIT]
        question = question.strip()
        category = (category or "").strip()
        updated = str(self._current_date) if self._current_date else ""

        # Preserve existing category when caller doesn't provide one
        if not category:
            old = self._mem_df.loc[self._mem_df["qid"] == qid, "category"]
            if not old.empty and old.iloc[0]:
                category = str(old.iloc[0]).strip()

        # Upsert: remove existing row for this qid
        self._mem_df = self._mem_df[self._mem_df["qid"] != qid]

        new_row = pd.DataFrame([{
            "qid": qid,
            "question": question,
            "last_updated": updated,
            "memory": memory,
            "category": category,
        }])
        self._mem_df = pd.concat([self._mem_df, new_row], ignore_index=True)

    def mem_update(self, qid: str, memory: str,
                   category: str = None) -> None:
        """Update an existing mem_df entry. If qid not found, treated as add with blank question."""
        qid = str(qid).strip()
        mask = self._mem_df["qid"] == qid
        if mask.any():
            self._mem_df.loc[mask, "memory"] = memory.strip()[:ACTIVE_MEM_CHAR_LIMIT]
            self._mem_df.loc[mask, "last_updated"] = str(self._current_date) if self._current_date else ""
            if category is not None and category.strip():
                self._mem_df.loc[mask, "category"] = category.strip()
        else:
            self.mem_add(qid, "", memory, category or "")

    def mem_delete(self, qid: str) -> bool:
        """Delete a mem_df entry by qid. Returns True if found and deleted."""
        qid = str(qid).strip()
        before = len(self._mem_df)
        self._mem_df = self._mem_df[self._mem_df["qid"] != qid]
        return len(self._mem_df) < before

    # --- Meta-insight delegation ---

    def add_entry(self, name: str, description: str, content: str) -> str:
        """Add a meta-insight entry (delegates to StructuredMemory)."""
        return self._meta.add_entry(name, description, content)

    def delete_entry(self, entry_name: str) -> bool:
        """Delete a meta-insight entry by name."""
        return self._meta.delete_entry(entry_name)

    def update_entry(self, entry_name: str, **kwargs) -> bool:
        """Update a meta-insight entry (delegates to StructuredMemory)."""
        return self._meta.update_entry(entry_name, **kwargs)

    def get_index(self) -> str:
        """Render meta-insight index (delegates to StructuredMemory)."""
        return self._meta.get_index()

    def retrieve(self, entry_name: str) -> Optional[str]:
        """Retrieve full meta-insight entry (delegates to StructuredMemory)."""
        return self._meta.retrieve(entry_name)

    @property
    def _max_entries(self) -> int:
        """Max meta-insight entries (for _handle_memory_action compat)."""
        return self._meta._max_entries

    @property
    def entry_count(self) -> int:
        """Number of meta-insight entries."""
        return self._meta.entry_count

    @property
    def mem_count(self) -> int:
        """Number of question-specific mem entries."""
        return len(self._mem_df)

    # --- Persistence ---

    def save(self, save_date: Optional[date] = None) -> None:
        """Write both mem CSV and meta-insight YAML to per-day directory."""
        save_date = save_date or self._current_date
        if not self._memory_dir or not save_date:
            return

        date_dir = self._memory_dir / str(save_date)
        date_dir.mkdir(parents=True, exist_ok=True)

        # Save mem_df as CSV
        csv_path = date_dir / "mem.csv"
        self._mem_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)

        # Save meta-insights YAML
        yaml_path = date_dir / "meta.yaml"
        data = [asdict(e) for e in self._meta._entries]
        yaml_path.write_text(yaml.safe_dump(data, default_flow_style=False, allow_unicode=True))

    def _save(self, save_date: Optional[date] = None) -> None:
        """Alias for save(), for compatibility with _run_memory_update_loop."""
        self.save(save_date)

    def __bool__(self) -> bool:
        return not self._mem_df.empty or bool(self._meta)

    def __len__(self) -> int:
        """Total rendered character count of meta-insights."""
        return len(self._meta)
