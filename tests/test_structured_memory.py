"""Tests for StructuredMemory and extract_memory_ops."""

import tempfile
import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml

from agents.utils.memory import (
    StructuredMemory, MemoryEntry, VALID_ENTRY_TYPES, FIELD_LIMITS, MAX_ENTRIES,
)
from agents.utils.forecast_parser import extract_memory_ops, extract_memory


# ── StructuredMemory tests ──────────────────────────────────────────────────


@pytest.fixture
def tmp_memory_dir(tmp_path):
    """Provide a temporary memory directory."""
    return str(tmp_path)


class TestStructuredMemoryBasics:
    def test_init_ephemeral(self):
        mem = StructuredMemory("agent1")
        assert mem.entry_count == 0
        assert not mem
        assert mem.get() == ""

    def test_init_with_dir(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        assert mem.entry_count == 0

    def test_add_entry(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        entry_id = mem.add_entry(
            "Q149 PSG prediction reasoning",
            "reasoning",
            "Q149",
            "Predicted PSG 0.70 because Sky Bet implied 55%.",
        )
        assert isinstance(entry_id, str)
        assert len(entry_id) == 8
        assert mem.entry_count == 1
        assert bool(mem) is True

    def test_add_entry_truncates_fields(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        long_name = "x" * 300
        long_content = "y" * 1500
        long_qids = "z" * 200
        mem.add_entry(long_name, "fact", long_qids, long_content)
        entry = mem._entries[0]
        assert len(entry.name) == FIELD_LIMITS["name"]
        assert len(entry.content) == FIELD_LIMITS["content"]
        assert len(entry.qids) == FIELD_LIMITS["qids"]

    def test_add_entry_invalid_type_defaults_to_insight(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("test", "invalid_type", "", "content")
        assert mem._entries[0].type == "insight"

    def test_delete_entry(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        eid = mem.add_entry("test", "fact", "", "content")
        assert mem.entry_count == 1
        assert mem.delete_entry(eid) is True
        assert mem.entry_count == 0

    def test_delete_nonexistent_entry(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        assert mem.delete_entry("nonexistent") is False

    def test_max_entries_enforcement(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        ids = []
        for i in range(MAX_ENTRIES + 5):
            eid = mem.add_entry(f"entry {i}", "fact", "", f"content {i}")
            ids.append(eid)
        assert mem.entry_count == MAX_ENTRIES
        # Oldest entries should have been dropped
        remaining_ids = {e.id for e in mem._entries}
        for old_id in ids[:5]:
            assert old_id not in remaining_ids
        # Newest entries should remain
        for new_id in ids[-MAX_ENTRIES:]:
            assert new_id in remaining_ids


class TestStructuredMemoryRendering:
    def test_render_empty(self):
        mem = StructuredMemory("agent1")
        assert mem.get() == ""

    def test_render_single_entry(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("Test entry", "reasoning", "Q1", "Some content here")
        rendered = mem.get()
        assert "[" in rendered  # has ID in brackets
        assert "(reasoning, Q1)" in rendered
        assert "Test entry" in rendered
        assert "Some content here" in rendered
        assert "added: 2025-06-01" in rendered

    def test_render_entry_without_qids(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("General insight", "insight", "", "Pattern observed")
        rendered = mem.get()
        assert "(insight)" in rendered
        assert ", )" not in rendered  # no dangling comma


class TestStructuredMemoryPersistence:
    def test_yaml_roundtrip(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("Entry A", "reasoning", "Q1", "Content A")
        mem.add_entry("Entry B", "calibration", "Q2, Q3", "Content B")

        # Load into a new instance
        mem2 = StructuredMemory("agent1", tmp_memory_dir)
        mem2.set_date(date(2025, 6, 2))  # loads 2025-06-01 snapshot
        assert mem2.entry_count == 2
        assert mem2._entries[0].name == "Entry A"
        assert mem2._entries[1].name == "Entry B"
        assert mem2._entries[1].qids == "Q2, Q3"

    def test_set_date_picks_most_recent(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        # Day 1
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("Day 1", "fact", "", "First day")
        # Day 2
        mem.set_date(date(2025, 6, 2))
        # Should load day 1 entries
        assert mem.entry_count == 1
        mem.add_entry("Day 2", "fact", "", "Second day")

        # Day 3 should load day 2 snapshot (which has 2 entries)
        mem2 = StructuredMemory("agent1", tmp_memory_dir)
        mem2.set_date(date(2025, 6, 3))
        assert mem2.entry_count == 2

    def test_txt_fallback_migration(self, tmp_memory_dir):
        """Loading old .txt memory files should migrate them to entries."""
        mem_dir = Path(tmp_memory_dir) / "memory"
        mem_dir.mkdir(parents=True)
        txt_path = mem_dir / "2025-06-01.txt"
        txt_path.write_text(
            "Key observations:\n1. Sports questions resolved well.\n\n"
            "Calibration notes:\nBookmaker odds were correct 80% of the time."
        )

        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 2))
        assert mem.entry_count > 0
        # Should have migrated the text into entries
        rendered = mem.get()
        assert "Key observations" in rendered or "Calibration notes" in rendered

    def test_yaml_preferred_over_txt(self, tmp_memory_dir):
        """When both .yaml and .txt exist for same date, .yaml should win."""
        mem_dir = Path(tmp_memory_dir) / "memory"
        mem_dir.mkdir(parents=True)

        # Write a .txt file
        txt_path = mem_dir / "2025-06-01.txt"
        txt_path.write_text("Old text memory")

        # Write a .yaml file for the same date
        yaml_data = [{"id": "abcd1234", "name": "YAML entry", "type": "fact",
                       "qids": "", "content": "From YAML", "added": "2025-06-01"}]
        yaml_path = mem_dir / "2025-06-01.yaml"
        yaml_path.write_text(yaml.safe_dump(yaml_data))

        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 2))
        assert mem.entry_count == 1
        assert mem._entries[0].name == "YAML entry"

    def test_ephemeral_add_no_crash(self):
        """Adding entries without a memory_dir should work (no disk writes)."""
        mem = StructuredMemory("agent1")
        mem._current_date = date(2025, 6, 1)
        eid = mem.add_entry("test", "fact", "", "content")
        assert mem.entry_count == 1
        assert len(eid) == 8


class TestStructuredMemoryBackwardCompat:
    def test_update_with_plain_text(self, tmp_memory_dir):
        """update() with plain text should replace entries (backward compat)."""
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("old", "fact", "", "old content")
        assert mem.entry_count == 1

        mem.update("New plain text memory content.\n\nSecond paragraph.")
        assert mem.entry_count == 2  # two paragraphs -> two entries
        rendered = mem.get()
        assert "New plain text memory content" in rendered

    def test_update_with_empty_clears(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("test", "fact", "", "content")
        mem.update("")
        assert mem.entry_count == 0


# ── Parser tests ────────────────────────────────────────────────────────────


class TestExtractMemoryOps:
    def test_single_add(self):
        response = """<reasoning>Good day.</reasoning>
<memory_add>
name: Q149 prediction reasoning
type: reasoning
qids: Q149
content: Predicted 0.70 based on bookmaker data.
</memory_add>"""
        adds, deletes = extract_memory_ops(response)
        assert len(adds) == 1
        assert adds[0]["name"] == "Q149 prediction reasoning"
        assert adds[0]["type"] == "reasoning"
        assert adds[0]["qids"] == "Q149"
        assert "0.70" in adds[0]["content"]
        assert len(deletes) == 0

    def test_multiple_adds_and_deletes(self):
        response = """<reasoning>Cleanup time.</reasoning>
<memory_add>
name: Entry one
type: fact
qids: Q1
content: Fact one content.
</memory_add>
<memory_delete>abc12345</memory_delete>
<memory_add>
name: Entry two
type: calibration
qids:
content: Calibration pattern here.
</memory_add>
<memory_delete>def67890</memory_delete>"""
        adds, deletes = extract_memory_ops(response)
        assert len(adds) == 2
        assert len(deletes) == 2
        assert adds[0]["name"] == "Entry one"
        assert adds[1]["type"] == "calibration"
        assert deletes == ["abc12345", "def67890"]

    def test_missing_required_fields_skips(self):
        response = """<memory_add>
type: fact
content: Missing name field.
</memory_add>"""
        adds, deletes = extract_memory_ops(response)
        assert len(adds) == 0  # skipped because no name

    def test_no_ops_returns_empty(self):
        response = "<reasoning>Nothing to add.</reasoning>"
        adds, deletes = extract_memory_ops(response)
        assert len(adds) == 0
        assert len(deletes) == 0

    def test_multiline_content(self):
        response = """<memory_add>
name: Multi-line test
type: insight
qids: Q1, Q2
content: First line of content.
Second line of content.
Third line with numbers: 0.75, 0.80.
</memory_add>"""
        adds, deletes = extract_memory_ops(response)
        assert len(adds) == 1
        assert "Second line" in adds[0]["content"]
        assert "Third line" in adds[0]["content"]

    def test_fallback_to_old_memory_tags(self):
        """extract_memory_ops returns empty, but extract_memory still works."""
        response = """<memory>
Old style full replacement memory content.
</memory>"""
        adds, deletes = extract_memory_ops(response)
        assert len(adds) == 0
        assert len(deletes) == 0
        # But old parser still works
        old = extract_memory(response)
        assert old == "Old style full replacement memory content."

    def test_qids_not_swallowed_by_content(self):
        """Regression: qids field must not capture content text when agent omits qids."""
        response = """<memory_add>
name: Champions Cup final matchup confirmed May 4
type: fact
qids:
content: On 2025-05-04, Bordeaux-Begles defeated Toulouse 35-18 in the Champions Cup semi-final, setting up a final against Northampton on May 24, 2025.
</memory_add>"""
        adds, _ = extract_memory_ops(response)
        assert len(adds) == 1
        assert adds[0]["qids"] == ""
        assert "Bordeaux" in adds[0]["content"]

    def test_qids_no_bleed_when_missing(self):
        """When agent skips qids line entirely, content should still parse."""
        response = """<memory_add>
name: All active questions have predictions
type: insight
content: As of 2025-05-02, all 295 unresolved questions have predictions. No new submissions needed.
</memory_add>"""
        adds, _ = extract_memory_ops(response)
        assert len(adds) == 1
        assert adds[0]["qids"] == ""
        assert "295" in adds[0]["content"]

    def test_qids_prose_is_cleared(self):
        """If qids somehow contains prose (>100 chars), it should be cleared."""
        response = """<memory_add>
name: Test entry
type: fact
qids: content: On 2025-05-02, searched for Q695 (Trumps claim about obliterated Iranian facility) and Q920 but they are not in dataset
content: Real content here.
</memory_add>"""
        adds, _ = extract_memory_ops(response)
        assert len(adds) == 1
        # The qids line contains prose — should be cleared
        assert len(adds[0]["qids"]) <= 100
        assert "Real content here" in adds[0]["content"]
