"""Tests for StructuredMemory (skills-style, name-keyed) and memory action parsing."""

import tempfile
from datetime import date
from pathlib import Path

import pytest
import yaml

from agents.utils.memory import (
    StructuredMemory, MemoryEntry, FIELD_LIMITS, MAX_ENTRIES, _strip_xml_tags,
)
from agents.utils.forecast_parser import (
    extract_memory_ops, extract_memory, parse_action, _parse_memory_entry_body,
    extract_mem_ops, _parse_mem_body,
)


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
        assert mem.get_index() == ""

    def test_init_with_dir(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        assert mem.entry_count == 0

    def test_add_entry(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry(
            "Q149 PSG prediction",
            "Reasoning behind PSG 0.70. Use for Q149 re-forecasting.",
            "Predicted PSG 0.70 because Sky Bet implied 55%.",
        )
        assert isinstance(name, str)
        assert name == "q149-psg-prediction"  # auto-normalized
        assert mem.entry_count == 1
        assert bool(mem) is True

    def test_add_entry_truncates_fields(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        long_name = "x" * 300
        long_desc = "d" * 500
        long_content = "y" * 2000
        mem.add_entry(long_name, long_desc, long_content)
        entry = mem._entries[0]
        assert len(entry.name) <= FIELD_LIMITS["name"]
        assert len(entry.description) == FIELD_LIMITS["description"]
        assert len(entry.content) == FIELD_LIMITS["content"]

    def test_add_entry_empty_description_uses_name(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("my-title", "", "some content")
        assert mem._entries[0].description == "my-title"

    def test_add_entry_strips_xml_tags(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("<b>Bold name</b>", "<i>Italic desc</i>", "<p>content</p>")
        assert "<b>" not in mem._entries[0].name
        assert "<i>" not in mem._entries[0].description
        assert "<p>" not in mem._entries[0].content

    def test_add_entry_normalizes_name(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("Q515 Brown Flag Awards!", "desc", "content")
        assert name == "q515-brown-flag-awards"
        assert mem._entries[0].name == "q515-brown-flag-awards"

    def test_add_entry_duplicate_raises(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("my-entry", "desc", "content")
        with pytest.raises(ValueError, match="already exists"):
            mem.add_entry("my-entry", "desc2", "content2")

    def test_add_entry_reserved_word_raises(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        with pytest.raises(ValueError, match="reserved word"):
            mem.add_entry("claude-helper", "desc", "content")
        with pytest.raises(ValueError, match="reserved word"):
            mem.add_entry("anthropic-tools", "desc", "content")

    def test_delete_entry(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("test-entry", "test desc", "content")
        assert mem.entry_count == 1
        assert mem.delete_entry(name) is True
        assert mem.entry_count == 0

    def test_delete_nonexistent_entry(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        assert mem.delete_entry("nonexistent") is False

    def test_max_entries_enforcement(self, tmp_memory_dir):
        max_ent = 10
        mem = StructuredMemory("agent1", tmp_memory_dir, max_entries=max_ent)
        mem.set_date(date(2025, 6, 1))
        names = []
        for i in range(max_ent + 5):
            name = mem.add_entry(f"entry-{i}", f"desc {i}", f"content {i}")
            names.append(name)
        assert mem.entry_count == max_ent
        # Oldest entries should have been dropped
        remaining_names = {e.name for e in mem._entries}
        for old_name in names[:5]:
            assert old_name not in remaining_names
        for new_name in names[-max_ent:]:
            assert new_name in remaining_names


class TestStructuredMemoryIndex:
    def test_get_index_empty(self):
        mem = StructuredMemory("agent1")
        assert mem.get_index() == ""

    def test_get_index_single(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("test-entry", "Some description here", "Full content")
        index = mem.get_index()
        assert f"[{name}]" in index
        assert "Some description here" in index
        assert "2025-06-01" in index
        assert "Full content" not in index  # content should NOT be in index

    def test_get_index_multiple(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name1 = mem.add_entry("entry-a", "Desc A", "Content A")
        name2 = mem.add_entry("entry-b", "Desc B", "Content B")
        index = mem.get_index()
        assert f"[{name1}]" in index
        assert f"[{name2}]" in index
        assert "Desc A" in index
        assert "Desc B" in index
        assert "Content A" not in index
        assert "Content B" not in index


class TestStructuredMemoryRetrieve:
    def test_retrieve_existing(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("test-entry", "Test desc", "Full content here")
        result = mem.retrieve(name)
        assert result is not None
        assert "test-entry" in result
        assert "Test desc" in result
        assert "Full content here" in result
        assert "2025-06-01" in result

    def test_retrieve_nonexistent(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        assert mem.retrieve("nonexistent") is None

    def test_retrieve_with_whitespace(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("test-entry", "desc", "content")
        assert mem.retrieve("  test-entry  ") is not None


class TestStructuredMemoryUpdate:
    def test_update_entry_partial(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("old-name", "Old desc", "Old content")
        ok = mem.update_entry(name, content="New content")
        assert ok is True
        assert mem._entries[0].description == "Old desc"
        assert mem._entries[0].content == "New content"

    def test_update_entry_all_fields(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("old-entry", "Old desc", "Old content")
        ok = mem.update_entry(name, description="New desc", content="New content")
        assert ok is True
        assert mem._entries[0].description == "New desc"
        assert mem._entries[0].content == "New content"

    def test_update_entry_nonexistent(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        ok = mem.update_entry("nonexistent", content="new")
        assert ok is False

    def test_update_entry_strips_xml(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("test-entry", "desc", "content")
        mem.update_entry(name, description="<i>Italic</i>")
        assert mem._entries[0].description == "Italic"


class TestStructuredMemoryRendering:
    def test_render_empty(self):
        mem = StructuredMemory("agent1")
        assert mem.get() == ""

    def test_render_single_entry(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("test-entry", "Q1 reasoning desc", "Some content here")
        rendered = mem.get()
        assert "[test-entry]" in rendered
        assert "Q1 reasoning desc" in rendered
        assert "Some content here" in rendered
        assert "2025-06-01" in rendered


class TestStructuredMemoryPersistence:
    def test_yaml_roundtrip(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("entry-a", "Desc A", "Content A")
        mem.add_entry("entry-b", "Desc B", "Content B")

        # Load into a new instance
        mem2 = StructuredMemory("agent1", tmp_memory_dir)
        mem2.set_date(date(2025, 6, 2))  # loads 2025-06-01 snapshot
        assert mem2.entry_count == 2
        assert mem2._entries[0].name == "entry-a"
        assert mem2._entries[0].description == "Desc A"
        assert mem2._entries[1].name == "entry-b"

    def test_set_date_picks_most_recent(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        # Day 1
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("day-1", "desc 1", "First day")
        # Day 2
        mem.set_date(date(2025, 6, 2))
        # Should load day 1 entries
        assert mem.entry_count == 1
        mem.add_entry("day-2", "desc 2", "Second day")

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
        # Should have migrated the text into entries with descriptions
        for e in mem._entries:
            assert e.description  # should be non-empty

    def test_yaml_preferred_over_txt(self, tmp_memory_dir):
        """When both .yaml and .txt exist for same date, .yaml should win."""
        mem_dir = Path(tmp_memory_dir) / "memory"
        mem_dir.mkdir(parents=True)

        # Write a .txt file
        txt_path = mem_dir / "2025-06-01.txt"
        txt_path.write_text("Old text memory")

        # Write a .yaml file for the same date (old schema with id/type/qids)
        yaml_data = [{"id": "abcd1234", "name": "YAML entry", "type": "fact",
                       "qids": "Q1", "content": "From YAML", "added": "2025-06-01"}]
        yaml_path = mem_dir / "2025-06-01.yaml"
        yaml_path.write_text(yaml.safe_dump(yaml_data))

        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 2))
        assert mem.entry_count == 1
        # Name should be normalized (id is ignored)
        assert mem._entries[0].name == "yaml-entry"

    def test_ephemeral_add_no_crash(self):
        """Adding entries without a memory_dir should work (no disk writes)."""
        mem = StructuredMemory("agent1")
        mem._current_date = date(2025, 6, 1)
        name = mem.add_entry("test-entry", "test desc", "content")
        assert mem.entry_count == 1
        assert name == "test-entry"

    def test_yaml_no_id_field(self, tmp_memory_dir):
        """Saved YAML should not contain id field."""
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("test-entry", "desc", "content")

        yaml_path = Path(tmp_memory_dir) / "memory" / "2025-06-01.yaml"
        data = yaml.safe_load(yaml_path.read_text())
        assert "id" not in data[0]
        assert data[0]["name"] == "test-entry"


class TestStructuredMemoryBackwardCompat:
    def test_load_old_yaml_with_type_qids(self, tmp_memory_dir):
        """Old YAML files with id/type/qids and no description should auto-migrate."""
        mem_dir = Path(tmp_memory_dir) / "memory"
        mem_dir.mkdir(parents=True)
        yaml_data = [
            {"id": "aaa11111", "name": "Old entry", "type": "reasoning",
             "qids": "Q42, Q99", "content": "Some reasoning content", "added": "2025-06-01"},
        ]
        yaml_path = mem_dir / "2025-06-01.yaml"
        yaml_path.write_text(yaml.safe_dump(yaml_data))

        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 2))
        assert mem.entry_count == 1
        entry = mem._entries[0]
        assert entry.name == "old-entry"  # normalized from "Old entry"
        # description should be auto-generated from type + qids + content
        assert "reasoning" in entry.description.lower() or "Q42" in entry.description
        assert entry.content == "Some reasoning content"

    def test_load_new_yaml_with_description(self, tmp_memory_dir):
        """New YAML files with description field should load directly (id ignored)."""
        mem_dir = Path(tmp_memory_dir) / "memory"
        mem_dir.mkdir(parents=True)
        yaml_data = [
            {"id": "bbb22222", "name": "new-entry", "description": "My custom desc",
             "content": "Full content", "added": "2025-06-01"},
        ]
        yaml_path = mem_dir / "2025-06-01.yaml"
        yaml_path.write_text(yaml.safe_dump(yaml_data))

        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 2))
        assert mem.entry_count == 1
        assert mem._entries[0].description == "My custom desc"
        assert mem._entries[0].name == "new-entry"

    def test_update_with_plain_text(self, tmp_memory_dir):
        """update() with plain text should replace entries (backward compat)."""
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("old-entry", "old desc", "old content")
        assert mem.entry_count == 1

        mem.update("New plain text memory content.\n\nSecond paragraph.")
        assert mem.entry_count == 2  # two paragraphs -> two entries
        rendered = mem.get()
        assert "new-plain-text-memory-content" in rendered or "New plain text" in rendered

    def test_update_with_empty_clears(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        mem.add_entry("test-entry", "desc", "content")
        mem.update("")
        assert mem.entry_count == 0


class TestNameValidation:
    def test_lowercase_normalization(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("MY UPPERCASE NAME", "desc", "content")
        assert name == "my-uppercase-name"

    def test_special_chars_stripped(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("Q515: Brown Flag Awards!", "desc", "content")
        assert name == "q515-brown-flag-awards"

    def test_underscores_to_hyphens(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("my_test_entry", "desc", "content")
        assert name == "my-test-entry"

    def test_xml_tags_stripped(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("<name>my-entry</name>", "desc", "content")
        assert name == "my-entry"

    def test_multiple_hyphens_collapsed(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("my---entry", "desc", "content")
        assert name == "my-entry"

    def test_empty_name_raises(self, tmp_memory_dir):
        mem = StructuredMemory("agent1", tmp_memory_dir)
        mem.set_date(date(2025, 6, 1))
        with pytest.raises(ValueError):
            mem.add_entry("!!!", "desc", "content")


# ── Helper tests ───────────────────────────────────────────────────────────


class TestStripXmlTags:
    def test_basic(self):
        assert _strip_xml_tags("<b>text</b>") == "text"
        assert _strip_xml_tags("no tags") == "no tags"
        assert _strip_xml_tags("<a href='x'>link</a>") == "link"
        assert _strip_xml_tags("") == ""


# ── Parser tests: action-based memory tools ────────────────────────────────


class TestParseMemoryActions:
    def test_memory_retrieve(self):
        response = '<reasoning>Need details.</reasoning>\n<action type="memory_retrieve">q149-psg-prediction</action>'
        parsed = parse_action(response)
        assert parsed.action_type == "memory_retrieve"
        assert parsed.memory_entry_name == "q149-psg-prediction"
        assert parsed.error is None

    def test_memory_retrieve_empty(self):
        response = '<action type="memory_retrieve"></action>'
        parsed = parse_action(response)
        assert parsed.action_type == "memory_retrieve"
        assert parsed.error is not None

    def test_memory_add(self):
        response = '''<reasoning>Adding entry.</reasoning>
<action type="memory_add">
<name>q149-psg-prediction</name>
<description>Reasoning for Q149 PSG 0.70. Use when re-forecasting Champions League.</description>
<content>Predicted PSG 0.70 because Sky Bet implied 55% and Inter eliminated.</content>
</action>'''
        parsed = parse_action(response)
        assert parsed.action_type == "memory_new"
        assert parsed.memory_new_data is not None
        assert parsed.memory_new_data["name"] == "q149-psg-prediction"
        assert "Q149" in parsed.memory_new_data["description"]
        assert "Sky Bet" in parsed.memory_new_data["content"]
        assert parsed.error is None

    def test_memory_add_plain_text_format(self):
        """Plain text key: value format should still work."""
        response = '''<action type="memory_add">
name: Q149 PSG prediction
description: Reasoning for Q149 PSG 0.70.
content: Predicted PSG 0.70 because Sky Bet implied 55%.
</action>'''
        parsed = parse_action(response)
        assert parsed.action_type == "memory_new"
        assert parsed.memory_new_data is not None
        assert "Q149" in parsed.memory_new_data["name"]

    def test_memory_add_missing_fields(self):
        response = '<action type="memory_add">\n<name>title only</name>\n</action>'
        parsed = parse_action(response)
        assert parsed.action_type == "memory_new"
        assert parsed.error is not None  # missing description and content

    def test_memory_update(self):
        response = '<action type="memory_update" name="q149-psg-prediction">\n<description>Updated desc</description>\n<content>Updated content</content>\n</action>'
        parsed = parse_action(response)
        assert parsed.action_type == "memory_update"
        assert parsed.memory_entry_name == "q149-psg-prediction"
        assert parsed.memory_update_data is not None
        assert parsed.memory_update_data["description"] == "Updated desc"
        assert parsed.memory_update_data["content"] == "Updated content"
        assert parsed.error is None

    def test_memory_update_with_id_backward_compat(self):
        """Old id= attribute should still work as fallback."""
        response = '<action type="memory_update" id="q149-psg-prediction">\n<content>Updated content</content>\n</action>'
        parsed = parse_action(response)
        assert parsed.action_type == "memory_update"
        assert parsed.memory_entry_name == "q149-psg-prediction"

    def test_memory_update_no_name(self):
        response = '<action type="memory_update">\n<content>new content</content>\n</action>'
        parsed = parse_action(response)
        assert parsed.action_type == "memory_update"
        assert parsed.error is not None  # missing name attribute

    def test_memory_update_partial(self):
        response = '<action type="memory_update" name="q149-psg-prediction">\n<content>Only updating content</content>\n</action>'
        parsed = parse_action(response)
        assert parsed.action_type == "memory_update"
        assert parsed.memory_update_data == {"content": "Only updating content"}
        assert "name" not in parsed.memory_update_data

    def test_memory_delete(self):
        response = '<action type="memory_delete">q149-psg-prediction</action>'
        parsed = parse_action(response)
        assert parsed.action_type == "memory_delete"
        assert parsed.memory_entry_name == "q149-psg-prediction"
        assert parsed.error is None

    def test_memory_delete_empty(self):
        response = '<action type="memory_delete"></action>'
        parsed = parse_action(response)
        assert parsed.action_type == "memory_delete"
        assert parsed.error is not None


class TestParseMemoryEntryBody:
    def test_all_fields_xml(self):
        body = "<name>Title</name>\n<description>Short desc</description>\n<content>Full content here</content>"
        result = _parse_memory_entry_body(body, require_all=True)
        assert result is not None
        assert result["name"] == "Title"
        assert result["description"] == "Short desc"
        assert result["content"] == "Full content here"

    def test_all_fields_plain(self):
        body = "name: Title\ndescription: Short desc\ncontent: Full content here"
        result = _parse_memory_entry_body(body, require_all=True)
        assert result is not None
        assert result["name"] == "Title"
        assert result["description"] == "Short desc"
        assert result["content"] == "Full content here"

    def test_xml_no_closing_tags(self):
        """XML without closing tags should auto-close at next opening tag."""
        body = "<name>Title\n<description>Short desc\n<content>Full content here"
        result = _parse_memory_entry_body(body, require_all=True)
        assert result is not None
        assert result["name"] == "Title"
        assert result["description"] == "Short desc"
        assert result["content"] == "Full content here"

    def test_multiline_content(self):
        body = "<name>Title</name>\n<description>Desc</description>\n<content>Line 1\nLine 2\nLine 3</content>"
        result = _parse_memory_entry_body(body, require_all=True)
        assert result is not None
        assert "Line 2" in result["content"]
        assert "Line 3" in result["content"]

    def test_multiline_description(self):
        body = "name: Title\ndescription: First part\nSecond part\ncontent: Content"
        result = _parse_memory_entry_body(body, require_all=True)
        assert result is not None
        assert "Second part" in result["description"]

    def test_partial_for_update(self):
        body = "<content>Just updating content</content>"
        result = _parse_memory_entry_body(body, require_all=False)
        assert result is not None
        assert result["content"] == "Just updating content"
        assert "name" not in result

    def test_require_all_fails_partial(self):
        body = "<name>Only name</name>"
        result = _parse_memory_entry_body(body, require_all=True)
        assert result is None  # missing description and content

    def test_empty_body(self):
        assert _parse_memory_entry_body("", require_all=True) is None
        assert _parse_memory_entry_body("", require_all=False) is None


# ── Parser tests: extract_memory_ops (end-of-day, backward compat) ─────────


class TestExtractMemoryOps:
    def test_new_format_add(self):
        response = """<memory_add>
name: Q149 prediction
description: Reasoning for Q149 PSG 0.70
content: Predicted 0.70 based on bookmaker data.
</memory_add>"""
        adds, deletes = extract_memory_ops(response)
        assert len(adds) == 1
        assert adds[0]["name"] == "Q149 prediction"
        assert "Q149" in adds[0]["description"]
        assert "0.70" in adds[0]["content"]
        assert len(deletes) == 0

    def test_old_format_add_backward_compat(self):
        """Old-style type/qids format should still work, folding into description."""
        response = """<memory_add>
name: Q149 prediction reasoning
type: reasoning
qids: Q149
content: Predicted 0.70 based on bookmaker data.
</memory_add>"""
        adds, deletes = extract_memory_ops(response)
        assert len(adds) == 1
        assert adds[0]["name"] == "Q149 prediction reasoning"
        # Old type/qids should be folded into description
        assert "description" in adds[0]
        assert "0.70" in adds[0]["content"]

    def test_multiple_adds_and_deletes(self):
        response = """<memory_add>
name: Entry one
description: Desc one
content: Content one.
</memory_add>
<memory_delete>q149-psg-prediction</memory_delete>
<memory_add>
name: Entry two
description: Desc two
content: Content two.
</memory_add>
<memory_delete>q515-brown-flag</memory_delete>"""
        adds, deletes = extract_memory_ops(response)
        assert len(adds) == 2
        assert len(deletes) == 2
        assert deletes == ["q149-psg-prediction", "q515-brown-flag"]

    def test_missing_required_fields_skips(self):
        response = """<memory_add>
content: Missing name and description.
</memory_add>"""
        adds, deletes = extract_memory_ops(response)
        assert len(adds) == 0

    def test_no_ops_returns_empty(self):
        response = "<reasoning>Nothing to add.</reasoning>"
        adds, deletes = extract_memory_ops(response)
        assert len(adds) == 0
        assert len(deletes) == 0

    def test_multiline_content(self):
        response = """<memory_add>
name: Multi-line test
description: Testing multiline content
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


# ── Parser tests: mem_df action types (mem_add/update/delete) ──────────────


class TestParseMemDfActions:
    """Test parse_action for mem_add, mem_update, mem_delete action types."""

    def test_mem_add_xml_tags(self):
        response = '''<action type="mem_add">
<qid>Q42</qid>
<question>Will X happen?</question>
<memory>Key evidence for prediction.</memory>
<category>politics</category>
</action>'''
        parsed = parse_action(response)
        assert parsed.action_type == "mem_add"
        assert parsed.error is None
        assert parsed.mem_data["qid"] == "Q42"
        assert parsed.mem_data["question"] == "Will X happen?"
        assert parsed.mem_data["memory"] == "Key evidence for prediction."
        assert parsed.mem_data["category"] == "politics"

    def test_mem_add_plain_text(self):
        response = '''<action type="mem_add">
qid: Q42
question: Will X happen?
memory: Key evidence for prediction.
category: politics
</action>'''
        parsed = parse_action(response)
        assert parsed.action_type == "mem_add"
        assert parsed.error is None
        assert parsed.mem_data["qid"] == "Q42"

    def test_mem_update_with_qid_attr(self):
        response = '''<action type="mem_update" qid="Q42">
<memory>Updated evidence here.</memory>
<category>economics</category>
</action>'''
        parsed = parse_action(response)
        assert parsed.action_type == "mem_update"
        assert parsed.error is None
        assert parsed.mem_qid == "Q42"
        assert parsed.mem_data["memory"] == "Updated evidence here."
        assert parsed.mem_data.get("category") == "economics"

    def test_mem_update_missing_qid(self):
        response = '<action type="mem_update">\n<memory>No qid</memory>\n</action>'
        parsed = parse_action(response)
        assert parsed.action_type == "mem_update"
        assert parsed.error is not None

    def test_mem_delete(self):
        response = '<action type="mem_delete">Q42</action>'
        parsed = parse_action(response)
        assert parsed.action_type == "mem_delete"
        assert parsed.error is None
        assert parsed.mem_qid == "Q42"

    def test_mem_delete_with_qid_attr(self):
        response = '<action type="mem_delete" qid="Q42"></action>'
        parsed = parse_action(response)
        assert parsed.action_type == "mem_delete"
        assert parsed.error is None
        assert parsed.mem_qid == "Q42"

    def test_mem_delete_empty(self):
        response = '<action type="mem_delete"></action>'
        parsed = parse_action(response)
        assert parsed.action_type == "mem_delete"
        assert parsed.error is not None


class TestParseMemBody:
    """Test _parse_mem_body helper."""

    def test_all_fields_xml(self):
        body = "<qid>Q1</qid>\n<question>Will X?</question>\n<memory>Evidence</memory>\n<category>test</category>"
        result = _parse_mem_body(body, require_question=True)
        assert result is not None
        assert result["qid"] == "Q1"
        assert result["question"] == "Will X?"
        assert result["memory"] == "Evidence"
        assert result["category"] == "test"

    def test_all_fields_plain(self):
        body = "qid: Q1\nquestion: Will X?\nmemory: Evidence\ncategory: test"
        result = _parse_mem_body(body, require_question=True)
        assert result is not None
        assert result["qid"] == "Q1"

    def test_no_confidence_field(self):
        body = "qid: Q1\nquestion: Will X?\nmemory: Evidence\nconfidence: 0.75\ncategory: test"
        result = _parse_mem_body(body, require_question=True)
        assert result is not None
        assert "confidence" not in result

    def test_missing_qid_fails(self):
        body = "<question>Will X?</question>\n<memory>Evidence</memory>"
        result = _parse_mem_body(body, require_question=True)
        assert result is None

    def test_require_question_false(self):
        body = "<memory>Just memory update</memory>"
        result = _parse_mem_body(body, require_question=False)
        assert result is not None
        assert result["memory"] == "Just memory update"


class TestExtractMemOps:
    """Test extract_mem_ops for both <mem_add> and <memo_add> tags."""

    def test_mem_add_tag(self):
        response = """<mem_add>
qid: Q42
question: Will X?
memory: Evidence.
category: test
</mem_add>"""
        adds, updates, deletes = extract_mem_ops(response)
        assert len(adds) == 1
        assert adds[0]["qid"] == "Q42"

    def test_memo_add_backward_compat(self):
        response = """<memo_add>
qid: Q42
question: Will X?
memory: Old style.
category: test
</memo_add>"""
        adds, updates, deletes = extract_mem_ops(response)
        assert len(adds) == 1
        assert adds[0]["qid"] == "Q42"

    def test_no_confidence_in_result(self):
        response = """<mem_add>
qid: Q1
question: Test
memory: Evidence
confidence: 0.9
category: test
</mem_add>"""
        adds, _, _ = extract_mem_ops(response)
        assert len(adds) == 1
        assert "confidence" not in adds[0]
