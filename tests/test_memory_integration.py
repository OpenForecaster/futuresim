"""Integration tests for the skills-style memory system.

Tests the full flow: parser → memory class → agent handler wiring,
simulating what happens when a model emits memory tool calls.
"""

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agents.utils.memory import StructuredMemory, ActiveMemory, FIELD_LIMITS, MAX_ENTRIES, ACTIVE_MEM_CHAR_LIMIT, MEM_DF_COLUMNS
from agents.utils.forecast_parser import parse_action, extract_memory_ops, extract_mem_ops
from agents.utils.budget import BudgetSettings, BudgetTracker


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def mem(tmp_path):
    """StructuredMemory with a temp dir, set to 2025-06-01."""
    m = StructuredMemory("test_agent", str(tmp_path))
    m.set_date(date(2025, 6, 1))
    return m


@pytest.fixture
def populated_mem(mem):
    """Memory with 3 pre-existing entries."""
    mem.add_entry("q42-election-reasoning",
                  "Reasoning for Q42 US election prediction. Use when re-forecasting Q42.",
                  "Predicted Biden 0.55 based on polling aggregates and incumbency advantage.")
    mem.add_entry("bookmaker-calibration",
                  "Tracks bookmaker accuracy across sports Qs. Use for weighting market odds.",
                  "Bookmaker odds correct 80% across 15 sports questions. Weight them 60-70%.")
    mem.add_entry("ecb-meeting-dates",
                  "Critical ECB dates for Q72, Q108. Check before forecasting interest rate Qs.",
                  "ECB next meeting June 5. Rate decision expected. Relevant to Q72, Q108.")
    return mem


# ── End-to-end: parse → execute on memory ─────────────────────────────────

class TestParseAndExecuteRetrieve:
    """Test that a model's memory_retrieve action parses and retrieves correctly."""

    def test_retrieve_existing_entry(self, populated_mem):
        name = populated_mem._entries[0].name
        response = f'<reasoning>Need Q42 details.</reasoning>\n<action type="memory_retrieve">{name}</action>'
        parsed = parse_action(response)

        assert parsed.action_type == "memory_retrieve"
        assert parsed.memory_entry_name == name
        assert parsed.error is None

        result = populated_mem.retrieve(parsed.memory_entry_name)
        assert result is not None
        assert "q42-election-reasoning" in result
        assert "Biden 0.55" in result
        assert "polling aggregates" in result

    def test_retrieve_nonexistent_entry(self, populated_mem):
        response = '<action type="memory_retrieve">nonexistent-entry</action>'
        parsed = parse_action(response)
        result = populated_mem.retrieve(parsed.memory_entry_name)
        assert result is None

    def test_retrieve_does_not_show_in_index(self, populated_mem):
        """Full content should NOT appear in the index view."""
        index = populated_mem.get_index()
        assert "Biden 0.55" not in index
        assert "polling aggregates" not in index
        # But descriptions should appear
        assert "Reasoning for Q42" in index
        assert "bookmaker accuracy" in index.lower() or "Bookmaker" in index


class TestParseAndExecuteAdd:
    """Test that a model's memory_add action parses and adds correctly."""

    def test_add_new_entry(self, mem):
        response = '''<reasoning>Storing Q99 analysis.</reasoning>
<action type="memory_add">
<name>q99-climate-prediction</name>
<description>Reasoning for Q99 climate target prediction. Use when re-forecasting Q99.</description>
<content>Paris agreement target unlikely to be met by 2030. Based on latest IPCC report projections showing 2.1C trajectory.</content>
</action>'''
        parsed = parse_action(response)

        assert parsed.action_type == "memory_add"
        assert parsed.error is None
        data = parsed.memory_add_data
        assert data["name"] == "q99-climate-prediction"
        assert "Q99" in data["description"]

        # Execute on memory
        before = mem.entry_count
        name = mem.add_entry(data["name"], data["description"], data["content"])
        assert mem.entry_count == before + 1
        assert name == "q99-climate-prediction"

        # Verify it shows in index
        index = mem.get_index()
        assert "q99-climate-prediction" in index
        assert "IPCC" not in index  # content not in index

        # Verify full retrieve works
        full = mem.retrieve(name)
        assert "IPCC" in full
        assert "2.1C" in full

    def test_add_truncates_long_fields(self, mem):
        response = f'''<action type="memory_add">
<name>{"x" * 200}</name>
<description>{"d" * 500}</description>
<content>{"c" * 3000}</content>
</action>'''
        parsed = parse_action(response)
        data = parsed.memory_add_data
        name = mem.add_entry(data["name"], data["description"], data["content"])
        entry = mem._entries[0]
        assert len(entry.name) <= FIELD_LIMITS["name"]
        assert len(entry.description) <= FIELD_LIMITS["description"]
        assert len(entry.content) <= FIELD_LIMITS["content"]

    def test_add_strips_xml_from_desc_and_content(self, mem):
        response = '''<action type="memory_add">
<name>bold-title</name>
<description><i>Italic description</i></description>
<content>Content with <tags> stripped.</content>
</action>'''
        parsed = parse_action(response)
        data = parsed.memory_add_data
        mem.add_entry(data["name"], data["description"], data["content"])
        entry = mem._entries[0]
        assert "<i>" not in entry.description
        assert "<tags>" not in entry.content


class TestParseAndExecuteUpdate:
    """Test that a model's memory_update action parses and updates correctly."""

    def test_update_content_only(self, populated_mem):
        name = populated_mem._entries[1].name  # bookmaker-calibration
        response = f'''<action type="memory_update" name="{name}">
<content>Updated: Bookmaker odds correct 85% across 20 sports questions now. Weight them 70-80%.</content>
</action>'''
        parsed = parse_action(response)

        assert parsed.action_type == "memory_update"
        assert parsed.memory_entry_name == name
        assert "description" not in parsed.memory_update_data

        ok = populated_mem.update_entry(parsed.memory_entry_name, **parsed.memory_update_data)
        assert ok is True
        assert "85%" in populated_mem._entries[1].content
        # Description should be unchanged
        assert "bookmaker accuracy" in populated_mem._entries[1].description.lower() or "Bookmaker" in populated_mem._entries[1].description

    def test_update_multiple_fields(self, populated_mem):
        name = populated_mem._entries[2].name  # ecb-meeting-dates
        response = f'''<action type="memory_update" name="{name}">
<description>Central bank meeting dates for Q72, Q108, Q150. Check before rate Qs.</description>
<content>ECB June 5, Fed June 12. Both expected to hold. Relevant to Q72, Q108, Q150.</content>
</action>'''
        parsed = parse_action(response)
        ok = populated_mem.update_entry(parsed.memory_entry_name, **parsed.memory_update_data)
        assert ok is True
        entry = populated_mem._entries[2]
        assert "Q150" in entry.description
        assert "Fed June 12" in entry.content

    def test_update_nonexistent(self, populated_mem):
        response = '<action type="memory_update" name="nonexistent-entry">\n<content>new stuff</content>\n</action>'
        parsed = parse_action(response)
        ok = populated_mem.update_entry(parsed.memory_entry_name, **parsed.memory_update_data)
        assert ok is False


class TestParseAndExecuteDelete:
    """Test that a model's memory_delete action parses and deletes correctly."""

    def test_delete_existing(self, populated_mem):
        name = populated_mem._entries[0].name
        response = f'<action type="memory_delete">{name}</action>'
        parsed = parse_action(response)

        assert parsed.action_type == "memory_delete"
        assert parsed.memory_entry_name == name

        before = populated_mem.entry_count
        ok = populated_mem.delete_entry(parsed.memory_entry_name)
        assert ok is True
        assert populated_mem.entry_count == before - 1
        # Verify entry is gone from index
        assert name not in populated_mem.get_index()

    def test_delete_nonexistent(self, populated_mem):
        response = '<action type="memory_delete">nonexistent-entry</action>'
        parsed = parse_action(response)
        ok = populated_mem.delete_entry(parsed.memory_entry_name)
        assert ok is False
        assert populated_mem.entry_count == 3  # unchanged


# ── Multi-step scenarios ──────────────────────────────────────────────────

class TestMultiStepScenarios:
    """Simulate multi-turn sequences a model would produce."""

    def test_add_then_retrieve_then_update(self, mem):
        """Simulate: add entry → retrieve it → update it."""
        # Step 1: Add
        add_resp = '''<action type="memory_add">
<name>q200-gdp-forecast</name>
<description>GDP growth prediction for Q200. Use when forecasting economic Qs.</description>
<content>US GDP growth predicted at 2.1% for Q3 2025 based on BEA advance estimate.</content>
</action>'''
        parsed = parse_action(add_resp)
        name = mem.add_entry(
            parsed.memory_add_data["name"],
            parsed.memory_add_data["description"],
            parsed.memory_add_data["content"],
        )
        assert mem.entry_count == 1

        # Step 2: Retrieve
        retrieve_resp = f'<action type="memory_retrieve">{name}</action>'
        parsed = parse_action(retrieve_resp)
        full = mem.retrieve(parsed.memory_entry_name)
        assert "2.1%" in full
        assert "BEA" in full

        # Step 3: Update with new data
        update_resp = f'''<action type="memory_update" name="{name}">
<content>US GDP growth revised to 2.3% for Q3 2025 based on BEA second estimate released Nov 2025.</content>
</action>'''
        parsed = parse_action(update_resp)
        ok = mem.update_entry(parsed.memory_entry_name, **parsed.memory_update_data)
        assert ok is True

        # Verify update
        full = mem.retrieve(name)
        assert "2.3%" in full
        assert "second estimate" in full

    def test_add_multiple_then_delete_stale(self, mem):
        """Simulate: add 3 entries → delete the stale one."""
        names = []
        for name, desc, content in [
            ("q10-resolved", "Q10 has resolved. Can be deleted.", "Q10 resolved Yes on June 1."),
            ("q20-active", "Q20 ongoing analysis. Use for Q20.", "Q20 still uncertain, 50/50."),
            ("q30-active", "Q30 prediction reasoning. Use for Q30.", "Q30 likely Yes based on trend."),
        ]:
            resp = f'<action type="memory_add">\n<name>{name}</name>\n<description>{desc}</description>\n<content>{content}</content>\n</action>'
            parsed = parse_action(resp)
            n = mem.add_entry(
                parsed.memory_add_data["name"],
                parsed.memory_add_data["description"],
                parsed.memory_add_data["content"],
            )
            names.append(n)

        assert mem.entry_count == 3

        # Delete stale Q10 entry
        del_resp = f'<action type="memory_delete">{names[0]}</action>'
        parsed = parse_action(del_resp)
        ok = mem.delete_entry(parsed.memory_entry_name)
        assert ok is True
        assert mem.entry_count == 2

        # Index should not contain Q10
        index = mem.get_index()
        assert "q10-resolved" not in index
        assert "q20-active" in index
        assert "q30-active" in index


# ── Persistence across days ───────────────────────────────────────────────

class TestPersistenceAcrossDays:
    """Test that skills-style memory persists correctly across simulation days."""

    def test_entries_persist_across_days(self, tmp_path):
        mem_dir = str(tmp_path)

        # Day 1: add entries
        mem = StructuredMemory("agent1", mem_dir)
        mem.set_date(date(2025, 6, 1))
        name1 = mem.add_entry("day1-entry", "Added on day 1", "Content from day 1")
        assert mem.entry_count == 1

        # Day 2: load → should see day 1 entries
        mem2 = StructuredMemory("agent1", mem_dir)
        mem2.set_date(date(2025, 6, 2))
        assert mem2.entry_count == 1
        assert mem2._entries[0].name == name1
        assert mem2._entries[0].description == "Added on day 1"

        # Add another on day 2
        name2 = mem2.add_entry("day2-entry", "Added on day 2", "Content from day 2")

        # Day 3: should see both
        mem3 = StructuredMemory("agent1", mem_dir)
        mem3.set_date(date(2025, 6, 3))
        assert mem3.entry_count == 2

        # Retrieve and index should both work
        assert mem3.retrieve(name1) is not None
        assert mem3.retrieve(name2) is not None
        index = mem3.get_index()
        assert "day1-entry" in index
        assert "day2-entry" in index

    def test_updates_persist(self, tmp_path):
        mem_dir = str(tmp_path)

        # Day 1: add and update
        mem = StructuredMemory("agent1", mem_dir)
        mem.set_date(date(2025, 6, 1))
        name = mem.add_entry("original-entry", "Original desc", "Original content")
        mem.update_entry(name, content="Updated content on day 1")

        # Day 2: should see updated content
        mem2 = StructuredMemory("agent1", mem_dir)
        mem2.set_date(date(2025, 6, 2))
        full = mem2.retrieve(name)
        assert "Updated content on day 1" in full

    def test_deletes_persist(self, tmp_path):
        mem_dir = str(tmp_path)

        # Day 1: add 2 entries, delete 1
        mem = StructuredMemory("agent1", mem_dir)
        mem.set_date(date(2025, 6, 1))
        name1 = mem.add_entry("keep-entry", "Keep desc", "Keep content")
        name2 = mem.add_entry("delete-entry", "Delete desc", "Delete content")
        mem.delete_entry(name2)

        # Day 2: should only have 1 entry
        mem2 = StructuredMemory("agent1", mem_dir)
        mem2.set_date(date(2025, 6, 2))
        assert mem2.entry_count == 1
        assert mem2._entries[0].name == name1


# ── Backward compat: old YAML → new index ────────────────────────────────

class TestOldYamlToNewIndex:
    """Test that old YAML files (with id/type/qids) work with new name-based methods."""

    def test_old_yaml_get_index(self, tmp_path):
        mem_dir = Path(tmp_path) / "memory"
        mem_dir.mkdir(parents=True)
        yaml_data = [
            {"id": "aaa11111", "name": "Old reasoning entry", "type": "reasoning",
             "qids": "Q42, Q99", "content": "Detailed reasoning about Q42 and Q99.",
             "added": "2025-05-30"},
            {"id": "bbb22222", "name": "Old calibration entry", "type": "calibration",
             "qids": "", "content": "Calibration pattern: overconfident on sports Qs.",
             "added": "2025-05-30"},
        ]
        (mem_dir / "2025-05-30.yaml").write_text(yaml.safe_dump(yaml_data))

        mem = StructuredMemory("agent1", str(tmp_path))
        mem.set_date(date(2025, 6, 1))
        assert mem.entry_count == 2

        # Index should use normalized names (id ignored)
        index = mem.get_index()
        assert "old-reasoning-entry" in index
        assert "old-calibration-entry" in index
        # Description should be auto-generated from type/qids
        assert "reasoning" in index.lower() or "Q42" in index

        # Retrieve by normalized name should work
        full = mem.retrieve("old-reasoning-entry")
        assert full is not None
        assert "Detailed reasoning" in full

    def test_old_yaml_can_be_updated(self, tmp_path):
        mem_dir = Path(tmp_path) / "memory"
        mem_dir.mkdir(parents=True)
        yaml_data = [
            {"id": "ccc33333", "name": "Old entry", "type": "fact",
             "qids": "Q50", "content": "Old fact content.", "added": "2025-05-30"},
        ]
        (mem_dir / "2025-05-30.yaml").write_text(yaml.safe_dump(yaml_data))

        mem = StructuredMemory("agent1", str(tmp_path))
        mem.set_date(date(2025, 6, 1))

        # Update the entry using normalized name
        ok = mem.update_entry("old-entry", description="Updated description", content="New fact content.")
        assert ok is True
        assert mem._entries[0].description == "Updated description"
        assert mem._entries[0].content == "New fact content."


# ── Budget integration ────────────────────────────────────────────────────

class TestBudgetIntegration:
    """Test that memory actions integrate with budget tracking."""

    def test_memory_action_consumes_action_budget(self):
        budget = BudgetTracker(BudgetSettings(max_actions=5))
        assert budget.actions_remaining == 5
        budget.consume_action()  # simulate memory_retrieve
        assert budget.actions_remaining == 4
        budget.consume_action()  # simulate memory_add
        assert budget.actions_remaining == 3

    def test_memory_mini_loop_token_only_budget(self):
        """End-of-day mini loop uses token-only budget (max_actions=None)."""
        budget = BudgetTracker(BudgetSettings(
            max_total_tokens=50000,
            submit_reserve_tokens=8192,
            force_submit_threshold_tokens=16384,
        ))
        # consume_action should be a no-op
        budget.consume_action()
        assert budget.actions_remaining is None
        assert not budget.is_exhausted()


# ── End-of-day extract_memory_ops compat ──────────────────────────────────

class TestEndOfDayOpsCompat:
    """Test that extract_memory_ops works with both old and new formats."""

    def test_new_format_add_then_apply(self, mem):
        response = """<reasoning>End of day cleanup.</reasoning>
<memory_add>
name: new-insight
description: Cross-question pattern about sports forecasting
content: Sports questions with bookmaker odds available resolve 80% in line with odds.
</memory_add>
<memory_delete>nonexistent-entry</memory_delete>"""

        adds, deletes = extract_memory_ops(response)
        assert len(adds) == 1
        assert len(deletes) == 1

        # Apply adds
        for add in adds:
            mem.add_entry(add["name"], add.get("description", add["name"]), add["content"])
        assert mem.entry_count == 1
        assert "sports forecasting" in mem.get_index().lower() or "Cross-question" in mem.get_index()

    def test_old_format_add_then_apply(self, mem):
        """Old-style type/qids format should still work end-to-end."""
        response = """<memory_add>
name: Legacy entry
type: calibration
qids: Q1, Q2
content: Old style entry that should still parse.
</memory_add>"""

        adds, deletes = extract_memory_ops(response)
        assert len(adds) == 1
        # Should have description (auto-generated from type/qids)
        assert "description" in adds[0]

        for add in adds:
            mem.add_entry(add["name"], add.get("description", add["name"]), add["content"])
        assert mem.entry_count == 1


# ── ActiveMemory delegation ──────────────────────────────────────────────

class TestActiveMemoryDelegation:
    """Test that ActiveMemory delegates index/retrieve/update to its meta StructuredMemory."""

    def test_active_memory_index_and_retrieve(self, tmp_path):
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))

        name = amem.add_entry("meta-insight", "Cross-question calibration pattern", "Details here")
        assert amem.entry_count == 1

        index = amem.get_index()
        assert "meta-insight" in index
        assert "Details here" not in index

        full = amem.retrieve(name)
        assert "Details here" in full

    def test_active_memory_update_entry(self, tmp_path):
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))

        name = amem.add_entry("old-meta", "Old desc", "Old content")
        ok = amem.update_entry(name, content="Updated meta content")
        assert ok is True
        full = amem.retrieve(name)
        assert "Updated meta content" in full


# ── ActiveMemory mem_df operations ────────────────────────────────────────

class TestActiveMemoryMemDf:
    """Test mem_add/update/delete on the per-question mem_df layer."""

    def test_mem_add_and_count(self, tmp_path):
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q42", question="Will X happen?", memory="Key evidence here", category="politics")
        assert amem.mem_count == 1
        df = amem.get_mem_df()
        assert len(df) == 1
        assert df.iloc[0]["qid"] == "Q42"
        assert df.iloc[0]["memory"] == "Key evidence here"
        assert df.iloc[0]["category"] == "politics"

    def test_mem_add_no_confidence_column(self, tmp_path):
        """mem_df should not have a confidence column."""
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q1", question="Test Q", memory="Test mem", category="test")
        df = amem.get_mem_df()
        assert "confidence" not in df.columns
        assert list(df.columns) == MEM_DF_COLUMNS

    def test_mem_update(self, tmp_path):
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q42", question="Will X happen?", memory="Old evidence", category="politics")
        amem.mem_update(qid="Q42", memory="New evidence with updates")
        df = amem.get_mem_df()
        assert df.iloc[0]["memory"] == "New evidence with updates"

    def test_mem_update_category(self, tmp_path):
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q42", question="Will X happen?", memory="Evidence", category="politics")
        amem.mem_update(qid="Q42", memory="Evidence", category="economics")
        df = amem.get_mem_df()
        assert df.iloc[0]["category"] == "economics"

    def test_mem_update_preserves_category_when_empty(self, tmp_path):
        """mem_update with category='' should preserve existing category."""
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q42", question="Will X happen?", memory="Old", category="politics")
        amem.mem_update(qid="Q42", memory="New", category="")
        df = amem.get_mem_df()
        assert df.iloc[0]["memory"] == "New"
        assert df.iloc[0]["category"] == "politics"

    def test_mem_update_preserves_category_when_none(self, tmp_path):
        """mem_update with category=None should preserve existing category."""
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q42", question="Will X happen?", memory="Old", category="politics")
        amem.mem_update(qid="Q42", memory="New", category=None)
        df = amem.get_mem_df()
        assert df.iloc[0]["memory"] == "New"
        assert df.iloc[0]["category"] == "politics"

    def test_mem_delete(self, tmp_path):
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q42", question="Will X happen?", memory="Evidence", category="politics")
        assert amem.mem_count == 1
        ok = amem.mem_delete("Q42")
        assert ok is True
        assert amem.mem_count == 0

    def test_mem_delete_nonexistent(self, tmp_path):
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        ok = amem.mem_delete("Q999")
        assert ok is False

    def test_mem_char_limit_is_1000(self):
        assert ACTIVE_MEM_CHAR_LIMIT == 1000

    def test_mem_summary_no_confidence(self, tmp_path):
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q42", question="Will X?", memory="Evidence here", category="test")
        summary = amem.mem_summary()
        assert "confidence" not in summary.lower()
        assert "Q42" in summary

    def test_mem_add_preserves_category_on_upsert(self, tmp_path):
        """mem_add with no category should preserve the existing category."""
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q42", question="Will X?", memory="Old evidence", category="politics")
        amem.mem_add(qid="Q42", question="Will X?", memory="New evidence")  # no category
        df = amem.get_mem_df()
        row = df[df["qid"] == "Q42"].iloc[0]
        assert row["memory"] == "New evidence"
        assert row["category"] == "politics"  # preserved

    def test_mem_add_overrides_category_when_provided(self, tmp_path):
        """mem_add with an explicit category should override the old one."""
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q42", question="Will X?", memory="Evidence", category="politics")
        amem.mem_add(qid="Q42", question="Will X?", memory="Evidence", category="sports")
        df = amem.get_mem_df()
        assert df[df["qid"] == "Q42"].iloc[0]["category"] == "sports"


# ── ActiveMemory directory format ─────────────────────────────────────────

class TestActiveMemoryDirectoryFormat:
    """Test new per-day directory format: memory/YYYY-MM-DD/{mem.csv, meta.yaml}."""

    def test_save_creates_directory(self, tmp_path):
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q1", question="Test", memory="Test memory", category="test")
        amem.add_entry("meta-1", "Meta desc", "Meta content")
        amem.save(date(2025, 6, 1))

        date_dir = tmp_path / "memory" / "2025-06-01"
        assert date_dir.is_dir()
        assert (date_dir / "mem.csv").exists()
        assert (date_dir / "meta.yaml").exists()

    def test_load_from_directory_format(self, tmp_path):
        # Save in directory format
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q1", question="Test Q", memory="Saved memory", category="test")
        amem.add_entry("meta-1", "Meta desc", "Meta content")
        amem.save(date(2025, 6, 1))

        # Load in new instance
        amem2 = ActiveMemory("agent1", str(tmp_path))
        amem2.set_date(date(2025, 6, 2))
        assert amem2.mem_count == 1
        df = amem2.get_mem_df()
        assert df.iloc[0]["qid"] == "Q1"
        assert df.iloc[0]["memory"] == "Saved memory"
        assert amem2.entry_count == 1
        assert amem2.retrieve("meta-1") is not None

    def test_restart_copy_directory_format(self, tmp_path):
        """Simulate restart: save → copytree to new dir → load from new dir."""
        import shutil

        src = tmp_path / "src"
        dst = tmp_path / "dst"

        # Save in source directory
        amem = ActiveMemory("agent1", str(src))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q1", question="Test Q1", memory="Evidence for Q1", category="politics")
        amem.mem_add(qid="Q2", question="Test Q2", memory="Evidence for Q2", category="sports")
        amem.add_entry("meta-pattern", "Cross-question pattern", "Sports Qs need bookmaker odds")
        amem.save(date(2025, 6, 1))

        # Simulate restart copy (like prepare_restart_directory)
        src_mem_dir = src / "memory"
        dst_mem_dir = dst / "memory"
        dst_mem_dir.mkdir(parents=True)
        for entry in src_mem_dir.iterdir():
            if entry.is_dir():
                shutil.copytree(entry, dst_mem_dir / entry.name)

        # Load from destination
        amem2 = ActiveMemory("agent1", str(dst))
        amem2.set_date(date(2025, 6, 2))
        assert amem2.mem_count == 2
        df = amem2.get_mem_df()
        assert set(df["qid"].tolist()) == {"Q1", "Q2"}
        assert amem2.entry_count == 1
        assert amem2.retrieve("meta-pattern") is not None

    def test_multi_day_directory_format(self, tmp_path):
        """Multiple days saved in directory format, set_date picks the most recent."""
        amem = ActiveMemory("agent1", str(tmp_path))

        # Day 1
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q1", question="Test Q1", memory="Day 1 evidence", category="test")
        amem.save(date(2025, 6, 1))

        # Day 2
        amem.set_date(date(2025, 6, 2))
        assert amem.mem_count == 1  # loaded from day 1
        amem.mem_add(qid="Q2", question="Test Q2", memory="Day 2 evidence", category="test")
        amem.save(date(2025, 6, 2))

        # Day 3: should load day 2 snapshot (2 entries)
        amem2 = ActiveMemory("agent1", str(tmp_path))
        amem2.set_date(date(2025, 6, 3))
        assert amem2.mem_count == 2

        # Restart from day 2: should only load day 1 snapshot (1 entry)
        amem3 = ActiveMemory("agent1", str(tmp_path))
        amem3.set_date(date(2025, 6, 2))
        assert amem3.mem_count == 1
        assert amem3.get_mem_df().iloc[0]["qid"] == "Q1"

    def test_backward_compat_flat_format(self, tmp_path):
        """Should load from old flat format (memo_YYYY-MM-DD.csv + YYYY-MM-DD.yaml)."""
        import csv
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir(parents=True)

        # Write old-format CSV with confidence column
        csv_path = mem_dir / "memo_2025-06-01.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["qid", "question", "last_updated", "memory", "confidence", "category"])
            writer.writerow(["Q42", "Will X happen?", "2025-06-01", "Old evidence", "0.75", "politics"])

        # Write old-format YAML
        yaml_path = mem_dir / "2025-06-01.yaml"
        yaml_data = [{"name": "old-meta", "description": "Old desc", "content": "Old content", "added": "2025-06-01"}]
        yaml_path.write_text(yaml.safe_dump(yaml_data))

        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 2))
        assert amem.mem_count == 1
        df = amem.get_mem_df()
        assert "confidence" not in df.columns  # confidence column dropped
        assert df.iloc[0]["qid"] == "Q42"
        assert amem.entry_count == 1


# ── Parse mem_add/update/delete action types ──────────────────────────────

class TestParseMemActions:
    """Test parsing of mem_add, mem_update, mem_delete action types."""

    def test_parse_mem_add(self):
        response = '''<reasoning>Storing Q42 analysis.</reasoning>
<action type="mem_add">
<qid>Q42</qid>
<question>Will X happen by Y?</question>
<memory>Key evidence: polls show 55% support. Confidence ~0.6.</memory>
<category>politics</category>
</action>'''
        parsed = parse_action(response)
        assert parsed.action_type == "mem_add"
        assert parsed.error is None
        assert parsed.mem_data is not None
        assert parsed.mem_data["qid"] == "Q42"
        assert parsed.mem_data["question"] == "Will X happen by Y?"
        assert "polls show 55%" in parsed.mem_data["memory"]
        assert parsed.mem_data["category"] == "politics"

    def test_parse_mem_add_plain_text(self):
        response = '''<action type="mem_add">
qid: Q42
question: Will X happen by Y?
memory: Key evidence here
category: politics
</action>'''
        parsed = parse_action(response)
        assert parsed.action_type == "mem_add"
        assert parsed.error is None
        assert parsed.mem_data["qid"] == "Q42"

    def test_parse_mem_add_missing_fields(self):
        response = '<action type="mem_add">\n<memory>Just memory, no qid</memory>\n</action>'
        parsed = parse_action(response)
        assert parsed.action_type == "mem_add"
        assert parsed.error is not None

    def test_parse_mem_update(self):
        response = '''<action type="mem_update" qid="Q42">
<memory>Updated evidence: new polls show 60%.</memory>
</action>'''
        parsed = parse_action(response)
        assert parsed.action_type == "mem_update"
        assert parsed.error is None
        assert parsed.mem_qid == "Q42"
        assert parsed.mem_data is not None
        assert "60%" in parsed.mem_data["memory"]

    def test_parse_mem_update_no_qid(self):
        response = '<action type="mem_update">\n<memory>Update without qid</memory>\n</action>'
        parsed = parse_action(response)
        assert parsed.action_type == "mem_update"
        assert parsed.error is not None

    def test_parse_mem_delete(self):
        response = '<action type="mem_delete">Q42</action>'
        parsed = parse_action(response)
        assert parsed.action_type == "mem_delete"
        assert parsed.error is None
        assert parsed.mem_qid == "Q42"

    def test_parse_mem_delete_empty(self):
        response = '<action type="mem_delete"></action>'
        parsed = parse_action(response)
        assert parsed.action_type == "mem_delete"
        assert parsed.error is not None


# ── Extract mem ops (end-of-day XML format) ───────────────────────────────

class TestExtractMemOps:
    """Test extract_mem_ops for both new <mem_add> and old <memo_add> tags."""

    def test_new_mem_add_tag(self):
        response = """<mem_add>
qid: Q42
question: Will X happen?
memory: Key evidence here.
category: politics
</mem_add>"""
        adds, updates, deletes = extract_mem_ops(response)
        assert len(adds) == 1
        assert adds[0]["qid"] == "Q42"
        assert adds[0]["memory"] == "Key evidence here."

    def test_old_memo_add_backward_compat(self):
        response = """<memo_add>
qid: Q42
question: Will X happen?
memory: Old style evidence.
category: politics
</memo_add>"""
        adds, updates, deletes = extract_mem_ops(response)
        assert len(adds) == 1
        assert adds[0]["qid"] == "Q42"

    def test_no_confidence_in_parsed_result(self):
        response = """<mem_add>
qid: Q42
question: Test Q
memory: Evidence
confidence: 0.75
category: test
</mem_add>"""
        adds, _, _ = extract_mem_ops(response)
        assert len(adds) == 1
        assert "confidence" not in adds[0]


# ── Warmup interop: ActiveMemory → StructuredMemory ───────────────────────

class TestWarmupInterop:
    """Test that ActiveMemory warmup can be loaded by StructuredMemory via flat YAML."""

    def test_active_warmup_interop_creates_flat_yaml(self, tmp_path):
        """Simulating _save_warmup_interop: flat yaml created from mem_df."""
        from pathlib import Path

        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q42", question="Will X happen?", memory="Key evidence for Q42", category="politics")
        amem.mem_add(qid="Q99", question="Will Y happen?", memory="Key evidence for Q99", category="sports")
        amem.save(date(2025, 6, 1))

        # Simulate interop write (same logic as _save_warmup_interop)
        memory_dir = tmp_path / "memory"
        flat_yaml_path = memory_dir / "2025-06-01.yaml"
        assert not flat_yaml_path.exists()

        entries = []
        for _, row in amem.get_mem_df().iterrows():
            qid = str(row["qid"])
            entries.append({
                "name": f"q{qid}-warmup",
                "description": f"Q{qid}: {str(row['question'])[:200]}",
                "content": str(row["memory"])[:1024],
                "added": "2025-06-01",
            })
        flat_yaml_path.write_text(yaml.safe_dump(entries, default_flow_style=False))

        assert flat_yaml_path.exists()
        loaded = yaml.safe_load(flat_yaml_path.read_text())
        assert len(loaded) == 2
        names = {e["name"] for e in loaded}
        assert "qQ42-warmup" in names
        assert "qQ99-warmup" in names

    def test_active_warmup_flat_yaml_loads_as_structured(self, tmp_path):
        """StructuredMemory.set_date() loads the flat yaml written by interop."""
        from agents.utils.memory import StructuredMemory

        # Create ActiveMemory warmup output
        amem = ActiveMemory("agent1", str(tmp_path))
        amem.set_date(date(2025, 6, 1))
        amem.mem_add(qid="Q42", question="Will X happen?", memory="Evidence for Q42", category="politics")
        amem.mem_add(qid="Q99", question="Will Y happen?", memory="Evidence for Q99", category="sports")
        amem.save(date(2025, 6, 1))

        # Write interop flat yaml
        memory_dir = tmp_path / "memory"
        entries = []
        for _, row in amem.get_mem_df().iterrows():
            qid = str(row["qid"])
            entries.append({
                "name": f"q{qid}-warmup",
                "description": f"Q{qid}: {str(row['question'])[:200]}",
                "content": str(row["memory"])[:1024],
                "added": "2025-06-01",
            })
        (memory_dir / "2025-06-01.yaml").write_text(yaml.safe_dump(entries, default_flow_style=False))

        # Load with StructuredMemory
        smem = StructuredMemory("agent1", str(tmp_path))
        smem.set_date(date(2025, 6, 2))
        assert smem.entry_count == 2
        index = smem.get_index()
        assert "Q42" in index
        assert "Q99" in index
        # Verify full content is retrievable
        q42_entry = [e for e in smem._entries if "Q42" in e.name or "Q42" in e.description][0]
        full = smem.retrieve(q42_entry.name)
        assert "Evidence for Q42" in full


# ── Limits and edge cases ────────────────────────────────────────────────

class TestLimitsAndEdgeCases:
    def test_max_entries_default_is_500(self):
        assert MAX_ENTRIES == 500

    def test_field_limits(self):
        assert FIELD_LIMITS["name"] == 64
        assert FIELD_LIMITS["description"] == 256
        assert FIELD_LIMITS["content"] == 1024

    def test_custom_max_entries(self, tmp_path):
        mem = StructuredMemory("agent1", str(tmp_path), max_entries=3)
        mem.set_date(date(2025, 6, 1))
        for i in range(10):
            mem.add_entry(f"entry-{i}", f"desc {i}", f"content {i}")
        assert mem.entry_count == 3  # only last 3 remain

    def test_parse_memory_add_with_special_chars(self):
        """Content with colons, quotes, etc. should parse correctly."""
        response = '''<action type="memory_add">
<name>tricky-content-test</name>
<description>Tests parsing with special chars in content field</description>
<content>Key ratio: 3:1. Quote: "hello world". URL: https://example.com/path?q=1&amp;r=2</content>
</action>'''
        parsed = parse_action(response)
        assert parsed.error is None
        assert "3:1" in parsed.memory_add_data["content"]

    def test_retrieve_with_whitespace_name(self, populated_mem):
        """Names with leading/trailing whitespace should be trimmed."""
        name = populated_mem._entries[0].name
        result = populated_mem.retrieve(f"  {name}  ")
        assert result is not None

    def test_empty_memory_index_is_empty_string(self):
        mem = StructuredMemory("agent1")
        assert mem.get_index() == ""

    def test_many_entries_index_performance(self, tmp_path):
        """Index generation should work fine with many entries."""
        mem = StructuredMemory("agent1", str(tmp_path), max_entries=200)
        mem.set_date(date(2025, 6, 1))
        for i in range(200):
            mem.add_entry(f"q{i}-entry", f"Description for Q{i}", f"Content for question {i}")
        index = mem.get_index()
        assert "q0-entry" in index
        assert "q199-entry" in index
        assert mem.entry_count == 200
