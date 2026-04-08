#!/usr/bin/env python3
"""
End-to-end test harness for ClaudeCodeAgent.

Three test levels:
  1. unit   -- MCP server tools work in isolation (no Claude Code, no GPU)
  2. smoke  -- Full pipeline with validation20 split through test_basic_agent.py.
               Uses real SimulationEnvironment, real search (LanceDB), real scoring,
               real articles, real answer matching. Proves a full run will work.
               Requires: Claude Code auth, LanceDB + embedding model, OpenRouter key.
  3. real   -- Arbitrary config for production evaluation runs.

Usage:
  # Unit test (fast, no API calls):
  python -m agents.claudeCodeAgent.test_harness --level unit

  # Smoke test (full features, validation20 split):
  python -m agents.claudeCodeAgent.test_harness --level smoke

  # Real eval run:
  python -m agents.claudeCodeAgent.test_harness --level real --config configs/claude_code_validation20.yaml
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

# Ensure repo root is on sys.path.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ── Unit tests ─────────────────────────────────────────────────────────

def test_unit():
    """Test MCP server tools in isolation — no Claude Code, no GPU."""
    from agents.claudeCodeAgent.mcp_server import (
        _reload_state,
        _write_json,
        _parse_date,
        search_news,
        submit_forecast,
    )
    import agents.claudeCodeAgent.mcp_server as srv

    workspace = Path(tempfile.mkdtemp(prefix="cc_test_unit_"))
    try:
        # Set up minimal state.
        for d in ("signals", "predictions", "memory"):
            (workspace / d).mkdir(parents=True, exist_ok=True)

        state = {
            "current_date": "2025-05-01",
            "start_date": "2025-05-01",
            "end_date": "2025-05-03",
            "search_cutoff_days": 0,
            "questions": [
                {
                    "qid": "q1",
                    "title": "Will it rain tomorrow?",
                    "background": "Weather forecast question.",
                    "resolution_criteria": "Rain recorded at station X.",
                    "answer_type": "binary",
                    "resolution_date": "2025-05-03",
                    "options": ["Yes", "No"],
                },
                {
                    "qid": "q2",
                    "title": "Who wins the match?",
                    "background": "Sports question.",
                    "resolution_criteria": "Final score.",
                    "answer_type": "multiple_choice",
                    "resolution_date": "2025-05-02",
                    "options": ["Team A", "Team B", "Draw"],
                },
            ],
            "resolution_events": [],
        }
        _write_json(workspace / "state.json", state)

        # Point module globals at our workspace.
        srv._workspace = workspace
        srv._reload_state()

        # Test 1: search_news without search DB returns graceful message.
        result = search_news("test query")
        assert "not available" in result.lower(), f"Expected 'not available', got: {result}"
        print("  [PASS] search_news without DB returns graceful message")

        # Test 2: submit_forecast writes predictions.
        result = submit_forecast("q1", {"Yes": 0.7, "No": 0.3})
        assert "recorded" in result.lower(), f"Expected 'recorded', got: {result}"
        pred_path = workspace / "predictions" / "2025-05-01.json"
        assert pred_path.exists(), "Predictions file not created"
        preds = json.loads(pred_path.read_text())
        assert len(preds) == 1
        assert preds[0]["question_id"] == "q1"
        assert preds[0]["outcomes"]["Yes"] == 0.7
        print("  [PASS] submit_forecast writes valid predictions file")

        # Test 3: submit_forecast validates probabilities.
        result = submit_forecast("q2", {"Team A": 0.6, "Team B": 0.5})
        assert "error" in result.lower(), f"Expected error for sum > 1, got: {result}"
        print("  [PASS] submit_forecast rejects probabilities > 1.0")

        # Test 4: submit second valid prediction.
        result = submit_forecast("q2", {"Team A": 0.5, "Team B": 0.3, "Draw": 0.2})
        assert "recorded" in result.lower()
        preds = json.loads(pred_path.read_text())
        assert len(preds) == 2
        print("  [PASS] submit_forecast accumulates predictions")

        # Test 5: _parse_date.
        assert _parse_date("2025-05-01") == date(2025, 5, 1)
        assert _parse_date(None) is None
        assert _parse_date("") is None
        print("  [PASS] _parse_date works correctly")

        print("\n  All unit tests passed!")

    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# ── Smoke test ─────────────────────────────────────────────────────────

_SMOKE_CONFIG = "configs/claude_code_validation20.yaml"


def test_smoke():
    """
    Full-feature integration test using the real simulation pipeline.

    Runs test_basic_agent.py with the validation20 config.  This exercises
    everything a production run would: dataset loading, LanceDB hybrid search,
    article file access, answer matching, Brier scoring, market.csv, and the
    full day loop through SimulationEnvironment.

    After the run completes, validates all expected output artifacts.

    Requires: Claude Code auth, FSIM_SEARCH_DB + FSIM_EMBEDDING_MODEL (GPU),
              OPENROUTER_API_KEY (answer matching).
    """
    import glob

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / _SMOKE_CONFIG

    if not config_path.exists():
        print(f"  ERROR: Config not found: {config_path}")
        sys.exit(1)

    # ── Pre-flight checks ──────────────────────────────────────────────
    print("  Pre-flight checks:")
    from dotenv import load_dotenv
    load_dotenv(repo_root / ".env")

    checks = {
        "FSIM_DATASET_PATH": os.environ.get("FSIM_DATASET_PATH", ""),
        "FSIM_SEARCH_DB": os.environ.get("FSIM_SEARCH_DB", ""),
        "FSIM_EMBEDDING_MODEL": os.environ.get("FSIM_EMBEDDING_MODEL", ""),
        "FSIM_OUTPUT_BASE": os.environ.get("FSIM_OUTPUT_BASE", ""),
    }
    all_ok = True
    for var, val in checks.items():
        exists = bool(val) and os.path.exists(val)
        status = "OK" if exists else "MISSING"
        print(f"    {var}: {status} ({val[:60]}{'...' if len(val) > 60 else ''})")
        if not exists:
            all_ok = False

    # Check validation20 split exists.
    ds_path = checks["FSIM_DATASET_PATH"]
    split_file = os.path.join(ds_path, "validation20-00000-of-00001.parquet") if ds_path else ""
    if split_file and os.path.exists(split_file):
        print(f"    validation20 split: OK")
    else:
        print(f"    validation20 split: MISSING — create it first:")
        print(f"      python -c \"import pandas as pd; "
              f"df = pd.read_parquet('{ds_path}/validation-00000-of-00001.parquet'); "
              f"df['_d'] = pd.to_datetime(df['resolution_date']); "
              f"df.sort_values('_d').head(20).drop(columns='_d')"
              f".to_parquet('{ds_path}/validation20-00000-of-00001.parquet', index=False)\"")
        all_ok = False

    # Check Claude Code CLI.
    claude_path = shutil.which("claude")
    if claude_path:
        print(f"    claude CLI: OK ({claude_path})")
    else:
        print(f"    claude CLI: MISSING")
        all_ok = False

    if not all_ok:
        print("\n  ERROR: Pre-flight checks failed. Fix the issues above and retry.")
        sys.exit(1)

    print()

    # ── Run the simulation ─────────────────────────────────────────────
    print(f"  Running: test_basic_agent.py --config {_SMOKE_CONFIG}")
    print(f"  This will take a while (20 questions, ~45 sim days)...\n")

    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "test_basic_agent.py"),
         "--config", str(config_path)],
        cwd=str(repo_root),
        timeout=14400,  # 4 hour hard timeout
    )

    if result.returncode != 0:
        print(f"\n  ERROR: test_basic_agent.py exited with code {result.returncode}")
        sys.exit(1)

    # ── Find output directory ──────────────────────────────────────────
    output_base = os.environ.get("FSIM_OUTPUT_BASE", "")
    sim_dirs = sorted(glob.glob(os.path.join(output_base, "claude_code_validation20", "*")))
    if not sim_dirs:
        print("  ERROR: No output directory found")
        sys.exit(1)
    output_dir = Path(sim_dirs[-1])  # most recent
    print(f"\n  Output directory: {output_dir}")

    # ── Validate outputs ───────────────────────────────────────────────
    print("\n  Validating outputs:\n")
    errors = []

    # 1. config.json
    config_json = output_dir / "config.json"
    if config_json.exists():
        print("  [PASS] config.json exists")
    else:
        errors.append("config.json missing")
        print("  [FAIL] config.json missing")

    # 2. actions.jsonl
    actions_file = output_dir / "actions.jsonl"
    if actions_file.exists():
        with open(actions_file) as f:
            actions = [json.loads(line) for line in f if line.strip()]
        predictions = [a for a in actions if a.get("type") == "prediction"]
        resolutions = [a for a in actions if a.get("type") == "resolution"]
        print(f"  [PASS] actions.jsonl: {len(predictions)} predictions, {len(resolutions)} resolutions")
        if predictions:
            print(f"  [PASS] Predictions were submitted")
        else:
            errors.append("No predictions in actions.jsonl")
            print(f"  [FAIL] No predictions in actions.jsonl")
        if resolutions:
            print(f"  [PASS] Questions were resolved ({len(resolutions)} resolutions)")
        else:
            print(f"  [WARN] No resolutions in actions.jsonl (may be expected if dates don't overlap)")
    else:
        errors.append("actions.jsonl missing")
        print("  [FAIL] actions.jsonl missing")

    # 3. daily_metrics.csv
    metrics_file = output_dir / "daily_metrics.csv"
    if metrics_file.exists():
        import pandas as pd
        metrics = pd.read_csv(metrics_file)
        print(f"  [PASS] daily_metrics.csv: {len(metrics)} rows")
        if "avg_brier" in metrics.columns and metrics["avg_brier"].notna().any():
            last = metrics.iloc[-1]
            print(f"         Last day: avg_brier={last.get('avg_brier', 'N/A')}, "
                  f"total_predictions={last.get('total_predictions', 'N/A')}")
    else:
        errors.append("daily_metrics.csv missing")
        print("  [FAIL] daily_metrics.csv missing")

    # 4. Agent directory
    agent_dirs = list((output_dir / "agents").glob("claudecode_*"))
    if agent_dirs:
        agent_dir = agent_dirs[0]
        print(f"  [PASS] Agent directory: {agent_dir.name}")

        # Check Claude Code logs
        stdout_log = agent_dir / "claude_code_stdout.jsonl"
        if stdout_log.exists():
            size_kb = stdout_log.stat().st_size / 1024
            print(f"  [PASS] Claude Code stdout log: {size_kb:.1f} KB")

            # Count MCP tool usage
            tool_counts = {}
            with open(stdout_log) as f:
                for line in f:
                    try:
                        d = json.loads(line.strip())
                        if d.get("type") == "assistant":
                            for c in (d.get("message", {}).get("content") or []):
                                if c.get("type") == "tool_use":
                                    name = c["name"]
                                    tool_counts[name] = tool_counts.get(name, 0) + 1
                    except (json.JSONDecodeError, KeyError):
                        pass

            mcp_tools = {k: v for k, v in tool_counts.items() if k.startswith("mcp__")}
            native_tools = {k: v for k, v in tool_counts.items() if not k.startswith("mcp__")}

            if mcp_tools:
                print(f"  [PASS] MCP tools used:")
                for t, c in sorted(mcp_tools.items(), key=lambda x: -x[1]):
                    print(f"           {t}: {c}")
            else:
                errors.append("No MCP tools were used")
                print(f"  [FAIL] No MCP tools used (MCP server may have failed to connect)")

            if "mcp__forecast__search_news" in mcp_tools:
                print(f"  [PASS] search_news was called ({mcp_tools['mcp__forecast__search_news']}x)")
            else:
                errors.append("search_news was never called")
                print(f"  [FAIL] search_news was never called")

            if "mcp__forecast__submit_forecast" in mcp_tools:
                print(f"  [PASS] submit_forecast was called ({mcp_tools['mcp__forecast__submit_forecast']}x)")
            else:
                errors.append("submit_forecast was never called")
                print(f"  [FAIL] submit_forecast was never called")

            if "mcp__forecast__next_day" in mcp_tools:
                print(f"  [PASS] next_day was called ({mcp_tools['mcp__forecast__next_day']}x)")
            else:
                errors.append("next_day was never called")
                print(f"  [FAIL] next_day was never called")

            if native_tools:
                print(f"  [PASS] Native tools used: {', '.join(f'{k}({v})' for k, v in sorted(native_tools.items(), key=lambda x: -x[1])[:8])}")
        else:
            errors.append("Claude Code stdout log missing")
            print("  [FAIL] Claude Code stdout log missing")

        # Check memory
        mem_dir = agent_dir / "memory"
        mem_files = list(mem_dir.glob("*")) if mem_dir.exists() else []
        print(f"  [INFO] Memory files: {len(mem_files)}")

        # Check predictions
        pred_dir = agent_dir / "predictions"
        pred_files = sorted(pred_dir.glob("*.json")) if pred_dir.exists() else []
        total_preds = 0
        for pf in pred_files:
            with open(pf) as f:
                total_preds += len(json.load(f))
        print(f"  [INFO] Prediction files: {len(pred_files)} ({total_preds} total predictions)")

        # Check article symlinks
        articles_dir = agent_dir / "articles"
        if articles_dir.exists():
            symlinks = list(articles_dir.rglob("*"))
            day_dirs = [s for s in symlinks if s.is_dir() and len(s.name) == 2]  # DD dirs
            print(f"  [PASS] Articles staging directory: {len(day_dirs)} day directories")
        else:
            print(f"  [WARN] No articles staging directory")

    else:
        errors.append("No agent directory found")
        print("  [FAIL] No agent directory found")

    # 5. Matcher cache
    matcher_files = list(output_dir.glob("matcher*"))
    if matcher_files:
        print(f"  [PASS] Matcher cache: {[f.name for f in matcher_files]}")
    else:
        print(f"  [WARN] No matcher cache found (may be fine if no resolutions)")

    # ── Final verdict ──────────────────────────────────────────────────
    print()
    if errors:
        print(f"  SMOKE TEST FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print(f"  SMOKE TEST PASSED — all checks OK")
        print(f"  Output: {output_dir}")


# ── Real test ──────────────────────────────────────────────────────────

def test_real(config_path: str):
    """
    Run a real simulation via the standard entry point.
    Just invokes test_basic_agent.py with the given config.
    """
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable, str(repo_root / "scripts" / "test_basic_agent.py"),
        "--config", config_path,
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(repo_root))


# ── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test ClaudeCodeAgent")
    parser.add_argument("--level", choices=["unit", "smoke", "real"], default="unit",
                        help="Test level: unit (no API), smoke (full pipeline validation20), "
                             "real (custom config)")
    parser.add_argument("--config", help="Config YAML for real test level")
    args = parser.parse_args()

    print(f"=== ClaudeCodeAgent Test: {args.level} ===\n")

    if args.level == "unit":
        test_unit()
    elif args.level == "smoke":
        test_smoke()
    elif args.level == "real":
        if not args.config:
            parser.error("--config required for real test level")
        test_real(args.config)


if __name__ == "__main__":
    main()
