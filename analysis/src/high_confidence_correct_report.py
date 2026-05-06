#!/usr/bin/env python3
"""
Generate a per-question evidence report for the questions where a single-agent
warmup-only run assigned >= THRESHOLD probability mass to the *correct*
(semantically-matched) answer.

Inputs (auto-discovered from --run-dir):
  - actions.jsonl              (predictions + resolutions)
  - market.csv                 (question text / background / criteria / ground truth)
  - matcher_cache.json         (predicted-string vs ground-truth equivalence cache)
  - daily_metrics.csv          (headline metrics)
  - agents/<id>/model_outputs.jsonl
  - agents/<id>/model_raw_warmup.jsonl

Output: a single Markdown file. Sections per the previous Q1 2026 report.

Usage:
  python analysis/src/high_confidence_correct_report.py \\
    --run-dir /is/cluster/.../ds_rg1_aljazeeraQ12026v16_r00/26-04-27-00-19-50 \\
    --threshold 0.8 \\
    --out analysis/reports/ds_rg1_aljazeeraQ12026v16_r00_high_confidence_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _norm(s: str) -> str:
    return str(s).strip().lower()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] bad JSON {path}:{ln}: {e}", file=sys.stderr)


def load_market(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[str(row["qid"])] = row
    return out


def load_matcher_cache(path: Path) -> Dict[Tuple[str, str, str], bool]:
    """Return mapping (predicted_lower, ground_truth_lower, qid) -> is_equivalent.

    Cache keys are JSON-encoded lists; we strip optional title (4th element).
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[Tuple[str, str, str], bool] = {}
    for k, v in raw.items():
        try:
            arr = json.loads(k)
        except json.JSONDecodeError:
            continue
        if not isinstance(arr, list) or len(arr) < 3:
            continue
        pred, gt, qid = str(arr[0]), str(arr[1]), str(arr[2])
        out[(_norm(pred), _norm(gt), qid)] = bool(v)
    return out


def load_predictions(actions_path: Path, agent_id_filter: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for d in iter_jsonl(actions_path):
        if d.get("type") != "prediction":
            continue
        if agent_id_filter and d.get("agent_id") != agent_id_filter:
            continue
        qid = str(d["question_id"])
        out[qid] = {str(k): float(v) for k, v in d.get("outcomes", {}).items()}
    return out


def load_resolutions(actions_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for d in iter_jsonl(actions_path):
        if d.get("type") != "resolution":
            continue
        out[str(d["question_id"])] = str(d.get("ground_truth", ""))
    return out


def correct_mass(
    distribution: Dict[str, float],
    ground_truth: str,
    qid: str,
    matcher: Dict[Tuple[str, str, str], bool],
) -> Tuple[float, List[str]]:
    """Sum probabilities over outcome strings that the matcher considered
    equivalent to ground_truth for this qid."""
    matched: List[str] = []
    total = 0.0
    gt_n = _norm(ground_truth)
    for outcome, p in distribution.items():
        out_n = _norm(outcome)
        if out_n == gt_n:
            matched.append(outcome)
            total += p
            continue
        if matcher.get((out_n, gt_n, qid), False):
            matched.append(outcome)
            total += p
    return total, matched


def find_agent_dir(run_dir: Path) -> Path:
    agents = run_dir / "agents"
    if not agents.is_dir():
        raise FileNotFoundError(f"No agents/ in {run_dir}")
    children = [p for p in agents.iterdir() if p.is_dir()]
    if len(children) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 agent dir in {agents}, found: {[c.name for c in children]}"
        )
    return children[0]


# ──────────────────────────────────────────────────────────────────────────────
# Per-qid evidence extraction
# ──────────────────────────────────────────────────────────────────────────────

ARTICLE_BLOCK_RE = re.compile(
    r"═══\s*\[(?P<idx>\d+)\]\s*═+\s*\n"
    r"HEADLINE:\s*(?P<headline>.*?)\n"
    r"SOURCE:\s*(?P<source>.*?)\n"
    r"PUBLISHED:\s*(?P<published>.*?)\s*\|\s*DOWNLOADED:\s*(?P<downloaded>.*?)\n"
    r"URL:\s*(?P<url>.*?)\n"
    r"\s*\n"
    r"(?P<body>.*?)(?=(?:═══\s*\[\d+\]\s*═+|\Z))",
    re.DOTALL,
)


def parse_article_chunks(tool_text: str) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    for m in ARTICLE_BLOCK_RE.finditer(tool_text):
        chunks.append({
            "idx": m.group("idx"),
            "headline": m.group("headline").strip(),
            "source": m.group("source").strip(),
            "published": m.group("published").strip(),
            "downloaded": m.group("downloaded").strip(),
            "url": m.group("url").strip(),
            "body": m.group("body").strip(),
        })
    return chunks


def extract_evidence_for_qid(
    qid: str,
    raw_warmup_path: Path,
) -> Dict[str, Any]:
    """Walk model_raw_warmup.jsonl filtered to qid, extract:
       - turns (chronological list of {phase, reasoning, queries, submit_response})
       - search_queries (flat list, derived from turns)
       - articles_seen (deduped by URL, in order of first appearance)
       - submit_response (final reasoning text)
       - turn_count, phases

    For reasoning models (e.g. deepseek-v4-pro), `metadata.reasoning` carries the
    chain-of-thought that produced the same turn's tool calls or submit.
    """
    turns: List[Dict[str, Any]] = []
    search_queries: List[Dict[str, Any]] = []
    articles_seen: Dict[str, Dict[str, str]] = {}  # url -> chunk
    article_order: List[str] = []
    submit_response: Optional[str] = None
    turn_count = 0
    phases: List[str] = []

    for d in iter_jsonl(raw_warmup_path):
        if str(d.get("qid")) != str(qid):
            continue
        turn_count += 1
        meta = d.get("metadata", {}) or {}
        phase = meta.get("phase", "?")
        phases.append(phase)
        reasoning = meta.get("reasoning") or ""

        turn_queries: List[Dict[str, Any]] = []
        if phase == "search":
            for tc in re.finditer(r"TOOL_CALLS:\s*(\[.*?\])\s*$", d.get("response", ""), re.DOTALL):
                try:
                    parsed = json.loads(tc.group(1))
                except json.JSONDecodeError:
                    continue
                for call in parsed:
                    if call.get("name") == "search_news":
                        args = call.get("arguments") or {}
                        if isinstance(args, dict):
                            turn_queries.append(args)
                            search_queries.append(args)

        # Tool messages (search results) live in input_delta of the *next* turn
        for m in d.get("input_delta") or []:
            if m.get("role") != "tool":
                continue
            content = str(m.get("content", ""))
            for chunk in parse_article_chunks(content):
                url = chunk["url"]
                if url and url not in articles_seen:
                    articles_seen[url] = chunk
                    article_order.append(url)

        turn_submit: Optional[str] = None
        if phase == "submit":
            turn_submit = d.get("response", "") or ""
            submit_response = turn_submit

        turns.append({
            "index": turn_count,
            "phase": phase,
            "reasoning": reasoning,
            "queries": turn_queries,
            "submit_response": turn_submit,
        })

    return {
        "turns": turns,
        "search_queries": search_queries,
        "articles": [articles_seen[u] for u in article_order],
        "submit_response": submit_response,
        "turn_count": turn_count,
        "phases": phases,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ──────────────────────────────────────────────────────────────────────────────


def fmt_distribution_md(dist: Dict[str, float], matched: List[str]) -> List[str]:
    matched_set = {m for m in matched}
    items = sorted(dist.items(), key=lambda kv: -kv[1])
    out = []
    for outcome, p in items:
        check = "  ✅" if outcome in matched_set else ""
        out.append(f"- `{outcome}`: **{p:.3f}**{check}")
    return out


def truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


def render_report(
    *,
    run_dir: Path,
    threshold: float,
    title: str,
    setup: Dict[str, str],
    headline_metrics: Dict[str, str],
    rows: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Run directory**: `{run_dir}`")
    lines.append("")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    for k, v in setup.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    for k, v in headline_metrics.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### Question-by-question summary table")
    lines.append("")
    lines.append("| # | QID | Resolution date | Ground truth | P(correct) | Title (truncated) |")
    lines.append("|---|-----|-----------------|--------------|-----------:|--------------------|")
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['qid']} | {r['resolution_date']} | {r['ground_truth']} | "
            f"{r['p_correct']:.3f} | {truncate(r['title'], 110)} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Per-question evidence trail")
    lines.append("")
    lines.append("For every high-confidence correct question, the report shows:")
    lines.append("")
    lines.append("1. The question, background, and resolution criteria as recorded in `market.csv`.")
    lines.append("2. The model's full probability distribution and the matched ground-truth probability.")
    lines.append("3. The chronological investigation trail — for each search turn, the model's chain-of-thought (`metadata.reasoning`) followed by the search query it issued. This shows *how* the model reasoned its way to the next piece of evidence.")
    lines.append("4. Every article chunk returned to the agent across all searches (deduplicated by URL), with headline, source, published date, downloaded date, URL, and the chunk body the model actually saw.")
    lines.append("5. The agent's final submit-phase chain-of-thought verbatim, followed by the actual submit-phase tool call (the forecast emitted).")
    lines.append("")

    for i, r in enumerate(rows, 1):
        lines.append(f"### {i}. [QID {r['qid']}] {r['title']}")
        lines.append("")
        lines.append(f"- **Ground truth**: `{r['ground_truth']}`")
        matched_repr = ", ".join(repr(m) for m in r["matched_outcomes"])
        lines.append(
            f"- **Model probability on correct answer**: **{r['p_correct']:.3f}**  "
            f"(matched outcome string(s): {matched_repr})"
        )
        lines.append(
            f"- **Resolution date**: `{r['resolution_date']}` "
            f"(model was permitted to see news ≤ `{r['resolution_date']}` − 1 day)"
        )
        lines.append(f"- **Answer type**: `{r['answer_type']}`")
        lines.append("")
        if r["background"]:
            lines.append("**Background**:")
            lines.append("")
            for ln in r["background"].splitlines():
                lines.append(f"> {ln}")
            lines.append("")
        if r["resolution_criteria"]:
            lines.append("**Resolution criteria**:")
            lines.append("")
            crit = r["resolution_criteria"]
            # Strip <ul>/<li> for readability if present
            crit = re.sub(r"\s*<ul>\s*", "", crit)
            crit = re.sub(r"\s*</ul>\s*", "", crit)
            crit = re.sub(r"\s*<li>\s*", "- ", crit)
            crit = re.sub(r"\s*</li>\s*", "\n", crit)
            crit = re.sub(r"<b>(.*?)</b>", r"**\1**", crit)
            crit = re.sub(r"<i>(.*?)</i>", r"*\1*", crit)
            for ln in crit.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(f"> {ln}")
            lines.append("")
        lines.append("**Model prediction (full distribution)**:")
        lines.append("")
        lines.extend(fmt_distribution_md(r["distribution"], r["matched_outcomes"]))
        lines.append("")
        sq = r["evidence"]["search_queries"]
        turns = r["evidence"].get("turns") or []
        search_turns = [t for t in turns if t["phase"] == "search"]
        lines.append(f"**Investigation trail ({len(sq)} search queries across {len(search_turns)} search turns)**:")
        lines.append("")
        lines.append("Each turn shows the model's chain-of-thought (`metadata.reasoning`) followed by the search query (or queries) it issued that turn. Search results returned to the model are deduplicated and listed in the next section.")
        lines.append("")
        for t in search_turns:
            lines.append(f"#### Turn {t['index']} — search")
            lines.append("")
            reasoning = (t.get("reasoning") or "").strip()
            if reasoning:
                lines.append("**Reasoning**:")
                lines.append("")
                lines.append("```")
                lines.append(reasoning)
                lines.append("```")
                lines.append("")
            else:
                lines.append("_(no reasoning content recorded for this turn)_")
                lines.append("")
            queries = t.get("queries") or []
            if queries:
                lines.append("**Search query/queries issued this turn**:")
                lines.append("")
                for q in queries:
                    lines.append(f"- `{json.dumps(q, sort_keys=True)}`")
                lines.append("")
        arts = r["evidence"]["articles"]
        lines.append(f"**Article chunks shown to the model ({len(arts)} unique URLs across all searches)**:")
        lines.append("")
        for j, a in enumerate(arts, 1):
            lines.append(f"**[{j}] {a['headline']}**")
            lines.append(
                f"- Source: `{a['source']}` · Published: `{a['published']}` · Downloaded: `{a['downloaded']}`"
            )
            lines.append(f"- URL: <{a['url']}>")
            lines.append("")
            for bl in a["body"].splitlines():
                lines.append(f"> {bl}")
            lines.append("")
        submit_turn = next((t for t in turns if t["phase"] == "submit"), None)
        submit_reasoning = (submit_turn or {}).get("reasoning") or ""
        sub = r["evidence"]["submit_response"] or ""
        lines.append("**Final submit-phase reasoning** (model's chain-of-thought before emitting the forecast):")
        lines.append("")
        if submit_reasoning.strip():
            lines.append("```")
            lines.append(submit_reasoning.strip())
            lines.append("```")
        else:
            lines.append("_(no reasoning content recorded for submit turn)_")
        lines.append("")
        lines.append("**Submit-phase tool call** (the actual forecast emitted):")
        lines.append("")
        lines.append("```")
        lines.append(sub.strip())
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    actions_p = run_dir / "actions.jsonl"
    market_p = run_dir / "market.csv"
    cache_p = run_dir / "matcher_cache.json"
    daily_p = run_dir / "daily_metrics.csv"
    agent_dir = find_agent_dir(run_dir)
    raw_warmup_p = agent_dir / "model_raw_warmup.jsonl"

    config_p = run_dir / "config.json"
    cfg = json.loads(config_p.read_text()) if config_p.is_file() else {}
    sim_meta = cfg.get("sim_meta", cfg)
    agent_id = agent_dir.name

    market = load_market(market_p)
    matcher_cache = load_matcher_cache(cache_p)
    predictions = load_predictions(actions_p, agent_id_filter=agent_id)
    resolutions = load_resolutions(actions_p)

    # Headline metrics from daily_metrics.csv
    headline = {}
    if daily_p.is_file():
        with daily_p.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("agent_id") == agent_id:
                    headline = row
                    break

    # Build candidate rows
    rows_all = []
    for qid, dist in predictions.items():
        gt = resolutions.get(qid, "")
        if not gt:
            continue
        p_correct, matched = correct_mass(dist, gt, qid, matcher_cache)
        rows_all.append((qid, p_correct, matched, gt, dist))

    high_conf = [(qid, pc, m, gt, dist) for (qid, pc, m, gt, dist) in rows_all if pc >= args.threshold]
    high_conf.sort(key=lambda t: (-t[1], t[0]))

    print(f"Total resolved questions: {len(rows_all)}")
    print(f"High-conf (>= {args.threshold}) correct: {len(high_conf)}")

    rows: List[Dict[str, Any]] = []
    for qid, p_correct, matched, gt, dist in high_conf:
        meta = market.get(qid, {})
        evidence = extract_evidence_for_qid(qid, raw_warmup_p)
        rows.append({
            "qid": qid,
            "p_correct": p_correct,
            "matched_outcomes": matched,
            "ground_truth": gt,
            "distribution": dist,
            "title": meta.get("title", ""),
            "background": meta.get("background", ""),
            "resolution_criteria": meta.get("resolution_criteria", ""),
            "resolution_date": meta.get("resolution_date", ""),
            "answer_type": meta.get("answer_type", ""),
            "evidence": evidence,
        })

    setup = {
        "Agent": f"`{agent_id}` (scaffold `{cfg.get('scaffold','allQ')}`, provider `openrouter`, "
                 f"model `{cfg.get('agents',[{}])[0].get('model','?') if cfg.get('agents') else '?'}`)",
        "Simulation date": f"{cfg.get('start_date','?')} (single-day warmup over the full question set)",
        "Resolution window": f"{cfg.get('resolution_start','?')} → {cfg.get('resolution_end','?')} ({cfg.get('split','?')})",
        "`timegap_days`": f"{cfg.get('timegap_days','?')} — for every question the model could only see news indexed **up to (resolution_date − 1)**",
        "`resolution_guard`": f"{cfg.get('resolution_guard','?')} (one-day buffer in addition to the timegap)",
        "Search backend": "LanceDB (Qwen3-Embedding-8B) over the deduped CommonCrawl corpus",
        "Matching": f"`{cfg.get('matcher','?')}` via OpenRouter (semantic answer-matching)",
    }

    headline_metrics = {
        f"Total resolved questions in this run": f"**{len(rows_all)}**",
        f"Questions where the model assigned **≥ {args.threshold:.2f} probability mass** to the correct answer":
            f"**{len(rows)}** ({100.0*len(rows)/max(1,len(rows_all)):.1f}%)",
    }
    if headline:
        headline_metrics["Daily metrics row"] = (
            f"avg Brier `{headline.get('avg_brier','?')}`, "
            f"accuracy `{headline.get('accuracy','?')}%`, "
            f"time-weighted score `{headline.get('tw_score','?')}`, "
            f"`{headline.get('total_predictions','?')}` predictions, "
            f"`{headline.get('daily_submissions','?')}` daily submissions (this is a single warmup day)."
        )

    title = args.title or f"DeepSeek — {cfg.get('split','?')} Forecasting: High-Confidence Correct Predictions"
    md = render_report(
        run_dir=run_dir,
        threshold=args.threshold,
        title=title,
        setup=setup,
        headline_metrics=headline_metrics,
        rows=rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(f"Wrote {args.out} ({len(md):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
