#!/usr/bin/env python3
"""
Analyze why forecast updates help or hurt over time.

Inputs per run:
- actions.jsonl
- daily_metrics.csv
- agents/*/model_outputs.jsonl
- agents/*/timing_stats.jsonl
- matcher_cache.json (optional)

Outputs per run (under plots/update_analysis/):
- forecast_events_scored.csv
- forecast_index_summary.csv
- qid_summary.csv
- day_summary.csv
- summary.txt
- png plots (global + per-qid trajectories)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class RunArtifacts:
    run_dir: Path
    output_dir: Path
    forecast_events: pd.DataFrame
    forecast_index_summary: pd.DataFrame
    qid_summary: pd.DataFrame
    day_summary: pd.DataFrame
    update_single_vs_group_summary: pd.DataFrame
    update_batch_position_summary: pd.DataFrame
    day0_date: pd.Timestamp
    run_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze update behavior for forecast runs.")
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory containing actions.jsonl and daily_metrics.csv (repeatable).",
    )
    parser.add_argument(
        "--per-qid-min-forecasts",
        type=int,
        default=2,
        help="Generate per-qid plots for qids with at least this many forecasts (default: 2).",
    )
    parser.add_argument(
        "--max-per-qid-plots",
        type=int,
        default=0,
        help="Cap number of per-qid plots (0 = no cap; default: 0).",
    )
    parser.add_argument(
        "--comparison-output-dir",
        default="analysis/plots/update_analysis",
        help="Directory for cross-run comparison plots (default: analysis/plots/update_analysis).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] Skipping malformed JSON line {lineno} in {path}")
    return rows


def norm_cache(text: str) -> str:
    return str(text).strip().lower()


def norm_loose(text: str) -> str:
    return "".join(str(text).lower().split()).strip()


def parse_tool_calls(response: Any) -> List[Dict[str, Any]]:
    if not isinstance(response, str):
        return []
    if not response.startswith("TOOL_CALLS:"):
        return []
    payload = response.split("TOOL_CALLS:", 1)[1].strip()
    try:
        parsed = json.loads(payload)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def canonicalize_outcomes(outcomes: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(outcomes, dict):
        return out
    for k, v in outcomes.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def outcomes_equal(a: Dict[str, float], b: Dict[str, float], tol: float = 1e-9) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a.keys():
        if abs(float(a[k]) - float(b[k])) > tol:
            return False
    return True


def match_outcome(
    predicted: str,
    truth: str,
    qid: str,
    matcher_cache: Dict[str, bool],
) -> bool:
    pred_norm = norm_cache(predicted)
    truth_norm = norm_cache(truth)

    if pred_norm == truth_norm:
        return True

    key_qid = f"{pred_norm}|||{truth_norm}|||{qid}"
    if key_qid in matcher_cache:
        return bool(matcher_cache[key_qid])

    key_none = f"{pred_norm}|||{truth_norm}|||None"
    if key_none in matcher_cache:
        return bool(matcher_cache[key_none])

    if norm_loose(predicted) == norm_loose(truth):
        return True

    return False


def score_prediction(
    outcomes: Dict[str, float],
    truth: str,
    qid: str,
    matcher_cache: Dict[str, bool],
) -> Dict[str, Any]:
    parsed_outcomes: Dict[str, float] = {}
    for k, v in outcomes.items():
        try:
            parsed_outcomes[str(k)] = float(v)
        except Exception:
            continue

    matched_outcome: Optional[str] = None
    if truth in parsed_outcomes:
        matched_outcome = truth
    else:
        for outcome in parsed_outcomes.keys():
            if match_outcome(outcome, truth, qid, matcher_cache):
                matched_outcome = outcome
                break

    brier = 0.0
    for outcome, prob in parsed_outcomes.items():
        y = 1.0 if outcome == matched_outcome else 0.0
        brier += (prob - y) ** 2
    if matched_outcome is None:
        brier += 1.0

    top_outcome = None
    top_prob = 0.0
    if parsed_outcomes:
        top_outcome, top_prob = max(parsed_outcomes.items(), key=lambda kv: kv[1])

    top_correct = False
    if top_outcome is not None:
        top_correct = match_outcome(top_outcome, truth, qid, matcher_cache)

    total_prob = float(sum(parsed_outcomes.values()))
    entropy = 0.0
    for p in parsed_outcomes.values():
        if p > 0:
            entropy -= p * math.log(p)

    return {
        "brier_skill": 1.0 - brier,
        "top_outcome": top_outcome,
        "top_prob": top_prob,
        "truth_prob": float(parsed_outcomes.get(matched_outcome, 0.0)) if matched_outcome else 0.0,
        "top_correct": int(bool(top_correct)),
        "total_prob": total_prob,
        "abstain_prob": max(0.0, 1.0 - total_prob),
        "n_outcomes": int(len(parsed_outcomes)),
        "entropy": entropy,
    }


def l1_shift(cur: Any, prev: Any) -> float:
    if not isinstance(cur, dict) or not isinstance(prev, dict):
        return 0.0
    keys = set(cur.keys()) | set(prev.keys())
    return float(sum(abs(float(cur.get(k, 0.0)) - float(prev.get(k, 0.0))) for k in keys))


def load_actions(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    actions_path = run_dir / "actions.jsonl"
    if not actions_path.exists():
        raise FileNotFoundError(f"Missing {actions_path}")

    prediction_rows: List[Dict[str, Any]] = []
    resolution_rows: List[Dict[str, Any]] = []

    with actions_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = record.get("type")
            sim_date = record.get("sim_date")
            if not isinstance(sim_date, str):
                continue

            if rtype == "prediction":
                qid = record.get("question_id")
                outcomes = record.get("outcomes")
                if qid is None or not isinstance(outcomes, dict):
                    continue
                prediction_rows.append(
                    {
                        "line_no": line_no,
                        "sim_date": pd.to_datetime(sim_date),
                        "qid": str(qid),
                        "outcomes": outcomes,
                    }
                )
            elif rtype == "resolution":
                qid = record.get("question_id")
                ground_truth = record.get("ground_truth")
                if qid is None or not isinstance(ground_truth, str):
                    continue
                resolution_rows.append(
                    {
                        "line_no": line_no,
                        "sim_date": pd.to_datetime(sim_date),
                        "qid": str(qid),
                        "ground_truth": ground_truth,
                    }
                )

    pred_df = pd.DataFrame(prediction_rows)
    res_df = pd.DataFrame(resolution_rows)
    if not pred_df.empty:
        pred_df = pred_df.sort_values("line_no").reset_index(drop=True)
    if not res_df.empty:
        res_df = res_df.sort_values("line_no").reset_index(drop=True)
    return pred_df, res_df


def load_matcher_cache(run_dir: Path) -> Dict[str, bool]:
    cache_path = run_dir / "matcher_cache.json"
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, bool] = {}
    for k, v in data.items():
        if isinstance(k, str):
            out[k] = bool(v)
    return out


def load_model_outputs(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "agents" / "allQ_gpt-oss-120b_001" / "model_outputs.jsonl"
    if not path.exists():
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for rec in load_jsonl(path):
        sim_date = rec.get("sim_date")
        if not isinstance(sim_date, str):
            continue
        md = rec.get("metadata") or {}
        if not isinstance(md, dict):
            md = {}
        tool_calls = parse_tool_calls(rec.get("response"))
        first_tool = tool_calls[0] if tool_calls else {}
        tool_name = first_tool.get("name") if isinstance(first_tool, dict) else None
        submit_size = 0
        if tool_name == "submit_forecasts":
            args = first_tool.get("arguments") if isinstance(first_tool, dict) else {}
            if isinstance(args, dict):
                forecasts = args.get("forecasts")
                if isinstance(forecasts, list):
                    submit_size = len(forecasts)

        rows.append(
            {
                "sim_date": pd.to_datetime(sim_date),
                "qid": rec.get("qid"),
                "phase": md.get("phase"),
                "actions_remaining": md.get("actions_remaining"),
                "reasoning_effort": md.get("reasoning_effort"),
                "reasoning_tokens_turn": md.get("reasoning_tokens"),
                "tool_name": tool_name,
                "submit_size": submit_size,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("sim_date").reset_index(drop=True)
    return df


def load_submit_calls(run_dir: Path) -> pd.DataFrame:
    """
    Load submit_forecasts tool calls with forecast order (XML/list index).
    """
    path = run_dir / "agents" / "allQ_gpt-oss-120b_001" / "model_outputs.jsonl"
    if not path.exists():
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    call_seq = 0
    for rec in load_jsonl(path):
        sim_date = rec.get("sim_date")
        if not isinstance(sim_date, str):
            continue
        tool_calls = parse_tool_calls(rec.get("response"))
        if not tool_calls:
            continue
        first_tool = tool_calls[0]
        if not isinstance(first_tool, dict):
            continue
        if first_tool.get("name") != "submit_forecasts":
            continue

        args = first_tool.get("arguments")
        forecasts = args.get("forecasts") if isinstance(args, dict) else None
        if not isinstance(forecasts, list):
            continue

        entries: List[Dict[str, Any]] = []
        for idx, f in enumerate(forecasts, start=1):
            if not isinstance(f, dict):
                continue
            qid = f.get("qid")
            outcomes = f.get("outcomes")
            if not isinstance(qid, str) or not isinstance(outcomes, dict):
                continue
            entries.append(
                {
                    "xml_index": idx,
                    "qid": str(qid),
                    "outcomes": canonicalize_outcomes(outcomes),
                }
            )

        call_seq += 1
        rows.append(
            {
                "sim_date": pd.to_datetime(sim_date),
                "submit_call_seq": call_seq,
                "submit_declared_size": len(entries),
                "submit_entries": entries,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["sim_date", "submit_call_seq"]).reset_index(drop=True)
    return df


def map_submit_calls_to_predictions(
    pred_df: pd.DataFrame,
    submit_calls_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map each submit_forecasts call entry to actions.jsonl prediction line_no.
    Mapping uses chronological order, same day, qid + outcomes match (qid-only fallback).
    """
    if pred_df.empty or submit_calls_df.empty:
        return pd.DataFrame()

    pred = pred_df.sort_values("line_no").copy()
    pred["canon_outcomes"] = pred["outcomes"].apply(canonicalize_outcomes)

    by_date: Dict[pd.Timestamp, List[Dict[str, Any]]] = defaultdict(list)
    for row in pred.itertuples(index=False):
        by_date[row.sim_date].append(
            {
                "line_no": int(row.line_no),
                "qid": str(row.qid),
                "canon_outcomes": row.canon_outcomes,
            }
        )

    consumed: set[int] = set()
    date_cursor: Dict[pd.Timestamp, int] = defaultdict(int)
    mapped_rows: List[Dict[str, Any]] = []

    for call in submit_calls_df.sort_values(["sim_date", "submit_call_seq"]).itertuples(index=False):
        date = call.sim_date
        entries = call.submit_entries if isinstance(call.submit_entries, list) else []
        day_preds = by_date.get(date, [])
        cursor = int(date_cursor.get(date, 0))
        matched: List[Tuple[int, int]] = []  # (line_no, xml_index)

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            qid = entry.get("qid")
            outcomes = entry.get("outcomes")
            xml_index = int(entry.get("xml_index", 0))
            if not isinstance(qid, str) or not isinstance(outcomes, dict):
                continue

            found_idx: Optional[int] = None
            # Pass 1: qid + exact outcomes.
            for i in range(cursor, len(day_preds)):
                p = day_preds[i]
                if p["line_no"] in consumed:
                    continue
                if p["qid"] != qid:
                    continue
                if outcomes_equal(p["canon_outcomes"], outcomes):
                    found_idx = i
                    break

            # Pass 2 fallback: qid-only.
            if found_idx is None:
                for i in range(cursor, len(day_preds)):
                    p = day_preds[i]
                    if p["line_no"] in consumed:
                        continue
                    if p["qid"] == qid:
                        found_idx = i
                        break

            if found_idx is None:
                continue

            p = day_preds[found_idx]
            consumed.add(p["line_no"])
            cursor = found_idx + 1
            matched.append((p["line_no"], xml_index))

        date_cursor[date] = cursor
        realized_size = len(matched)
        for line_no, xml_index in matched:
            mapped_rows.append(
                {
                    "line_no": int(line_no),
                    "sim_date": date,
                    "submit_call_seq": int(call.submit_call_seq),
                    "submit_batch_size_realized": int(realized_size),
                    "submit_batch_size_declared": int(call.submit_declared_size),
                    "submit_xml_index": int(xml_index),
                }
            )

    out = pd.DataFrame(mapped_rows)
    if not out.empty:
        out = out.sort_values(["line_no"]).reset_index(drop=True)
    return out


def attach_submit_batch_info(scored: pd.DataFrame, batch_map: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = scored.copy()
    if batch_map.empty:
        out["submit_call_seq"] = np.nan
        out["submit_batch_size_realized"] = np.nan
        out["submit_batch_size_declared"] = np.nan
        out["submit_xml_index"] = np.nan
    else:
        keep = [
            "line_no",
            "submit_call_seq",
            "submit_batch_size_realized",
            "submit_batch_size_declared",
            "submit_xml_index",
        ]
        out = out.merge(batch_map[keep], on="line_no", how="left")
    out["raw_brier"] = 1.0 - out["brier_skill"]
    out["submit_is_single"] = out["submit_batch_size_realized"] == 1
    out["submit_is_group"] = out["submit_batch_size_realized"] > 1
    return out


def load_timing_stats(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "agents" / "allQ_gpt-oss-120b_001" / "timing_stats.jsonl"
    if not path.exists():
        return pd.DataFrame()

    rows = load_jsonl(path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def build_scored_events(
    pred_df: pd.DataFrame,
    res_df: pd.DataFrame,
    matcher_cache: Dict[str, bool],
) -> pd.DataFrame:
    if pred_df.empty or res_df.empty:
        return pd.DataFrame()

    truth_by_qid = {
        str(row.qid): str(row.ground_truth)
        for row in res_df.drop_duplicates("qid").itertuples(index=False)
    }
    pred_scored = pred_df[pred_df["qid"].isin(truth_by_qid.keys())].copy()
    if pred_scored.empty:
        return pred_scored

    metric_rows: List[Dict[str, Any]] = []
    for row in pred_scored.itertuples(index=False):
        truth = truth_by_qid[str(row.qid)]
        metrics = score_prediction(row.outcomes, truth, str(row.qid), matcher_cache)
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)
    scored = pd.concat([pred_scored.reset_index(drop=True), metrics_df], axis=1)
    scored["ground_truth"] = scored["qid"].map(truth_by_qid)

    scored = scored.sort_values(["qid", "line_no"]).reset_index(drop=True)
    scored["forecast_index"] = scored.groupby("qid").cumcount() + 1
    scored["first_forecast_date"] = scored.groupby("qid")["sim_date"].transform("min")
    scored["is_update"] = scored["forecast_index"] > 1
    scored["prev_brier_skill"] = scored.groupby("qid")["brier_skill"].shift(1)
    scored["skill_delta"] = scored["brier_skill"] - scored["prev_brier_skill"]
    scored["prev_top_prob"] = scored.groupby("qid")["top_prob"].shift(1)
    scored["top_prob_delta"] = scored["top_prob"] - scored["prev_top_prob"]
    scored["prev_outcomes"] = scored.groupby("qid")["outcomes"].shift(1)
    scored["changed_from_prev"] = scored.apply(
        lambda r: bool(isinstance(r.prev_outcomes, dict) and r.outcomes != r.prev_outcomes), axis=1
    )
    scored["l1_shift"] = scored.apply(
        lambda r: l1_shift(r.outcomes, r.prev_outcomes), axis=1
    )
    scored = scored.sort_values("line_no").reset_index(drop=True)
    return scored


def summarize_forecast_index(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()

    summary = (
        scored.groupby("forecast_index")
        .agg(
            n=("qid", "count"),
            brier_skill=("brier_skill", "mean"),
            accuracy=("top_correct", "mean"),
            top_prob=("top_prob", "mean"),
            truth_prob=("truth_prob", "mean"),
            entropy=("entropy", "mean"),
        )
        .reset_index()
    )
    summary["overconfidence_gap"] = summary["top_prob"] - summary["accuracy"]
    return summary


def summarize_qid(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for qid, g in scored.sort_values(["qid", "line_no"]).groupby("qid"):
        first = g.iloc[0]
        last = g.iloc[-1]
        updates = g[g["forecast_index"] > 1]
        rows.append(
            {
                "qid": qid,
                "n_forecasts": int(len(g)),
                "first_date": first["sim_date"],
                "last_date": last["sim_date"],
                "first_skill": float(first["brier_skill"]),
                "last_skill": float(last["brier_skill"]),
                "skill_change_last_minus_first": float(last["brier_skill"] - first["brier_skill"]),
                "first_accuracy": int(first["top_correct"]),
                "last_accuracy": int(last["top_correct"]),
                "changed_updates": int((updates["changed_from_prev"]).sum()),
                "unchanged_updates": int((~updates["changed_from_prev"]).sum()) if len(updates) else 0,
                "worsening_updates": int((updates["skill_delta"] < 0).sum()) if len(updates) else 0,
                "improving_updates": int((updates["skill_delta"] > 0).sum()) if len(updates) else 0,
                "mean_update_skill_delta": float(updates["skill_delta"].mean()) if len(updates) else np.nan,
                "mean_update_top_prob_delta": float(updates["top_prob_delta"].mean()) if len(updates) else np.nan,
                "mean_l1_shift_changed_updates": float(
                    updates.loc[updates["changed_from_prev"], "l1_shift"].mean()
                )
                if len(updates)
                else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["skill_change_last_minus_first", "n_forecasts"], ascending=[True, False]).reset_index(drop=True)
    return out


def summarize_update_single_vs_group(
    scored: pd.DataFrame,
    day0_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compare non-day0 updates submitted alone (batch size 1) vs grouped (>1).
    """
    if scored.empty:
        return pd.DataFrame()

    base = scored[
        (scored["sim_date"] > day0_date)
        & (scored["is_update"])
        & (scored["submit_batch_size_realized"].notna())
    ].copy()
    if base.empty:
        return pd.DataFrame()

    rows = []
    for label, mask in [
        ("single_update_submit", base["submit_batch_size_realized"] == 1),
        ("grouped_update_submit", base["submit_batch_size_realized"] > 1),
    ]:
        g = base[mask]
        rows.append(
            {
                "group_type": label,
                "n": int(len(g)),
                "accuracy": float(g["top_correct"].mean()) if len(g) else np.nan,
                "brier_skill": float(g["brier_skill"].mean()) if len(g) else np.nan,
                "raw_brier": float(g["raw_brier"].mean()) if len(g) else np.nan,
                "top_prob": float(g["top_prob"].mean()) if len(g) else np.nan,
                "overconfidence_gap": float(g["top_prob"].mean() - g["top_correct"].mean()) if len(g) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 2:
        s = out.iloc[0]
        g = out.iloc[1]
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "group_type": "single_minus_group",
                            "n": np.nan,
                            "accuracy": s["accuracy"] - g["accuracy"],
                            "brier_skill": s["brier_skill"] - g["brier_skill"],
                            "raw_brier": s["raw_brier"] - g["raw_brier"],
                            "top_prob": s["top_prob"] - g["top_prob"],
                            "overconfidence_gap": s["overconfidence_gap"] - g["overconfidence_gap"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return out


def summarize_update_batch_position(
    scored: pd.DataFrame,
    day0_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    For grouped update submits, summarize performance by within-submit XML index.
    """
    if scored.empty:
        return pd.DataFrame()

    df = scored[
        (scored["sim_date"] > day0_date)
        & (scored["is_update"])
        & (scored["submit_batch_size_realized"] > 1)
        & (scored["submit_xml_index"].notna())
    ].copy()
    if df.empty:
        return pd.DataFrame()

    df["submit_xml_index"] = df["submit_xml_index"].astype(int)
    df["submit_batch_size_realized"] = df["submit_batch_size_realized"].astype(int)
    df["submit_xml_index_norm"] = df["submit_xml_index"] / df["submit_batch_size_realized"].replace(0, np.nan)

    out = (
        df.groupby("submit_xml_index")
        .agg(
            n=("qid", "count"),
            accuracy=("top_correct", "mean"),
            brier_skill=("brier_skill", "mean"),
            raw_brier=("raw_brier", "mean"),
            top_prob=("top_prob", "mean"),
            overconfidence_gap=("top_prob", lambda s: float(s.mean())),
            mean_batch_size=("submit_batch_size_realized", "mean"),
            mean_index_norm=("submit_xml_index_norm", "mean"),
        )
        .reset_index()
        .sort_values("submit_xml_index")
    )
    out["overconfidence_gap"] = out["top_prob"] - out["accuracy"]
    return out


def summarize_day_level(
    pred_df_all: pd.DataFrame,
    scored: pd.DataFrame,
    timing_df: pd.DataFrame,
    outputs_df: pd.DataFrame,
    daily_metrics_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    if pred_df_all.empty:
        raise ValueError("No predictions found in actions.jsonl")

    pred_counts = (
        pred_df_all.groupby("sim_date")
        .size()
        .rename("predictions_today")
        .reset_index()
        .rename(columns={"sim_date": "date"})
    )

    day0_date = pred_counts["date"].min()

    tool_rows = []
    if not outputs_df.empty:
        grouped = outputs_df.groupby("sim_date")
        for day, g in grouped:
            tool_counts = Counter(x for x in g["tool_name"] if isinstance(x, str))
            tool_rows.append(
                {
                    "date": day,
                    "model_output_rows": int(len(g)),
                    "qid_nonnull_rows": int(g["qid"].notna().sum()),
                    "qid_null_rows": int(g["qid"].isna().sum()),
                    "submit_calls": int(tool_counts.get("submit_forecasts", 0)),
                    "submit_forecasts_total": int(g.loc[g["tool_name"] == "submit_forecasts", "submit_size"].sum()),
                    "search_calls_output": int(tool_counts.get("search_news", 0)),
                    "query_calls_output": int(tool_counts.get("query_df", 0)),
                    "next_day_calls_output": int(tool_counts.get("next_day", 0)),
                    "memory_update_calls_output": int(tool_counts.get("update_memory", 0)),
                }
            )
    tool_df = pd.DataFrame(tool_rows)

    day_df = pred_counts.copy()
    if not timing_df.empty:
        cols = [
            "date",
            "llm_count",
            "search_count",
            "df_query_count",
            "reasoning_tokens",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ]
        available = [c for c in cols if c in timing_df.columns]
        day_df = day_df.merge(timing_df[available], on="date", how="left")
    if not tool_df.empty:
        day_df = day_df.merge(tool_df, on="date", how="left")

    if not daily_metrics_df.empty:
        dm = daily_metrics_df.copy()
        dm["date"] = pd.to_datetime(dm["date"])
        keep = ["date", "avg_brier", "accuracy", "total_predictions"]
        keep = [c for c in keep if c in dm.columns]
        day_df = day_df.merge(dm[keep], on="date", how="left")

    day_df["search_per_prediction"] = day_df["search_count"] / day_df["predictions_today"]
    day_df["llm_per_prediction"] = day_df["llm_count"] / day_df["predictions_today"]
    day_df["reasoning_tokens_per_prediction"] = day_df["reasoning_tokens"] / day_df["predictions_today"]
    day_df["submit_calls_per_prediction"] = day_df["submit_calls"] / day_df["predictions_today"]
    day_df["submit_batch_size"] = day_df["submit_forecasts_total"] / day_df["submit_calls"].replace(0, np.nan)
    day_df["is_day0"] = day_df["date"] == day0_date
    day_df["is_update_day"] = day_df["date"] > day0_date
    day_df = day_df.sort_values("date").reset_index(drop=True)

    if not scored.empty:
        changed = (
            scored[scored["is_update"]]
            .groupby("sim_date")
            .agg(
                update_events=("qid", "count"),
                changed_updates=("changed_from_prev", "sum"),
                worsening_updates=("skill_delta", lambda s: int((s < 0).sum())),
                improving_updates=("skill_delta", lambda s: int((s > 0).sum())),
            )
            .reset_index()
            .rename(columns={"sim_date": "date"})
        )
        day_df = day_df.merge(changed, on="date", how="left")

    return day_df, day0_date


def build_fixed_cohort_trend(
    scored: pd.DataFrame,
    daily_metrics_df: pd.DataFrame,
    day0_date: pd.Timestamp,
) -> pd.DataFrame:
    if scored.empty or daily_metrics_df.empty:
        return pd.DataFrame()

    eval_days = pd.to_datetime(daily_metrics_df["date"]).sort_values().unique()
    eval_index = pd.DatetimeIndex(eval_days)

    timeline_chunks: List[pd.DataFrame] = []
    by_qid = scored.sort_values(["qid", "sim_date", "line_no"]).groupby("qid")
    for qid, g in by_qid:
        # Keep only the final update each day for snapshot carry-forward.
        per_day = g.drop_duplicates(subset=["sim_date"], keep="last")[
            ["sim_date", "brier_skill", "top_correct", "forecast_index"]
        ].copy()
        per_day = per_day.set_index("sim_date").reindex(eval_index).ffill()
        per_day["qid"] = qid
        per_day["date"] = eval_index
        timeline_chunks.append(per_day.reset_index(drop=True))

    timeline = pd.concat(timeline_chunks, ignore_index=True)

    first_dates = (
        scored.groupby("qid")["first_forecast_date"]
        .min()
        .rename("first_forecast_date")
        .reset_index()
    )
    timeline = timeline.merge(first_dates, on="qid", how="left")
    timeline["in_day0_cohort"] = timeline["first_forecast_date"] == day0_date

    out_rows: List[pd.DataFrame] = []
    for cohort_name, mask in [
        ("day0_cohort", timeline["in_day0_cohort"]),
        ("all_resolved", timeline["qid"].notna()),
    ]:
        g = (
            timeline[mask]
            .groupby("date")
            .agg(
                mean_skill=("brier_skill", "mean"),
                mean_accuracy=("top_correct", "mean"),
                qid_count=("qid", "nunique"),
            )
            .reset_index()
        )
        g["cohort"] = cohort_name
        out_rows.append(g)

    out = pd.concat(out_rows, ignore_index=True)
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_forecast_index_summary(summary: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    if summary.empty:
        return
    df = summary[summary["n"] >= 5].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["forecast_index"], df["brier_skill"], marker="o", label="Mean Brier Skill")
    ax.plot(df["forecast_index"], df["accuracy"], marker="s", label="Mean Accuracy")
    ax.plot(df["forecast_index"], df["top_prob"], marker="^", label="Mean Top Probability")
    ax.set_xlabel("Forecast Index (1=first forecast)")
    ax.set_ylabel("Score / Probability")
    ax.set_title(f"{title_prefix}: Score vs Forecast Index")
    ax.grid(alpha=0.3)
    ax.legend()

    # Show support size on secondary axis.
    ax2 = ax.twinx()
    ax2.bar(df["forecast_index"], df["n"], alpha=0.15, color="gray", label="N forecasts")
    ax2.set_ylabel("Count")
    ax2.set_ylim(0, max(df["n"]) * 1.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_calibration_by_index(summary: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    if summary.empty:
        return
    df = summary[summary["n"] >= 5].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["forecast_index"], df["top_prob"], marker="^", label="Mean Top Prob")
    ax.plot(df["forecast_index"], df["accuracy"], marker="o", label="Empirical Accuracy")
    ax.plot(df["forecast_index"], df["overconfidence_gap"], marker="s", label="Overconfidence Gap")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel("Forecast Index")
    ax.set_ylabel("Probability / Gap")
    ax.set_title(f"{title_prefix}: Calibration by Forecast Index")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_update_delta_hist(scored: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    if scored.empty:
        return
    updates = scored[scored["is_update"]].copy()
    if updates.empty:
        return

    all_delta = updates["skill_delta"].dropna()
    changed_delta = updates.loc[updates["changed_from_prev"], "skill_delta"].dropna()
    if all_delta.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(all_delta, bins=60, alpha=0.45, label="All updates")
    if not changed_delta.empty:
        ax.hist(changed_delta, bins=60, alpha=0.55, label="Changed updates only")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Brier Skill Delta vs Previous Forecast")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix}: Update Skill Delta Distribution")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_day0_vs_update_tools(day_summary: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    if day_summary.empty:
        return
    day0 = day_summary[day_summary["is_day0"]]
    updates = day_summary[day_summary["is_update_day"]]
    if day0.empty or updates.empty:
        return

    d0 = day0.iloc[0]
    upd_mean = updates.mean(numeric_only=True)

    metrics = [
        ("search_per_prediction", "Search calls / prediction"),
        ("llm_per_prediction", "LLM turns / prediction"),
        ("reasoning_tokens_per_prediction", "Reasoning tokens / prediction"),
        ("submit_calls_per_prediction", "Submit calls / prediction"),
    ]

    labels = []
    v0 = []
    vu = []
    for col, label in metrics:
        if col in day_summary.columns:
            labels.append(label)
            v0.append(float(d0.get(col, np.nan)))
            vu.append(float(upd_mean.get(col, np.nan)))

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, v0, width=w, label=f"Day 0 ({d0['date'].date()})")
    ax.bar(x + w / 2, vu, width=w, label="Update-day mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(f"{title_prefix}: Day 0 vs Update-Day Tool Usage")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_update_single_vs_group(
    single_group_df: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
) -> None:
    if single_group_df.empty:
        return
    df = single_group_df[single_group_df["group_type"].isin(["single_update_submit", "grouped_update_submit"])].copy()
    if df.empty:
        return

    order = ["single_update_submit", "grouped_update_submit"]
    df["group_type"] = pd.Categorical(df["group_type"], categories=order, ordered=True)
    df = df.sort_values("group_type")
    labels = ["Single", "Grouped"]
    x = np.arange(len(df))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.bar(x - w / 2, df["accuracy"], width=w, label="Accuracy")
    ax1.bar(x + w / 2, df["brier_skill"], width=w, label="Brier skill")
    ax1.set_ylabel("Score")
    ax1.set_title(f"{title_prefix}: Non-day0 Updates - Single vs Grouped Submit")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()

    ax2.bar(x - w / 2, df["raw_brier"], width=w, label="Raw Brier (lower better)")
    ax2.bar(x + w / 2, df["top_prob"], width=w, label="Top probability")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{l}\n(n={int(n)})" for l, n in zip(labels, df["n"])])
    ax2.set_ylabel("Value")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_update_batch_position_trend(
    batch_pos_df: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
) -> None:
    if batch_pos_df.empty:
        return
    df = batch_pos_df.copy().sort_values("submit_xml_index")
    # Keep at least modestly supported points to avoid extreme-noise tails.
    plot_df = df[df["n"] >= 3].copy()
    if plot_df.empty:
        plot_df = df.copy()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(plot_df["submit_xml_index"], plot_df["accuracy"], marker="o", label="Accuracy")
    ax.plot(plot_df["submit_xml_index"], plot_df["brier_skill"], marker="s", label="Brier skill")
    ax.plot(plot_df["submit_xml_index"], plot_df["raw_brier"], marker="^", label="Raw Brier (lower better)")
    ax.set_xlabel("Submit XML Index within batch (1,2,3,...)")
    ax.set_ylabel("Performance")
    ax.set_title(f"{title_prefix}: Grouped Updates - Submit Index vs Performance")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.bar(
        plot_df["submit_xml_index"],
        plot_df["n"],
        alpha=0.18,
        color="gray",
        label="N",
    )
    ax2.set_ylabel("Count")
    ax2.set_ylim(0, max(plot_df["n"]) * 1.25 if len(plot_df) else 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fixed_cohort_trend(
    trend_df: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
) -> None:
    if trend_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for cohort, g in trend_df.groupby("cohort"):
        ax1.plot(g["date"], g["mean_skill"], label=cohort)
        ax2.plot(g["date"], g["mean_accuracy"], label=cohort)

    ax1.set_ylabel("Mean Brier Skill")
    ax1.set_title(f"{title_prefix}: Fixed-Cohort Trend (carry-forward snapshots)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_ylabel("Mean Accuracy")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_qid_trajectory(g: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    if g.empty:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(g["forecast_index"], g["brier_skill"], marker="o")
    ax1.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax1.set_ylabel("Brier Skill")
    ax1.grid(alpha=0.3)

    ax2.plot(g["forecast_index"], g["top_correct"], marker="o", label="Top-choice correct (0/1)")
    ax2.plot(g["forecast_index"], g["top_prob"], marker="^", label="Top probability")
    ax2.set_xlabel("Forecast Index")
    ax2.set_ylabel("Accuracy / Confidence")
    ax2.grid(alpha=0.3)
    ax2.legend()

    qid = g["qid"].iloc[0]
    gt = g["ground_truth"].iloc[0]
    ax1.set_title(f"{title_prefix}: qid={qid} | truth={gt}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_summary_text(
    out_path: Path,
    run_name: str,
    day0_date: pd.Timestamp,
    scored: pd.DataFrame,
    day_summary: pd.DataFrame,
    qid_summary: pd.DataFrame,
    update_single_vs_group_summary: pd.DataFrame,
    update_batch_position_summary: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append(f"Run: {run_name}")
    lines.append(f"Day 0: {day0_date.date()}")
    lines.append("")

    if scored.empty:
        lines.append("No scored forecast events (resolved qids) found.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    first = scored[scored["forecast_index"] == 1]
    updates = scored[scored["is_update"]]
    changed_updates = updates[updates["changed_from_prev"]]

    def pct(x: float) -> str:
        return f"{100.0 * x:.1f}%"

    lines.append("Forecast-level:")
    lines.append(f"- Resolved qids analyzed: {scored['qid'].nunique()}")
    lines.append(f"- Forecast events analyzed: {len(scored)}")
    lines.append(
        f"- First forecasts: skill={first['brier_skill'].mean():.4f}, "
        f"acc={first['top_correct'].mean():.4f}, top_prob={first['top_prob'].mean():.4f}, "
        f"overconfidence_gap={(first['top_prob'].mean() - first['top_correct'].mean()):.4f}"
    )
    lines.append(
        f"- Update forecasts: skill={updates['brier_skill'].mean():.4f}, "
        f"acc={updates['top_correct'].mean():.4f}, top_prob={updates['top_prob'].mean():.4f}, "
        f"overconfidence_gap={(updates['top_prob'].mean() - updates['top_correct'].mean()):.4f}"
    )
    lines.append(
        f"- Update skill deltas: mean={updates['skill_delta'].mean():.4f}, "
        f"improving={pct((updates['skill_delta'] > 0).mean())}, "
        f"worsening={pct((updates['skill_delta'] < 0).mean())}, "
        f"unchanged={pct((updates['skill_delta'] == 0).mean())}"
    )
    if not changed_updates.empty:
        lines.append(
            f"- Changed updates only: n={len(changed_updates)}, "
            f"mean_skill_delta={changed_updates['skill_delta'].mean():.4f}, "
            f"worsening={pct((changed_updates['skill_delta'] < 0).mean())}, "
            f"mean_l1_shift={changed_updates['l1_shift'].mean():.4f}"
        )

    if not day_summary.empty:
        lines.append("")
        lines.append("Day-level tools:")
        d0 = day_summary[day_summary["is_day0"]]
        upd = day_summary[day_summary["is_update_day"]]
        if not d0.empty and not upd.empty:
            d0 = d0.iloc[0]
            upd_mean = upd.mean(numeric_only=True)
            lines.append(
                f"- Search/prediction: day0={d0.get('search_per_prediction', np.nan):.4f}, "
                f"updates_mean={upd_mean.get('search_per_prediction', np.nan):.4f}"
            )
            lines.append(
                f"- LLM turns/prediction: day0={d0.get('llm_per_prediction', np.nan):.4f}, "
                f"updates_mean={upd_mean.get('llm_per_prediction', np.nan):.4f}"
            )
            lines.append(
                f"- Reasoning tokens/prediction: day0={d0.get('reasoning_tokens_per_prediction', np.nan):.1f}, "
                f"updates_mean={upd_mean.get('reasoning_tokens_per_prediction', np.nan):.1f}"
            )
            lines.append(
                f"- Submit batch size: day0={d0.get('submit_batch_size', np.nan):.2f}, "
                f"updates_mean={upd_mean.get('submit_batch_size', np.nan):.2f}"
            )

    if not qid_summary.empty:
        lines.append("")
        lines.append("Worst last-vs-first skill qids (top 10):")
        for row in qid_summary.head(10).itertuples(index=False):
            lines.append(
                f"- qid={row.qid}: n={row.n_forecasts}, "
                f"skill_change={row.skill_change_last_minus_first:.4f}, "
                f"changed_updates={row.changed_updates}, "
                f"worsening_updates={row.worsening_updates}"
            )

    if not update_single_vs_group_summary.empty:
        lines.append("")
        lines.append("Non-day0 updates: single-submit vs grouped-submit:")
        for row in update_single_vs_group_summary.itertuples(index=False):
            n_val = "nan" if pd.isna(row.n) else f"{int(row.n)}"
            lines.append(
                f"- {row.group_type}: n={n_val}, "
                f"acc={row.accuracy:.4f}, brier_skill={row.brier_skill:.4f}, "
                f"raw_brier={row.raw_brier:.4f}, top_prob={row.top_prob:.4f}, "
                f"overconfidence_gap={row.overconfidence_gap:.4f}"
            )

    if not update_batch_position_summary.empty:
        lines.append("")
        lines.append("Grouped-update batch position trend (first 10 indices):")
        for row in update_batch_position_summary.sort_values("submit_xml_index").head(10).itertuples(index=False):
            lines.append(
                f"- idx={int(row.submit_xml_index)}: n={int(row.n)}, "
                f"acc={row.accuracy:.4f}, brier_skill={row.brier_skill:.4f}, raw_brier={row.raw_brier:.4f}"
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def analyze_run(
    run_dir: Path,
    per_qid_min_forecasts: int,
    max_per_qid_plots: int,
) -> RunArtifacts:
    run_dir = run_dir.resolve()
    run_name = f"{run_dir.parent.name}/{run_dir.name}"
    output_dir = run_dir / "plots" / "update_analysis"
    ensure_dir(output_dir)

    pred_df_all, res_df = load_actions(run_dir)
    matcher_cache = load_matcher_cache(run_dir)
    scored = build_scored_events(pred_df_all, res_df, matcher_cache)

    daily_metrics_path = run_dir / "daily_metrics.csv"
    daily_metrics_df = pd.read_csv(daily_metrics_path) if daily_metrics_path.exists() else pd.DataFrame()

    timing_df = load_timing_stats(run_dir)
    outputs_df = load_model_outputs(run_dir)
    submit_calls_df = load_submit_calls(run_dir)
    batch_map_df = map_submit_calls_to_predictions(pred_df_all, submit_calls_df)
    scored = attach_submit_batch_info(scored, batch_map_df)
    day_summary, day0_date = summarize_day_level(
        pred_df_all=pred_df_all,
        scored=scored,
        timing_df=timing_df,
        outputs_df=outputs_df,
        daily_metrics_df=daily_metrics_df,
    )

    forecast_index_summary = summarize_forecast_index(scored)
    qid_summary = summarize_qid(scored)
    trend_df = build_fixed_cohort_trend(scored, daily_metrics_df, day0_date)
    update_single_vs_group_summary = summarize_update_single_vs_group(scored, day0_date)
    update_batch_position_summary = summarize_update_batch_position(scored, day0_date)

    # Persist tabular outputs.
    if not scored.empty:
        scored_to_save = scored.copy()
        scored_to_save["sim_date"] = scored_to_save["sim_date"].dt.strftime("%Y-%m-%d")
        scored_to_save["first_forecast_date"] = scored_to_save["first_forecast_date"].dt.strftime("%Y-%m-%d")
        scored_to_save.to_csv(output_dir / "forecast_events_scored.csv", index=False)
    if not batch_map_df.empty:
        batch_map_to_save = batch_map_df.copy()
        batch_map_to_save["sim_date"] = batch_map_to_save["sim_date"].dt.strftime("%Y-%m-%d")
        batch_map_to_save.to_csv(output_dir / "submit_call_mapping.csv", index=False)
    if not forecast_index_summary.empty:
        forecast_index_summary.to_csv(output_dir / "forecast_index_summary.csv", index=False)
    if not qid_summary.empty:
        qid_summary_to_save = qid_summary.copy()
        qid_summary_to_save["first_date"] = pd.to_datetime(qid_summary_to_save["first_date"]).dt.strftime("%Y-%m-%d")
        qid_summary_to_save["last_date"] = pd.to_datetime(qid_summary_to_save["last_date"]).dt.strftime("%Y-%m-%d")
        qid_summary_to_save.to_csv(output_dir / "qid_summary.csv", index=False)
    if not update_single_vs_group_summary.empty:
        update_single_vs_group_summary.to_csv(output_dir / "update_single_vs_group_summary.csv", index=False)
    if not update_batch_position_summary.empty:
        update_batch_position_summary.to_csv(output_dir / "update_batch_position_summary.csv", index=False)
    if not day_summary.empty:
        day_summary_to_save = day_summary.copy()
        day_summary_to_save["date"] = day_summary_to_save["date"].dt.strftime("%Y-%m-%d")
        day_summary_to_save.to_csv(output_dir / "day_summary.csv", index=False)
    if not trend_df.empty:
        trend_to_save = trend_df.copy()
        trend_to_save["date"] = trend_to_save["date"].dt.strftime("%Y-%m-%d")
        trend_to_save.to_csv(output_dir / "fixed_cohort_trend.csv", index=False)

    # Global plots.
    plot_forecast_index_summary(
        forecast_index_summary,
        output_dir / "forecast_index_metrics.png",
        title_prefix=run_name,
    )
    plot_calibration_by_index(
        forecast_index_summary,
        output_dir / "calibration_by_forecast_index.png",
        title_prefix=run_name,
    )
    plot_update_delta_hist(
        scored,
        output_dir / "update_delta_distribution.png",
        title_prefix=run_name,
    )
    plot_day0_vs_update_tools(
        day_summary,
        output_dir / "day0_vs_update_tool_usage.png",
        title_prefix=run_name,
    )
    plot_update_single_vs_group(
        update_single_vs_group_summary,
        output_dir / "update_single_vs_group.png",
        title_prefix=run_name,
    )
    plot_update_batch_position_trend(
        update_batch_position_summary,
        output_dir / "update_batch_position_trend.png",
        title_prefix=run_name,
    )
    plot_fixed_cohort_trend(
        trend_df,
        output_dir / "fixed_cohort_trend.png",
        title_prefix=run_name,
    )

    # Per-qid trajectory plots.
    per_qid_dir = output_dir / "per_qid"
    ensure_dir(per_qid_dir)
    if not scored.empty:
        candidates = scored.groupby("qid").size().rename("n_forecasts").reset_index()
        candidates = candidates[candidates["n_forecasts"] >= per_qid_min_forecasts]
        if not qid_summary.empty:
            candidates = candidates.merge(
                qid_summary[["qid", "skill_change_last_minus_first"]],
                on="qid",
                how="left",
            ).sort_values(["skill_change_last_minus_first", "n_forecasts"], ascending=[True, False])
        qids = list(candidates["qid"])
        if max_per_qid_plots > 0:
            qids = qids[:max_per_qid_plots]
        for qid in qids:
            g = scored[scored["qid"] == qid].sort_values("forecast_index").copy()
            plot_per_qid_trajectory(
                g,
                per_qid_dir / f"qid_{qid}.png",
                title_prefix=run_name,
            )

    write_summary_text(
        output_dir / "summary.txt",
        run_name=run_name,
        day0_date=day0_date,
        scored=scored,
        day_summary=day_summary,
        qid_summary=qid_summary,
        update_single_vs_group_summary=update_single_vs_group_summary,
        update_batch_position_summary=update_batch_position_summary,
    )

    return RunArtifacts(
        run_dir=run_dir,
        output_dir=output_dir,
        forecast_events=scored,
        forecast_index_summary=forecast_index_summary,
        qid_summary=qid_summary,
        day_summary=day_summary,
        update_single_vs_group_summary=update_single_vs_group_summary,
        update_batch_position_summary=update_batch_position_summary,
        day0_date=day0_date,
        run_name=run_name,
    )


def plot_cross_run_comparison(artifacts: List[RunArtifacts], output_dir: Path) -> None:
    if not artifacts:
        return
    ensure_dir(output_dir)

    # Comparison: mean skill vs forecast index.
    fig, ax = plt.subplots(figsize=(11, 6))
    for a in artifacts:
        df = a.forecast_index_summary
        if df.empty:
            continue
        d = df[df["n"] >= 5]
        if d.empty:
            continue
        ax.plot(d["forecast_index"], d["brier_skill"], marker="o", label=a.run_name)
    ax.set_xlabel("Forecast Index")
    ax.set_ylabel("Mean Brier Skill")
    ax.set_title("Cross-run: Mean Brier Skill vs Forecast Index")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_brier_skill_by_forecast_index.png", dpi=180)
    plt.close(fig)

    # Comparison: calibration gap vs forecast index.
    fig, ax = plt.subplots(figsize=(11, 6))
    for a in artifacts:
        df = a.forecast_index_summary
        if df.empty:
            continue
        d = df[df["n"] >= 5]
        if d.empty:
            continue
        ax.plot(d["forecast_index"], d["overconfidence_gap"], marker="o", label=a.run_name)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Forecast Index")
    ax.set_ylabel("TopProb - Accuracy")
    ax.set_title("Cross-run: Overconfidence Gap vs Forecast Index")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_overconfidence_gap_by_forecast_index.png", dpi=180)
    plt.close(fig)

    # Comparison: day0 vs update search/forecast.
    rows = []
    for a in artifacts:
        if a.day_summary.empty:
            continue
        d0 = a.day_summary[a.day_summary["is_day0"]]
        upd = a.day_summary[a.day_summary["is_update_day"]]
        if d0.empty or upd.empty:
            continue
        d0 = d0.iloc[0]
        upd_mean = upd.mean(numeric_only=True)
        rows.append(
            {
                "run_name": a.run_name,
                "search_per_pred_day0": float(d0.get("search_per_prediction", np.nan)),
                "search_per_pred_updates": float(upd_mean.get("search_per_prediction", np.nan)),
                "llm_per_pred_day0": float(d0.get("llm_per_prediction", np.nan)),
                "llm_per_pred_updates": float(upd_mean.get("llm_per_prediction", np.nan)),
            }
        )
    if rows:
        t = pd.DataFrame(rows)
        x = np.arange(len(t))
        w = 0.2
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - 1.5 * w, t["search_per_pred_day0"], width=w, label="Search/pred day0")
        ax.bar(x - 0.5 * w, t["search_per_pred_updates"], width=w, label="Search/pred updates")
        ax.bar(x + 0.5 * w, t["llm_per_pred_day0"], width=w, label="LLM/pred day0")
        ax.bar(x + 1.5 * w, t["llm_per_pred_updates"], width=w, label="LLM/pred updates")
        ax.set_xticks(x)
        ax.set_xticklabels(t["run_name"], rotation=15, ha="right")
        ax.set_ylabel("Calls per prediction")
        ax.set_title("Cross-run: Day0 vs Update tool intensity")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "comparison_day0_vs_update_tool_intensity.png", dpi=180)
        plt.close(fig)

    # Comparison: single vs grouped updates.
    rows = []
    for a in artifacts:
        g = a.update_single_vs_group_summary
        if g.empty:
            continue
        single = g[g["group_type"] == "single_update_submit"]
        grouped = g[g["group_type"] == "grouped_update_submit"]
        if single.empty or grouped.empty:
            continue
        single = single.iloc[0]
        grouped = grouped.iloc[0]
        rows.append(
            {
                "run_name": a.run_name,
                "acc_single": float(single["accuracy"]),
                "acc_group": float(grouped["accuracy"]),
                "skill_single": float(single["brier_skill"]),
                "skill_group": float(grouped["brier_skill"]),
            }
        )
    if rows:
        t = pd.DataFrame(rows)
        x = np.arange(len(t))
        w = 0.2
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - 1.5 * w, t["acc_single"], width=w, label="Accuracy single")
        ax.bar(x - 0.5 * w, t["acc_group"], width=w, label="Accuracy grouped")
        ax.bar(x + 0.5 * w, t["skill_single"], width=w, label="Brier skill single")
        ax.bar(x + 1.5 * w, t["skill_group"], width=w, label="Brier skill grouped")
        ax.set_xticks(x)
        ax.set_xticklabels(t["run_name"], rotation=15, ha="right")
        ax.set_ylabel("Performance")
        ax.set_title("Cross-run: Non-day0 updates single vs grouped submits")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "comparison_update_single_vs_group.png", dpi=180)
        plt.close(fig)

    # Comparison: grouped-update batch index trend (accuracy only).
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    for a in artifacts:
        df = a.update_batch_position_summary
        if df.empty:
            continue
        d = df[df["n"] >= 5].sort_values("submit_xml_index")
        if d.empty:
            continue
        ax.plot(d["submit_xml_index"], d["accuracy"], marker="o", label=a.run_name)
        plotted = True
    if plotted:
        ax.set_xlabel("Submit XML Index")
        ax.set_ylabel("Accuracy")
        ax.set_title("Cross-run: Grouped-update batch position vs accuracy")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "comparison_batch_position_accuracy.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dirs = [Path(x).expanduser() for x in args.run_dir]

    artifacts: List[RunArtifacts] = []
    for run_dir in run_dirs:
        print(f"[analyze] {run_dir}")
        art = analyze_run(
            run_dir=run_dir,
            per_qid_min_forecasts=args.per_qid_min_forecasts,
            max_per_qid_plots=args.max_per_qid_plots,
        )
        artifacts.append(art)
        print(f"[saved] {art.output_dir}")

    comparison_dir = Path(args.comparison_output_dir).expanduser()
    plot_cross_run_comparison(artifacts, comparison_dir)
    print(f"[saved] {comparison_dir}")


if __name__ == "__main__":
    main()
