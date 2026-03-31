#!/usr/bin/env python3
"""
Union coverage: count distinct resolved questions where any of several runs
got a high-confidence top-k answer semantically correct — using **only**
on-disk matcher artifacts (matcher_cache.json + matcher.jsonl), no API calls.

Example (paths are examples; use directories that contain actions.jsonl):

  python analysis/src/union_highconf_topk_correct.py \\
    --run gpt41:/fast/sgoel/forecasting/current_sim/warmup_only_openrouter_gpt-4.1-mini_simday_2025-01-01_aljazeeraRecent2_reasoning_high_r00/<timestamp>:2025-01-01 \\
    --run qwen27:/fast/sgoel/forecasting/current_sim/warmup_only_qwen3.5-27b_simday_2025-08-01_aljazeeraRecent2_tokenbudget_r00/26-03-23-00-22-09:2025-08-01 \\
    --run glm5:/fast/sgoel/forecasting/current_sim/warmup_only_openrouter_glm-5_simday_2025-08-01_aljazeeraRecent2_actions20_r00/<timestamp>:2025-08-01 \\
    --run gemini:/fast/sgoel/forecasting/current_sim/warmup_only_openrouter_gemini-3-flash_simday_2025-08-01_aljazeeraRecent2_actions20_r00/<timestamp>:2025-08-01 \\
    --run deepseek:/fast/sgoel/forecasting/current_sim/warmup_only_openrouter_deepseek-v3.2_simday_2025-08-01_aljazeeraRecent2_actions10_reasoning_high_r00/<timestamp>:2025-08-01 \\
    --write-csv analysis/plots/union_highconf_qids.csv

Note: GPT-4.1-mini above uses sim day 2025-01-01; others use 2025-08-01 — not a
controlled same-information benchmark across those runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _norm(s: str) -> str:
    return str(s).strip().lower()


def _parse_run_arg(spec: str) -> Tuple[str, Path, str]:
    parts = spec.split(":")
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(
            "Each --run must be NAME:PATH:YYYY-MM-DD (colon in PATH is not supported)"
        )
    name = parts[0]
    sim_date = parts[-1]
    path_str = ":".join(parts[1:-1])
    if not name or not path_str:
        raise argparse.ArgumentTypeError(f"Invalid --run {spec!r}")
    return name, Path(path_str).expanduser(), sim_date


def resolve_actions_dir(path: Path) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Not a directory: {path}")
    if (path / "actions.jsonl").is_file():
        return path
    # Single timestamp child (common layout: ..._r00/26-..-..-..-..-..)
    children = [p for p in path.iterdir() if p.is_dir()]
    with_actions = [p for p in children if (p / "actions.jsonl").is_file()]
    if len(with_actions) == 1:
        return with_actions[0]
    if not with_actions:
        raise FileNotFoundError(
            f"No actions.jsonl in {path} or in exactly one subdirectory"
        )
    raise FileNotFoundError(
        f"Multiple subdirs with actions.jsonl under {path}; pass the timestamp dir explicitly"
    )


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] skip bad JSON {path}:{lineno}: {e}", file=sys.stderr)


def load_matcher_cache_json(path: Path) -> Dict[str, bool]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] could not read {path}: {e}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, bool] = {}
    for k, v in data.items():
        if isinstance(k, str):
            out[k] = bool(v)
    return out


def merge_bool_dict(dst: Dict[str, bool], src: Dict[str, bool]) -> None:
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
        elif v and not dst[k]:
            dst[k] = True


_JSONL_EQ_TYPES = frozenset({"check_guess", "is_equivalent", "expand_set"})


def ingest_matcher_jsonl(path: Path, dst: Dict[str, bool]) -> int:
    """Merge is_equivalent rows into dst; returns count of usable jsonl records."""
    if not path.is_file():
        return 0
    n = 0
    for rec in iter_jsonl(path):
        inp = rec.get("input")
        out = rec.get("output")
        if not isinstance(inp, dict) or not isinstance(out, dict):
            continue
        mtype = inp.get("type")
        if mtype not in _JSONL_EQ_TYPES:
            continue
        if "is_equivalent" not in out:
            continue
        pred = inp.get("predicted")
        gt = inp.get("ground_truth")
        if pred is None or gt is None:
            continue
        qid_raw = inp.get("question_id")
        qid = str(qid_raw) if qid_raw is not None else "None"
        pn = _norm(str(pred))
        gn = _norm(str(gt))
        key = f"{pn}|||{gn}|||{qid}"
        val = bool(out["is_equivalent"])
        n += 1
        if key not in dst:
            dst[key] = val
        elif val and not dst[key]:
            dst[key] = True
    return n


def cache_lookup(equiv_map: Dict[str, bool], candidate: str, ground_truth: str, qid: str) -> bool:
    cn = _norm(candidate)
    gn = _norm(ground_truth)
    if cn == gn:
        return True
    q = str(qid)
    for qkey in (q, "None"):
        key = f"{cn}|||{gn}|||{qkey}"
        if key in equiv_map and equiv_map[key]:
            return True
    return False


def load_actions_predictions_resolutions(
    actions_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    preds: List[Dict[str, Any]] = []
    ress: List[Dict[str, Any]] = []
    for rec in iter_jsonl(actions_path):
        rtype = rec.get("type")
        if rtype == "prediction":
            preds.append(rec)
        elif rtype == "resolution":
            ress.append(rec)
    return preds, ress


def last_prediction_per_qid(
    preds: Sequence[Dict[str, Any]], sim_date: str
) -> Dict[str, Dict[str, float]]:
    """Last prediction row per question_id for exact sim_date string."""
    last: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for i, rec in enumerate(preds):
        if rec.get("sim_date") != sim_date:
            continue
        qid = rec.get("question_id")
        if qid is None:
            continue
        outcomes = rec.get("outcomes")
        if not isinstance(outcomes, dict):
            continue
        qs = str(qid)
        parsed: Dict[str, float] = {}
        for k, v in outcomes.items():
            try:
                parsed[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        if not parsed:
            continue
        last[qs] = (i, {"outcomes": parsed})
    out: Dict[str, Dict[str, float]] = {}
    for qid, (_, payload) in last.items():
        out[qid] = payload["outcomes"]
    return out


def topk_highconf_candidates(
    outcomes: Dict[str, float], top_k: int, min_prob: float
) -> List[str]:
    ranked = sorted(outcomes.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [o for o, p in ranked if p >= min_prob]


def model_correct_on_q(
    outcomes: Dict[str, float],
    ground_truth: str,
    qid: str,
    equiv_map: Dict[str, bool],
    top_k: int,
    min_prob: float,
) -> bool:
    for c in topk_highconf_candidates(outcomes, top_k, min_prob):
        if cache_lookup(equiv_map, c, ground_truth, qid):
            return True
    return False


@dataclass
class RunSpec:
    name: str
    actions_dir: Path
    sim_date: str


def _truth_parquet_path_from_run(run_dir: Path) -> Path:
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"No config.json in {run_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    dp = Path(str(cfg["dataset_path"]))
    split = str(cfg["split"])
    cand = dp / f"{split}-00000-of-00001.parquet"
    if cand.is_file():
        return cand
    matches = sorted(dp.glob(f"{split}-*.parquet"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No parquet for split {split!r} under {dp}")
    raise FileNotFoundError(
        f"Multiple parquets for split {split!r} under {dp}: pick one with --ground-truth-parquet"
    )


def load_truth_from_parquet(path: Path) -> Dict[str, str]:
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("parquet ground truth requires pandas; pip/uv install pandas") from e
    df = pd.read_parquet(path, columns=["qid", "answer"])
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        qid = str(row["qid"])
        ans = row["answer"]
        if pd.isna(ans):
            continue
        s = str(ans).strip()
        if s and s.lower() != "nan":
            out[qid] = s
    return out


def load_titles_from_parquet(path: Path) -> Dict[str, str]:
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("parquet titles require pandas; pip/uv install pandas") from e
    df = pd.read_parquet(path, columns=["qid", "question_title"])
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        qid = str(row["qid"])
        t = row["question_title"]
        if pd.isna(t):
            continue
        s = str(t).strip()
        if s:
            out[qid] = s
    return out


def top1_outcome_prob(outcomes: Optional[Dict[str, float]]) -> Optional[Tuple[str, float]]:
    if not outcomes:
        return None
    o, p = max(outcomes.items(), key=lambda kv: kv[1])
    return (str(o), float(p))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Union count of questions any run got right (top-k, p>=tau) using cache-only matcher."
    )
    p.add_argument(
        "--run",
        action="append",
        required=True,
        type=_parse_run_arg,
        metavar="NAME:PATH:YYYY-MM-DD",
        help="Repeat per model: label, directory (or _r00 parent), sim_date for predictions",
    )
    p.add_argument("--top-k", type=int, default=5, help="Top outcomes by probability (default 5)")
    p.add_argument(
        "--min-prob",
        type=float,
        default=0.4,
        help="Minimum probability among top-k to count as candidate (default 0.4)",
    )
    p.add_argument(
        "--write-csv",
        type=str,
        default=None,
        help="Optional output CSV path (parent dirs created as needed)",
    )
    p.add_argument(
        "--ground-truth-mode",
        choices=("parquet", "actions"),
        default="parquet",
        help=(
            "parquet: qid -> answer from OpenForesight split parquet (path from first --run "
            "config.json, or --ground-truth-parquet). "
            "actions: only qids with type=resolution lines in actions.jsonl (often very few)."
        ),
    )
    p.add_argument(
        "--ground-truth-parquet",
        type=str,
        default=None,
        help="Override parquet path for labels (default: infer from first run's config.json).",
    )
    p.add_argument(
        "--universe",
        choices=("union_resolved", "intersection"),
        default="union_resolved",
        help="Only for --ground-truth-mode actions: how to merge resolution qids across runs.",
    )
    p.add_argument(
        "--print-any-correct-detail",
        action="store_true",
        help=(
            "After the summary, print each question where at least one model was correct: "
            "qid, title (from parquet), ground-truth answer, and each model's single "
            "highest-probability outcome. Requires --ground-truth-mode parquet."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs: List[RunSpec] = []
    for name, raw_path, sim_date in args.run:
        adir = resolve_actions_dir(raw_path)
        runs.append(RunSpec(name=name, actions_dir=adir, sim_date=sim_date))

    equiv_map: Dict[str, bool] = {}
    n_cache_files = 0
    n_jsonl_lines_used = 0
    for spec in runs:
        mc = spec.actions_dir / "matcher_cache.json"
        d = load_matcher_cache_json(mc)
        if d:
            n_cache_files += 1
        merge_bool_dict(equiv_map, d)
        n_jsonl_lines_used += ingest_matcher_jsonl(spec.actions_dir / "matcher.jsonl", equiv_map)

    truth_by_qid: Dict[str, str] = {}
    resolutions_by_run: List[Dict[str, str]] = []
    truth_source_desc = ""
    pq: Optional[Path] = None

    if args.ground_truth_mode == "parquet":
        pq = (
            Path(args.ground_truth_parquet).expanduser()
            if args.ground_truth_parquet
            else _truth_parquet_path_from_run(runs[0].actions_dir)
        )
        truth_by_qid = load_truth_from_parquet(pq)
        truth_source_desc = str(pq.resolve())
    else:
        for spec in runs:
            _, ress = load_actions_predictions_resolutions(spec.actions_dir / "actions.jsonl")
            m: Dict[str, str] = {}
            for rec in ress:
                qid = rec.get("question_id")
                gt = rec.get("ground_truth")
                if qid is None or not isinstance(gt, str):
                    continue
                qs = str(qid)
                m[qs] = gt
                if qs not in truth_by_qid:
                    truth_by_qid[qs] = gt
                elif _norm(truth_by_qid[qs]) != _norm(gt):
                    print(
                        f"[warn] ground_truth mismatch for qid={qs!r} across runs; keeping first",
                        file=sys.stderr,
                    )
            resolutions_by_run.append(m)
        truth_source_desc = "actions.jsonl resolution records only"

    if args.ground_truth_mode == "parquet":
        universe_qids = set(truth_by_qid.keys())
    elif args.universe == "union_resolved":
        universe_qids = set(truth_by_qid.keys())
    else:
        universe_qids = set(truth_by_qid.keys())
        for m in resolutions_by_run:
            universe_qids &= set(m.keys())

    preds_by_run: Dict[str, Dict[str, Dict[str, float]]] = {}
    for spec in runs:
        preds, _ = load_actions_predictions_resolutions(spec.actions_dir / "actions.jsonl")
        preds_by_run[spec.name] = last_prediction_per_qid(preds, spec.sim_date)

    per_model_correct: Dict[str, set] = {spec.name: set() for spec in runs}
    n_models_per_q: Dict[str, int] = {}

    for qid in sorted(universe_qids):
        gt = truth_by_qid.get(qid)
        if not gt:
            continue
        n_ok = 0
        for spec in runs:
            outcomes = preds_by_run[spec.name].get(qid)
            if not outcomes:
                continue
            if model_correct_on_q(
                outcomes, gt, qid, equiv_map, args.top_k, args.min_prob
            ):
                per_model_correct[spec.name].add(qid)
                n_ok += 1
        if n_ok:
            n_models_per_q[qid] = n_ok

    any_model_correct = (
        set().union(*per_model_correct.values()) if per_model_correct else set()
    )
    n_with_any_prediction = sum(
        1
        for qid in universe_qids
        if any(qid in preds_by_run[s.name] for s in runs)
    )

    sim_dates = sorted({f"{s.name}={s.sim_date}" for s in runs})
    print()
    print(
        f"Number of questions where at least one model got a correct answer "
        f"(top-{args.top_k} outcomes, probability >= {args.min_prob}, cache-only matcher): "
        f"{len(any_model_correct)}"
    )
    print(
        f"Questions with ground truth in this evaluation: {len(universe_qids)} "
        f"(source: {truth_source_desc})"
    )
    print(
        f"Questions with at least one model prediction on the listed sim_date: {n_with_any_prediction}"
    )
    print()
    print("Details (cache-only semantics: no live matcher API).")
    print(
        "A model is correct on a question if some candidate outcome (top-k, p>=threshold) "
        "matches the label via normalized string equality or a True entry in merged "
        "matcher_cache.json / matcher.jsonl."
    )
    print(f"Merged matcher lookup size: {len(equiv_map)} entries")
    print(
        f"  (matcher_cache.json from {n_cache_files} run dir(s); "
        f"matcher.jsonl records merged: {n_jsonl_lines_used})"
    )
    print(f"Runs / sim_date: {', '.join(sim_dates)}")
    print(
        "Note: mixed sim dates (e.g. 2025-01-01 vs 2025-08-01) mean models did not all see "
        "the same information cutoff."
    )
    print()
    print("Per-model counts (questions that model got right under the rule above):")
    for spec in runs:
        print(f"  {spec.name}: {len(per_model_correct[spec.name])}")
    print()

    hist = Counter(n_models_per_q.values())
    if hist:
        print("How many models got each question right (among questions with at least one correct):")
        for k in sorted(hist.keys()):
            print(f"  {k} model(s): {hist[k]} question(s)")

    if args.print_any_correct_detail:
        if pq is None:
            print(
                "[warn] --print-any-correct-detail only applies with --ground-truth-mode parquet",
                file=sys.stderr,
            )
        elif any_model_correct:
            titles_by_qid = load_titles_from_parquet(pq)

            def _qid_sort_key(q: str) -> Tuple[int, str]:
                if str(q).isdigit():
                    return (0, str(int(q)))
                return (1, str(q))

            print()
            print(
                "=== Per-question detail (at least one model correct under top-k / min-prob rule; "
                "top prediction = single highest-probability outcome that day) ==="
            )
            for qid in sorted(any_model_correct, key=_qid_sort_key):
                title = titles_by_qid.get(qid, "").replace("\n", " ").strip()
                gt = truth_by_qid.get(qid, "")
                print()
                print(f"qid: {qid}")
                print(f"title: {title}")
                print(f"answer: {gt}")
                for spec in runs:
                    tp = top1_outcome_prob(preds_by_run[spec.name].get(qid))
                    if tp is None:
                        print(f"  {spec.name}: (no prediction)")
                    else:
                        o, pr = tp
                        print(f"  {spec.name}: {o!r} @ p={pr:.6f}")

    if args.write_csv:
        out_path = Path(args.write_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["qid", "ground_truth", "n_models_correct", "any_model_correct"]
        for spec in runs:
            fieldnames.append(f"correct__{spec.name}")
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for qid in sorted(universe_qids):
                gt = truth_by_qid.get(qid, "")
                row: Dict[str, Any] = {
                    "qid": qid,
                    "ground_truth": gt,
                    "n_models_correct": n_models_per_q.get(qid, 0),
                    "any_model_correct": 1 if qid in any_model_correct else 0,
                }
                for spec in runs:
                    row[f"correct__{spec.name}"] = (
                        1 if qid in per_model_correct[spec.name] else 0
                    )
                w.writerow(row)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
