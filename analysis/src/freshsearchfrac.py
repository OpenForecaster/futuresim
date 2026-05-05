#!/usr/bin/env python3
"""
Compute and plot the daily fraction of search results published after simulation day 0.

Input logs are expected to be model_raw*.jsonl files that contain prompt deltas with:
  SEARCH RESULTS:
  ...
  PUBLISHED: YYYY-MM-DD ...
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is not installed. Please install it with 'pip install matplotlib'.")
    sys.exit(1)

import plot_config  # noqa: F401  (applies project-wide science+serif style)


PUBLISHED_DATE_RE = re.compile(r"PUBLISHED:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class SearchEvent:
    sim_date: date
    published_dates: List[date]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze model_raw search results and plot the daily fraction of results "
            "published after simulation day 0."
        )
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help=(
            "Path to a run dir (containing agents/*/model_raw*.jsonl), an agent dir "
            "(containing model_raw*.jsonl), or a model_raw*.jsonl file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files. Defaults to analysis/plots/freshsearchfrac.",
    )
    parser.add_argument(
        "--plot-name",
        default=None,
        help="Output plot filename. Default includes simulation name.",
    )
    parser.add_argument(
        "--csv-name",
        default=None,
        help="Output CSV filename. Default includes simulation name.",
    )
    parser.add_argument(
        "--day0",
        default=None,
        help="Optional override for day 0 in YYYY-MM-DD format. Defaults to earliest sim_date.",
    )
    parser.add_argument(
        "--ema-span",
        type=int,
        default=7,
        help="EMA span for the main line (default: 7).",
    )
    parser.add_argument(
        "--raw-alpha",
        type=float,
        default=0.22,
        help="Opacity for the background raw line (default: 0.22).",
    )
    return parser.parse_args()


def parse_yyyy_mm_dd(value: str) -> date:
    parts = value.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid date format: {value}")
    year, month, day = (int(x) for x in parts)
    return date(year, month, day)


def parse_sim_date(value: object) -> Optional[date]:
    if not isinstance(value, str):
        return None
    if len(value) < 10:
        return None
    token = value[:10]
    try:
        return parse_yyyy_mm_dd(token)
    except Exception:
        return None


def extract_published_dates(prompt: str) -> List[date]:
    out: List[date] = []
    for match in PUBLISHED_DATE_RE.finditer(prompt):
        token = match.group(1)
        try:
            out.append(parse_yyyy_mm_dd(token))
        except Exception:
            continue
    return out


def _is_model_raw_file(path: Path) -> bool:
    return path.name in {"model_raw.jsonl", "model_raw_daily.jsonl", "model_raw_warmup.jsonl"}


def discover_model_raw_files(input_path: Path) -> List[Tuple[str, Path]]:
    """
    Return list of (agent_label, model_raw_path).
    """
    input_path = input_path.expanduser()
    files: List[Tuple[str, Path]] = []

    if input_path.is_file():
        if not _is_model_raw_file(input_path):
            raise FileNotFoundError(f"Expected a model_raw*.jsonl file, got: {input_path}")
        files.append((input_path.parent.name, input_path))
        return files

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    direct_candidates = [input_path / name for name in ("model_raw.jsonl", "model_raw_daily.jsonl", "model_raw_warmup.jsonl")]
    direct = [p for p in direct_candidates if p.is_file()]
    if direct:
        for p in direct:
            files.append((input_path.name, p))
        return files

    agents_dir = input_path / "agents"
    if agents_dir.is_dir():
        for p in sorted(agents_dir.glob("*/model_raw*.jsonl")):
            if _is_model_raw_file(p):
                files.append((p.parent.name, p))
        if files:
            return files

    for p in sorted(input_path.rglob("model_raw*.jsonl")):
        if _is_model_raw_file(p):
            files.append((p.parent.name, p))

    if not files:
        raise FileNotFoundError(f"No model_raw*.jsonl files found under: {input_path}")
    return files


def infer_default_output_dir() -> Path:
    return Path("analysis/plots/freshsearchfrac")


def infer_fallback_output_dir(input_path: Path) -> Path:
    base = Path("analysis/plots/freshsearchfrac")
    if input_path.is_file():
        label = input_path.parent.name
    else:
        label = input_path.name
    label = label.replace("/", "_")
    return base / label


def sanitize_name(value: str) -> str:
    value = value.strip()
    if not value:
        return "unknown_sim"
    return SAFE_NAME_RE.sub("_", value)


def infer_sim_name(input_path: Path, model_raw_files: Sequence[Tuple[str, Path]]) -> str:
    """
    Try to infer the simulation name.
    For .../<sim_name>/<run_timestamp>/agents/<agent>/model_raw*.jsonl, use <sim_name>.
    """
    candidates: List[str] = []
    for _, p in model_raw_files:
        # Expected path depth near run format.
        # p.parents[0]=agent_dir, [1]=agents, [2]=run_timestamp, [3]=sim_name
        if len(p.parents) >= 4 and p.parents[1].name == "agents":
            candidates.append(p.parents[3].name)
        elif len(p.parents) >= 3 and p.parents[0].name == "agents":
            candidates.append(p.parents[2].name)

    unique = sorted({x for x in candidates if x})
    if len(unique) == 1:
        return sanitize_name(unique[0])

    if input_path.is_file():
        return sanitize_name(input_path.parent.name)
    return sanitize_name(input_path.name)


def parse_search_events(model_raw_path: Path) -> Tuple[List[date], List[SearchEvent], int]:
    sim_dates: List[date] = []
    events: List[SearchEvent] = []
    bad_json_lines = 0

    with model_raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_json_lines += 1
                continue

            sim_date = parse_sim_date(record.get("sim_date"))
            if sim_date is None:
                continue
            sim_dates.append(sim_date)

            prompt = record.get("prompt")
            if not isinstance(prompt, str):
                continue
            if "SEARCH RESULTS:" not in prompt:
                continue

            published_dates = extract_published_dates(prompt)
            events.append(SearchEvent(sim_date=sim_date, published_dates=published_dates))

    return sim_dates, events, bad_json_lines


def aggregate_daily(
    sim_dates: Sequence[date],
    events: Sequence[SearchEvent],
    day0: date,
) -> List[Dict[str, object]]:
    all_days = sorted(set(sim_dates))

    search_prompts: Dict[date, int] = defaultdict(int)
    empty_search_prompts: Dict[date, int] = defaultdict(int)
    total_results: Dict[date, int] = defaultdict(int)
    fresh_results: Dict[date, int] = defaultdict(int)

    for event in events:
        search_prompts[event.sim_date] += 1
        if not event.published_dates:
            empty_search_prompts[event.sim_date] += 1
            continue
        for pub_date in event.published_dates:
            total_results[event.sim_date] += 1
            if pub_date > day0:
                fresh_results[event.sim_date] += 1

    rows: List[Dict[str, object]] = []
    for d in all_days:
        total = int(total_results.get(d, 0))
        fresh = int(fresh_results.get(d, 0))
        frac = (fresh / total) if total > 0 else None
        rows.append(
            {
                "sim_date": d.isoformat(),
                "day_index": (d - day0).days,
                "search_prompts": int(search_prompts.get(d, 0)),
                "search_prompts_no_results": int(empty_search_prompts.get(d, 0)),
                "results_total": total,
                "results_after_day0": fresh,
                "results_on_or_before_day0": total - fresh,
                "fresh_fraction": frac,
            }
        )
    return rows


def save_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "sim_date",
        "day_index",
        "search_prompts",
        "search_prompts_no_results",
        "results_total",
        "results_after_day0",
        "results_on_or_before_day0",
        "fresh_fraction",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_ema(values: Sequence[float], span: int) -> Tuple[List[float], float]:
    if span < 1:
        raise ValueError("--ema-span must be >= 1")

    alpha = 2.0 / (span + 1.0)
    ema_vals: List[float] = []
    prev: Optional[float] = None
    for v in values:
        if prev is None:
            cur = v
        else:
            cur = alpha * v + (1.0 - alpha) * prev
        ema_vals.append(cur)
        prev = cur
    return ema_vals, alpha


def save_plot(
    rows: Sequence[Dict[str, object]],
    day0: date,
    out_path: Path,
    ema_span: int,
    raw_alpha: float,
    sim_name: str,
) -> float:
    x: List[int] = []
    y: List[float] = []

    for row in rows:
        frac = row["fresh_fraction"]
        if frac is None:
            continue
        x.append(int(row["day_index"]))
        y.append(float(frac))

    plt.figure(figsize=(11, 6))
    if x:
        ema_y, ema_alpha = compute_ema(y, span=ema_span)
        plt.plot(
            x,
            y,
            marker="o",
            linewidth=1.2,
            alpha=raw_alpha,
            color="C0",
            label="Raw daily fresh fraction",
            zorder=1,
        )
        plt.plot(
            x,
            ema_y,
            linewidth=2.5,
            alpha=1.0,
            color="C0",
            label=f"EMA (span={ema_span}, alpha={ema_alpha:.3f})",
            zorder=2,
        )
    else:
        ema_alpha = 2.0 / (ema_span + 1.0)
        plt.text(
            0.5,
            0.5,
            "No search result publication dates found",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.ylim(-0.02, 1.02)
    plt.xlabel(f"Simulation day index (day 0 = {day0.isoformat()})")
    plt.ylabel("Fraction of search results published after day 0")
    plt.title(f"Fresh Search Result Fraction by Simulation Day\n{sim_name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return ema_alpha


def main() -> None:
    args = parse_args()
    if args.ema_span < 1:
        raise ValueError("--ema-span must be >= 1")
    if args.raw_alpha < 0 or args.raw_alpha > 1:
        raise ValueError("--raw-alpha must be in [0, 1]")

    input_path = Path(args.input_path).expanduser()
    model_raw_files = discover_model_raw_files(input_path)
    sim_name = infer_sim_name(input_path=input_path, model_raw_files=model_raw_files)

    all_sim_dates: List[date] = []
    all_events: List[SearchEvent] = []
    bad_json_total = 0

    for _, path in model_raw_files:
        sim_dates, events, bad_json = parse_search_events(path)
        all_sim_dates.extend(sim_dates)
        all_events.extend(events)
        bad_json_total += bad_json

    if not all_sim_dates:
        raise ValueError("No valid sim_date entries found in provided logs.")

    if args.day0 is not None:
        day0 = parse_yyyy_mm_dd(args.day0)
    else:
        day0 = min(all_sim_dates)

    rows = aggregate_daily(sim_dates=all_sim_dates, events=all_events, day0=day0)

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else infer_default_output_dir()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback_dir = infer_fallback_output_dir(input_path)
        fallback_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Warning: cannot write to '{output_dir}'. "
            f"Using fallback output dir '{fallback_dir}'."
        )
        output_dir = fallback_dir

    plot_name = args.plot_name if args.plot_name else f"freshsearchfrac_{sim_name}.png"
    csv_name = args.csv_name if args.csv_name else f"freshsearchfrac_daily_{sim_name}.csv"
    csv_path = output_dir / csv_name
    plot_path = output_dir / plot_name
    save_csv(rows, csv_path)
    ema_alpha = save_plot(
        rows=rows,
        day0=day0,
        out_path=plot_path,
        ema_span=args.ema_span,
        raw_alpha=args.raw_alpha,
        sim_name=sim_name,
    )

    days_with_search_results = sum(1 for r in rows if int(r["results_total"]) > 0)
    total_results = sum(int(r["results_total"]) for r in rows)
    total_fresh = sum(int(r["results_after_day0"]) for r in rows)
    overall_frac = (total_fresh / total_results) if total_results > 0 else None

    print(f"Analyzed {len(model_raw_files)} model_raw*.jsonl file(s).")
    print(f"Day 0: {day0.isoformat()}")
    print(f"Days in logs: {len(rows)}")
    print(f"Days with >=1 parsed search result: {days_with_search_results}")
    print(f"Total parsed search results: {total_results}")
    if overall_frac is None:
        print("Overall fresh fraction: n/a (no parsed publication dates)")
    else:
        print(f"Overall fresh fraction: {overall_frac:.4f}")
    if bad_json_total > 0:
        print(f"Warning: skipped {bad_json_total} malformed JSON line(s).")
    print(
        "EMA config: "
        f"span={args.ema_span}, alpha={ema_alpha:.4f}, "
        "equation=EMA_t = alpha * x_t + (1-alpha) * EMA_(t-1), EMA_0 = x_0"
    )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
