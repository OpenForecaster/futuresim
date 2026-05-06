"""Memory vs no-memory curves for Qwen and DeepSeek.

Compact, presentation-style figure (small canvas, thick lines, large fonts)
comparing NeurIPS no-memory runs against NeurIPS our-harness memory runs.

The no-memory run starts its metrics at 2025-12-25, so we prepend the memory
run's 2025-12-24 row to each no-memory curve as the shared day0 anchor.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

from plot_config import color_for_label, style_axes  # noqa: F401  (style side-effects)

DEFAULT_NEURIPS_DIR = Path(
    os.getenv("FSIM_OUTPUT_BASE", "/fast/sgoel/forecasting/current_sim")
) / "neurips_runs"
DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parents[1] / "plots" / "mem_vs_nomem"
)


@dataclass(frozen=True)
class PairSpec:
    label: str
    nomem_rel: Path
    mem_rel: Path


PAIRS: tuple[PairSpec, ...] = (
    PairSpec(
        label="deepseek-v4-pro",
        nomem_rel=Path("no_memory/ds-v4-pro/26-05-05-04-58-01"),
        mem_rel=Path("our_harness/ds-v4-pro/26-05-05-02-43-54"),
    ),
    PairSpec(
        label="qwen-3.6-plus",
        nomem_rel=Path("no_memory/qwen-3.6-plus/26-05-05-14-12-52"),
        mem_rel=Path("our_harness/qwen-3.6-plus/26-05-05-03-46-54"),
    ),
)

METRIC_LABELS = {
    "accuracy": "Accuracy (%)",
    "avg_brier": "Brier Skill Score (↑)",
}

LINEWIDTH = 6.0
LEGEND_LINEWIDTH = 4.0
BASELINE_COLOR = "#9A9A9A"
BASELINE_LINESTYLE = (0, (18, 10))
BASELINE_LINEWIDTH = 1.2
WITH_MEM_LINESTYLE = "-"
NO_MEM_LINESTYLE = ":"


def _latest_ts(run_dir: Path) -> Path:
    candidates = sorted(p for p in run_dir.iterdir() if p.is_dir())
    if not candidates:
        raise SystemExit(f"No timestamp dirs under {run_dir}")
    return candidates[-1]


def _load_metrics(ts_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(ts_dir / "daily_metrics.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _prepare_nomem_with_day0(am: pd.DataFrame, nomem: pd.DataFrame) -> pd.DataFrame:
    day0_date = am["date"].iloc[0]
    if nomem["date"].iloc[0] == day0_date:
        return nomem
    day0_row = am.iloc[[0]].copy()
    day0_row["agent_id"] = nomem["agent_id"].iloc[0]
    return pd.concat([day0_row, nomem], ignore_index=True)


def _plot_metric(metric: str, records: list[tuple[PairSpec, pd.DataFrame, pd.DataFrame]],
                 out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    style_axes(ax)

    for spec, am, nomem in records:
        color = color_for_label(spec.label)
        ax.plot(
            am["date"],
            am[metric],
            color=color,
            linewidth=LINEWIDTH,
            linestyle=WITH_MEM_LINESTYLE,
            solid_capstyle="round",
        )
        ax.plot(
            nomem["date"],
            nomem[metric],
            color=color,
            linewidth=LINEWIDTH,
            linestyle=NO_MEM_LINESTYLE,
            dash_capstyle="round",
        )

    if metric == "avg_brier":
        ax.axhline(
            0.0,
            color=BASELINE_COLOR,
            linestyle=BASELINE_LINESTYLE,
            linewidth=BASELINE_LINEWIDTH,
            zorder=1,
        )

    ax.set_ylabel(METRIC_LABELS[metric], fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=20, length=5)
    ax.minorticks_off()

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    tick_dates = pd.to_datetime(
        ["2026-01-01", "2026-02-01", "2026-03-01"]
    )
    ax.set_xticks(tick_dates)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=15)
    last_date = min(
        min(am["date"].iloc[-1], nomem["date"].iloc[-1])
        for _spec, am, nomem in records
    )
    ax.set_xlim(pd.Timestamp("2025-12-24"), pd.Timestamp(last_date))
    ax.margins(x=0.02)

    model_handles = [
        mlines.Line2D(
            [],
            [],
            color=color_for_label(spec.label),
            linewidth=LEGEND_LINEWIDTH,
            linestyle="-",
            label=spec.label,
        )
        for spec, _am, _nomem in records
    ]
    style_handles = [
        mlines.Line2D(
            [],
            [],
            color="#333333",
            linewidth=LEGEND_LINEWIDTH,
            linestyle=WITH_MEM_LINESTYLE,
            label="with memory",
        ),
        mlines.Line2D(
            [],
            [],
            color="#333333",
            linewidth=LEGEND_LINEWIDTH,
            linestyle=NO_MEM_LINESTYLE,
            label="without memory",
        ),
    ]
    ax.legend(
        handles=model_handles + style_handles,
        loc="upper center",
        fontsize=15,
        frameon=False,
        borderaxespad=0.4,
        handlelength=2.4,
        bbox_to_anchor=(0.5, 1.24),
        ncol=2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurips_dir", type=Path, default=DEFAULT_NEURIPS_DIR)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--skip_day0", action="store_true", default=False)
    args = ap.parse_args()

    records = []
    for spec in PAIRS:
        nomem_ts = args.neurips_dir / spec.nomem_rel
        am_ts = args.neurips_dir / spec.mem_rel
        print(f"{spec.label}:")
        print(f"  no-memory : {nomem_ts}")
        print(f"  with memory: {am_ts}")

        nomem = _prepare_nomem_with_day0(_load_metrics(am_ts), _load_metrics(nomem_ts))
        am = _load_metrics(am_ts)

        if args.skip_day0:
            nomem = nomem.iloc[1:].reset_index(drop=True)
            am = am.iloc[1:].reset_index(drop=True)

        records.append((spec, am, nomem))

    for metric in ("accuracy", "avg_brier"):
        _plot_metric(metric, records, args.output_dir / f"{metric}.png")


if __name__ == "__main__":
    main()
