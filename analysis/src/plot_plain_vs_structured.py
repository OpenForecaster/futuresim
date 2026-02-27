"""Compare plain-text memory vs structured (YAML) memory simulation runs.

Plots forecasting metrics side-by-side, plus a memory-specific panel showing
entry counts and character usage over time for the structured run.
"""

import pandas as pd
import argparse
import os
import sys
import glob
from datetime import date as dt_date

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import yaml
except ImportError as e:
    print(f"Error: missing dependency — {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare plain-text vs structured memory simulation runs."
    )
    parser.add_argument(
        "--plain_dir", type=str, required=True,
        help="Path to plain-text memory run directory",
    )
    parser.add_argument(
        "--structured_dir", type=str, required=True,
        help="Path to structured memory run directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Directory to save plots (default: current dir)",
    )
    parser.add_argument("--plain_label", type=str, default="Plain Memory")
    parser.add_argument("--structured_label", type=str, default="Structured Memory")
    return parser.parse_args()


def load_metrics(directory):
    csv_path = os.path.join(directory, "daily_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date")
    return df


def load_memory_stats(run_dir, ext, parser_fn):
    """Walk agent memory dirs and collect per-day stats.

    Returns a DataFrame with columns: date, entry_count, total_chars.
    """
    agents_dir = os.path.join(run_dir, "agents")
    if not os.path.isdir(agents_dir):
        return pd.DataFrame(columns=["date", "entry_count", "total_chars"])

    rows = []
    for agent_dir in sorted(glob.glob(os.path.join(agents_dir, "*"))):
        mem_dir = os.path.join(agent_dir, "memory")
        if not os.path.isdir(mem_dir):
            continue
        for fpath in sorted(glob.glob(os.path.join(mem_dir, f"*.{ext}"))):
            fname = os.path.basename(fpath)
            try:
                file_date = dt_date.fromisoformat(os.path.splitext(fname)[0])
            except ValueError:
                continue
            entry_count, total_chars = parser_fn(fpath)
            rows.append({
                "date": pd.Timestamp(file_date),
                "entry_count": entry_count,
                "total_chars": total_chars,
            })
    return pd.DataFrame(rows)


def parse_yaml_memory(fpath):
    """Parse a structured YAML memory file -> (entry_count, total_chars)."""
    try:
        with open(fpath) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            return 0, 0
        total = sum(len(str(d.get("content", ""))) for d in data)
        return len(data), total
    except Exception:
        return 0, 0


def parse_txt_memory(fpath):
    """Parse a plain-text memory file -> (paragraph_count, total_chars)."""
    try:
        text = open(fpath).read().strip()
        if not text:
            return 0, 0
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return len(paragraphs) if paragraphs else 1, len(text)
    except Exception:
        return 0, 0


def count_yaml_type_distribution(run_dir):
    """Count entry types across all YAML memory files (latest per agent).

    Returns a dict like {"reasoning": 42, "calibration": 15, ...}.
    """
    agents_dir = os.path.join(run_dir, "agents")
    if not os.path.isdir(agents_dir):
        return {}

    type_counts = {}
    for agent_dir in sorted(glob.glob(os.path.join(agents_dir, "*"))):
        mem_dir = os.path.join(agent_dir, "memory")
        yamls = sorted(glob.glob(os.path.join(mem_dir, "*.yaml")))
        if not yamls:
            continue
        # Use the latest YAML file per agent
        try:
            with open(yamls[-1]) as f:
                data = yaml.safe_load(f)
            if isinstance(data, list):
                for entry in data:
                    t = entry.get("type", "unknown")
                    type_counts[t] = type_counts.get(t, 0) + 1
        except Exception:
            continue
    return type_counts


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df_plain = load_metrics(args.plain_dir)
    df_struct = load_metrics(args.structured_dir)

    # Align on common date range
    common_start = max(df_plain["date"].min(), df_struct["date"].min())
    common_end = min(df_plain["date"].max(), df_struct["date"].max())
    df_plain = df_plain[(df_plain["date"] >= common_start) & (df_plain["date"] <= common_end)]
    df_struct = df_struct[(df_struct["date"] >= common_start) & (df_struct["date"] <= common_end)]

    print(f"Common date range: {common_start.date()} to {common_end.date()} "
          f"({len(df_plain)} / {len(df_struct)} days)")

    # Load memory stats
    mem_plain = load_memory_stats(args.plain_dir, "txt", parse_txt_memory)
    mem_struct = load_memory_stats(args.structured_dir, "yaml", parse_yaml_memory)
    mem_plain = mem_plain[(mem_plain["date"] >= common_start) & (mem_plain["date"] <= common_end)]
    mem_struct = mem_struct[(mem_struct["date"] >= common_start) & (mem_struct["date"] <= common_end)]

    # Entry type distribution for structured run
    type_dist = count_yaml_type_distribution(args.structured_dir)

    # ── Plot ────────────────────────────────────────────────────────────
    forecast_metrics = [
        ("avg_brier", "Average Brier Skill Score", "higher is better"),
        ("accuracy", "Accuracy (%)", "higher is better"),
        ("exp_acc", "Expected Accuracy", "higher is better"),
        ("total_predictions", "Total Predictions", ""),
    ]
    n_forecast = len(forecast_metrics)
    # Extra panels: entry count over time, total chars over time, type distribution
    has_mem_stats = not mem_struct.empty
    n_mem_panels = 3 if (has_mem_stats and type_dist) else (2 if has_mem_stats else 0)
    n_rows = n_forecast + n_mem_panels

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.8 * n_rows))
    if n_rows == 1:
        axes = [axes]

    colors = {"plain": "#FF5722", "struct": "#2196F3"}

    # ── Forecast metric panels ──────────────────────────────────────────
    for ax, (col, title, direction) in zip(axes[:n_forecast], forecast_metrics):
        ax.plot(df_plain["date"], df_plain[col], label=args.plain_label,
                color=colors["plain"], linewidth=1.5, alpha=0.9)
        ax.plot(df_struct["date"], df_struct[col], label=args.structured_label,
                color=colors["struct"], linewidth=1.5, alpha=0.9)

        subtitle = f"  ({direction})" if direction else ""
        ax.set_title(f"{title}{subtitle}", fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel(col.replace("_", " ").title(), fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)

        # Final-value annotations
        for df, label, color in [
            (df_plain, args.plain_label, colors["plain"]),
            (df_struct, args.structured_label, colors["struct"]),
        ]:
            if df.empty:
                continue
            last = df.iloc[-1]
            val_str = f"{last[col]:.4f}" if isinstance(last[col], float) else f"{last[col]}"
            ax.annotate(val_str, xy=(last["date"], last[col]),
                        fontsize=8, color=color, fontweight="bold",
                        xytext=(5, 0), textcoords="offset points", va="center")

    # ── Memory stat panels ──────────────────────────────────────────────
    if has_mem_stats:
        # Panel: entry count over time
        ax_count = axes[n_forecast]
        if not mem_plain.empty:
            ax_count.plot(mem_plain["date"], mem_plain["entry_count"],
                          label=f"{args.plain_label} (paragraphs)",
                          color=colors["plain"], linewidth=1.5, alpha=0.9)
        if not mem_struct.empty:
            ax_count.plot(mem_struct["date"], mem_struct["entry_count"],
                          label=f"{args.structured_label} (entries)",
                          color=colors["struct"], linewidth=1.5, alpha=0.9)
        ax_count.set_title("Memory Entry Count Over Time", fontsize=12,
                           fontweight="bold", loc="left")
        ax_count.set_ylabel("Entries", fontsize=10)
        ax_count.grid(True, linestyle="--", alpha=0.4)
        ax_count.legend(loc="upper right", fontsize=9)

        # Panel: total chars over time
        ax_chars = axes[n_forecast + 1]
        if not mem_plain.empty:
            ax_chars.plot(mem_plain["date"], mem_plain["total_chars"],
                          label=args.plain_label, color=colors["plain"],
                          linewidth=1.5, alpha=0.9)
        if not mem_struct.empty:
            ax_chars.plot(mem_struct["date"], mem_struct["total_chars"],
                          label=args.structured_label, color=colors["struct"],
                          linewidth=1.5, alpha=0.9)
        ax_chars.set_title("Memory Size Over Time (content chars)", fontsize=12,
                           fontweight="bold", loc="left")
        ax_chars.set_ylabel("Characters", fontsize=10)
        ax_chars.grid(True, linestyle="--", alpha=0.4)
        ax_chars.legend(loc="upper right", fontsize=9)

        # Panel: type distribution (bar chart, structured only)
        if type_dist and n_mem_panels == 3:
            ax_types = axes[n_forecast + 2]
            types = sorted(type_dist.keys())
            counts = [type_dist[t] for t in types]
            bar_colors = {
                "reasoning": "#42A5F5", "calibration": "#66BB6A",
                "insight": "#FFA726", "fact": "#AB47BC",
            }
            ax_types.bar(types, counts,
                         color=[bar_colors.get(t, "#90A4AE") for t in types],
                         edgecolor="white", linewidth=0.8)
            for i, (t, c) in enumerate(zip(types, counts)):
                ax_types.text(i, c + 0.3, str(c), ha="center", fontsize=10,
                              fontweight="bold")
            ax_types.set_title("Entry Type Distribution (latest structured snapshot)",
                               fontsize=12, fontweight="bold", loc="left")
            ax_types.set_ylabel("Count", fontsize=10)
            ax_types.grid(True, linestyle="--", alpha=0.4, axis="y")

    # ── Shared x-axis formatting ────────────────────────────────────────
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    axes[-1].set_xlabel("Date", fontsize=11)

    fig.suptitle("Plain vs Structured Memory Comparison",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, "plain_vs_structured_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")

    # ── Print summary stats ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary (over common date range)")
    print(f"{'='*60}")
    for col, title, _ in forecast_metrics:
        p_mean = df_plain[col].mean()
        s_mean = df_struct[col].mean()
        diff = s_mean - p_mean
        pct = (diff / abs(p_mean) * 100) if p_mean != 0 else 0
        print(f"  {title:30s}  plain={p_mean:.4f}  struct={s_mean:.4f}  "
              f"diff={diff:+.4f} ({pct:+.1f}%)")

    if has_mem_stats and not mem_struct.empty:
        print(f"\nStructured memory (latest snapshot):")
        print(f"  Entries: {mem_struct.iloc[-1]['entry_count']:.0f}")
        print(f"  Content chars: {mem_struct.iloc[-1]['total_chars']:.0f}")
        if type_dist:
            print(f"  Type distribution: {type_dist}")


if __name__ == "__main__":
    main()
