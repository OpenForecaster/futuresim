"""
Statistical testing for memory vs no-memory forecasting runs.

Extracts per-question resolution scores from actions.jsonl and performs
paired statistical tests to determine if memory provides a significant advantage.

Usage:
    # Single pair of runs:
    python analysis/src/stat_test_memory.py \
        --mem_dir /path/to/mem/run --nomem_dir /path/to/nomem/run

    # Multi-run (pools questions across runs):
    python analysis/src/stat_test_memory.py \
        --mem_parent /path/to/mem_parent --nomem_parent /path/to/nomem_parent \
        --max_runs 3
"""

import argparse
import json
import os
import sys
import glob
from collections import defaultdict
from datetime import datetime

import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:
    print("Error: scipy is required. pip install scipy")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Statistical testing: memory vs no-memory forecasting runs."
    )
    # Single-run mode
    parser.add_argument("--mem_dir", type=str, help="Single memory run directory")
    parser.add_argument("--nomem_dir", type=str, help="Single no-memory run directory")
    # Multi-run mode
    parser.add_argument("--mem_parent", type=str, help="Parent dir with memory run subdirectories")
    parser.add_argument("--nomem_parent", type=str, help="Parent dir with no-memory run subdirectories")
    parser.add_argument("--max_runs", type=int, default=None,
                        help="Use only the first N runs (sorted ascending by date). Default: all.")
    # Output
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Save per-question deltas to CSV")
    parser.add_argument("--n_bootstrap", type=int, default=10000,
                        help="Number of bootstrap resamples (default: 10000)")
    # Scoring orientation
    parser.add_argument("--scorer", type=str, default="brier_skill",
                        choices=["brier_skill", "binary_brier"],
                        help="Scorer type: brier_skill (higher=better) or binary_brier (lower=better)")
    return parser.parse_args()


def extract_resolutions(run_dir):
    """Extract per-question raw_brier scores from a run's actions.jsonl.

    Returns:
        dict: {question_id: {agent_id: raw_brier_score}}
    """
    actions_path = os.path.join(run_dir, "actions.jsonl")
    if not os.path.isfile(actions_path):
        print(f"  Warning: {actions_path} not found, skipping")
        return {}

    resolutions = {}
    with open(actions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") != "resolution":
                continue
            qid = record["question_id"]
            raw_brier = record.get("raw_brier", {})
            sim_date = record.get("sim_date", "")
            resolutions[qid] = {
                "scores": raw_brier,
                "sim_date": sim_date,
                "ground_truth": record.get("ground_truth", ""),
            }
    return resolutions


def get_sorted_subdirs(parent_dir, max_runs=None):
    """Get subdirectories sorted ascending by name (date-based names)."""
    subdirs = sorted(glob.glob(os.path.join(parent_dir, "*")))
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    if max_runs is not None:
        subdirs = subdirs[:max_runs]
    return subdirs


def load_multi_run_resolutions(parent_dir, max_runs=None):
    """Load resolutions from all runs under a parent directory.

    Returns:
        dict: {question_id: [list of raw_brier scores across runs]}
        Also returns per-run details for reporting.
    """
    subdirs = get_sorted_subdirs(parent_dir, max_runs)
    if not subdirs:
        print(f"Error: No subdirectories found in {parent_dir}")
        sys.exit(1)

    print(f"  Loading {len(subdirs)} runs from {parent_dir}")

    # {qid: [score1, score2, ...]} pooled across runs
    pooled = defaultdict(list)
    # {qid: {run_name: score}} for per-run breakdown
    per_run = defaultdict(dict)
    # {qid: sim_date} from any run (for stratification)
    resolution_dates = {}

    for subdir in subdirs:
        run_name = os.path.basename(subdir)
        resolutions = extract_resolutions(subdir)
        n_resolved = len(resolutions)
        print(f"    {run_name}: {n_resolved} resolved questions")

        for qid, data in resolutions.items():
            # Take the first agent's score (single-agent setup)
            scores = data["scores"]
            if scores:
                agent_id = list(scores.keys())[0]
                score = scores[agent_id]
                pooled[qid].append(score)
                per_run[qid][run_name] = score
                if qid not in resolution_dates:
                    resolution_dates[qid] = data["sim_date"]

    return dict(pooled), dict(per_run), resolution_dates


def compute_deltas(mem_pooled, nomem_pooled, higher_is_better=True):
    """Compute per-question deltas: mean(mem) - mean(nomem) for each question.

    Args:
        higher_is_better: If True, positive delta = memory helped.
                          If False (binary_brier), flip sign.

    Returns:
        dict: {qid: delta} where positive = memory advantage
        dict: {qid: (mean_mem, mean_nomem)} for reporting
    """
    common_qids = set(mem_pooled.keys()) & set(nomem_pooled.keys())
    if not common_qids:
        print("Error: No common resolved questions between mem and nomem runs.")
        sys.exit(1)

    deltas = {}
    details = {}
    for qid in sorted(common_qids):
        mean_mem = np.mean(mem_pooled[qid])
        mean_nomem = np.mean(nomem_pooled[qid])
        delta = mean_mem - mean_nomem
        if not higher_is_better:
            delta = -delta  # Flip so positive = memory advantage
        deltas[qid] = delta
        details[qid] = (mean_mem, mean_nomem)

    return deltas, details


def bootstrap_ci(deltas_array, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    n = len(deltas_array)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(deltas_array, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return lo, hi, boot_means


def cohens_d(deltas_array):
    """Cohen's d for paired differences (one-sample effect size)."""
    if np.std(deltas_array, ddof=1) == 0:
        return float('inf') if np.mean(deltas_array) != 0 else 0.0
    return np.mean(deltas_array) / np.std(deltas_array, ddof=1)


def stratify_by_resolution_date(deltas, resolution_dates):
    """Split questions into early/mid/late thirds by resolution date."""
    dated_qids = []
    for qid in deltas:
        if qid in resolution_dates and resolution_dates[qid]:
            try:
                d = datetime.strptime(resolution_dates[qid], "%Y-%m-%d")
                dated_qids.append((d, qid))
            except ValueError:
                pass

    if len(dated_qids) < 6:
        return None  # Not enough for meaningful stratification

    dated_qids.sort()
    n = len(dated_qids)
    third = n // 3
    strata = {
        "early": [qid for _, qid in dated_qids[:third]],
        "mid": [qid for _, qid in dated_qids[third:2*third]],
        "late": [qid for _, qid in dated_qids[2*third:]],
    }
    return strata


def print_results(deltas, details, resolution_dates, n_bootstrap, higher_is_better):
    """Print comprehensive statistical report."""
    qids = sorted(deltas.keys())
    deltas_array = np.array([deltas[q] for q in qids])
    n = len(deltas_array)

    score_label = "Brier Skill (higher=better)" if higher_is_better else "Binary Brier (lower=better)"

    print("\n" + "=" * 70)
    print("MEMORY vs NO-MEMORY: STATISTICAL TEST RESULTS")
    print("=" * 70)
    print(f"Scoring: {score_label}")
    print(f"Questions resolved in both conditions: {n}")
    print(f"Convention: positive delta = memory advantage")

    # Descriptive stats
    mean_delta = np.mean(deltas_array)
    median_delta = np.median(deltas_array)
    std_delta = np.std(deltas_array, ddof=1)
    se_delta = std_delta / np.sqrt(n)

    print(f"\n--- Descriptive Statistics ---")
    print(f"Mean delta:   {mean_delta:+.6f} (SE = {se_delta:.6f})")
    print(f"Median delta: {median_delta:+.6f}")
    print(f"Std dev:      {std_delta:.6f}")
    print(f"Memory better on {np.sum(deltas_array > 0)}/{n} questions "
          f"({100*np.mean(deltas_array > 0):.1f}%)")
    print(f"Nomem better on  {np.sum(deltas_array < 0)}/{n} questions "
          f"({100*np.mean(deltas_array < 0):.1f}%)")
    print(f"Tied:            {np.sum(deltas_array == 0)}/{n}")

    # Paired t-test
    t_stat, t_pval = scipy_stats.ttest_1samp(deltas_array, 0)
    print(f"\n--- Paired t-test (H0: mean delta = 0) ---")
    print(f"t = {t_stat:.4f}, p = {t_pval:.6f}")
    print(f"{'*** Significant at p<0.05' if t_pval < 0.05 else 'Not significant at p<0.05'}")

    # Wilcoxon signed-rank test
    nonzero = deltas_array[deltas_array != 0]
    if len(nonzero) >= 10:
        w_stat, w_pval = scipy_stats.wilcoxon(nonzero, alternative='two-sided')
        print(f"\n--- Wilcoxon signed-rank test (H0: median delta = 0) ---")
        print(f"W = {w_stat:.4f}, p = {w_pval:.6f}")
        print(f"{'*** Significant at p<0.05' if w_pval < 0.05 else 'Not significant at p<0.05'}")
    else:
        print(f"\n--- Wilcoxon test: skipped (only {len(nonzero)} non-zero deltas, need >= 10) ---")

    # Bootstrap CI
    lo, hi, boot_means = bootstrap_ci(deltas_array, n_bootstrap)
    print(f"\n--- Bootstrap 95% CI (n={n_bootstrap}) ---")
    print(f"Mean delta: {mean_delta:+.6f}  [{lo:+.6f}, {hi:+.6f}]")
    ci_excludes_zero = (lo > 0) or (hi < 0)
    print(f"{'*** CI excludes zero' if ci_excludes_zero else 'CI includes zero'}")

    # Effect size
    d = cohens_d(deltas_array)
    magnitude = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
    print(f"\n--- Effect Size ---")
    print(f"Cohen's d = {d:.4f} ({magnitude})")

    # Top contributing questions
    print(f"\n--- Top 10 Questions by |Delta| ---")
    sorted_qids = sorted(qids, key=lambda q: abs(deltas[q]), reverse=True)
    print(f"{'QID':>8}  {'Delta':>10}  {'Mem Score':>10}  {'Nomem Score':>12}  {'Resolved':>12}  Direction")
    print("-" * 75)
    for qid in sorted_qids[:10]:
        d_val = deltas[qid]
        mem_s, nomem_s = details[qid]
        res_date = resolution_dates.get(qid, "?")
        direction = "MEM+" if d_val > 0 else "NOMEM+"
        print(f"{qid:>8}  {d_val:>+10.4f}  {mem_s:>10.4f}  {nomem_s:>12.4f}  {res_date:>12}  {direction}")

    # Stratified analysis
    strata = stratify_by_resolution_date(deltas, resolution_dates)
    if strata:
        print(f"\n--- Stratified by Resolution Timing ---")
        for stratum_name, stratum_qids in strata.items():
            stratum_deltas = np.array([deltas[q] for q in stratum_qids])
            s_mean = np.mean(stratum_deltas)
            s_n = len(stratum_deltas)
            s_pos = np.sum(stratum_deltas > 0)
            print(f"  {stratum_name:>5} (n={s_n:>3}): mean delta = {s_mean:+.6f}, "
                  f"mem better on {s_pos}/{s_n} ({100*s_pos/s_n:.0f}%)")

    print("\n" + "=" * 70)

    return {
        "n": n,
        "mean_delta": mean_delta,
        "t_pval": t_pval,
        "bootstrap_ci": (lo, hi),
        "cohens_d": d,
    }


def save_csv(deltas, details, resolution_dates, output_path):
    """Save per-question deltas to CSV."""
    with open(output_path, "w") as f:
        f.write("question_id,delta,mem_score,nomem_score,resolution_date\n")
        for qid in sorted(deltas.keys(), key=lambda q: -abs(deltas[q])):
            mem_s, nomem_s = details[qid]
            res_date = resolution_dates.get(qid, "")
            f.write(f"{qid},{deltas[qid]:.6f},{mem_s:.6f},{nomem_s:.6f},{res_date}\n")
    print(f"\nPer-question deltas saved to {output_path}")


def main():
    args = parse_args()
    higher_is_better = (args.scorer == "brier_skill")

    # Determine mode
    if args.mem_parent and args.nomem_parent:
        # Multi-run mode
        print("=== Multi-run pooled analysis ===")
        mem_pooled, mem_per_run, mem_dates = load_multi_run_resolutions(
            args.mem_parent, args.max_runs
        )
        nomem_pooled, nomem_per_run, nomem_dates = load_multi_run_resolutions(
            args.nomem_parent, args.max_runs
        )
        # Merge resolution dates (prefer mem, fallback to nomem)
        resolution_dates = {**nomem_dates, **mem_dates}

    elif args.mem_dir and args.nomem_dir:
        # Single-run mode
        print("=== Single-run pair analysis ===")
        mem_res = extract_resolutions(args.mem_dir)
        nomem_res = extract_resolutions(args.nomem_dir)

        mem_pooled = {}
        nomem_pooled = {}
        resolution_dates = {}
        for qid, data in mem_res.items():
            scores = data["scores"]
            if scores:
                agent_id = list(scores.keys())[0]
                mem_pooled[qid] = [scores[agent_id]]
                resolution_dates[qid] = data["sim_date"]
        for qid, data in nomem_res.items():
            scores = data["scores"]
            if scores:
                agent_id = list(scores.keys())[0]
                nomem_pooled[qid] = [scores[agent_id]]
                if qid not in resolution_dates:
                    resolution_dates[qid] = data["sim_date"]

        print(f"  Mem run: {len(mem_pooled)} resolved questions")
        print(f"  Nomem run: {len(nomem_pooled)} resolved questions")
    else:
        print("Error: Provide either --mem_dir/--nomem_dir or --mem_parent/--nomem_parent")
        sys.exit(1)

    # Compute deltas
    deltas, details = compute_deltas(mem_pooled, nomem_pooled, higher_is_better)

    # Print results
    print_results(deltas, details, resolution_dates, args.n_bootstrap, higher_is_better)

    # Save CSV if requested
    if args.output_csv:
        save_csv(deltas, details, resolution_dates, args.output_csv)


if __name__ == "__main__":
    main()
