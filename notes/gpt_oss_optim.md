# GPT-OSS 20B AllQ Optimization Log

Date: 2026-02-12

## Objective
Optimize GPT-OSS 20B AllQ scaffold/config for one-day performance.

Required evaluation setup:
- Simulation day: `2025-04-24` only.
- Resolution window: `2025-05-01` to `2025-08-23`.
- Expected questions: 302.
- Priorities: maximize `avg_brier`, `accuracy`, and `total_predictions`.

## Validation
Question window check:
- `total_count=302`
- `active_count=302`
- `date_range=(2025-05-01, 2025-08-23)`

Command used:
```bash
python - <<'PY'
from datetime import date
from environment.data_loader import QuestionPool
qp = QuestionPool(
    dataset='openforesight',
    dataset_path='/is/cluster/fast/sgoel/forecasting/qs/OpenForesight/data/',
    dataset_cache='/is/cluster/fast/sgoel/forecasting/qs/cache',
    split='test',
    resolution_start=date.fromisoformat('2025-05-01'),
    resolution_end=date.fromisoformat('2025-08-23'),
    min_forecasters=10,
    resolved_only=True,
)
print(qp.total_count, qp.active_count, qp.get_date_range())
PY
```

## Code/Scaffold Changes
1. Added optimized scaffold package: `agents/gpt_oss_allq_optim/`
- `GPTOSSAllQOptimAgentV1`: one-search-budget warmup, strict submit enforcement, rescue submit pass.
- `GPTOSSAllQOptimAgentV2`: submit-only warmup variant for higher coverage/throughput.
- Added outcome sanitization: clip to `[0,1]`, enforce non-empty outcomes, normalize if sum > 1.

2. Wired new scaffold names in `scripts/test_basic_agent.py`
- `gpt_oss_allq_optim_v1`
- `gpt_oss_allq_optim_v2`

3. Added configs in `configs/gpt_oss_optim/`
- `allq_gptoss_optim_v1_a.yaml`
- `allq_gptoss_optim_v1_b.yaml`
- `allq_gptoss_optim_v2_a.yaml`
- `allq_gptoss_optim_v2_b.yaml`

4. Submission tooling robustness (`mpi_scripts/run_sim/submit_sim.py`)
- Fallback from `htcondor2` to CLI-based submission when Python HTCondor bindings are unavailable.
- Use `condor_submit_bid <bid>` in fallback path.
- Fix cluster-id regex parsing.

## Initial Submitted Experiments
Primary running set (cluster IDs):
- `16845555` -> `v1_a_r00`
- `16845556` -> `v1_b_r00`
- `16845557` -> `v2_b_r00`
- `16845558` -> `v2_a_r00`

Expected output directories:
- `/is/cluster/fast/sgoel/forecasting/current_sim/allq_gptoss_optim/v1_a_r00/26-02-12-01-47-47`
- `/is/cluster/fast/sgoel/forecasting/current_sim/allq_gptoss_optim/v1_b_r00/26-02-12-01-47-48`
- `/is/cluster/fast/sgoel/forecasting/current_sim/allq_gptoss_optim/v2_b_r00/26-02-12-01-47-49`
- `/is/cluster/fast/sgoel/forecasting/current_sim/allq_gptoss_optim/v2_a_r00/26-02-12-01-47-49`

## Results
(Use only first data row from each `daily_metrics.csv`.)

| config | cluster_id | output_dir | day1_date | agent_id | avg_brier | accuracy | total_predictions | status |
|---|---:|---|---|---|---:|---:|---:|---|
| v1_a_r00 | 16845555 | /is/cluster/fast/sgoel/forecasting/current_sim/allq_gptoss_optim/v1_a_r00/26-02-12-01-47-47 | 2025-04-24 | gpt_oss_allq_optim_v1_gpt-oss-20b_001 | -0.0455 | 26.83 | 287 | complete |
| v1_b_r00 | 16845556 | /is/cluster/fast/sgoel/forecasting/current_sim/allq_gptoss_optim/v1_b_r00/26-02-12-01-47-48 | 2025-04-24 | gpt_oss_allq_optim_v1_gpt-oss-20b_001 | -0.0672 | 25.42 | 299 | complete |
| v2_b_r00 | 16845557 | /is/cluster/fast/sgoel/forecasting/current_sim/allq_gptoss_optim/v2_b_r00/26-02-12-01-47-49 | 2025-04-24 | gpt_oss_allq_optim_v2_gpt-oss-20b_001 | -0.1431 | 20.53 | 302 | complete |
| v2_a_r00 | 16845558 | /is/cluster/fast/sgoel/forecasting/current_sim/allq_gptoss_optim/v2_a_r00/26-02-12-01-47-49 | 2025-04-24 | gpt_oss_allq_optim_v2_gpt-oss-20b_001 | -0.1140 | 22.71 | 295 | complete |
| v1_c_r00 | 16845565 | /is/cluster/fast/sgoel/forecasting/current_sim/allq_gptoss_optim/v1_c_r00/26-02-12-01-59-09 | - | - | - | - | - | queued/running |
| v1_c_r01 | 16845566 | /is/cluster/fast/sgoel/forecasting/current_sim/allq_gptoss_optim/v1_c_r01/26-02-12-01-59-10 | - | - | - | - | - | queued/running |
| v1_a_fix_r00 | 16845567 | /is/cluster/fast/sgoel/forecasting/current_sim/allq_gptoss_optim/v1_a_fix_r00/26-02-12-01-59-32 | - | - | - | - | - | queued/running |

## Notes
- Best `avg_brier` and best `accuracy` in wave-1: `v1_a_r00`.
- Best `total_predictions` in wave-1: `v2_b_r00` (302), but much worse brier/accuracy.
- Warmup with search (`v1`) currently dominates submit-only (`v2`) on brier/accuracy in this setup.
- Added robustness patch in `agents/gpt_oss_allq_optim/agent.py` to avoid losing an entire question on per-call failures and still attempt rescue submit.
- Current optimizer pass is focused on warmup quality/coverage because this setup runs only one simulation day.

## Latest Standings (Expanded)
Additional discovered runs (same one-day setup):
- `v1_a_rep2_r00`: `avg_brier=-0.0448`, `accuracy=26.76`, `predictions=284`
- `v1_b_rep2_r00`: `avg_brier=-0.0428`, `accuracy=26.87`, `predictions=294` (current best)
- `v1_a_rep3_r00`: `avg_brier=-0.0846`, `accuracy=23.78`, `predictions=286`
- `v2_a_rep2_r00`: `avg_brier=-0.1023`, `accuracy=22.90`, `predictions=297`
- `v2_b_rep2_r00`: `avg_brier=-0.1242`, `accuracy=23.26`, `predictions=301`
- `v1_c_r00`: `avg_brier=-0.0472`, `accuracy=27.18`, `predictions=287`

## Overnight Automation
Background 8-hour loop launched:
- Script: `scripts/run_gpt_oss_overnight.sh`
- PID at launch: `764568`
- Live log: `notes/gpt_oss_optim_live.log`
- Nohup output: `/tmp/gpt_oss_overnight_nohup.out`

The loop does:
- Poll active `allq_gptoss_optim` jobs.
- Keep filling up to 8 concurrent jobs (16 GPUs total) from a queued config list.
- Auto-detect newly completed `daily_metrics.csv` rows and append them to the live log.
- Print rolling best-so-far (`avg_brier`, `accuracy`, `predictions`) snapshots.

## New Baseline Wave (Running)
Submitted baseline `allQ` variants (intended to beat custom scaffold):
- `base_a_r00` (cluster `16845568`)
- `base_b_r00` (cluster `16845569`)
- `base_c_r00` (cluster `16845570`)
- `base_d_r00` (cluster `16845571`)
- `base_e_r00` (cluster `16845572`)
