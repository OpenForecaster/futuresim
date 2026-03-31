# Loop And State

## High-Level Lifecycle

1. The environment selects active questions for the current wakeup date and writes `market.csv`.
2. Agents act against the same frozen market snapshot for that wakeup session.
3. `BasicAgent`-style scaffolds can inspect `df`, use tools, and submit forecasts.
4. Predictions are appended to `PredictionHistory`; later updates become the current forecast without deleting prior ones.
5. End-of-session logic updates aggregates, logs metrics, resolves questions when due, and scaffold-owned session logic may update memory or agent logs.

With `timegap_days > 1`, a single session represents one wakeup that covers the interval from `current_date` through `min(current_date + timegap_days - 1, end_date)` for active-question metrics.

## Agent-Visible DataFrame

Agents work with a pandas DataFrame built from `market.csv`.

| Column | Meaning |
| --- | --- |
| `qid` | unique question id |
| `title` | question text |
| `background` | background/context |
| `resolution_criteria` | resolution instructions |
| `answer_type` | `yes/no`, `string`, `numeric`, etc. |
| `resolution_date` | resolution date |
| `is_resolved` | whether the question has resolved |
| `ground_truth` | resolved answer when available |
| `market_aggregate` | JSON-encoded aggregate forecast |
| `num_predictions` | total predictions recorded so far |
| `options` | JSON-encoded explicit options when present |
| `my_prediction` | agent-specific current forecast added by `DfInterface` |
| `my_prediction_date` | date of the agent's last forecast |

`my_prediction` and `my_prediction_date` are derived at load time and are not stored in the CSV itself.

## Shared Artifacts

- `market.csv`: market snapshot for the current wakeup session
- `actions.jsonl`: central event log for predictions and resolutions
- `daily_metrics.csv`: cumulative metrics by agent, one row per wakeup session, plus submission-count / submission-shift columns for that session
- `test_daily_metrics.csv`: the same metrics restricted to questions with `source_split == "test"`
- `agents/<agent_id>/model_outputs.jsonl`: cleaned model outputs written by the agent scaffold
- `agents/<agent_id>/model_raw_warmup.jsonl`: warmup raw logs, sorted by qid and storing only per-turn input deltas
- `agents/<agent_id>/model_raw_daily.jsonl`: post-warmup raw logs storing only per-turn input deltas

## Agent-Visible Cadence State

The forecast interface exposes:

- `timegap_days`: configured wakeup spacing
- `last_active_date`: previous wakeup date, if any
- `next_active_date`: next scheduled wakeup date, if any

Scaffolds can use these fields to explain cadence if they want. Warmup prompts should stay focused on the current question rather than adding cadence instructions.
