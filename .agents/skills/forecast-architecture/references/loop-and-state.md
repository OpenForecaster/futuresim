# Loop And State

## High-Level Lifecycle

1. The environment selects active questions for the current day and writes `market.csv`.
2. Agents act against the same frozen market snapshot for that day.
3. `BasicAgent`-style scaffolds can inspect `df`, use tools, and submit forecasts.
4. Predictions are appended to `PredictionHistory`; later updates become the current forecast without deleting prior ones.
5. End-of-day logic updates aggregates, logs metrics, resolves questions when due, and may prompt a memory update.

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

- `market.csv`: daily market snapshot for agents
- `actions.jsonl`: central event log for predictions and resolutions
- `daily_metrics.csv`: per-day metrics by agent
- `agents/<agent_id>/model_outputs.jsonl`: cleaned model outputs
- `agents/<agent_id>/model_raw.jsonl`: prompt + raw response log
