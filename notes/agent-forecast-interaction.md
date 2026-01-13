# Agent-Forecast Interaction Design

This document describes how forecasting agents interact with the simulation environment.

---

## Overview

```mermaid
sequenceDiagram
    participant Env as Environment
    participant Agent as BasicAgent
    
    Note over Env: Start of day
    Env->>Agent: Initial prompt with DataFrame info + scoring rules
    
    loop Query Phase (max_queries tries)
        Agent->>Env: Python code on DataFrame
        Env->>Env: Execute via eval() with timeout
        Env->>Agent: Result + "Tries remaining: N"
    end
    
    Note over Agent: Agent outputs "SUBMIT:" with XML forecasts
    
    loop Submit Phase (max_retries)
        alt Parse success
            Env->>Env: Record predictions
        else Parse failure
            Env->>Agent: Error + "Retries remaining: N"
        end
    end
    
    Note over Env: End of day, update aggregates
```

---

## Questions DataFrame Schema

Agents receive access to a pandas DataFrame `df` with all questions:

| Column | Type | Description |
|--------|------|-------------|
| `qid` | str | Unique question ID |
| `title` | str | Question text |
| `background` | str | Context/background |
| `resolution_criteria` | str | How resolved |
| `answer_type` | str | "yes/no", "string", "numeric" |
| `resolution_date` | date | When resolves |
| `is_resolved` | bool | True if resolved |
| `ground_truth` | str/None | Answer (only if resolved) |
| `market_aggregate` | str (JSON) | e.g. '{"Yes": 0.6}' |
| `num_predictions` | int | Total predictions from all agents |
| `my_prediction` | str (JSON)/None | This agent's current prediction |
| `my_prediction_date` | date/None | When agent last predicted |

---

## Scoring Rules

- **Brier Skill Score**: Range [-1, +1], higher = better. Score of 0 = abstainer baseline.
- **Peer score** = 100 × (your_brier - average_of_others). Zero-sum.
- **Single-agent**: Compared against virtual abstainer (score 0) → peer_score = 100 × brier_score

---

## Agent Initial Prompt Template

```
=== FORECASTING SIMULATION - Day {current_date} ===

You are a forecasting agent. Your goal: make accurate probability predictions.

## SCORING RULES
- Brier Skill Score: Range [-1, +1], higher = better. Score of 0 = abstainer baseline.
- Peer score = 100 × (your_brier - average_of_others). Zero-sum with other agents.
- If you don't predict on a question, you're scored as if you predicted the market aggregate.

## AVAILABLE DATA
DataFrame `df` with {n_rows} rows. Columns:
{columns_desc}

Current simulation date available as `today`.

## INTERACTION RULES

QUERY PHASE (max {max_queries} queries):
- Write Python code in ```python ... ``` blocks to explore the DataFrame
- Code has access to: df, pd, today, date, datetime, timedelta

SUBMIT PHASE:
- Say "SUBMIT:" followed by XML forecasts:
<submit>
  <forecast qid="QUESTION_ID">
    <outcome name="Yes" prob="0.7"/>
    <outcome name="No" prob="0.3"/>
  </forecast>
</submit>

Rules: probs sum ≤ 1.0, max 5 outcomes/question
```

---

## Forecast Submission Format (XML)

```xml
<submit>
  <forecast qid="15573">
    <outcome name="Intel" prob="0.55"/>
    <outcome name="Amazon" prob="0.25"/>
    <outcome name="Other" prob="0.15"/>
  </forecast>
  <forecast qid="15580">
    <outcome name="Yes" prob="0.7"/>
    <outcome name="No" prob="0.3"/>
  </forecast>
</submit>
```

---

## Code Execution

> ⚠️ **SAFETY WARNING**: Current implementation uses `eval()` which is NOT SAFE for untrusted code. This is acceptable for testing with controlled agents, but needs proper sandboxing for production. Options to explore later:
> - RestrictedPython
> - AST whitelisting  
> - Subprocess isolation
> - Docker/container execution

Agent code has access to:
- `df` - pandas DataFrame (copy)
- `pd` - pandas module
- `today` - current simulation date
- `date`, `datetime`, `timedelta` - date utilities

Timeout: 5 seconds per query.

---

## Logging Directory Structure

```
/is/cluster/fast/sgoel/forecasting/current_sim/
└── {sim_name}/
    └── {YY-MM-DD-HH-MM-SS}/
        ├── config.json          # All hyperparameters
        └── logs/
            ├── actions.jsonl     # Predictions
            └── model_outputs.jsonl # LLM prompts/responses
```

---

## Running Tests

```bash
# Get GPU session on cluster
condor_submit_bid 25 -i -append request_gpus=1 -append "requirements=TARGET.CUDACapability == 8.0" -append request_memory=40960

# Run test (Dec 25-27, 2024)
python scripts/test_basic_agent.py --sim_name test_run --start_date 2024-12-25 --end_date 2024-12-27

# Run without LLM (test setup)
python scripts/test_basic_agent.py --no_inference
```

---

## Files Overview

| File | Purpose |
|------|---------|
| `agents/basicAgent.py` | BasicAgent with query loop + XML parsing |
| `environment/safe_executor.py` | Query execution with timeout |
| `environment/env.py` | SimulationEnvironment + SimForecastInterface |
| `environment/scoring/__init__.py` | Scoring with single-agent baseline |
| `scripts/test_basic_agent.py` | CLI test script |

---

## Question Counts (Dec 25-31, 2024)

| Date | Resolving |
|------|-----------|
| 2024-12-25 | 27 |
| 2024-12-26 | 33 |
| 2024-12-27 | 41 |
| 2024-12-28 | 38 |
| 2024-12-29 | 27 |
| 2024-12-30 | 48 |
| 2024-12-31 | 169 |
| **Total** | **383** |
