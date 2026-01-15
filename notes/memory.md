# Forecasting Simulator: Design Context & Decisions

This document captures the goal of my project, design decisions, and context that isn't obvious from code alone. For future developers/models working on this codebase. If you have better ideas for any of the design choices that better support my project goal, always let me know. I am always open to feedback and criticism for my ideas, as I'm just a researcher exploring the unknown. If you have any questions, or any uncertainty about any design decision whatsoever, please ask me. Its okay to take obvious next steps on your own, but if there are any assumptions I or future collaborators/agents working on this project should know of, please let me know, and also update them here.

---

## Project Goal

Build a **multi-agent forecasting simulator** where LLM agents:
1. Read news/context articles
2. Make probabilistic predictions on free-form questions
3. Are scored fairly against each other
4. Learn behaviors that transfer to real prediction markets

The ultimate goal is **training agents whose forecasting behaviors generalize to real-world prediction markets**.

---

## How to run

We have a uv environment called fsim which we are using for this project. 

For the /is/cluster, to get GPU (say to debug) you will have to get an interactive job. One way to do that (A100) is:
```
(miniforge3)sgoel@login3:~/forecast-sim$ condor_submit_bid 25 -i -append request_gpus=1 -append "requirements=TARGET.CUDACapability == 8.0" -append request_memory=40960
```
Local open-weight models are stored in /fast/rolmedo/models on the /is/cluster. For debugging, its good to use qwen3-4b-it-2507. The OpenForesight dataset of forecasting questions is at /fast/sgoel/forecasting/qs/OpenForesight/data

Test script:
```bash
python scripts/test_basic_agent.py \
  --start_date 2024-12-25 \
  --end_date 2024-12-27 \
  --lookback_days 7 \
  --sim_name test_run
```

On seal-node we already have GPUs, so you can go for it directly.

## Scoring Approach

**Method**: Brier-based peer scores + time-weighted averaging.

**How it works**:
1. Agents predict `{outcome: probability}` pairs (probabilities need not sum to 1)
2. At each time snapshot, compare all agents' current (carried-forward) predictions via Brier score
3. Peer score = how much better than average (zero-sum)
4. Final score = time-weighted average of peer scores

**Good properties**:
- Proper for free-form: no incentive to inflate probabilities on named outcomes
- Rewards early prediction (weighted by duration)
- Zero-sum (measures relative skill)
- Pluggable: can swap in alternative scorers

**Potential drawbacks**:
- Peer scores are NOT strictly proper (optimal strategy may depend on others' predictions)
- Abstention incentive: agents who skip questions get 0, which may be better than a bad prediction. (Open issue: consider assigning crowd aggregate to non-predictors.)

---

## Key Design Decisions

### 1. Agents report `{outcome: probability}`, not stakes

Agents output probability distributions over outcomes. This:
- Is directly interpretable as beliefs
- Enables proper scoring rule evaluation
- Allows Brier score analysis alongside peer scores
- Matches how researchers think about forecasts

### 2. "Other" outcome receives no payout

If agent outputs `{"Yes": 0.7, "No": 0.2}`, remaining 0.1 is implicit "Other".
If ground truth doesn't match any predicted outcome, agent gets **zero score**.

This prevents gaming: can't just say `{"Other": 0.99}` and collect safe points.

### 3. Answer matching uses Union-Find for transitivity

LLM-based semantic matching is inconsistent. Without enforcement:
- LLM says "Hinton" ≈ "Geoffrey Hinton"
- LLM says "Geoffrey Hinton" ≈ "Prof. Hinton"  
- LLM says "Hinton" ≠ "Prof. Hinton" ← contradiction!

We use Union-Find to:
- Enforce transitivity once equivalences are established
- Detect and log inconsistencies when LLM contradicts transitive closure
- Reduce LLM calls (check Union-Find before asking LLM)

### 4. Frozen daily aggregates for fairness

All agents on a given day see the **same frozen aggregate** from end of previous day.
This ensures:
- No agent has information advantage from order
- Peer scores are computed fairly against same baseline
- Parallelization is possible (agents prompt concurrently)

### 5. Predictions are additive, not replacement

Each agent can predict multiple times on same question. All predictions are stored.
Time-weighted scoring uses duration each prediction was "active".

This incentivizes:
- Predicting early (locks in early beliefs for scoring)
- Updating when new info arrives (improves later predictions)

---

## Context Data Organization (COMPLETED)

News articles are organized in `/is/cluster/fast/sgoel/forecasting/news/deduped_articles/`.

**Directory structure:**
```
deduped_articles/
├── data/
│   └── {YYYY}/{MM}/{DD}/
│       ├── articles_b0000.parquet  # Per-batch parquet files
│       ├── articles_b0001.parquet
│       ├── headlines_b0000.json    # Lightweight headline index
│       └── headlines_b0001.json
├── index/
│   ├── date_range.json   # {min, max, total_days, total_articles}
│   └── sources.json      # List of all source domains
├── current_sim/          # For simulation state (empty initially)
└── README.md
```

**Key decisions:**
- Multiple parquet files per day (one per processing batch) - PyArrow reads them as a single dataset
- ~20M+ articles across ~3000 days
- Headlines JSON files are lightweight for quick browsing without loading full articles

---

## Cluster Job Tips

- **htcondor v25+ uses `htcondor2` module**, not `htcondor`
- **Cluster access**: `kinit` on seal-node1, then `ssh login.cluster.is.localnet`
- **Memory limit**: Jobs killed if exceeding requested memory. Request 100GB+ for large data jobs.
- **Batch processing**: For large data, process in batches to avoid memory spikes. See `scripts/convert_jsonl_to_parquet.py` for example.

## Scoring Implementation

```
environment/scoring/
├── base.py       # BaseScorer ABC (extend this for new scorers)
├── brier.py      # BrierScorer (default) - Brier Skill Score
└── log_score.py  # LogScorer (Metaculus-style, not recommended for free-form)
```

**Brier Skill Score** (default): `1 - Σ(p_i - y_i)²` over `{named outcomes} ∪ {truth}`
- p_i = probability assigned (0 if not named)
- y_i = 1 if truth, 0 otherwise
- Range: -1 to +1, **0 = abstainer baseline**
- Higher is better

**Log** (alternative): `ln(p)` where p = P(truth). ⚠️ Incentivizes overconfidence in free-form.

**Peer score**: `100 × (my_score - avg_others)`. Zero-sum across agents.

---

## Rejected Scoring Alternatives

| Approach | Why Rejected |
|----------|-------------|
| LMSR | Path-dependent; simultaneous trades have no clean solution |
| Parimutuel | No incentive to predict early |
| Log score (for free-form) | Incentivizes overconfidence on named outcomes |
| Score implicit P(other) | Gameable via `{garbage: 0.01}` + P(other)=0.99 |

---

## Files Overview

| File | Purpose |
|------|---------|
| `environment/scoring/` | Modular scoring (Brier default, Log alternative) |
| `environment/ansmatching.py` | Union-Find based answer matching with LLM |
| `environment/interfaces.py` | `PredictionSubmission` dataclass |
| `environment/env.py` | SimulationEnvironment + SimForecastInterface + MarketWriter |
| `environment/data_loader.py` | QuestionPool with heap-based resolution tracking |
| `agents/base.py` | BaseAgent abstract class |
| `agents/basicAgent.py` | BasicAgent with DataFrame queries + XML forecasts |
| `agents/utils/memory.py` | BasicMemory class for persistent memory |
| `agents/utils/df_interface.py` | DfInterface for loading market.csv + query execution |
| `agents/utils/forecast_parser.py` | XML forecast parsing utilities |
| `environment/safe_executor.py` | QueryExecutor with eval() + timeout |
| `scripts/test_basic_agent.py` | CLI test script for BasicAgent |
| `notes/agent-forecast-interaction.md` | Agent interaction design documentation |

---

## How to Run

To run the BasicAgent test for the Dec 25-27 resolution window (with a 7-day lookback):

```bash
python scripts/test_basic_agent.py \
  --start_date 2024-12-25 \
  --end_date 2024-12-27 \
  --lookback_days 7 \
  --sim_name test_run
```

Arguments:
- `--start_date` / `--end_date`: Bounds for **resolution dates** (which questions resolve).
- `--lookback_days`: How many days before the first resolution to start the simulation (agent starts forecasting).
- `--sim_name`: Subdirectory name under `/is/cluster/fast/sgoel/forecasting/current_sim/`.

---

## Agent-Forecast Interaction (Summary)

See `notes/agent-forecast-interaction.md` for detailed design.

### High-Level Flow
```
1. Environment writes market.csv at start of day
2. Environment calls agent.act(None, forecast_interface, current_date)
3. Agent loads market.csv via DfInterface (adds my_prediction columns)
4. Query loop: Agent writes code → DfInterface executes → returns results (max N queries)
5. Submit loop: Agent outputs <submit>...</submit> → env parses and records predictions
6. Memory update: Agent optionally updates memory via <memory> tags
7. At resolution: time-weighted Brier peer scores computed
```

### Key Components
| Component | Purpose |
|-----------|---------|
| `VLLMInference.chat()` | Chat completion with messages format `[{role, content}]` |
| `SimForecastInterface` | Exposes `get_market_csv_path()`, `get_agent_predictions()`, `submit_prediction()` |
| `DfInterface` | Loads market.csv, adds agent-specific columns, executes queries |
| `QueryExecutor` | Runs agent code with eval() + 5s timeout |
| `BasicAgent` | Reference agent with CoT using `<reasoning>` and `<action>` tags |

### Agent Response Format
```xml
<reasoning>
Analysis of the questions and likely outcomes...
</reasoning>
<action>
```python
df[df['is_resolved']==False][['qid','title']].head()
```
</action>
```
Or for submission:
```xml
<action>
<submit>
  <forecast qid="12345">
    <outcome name="Tokyo" prob="0.4"/>
    ... allows multiple outcome prediction as long as sum of probabilities is 1 or lower.
  </forecast>
</submit>
</action>
```

### Scoring
- **Brier Skill Score**: `1 - Σ(p_i - y_i)²` over named outcomes + truth
- **Single-agent**: `peer_score = 100 × brier_score` (vs virtual abstainer at 0)
- **Multi-agent**: Zero-sum peer comparison
- Scores are time-weighted by days prediction was active before resolution

### DataFrame Schema (provided to agent)
```
qid, title, background, resolution_criteria, answer_type,
resolution_date, is_resolved, ground_truth, market_aggregate,
num_predictions, my_prediction, my_prediction_date
```

---

## ⚠️ Known Safety Issue

**Code execution in `safe_executor.py` uses `eval()`** which is NOT safe for untrusted agent code. This is acceptable for testing with controlled agents (like BasicAgent), but needs proper sandboxing for production/untrusted agents.

Options to explore later:
- RestrictedPython
- AST whitelisting
- Subprocess isolation
- Docker/container execution

---

## Future Work

### In Progress / High Priority

1. **Implement SimDocInterface**: Read articles from organized parquet files for RAG-style forecasting
2. **Build FAISS index**: Semantic search over articles for efficient context retrieval
3. **Sandbox code execution**: Replace eval() with safe execution (RestrictedPython, AST whitelisting, or subprocess isolation)

### Medium Priority

4. **Checkpointing**: Save/resume simulation state mid-run
5. **vLLM Batching for Multi-Agent**: Batch inference requests across agents for better throughput with local models
6. **Evaluation dashboard**: Track Brier scores, calibration curves, and prediction accuracy over time

### Low Priority / Nice-to-Have

7. **Deduplication across news batches**: Current approach may have duplicates if same article appears in multiple JSONL files
8. **Alternative agent scaffolds**: Create different agent architectures (e.g., ReAct, tree-of-thought) for comparison

### ✅ Completed

- ~~**Agent Memory**~~ - Persistent memory via `<memory>` tags in `BasicMemory` class
- ~~**Async agent execution**~~ - Multi-agent parallel via ThreadPoolExecutor
- ~~**Answer matching modes**~~ - `--matching exact|openrouter|vllm` with configurable matcher model
- ~~**Modular agent utilities**~~ - Extracted `DfInterface`, `BasicMemory`, `forecast_parser` to `agents/utils/`
- ~~**File-based market state**~~ - Environment writes `market.csv` for agent consumption
