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
condor_submit_bid 25 -i -append request_gpus=1 -append "requirements=TARGET.CUDACapability >= 8.0" -append request_memory=80960
cd ~/forecast-sim
source fsim/bin/activate
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

With search:
```bash
python scripts/test_basic_agent.py \
    --sim_name search_xiaomi_test \
    --provider openrouter \
    --openrouter_model xiaomi/mimo-v2-flash:free \
    --search_db /is/cluster/fast/sgoel/forecasting/news/deduped_articles/lance/Qwen3-Embedding-8B
```

On seal-node we already have GPUs, so you can go for it directly.

## AllQAgent (Warmup Mode)
This agent performs a "warmup" phase on Day 0 where it predicts on **every single active question** before the simulation starts.
- **Goal**: Establish a baseline score by ensuring coverage of all questions.
- **Behavior**: 
    - **Day 0**: Iterates through all questions. For each, it runs a mini-loop (default max 10 actions) to search and submit a forecast.
    - **Day 1+**: Behaves like BasicAgent, but with a prompt reminder that initial predictions exist.
- **Memory**: The warmup phase does **not** read/write persistent memory. Memory is enabled starting Day 1.
- **Config**: Use `scaffold: "allQ"` and `warmup_max_actions: 10`.

## Simulation Resume & Restart

### Resume (continue from last day)
Use `--resume <output_dir>` to continue a simulation from where it left off. State is rebuilt from `actions.jsonl`.

### Restart from Specific Day
Use `--restart_from <source_dir> --restart_from_day YYYY-MM-DD` to re-run from a specific day:
```bash
python scripts/test_basic_agent.py \
    --restart_from /path/to/original/run \
    --restart_from_day 2025-04-05
```
- Creates a **new** output directory (`{sim_name}_restart/...`)
- Copies `actions.jsonl` entries and memory snapshots **before** restart day
- Uses existing `--resume` logic internally
- **Warmup predictions preserved** (main use case: avoid costly Day 0 re-runs)

### Per-Day Memory Storage
Memory is saved per-day as `agents/<agent_id>/memory/{YYYY-MM-DD}.txt`.
- On each day, `set_date(current_date)` loads the most recent memory file **before** that date
- Enables restarting from any day with correct memory state
- Backward compatible: if no memory files exist, starts with empty memory

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

### Directions

#### agent scaffold design

- decouple q choosing: e.g. always let all agents predict on all qs on day 1 based on past data, and then only “update” predictions based on the news they explore (should we only limit to that days news?)
- agent skills (in memory) but you learn from resolved questions
- context tools compare, incl. diff. retrieval models / techniques, rlm
- run sth like “hyperagents” (gepa on the agent scaffold)

#### training

- train (grpo?) each day based on projected brier score update for that day
- test-time / continual training on each day with same recipe as above?
- how to reward the memory which is carried forward to future days?

#### agent benchmark

- run claude code / codex / agy / opencode etc. and different frontier models
- launch as an env for people to do all sorts of research on
- studying different “behaviours” of different models

#### continual learning

- icl compaction / agent skills writing in memory
- test-time training
- just search agent

#### multi agent

- is relative ranking maintained between N different agents in a multi-agent sim vs single agent? or do separate “MA effects” kick in?
- the agents could be different scaffold/training techniques we expt with
- could also be copies of the same agent model+scaffold. “selfplay”

## Search Infrastructure

### Overview

LanceDB-based article search allows agents to query 14.7M+ news article chunks using hybrid (vector + keyword) search.

**Key locations:**
- Articles: `/is/cluster/fast/sgoel/forecasting/news/deduped_articles/data/YYYY/MM/DD/`
- Embeddings: `/is/cluster/fast/sgoel/forecasting/news/embeddings/Qwen3-Embedding-8B/`
- LanceDB: `/is/cluster/fast/sgoel/forecasting/news/deduped_articles/lance/Qwen3-Embedding-8B/`
- Embedding model: `/is/cluster/fast/sgoel/models/Qwen3-Embedding-8B`

### Key Scripts

| Script | Purpose |
|--------|---------|
| `mpi_scripts/embed/submit_job.py` | Submit embedding jobs to cluster |
| `scripts/build_lancedb.py` | Build LanceDB from articles + embeddings |
| `scripts/build_lancedb_index.py` | Create IVF vector index (CRITICAL for performance) |

### Performance

| Index State | Query Time |
|-------------|-----------|
| No vector index | ~300 sec/query ❌ |
| With IVF index | ~10-25 sec/query ✅ |

**IVF Index Parameters:**
- `num_partitions=256`
- `num_sub_vectors=64` (must divide vector dim 4096 evenly)
- `metric=cosine`

### GPU Memory Allocation

When using both embedding model and matcher model on same GPU:
- `--embedding_gpu_mem 0.4` (40% for Qwen3-Embedding-8B)
- `--matcher_gpu_mem 0.3` (30% for qwen3-4b-it matcher)

### Agent Search Action Format

```xml
<!-- Basic search -->
<action type="search">
query text here
</action>

<!-- With date range (YYYY-MM-DD format) -->
<action type="search" from="2024-12-01" to="2024-12-15">
query text here
</action>
```

- `to` date is capped at simulation date (no future leakage)
- Parser in `agents/utils/forecast_parser.py` handles the from/to attributes

### Search Result Format

```
═══ [1] ═══════════════════════════════════════
HEADLINE: Article Title Here
SOURCE: example.com
PUBLISHED: 2024-12-15 | DOWNLOADED: 2024-12-20
URL: https://...

Full chunk content (512 tokens max, not truncated)
```

### Timing Metrics

Agent timing stats saved to `<agent_dir>/timing_stats.jsonl`.

### Full Setup

See `agents/search_tools/README.md` for complete setup instructions.

### In Progress / High Priority

Q: why does search yield no results sometimes? it should still pull the most relevant articles instead of outright wasting queries

Verify each line of env code



### Medium Priority

**Checkpointing**: Test Save/resume simulation state mid-run
**vLLM Batching for Multi-Agent**: Batch inference requests across agents for better throughput with local modelse

### Low Priority / Nice-to-Have

**standard agent action format**: should i switch to openai harmony format for example?

**Deduplication across news batches**: Current approach may have duplicates if same article appears in multiple JSONL files

Q: in trying to make many submissions models eg grok dont get to use cot/reasoning. what to do about this?

1. **Sandbox code execution**: Replace eval() with safe execution (RestrictedPython, AST whitelisting, or subprocess isolation)

## Market Integration Roadmap (Planned)

### Deferred to Future Work
1. **Additional Platforms**:
   - **Kalshi**: Need specialized fetcher and scoring.
   - **Polymarket**: Prediction market format (shares/cents).
   - **Manifold**: Play-money market.

2. **Scoring for Prediction Markets**:
   - Implement **Log Score** or **Profit/Loss** metrics for market-based platforms.
   - Current Brier score is suited for probability forecasts but markets might need ROI-based metrics.

3. **Complex Question Types**:
   - **Numeric**: Range scoring, PDF construction.
   - **Date**: Evaluation of date proximity (e.g. L1/L2 distance).
   - Current Metaculus integration only supports Binary and MCQ.

4. **Metaculus Fetching**:
   - Add supports for `numeric` and `date` types in `MetaculusFetcher`.
   - Implement `resolution` value parsing logic for these types.