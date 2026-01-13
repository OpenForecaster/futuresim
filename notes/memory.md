# Forecasting Simulator: Design Context & Decisions

This document captures design decisions, rejected alternatives, and context that isn't obvious from code alone. For future developers/models working on this codebase.

---

## Project Goal

Build a **multi-agent forecasting simulator** where LLM agents:
1. Read news/context articles
2. Make probabilistic predictions on free-form questions
3. Are scored fairly against each other
4. Learn behaviors that transfer to real prediction markets

The ultimate goal is **training agents whose forecasting behaviors generalize to real-world prediction markets**.

---

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
| `environment/interfaces.py` | QuestionView, PredictionSubmission datatypes |
| `environment/env.py` | SimulationEnvironment orchestrating daily flow |
| `environment/data_loader.py` | QuestionPool with heap-based resolution tracking |
| `agents/base.py` | BaseAgent abstract class |
| `scripts/run_sim.py` | CLI entry point with stub agents |
| `scripts/convert_jsonl_to_parquet.py` | Streaming JSONL→Parquet conversion |
| `mpi_scripts/organize_news/submit_job.py` | HTCondor job submission (uses htcondor2) |
| `mpi_scripts/organize_news/run_conversion.sh` | Shell wrapper for cluster jobs |

---

## Future Work

1. **Implement SimDocInterface**: Read articles from organized parquet files
2. **Build FAISS index**: Semantic search over articles for RAG-style agents
3. **Real LLM agents**: Replace stub agents with actual inference
4. **Async agent execution**: Parallel prompts for speed
5. **Evaluation metrics**: Track Brier scores separately from peer scores
6. **Checkpointing**: Save/resume simulation state
7. **Deduplication across batches**: Current approach may have duplicates if same article appears in multiple JSONL files
