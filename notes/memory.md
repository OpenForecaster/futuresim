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

## Why We Chose Metaculus-Style Scoring (Not LMSR)

### LMSR (Logarithmic Market Scoring Rule) - REJECTED

We initially implemented LMSR but rejected it for several reasons:

1. **Simultaneous trades are problematic**: LMSR is path-dependent. If Agent A and B both bet at the "same time", execution order affects prices. No mathematically clean way to handle true simultaneity.

2. **Stake ≠ Probability**: LMSR conflates belief strength with budget. An agent with $1000 moves the market more than one with $10, even if beliefs are identical.

### Parimutuel - CONSIDERED

Parimutuel (pool all bets, winners split pool) was considered because:
- ✅ Naturally simultaneous (no ordering issues)
- ✅ Handles N outcomes trivially
- ❌ No incentive to predict EARLY (only final pool ratio matters)
- ❌ Late bettors see more information, no disadvantage

### Metaculus-Style Peer Scoring - CHOSEN

Final choice: Log score + peer comparison + time-weighted averaging.

**Why it works:**
- **Proper scoring rule** (log score): Optimal strategy is to report true beliefs
- **Peer comparison**: Score = 100 × (my_log - avg_of_others). Measures relative skill.
- **Time-weighted**: Predictions weighted by how long they were active. Early correct predictions earn more.
- **Simultaneous**: All agents on same day scored against each other. No ordering issues.

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

## Context Data Organization (NOT YET IMPLEMENTED)

We discussed but deferred implementing context (news articles) organization.

**Preferred approach: Hybrid**
- **Canonical storage**: Single `articles.parquet` file
- **Filesystem view**: Build script generates `{date}/{source}/article.json` tree
- **Search index**: FAISS index built from same parquet source

Agents need both:
- **Filesystem navigation**: List dates → list sources → list articles → read content
- **Semantic search**: Query → relevant articles (for RAG-style agents)

The `SimDocInterface` is currently stubbed, waiting for context format finalization.

---

## Scoring Math Reference

### Log Score
```
log_score(p) = ln(p)  # where p = probability assigned to true outcome
```
- Perfect (p=1): score = 0
- Worst (p→0): score → -∞
- We clamp to [0.001, 0.999] to avoid infinities

### Peer Score
```
peer_score = 100 × (my_log_score - mean(others_log_scores))
```
- Positive = better than average
- Zero-sum across all agents per question

### Time-Weighted Average
```
For each prediction interval [t_start, t_end]:
    weight = duration
    contribution = peer_score × weight

final_score = sum(contributions) / sum(weights)
```

---

## Rejected Alternatives

| Idea | Why Rejected |
|------|--------------|
| LMSR market | Path-dependent, ordering issues, stakes ≠ beliefs |
| Parimutuel | No early prediction incentive |
| Brier score only | No early prediction incentive (best to predict at end) |
| Sequential agent execution | Unfair ordering, first agent sees different state |
| Stakes determine weights | Conflates budget with belief strength |
| LLM-only answer matching | LLM is inconsistent, violates transitivity |

---

## Files Overview

| File | Purpose |
|------|---------|
| `environment/scoring.py` | Log score, peer score, time-weighted averaging |
| `environment/ansmatching.py` | Union-Find based answer matching with LLM |
| `environment/interfaces.py` | QuestionView, PredictionSubmission datatypes |
| `environment/env.py` | SimulationEnvironment orchestrating daily flow |
| `environment/data_loader.py` | QuestionPool with heap-based resolution tracking |
| `agents/base.py` | BaseAgent abstract class |
| `scripts/run_sim.py` | CLI entry point with stub agents |

---

## Future Work

1. **Implement SimDocInterface**: Read articles from context directory
2. **Build context organization**: Parquet + filesystem view + FAISS index  
3. **Real LLM agents**: Replace stub agents with actual inference
4. **Async agent execution**: Parallel prompts for speed
5. **Evaluation metrics**: Track Brier scores separately from peer scores
6. **Checkpointing**: Save/resume simulation state
