# Forecast-Sim: Issues & Improvement Areas

A comprehensive list of low-level issues that impact efficiency, reproducibility, model performance, and benchmark signal. These need to be addressed for forecast-sim to become a rigorous frontier benchmark.

---

## 1. Reproducibility

### 1.1 Incomplete Seeding
**File**: `scripts/run_sim.py`

Only `random.seed(args.seed)` is called (line 96). Missing:
- `numpy.random.seed()`
- `torch.manual_seed()` / `torch.cuda.manual_seed_all()`

**Impact**: Any component using numpy (pandas internals, lancedb) or torch (embedding models) introduces non-determinism. Running the same simulation twice with the same seed can produce different results.

**Fix**: Create a `set_global_seed(seed)` utility that sets all three.

---

### 1.2 Non-Deterministic Inference Settings
**File**: `configs/default_sim.yaml`

Default `temperature: 0.7` makes LLM outputs stochastic.

**Impact**: Benchmark runs are not reproducible even with identical code/seed.

**Fix**: Add a `benchmark_mode` config that forces `temperature: 0.0` for reproducible evaluations.

---

### 1.3 Undefined Float Precision in Logs
**File**: `environment/env.py`

Scores logged with inconsistent precision:
- `log_daily_metrics`: uses `.4f` for brier, peer (line 99)
- `_print_final_summary`: uses `.3f`, `.2f` in different places

**Impact**: Floating point drift can cause "butterfly effects" in long simulations where agent decisions depend on aggregate scores (reading them back from CSV).

**Fix**: Standardize to a fixed precision (e.g., 6 decimal places) across all logging.

---

## 2. Efficiency (Time)

### 2.1 Global Lock Contention in Logging
**File**: `environment/env.py` – `SimLogger` class

Uses `threading.Lock` for every write operation:
- `_actions_lock` (line 35)
- `_metrics_lock` (line 38)
- `_matcher_lock` (line 44)

**Impact**: With 50+ parallel agents, threads spend significant time waiting for locks, negating parallelism benefits.

**Fix**: Refactor to a queue-based logging pattern. Agents write to a non-blocking queue; a single dedicated thread drains to disk.

---

### 2.2 Quadratic Warmup Cost
**File**: `agents/allQAgent/agent.py`

`AllQAgent.warmup()` iterates through *every* active question on Day 0 to make initial predictions.

**Impact**: With 5,000 active questions, this means 5,000+ LLM calls on the first simulated day. Hits rate limits immediately and wastes massive tokens on low-priority questions.

**Fix**: Prioritize "resolving soon" questions, use batching, or use a cheaper/faster model for initial bulk predictions.

---

### 2.3 No Embedding Cache
**File**: `agents/search_tools/lancedb/store.py`

`_encode_query()` re-encodes the query string every time, even if identical queries are issued.

**Impact**: If multiple agents search for "US Election 2024", the embedding is computed N times.

**Fix**: Add an LRU cache for query embeddings.

---

## 3. Efficiency (Tokens)

### 3.1 Unbounded Memory Growth
**File**: `agents/utils/memory.py` – `BasicMemory` class

`update()` completely replaces memory with whatever the agent outputs. No structured summarization, pruning, or sliding window.

**Impact**: Memory can grow unboundedly, eventually overflowing context window or becoming prohibitively expensive.

**Fix**: Implement `SummarizingMemory` that:
- Keeps last N days raw
- Summarizes older days into a "long-term context" block using a cheap model
- Enforces a hard token limit

---

### 3.2 No Token Accounting
**Files**: `agents/basicAgent/agent.py`, `inference/openrouter.py`

While `_timer.record_tokens(usage)` exists, there's no aggregation or reporting of tokens consumed per agent/day.

**Impact**: Cannot measure "accuracy per token" or optimize prompt efficiency. Critical for benchmark comparisons.

**Fix**: Add a token accounting wrapper that logs cumulative input/output tokens and includes them in `daily_metrics.csv`.

---

## 4. Benchmark Signal & Rigor

### 4.1 Silent Search Degradation
**File**: `agents/search_tools/lancedb/store.py`

In `search()` (lines 102-128), if embedding fails:
- Falls back silently to keyword search (FTS)
- Catches exceptions and returns empty results

**Impact**: A benchmark run may get wildly different results (and lower performance) simply because the embedding server was busy, without any indication to the user.

**Fix**: 
- Add a config flag `strict_search: true` that raises an error instead of falling back
- Log warnings when fallback occurs
- Retry with backoff before failing

---

### 4.2 Data Loading Bottleneck
**File**: `environment/data_loader.py` – `QuestionPool` class

Loads the *entire* dataset into memory and builds a heap index at startup (`_build_index()`).

**Impact**: Prevents testing on "frontier" scale datasets (1M+ questions) due to RAM and startup time.

**Fix**: Implement lazy loading or streaming from the HuggingFace dataset.

---

### 4.3 No Regression Testing Infrastructure
**Files**: `tests/` directory

Only `test_scoring.py` exists. No reproducibility tests.

**Impact**: Cannot verify that code changes don't break determinism or alter benchmark results.

**Fix**: Add `tests/reproducibility/` with:
- A small golden dataset (3-day simulation)
- Hash comparison of final `market.csv` and `daily_metrics.csv`
- Run as part of CI

---

## 5. Prompt Architecture

### 5.1 Tightly Coupled Prompt Construction
**File**: `agents/basicAgent/agent.py` – `_build_instructions()` (lines 310-446)

The system prompt is built as a single massive f-string with:
- Inline conditional sections (search enabled/disabled)
- Hardcoded scoring explanations
- DataFrame info embedded directly
- Memory section inline

**Problems**:
- Hard to A/B test different prompt variants
- Changing one section risks breaking others
- No separation between "rules" and "context"
- Difficult to version control prompt changes independently

**Fix**: Modular prompt builder with:
```python
class PromptBuilder:
    def __init__(self):
        self.sections = []
    
    def add_section(self, name: str, content: str, priority: int = 0):
        ...
    
    def build(self) -> str:
        ...
```
Each section (scoring rules, data schema, action format, etc.) defined separately and composable.

---

### 5.2 Memory Prompt Duplication
**Files**: `agents/basicAgent/agent.py` (lines 456-480), `agents/allQAgent/agent.py`

Memory update prompt is duplicated/hardcoded in multiple places. The format instructions for `<memory>` tags are embedded in the prompt.

**Fix**: Extract memory prompt to a separate template that can be versioned independently.

---

## 6. Message Format & Model Input/Output

### 6.1 Tool Outputs as `user` vs Dedicated Role
**File**: `agents/basicAgent/agent.py` – action handlers

All tool outputs (query results, search results, submit confirmations) are appended as `{"role": "user", "content": feedback}`:
- Line 201: `messages.append({"role": "user", "content": feedback})`
- Line 246: `messages.append({"role": "user", "content": feedback})`
- Line 275: `messages.append({"role": "user", "content": feedback})`

**问题**: 
- Modern LLMs (GPT-4, Claude) have native `tool` or `function` roles that are trained differently
- Using `user` role may confuse the model about who is "speaking"
- No clear visual/semantic distinction between human instructions and system feedback

**Considerations**:
- OpenRouter may not support `tool` role for all models
- Some models perform better with `user` role for tool outputs
- Need empirical testing to determine optimal approach

**Options to Explore**:
1. Use `tool` role where supported
2. Use `assistant` role with a prefix like `[SYSTEM FEEDBACK]:`
3. Use structured markers within user messages: `<tool_output>...</tool_output>`
4. Keep as-is but document the decision

---

### 6.2 No Message Compression/Truncation
**File**: `agents/basicAgent/agent.py`

The full conversation history is passed to every LLM call. No truncation strategy.

**Impact**: 
- Context window can overflow after many actions
- Older query results (potentially irrelevant) consume tokens
- No way to prioritize recent vs. important information

**Fix**: Implement a message manager that:
- Keeps system prompt and last N turns
- Summarizes/drops old tool outputs
- Prioritizes recent actions over historical ones

---

### 6.3 Inconsistent Action Parsing Feedback
**File**: `agents/basicAgent/agent.py` – `_handle_invalid()` (lines 278-287)

When action parsing fails, the feedback is generic:
```python
feedback = f"No valid action found. {parsed.error or 'Use <action type=\"...\">...</action> format.'}"
```

**Impact**: Model doesn't get specific guidance on *what* was wrong (missing closing tag? wrong attribute? malformed XML?).

**Fix**: Improve parser error messages to be more diagnostic.

---

### 6.4 Reasoning Not Consistently Available
**File**: `agents/basicAgent/agent.py`

Reasoning is extracted from `usage.get("_reasoning_content")` which is provider-specific (OpenRouter extended thinking).

```python
reasoning = usage.get("_reasoning_content") if usage else None
```

**Impact**: 
- Only works with specific providers/models
- Analysis tools can't reliably access chain-of-thought
- No fallback for models that don't support extended thinking

**Fix**: Also parse `<reasoning>` tags from the response itself as a fallback.

---

### 6.5 Feedback Message Prefix Inconsistency
**File**: `agents/basicAgent/agent.py`

Different feedback types use different prefixes:
- `QUERY RESULT:` / `QUERY ERROR:` (lines 192-194)
- `SEARCH RESULTS:` / `SEARCH ERROR:` (lines 237-239)
- `Submitted X forecast(s).` (line 267) – no prefix
- `SUBMIT ERROR:` (line 272)

**Impact**: Inconsistent formatting may confuse the model about message types.

**Fix**: Standardize all feedback with consistent formatting:
```
[TOOL: query] SUCCESS
{result}

[TOOL: search] ERROR
{error_message}
```

---

## 7. Scalability

### 7.1 Single-Process Simulation
**File**: `environment/env.py`

All agents run in threads within a single Python process.

**Impact**: 
- Limited by GIL for CPU-bound operations
- Cannot scale across multiple machines
- Memory pressure from all agents sharing one process

**Future Fix**: Support distributed simulation with Ray or similar.

---

### 7.2 No Checkpointing During Long Runs
**File**: `environment/env.py`

Resume only works from `actions.jsonl` replay. No periodic checkpointing.

**Impact**: If a 100-day simulation crashes on day 80, must replay 80 days of actions.

**Fix**: Add periodic state snapshots that can be loaded directly.

---

## Summary Priority Matrix

| Issue | Impact | Effort | Priority |
|-------|--------|--------|----------|
| 1.1 Incomplete Seeding | High | Low | **P0** |
| 4.1 Silent Search Degradation | High | Low | **P0** |
| 5.1 Tightly Coupled Prompts | High | Medium | **P1** |
| 3.1 Unbounded Memory | High | Medium | **P1** |
| 6.1 Tool Output Role | Medium | Low | **P1** |
| 2.1 Lock Contention | Medium | Medium | **P2** |
| 6.2 No Message Truncation | Medium | Medium | **P2** |
| 3.2 No Token Accounting | Medium | Low | **P2** |
| 2.2 Quadratic Warmup | Medium | Medium | **P2** |
| 4.3 No Regression Tests | Medium | Medium | **P2** |
| 1.3 Float Precision | Low | Low | **P3** |
| 2.3 No Embedding Cache | Low | Low | **P3** |
| 6.5 Feedback Prefix Consistency | Low | Low | **P3** |
