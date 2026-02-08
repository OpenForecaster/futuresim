# Installation

To reproduce the environment, first make sure you have `uv` installed.
Then:

```bash
# Initialize and sync environment
uv sync

# Initialize news pipeline dependencies (applies patches)
./data/news/scripts/setup_news_pipeline.sh

# Activate environment
source .venv/bin/activate
```

See scripts/test_basic_agent.py launch commands are at the top of it.

notes/memory.md has a summary to understand the codebase / design choices.

# Running AllQAgent
To run the agent that predicts on ALL questions during Day 0 warmup:
```bash
python scripts/test_basic_agent.py --config configs/allq_sim.yaml
```
(Or use `--scaffold allQ` with CLI args).
