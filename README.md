See scripts/test_basic_agent.py launch commands are at the top of it.

notes/memory.md has a summary to understand the codebase / design choices.

# Running AllQAgent
To run the agent that predicts on ALL questions during Day 0 warmup:
```bash
python scripts/test_basic_agent.py --config configs/allq_sim.yaml
```
(Or use `--scaffold allQ` with CLI args).
