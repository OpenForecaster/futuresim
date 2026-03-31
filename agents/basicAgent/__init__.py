"""
BasicAgent: A simple LLM-based forecasting agent.

This package provides a basic forecasting agent that:
1. Receives a DataFrame of active/resolved questions
2. Writes Python code to explore the data
3. Submits probability forecasts through Chat Completions tools
4. Can search news article chunks for context (retrieval count and chunk size depend on search config)
"""

from .agent import BasicAgent
from .config import AgentConfig

__all__ = ['BasicAgent', 'AgentConfig']
