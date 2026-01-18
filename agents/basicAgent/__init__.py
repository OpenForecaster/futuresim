"""
BasicAgent: A simple LLM-based forecasting agent.

This package provides a basic forecasting agent that:
1. Receives a DataFrame of active/resolved questions
2. Writes Python code to explore the data
3. Submits probability forecasts in XML format
4. Can search news articles for context (when search tool available)
"""

from .agent import BasicAgent
from .config import AgentConfig

__all__ = ['BasicAgent', 'AgentConfig']
