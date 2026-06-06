"""Source-tree alias for the local agent implementations.

The public wheel installs the files from ``agents/`` under the package name
``futuresim_agents`` so it does not collide with Verifiers' ``openai-agents``
dependency, which also imports as ``agents``.
"""

from pathlib import Path

_SOURCE_AGENTS = Path(__file__).resolve().parents[1] / "agents"
__path__ = [str(_SOURCE_AGENTS)]
