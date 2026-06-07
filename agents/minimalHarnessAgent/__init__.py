__all__ = [
    "MinimalHarnessAgent",
    "MinimalHarnessConfig",
    "ClaudeCodeAgent",
    "CodexAgent",
    "OpenCodeAgent",
]


def __getattr__(name):
    if name == "MinimalHarnessConfig":
        from .config import MinimalHarnessConfig

        return MinimalHarnessConfig
    if name == "MinimalHarnessAgent":
        from .agent import MinimalHarnessAgent

        return MinimalHarnessAgent
    if name == "ClaudeCodeAgent":
        from .claude_code_agent import ClaudeCodeAgent

        return ClaudeCodeAgent
    if name == "CodexAgent":
        from .codex_agent import CodexAgent

        return CodexAgent
    if name == "OpenCodeAgent":
        from .opencode_agent import OpenCodeAgent

        return OpenCodeAgent
    raise AttributeError(name)
