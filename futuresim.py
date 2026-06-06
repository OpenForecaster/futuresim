"""Public Verifiers entrypoint for the Futuresim environment."""

from __future__ import annotations

from typing import Any


def load_environment(*args: Any, **kwargs: Any) -> Any:
    """Load the Futuresim Verifiers environment.

    This small shim is what ``verifiers.load_environment("futuresim")`` imports
    after the package is installed from a wheel or Prime Environment Hub.
    """
    from integrations.verifiers.futuresim_env import load_environment as _load_environment

    return _load_environment(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "FuturesimVerifiersEnv":
        from integrations.verifiers.futuresim_env import FuturesimVerifiersEnv

        return FuturesimVerifiersEnv
    raise AttributeError(name)


__all__ = ["FuturesimVerifiersEnv", "load_environment"]
