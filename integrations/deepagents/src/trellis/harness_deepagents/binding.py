"""Finding the execution a Deep Agents graph is running inside.

A compiled Deep Agents graph is expensive to build and is meant to be built once, at import
time, next to the tools it uses. The harness pieces it needs — the instrumented model client,
the memory runtime, the event stream — are per *run*. Binding them at construction would
force every application into a factory shape: build the graph again for every turn.

It is not needed. The harness already binds the current :class:`AgentRuntime` to a context
variable for the duration of an execution, so the middleware, the chat model and the backend
resolve it per call and one compiled graph serves every run. An explicit object still wins
when one is passed, which is what tests and nested runs use.
"""

from __future__ import annotations

from typing import Any

from trellis.harness.runtime.propagation import require_runtime

__all__ = ["HINT", "active_runtime"]

HINT = "run the Deep Agents graph through harness.deepagents.agent(...) / .wrap(...)"


def active_runtime(explicit: Any = None) -> Any:
    """The harness runtime to use: the one passed in, else the running execution's."""
    return require_runtime(explicit, hint=HINT)
