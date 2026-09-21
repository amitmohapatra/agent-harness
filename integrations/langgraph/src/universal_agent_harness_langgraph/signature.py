"""Building a wrapper LangGraph will hand the right arguments to.

LangGraph injects node arguments **by parameter name and annotation** (``config``,
``store``, ``writer``, ``previous``, ``runtime``) — that is the documented way to ask for
them. A wrapper therefore has to *declare* the parameters the wrapped node wants, so this
module generates a wrapper function with exactly that signature and forwards the injected
values through. Nothing here reaches into LangGraph's internals: it only writes a function
whose signature follows the public convention.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

#: Injectables a node may declare, with the annotation LangGraph matches on.
INJECTABLES = ("config", "store", "writer", "previous", "runtime")


def declared_injectables(fn: Callable[..., Any]) -> tuple[str, ...]:
    """Which injectables the wrapped node asks for, in signature order."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return ()
    valid = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return tuple(name for name in INJECTABLES if name in params and params[name].kind in valid)


def positional_arity(fn: Callable[..., Any]) -> int:
    """Positional parameters that are not injectables — 1 for ``(state)``, 2 for
    ``(state, agent)`` (a runtime-aware node)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return 1
    return len(
        [
            p
            for name, p in params.items()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and name not in INJECTABLES
        ]
    )


def build_node(
    impl: Callable[..., Any],
    *,
    name: str,
    injectables: tuple[str, ...],
    is_async: bool = True,
) -> Callable[..., Any]:
    """Create ``async def <name>(state, config: RunnableConfig, ...)`` calling ``impl``.

    ``impl(state, injected: dict)`` receives whatever LangGraph passed, so the wrapped node
    can be given back exactly the arguments it declared.
    """
    # LangGraph types are imported inside the adapter only, and only when building a node.
    from langchain_core.runnables import RunnableConfig  # noqa: PLC0415
    from langgraph.store.base import BaseStore  # noqa: PLC0415
    from langgraph.types import StreamWriter  # noqa: PLC0415

    params = ["state", "config: RunnableConfig = None"]
    forwarded = ["'config': config"]
    for injectable in injectables:
        if injectable == "config":
            continue
        annotation = {
            "store": "store: Optional[BaseStore] = None",
            "writer": "writer: StreamWriter = None",
            "previous": "previous = None",
            "runtime": "runtime = None",
        }[injectable]
        params.append(annotation)
        forwarded.append(f"'{injectable}': {injectable}")

    signature = ", ".join(params)
    injected = "{" + ", ".join(forwarded) + "}"
    source = (
        f"async def _node({signature}):\n    return await impl(state, {injected})\n"
        if is_async
        else f"def _node({signature}):\n    return run_sync(impl(state, {injected}))\n"
    )

    # ``Optional[BaseStore]`` (not ``BaseStore | None``) is the annotation LangGraph
    # matches when deciding whether to inject the store.
    from typing import Optional  # noqa: PLC0415

    from universal_agent_harness.execution.sync import run_sync  # noqa: PLC0415

    namespace: dict[str, Any] = {
        "impl": impl,
        "RunnableConfig": RunnableConfig,
        "BaseStore": BaseStore,
        "StreamWriter": StreamWriter,
        "Optional": Optional,
        "run_sync": run_sync,
    }
    exec(compile(source, f"<harness-node:{name}>", "exec"), namespace)
    node = namespace["_node"]
    node.__name__ = name
    node.__qualname__ = name
    return node
