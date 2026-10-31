"""The variants the benchmark compares, each a ``(target, h.wrap options)`` built from the
same instructions and model. A new cost feature lands as one more entry of :data:`VARIANTS`
(its middleware left out the way to turn it off), so its saving is measured against these two.

* ``baseline`` — the plain tool-calling loop, everything off: LangChain's ``create_agent``
  with only what runs the harness's tools and bounds the loop (``HarnessTools``,
  ``StallGuard``, ``StepLimit``, ``PatchToolCallsMiddleware``, ``ModelHooks``), no result
  paging, no context editing, no summarization, no automatic feature;
  ``without={"code_mode", "tool_search"}``.
* ``main`` — ``ReAct(...)`` as this branch builds it, nothing changed: with no Deep Agents
  capability asked for, LangChain's ``create_agent`` with ``trellis.middleware()`` (and Deep
  Agents' ``PatchToolCallsMiddleware``) — planning on a long task, older results cleared past
  half the window, the request fitted to the window (``FitWindow``).
* ``deep`` — ``ReAct`` on Deep Agents (a backend given: the journal's), ``create_deep_agent``
  with the harness's middleware: result paging (``FilesystemMiddleware``: a large result saved,
  a preview sent), summarization near the window's end, the scratch files and the ``task``
  helper on a long task.
* ``pieces`` — LangChain's ``create_agent`` with ``trellis.middleware()``'s pieces and Deep
  Agents' own pieces attached by hand (its ``FilesystemMiddleware`` on the journal, its
  summarization placed after the narrowing, ``PatchToolCallsMiddleware``; the scratch tools on a
  long task): Deep Agents' capabilities gated per call without ``create_deep_agent``'s assembly.
* ``main-planning``, ``deep-files`` — without a feature turned on by itself (:data:`AUTOMATIC`:
  its middleware left out); ``main@8k`` — ``main`` on an 8,192-token window;
* ``deep-<piece>`` — ``deep`` with one of Deep Agents' pieces off (:data:`PIECES`: ``paging``,
  ``context_editing``, ``summarization``), each measured on its own (``react.defaults_off``).

:data:`DEFAULT` are the variants a run compares unless ``--variants`` says.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, cast

from trellis import ReAct
from trellis.harness.react import PLACED, defaults_off


@dataclass(frozen=True)
class Setup:
    """What every variant is built from."""

    system: str
    model: Any  # a gateway model name, or a LangChain chat model
    window: int | None
    max_steps: int
    #: the harness tools every variant runs (a ``ReAct``'s ``tools=``)
    tools: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    build: Callable[[Setup], tuple[Any, dict[str, Any]]]


def baseline(setup: Setup) -> tuple[Any, dict[str, Any]]:
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from langchain.agents import create_agent

    from trellis.harness import middleware as ours
    from trellis.harness.adapters.base import TOOLS_ATTRIBUTE
    from trellis.harness.gateways import routed_chat_model

    graph = create_agent(
        routed_chat_model(setup.model) if isinstance(setup.model, str) else setup.model,
        tools=[],
        system_prompt=setup.system,
        middleware=[
            ours.HarnessTools(),
            ours.StallGuard(ours.MAX_REPEATS),
            ours.StepLimit(setup.max_steps),
            PatchToolCallsMiddleware(),
            ours.ModelHooks(),
        ],
        checkpointer=ours.RunCheckpointer(),
        name="baseline",
    )
    setattr(graph, TOOLS_ATTRIBUTE, list(setup.tools))  # run by HarnessTools, as ReAct's are
    return graph, {"without": {"code_mode", "tool_search"}}


def main(
    setup: Setup,
    *,
    off: tuple[str, ...] = (),
    window: int | None = None,
    deep: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """``ReAct`` as it is built — ``deep``: on Deep Agents (a backend given: the journal's)."""
    from trellis.harness.automatic import JournalBackend

    named = PIECES | AUTOMATIC
    with defaults_off(*(named[name] for name in off)):
        target = ReAct(
            setup.system,
            setup.model,
            max_steps=setup.max_steps,
            context_window=window or setup.window,
            tools=setup.tools,
            backend=JournalBackend() if deep else None,
        )
    return target, {}


def pieces(setup: Setup) -> tuple[Any, dict[str, Any]]:
    """``create_agent`` with the harness's middleware and Deep Agents' pieces, gated per call."""
    from deepagents.middleware import FilesystemMiddleware
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from deepagents.middleware.summarization import create_summarization_middleware
    from langchain.agents import create_agent

    from trellis.harness import automatic, react
    from trellis.harness import middleware as ours
    from trellis.harness.adapters.base import TOOLS_ATTRIBUTE
    from trellis.harness.gateways import routed_chat_model

    chat = routed_chat_model(setup.model) if isinstance(setup.model, str) else setup.model
    model = react._profiled(chat, setup.window)
    backend = automatic.JournalBackend()
    stack = react._middleware(
        max_steps=setup.max_steps,
        max_repeats=ours.MAX_REPEATS,
        call_timeout=None,
        context_window=setup.window,
        system_prompt=None,
        held=automatic.SCRATCH,
        files=False,
        summarization=react._placed(create_summarization_middleware(model, backend), react.PLACED),
    )
    graph = create_agent(
        model,
        tools=[],
        system_prompt=setup.system,
        middleware=cast(
            list[Any], [FilesystemMiddleware(backend=backend), PatchToolCallsMiddleware(), *stack]
        ),
        checkpointer=ours.RunCheckpointer(),
        name="pieces",
    )
    setattr(graph, TOOLS_ATTRIBUTE, list(setup.tools))  # run by HarnessTools, as ReAct's are
    return graph, {}


#: ``deep``'s pieces a ``deep-<piece>`` variant turns off, by the name of their middleware.
PIECES: Final = {
    "paging": "FilesystemMiddleware",
    "context_editing": "ContextEditingMiddleware",
    "summarization": PLACED,
}


#: The features ``ReAct`` turns on by itself (``trellis.harness.automatic``), by the name of
#: their middleware: a ``main-<feature>`` variant is ``main`` with it left out.
AUTOMATIC: Final = {"planning": "TodoListMiddleware", "files": "LongTaskTools"}
#: The window ``main@8k`` assumes whatever the model's: a small local model's.
SMALL_WINDOW: Final = 8_192


VARIANTS: Final[dict[str, Variant]] = {
    "baseline": Variant("baseline", "plain tools, everything off", baseline),
    "main": Variant(
        "main",
        "ReAct as it is built: create_agent with middleware() (no Deep Agents capability asked)",
        main,
    ),
    "deep": Variant(
        "deep",
        "ReAct on Deep Agents (a backend given): its file tools, paging, summarization, helper",
        functools.partial(main, deep=True),
    ),
    **{
        f"deep-{piece}": Variant(
            f"deep-{piece}",
            f"deep with {piece.replace('_', ' ')} off",
            functools.partial(main, off=(piece,), deep=True),
        )
        for piece in PIECES
    },
    "main-planning": Variant(
        "main-planning", "main without planning", functools.partial(main, off=("planning",))
    ),
    "deep-files": Variant(
        "deep-files",
        "deep without its long-task files",
        functools.partial(main, off=("files",), deep=True),
    ),
    "pieces": Variant(
        "pieces",
        "create_agent with middleware() and Deep Agents' pieces attached, gated per call",
        pieces,
    ),
    "main@16k": Variant(
        "main@16k",
        "main on a 16,384-token window (the model server's own: -c 16384)",
        functools.partial(main, window=16_384),
    ),
    "main@8k": Variant(
        "main@8k",
        "main on an 8,192-token window, whatever the model's",
        functools.partial(main, window=SMALL_WINDOW),
    ),
}
#: The variants a run compares by default.
DEFAULT: Final = ("baseline", "main")
