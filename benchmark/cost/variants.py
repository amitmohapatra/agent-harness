"""The variants the benchmark compares, each a ``(target, h.wrap options)`` built from the
same instructions and model. A new cost feature lands as one more entry of :data:`VARIANTS`
(its middleware left out the way to turn it off), so its saving is measured against these two.

* ``baseline`` — the plain tool-calling loop, everything off: LangChain's ``create_agent``
  with only what runs the harness's tools and bounds the loop (``HarnessTools``,
  ``StallGuard``, ``StepLimit``, ``PatchToolCallsMiddleware``, ``ModelHooks``), no result
  paging, no context editing, no summarization, no automatic feature;
  ``without={"code_mode", "tool_search"}``.
* ``main`` — ``ReAct(...)`` as this branch builds it, nothing changed: Deep Agents'
  ``create_deep_agent`` with the harness's middleware — the features it turns on by itself
  (planning, scratch files, the ``task`` helper, tool search...), result paging
  (``FilesystemMiddleware``: a large result saved, a preview sent), older results cleared past
  half the window, summarization near its end, ``ToolGuidance`` and ``AnswerCheck`` automatic
  (``None``), Code Mode where it applies (read-only Code Mode MCP servers of the gateway; this
  benchmark's catalogues are the harness's own tools, so it does not).
* ``agent`` — LangChain's ``create_agent`` with ``trellis.middleware()`` (and Deep Agents'
  ``PatchToolCallsMiddleware``): the harness's part without Deep Agents' (no result paging, no
  summarization, no file tools or helper) — the other base ``ReAct`` could have.
* ``pieces`` — LangChain's ``create_agent`` with ``trellis.middleware()``'s pieces and Deep
  Agents' own pieces attached by hand (its ``FilesystemMiddleware`` on the journal, its
  summarization placed after the narrowing, ``PatchToolCallsMiddleware``; the scratch tools on a
  long task): Deep Agents' capabilities on demand without ``create_deep_agent``'s assembly.
* ``main+guidance``, ``main+check`` — ``main`` with ``tool_guidance=True`` or
  ``answer_check=True``: what each is worth when forced on.
* ``main-<feature>`` — ``main`` without each feature it turns on by itself
  (:data:`AUTOMATIC`: its middleware left out); ``main@8k`` — ``main`` on an 8,192-token
  window;
* ``main-<piece>`` — ``main`` with one of its pieces off (:data:`PIECES`: ``paging``,
  ``context_editing``, ``summarization``), so each piece is measured on its own: the default
  middleware of that name left out (``react.defaults_off``).

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
    switches: dict[str, bool] | None = None,
) -> tuple[Any, dict[str, Any]]:
    with defaults_off(*({**PIECES, **AUTOMATIC}[name] for name in off)):
        target = ReAct(
            setup.system,
            setup.model,
            max_steps=setup.max_steps,
            context_window=window or setup.window,
            tools=setup.tools,
            tool_guidance=(switches or {}).get("tool_guidance"),
            answer_check=(switches or {}).get("answer_check"),
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
        tool_guidance=None,
        answer_check=None,
        capabilities=None,
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


def agent(setup: Setup) -> tuple[Any, dict[str, Any]]:
    """``create_agent`` with the harness's middleware: ``ReAct``'s settings, no Deep Agents."""
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from langchain.agents import create_agent

    from trellis import middleware
    from trellis.harness import middleware as ours
    from trellis.harness.adapters.base import TOOLS_ATTRIBUTE
    from trellis.harness.gateways import routed_chat_model

    graph = create_agent(
        routed_chat_model(setup.model) if isinstance(setup.model, str) else setup.model,
        tools=[],
        system_prompt=setup.system,
        middleware=[
            PatchToolCallsMiddleware(),
            *middleware(max_steps=setup.max_steps, context_window=setup.window),
        ],
        checkpointer=ours.RunCheckpointer(),
        name="agent",
    )
    setattr(graph, TOOLS_ATTRIBUTE, list(setup.tools))  # run by HarnessTools, as ReAct's are
    return graph, {}


#: ``main``'s pieces a ``main-<piece>`` variant turns off, by the name of their middleware.
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
        "ReAct as main builds it: result paging, context editing, summarization; ToolGuidance off",
        main,
    ),
    **{
        f"main-{piece}": Variant(
            f"main-{piece}",
            f"main with {piece.replace('_', ' ')} off",
            functools.partial(main, off=(piece,)),
        )
        for piece in PIECES
    },
    **{
        f"main-{feature}": Variant(
            f"main-{feature}",
            f"main without {feature}",
            functools.partial(main, off=(feature,)),
        )
        for feature in AUTOMATIC
    },
    "agent": Variant(
        "agent",
        "create_agent with trellis.middleware(): the harness's part without Deep Agents",
        agent,
    ),
    "pieces": Variant(
        "pieces",
        "create_agent with middleware() and Deep Agents' pieces attached, gated per call",
        pieces,
    ),
    "main+guidance": Variant(
        "main+guidance",
        "main with tool_guidance=True",
        functools.partial(main, switches={"tool_guidance": True}),
    ),
    "main+check": Variant(
        "main+check",
        "main with answer_check=True",
        functools.partial(main, switches={"answer_check": True}),
    ),
    "main@8k": Variant(
        "main@8k",
        "main on an 8,192-token window, whatever the model's",
        functools.partial(main, window=SMALL_WINDOW),
    ),
}
#: The variants a run compares by default.
DEFAULT: Final = ("baseline", "main")
