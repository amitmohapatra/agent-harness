"""The variants the benchmark compares, each a ``(target, h.wrap options)`` built from the
same instructions and model. A new cost feature lands as one more entry of :data:`VARIANTS`
(its middleware left out the way to turn it off), so its saving is measured against these two.

* ``baseline`` — the plain tool-calling loop, everything off: LangChain's ``create_agent``
  with only what runs the harness's tools and bounds the loop (``HarnessTools``, the loop's
  limits — ``react.limits`` — ``PatchToolCallsMiddleware``, ``ModelHooks``), no result
  paging, no context editing, no summarization, no automatic feature;
  ``without={"code_mode", "tool_search"}``.
* ``main`` — ``ReAct(...)`` as the ``trellis`` on the path builds it, nothing changed (run on a
  worktree of origin/main it is *today's* ReAct, the arm every change is measured against;
  the report names the source it ran: ``environment.trellis``): with no Deep Agents
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

The ReAct-loop arms (``docs/frameworks/react.md#measuring-the-loop``), each a source that has
the library's limits (``react.limits``; refused on one without them):

* ``all`` — ``main`` as this branch builds it: the library's limits with the repeat check, tool
  errors as results, and model calls retried by ``ModelRetryMiddleware`` where no gateway does;
* ``limitsA`` — LangChain's limits only (``ModelCallLimitMiddleware``,
  ``ToolCallLimitMiddleware`` on the memory search tools), the repeat check (``StallGuard``)
  and the errors' pieces off;
* ``limitsB`` — ``limitsA`` and the repeat check (the errors' pieces off);
* ``errorsA`` — ``limitsA`` and the errors' pieces (``ToolErrorMiddleware``,
  ``ModelRetryMiddleware``): the errors change measured on the base without the repeat check
  (``all`` is it on ``limitsB``);
* ``context-native`` — prepared, not measured yet: LangChain's ``ContextEditingMiddleware``
  (its own placeholder) and ``SummarizationMiddleware`` in place of ``FitWindow`` and
  ``read_result``.

Helpers (:data:`HELPERS`):

* ``task-only`` — ``main`` without ``delegate`` (``automatic.Delegation`` held back: what
  removing it leaves, the native ``task`` helper only), against ``main`` as ``delegate``; on a
  source without it, the same as ``main`` (the report says which source);
* ``helpers`` and ``helpers-admitted`` — ``ReAct`` with a Deep Agents sub-agent of its own (its
  ``task`` tool, each call a child run), and the same with each ``task`` call admitted by the
  run's gate first (``admission.branch``: :class:`AdmittedTasks`), for the ``helpers`` class.

Reasoning techniques (``benchmark.cost.techniques``), each ``main`` and one candidate:

* ``think``, ``slot-reuse``, ``temp0``, ``serial-tools``, ``anthropic-cache``,
  ``openai-cache-key`` — a native switch or a library's middleware given to ``ReAct``
  (``techniques.TECHNIQUES``);
* ``code`` — ``ReAct(backend=sandbox())``: code run in the run's own sandbox (Deep Agents'
  ``execute``; Docker unless ``DOCKER_HOST`` says), against ``deep`` (the same graph, no
  sandbox) as well as ``main``;
* ``ask-off``, ``ask``, ``ask-large`` — ``main`` without, with and (a large model only) with the
  ``ask_user`` tool (``techniques.ask_user``: the run's own ``ask``), each run's questions
  answered from the task's gold data (``run.answered``), the same for all three; ``hitl`` and
  ``ask+hitl`` — ``ask-off`` and ``ask`` with every approval given by a careful reviewer who
  edits a wrong argument to the gold value, the run going on;
* ``examples`` — ``main`` run for a user of its own whose documents are the reasoning class's
  dev items as solved examples (``techniques.seed_examples``, with ``--memory``): the memory
  service's own retrieval brings the similar ones into each run's context.

:data:`DEFAULT` are the variants a run compares unless ``--variants`` says.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, cast

from langchain.agents.middleware import AgentMiddleware

from trellis import ReAct
from trellis.harness.react import PLACED, defaults_off, defaults_on


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
    from trellis.harness import react
    from trellis.harness.adapters.base import TOOLS_ATTRIBUTE
    from trellis.harness.gateways import routed_chat_model

    graph = create_agent(
        routed_chat_model(setup.model) if isinstance(setup.model, str) else setup.model,
        tools=[],
        system_prompt=setup.system,
        middleware=[
            ours.HarnessTools(),
            *react.limits(max_steps=setup.max_steps, max_repeats=ours.MAX_REPEATS, scoped=True),
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
    on: tuple[str, ...] = (),
    window: int | None = None,
    deep: bool = False,
    loop: bool = False,
    subagents: tuple[dict[str, Any], ...] = (),
    middleware: tuple[Any, ...] = (),
    backend: Any = None,
    options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """``ReAct`` as it is built — ``deep``: on Deep Agents (a backend given: the journal's);
    ``loop``: an arm of the ReAct loop's changes (refused on a source without them);
    ``subagents``, ``middleware``, ``backend``: given to ``ReAct`` as they are; ``options``: the
    arm's (``run.run_cell``: ``user``, ``prepare``)."""
    from trellis.harness import react
    from trellis.harness.automatic import JournalBackend

    if loop and not hasattr(react, "limits"):
        raise RuntimeError(
            "a ReAct-loop arm measures the library's limits (react.limits): run it on a source "
            "that has them; today's ReAct is the main arm"
        )
    named = PIECES | AUTOMATIC | LOOP | HELPERS
    with defaults_off(*(named[name] for name in off)), defaults_on(*on):
        target = ReAct(
            setup.system,
            setup.model,
            max_steps=setup.max_steps,
            context_window=window or setup.window,
            tools=setup.tools,
            backend=backend if backend is not None else JournalBackend() if deep else None,
            subagents=[dict(s) for s in subagents],
            middleware=list(middleware),
        )
    return target, dict(options or {})


def technique(setup: Setup, *, name: str) -> tuple[Any, dict[str, Any]]:
    """``main`` with the technique ``name``'s middleware (``techniques.TECHNIQUES``)."""
    from benchmark.cost import techniques

    return main(setup, middleware=(techniques.TECHNIQUES[name](),))


def ask(setup: Setup, *, offered: str, review: bool = False) -> tuple[Any, dict[str, Any]]:
    """``main`` with the ``ask_user`` tool (``techniques.ask_user``): ``offered`` to every
    model (``all``), to a large one only (``large``: ``automatic.model_tier``), or not at all
    (``none``); every arm's questions answered from the task's gold data (``run.answered``)
    and, with ``review``, its approvals given by a careful reviewer (a wrong argument edited to
    the gold one, the run going on: human in the loop)."""
    from benchmark.cost import techniques
    from trellis.harness.automatic import model_tier

    name = setup.model if isinstance(setup.model, str) else getattr(setup.model, "model_name", None)
    tier = model_tier(name if isinstance(name, str) else None, setup.window)
    given = offered == "all" or (offered == "large" and tier == "large")
    tools = (*setup.tools, techniques.ask_user()) if given else setup.tools
    return main(
        dataclasses.replace(setup, tools=tools), options={"answers": True, "review": review}
    )


def code(setup: Setup) -> tuple[Any, dict[str, Any]]:
    """``ReAct`` whose code runs in the run's own sandbox (``backend=sandbox()``)."""
    from trellis import sandbox

    return main(setup, backend=sandbox())


def examples(setup: Setup) -> tuple[Any, dict[str, Any]]:
    """``main``, run for the worked examples' user, whose documents they are."""
    from benchmark.cost import techniques

    return main(
        setup,
        options={"user": techniques.EXAMPLES_USER, "prepare": techniques.seed_examples},
    )


def helpers(setup: Setup, *, admitted: bool = False) -> tuple[Any, dict[str, Any]]:
    """``ReAct`` with a Deep Agents sub-agent of its own, ``researcher``, given the set's own
    tools (each call through the bridge, in the helper's child run) — ``admitted``: each of its
    ``task`` calls admitted by the run's gate first (:class:`AdmittedTasks`)."""
    from trellis.harness.tools.base import Tool
    from trellis.harness.tools.convert import langchain as converted

    # the set's own tools (a function tool's ``Tool``), not the distractors' source
    found = [getattr(t, "tool", t) for t in setup.tools]
    own = [t for t in found if isinstance(t, Tool)]
    researcher = {
        "name": "researcher",
        "description": "Looks one item up with the tools and answers in one sentence: give it "
        "one item per call, and call it once per item, all at once.",
        "system_prompt": setup.system,
        "tools": converted.convert(own),
    }
    return main(
        setup,
        subagents=(researcher,),
        middleware=(AdmittedTasks(),) if admitted else (),
    )


class AdmittedTasks(AgentMiddleware):
    """Each Deep Agents ``task`` call (a helper's child run) admitted by the run's gate before it
    runs (``admission.branch``: queued, a slot taken under the adaptive limit, given back however
    it ends) — what ``helpers-admitted`` measures; the harness itself does not. Inside
    ``GovernTools``, so it wraps the helper's run (``middleware._helped``'s ``ran``)."""

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        from trellis.harness import admission
        from trellis.harness.adapters.langgraph import TASK_TOOL
        from trellis.harness.runtime import current

        runtime = current()
        call = request.tool_call
        if runtime is None or call["name"] != TASK_TOOL:
            return await handler(request)
        kind = str(call["args"].get("subagent_type", "task"))
        async with admission.branch(runtime, f"{runtime.run_id}:{call['id']}", kind) as found:
            done = await handler(request)
            found.ended("error" if getattr(done, "status", None) == "error" else "success")
            return done


def context_native(setup: Setup) -> tuple[Any, dict[str, Any]]:
    """``create_agent`` with the branch's stack, ``FitWindow``, ``ReadTools`` and the clearing
    of ``read_result`` left out, and LangChain's own in their place: ``ContextEditingMiddleware``
    (``ClearToolUsesEdit``, its own placeholder, at half the window) and
    ``SummarizationMiddleware`` (at 85% of it, the last 20 messages kept). Prepared: measured
    once the loop's arms are decided."""
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from langchain.agents import create_agent
    from langchain.agents.middleware import (
        ClearToolUsesEdit,
        ContextEditingMiddleware,
        SummarizationMiddleware,
    )

    from trellis.harness import middleware as ours
    from trellis.harness import react
    from trellis.harness.adapters.base import TOOLS_ATTRIBUTE
    from trellis.harness.gateways import routed_chat_model

    if setup.window is None:
        raise RuntimeError("context-native needs the model's window (--window)")
    chat = routed_chat_model(setup.model) if isinstance(setup.model, str) else setup.model
    stack = react._middleware(
        max_steps=setup.max_steps,
        max_repeats=ours.MAX_REPEATS,
        call_timeout=None,
        context_window=setup.window,
        system_prompt=None,
        scoped=True,
        retry=isinstance(setup.model, str),
    )
    replaced = {"FitWindow", "ReadTools", "ContextEditingMiddleware"}
    kept = [m for m in stack if m.name not in replaced]
    native = [
        ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=int(0.5 * setup.window))]),
        SummarizationMiddleware(
            chat, trigger=("tokens", int(0.85 * setup.window)), keep=("messages", 20)
        ),
    ]
    graph = create_agent(
        chat,
        tools=[],
        system_prompt=setup.system,
        middleware=cast(list[Any], [PatchToolCallsMiddleware(), *kept[:-1], *native, kept[-1]]),
        checkpointer=ours.RunCheckpointer(),
        name="context-native",
    )
    setattr(graph, TOOLS_ATTRIBUTE, list(setup.tools))  # run by HarnessTools, as ReAct's are
    return graph, {}


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
        scoped=True,
        retry=isinstance(setup.model, str),
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
#: The ReAct loop's pieces an arm turns off, by the name of their middleware: the repeat check,
#: and the errors' change (a tool that raised read as a result; model calls retried).
LOOP: Final = {
    "repeats": "StallGuard",
    "tool_errors": "ToolErrorMiddleware",
    "retries": "ModelRetryMiddleware",
}
#: What ``task-only`` holds back: ``delegate`` (where the source still has it).
HELPERS: Final = {"delegate": "Delegation"}
#: The errors' change, both its pieces.
ERRORS: Final = ("tool_errors", "retries")
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
    "main+selector": Variant(
        "main+selector",
        "main with LangChain's tool selector when no hints and over 20 tools",
        functools.partial(main, on=("ToolSelection",)),
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
    "all": Variant(
        "all",
        "the ReAct loop's changes together: the library's limits with the repeat check, tool "
        "errors as results, model retries where no gateway retries",
        functools.partial(main, loop=True),
    ),
    "limitsA": Variant(
        "limitsA",
        "LangChain's limits only (model calls, the memory search tools' calls)",
        functools.partial(main, loop=True, off=("repeats", *ERRORS)),
    ),
    "limitsB": Variant(
        "limitsB",
        "LangChain's limits and the repeat check (the same call not run again)",
        functools.partial(main, loop=True, off=ERRORS),
    ),
    "errorsA": Variant(
        "errorsA",
        "limitsA and the errors' change (ToolErrorMiddleware, ModelRetryMiddleware)",
        functools.partial(main, loop=True, off=("repeats",)),
    ),
    "context-native": Variant(
        "context-native",
        "LangChain's context editing and summarization in place of FitWindow and read_result",
        context_native,
    ),
    "task-only": Variant(
        "task-only",
        "main without delegate: the native task helper only",
        functools.partial(main, off=("delegate",)),
    ),
    "helpers": Variant(
        "helpers",
        "ReAct with a Deep Agents sub-agent of its own (task calls, each a child run)",
        helpers,
    ),
    "helpers-admitted": Variant(
        "helpers-admitted",
        "helpers, each task call admitted by the run's gate (admission.branch)",
        functools.partial(helpers, admitted=True),
    ),
    **{
        name: Variant(name, description, functools.partial(technique, name=name))
        for name, description in {
            "think": "main, the chat template's thinking switch on every call",
            "slot-reuse": "main, each run on one llama.cpp slot (id_slot, cache_prompt)",
            "temp0": "main, temperature 0 on every call offered tools",
            "serial-tools": "main, parallel tool calls off",
            "anthropic-cache": "main with langchain_anthropic's prompt caching middleware",
            "openai-cache-key": "main with the OpenAI API's prompt_cache_key",
        }.items()
    },
    "code": Variant("code", "ReAct with code run in the run's own sandbox", code),
    "ask-off": Variant(
        "ask-off",
        "main, its questions answered as the ask arms' are (no ask_user)",
        functools.partial(ask, offered="none"),
    ),
    "ask": Variant(
        "ask", "main with ask_user, offered to every model", functools.partial(ask, offered="all")
    ),
    "ask-large": Variant(
        "ask-large",
        "main with ask_user, offered to a large model only",
        functools.partial(ask, offered="large"),
    ),
    "hitl": Variant(
        "hitl",
        "main, its changes approved by a careful reviewer who edits a wrong argument",
        functools.partial(ask, offered="none", review=True),
    ),
    "ask+hitl": Variant(
        "ask+hitl",
        "ask and hitl together",
        functools.partial(ask, offered="all", review=True),
    ),
    "examples": Variant(
        "examples", "main with the reasoning dev items as solved examples in memory", examples
    ),
}
#: The variants a run compares by default.
DEFAULT: Final = ("baseline", "main")
