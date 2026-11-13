"""The OpenAI Agents SDK's tools that run on this machine — a ``LocalShellTool``, a
``ShellTool`` with a local executor, an ``ApplyPatchTool``'s editor, a ``ComputerTool``'s
computer, a ``CustomTool`` — each of their calls run as a harness tool call is
(``tools.bridge.call``): the run's hooks, governance by the tool's risk (as the catalog says;
a tool that declares nothing asks), a person when it asks — shown the call as the model made
it —, the journal (a resume does not run a call again), the record, the run's count.

One seam for every kind: where the SDK hands a call to the code that carries it out (the
tool's executor, its editor's operation, its computer's action), the copy of the tool the run
gets hands it to the bridge instead, which runs that code when the call may run
(:func:`run_here`). The SDK's own work around the call — its ``needs_approval`` and
``on_approval``, its output items, its hooks — is unchanged: a call its own gate approved is
the approval governance takes for it (``bridge``: one call, one question).

What the model reads of a call that does not run is the bridge's (the approver's reason, the
hook's); of one whose code failed, what the SDK makes of the failure; of a call that paused
the run, nothing — the run stops there (:func:`paused_inside`), as a harness tool's pause
stops it. An edited call runs with the edited arguments. A computer's action that does not
run is skipped — the SDK tells the model of a computer call by the screenshot after it — and
the model is told why in its next model call's input (:func:`unsaid`). The actions of one
computer call are one model decision (:func:`computer_started`)."""

from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
import json
import weakref
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Final

from trellis.contracts import ToolError, ToolSpec
from trellis.contracts.errors import is_pause_signal
from trellis.harness.runtime import Runtime, current
from trellis.harness.tools import bridge
from trellis.harness.tools.base import Tool

#: The fields of a call item that say which call it is, not what it does: left out of what a
#: person is shown and what governance decides on (the call's id is the bridge's own).
IDENTITY: Final = frozenset({"id", "call_id", "status", "type"})

#: A pause raised inside a call whose code the SDK runs within a ``try`` of its own (it turns
#: an exception into a failed output and goes on), by run: raised again from the run's hooks
#: as the run's next step begins (:func:`paused_inside`).
_PAUSED: weakref.WeakKeyDictionary[Runtime, BaseException] = weakref.WeakKeyDictionary()
#: Per run: the computer calls of its latest model reply not yet carried out, in order; the one
#: being carried out (:func:`computer_started`); and what the model is to be told of computer
#: actions that did not run (:func:`unsaid`).
_COMPUTER_CALLS: weakref.WeakKeyDictionary[Runtime, deque[str]] = weakref.WeakKeyDictionary()
_COMPUTER_CALL: weakref.WeakKeyDictionary[Runtime, Hashable] = weakref.WeakKeyDictionary()
_UNSAID: weakref.WeakKeyDictionary[Runtime, list[str]] = weakref.WeakKeyDictionary()


def call_args(raw: Any) -> dict[str, Any]:
    """What a call does, as the model made it (a call item of the SDK's, or the JSON a saved
    run state keeps of one): its fields less :data:`IDENTITY`, with no empty ones."""
    data = raw.model_dump(mode="json", exclude_none=True) if hasattr(raw, "model_dump") else raw
    return {k: v for k, v in _plain(dict(data)).items() if k not in IDENTITY}


def _plain(value: Any) -> Any:
    """``value`` as JSON data with no ``None`` in it."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if v is not None}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def paused_inside(runtime: Runtime | None) -> BaseException | None:
    """The pause a call of a tool the SDK runs here raised inside the SDK's own ``try`` (once):
    the run's hooks raise it, so the run stops there."""
    return _PAUSED.pop(runtime, None) if runtime is not None else None


def run_here(tool: Any) -> Any:
    """``tool`` — one of the SDK's that runs on this machine — as a copy whose calls go through
    the bridge (the module's seam); any other tool as it is."""
    from agents import ApplyPatchTool, ComputerTool, CustomTool, LocalShellTool, ShellTool

    if isinstance(tool, LocalShellTool | ShellTool) and getattr(tool, "executor", None):
        return dataclasses.replace(tool, executor=_executor(tool))
    if isinstance(tool, CustomTool):
        return dataclasses.replace(tool, on_invoke_tool=_custom(tool))
    if isinstance(tool, ApplyPatchTool):
        return dataclasses.replace(tool, editor=_Editor(tool))
    if isinstance(tool, ComputerTool):
        return _computing(tool)
    return tool


async def through(
    name: str,
    args: dict[str, Any],
    run: Callable[[dict[str, Any]], Any],
    *,
    call_id: str | None = None,
    decision: Hashable | None = None,
    on_refusal: Callable[[str], Any] = str,
) -> Any:
    """``run(args)`` — the code that carries out a call of the SDK's tool ``name`` — as the run's
    call: what ``run`` returned (live, as the SDK takes it), the journal's record of it on a
    resume, ``on_refusal`` of what the model reads when the call does not run; what ``run``
    raised is raised again, for the SDK to deal with as it does. Outside a run, ``run(args)``.
    ``decision``: the model decision the call is part of (``bridge.call``: a computer call, for
    each of its actions).

    A call whose outcome the run's ``after_tool`` hooks changed (``bridge.hooked``) is theirs:
    a result is returned as the journal keeps it (what a resume returns), and a failure is
    raised as the SDK takes a failed call — the error, its class kept, saying the hooks' words
    (:func:`reworded`): the SDK words what the model reads of it from the error itself."""
    runtime = current()
    if runtime is None:
        return await _awaited(run(args))
    live: list[Any] = []
    raised: list[Exception] = []

    async def ran(given: dict[str, Any]) -> Any:
        try:
            answer = await _awaited(run(given))
        except Exception as error:
            if not is_pause_signal(error):
                raised.append(error)
            raise
        live.append(answer)
        return _recorded(answer)

    spec = runtime.native.get(name) or ToolSpec(name=name)
    try:
        outcome = await bridge.call(Tool(spec, ran), args, call_id=call_id, decision=decision)
    except Exception as error:
        if is_pause_signal(error):  # the SDK may swallow it: its hooks stop the run
            _PAUSED[runtime] = error
        raise
    if bridge.hooked(outcome):  # the run's after_tool hooks have the last word on the call
        if outcome.ok:
            return outcome.output  # as the journal keeps it, as a resume replays it
        raise reworded(raised[-1] if raised else None, str(outcome.output))
    if live:
        return live[-1]
    if raised:
        raise raised[-1]
    if outcome.ok:
        return outcome.output  # the journal's: run before a resume
    return on_refusal(str(outcome.output))


def reworded(error: Exception | None, said: str) -> Exception:
    """``error`` saying ``said`` in place of its own message: a copy of it — its class kept,
    not its cause or traceback, so the words it had are not carried along — or, when it
    cannot be copied to say exactly that (no error; one whose class words itself), a
    ``ToolError`` saying it."""
    if error is not None:
        try:
            copied = copy.copy(error)
            copied.args = (said,)
        except Exception:  # an error its class cannot rebuild from its arguments
            copied = None
        if copied is not None and str(copied) == said:
            return copied
    return ToolError(said, source="tools")


async def _awaited(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _recorded(answer: Any) -> Any:
    """A call's result as the record and the journal keep it: JSON data, or its text."""
    if answer is None or isinstance(answer, str | int | float | bool | list | dict):
        return answer
    if dataclasses.is_dataclass(answer) and not isinstance(answer, type):
        return _plain(dataclasses.asdict(answer))
    return str(answer)


def _executor(tool: Any) -> Any:
    """A shell's executor (``LocalShellTool``, ``ShellTool``): the request's call as its
    arguments, the request made again from edited ones."""
    executor = tool.executor

    async def execute(request: Any) -> Any:
        data = request.data
        raw = getattr(data, "raw", data)  # a shell's parsed call keeps the model's item
        args = (
            call_args(raw)
            if raw is not None
            else {"action": _plain(dataclasses.asdict(data.action))}
        )

        def run(given: dict[str, Any]) -> Any:
            return executor(request if given == args else _remade(request, given))

        return await through(tool.name, args, run, call_id=getattr(data, "call_id", None))

    return execute


def _remade(request: Any, args: dict[str, Any]) -> Any:
    """A shell's request with the call an approver edited."""
    from agents import ShellActionRequest

    data = request.data
    if hasattr(data, "model_validate"):  # a local shell's: the model's own item
        edited = type(data).model_validate({**data.model_dump(), **args})
        return dataclasses.replace(request, data=edited)
    action = args.get("action") or {}
    fields = {f.name for f in dataclasses.fields(ShellActionRequest)}
    remade = ShellActionRequest(**{k: v for k, v in action.items() if k in fields})
    return dataclasses.replace(request, data=dataclasses.replace(data, action=remade))


def _custom(tool: Any) -> Any:
    """A ``CustomTool``'s invoke: the call's input, as the model wrote it."""
    invoke = tool.on_invoke_tool

    async def run_custom(context: Any, given_input: str) -> Any:
        call = getattr(context, "tool_call", None)
        args = call_args(call) if call is not None else {"name": tool.name, "input": given_input}

        def run(given: dict[str, Any]) -> Any:
            return invoke(context, str(given.get("input", given_input)))

        return await through(tool.name, args, run, call_id=getattr(context, "tool_call_id", None))

    return run_custom


class _Editor:
    """An ``ApplyPatchTool``'s editor whose operations are each a call of the tool: the
    operation (as the model's call item has it) its arguments."""

    def __init__(self, tool: Any) -> None:
        self._tool = tool
        self._editor = tool.editor

    def _operation(self, method: str, operation: Any) -> Any:
        from agents import ApplyPatchOperation

        fields = ("type", "path", "diff", "move_to")
        args = {"operation": _plain({f: getattr(operation, f) for f in fields})}

        def run(given: dict[str, Any]) -> Any:
            edited = given.get("operation") or {}
            ctx = operation.ctx_wrapper
            changed = ApplyPatchOperation(**{**edited, "ctx_wrapper": ctx})
            return getattr(self._editor, method)(operation if given == args else changed)

        return through(self._tool.name, args, run)

    def create_file(self, operation: Any) -> Any:
        return self._operation("create_file", operation)

    def update_file(self, operation: Any) -> Any:
        return self._operation("update_file", operation)

    def delete_file(self, operation: Any) -> Any:
        return self._operation("delete_file", operation)


def _computing(tool: Any) -> Any:
    """A ``ComputerTool`` whose computer (given, or made per run by its provider) acts through
    the bridge, one call per action the model asked for."""
    from agents import ComputerProvider

    given = tool.computer
    maker = given.create if isinstance(given, ComputerProvider) else given

    async def made(**kw: Any) -> Any:  # a computer made per run: governed once made
        return computing(await _awaited(maker(**kw)), tool)

    if isinstance(given, ComputerProvider):
        computer: Any = dataclasses.replace(given, create=made)
    elif callable(given) and not hasattr(given, "screenshot"):
        computer = made
    else:
        computer = computing(given, tool)
    return dataclasses.replace(tool, computer=computer)


def computing(inner: Any, tool: Any) -> Any:
    """``inner`` (the SDK's ``Computer`` or ``AsyncComputer``) as an ``AsyncComputer`` whose
    every action — each method of the SDK's computer interface — is a call of ``tool``:
    ``{"action": {"type": <the action>, <its arguments>}}``, skipped when it may not run.
    What the SDK reads of the computer besides (its environment, its size) is the inner's."""
    return _computer_class()(inner, tool)


@functools.cache
def _computer_class() -> type:
    from agents import AsyncComputer

    def acting(name: str) -> Any:
        async def act(self: Any, *args: Any, **kwargs: Any) -> Any:
            return await self._act(name, args, kwargs)

        act.__name__ = name
        return act

    actions = sorted(AsyncComputer.__abstractmethods__)
    body: dict[str, Any] = {name: acting(name) for name in actions}

    def __init__(self: Any, inner: Any, tool: Any) -> None:
        self._inner, self._tool = inner, tool

    async def _act(self: Any, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        method = getattr(self._inner, name)
        accepted = inspect.signature(method).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        bound = inspect.signature(method).bind_partial(*args, **kwargs)
        action = {"type": name, **_plain(json.loads(json.dumps(bound.arguments, default=str)))}

        def run(given: dict[str, Any]) -> Any:
            if given == {"action": action}:
                return method(*args, **kwargs)
            edited = dict(given.get("action") or {})
            return getattr(self._inner, str(edited.pop("type", name)))(**edited)

        runtime = current()
        return await through(
            self._tool.name,
            {"action": action},
            run,
            decision=None if runtime is None else _COMPUTER_CALL.get(runtime),
            on_refusal=_skipped,
        )

    body.update(
        __init__=__init__,
        _act=_act,
        environment=property(lambda self: self._inner.environment),
        dimensions=property(lambda self: self._inner.dimensions),
    )
    return type("GovernedComputer", (AsyncComputer,), body)


def _skipped(said: str) -> None:
    """A computer action that may not run: nothing done — the model sees the screen as it is,
    and is told why in the input of its next model call (:func:`unsaid`: a computer call's
    output is a screenshot, with nowhere to say it)."""
    runtime = current()
    if runtime is not None:
        _UNSAID.setdefault(runtime, []).append(said)


def unsaid(runtime: Runtime) -> list[str]:
    """What the model is to be told of the computer actions that did not run since its last
    model call (why each did not: the bridge's words), once: the adapter tells it in its next
    model call's input (``adapters.openai_agents``)."""
    return _UNSAID.pop(runtime, [])


def computer_replied(runtime: Runtime, items: Sequence[Any]) -> None:
    """A model reply's computer calls (``items``: its output), in order: the SDK carries them
    out one after another, each its actions and then a screenshot (:func:`computer_started`)."""
    from openai.types.responses import ResponseComputerToolCall

    calls = [i.call_id for i in items if isinstance(i, ResponseComputerToolCall)]
    _COMPUTER_CALLS[runtime] = deque(calls)


def computer_started(runtime: Runtime) -> None:
    """The SDK starts carrying out a computer call (its ``on_tool_start``, before the call's
    actions): its actions are each a call of the computer tool and one model decision — the
    computer call's id (``bridge.call``'s ``decision``): the stall rule reads a batch of three
    ``Down`` presses as one decision, not a model looping, and a person's answer about an
    action covers the same action again in it. A call carried out with no reply read before it
    (a resumed run's) is a decision of its own all the same."""
    pending = _COMPUTER_CALLS.get(runtime)
    _COMPUTER_CALL[runtime] = ("computer", pending.popleft() if pending else object())


__all__ = [
    "call_args",
    "computer_replied",
    "computer_started",
    "computing",
    "paused_inside",
    "reworded",
    "run_here",
    "through",
    "unsaid",
]
