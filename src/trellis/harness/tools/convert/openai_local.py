"""The OpenAI Agents SDK's tools that run on this machine — a ``LocalShellTool``, a
``ShellTool`` with a local executor, an ``ApplyPatchTool``'s editor, a ``ComputerTool``'s
computer, a ``CustomTool`` — each of their calls run as a harness tool call is
(``tools.bridge.call``): the run's hooks, governance by the tool's risk (as the catalog says;
a tool that declares nothing asks — but a shell, the framework's own, is a write that runs
announced, never asked, as every framework's shell does), a person when it asks — shown the
call as the model made it —, the journal (a resume does not run a call again), the record,
the run's count.

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
from collections.abc import Awaitable, Callable, Collection, Hashable, Sequence
from typing import Any, Final, cast

from trellis.contracts import ToolError, ToolSpec, is_pause_signal
from trellis.harness.runtime import Runtime, current
from trellis.harness.tools import bridge
from trellis.harness.tools.base import Tool, not_run

#: The fields of a call item that say which call it is, not what it does: left out of what a
#: person is shown and what governance decides on (the call's id is the bridge's own).
IDENTITY: Final = frozenset({"id", "call_id", "status", "type"})


@dataclasses.dataclass(slots=True)
class Local:
    """What one attempt keeps of the calls the SDK carries out on this machine: on its
    ``Runtime`` (``Runtime.local_calls``, :func:`_local`), gone with it."""

    #: a pause raised inside a call whose code the SDK runs within a ``try`` of its own (it
    #: turns an exception into a failed output and goes on): raised again from the run's hooks
    #: as the run's next step begins (:func:`paused_inside`)
    paused: BaseException | None = None
    #: the computer calls of the latest model reply not yet carried out, in order
    computer_calls: deque[str] = dataclasses.field(default_factory=deque)
    #: the computer call being carried out (:func:`computer_started`): one model decision
    computer_call: Hashable | None = None
    #: what the model is to be told of computer actions that did not run (:func:`unsaid`)
    unsaid: list[str] = dataclasses.field(default_factory=list)


def _local(runtime: Runtime) -> Local:
    """The attempt's :class:`Local`, made on its first use."""
    if runtime.local_calls is None:
        runtime.local_calls = Local()
    return cast("Local", runtime.local_calls)


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
    if runtime is None:
        return None
    local = _local(runtime)
    paused, local.paused = local.paused, None
    return paused


def run_here(tool: Any) -> Any:
    """``tool`` — one of the SDK's that runs on this machine — as a copy whose calls go through
    the bridge (the module's seam); any other tool as it is."""
    from agents import ApplyPatchTool, ComputerTool, CustomTool, LocalShellTool, ShellTool

    if isinstance(tool, LocalShellTool | ShellTool) and getattr(tool, "executor", None):
        return with_fields(tool, executor=_executor(tool))
    if isinstance(tool, CustomTool):
        return with_fields(tool, on_invoke_tool=_custom(tool))
    if isinstance(tool, ApplyPatchTool):
        return with_fields(tool, editor=_Editor(tool))
    if isinstance(tool, ComputerTool):
        return _computing(tool)
    return tool


def with_fields(tool: Any, **fields: Any) -> Any:
    """A copy of ``tool`` (one of the SDK's tool dataclasses) with ``fields`` changed: made
    again as the SDK makes it (``dataclasses.replace``), or — a subclass whose constructor
    takes fields of its own (a sandbox capability's ``exec_command``, its ``apply_patch``) —
    copied as it is. What the harness marked on the tool itself (not a field: which capability
    made it) stays on the copy."""
    if dataclasses.is_dataclass(type(tool)) and type(tool).__module__ == "agents.tool":
        made = dataclasses.replace(tool, **fields)
        marks = vars(tool).keys() - {f.name for f in dataclasses.fields(tool)}
        for name in marks:
            setattr(made, name, getattr(tool, name))
        return made
    made = copy.copy(tool)
    for name, value in fields.items():
        setattr(made, name, value)
    return made


def spec_of(runtime: Runtime, name: str, **said: Any) -> ToolSpec:
    """The SDK's tool ``name`` as the run's tool set has it, else undeclared: what it ``said``
    of itself (its description, its schema), no risk."""
    return runtime.native.get(name) or ToolSpec(name=name, **said)


@dataclasses.dataclass(slots=True)
class Carried:
    """A call's own code (``run``) as the bridge runs it (``Tool.run``): what it returned kept
    (``live``: as the SDK takes it from the tool itself) and what it raised (``raised``: a
    pause is no failure), for the SDK to be handed once the bridge has governed, journaled
    and recorded the call."""

    run: Callable[[dict[str, Any]], Any]
    live: list[Any] = dataclasses.field(default_factory=list)
    raised: list[Exception] = dataclasses.field(default_factory=list)

    async def __call__(self, given: dict[str, Any]) -> Any:
        try:
            answer = await _awaited(self.run(given))
        except Exception as error:
            if not is_pause_signal(error):
                self.raised.append(error)
            raise
        self.live.append(answer)
        return answer


async def through(
    name: str,
    args: dict[str, Any],
    run: Callable[[dict[str, Any]], Any],
    *,
    call_id: str | None = None,
    decision: Hashable | None = None,
    on_refusal: Callable[[str], Any] = str,
    shell: bool = False,
) -> Any:
    """``run(args)`` — the code that carries out a call of the SDK's tool ``name`` — as the run's
    call: what ``run`` returned (live, as the SDK takes it), the journal's record of it on a
    resume, ``on_refusal`` of what the model reads when the call does not run; what ``run``
    raised is raised again, for the SDK to deal with as it does. Outside a run, ``run(args)``.
    ``decision``: the model decision the call is part of (``bridge.call``: a computer call, for
    each of its actions). ``shell``: the framework's own shell (:func:`_executor`), run as the
    framework runs it — a write unless the catalog says more, announced and recorded, never
    asked (``Tool.confined``), as Claude Code's ``Bash`` and Deep Agents' ``execute`` are.

    A call whose outcome the run's ``after_tool`` hooks changed (``bridge.hooked``) is theirs:
    a result is returned as the journal keeps it (what a resume returns), and a failure is
    raised as the SDK takes a failed call — the error, its class kept, saying the hooks' words
    (:func:`reworded`): the SDK words what the model reads of it from the error itself."""
    runtime = current()
    if runtime is None:
        return await _awaited(run(args))
    carried = Carried(run)
    live, raised = carried.live, carried.raised

    async def ran(given: dict[str, Any]) -> Any:
        return _recorded(await carried(given))

    spec = spec_of(runtime, name)
    if shell and spec.side_effects == "unknown":
        spec = spec.model_copy(update={"side_effects": "write"})
    try:
        outcome = await bridge.call(
            Tool(spec, ran, confined=shell), args, call_id=call_id, decision=decision
        )
    except Exception as error:
        if is_pause_signal(error):  # the SDK may swallow it: its hooks stop the run
            _local(runtime).paused = error
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

        return await through(
            tool.name, args, run, call_id=getattr(data, "call_id", None), shell=True
        )

    return execute


#: The runs whose model was told where the files of the shell the harness gave it go.
_TOLD: Final[weakref.WeakSet[Any]] = weakref.WeakSet()


def given_shell(command: Callable[[str], Awaitable[dict[str, Any]]], says: str) -> Any:
    """The SDK's own ``ShellTool`` on the working folder the harness gives a run: each of the
    model's commands run by ``command`` (``SandboxSource.command``: as ``sandbox_exec`` runs
    it), its result the SDK's own shape. A hosted tool has no description to say where the
    user's files go: ``says`` is told with the run's first result, once."""
    from agents import ShellCallOutcome, ShellCommandOutput, ShellResult, ShellTool

    async def execute(request: Any) -> Any:
        outputs = []
        for asked in request.data.action.commands:
            ran = await command(asked)
            told = [str(ran[k]) for k in ("hint", "outputs") if ran.get(k)]
            runtime = current()
            if runtime is not None and runtime not in _TOLD:
                _TOLD.add(runtime)
                told.append(says)
            stdout = "\n".join(part for part in (str(ran.get("stdout") or ""), *told) if part)
            outcome = ShellCallOutcome(type="exit", exit_code=ran.get("exit_code"))
            outputs.append(
                ShellCommandOutput(
                    stdout=stdout,
                    stderr=str(ran.get("stderr") or ""),
                    outcome=outcome,
                    command=asked,
                )
            )
        return ShellResult(output=outputs)

    return ShellTool(executor=execute)


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

    actions = AsyncComputer.__abstractmethods__  # the interface's actions: nothing else runs
    body: dict[str, Any] = {name: acting(name) for name in sorted(actions)}

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
            edited = {"type": name, **(given.get("action") or {})}
            return _edited_action(self._inner, edited, actions, self._tool.name)

        runtime = current()
        try:
            return await through(
                self._tool.name,
                {"action": action},
                run,
                decision=None if runtime is None else _local(runtime).computer_call,
                on_refusal=_skipped,
            )
        except NotAnAction as refused:  # recorded as failed; nothing of the computer called
            return _skipped(refused.message)

    body.update(
        __init__=__init__,
        _act=_act,
        environment=property(lambda self: self._inner.environment),
        dimensions=property(lambda self: self._inner.dimensions),
    )
    return type("GovernedComputer", (AsyncComputer,), body)


class NotAnAction(ToolError):
    """A computer call edited (a person's edit, a hook's rewrite) into what is no action of the
    SDK's computer interface, or an action with arguments that are not its own: not carried
    out."""


def _edited_action(inner: Any, edited: dict[str, Any], actions: Collection[str], tool: str) -> Any:
    """An edited computer action carried out on ``inner``: only one of the interface's own
    ``actions`` (the SDK's ``AsyncComputer`` abstract methods — never another method or
    attribute of the team's computer object), its arguments bound to that method's signature;
    anything else is :class:`NotAnAction`, raised before the computer is touched."""
    kind = str(edited.pop("type"))
    if kind not in actions:
        raise NotAnAction(not_run(tool, f"{kind!r} is not an action of the computer"))
    method = getattr(inner, kind)
    try:
        bound = inspect.signature(method).bind(**edited)
    except TypeError as exc:
        raise NotAnAction(not_run(tool, f"the arguments of {kind} do not fit ({exc})")) from exc
    return method(*bound.args, **bound.kwargs)


def _skipped(said: str) -> None:
    """A computer action that may not run: nothing done — the model sees the screen as it is,
    and is told why in the input of its next model call (:func:`unsaid`: a computer call's
    output is a screenshot, with nowhere to say it)."""
    runtime = current()
    assert runtime is not None  # a call is refused only inside a run (:func:`through`)
    _local(runtime).unsaid.append(said)


def unsaid(runtime: Runtime) -> list[str]:
    """What the model is to be told of the computer actions that did not run since its last
    model call (why each did not: the bridge's words), once: the adapter tells it in its next
    model call's input (``adapters.openai_agents``)."""
    local = _local(runtime)
    said, local.unsaid = local.unsaid, []
    return said


def computer_replied(runtime: Runtime, items: Sequence[Any]) -> None:
    """A model reply's computer calls (``items``: its output), in order: the SDK carries them
    out one after another, each its actions and then a screenshot (:func:`computer_started`)."""
    from openai.types.responses import ResponseComputerToolCall

    calls = [i.call_id for i in items if isinstance(i, ResponseComputerToolCall)]
    _local(runtime).computer_calls = deque(calls)


def computer_started(runtime: Runtime) -> None:
    """The SDK starts carrying out a computer call (its ``on_tool_start``, before the call's
    actions): its actions are each a call of the computer tool and one model decision — the
    computer call's id (``bridge.call``'s ``decision``): the stall rule reads a batch of three
    ``Down`` presses as one decision, not a model looping, and a person's answer about an
    action covers the same action again in it. A call carried out with no reply read before it
    (a resumed run's) is a decision of its own all the same."""
    local = _local(runtime)
    pending = local.computer_calls
    local.computer_call = ("computer", pending.popleft() if pending else object())


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
