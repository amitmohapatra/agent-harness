"""Commands on this machine are the deployment's to allow, never a run's user's
(``Settings.host_execution``, ``TRELLIS_HOST_EXECUTION``).

The person a run is for decides the actions taken in their name: governance asks them. The
machine the harness runs on is not theirs — a command there reaches the deployment's files,
credentials, network and other users' runs — so whether agents may run commands on it is an
administrator's policy, set once for the deployment. An agent that would run a shell command
here, not in an isolated sandbox, is refused where it is wrapped (``h.wrap``) and where a
run's own ``framework_options`` would make it one (:func:`refuse`): a ``ConfigurationError``
naming the setting and the sandbox to give it instead, before anything runs, unless the
deployment allows it (``TRELLIS_HOST_EXECUTION=allow``). Allowed, governance decides each call
as before — and the user's and tenant's boundaries still hold on the host: commands run through
the isolation layer installed for this machine (:func:`install`: a workspace of the run's own,
an unprivileged uid of its own inside the tenant's boundary, limits, an environment with no
secret, a network policy, clean-up, a reaper, audit), not as the service's own user (with none
installed they would run as that user). A layer
that says it cannot isolate here (it cannot switch uid, say) refuses host execution, naming
why (:data:`UNISOLATED`). The layer is built apart (``trellis.harness.sandbox``'s isolation);
this module is where it is called.

What runs a command here (:func:`found`), by each framework's own word for it:

* Deep Agents' ``execute`` on a ``LocalShellBackend`` (alone, or the default of a
  ``CompositeBackend``, where ``execute`` goes), in a graph of your own, its ``task``
  sub-agents' or ``ReAct(backend=)``;
* LangChain's ``ShellToolMiddleware`` with any execution policy but Docker's or Codex's
  sandbox (its default is ``HostExecutionPolicy``);
* an OpenAI agent's ``LocalShellTool`` or ``ShellTool`` in the SDK's local environment (its
  executor is code on this machine), on the agent or one it reaches (a handoff, an agent
  tool), and the SDK's ``UnixLocalSandboxClient`` (its session, its state) in the run's
  ``run_config``;
* Claude Code with its shell (``Bash``, ``PowerShell``, ``REPL``, ``Monitor`` — its tools
  unless a list leaves them out or ``disallowed_tools`` names them) and no sandbox, or a
  sandbox that lets a command out (``allowUnsandboxedCommands``, true unless set false;
  ``excludedCommands``);
* an ACP agent (Kimi Code) with no ``sandbox``.

An isolated sandbox is never refused: the harness's ``sandbox()``, a Deep Agents sandbox
backend, a hosted client's, LangChain's Docker policy, and a framework's own confinement of
its shell (Claude Code's sandbox with nothing let out, Codex's). Neither is an agent with no
shell, nor code of your own that starts a process (a tool you wrote: you decided it)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

from trellis.contracts import ConfigurationError

if TYPE_CHECKING:
    from trellis.harness.agent import Agent

#: The variable an administrator allows commands on this machine with.
HOST_EXECUTION_ENV: Final = "TRELLIS_HOST_EXECUTION"
#: What an agent that would run commands here is refused with.
REFUSED: Final = (
    "agent {agent!r} would run commands on this machine, outside a sandbox: {found}. This "
    "deployment does not allow it, and a run's user cannot approve it. Give it a sandbox — "
    "{fix} — or, where the deployment's administrator decides that commands may run here, "
    f"set {HOST_EXECUTION_ENV}=allow (Settings(host_execution='allow'))"
)
#: What an agent allowed to run commands here is refused with when the isolation layer cannot
#: isolate them.
UNISOLATED: Final = (
    "agent {agent!r} would run commands on this machine ({found}), which this deployment "
    "allows ({env}=allow), but they cannot be isolated here: {why}. Commands on the host run "
    "only through the isolation layer, never as the service's own user"
).replace("{env}", HOST_EXECUTION_ENV)
#: Claude Code's built-in tools that run a command or code on the machine its CLI runs on.
CLAUDE_SHELLS: Final = frozenset({"Bash", "PowerShell", "REPL", "Monitor"})
#: How the Deep Agents fix reads, for a graph of your own and for ``ReAct``.
DEEP_AGENTS_FIX: Final = (
    "ReAct(backend=sandbox()), or create_deep_agent(**sandbox().deep_agent()), in place of "
    "the LocalShellBackend"
)


@dataclass(frozen=True, slots=True)
class HostShell:
    """One way an agent would run commands on this machine: what it is, and the sandbox to
    give it instead."""

    what: str
    fix: str


class HostIsolation(Protocol):
    """The isolation layer commands on this machine run through (:func:`install`)."""

    def cannot_isolate(self) -> str | None:
        """Why commands cannot be isolated on this machine now (no unprivileged uid to switch
        to, say), else ``None``."""
        ...


#: The isolation layer installed for this machine's commands (:func:`install`), if any.
_isolation: HostIsolation | None = None


def install(layer: HostIsolation | None) -> HostIsolation | None:
    """Make ``layer`` the isolation layer every allowed command on this machine runs through
    (``None``: none); the one it replaces is returned."""
    global _isolation
    replaced, _isolation = _isolation, layer
    return replaced


def refuse(agent: Agent, options: Mapping[str, Any]) -> None:
    """Refuse ``agent`` run with framework ``options`` (its own, or a run's over them) when it
    would run commands on this machine (:func:`found`) and the deployment does not allow it,
    or allows it but the isolation layer cannot isolate them (:data:`UNISOLATED`)."""
    shells = found(agent.adapter.name, agent.target, options)
    if not shells:
        return
    what = "; ".join(dict.fromkeys(s.what for s in shells))
    if agent.harness.settings.host_execution != "allow":
        fix = "; ".join(dict.fromkeys(s.fix for s in shells))
        raise ConfigurationError(REFUSED.format(agent=agent.id, found=what, fix=fix))
    why = None if _isolation is None else _isolation.cannot_isolate()
    if why is not None:
        raise ConfigurationError(UNISOLATED.format(agent=agent.id, found=what, why=why))


def found(adapter: str, target: Any, options: Mapping[str, Any]) -> list[HostShell]:
    """The ways the ``adapter``'s ``target``, run with ``options``, would run commands on this
    machine (the module's list); ``[]`` for a framework that runs none of its own."""
    finder = FINDERS.get(adapter)
    return [] if finder is None else finder(target, options)


def _graph(target: Any, options: Mapping[str, Any]) -> list[HostShell]:
    """A compiled graph's: Deep Agents' ``execute`` on a ``LocalShellBackend``, LangChain's
    ``ShellToolMiddleware`` on the host — in its tool nodes and its sub-agents' graphs."""
    from trellis.harness.adapters.langgraph import graph_tools
    from trellis.harness.middleware import files_backend

    shells: list[HostShell] = []
    for tool in graph_tools(target):
        if tool.name == "execute" and _local_shell(files_backend(tool)):
            shells.append(HostShell("Deep Agents' execute on a LocalShellBackend", DEEP_AGENTS_FIX))
        middleware = _shell_middleware(tool)
        if middleware is not None and not _isolated(middleware):
            shells.append(
                HostShell(
                    "LangChain's ShellToolMiddleware on the host",
                    "ShellToolMiddleware(execution_policy=DockerExecutionPolicy())",
                )
            )
    return shells


def _local_shell(backend: Any) -> bool:
    """Whether Deep Agents' ``execute`` on ``backend`` runs on this machine: a
    ``LocalShellBackend``, or a ``CompositeBackend`` whose default is one."""
    from deepagents.backends import CompositeBackend, LocalShellBackend

    if isinstance(backend, CompositeBackend):
        return _local_shell(backend.default)
    return isinstance(backend, LocalShellBackend)


def _shell_middleware(tool: Any) -> Any | None:
    """The ``ShellToolMiddleware`` whose shell ``tool`` is (its code closes over it)."""
    from langchain.agents.middleware import ShellToolMiddleware

    for cell in getattr(getattr(tool, "func", None), "__closure__", None) or ():
        try:
            held = cell.cell_contents
        except ValueError:  # an empty cell
            continue
        if isinstance(held, ShellToolMiddleware):
            return held
    return None


def _isolated(middleware: Any) -> bool:
    """Whether a ``ShellToolMiddleware``'s policy confines its shell: Docker's, or Codex's
    sandbox; any other (its default ``HostExecutionPolicy``, one of your own) is the host."""
    from langchain.agents.middleware import CodexSandboxExecutionPolicy, DockerExecutionPolicy

    policy = getattr(middleware, "_execution_policy", None)
    return isinstance(policy, DockerExecutionPolicy | CodexSandboxExecutionPolicy)


def _openai(target: Any, options: Mapping[str, Any]) -> list[HostShell]:
    """An OpenAI agent's: a local shell tool on it or on an agent it reaches, the SDK's
    Unix-local sandbox in the run's ``run_config``."""
    from agents import LocalShellTool, ShellTool
    from agents.sandbox.sandboxes.unix_local import (
        UnixLocalSandboxClient,
        UnixLocalSandboxSession,
        UnixLocalSandboxSessionState,
    )

    from trellis.harness.adapters.openai_agents import reachable

    shells: list[HostShell] = []
    for agent in reachable(target):
        for tool in agent.tools:
            local = isinstance(tool, ShellTool) and (tool.environment or {}).get("type") == "local"
            if isinstance(tool, LocalShellTool) or local:
                shells.append(
                    HostShell(
                        f"the OpenAI Agents SDK's {type(tool).__name__} {tool.name!r} of agent "
                        f"{agent.name!r}, run by its executor here",
                        "a ShellTool in a hosted environment (environment={'type': "
                        "'container_auto'}), or a SandboxAgent with a Docker or hosted client",
                    )
                )
    # a run config, and its sandbox, as an object or the dict ``Runner.run`` takes (one with a
    # field the SDK does not have is the SDK's to refuse when the run is made)
    sandbox = _field(options.get("run_config"), "sandbox")
    held = [_field(sandbox, name) for name in ("client", "session", "session_state")]
    local = (UnixLocalSandboxClient, UnixLocalSandboxSession, UnixLocalSandboxSessionState)
    if any(isinstance(given, local) for given in held):
        shells.append(
            HostShell(
                "the OpenAI Agents SDK's UnixLocalSandboxClient (no isolation)",
                "SandboxRunConfig(client=DockerSandboxClient(...)) or a hosted client "
                "(agents.extensions.sandbox)",
            )
        )
    return shells


def _field(given: Any, name: str) -> Any:
    """``given``'s field ``name``: a dict's key, else an object's attribute (``None``: none)."""
    if isinstance(given, Mapping):
        return given.get(name)
    return getattr(given, name, None)


def _claude(target: Any, options: Mapping[str, Any]) -> list[HostShell]:
    """Claude Code's shell with no sandbox, or with a sandbox that lets a command out."""

    def given(name: str) -> Any:
        return options[name] if name in options else getattr(target, name, None)

    tools, denied = given("tools"), set(given("disallowed_tools") or ())
    offered = set(tools) if isinstance(tools, list) else set(CLAUDE_SHELLS)
    if not (offered & CLAUDE_SHELLS) - denied:
        return []
    box = given("sandbox") or {}
    if not box.get("enabled"):
        what = "Claude Code's shell with no sandbox"
    elif box.get("allowUnsandboxedCommands", True):
        what = (
            "Claude Code's sandbox lets a command out (allowUnsandboxedCommands, true unless "
            "set false)"
        )
    elif box.get("excludedCommands"):
        what = (
            f"Claude Code's sandbox runs {', '.join(map(str, box['excludedCommands']))} "
            "outside it (excludedCommands)"
        )
    else:
        return []
    fix = (
        "ClaudeAgentOptions(sandbox={'enabled': True, 'allowUnsandboxedCommands': False}) "
        "with no excludedCommands, or tools without its shell ("
        + ", ".join(sorted(CLAUDE_SHELLS))
        + ")"
    )
    return [HostShell(what, fix)]


def _acp(target: Any, options: Mapping[str, Any]) -> list[HostShell]:
    """An ACP agent's (Kimi Code's) shell with no sandbox: it runs its commands itself."""
    if getattr(target, "sandbox", None) is not None:
        return []
    profile = getattr(target, "profile", None)
    label = getattr(profile, "label", None) or "the ACP agent"
    return [
        HostShell(
            f"{label}'s shell with no sandbox",
            "ACPAgent(sandbox=sandbox()), kimi(sandbox=sandbox())",
        )
    ]


#: Where each framework's ways are found, by its adapter's name.
FINDERS: Final[Mapping[str, Callable[[Any, Mapping[str, Any]], list[HostShell]]]] = {
    "langgraph": _graph,
    "openai_agents": _openai,
    "claude_agent_sdk": _claude,
    "acp": _acp,
}
