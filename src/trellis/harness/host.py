"""Shell commands on this machine: allowed by default
(``Settings.host_execution``, ``TRELLIS_HOST_EXECUTION``: ``allow``, or ``deny``, the
administrator's switch that turns them off).

A framework's own shell runs as the framework runs it — on this machine, as the service's
user, unless the agent is given a sandbox — and every call of it is the run's: governed,
announced and recorded as the framework's tools are. What runs a command here (:func:`found`),
by each framework's own word for it:

* **Claude Code's and Kimi Code's own shell** (Claude's ``Bash``, ``PowerShell``, ``REPL``,
  ``Monitor``; Kimi's ``Bash`` with no ``sandbox=``), in their working folder. A team's own
  Claude ``SandboxSettings`` is honoured as given; ``kimi(sandbox=sandbox())`` isolates Kimi's.
* **Deep Agents' ``execute`` on a ``LocalShellBackend``** (alone, a ``CompositeBackend``'s
  default, a sub-agent's; ``ReAct(backend=)`` too).
* **LangChain's ``ShellToolMiddleware`` on the host**.
* **An OpenAI agent's ``LocalShellTool`` or local ``ShellTool``** (its executor: code of yours
  on this machine), **the SDK's ``UnixLocalSandboxClient``**.

An agent with none of these gets the harness's own, with nothing to write
(``sandbox.code_for``: a working folder of the run's own on this machine, with no isolation —
``ReAct`` through Deep Agents' ``execute``, any other as ``sandbox_exec``).

``deny`` refuses an agent with a shell of its own at ``h.wrap``, and a run whose own
``framework_options`` would make one, naming the setting; the harness's own is not given
there. Isolated sandboxes are never refused: ``sandbox()``, a Deep Agents sandbox backend, a
hosted client, LangChain's Docker or Codex policy, Kimi with a sandbox. The deployment may
install the isolation layer it configured (:func:`install`: a ``HostSandbox`` with a root
helper, limits, network), refused at start when it cannot isolate; ``sandbox()`` and
``isolated_run`` then use it."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from trellis.contracts import ConfigurationError

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.sandbox.isolation import HostSandbox

#: The variable an administrator switches shell commands on this machine off with.
HOST_EXECUTION_ENV: Final = "TRELLIS_HOST_EXECUTION"
#: What an agent that would run commands here is refused with where the deployment says deny.
REFUSED: Final = (
    "agent {agent!r} would run commands on this machine: {found}. This deployment switched "
    f"them off ({HOST_EXECUTION_ENV}=deny), and a run's user cannot approve them. Give it a "
    "sandbox — {fix}"
)
#: What installing an isolation layer that cannot isolate here is refused with.
UNISOLATED: Final = (
    "commands on this machine cannot be isolated here: {why}. The harness never runs them as "
    "the service's own user"
)
#: Claude Code's built-in tools that run a command or code on the machine its CLI runs on.
CLAUDE_SHELLS: Final = frozenset({"Bash", "PowerShell", "REPL", "Monitor"})
#: How the Deep Agents fix reads, for a graph of your own and for ``ReAct``.
DEEP_AGENTS_FIX: Final = (
    "ReAct(backend=sandbox()), or create_deep_agent(**sandbox().deep_agent()), in place of "
    "the LocalShellBackend"
)
#: What a way to a shell is: a framework's own (Claude Code's, Kimi Code's, Deep Agents',
#: LangChain's, the OpenAI SDK's), or an executor of yours — each runs as it does unless the
#: deployment says ``deny``.
Kind = Literal["framework", "yours"]


@dataclass(frozen=True, slots=True)
class HostShell:
    """One way an agent would run commands on this machine: what it is, the sandbox to give it
    instead, and its kind."""

    what: str
    fix: str
    kind: Kind = "framework"


#: The isolation layer the deployment installed (:func:`install`), if any.
_isolation: HostSandbox | None = None


def install(layer: HostSandbox | None) -> HostSandbox | None:
    """Make ``layer`` this machine's isolation layer — ``sandbox()``'s and ``isolated_run``'s
    (``sandbox.isolation.use_host``; ``None``: the default one) — refused at start
    (:data:`UNISOLATED`) when it says it cannot isolate here. The layer it replaces is
    returned."""
    from trellis.harness.sandbox.isolation import use_host

    why = None if layer is None else layer.cannot_isolate()
    if why is not None:
        raise ConfigurationError(UNISOLATED.format(why=why))
    global _isolation
    use_host(layer)
    replaced, _isolation = _isolation, layer
    return replaced


def refuse(agent: Agent, options: Mapping[str, Any]) -> None:
    """Refuse ``agent`` run with framework ``options`` (its own, or a run's over them) when it
    would run commands here with a shell of its own and the deployment says ``deny``
    (:data:`REFUSED`)."""
    shells = found(agent.adapter.name, agent.target, options)
    if shells and agent.harness.settings.host_execution == "deny":
        raise ConfigurationError(REFUSED.format(agent=agent.id, **_said(shells)))


def _said(shells: list[HostShell]) -> dict[str, str]:
    return {
        "found": "; ".join(dict.fromkeys(s.what for s in shells)),
        "fix": "; ".join(dict.fromkeys(s.fix for s in shells)),
    }


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
                    "framework",
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
                        "yours",
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
                "framework",
            )
        )
    return shells


def _field(given: Any, name: str) -> Any:
    """``given``'s field ``name``: a dict's key, else an object's attribute (``None``: none)."""
    if isinstance(given, Mapping):
        return given.get(name)
    return getattr(given, name, None)


def _claude(target: Any, options: Mapping[str, Any]) -> list[HostShell]:
    """Claude Code's own shell: it runs as it does (its own ``SandboxSettings``, as given)."""

    def given(name: str) -> Any:
        return options[name] if name in options else getattr(target, name, None)

    tools, denied = given("tools"), set(given("disallowed_tools") or ())
    offered = set(tools) if isinstance(tools, list) else set(CLAUDE_SHELLS)
    if not (offered & CLAUDE_SHELLS) - denied:
        return []
    box = given("sandbox") or {}
    confined = box.get("enabled") and box.get("allowUnsandboxedCommands") is False
    if confined and not box.get("excludedCommands"):
        return []
    fix = (
        "ClaudeAgentOptions(sandbox={'enabled': True, 'allowUnsandboxedCommands': False}) "
        "with no excludedCommands, or tools without its shell ("
        + ", ".join(sorted(CLAUDE_SHELLS))
        + ")"
    )
    return [HostShell("Claude Code's own shell, as the service's user", fix, "framework")]


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
            "framework",
        )
    ]


#: Where each framework's ways are found, by its adapter's name.
FINDERS: Final[Mapping[str, Callable[[Any, Mapping[str, Any]], list[HostShell]]]] = {
    "langgraph": _graph,
    "openai_agents": _openai,
    "claude_agent_sdk": _claude,
    "acp": _acp,
}
