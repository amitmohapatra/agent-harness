"""Commands on the host are the deployment's to allow, never the end user's
(``Settings.host_execution``, ``TRELLIS_HOST_EXECUTION``).

An agent that would run a shell command on the machine the harness runs on — not in an
isolated sandbox — is refused where it is wrapped (``h.wrap``), and where a run's own
``framework_options`` would make it one, with a ``ConfigurationError`` naming the setting,
unless the deployment allows it. Every way the harness can reach a host shell: Deep Agents'
``LocalShellBackend`` (alone, or the default of a ``CompositeBackend``; in a graph of your own
or ``ReAct(backend=)``), LangChain's ``ShellToolMiddleware`` on the host, an OpenAI agent's
``LocalShellTool`` and local ``ShellTool`` (on it or on an agent it hands off to) and its
Unix-local sandbox client, Claude Code's shell without its sandbox (or with commands let out
of it), an ACP agent such as Kimi Code without a sandbox. Allowed, each wraps and runs; an
isolated sandbox (Docker, the harness's ``sandbox()``, a hosted one) is never affected."""

from __future__ import annotations

import tempfile
import types
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from claude_agent_sdk import ClaudeAgentOptions
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend, StateBackend

from tests.support.chat_model import ScriptedChatModel
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, ReAct, sandbox
from trellis.contracts import ConfigurationError
from trellis.harness import host
from trellis.harness.adapters.acp import ACPAgent, kimi
from trellis.harness.host import _shell_middleware
from trellis.harness.settings import Settings

#: What every refusal names: the one setting an administrator turns.
NAMED = r"TRELLIS_HOST_EXECUTION=deny"


def folder() -> str:
    return tempfile.mkdtemp(prefix="trellis-host-")


def model() -> ScriptedChatModel:
    return ScriptedChatModel(turns=[])


def deep_local() -> Any:
    return create_deep_agent(model=model(), backend=LocalShellBackend(root_dir=folder()))


def deep_composite() -> Any:
    backend = CompositeBackend(
        default=LocalShellBackend(root_dir=folder()), routes={"/memories/": StateBackend()}
    )
    return create_deep_agent(model=model(), backend=backend)


def react_local() -> Any:
    return ReAct("s", model(), backend=LocalShellBackend(root_dir=folder()))


def react_subagent_local() -> Any:
    """A Deep Agent whose own backend is the graph's state, with a sub-agent of a host shell."""
    helper = create_deep_agent(model=model(), backend=LocalShellBackend(root_dir=folder()))
    spec: Any = {"name": "ops", "description": "runs commands", "runnable": helper}
    return create_deep_agent(model=model(), subagents=[spec])


def shell_middleware() -> Any:
    from langchain.agents import create_agent
    from langchain.agents.middleware import ShellToolMiddleware

    return create_agent(
        model(), tools=[], middleware=[ShellToolMiddleware(workspace_root=folder())]
    )


async def executor(request: Any) -> str:
    return "ran"


def openai_local_shell() -> Any:
    from agents import Agent, LocalShellTool

    return Agent(name="ops", tools=[LocalShellTool(executor=executor)])


def openai_shell() -> Any:
    from agents import Agent, ShellTool

    return Agent(name="ops", tools=[ShellTool(executor=executor)])


def openai_handoff_shell() -> Any:
    from agents import Agent, ShellTool

    ops = Agent(name="ops", tools=[ShellTool(executor=executor)])
    return Agent(name="triage", handoffs=[ops])


def unix_local() -> dict[str, Any]:
    from agents import RunConfig
    from agents.sandbox import SandboxRunConfig
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    return {"run_config": RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()))}


def openai_sandbox_agent() -> Any:
    from agents.sandbox import SandboxAgent

    return SandboxAgent(name="box", instructions="work")


#: Each way to a shell on this machine outside a sandbox that runs by default and that
#: ``deny`` switches off: the target, the framework options it is wrapped with, and what the
#: refusal names besides the setting (the way to a sandbox instead).
HOST_PATHS: dict[str, tuple[Callable[[], Any], Callable[[], dict[str, Any]], str]] = {
    "openai-local-shell-tool": (openai_local_shell, dict, "LocalShellTool"),
    "openai-shell-tool": (openai_shell, dict, "ShellTool"),
    "openai-handoff": (openai_handoff_shell, dict, "ShellTool"),
    "claude-unsandboxed": (ClaudeAgentOptions, dict, "allowUnsandboxedCommands"),
    "claude-unsandboxed-allowed": (
        lambda: ClaudeAgentOptions(sandbox={"enabled": True}),  # Claude's default lets it out
        dict,
        "allowUnsandboxedCommands",
    ),
    "claude-excluded-commands": (
        lambda: ClaudeAgentOptions(
            sandbox={
                "enabled": True,
                "allowUnsandboxedCommands": False,
                "excludedCommands": ["docker"],
            }
        ),
        dict,
        "excludedCommands",
    ),
    "acp-unsandboxed": (lambda: ACPAgent(command=("some-agent", "acp")), dict, "sandbox=sandbox()"),
    "kimi-unsandboxed": (kimi, dict, "sandbox=sandbox()"),
}
#: The ways to a shell on this machine with no isolation the harness could give: refused
#: whatever the setting, naming ``sandbox()``.
UNISOLATED_PATHS: dict[str, tuple[Callable[[], Any], Callable[[], dict[str, Any]], str]] = {
    "deepagents-local-shell": (deep_local, dict, "LocalShellBackend"),
    "deepagents-composite-default": (deep_composite, dict, "LocalShellBackend"),
    "deepagents-subagent": (react_subagent_local, dict, "LocalShellBackend"),
    "react-backend": (react_local, dict, "backend=sandbox()"),
    "langchain-shell-middleware": (shell_middleware, dict, "DockerExecutionPolicy"),
    "openai-unix-local": (openai_sandbox_agent, unix_local, "UnixLocalSandboxClient"),
}


def denying() -> Harness:
    return Harness(config=Settings(host_execution="deny"), memory=False)


def allowing() -> Harness:
    return Harness(config=Settings(), memory=False)


def test_the_setting_allows_by_default_and_the_environment_denies() -> None:
    assert Settings().host_execution == "allow"
    assert Settings.from_env({}).host_execution == "allow"
    assert Settings.from_env({"TRELLIS_HOST_EXECUTION": "deny"}).host_execution == "deny"
    assert Settings.from_env({"TRELLIS_HOST_EXECUTION": "allow"}).host_execution == "allow"
    with pytest.raises(ValueError, match="host_execution"):
        Settings.from_env({"TRELLIS_HOST_EXECUTION": "yes"})


def test_the_old_require_sandbox_setting_is_refused_naming_the_new_one() -> None:
    """``require_sandbox`` asked for what is now the default: given in code it is refused,
    naming the setting that replaces it."""
    with pytest.raises(ConfigurationError, match="host_execution"):
        Settings(require_sandbox=True)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize("way", sorted(HOST_PATHS))
def test_a_host_shell_is_refused_where_the_deployment_denies_naming_the_setting(way: str) -> None:
    target, options, fix = HOST_PATHS[way]
    h = denying()
    with pytest.raises(ConfigurationError, match=NAMED) as refused:
        h.wrap(target(), id=way, framework_options=options())
    assert fix in str(refused.value)
    assert repr(way) in str(refused.value)
    assert way not in h.agents


@pytest.mark.parametrize("way", sorted(HOST_PATHS))
def test_a_host_shell_wraps_by_default(way: str) -> None:
    target, options, _ = HOST_PATHS[way]
    h = allowing()
    h.wrap(target(), id=way, framework_options=options())
    assert way in h.agents


class HostedClient:
    """A sandbox client of a hosted provider's (E2B, Modal...: ``agents.extensions.sandbox``),
    as far as the harness looks: not the SDK's Unix-local one."""


def boxed_openai() -> tuple[Any, dict[str, Any]]:
    from agents import RunConfig
    from agents.sandbox import SandboxRunConfig

    config = RunConfig(sandbox=SandboxRunConfig(client=HostedClient()))  # pyright: ignore[reportArgumentType]
    return openai_sandbox_agent(), {"run_config": config}


def hosted_shell() -> Any:
    from agents import Agent, ShellTool

    return Agent(name="ops", tools=[ShellTool(environment={"type": "container_auto"})])


def docker_middleware() -> Any:
    from langchain.agents import create_agent
    from langchain.agents.middleware import DockerExecutionPolicy, ShellToolMiddleware

    shell = ShellToolMiddleware(workspace_root=folder(), execution_policy=DockerExecutionPolicy())
    return create_agent(model(), tools=[], middleware=[shell])


async def plain(input: str) -> str:
    return input


#: What runs no command on the host: an isolated sandbox, or no shell at all.
NOT_HOST: dict[str, Callable[[], tuple[Any, dict[str, Any]]]] = {
    "react-sandbox": lambda: (ReAct("s", model(), backend=sandbox(FakeSandboxes())), {}),
    "deepagents-sandbox": lambda: (
        create_deep_agent(model=model(), **sandbox(FakeSandboxes()).deep_agent()),
        {},
    ),
    "deepagents-state": lambda: (create_deep_agent(model=model()), {}),
    "deepagents-disk-files": lambda: (
        create_deep_agent(model=model(), backend=FilesystemBackend(root_dir=folder())),
        {},
    ),
    "langchain-docker-shell": lambda: (docker_middleware(), {}),
    "openai-hosted-sandbox": boxed_openai,
    "openai-hosted-shell": lambda: (hosted_shell(), {}),
    "claude-sandboxed": lambda: (
        ClaudeAgentOptions(sandbox={"enabled": True, "allowUnsandboxedCommands": False}),
        {},
    ),
    "claude-no-shell": lambda: (ClaudeAgentOptions(tools=["Read", "Grep", "Glob"]), {}),
    "claude-shell-disallowed": lambda: (
        ClaudeAgentOptions(disallowed_tools=["Bash", "PowerShell", "REPL", "Monitor"]),
        {},
    ),
    "kimi-sandboxed": lambda: (kimi(sandbox=sandbox(FakeSandboxes())), {}),
    "function": lambda: (plain, {}),
}


@pytest.mark.parametrize("way", sorted(NOT_HOST))
def test_a_sandbox_or_no_shell_is_unaffected(way: str) -> None:
    target, options = NOT_HOST[way]()
    h = denying()
    h.wrap(target, id=way, framework_options=options)
    assert way in h.agents


async def test_a_runs_own_options_that_let_a_shell_out_are_refused() -> None:
    """A run's ``framework_options`` go over the agent's: where the deployment denies, a Claude
    run that turns its sandbox off is refused before the run is made (by default it is made);
    an OpenAI run given a Unix-local sandbox is refused whatever the setting."""
    h = denying()
    boxed: Any = {"enabled": True, "allowUnsandboxedCommands": False}
    claude = h.wrap(ClaudeAgentOptions(sandbox=boxed), id="claude")
    with pytest.raises(ConfigurationError, match=NAMED):
        await claude.start("hi", user="u", framework_options={"sandbox": {"enabled": False}})
    target, options = boxed_openai()
    box = h.wrap(target, id="box", framework_options=options)
    with pytest.raises(ConfigurationError, match=r"sandbox\(\)"):
        await box.run("hi", user="u", framework_options=unix_local())

    allowed = allowing()
    claude = allowed.wrap(ClaudeAgentOptions(sandbox=boxed), id="claude")
    run_id = await claude.start("hi", user="u", framework_options={"sandbox": {"enabled": False}})
    assert run_id


def test_a_tool_whose_code_holds_an_empty_cell_is_no_shell() -> None:
    """A graph's tool whose code closes over a name not yet bound (an empty cell) is read past,
    as any tool that is not a ``ShellToolMiddleware``'s shell."""

    def holding() -> Callable[[], int]:
        held = 0

        def code() -> int:
            return held

        return code

    empty = types.FunctionType(holding().__code__, {}, "code", None, (types.CellType(),))
    assert _shell_middleware(SimpleNamespace(func=empty)) is None


class Layer:
    """The isolation layer as the harness calls it (``sandbox.isolation.HostSandbox``): whether
    it can isolate here."""

    def __init__(self, why: str | None = None) -> None:
        self.why = why

    def cannot_isolate(self) -> str | None:
        return self.why


@pytest.fixture
def layer() -> Iterator[Layer]:
    """A layer installed for the test, the one before put back after it."""
    made = Layer()
    before = host.install(cast(Any, made))
    yield made
    host.install(before)


def test_a_layer_that_cannot_isolate_refuses_host_execution_at_start() -> None:
    """Installing the isolation layer refuses at once when it says it cannot isolate here (no
    uid to switch to): never a command as the service's own user."""
    with pytest.raises(ConfigurationError, match="cannot switch to an unprivileged uid"):
        host.install(cast(Any, Layer("cannot switch to an unprivileged uid")))


def test_an_installed_layer_is_the_hosts_and_returns_the_one_it_replaced(layer: Layer) -> None:
    """The layer the deployment installs is ``sandbox()``'s and ``isolated_run``'s
    (``use_host``); installing another returns it."""
    from trellis.harness.sandbox.isolation import host_layer

    assert host_layer() is cast(Any, layer)
    other = Layer()
    assert host.install(cast(Any, other)) is layer
    assert host_layer() is cast(Any, other)


def test_a_run_config_given_as_dicts_is_read_as_the_sdk_takes_it() -> None:
    """``run_config`` — and its sandbox — as the dicts ``Runner.run`` takes: a Unix-local
    client in them is the host as in the SDK's objects; a field the SDK does not have is left
    to the SDK, which refuses it when the run is made."""
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    agent = openai_sandbox_agent()
    local = {"run_config": {"sandbox": {"client": UnixLocalSandboxClient()}}}
    assert [s.what for s in host.found("openai_agents", agent, local)] == [
        "the OpenAI Agents SDK's UnixLocalSandboxClient (no isolation)"
    ]
    assert host.found("openai_agents", agent, {"run_config": {"no_such": 1}}) == []
