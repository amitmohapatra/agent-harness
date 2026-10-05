# Sandboxes: where the model runs code

`sandbox()` gives an agent three tools — run a shell command, read a file, write a file — that
work in a sandbox of its run's own: a container (or a hosted microVM) made at the run's first
sandbox call and deleted when the run ends. Every call is a harness tool call, so it is
governed, journaled, recorded, redacted and bounded like any other, in every framework. You
choose only where sandboxes come from (`SANDBOX=docker`, or a provider object) and, if you
like, what each is made of; the rest is automatic.

```python
from trellis import Harness, ReAct, sandbox

h = Harness()  # SANDBOX=docker
agent = h.wrap(ReAct(system="You analyse data.", model=model), id="analyst", tools=[sandbox()])
result = await agent.run("How many units did we sell?", user="ada")
```

| You write | Automatic |
|---|---|
| `tools=[sandbox()]` (or `h.tools(sandbox(), framework=...)`) and `SANDBOX=docker` | one sandbox per run: made at its first call, named and labelled after the run, recorded in its journal before any call uses it, attached again after a pause or a crash, snapshotted and paused while a person answers, deleted at the end; orphans reaped |
| optionally `sandbox(provider, SandboxSpec(...), timeout=)` | the provider (else `SANDBOX`), the image, CPU, memory, network and starting files (else the provider's image, its limits, no network), the most one call may take (else 120 s) |

## What

The tools, all three taking and returning text:

| Tool | Arguments | Returns | Side effects |
|---|---|---|---|
| `sandbox_exec` | `command` | `{"exit_code", "stdout", "stderr"}` — `sh -c` in the sandbox's working directory; a non-zero exit is a result, not an error | `write` |
| `sandbox_read` | `path` (relative to the working directory, or absolute) | the file, as text (UTF-8, invalid bytes replaced); at most 256 KiB (`READ_BYTES`): a larger file is read in parts with a command | `read` |
| `sandbox_write` | `path`, `content` | `"wrote N bytes to <path>"`; its directories are made | `write`, idempotent (the same content at the same path has one effect) |

Behind them, `trellis.harness.sandbox` is a provider-neutral interface
([`base.py`](../src/trellis/harness/sandbox/base.py)): a `SandboxProvider` (`create(ref, spec)`,
`attach(ref)`, `delete(ref)`, `labelled(labels)`) and the `Sandbox` it gives (`exec(command,
*, timeout, env)`, `read(path)`, `write(path, data)`), with optional capabilities a provider
has or not (`SupportsPause`, `SupportsSnapshot`). `DockerSandbox` is the provider this
repository ships: containers of a Docker daemon, through its Engine API.

## When

* The model should compute rather than guess: analyse a file, run a calculation, transform data,
  try code and read the error.
* Code the model writes must not run on your host, and must not reach the network.
* Several steps work on the same files, across a pause for a person and a worker's crash.

Not for calling an API or a service: that is a tool (`@tool`, `openapi`, an MCP server), which
governance sees by name and arguments. A sandbox command is opaque text; govern it by what it
can reach (no network) and, when needed, approve it ([governance](#automatic)).

## Where

Way 1, every adapter: `tools=[sandbox()]` for a function, `ReAct`, an OpenAI Agents `Agent` or
`ClaudeAgentOptions`; `await h.tools(sandbox(), framework="langgraph" | "deepagents" |
"openai_agents" | "claude_agent_sdk")` for a graph built with its tools (and an OpenAI Agents
handoff's specialist). Every mode: `run`, `stream`, `start` + workers, a resume in another
process, `serve_chat`, `serve_a2a`, `h.evaluate` — a resume elsewhere reaches the same sandbox
when its provider is reachable from there (a hosted provider; for Docker, the same daemon).
Way 2: the provider is a block of its own ([below](#way-2-without-a-harness)).

## How

```python
from trellis import sandbox
from trellis.harness.sandbox import DockerSandbox, SandboxSpec

# the image (else the provider's: SANDBOX_IMAGE for the deployment's), at most 1 core and
# 512 MiB, no network (the default; "open", or the hosts it may reach where the provider can
# enforce that), and a file written before its first command
spec = SandboxSpec(
    image="python:3.12-slim", cpu=1, memory=512, network="none", files={"sales.csv": csv_text}
)
tools = [sandbox(DockerSandbox(), spec, timeout=300)]
```

| Name | |
|---|---|
| `sandbox(provider=None, spec=None, *, timeout=120)` | the tool source. `provider`: where sandboxes come from (else the deployment's: `SANDBOX`); `spec`: what each is made of; `timeout`: the most one call may take, in seconds (`REMOTE_TIMEOUT_SECONDS`, the one default of a remote tool) |
| `SandboxSpec(image=None, cpu=None, memory=None, network="none", files={})` | frozen; `cpu` and `memory` over 0 |
| `DockerSandbox(image=None, *, socket="/var/run/docker.sock")` | the Docker provider: `image` for a spec that names none (else `python:3.12-slim`), `socket` the daemon's |
| `configured(settings)` | the deployment's provider: `SANDBOX=docker` is `DockerSandbox(SANDBOX_IMAGE)` |
| `await reap(provider, runs, *, tenant)` | deletes the tenant's sandboxes whose runs have ended (and returns them): what runs it automatically does by hand, for a cron |
| `SandboxRef(provider, id, labels, snapshot)` | which sandbox, as a run's journal keeps it: never a secret or a host path |

The environment: `SANDBOX=docker` names the deployment's provider and `SANDBOX_IMAGE` its image
([configuration.md](configuration.md)); nothing else is configured.

## Automatic

**Its life is the run's.**

* **Made** at the run's first sandbox call — one, however many calls come at once — named
  `trellis-<run id>` (a run id no provider takes as a name: its digest) and labelled with the
  run, its tenant and its agent (`trellis.run_id`, `trellis.tenant`, `trellis.agent_id`).
  `create` adopts a sandbox of that name, so the attempt after a crash between making and
  recording it works in the same one.
* **Recorded** in the run's journal (`Journal.sandbox`) and saved as its progress as soon as it
  exists, before any call uses it. A sub-agent's run has its own.
* **Attached** by every later attempt — after a pause, after a crash, on another worker — never
  replaced blindly: one that is gone is made again from the snapshot of the run's last pause
  while no call has changed it since; otherwise every call says it is lost.
* **Snapshotted, then paused** when the run pauses for a person, as far as the provider can
  (Docker: a commit of its filesystem, its processes frozen); the snapshot is named in the
  journal saved with the pause; the first call that changes the sandbox after the resume
  forgets it.
* **Deleted** when the run ends, whichever way (success, error, time out, cancel — a queued or
  paused run cancelled included), with its snapshots. A queued run that failed with an error
  agent-runs runs it again after keeps it: its next attempt goes on in it.
* **Reaped**: a process deletes, in the background, a tenant's sandboxes whose runs have ended
  — what a process that died before deleting its own left — when it first makes a sandbox for
  that tenant, then at most every 10 minutes (`REAP_SECONDS`).

The run's stream says so: a `sandbox` custom event with `action` `created`, `attached`,
`restored` or `paused` and the sandbox's name; a `warning` with code `sandbox` when it could
not be paused.

**Every call is a tool call** ([tools.md](tools.md#every-call)):

* *governed* — `sandbox_exec` and `sandbox_write` write (announced: `tool_notice`),
  `sandbox_read` reads; the catalog overrides it like any tool's: `risk: irreversible` on
  `sandbox_exec` asks a person before every command, an `approve_when` on its `command`
  argument before some ([governance.md](governance.md));
* *journaled* — a resumed run reads every command's recorded output and runs nothing again;
* *recorded and redacted* — the arguments and the output on the run's stream and in the memory
  service's tool records pass the redactor (a credential a command printed is
  `[redacted]` there); the model gets them as they are;
* *bounded* — at most the source's `timeout` and what is left of the run's time; a command past
  it is **killed, with everything it started** (its process group), and the model reads
  `"sandbox_exec timed out after 120s; it may or may not have taken effect: check before
  calling it again"` ([unknown outcomes](reliability.md#unknown-outcomes));
* *retried* — a read after an error that may pass (the daemon unreachable), within its time; a
  write too (it is idempotent); a command never.

| What happened | What the next call finds |
|---|---|
| the run paused for a person, and is answered | the same sandbox, resumed |
| the worker died after the sandbox was made, before it was recorded | the same sandbox, adopted by name |
| the worker died while a command ran | the command's effect is unknown (`"sandbox_exec was interrupted by a crash; …"`); it is not run again |
| the worker died between calls | the same sandbox, attached; the calls made replay from the journal |
| the sandbox is gone, and nothing changed it since the run's last pause | the sandbox made again from that pause's snapshot |
| the sandbox is gone otherwise (or the deployment's provider changed) | `"<tool> failed: the sandbox trellis-… is gone"` on every call: the model, or a person, decides |

**Docker.** A container named after the run, kept alive doing nothing between commands, made
of the spec's image (pulled when the daemon lacks it); no network (`--network none`) unless the
spec says `open`; every capability dropped, no new privileges, at most 512 processes, the
spec's CPU and memory as its limits, an init process that reaps what a killed command leaves.
A command runs as `sh -c` in `/workspace`; its output is kept to its last 256 KiB per stream
(the rest cut, and said). A snapshot is a commit (`trellis-snapshot:<name>`, labelled with the
sandbox it was taken of: the filesystem, not the processes or memory); a pause is Docker's.

## Way 2: without a harness

The provider is a block of its own: make a sandbox, and govern its commands like any tool of
yours ([blocks/governance.md](blocks/governance.md)) — you own its life (delete it when your
run ends) and its replay (your framework's checkpoint).

```python
from trellis.harness.governance import Governance, governed
from trellis.harness.sandbox import DockerSandbox, SandboxRef, SandboxSpec

provider = DockerSandbox()
box = await provider.create(SandboxRef(provider="docker", id=f"mine-{run_id}"), SandboxSpec())


async def sandbox_exec(command: str) -> dict:
    return (await box.exec(command, timeout=60, env={"PIP_INDEX_URL": index})).model_dump()


run_code = governed(sandbox_exec, Governance.from_env(), on_ask=ask_a_person)
...
await provider.delete(box.ref)
```

`env=` is per command (a short-lived credential goes there, never into the spec: a spec and a
ref travel in checkpoints and snapshots). Within a wrapped agent, `h.tools(sandbox(...),
framework=...)` gives a framework's own agent the same three tools, every call still the
harness's.

## On failure

* No provider (`sandbox()` without one, `SANDBOX` unset): each call is an error the model reads,
  saying so. `SANDBOX` other than `docker`: `Settings` refuses it (`ValidationError`).
* A spec the provider cannot enforce — a list of hosts for Docker — is refused when the sandbox
  is made (`ConfigurationError`, the model reads it), never weakened.
* An image that cannot be pulled, a daemon that refuses: the call's error (`"Docker answered
  500: …"`; an error of the daemon itself may pass, and a read is retried).
* The sandbox lost (deleted, expired, on a daemon this worker cannot reach): `SandboxLost`, the
  call's error, on every call of the run — never a blank replacement.
* A snapshot or pause that fails: a warning; the run pauses all the same. A delete that fails:
  logged; the reaper deletes the sandbox later.
* A file larger than 256 KiB, or not a file: the read's error, saying how to read it in parts.

## Native sandboxes: theirs or ours

Some frameworks bring a sandbox of their own. Their tools are the framework's, so they do not
pass the harness's bridge: no harness governance, journal, timeouts, records or lifecycle.

| Framework | Its own | Use it when | Use `sandbox()` when |
|---|---|---|---|
| Deep Agents | backends (`StateBackend`, `FilesystemBackend`, sandbox backends for Daytona, Modal, Runloop...) behind its file tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) and `execute` | its file-editing tools are what the agent needs, its `interrupt_on` is your gate, and a checkpointer resumes the graph in place | commands must be governed by the catalog, journaled (a re-run does not repeat them), bounded and killed, recorded, and deleted with the run |
| OpenAI Agents SDK | `SandboxAgent` with `SandboxRunConfig(client=...)` (Unix-local, Docker, E2B, Modal, Daytona...): a manifest, mounts, `apply_patch`, its session state in the `RunState` | you build on its sandbox capabilities (manifest, mounts, memory) and its own approvals | the same sandbox semantics for every framework of the deployment, under the harness's governance and journal |
| Claude Agent SDK | `ClaudeAgentOptions(sandbox=SandboxSettings(...))`: Claude's own `Bash` and file tools confined on the machine the CLI runs on (bubblewrap, Seatbelt; a network allowlist through a local proxy); and Anthropic's hosted code execution tool (no internet, container reuse) | Claude should use its built-in tools on a workspace, and the machine is the boundary you trust | each run needs a container of its own, the calls governed, journaled and recorded, and the sandbox deleted with the run |

The two can coexist (a framework's sandbox for its own tools, `sandbox()` for the harness's),
but they are two sandboxes that do not share files; prefer one.

## Other providers: E2B, Daytona, Modal

Only Docker is implemented here: a hosted provider's client is a new dependency and needs an
account key, so none ships with this repository. One plugs in by implementing the interface —
nothing in the harness changes:

| Provider method | E2B | Daytona | Modal |
|---|---|---|---|
| `name` | `"e2b"` | `"daytona"` | `"modal"` |
| `create(ref, spec)`: idempotent by `ref.id` | `Sandbox.create(template, metadata=ref.labels, …)` after `list` with the run's metadata finds none | `create` with `ref.id` as name and the labels, or the existing one | `Sandbox.create(name=ref.id, …)` and `set_tags(ref.labels)` after `from_name` finds none |
| from `ref.snapshot` | `create` from the snapshot's id | `create` from the snapshot | `create(image=snapshot)` |
| `attach(ref)`: resume a paused one, `SandboxLost` when gone | `Sandbox.connect(id)` (it resumes a paused one) | `start` a stopped one | `from_name` |
| `delete(ref)`: and its snapshots | `kill`, then delete the snapshots | `delete` | `terminate` |
| `labelled(labels)` | `list` filtered by metadata | `list` by labels | `list(tags=…)` |
| `exec(command, timeout=, env=)` | `commands.run(cmd, timeout=, envs=)`; `commands.kill(pid)` when cancelled | `process.exec(cmd, timeout=, env=)` | `exec("sh", "-c", cmd, timeout=, env=)` |
| `read` / `write` | `files.read` / `files.write` | `fs.download_file` / `fs.upload_file` | `filesystem.read_bytes` / `write_bytes` |
| `SupportsPause` | `pause()` | `stop()` | — |
| `SupportsSnapshot` | `create_snapshot()` | experimental | `snapshot_filesystem()` |
| `network` hosts | `network={"allow_out": [...]}` | CIDRs only: refuse hosts | `outbound_domain_allowlist` |

Its `SandboxRef` holds the sandbox's id and the labels only; the key stays in the provider
object (from its own environment variable). `examples/sandbox.py` has a provider of its own (a
temporary directory, with no isolation) that shows the whole interface working.

## Limits

* One sandbox per run: a thread's next run gets a new one.
* Docker: one daemon — a run resumed by a worker that reaches another daemon finds its sandbox
  lost; a shared kernel — for code nobody reviewed, run the daemon with gVisor (`runsc`) or use
  a microVM provider; no list of reachable hosts; a snapshot keeps the filesystem, not the
  processes.
* The reaper runs where sandboxes are made (and by hand: `reap`); a sandbox whose run the store
  does not know — runs kept in a process that is gone, without `RUNS_URL` — is left alone.
  A paused sub-agent's run cancelled with its parent, or a paused run ended by agent-runs'
  ticker, has its sandbox reaped rather than deleted at once.
* A sandbox only an OpenAI Agents handoff's specialist uses (its tools from `h.tools`, not the
  wrapped agent's own) is neither paused nor deleted by the run: the reaper deletes it.
* Files are text to the tools (`sandbox_read`, `sandbox_write`); binary data moves with a
  command (`base64`).

## Run it

* [`examples/sandbox.py`](../examples/sandbox.py) — a `ReAct` analyst writing and running a
  script in its sandbox (Docker with `SANDBOX=docker`, else a provider of the example's own),
  then the provider governed without a harness (Way 2).
* Tests: `tests/integration/test_sandbox.py` (every adapter, calls at once, a pause, a lost
  sandbox, crashes, a timeout, governance, redaction, every ending, the reaper),
  `tests/unit/test_docker_sandbox.py` (the Engine API calls), and against the real daemon
  `tests/live/test_live_sandbox.py`.
