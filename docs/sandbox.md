# Sandboxes: where the model runs code

A sandbox is where a run's commands run, away from the service: by default a **workspace of
the run's own, with a uid of its own, in its tenant's shared container** ([the shared
sandbox](#the-shared-sandbox-one-per-tenant-isolated-per-run)), made at the run's first command
and deleted when the run ends; or a container (or a hosted microVM) of the run's own, from a
provider you give. One entry, `sandbox()`, given where each engine takes it — one
sandbox behind every way:

| Engine | Where `sandbox()` goes |
|---|---|
| `ReAct` | `ReAct(backend=sandbox())`: Deep Agents' `execute` runs a command there, and its file tools (`ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`) work on its files |
| Deep Agents (your own graph) | `create_deep_agent(model, **sandbox().deep_agent())`: the same, as its `backend=` |
| plain function, LangGraph | `sandbox().tools()`: three tools — run a shell command, read a file, write a file — `await agent.tools.use(sandbox().tools())` |
| OpenAI Agents SDK, Claude Agent SDK | their own sandbox ([below](#native-sandboxes-theirs-or-ours)); `sandbox().tools()` only where every command must be the harness's (governed by the catalog, journaled) |
| Kimi Code | `kimi(sandbox=sandbox())`: Kimi's `Bash` and file writes go to the run's sandbox through ACP's terminal and file capabilities. Its reads are **not** isolated: Kimi's `Read`, `Glob` and `Grep` read the host ([frameworks/kimi.md](frameworks/kimi.md#native-or-ours-sandbox-skills-hooks)) |

Every call is a harness tool call, so it is governed, journaled, recorded, redacted and bounded
like any other. You choose only where sandboxes come from (Docker, or a provider object) and, if
you like, what each is made of; the rest is automatic.

```python
from trellis import Harness, ReAct, sandbox

h = Harness()  # sandbox(): Docker, the daemon DOCKER_HOST names
agent = h.wrap(
    ReAct(system_prompt="You analyse data.", model=model, backend=sandbox()), id="analyst"
)
result = await agent.run("How many units did we sell?", user="ada")
```

| You write | Automatic |
|---|---|
| `ReAct(backend=sandbox())`, `create_deep_agent(**sandbox().deep_agent())`, or a function target's `agent.tools.use(sandbox().tools())` | one sandbox per run: made at its first call, named and labelled after the run, recorded in its journal before any call uses it, attached again after a pause or a crash, snapshotted and paused while a person answers, deleted at the end; orphans reaped |
| optionally `sandbox(provider, image=, spec=SandboxSpec(...), call_timeout=)` | the provider (else Docker), the image, CPU, memory, network and starting files (else the provider's image, its limits, no network), the most one call may take (else 120 s) |

## What

As a backend (`backend=sandbox()`, `sandbox().deep_agent()`), Deep Agents' `BaseSandbox` builds every file tool on three
operations the run's sandbox does — a command, a file written, a file read — and `execute` is a
command: governed as a write, announced as the harness's own `sandbox_exec` is (the file tools
that change files are writes that ask, as on any real backend, the others read; the catalog may
say more), its output read by the model. The harness runs the graph
asynchronously: the backend has no synchronous operations.

As tools (`sandbox().tools()`), all three taking and returning text:

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

Way 1: `ReAct(backend=sandbox())` (or a `create_deep_agent(**sandbox().deep_agent())` of your
own), and a function target's `agent.tools.call(sandbox().tools(), ...)` (or
`agent.tools.use(sandbox().tools())`). `ReAct(tools=[sandbox()])` is refused, naming the backend,
and so is a bare `sandbox()` given as a tool (`agent.tools.use(sandbox())`), naming `.tools()`.
A framework's own agent uses its own sandbox (below). Every
mode: `run`, `stream`, `start` + workers, a resume in another
process, `serve_chat`, `serve_a2a`, `h.evaluate` — a resume elsewhere reaches the same sandbox
when its provider is reachable from there (a hosted provider; for Docker, the same daemon).
Way 2: the provider is a block of its own ([below](#way-2-without-a-harness)).

## How

```python
from trellis import sandbox
from trellis.harness.sandbox import DockerSandbox, SandboxSpec

# the image (else the provider's own), at most 1 core and
# 512 MiB, no network (the default; "open", or the hosts it may reach where the provider can
# enforce that), and a file written before its first command
spec = SandboxSpec(
    image="python:3.12-slim", cpu=1, memory=512, network="none", files={"sales.csv": csv_text}
)
box = sandbox(DockerSandbox(), spec=spec, call_timeout=300)
# ReAct(..., backend=box) · create_deep_agent(..., **box.deep_agent()) · agent.tools.use(box.tools())
```

| Name | |
|---|---|
| `sandbox(provider=None, *, image=None, spec=None, call_timeout=120)` | the run's sandbox (`SandboxSource`): `ReAct(backend=)` takes it as it is; `.deep_agent()` is `create_deep_agent`'s `{"backend": ...}`; `.tools()` the three tools, a tool source (the same object every time). `provider`: where sandboxes come from (else `DockerSandbox(image)`); `image`: what each is made of (else the spec's, else the provider's); `spec`: the rest of what each is made of; `call_timeout`: the most one call may take, in seconds (`REMOTE_TIMEOUT_SECONDS`, the one default of a remote tool) |
| `SandboxSpec(image=None, cpu=None, memory=None, network="none", files={})` | frozen; `cpu` and `memory` over 0 |
| `DockerSandbox(image=None, *, host=None, socket=None)` | the Docker provider: `image` for a spec that names none (else `python:3.12-slim`); the daemon at `host` (`unix:///path` or `tcp://host:port`) or `socket`, else where docker-py would find it: `DOCKER_HOST`, else the current docker context's endpoint, else `/var/run/docker.sock` |
| `await reap(provider, runs, *, tenant)` | deletes the tenant's sandboxes whose runs have ended (and returns them): what runs it automatically does by hand, for a cron |
| `SandboxRef(provider, id, labels, snapshot)` | which sandbox, as a run's journal keeps it: never a secret or a host path |

Nothing is configured in the environment: `sandbox()` is Docker (the daemon `DOCKER_HOST`
names, Docker's own variable) — the tenant's shared container (`SharedDockerSandbox`, below);
`sandbox(DockerSandbox())` a container of each run's own — and `sandbox(image=)` its image.

## The shared sandbox: one per tenant, isolated per run

Commands run in a **shared** sandbox, not a container per user: one container per tenant
(`SharedDockerSandbox`, `sandbox()`'s default), and inside it the harness enforces the user and
run boundaries itself, through one isolation layer
([`isolation.py`](../src/trellis/harness/sandbox/isolation.py)) — the same layer commands on the
host itself go through when an administrator allows them ([`isolated_run`](#commands-on-the-host-the-same-boundaries)).

**One container per tenant, never shared across tenants.** Its name is a digest of the tenant
and of what it is made of (image, network, CPU, memory: `trellis-pool-<digest>`), and it is
labelled with the tenant (`trellis.pool`). Two tenants never get the same container, so
nothing one tenant's code does — a kernel-level trick, a file left in a shared folder, a
process — reaches another's; the per-run boundaries below are the second wall, inside one
tenant. The container stays for the tenant's next run (no start-up per run); its CPU and memory
(`SandboxSpec(cpu=, memory=)`) are the tenant's together. Each tenant also gets a range of uids
of its own (a digest of the tenant), so no two tenants' runs share a uid on the machine.

**Inside it, each run:**

| Boundary | How |
|---|---|
| a workspace of its own | `/workspaces/t-<tenant digest>/trellis-<run id>`, mode 0700, owned by the run's uid; the tenant's folder 0711 (no listing) |
| an OS identity of its own | an unprivileged uid per run, from the tenant's range, made on demand and freed at the run's end (no `/etc/passwd` entry: a number) |
| cannot read, list or write another run's workspace | its owner and mode; `cat`, `ls` and a write into another workspace fail (`Permission denied`) |
| cannot see or signal another run's processes | each command runs in a process namespace of its own with a `/proc` of its own (`ps` shows its own processes only), as another uid (`kill` fails) |
| cannot read the service's files | the layer's registry (`/workspaces/.trellis`: uids, labels, pid files) is root's alone (0700); every filesystem but the workspace is read-only to the command; on the host, the service's working folder, its home, `/root` and `/home` are covered (`hidden=`) |
| no secret | a command's environment is `PATH`, `HOME` and `TMPDIR` (in its workspace), `LANG`, `USER` and what its caller passes (Kimi's terminal `env`, Way 2's `env=`) — never the service's (`env -i`); the container never receives the service's environment, keys or credentials |
| network | the container's: none unless the spec opens it (`SandboxSpec(network="open")`); a list of hosts is refused (Docker cannot enforce one) |
| limits | per command (`Limits`, below); a runaway command is stopped and its result says which limit (`limit`, and a `hint`); other runs go on |
| `/tmp`, `/var/tmp`, `/dev/shm` | the workspace's own `.tmp` (nothing left behind in a shared folder) |
| nothing outlives its command | a command's process namespace goes with it: a background process (`cmd &`) ends when the command does |

**Limits** (`Limits`, given to the provider: `SharedDockerSandbox(limits=Limits(...))`,
`HostSandbox(limits=...)`, or per call to `isolated_run`):

| Setting | Default | What it bounds |
|---|---|---|
| `cpu_seconds` | 120 | CPU time of each process (`limit: "cpu"`, exit 152) |
| `memory_mib` | 2048 | address space of each process (`limit: "memory"`: `MemoryError`, `Cannot allocate memory`) |
| `processes` | 256 | processes of the run's uid at once, every command of the run together (`limit: "processes"`) |
| `file_mib` | 1024 | the largest file a command writes (`limit: "file"`, exit 153) |
| `open_files` | 1024 | files one process has open (`limit: "files"`) |
| wall-clock | the call's timeout | `sandbox(call_timeout=)` (120 s) and what is left of the run's time: past it the command is killed with everything it started |

The tenant's container: at most 4096 processes (`POOL_PIDS`), the spec's CPU and memory. A
process the kernel kills for memory is the largest of the tenant's (each command's
`oom_score_adj` is 1000): the runaway one, not the layer.

**Clean-up, right after each run.** When a run ends — success, error, time out, cancel, a run
taken over after its worker crashed — what must outlive it is kept first, as without the
shared sandbox: each file in `outputs/` is delivered to the user (the file store) when the run
succeeds, and the results its conversation carries stay in the run's journal (`tools.results`),
written into the next turn's fresh workspace at the path the model was told. Then its
processes are killed, its workspace deleted and its uid freed. A run taken over after a crash
kills what the dead attempt left running when it attaches (the same workspace, the same uid).
**The reaper** removes the workspaces (and processes) of runs that ended without it — a
process that died — when a process first makes a sandbox for a tenant, then at most every 10
minutes (`REAP_SECONDS`), and by hand: `reap(provider, runs, tenant=)`.

**Audit.** Every command is recorded: the run's stream gets a `sandbox_command` event (`user`,
`tenant`, `run`, `sandbox`, `uid`, `command`, `exit_code`, `duration_ms`, `limit`), the log
`trellis.sandbox.audit` the same as one JSON line, and the command's result (journaled with the
call) its `duration_ms` and `limit`. No new store.

**What the container holds.** The layer's own root commands need to switch uid, make the
command's namespaces and mounts, own and remove a run's files and kill its processes, so the
container keeps `SYS_ADMIN`, `SETUID`, `SETGID`, `SETPCAP`, `CHOWN`, `DAC_OVERRIDE`, `FOWNER`
and `KILL` of its capabilities (every other dropped, no new privileges). A run's command has
none (`setpriv --inh-caps=-all --bounding-set=-all --no-new-privs`). It needs util-linux
(`unshare`, `setpriv`, `prlimit`) in the image (Debian's and Ubuntu's have it; Alpine's does
not). Docker's default seccomp profile lets a process with `SYS_ADMIN` among the container's
capabilities make a user namespace: a run's command can make one of its own (it gains nothing
over another run's files or processes, but it is kernel surface) — for code nobody reviewed,
run the daemon with gVisor (`runsc`).

**Not kept from a container of its own:** a pause or snapshot (a workspace lost with its
container is `SandboxLost`), and a background process across commands.

### Commands on the host: the same boundaries

When an administrator allows commands on the server itself (the `-hostshell` setting), they go
through the same layer: `HostSandbox` and its one entry point —

```python
from trellis.harness.sandbox import HostSandbox, Limits, isolated_end, isolated_run

result = await isolated_run(tenant, user, run_id, "python3 report.py", Limits(cpu_seconds=30))
# result.exit_code, .stdout, .stderr, .duration_ms, .limit
await isolated_end(run_id)  # outside a harness run; inside one, the run's end does it
```

The host's workspaces are under `HostSandbox(root=)` (`/var/lib/trellis/workspaces`): a base
folder and a uid range per tenant, a workspace (0700) and uid per run, the same limits, an
environment of its own, the network off (a network namespace of its own; `HostSandbox(network=
"open")` is the administrator's to give), the service's working folder, home, `/root` and
`/home` covered (`hidden=`), clean-up at the run's end and the reaper, the same audit. It needs
root: the service itself, or a helper that runs a command as root (`HostSandbox(helper=("sudo",
"-n"))`). **Without it, host execution refuses to start** (`cannot_isolate()` says why;
`isolated_run` raises `ConfigurationError`), never running a command as the service's own user.
Kept results are bound at `/large_tool_results` (an empty mount point the layer makes once).

### Where the harness cannot enforce it

| Adapter | Commands through the layer |
|---|---|
| `ReAct(backend=sandbox())`, `create_deep_agent(**sandbox().deep_agent())` | yes (`execute` and the file tools) |
| a function or LangGraph agent with `sandbox().tools()`, any framework given those tools | yes |
| Kimi Code, `kimi(sandbox=sandbox())` | its `Bash` and file writes: yes (its working folder is the workspace); its own reads are not isolated (they read the host) |
| Claude Agent SDK's own `Bash` (its `SandboxSettings`) | no: the CLI's own sandbox on the machine it runs on; give it `sandbox().tools()` for commands the layer runs |
| OpenAI Agents SDK's `SandboxAgent` and its clients | no: the SDK's own sandbox (Docker, Unix-local, hosted); give it a hosted or Docker client, or `sandbox().tools()` |
| Deep Agents with a backend of its own (`LocalShellBackend`) | no: refused unless the administrator allows host commands |

## The image: build in the libraries the agent needs

A sandbox has no network by default, so nothing is installed while a run is going. The
libraries the agent's code needs are built into the image beforehand, from your package index
(a Nexus or JFrog proxy of PyPI), by the names pip and uv read: `PIP_INDEX_URL`, `UV_INDEX_URL`.

```dockerfile
FROM python:3.12-slim
# A URL without credentials can be a build argument. A URL with credentials goes in a build
# secret, since build arguments stay in the image's history.
ARG PIP_INDEX_URL
ARG UV_INDEX_URL
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=secret,id=PIP_INDEX_URL,env=PIP_INDEX_URL \
    pip install --no-cache-dir --disable-pip-version-check -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
```

```bash
docker build --secret id=PIP_INDEX_URL,env=PIP_INDEX_URL -t acme/analyst-sandbox:1 .
```

Then name the image in code: `sandbox(image="acme/analyst-sandbox:1")`.

**The environment card.** The model is told what the image holds before it writes code: the
sandbox's command tool (`sandbox_exec`, or Deep Agents' `execute` under `ReAct(backend=sandbox())`)
is described with the image's Python version and installed distributions, and with no network,
that nothing else can be installed. For example:

```text
Its environment: Python 3.12.16; installed: et_xmlfile 2.0.0, openpyxl 3.1.5. It has no network: nothing else can be installed.
```

The card is read once per process, in a sandbox made only for it and deleted at once (about half a
second on Docker). It names at most 80 distributions and counts the rest. An image without Python
is said to have none. When the card cannot be read (no daemon), the tool keeps its plain
description, and a warning is logged. A missing module that a command still imports is
explained by its hint (see [On failure](#on-failure)).

## Files for the user: `outputs/`

The command tool tells the model that the files it writes in `outputs/` (in the working
directory) are given to the user. When the run succeeds, each file directly in `outputs/` is kept
in the file store (`TRELLIS_FILES_URL`, [configuration.md](configuration.md)) before the sandbox
is deleted. The file is named in the run's journal and announced with the `file` event, and
`serve_chat` serves it to the run's user ([surfaces.md](surfaces.md)). A file the store cannot
take is left behind and reported in a warning: a name that is not one safe path part, or a file
over 50 MiB. A run that fails keeps none.

A file that is not the user's is known at once, with the call that wrote it, before the answer
says it was saved:

- A command that names an `outputs` folder is told what the kept folder holds and where it is
  ("In /workspace/outputs/ for the user: report.csv (85731 bytes)." or "No file is in
  /workspace/outputs/ yet: only a file written directly there is given to the user."). Another
  `outputs` (the root's `/outputs/`) is not the kept one.
- A file written by its path (`write_file`, `sandbox_write`) is told the same when it is
  directly in the kept folder, and otherwise that it is not the user's, by its path ("Not in
  /workspace/outputs/, so not given to the user: /outputs/chart.svg.").
- A command that left a new file directly in the working directory, beside the kept folder, is
  told so, once per file ("Not in /workspace/outputs/, so not given to the user: chart.png.").
  A file the sandbox started with is not named, nor one already told of when it was written
  by its path.

It costs one listing in the sandbox after each command of the model's (and one before its
first), and one after a file written into the kept folder.

**The run's data, the query and the sandbox.** A tool result too large to show whole is kept
by the run, as without a sandbox: the model is shown a preview with the result's file path and,
for rows, the table `query_result` reads them from, so a count, a sum, a filter, a ranking or a
join is one exact query. The sandbox gets the same file at that path before the next command.
A query's own result of more than one row is kept the same way and is a CSV file in the
sandbox, named in the query's answer: what code makes of it (a chart, a formatted workbook, a
script) starts from that file. Both tools' descriptions say which does what; the model chooses.
The same question gives the same figure with or without a sandbox.

Claude Code and Kimi Code work in a folder of their own on this machine (their `cwd`) — their
shells unsandboxed only where the deployment allows commands on this machine
(`TRELLIS_HOST_EXECUTION=allow`, [governance.md](governance.md#commands-on-the-host-the-deployments-to-allow)). Its
`outputs/` is the user's the same way (`sandbox.delivered`, one implementation): Claude is told
so after its system prompt, Kimi in a block before its prompt — "Files you write in outputs/ of
your working directory are given to the user when you finish. Write a file the user names, or
gives a path or folder for, exactly there (a bare name is in your working directory); use
outputs/ for a file you make to give the user that they named no place for." — so a file the
user names (`hello.txt`: the working folder) is written where they said, as the SDK's own
`Write` does without the harness; when the run succeeds each file
directly in that `outputs/` is kept, announced and served as above, then removed from the folder
(so a later run does not give it again); the rest of the folder is left alone. Runs that share a
working folder share its `outputs/`: give each user's runs a folder of their own. A sandboxed
Kimi run writes in its sandbox instead.

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
  paused run cancelled included), with its snapshots and the anonymous volumes its image
  declares (Docker: `VOLUME`). A queued run that failed with an error
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
* *recorded* — the arguments and the output on the run's stream and in the memory service's
  tool records, as they are; the command's span masks credentials, and an `on_output` hook
  masks what a user sees ([hooks.md](hooks.md#mask-what-the-ui-sees));
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


run_code = governed(sandbox_exec, Governance.from_env(), side_effects="write", on_ask=ask_a_person)
...
await provider.delete(box.ref)
```

`env=` is per command (a short-lived credential goes there, never into the spec: a spec and a
ref travel in checkpoints and snapshots).

## On failure

* A command that fails says, beside its output, what the sandbox knows about why that the output
  does not (`hint`: a field of `sandbox_exec`'s result, the last line of `execute`'s output):
  killed (exit 137) or Python's `MemoryError` — the sandbox's memory limit it went past; Python's
  `ModuleNotFoundError` — whether the module can be installed there (no network: it cannot). Any
  other failure has no hint: its output says all there is.
* No Docker daemon reachable (`sandbox()` without a provider): each call is an error the model
  reads, saying so.
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

Prefer the framework's own sandbox on the OpenAI Agents SDK, Deep Agents and the Claude Agent
SDK: its tools are the ones the framework's model is built to use. `sandbox()` is for `ReAct`,
plain-function and LangGraph agents, which have none, and wherever every command must be
governed by the catalog and journaled (a replay does not run it again). A framework's own tools
do not pass the harness's bridge: no catalog, journal, timeout, record or lifecycle of the
harness's, except where the framework asks permission or gives a seam for its tools (below:
an OpenAI `SandboxAgent`'s capability tools are governed there).

| Framework | Its own (in the installed package) | Under `h.wrap` |
|---|---|---|
| Deep Agents | a backend behind its file tools and `execute` (`create_deep_agent(backend=)`): `execute` runs commands when the backend implements `SandboxBackendProtocol` — `BaseSandbox` (`deepagents/backends/sandbox.py`: a subclass implements `execute` and `upload_files`), `LangSmithSandbox`, and `LocalShellBackend` (`deepagents/backends/local_shell.py`: the host's shell, with no isolation: refused at `h.wrap` unless `TRELLIS_HOST_EXECUTION=allow`) | the graph runs as it is; gate `execute` with `interrupt_on`, which the harness turns into an approval ([frameworks/deepagents.md](frameworks/deepagents.md)) |
| OpenAI Agents SDK | `SandboxAgent` (`agents/sandbox`) with capabilities `Filesystem`, `Shell` and `Compaction` by default, `Skills` and `Memory` on request; a session from `RunConfig(sandbox=SandboxRunConfig(client=...))`: `DockerSandboxClient`, `UnixLocalSandboxClient` (`agents/sandbox/sandboxes`: no OS isolation on Linux; refused, as a `LocalShellTool` or a local `ShellTool` is, unless `TRELLIS_HOST_EXECUTION=allow`), and hosted clients under `agents.extensions.sandbox` (E2B, Modal, Daytona, Runloop, Vercel, Blaxel, Cloudflare) | the SDK refuses a `SandboxAgent` without a `RunConfig(sandbox=...)` (`UserError`): give it as `framework_options={"run_config": ...}` (below), or run it with `Runner.run` yourself and plug the blocks in ([blocks/openai-agents.md](blocks/openai-agents.md)). Its capabilities' tools (`exec_command`, `write_stdin`, `view_image`, `apply_patch`, `load_skill`) are governed and recorded like the agent's own function tools; a live `SandboxRunConfig(session=...)` on `h.wrap` (one sandbox for every user) is refused: give it per run, one per conversation |
| Claude Agent SDK | the CLI's `Bash` and file tools (`ClaudeAgentOptions(tools=[...])`), confined by `ClaudeAgentOptions(sandbox=SandboxSettings(enabled=True, allowUnsandboxedCommands=False))` on the machine the CLI runs on (macOS and Linux; a `network` setting of its own); unconfined, or with a command let out (`allowUnsandboxedCommands` unset is true, `excludedCommands`), refused at `h.wrap` unless `TRELLIS_HOST_EXECUTION=allow` | every built-in call goes through the harness's `PreToolUse` hook (hooks, governance by risk), then Claude Code's own check; what it asks about, the harness's `can_use_tool` answers (a read runs, any other asks a person) ([frameworks/claude-agent-sdk.md](frameworks/claude-agent-sdk.md)). With the sandbox on, a sandboxed `Bash` command runs unasked, announced and recorded — the CLI approves it itself (`autoAllowBashIfSandboxed`, true by default), and when it asks, the harness lets it through: a command in the sandbox is the deployment's, not the user's to answer ([governance.md](governance.md#who-decides-what)) |

The two can coexist (a framework's sandbox for its own tools, `sandbox()` for the harness's),
but they are two sandboxes that do not share files; prefer one.

Every way to a shell on this machine rather than in a sandbox — the `LocalShellBackend`, the
`UnixLocalSandboxClient`, an unconfined Claude Code or Kimi Code above, LangChain's
`ShellToolMiddleware` on the host — is refused at `h.wrap` unless the deployment's
administrator allows it (`TRELLIS_HOST_EXECUTION=allow`): a run's user cannot approve a command
on the host ([governance.md](governance.md#commands-on-the-host-the-deployments-to-allow)).

A framework's own sandbox runs under `h.wrap` too: its configuration is the framework's run
options. An OpenAI `SandboxAgent` needs `RunConfig(sandbox=...)` on `Runner.run`, so give it as
`framework_options` on `h.wrap` — an object, kept in code and built again in every process
([openai-agents.md](frameworks/openai-agents.md#using-an-existing-openai-agents-project)):

```python
from agents import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

# on this machine, with no isolation: only with TRELLIS_HOST_EXECUTION=allow (a Docker or a
# hosted client otherwise)
box = RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()))
agent = h.wrap(
    SandboxAgent(name="analyst", instructions="..."),
    id="analyst",
    framework_options={"run_config": box},
)
```

Claude's `sandbox=SandboxSettings(...)` is a `ClaudeAgentOptions` field: on the options you
wrap, or a run's own `framework_options={"sandbox": ...}`.

## Other providers: E2B, Daytona, Modal

*A design note: none of these providers is built.* Only Docker is implemented here: a hosted provider's client is a new dependency and needs an
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
object (from its own environment variable). [`examples/05_features/sandbox.py`](../examples/05_features/sandbox.py) has a provider of its own (a
temporary directory, with no isolation) that shows the whole interface working.

## Limits

* One sandbox per run (a workspace in the shared container, by default): a thread's next run
  gets a new one.
* Docker: one daemon — a run resumed by a worker that reaches another daemon finds its sandbox
  lost; a shared kernel — for code nobody reviewed, run the daemon with gVisor (`runsc`) or use
  a microVM provider; no list of reachable hosts; a snapshot keeps the filesystem, not the
  processes.
* The reaper runs where sandboxes are made (and by hand: `reap`); a sandbox whose run the store
  does not know — runs kept in a process that is gone, without `RUNS_URL` — is left alone.
  A paused sub-agent's run cancelled with its parent, or a paused run ended by agent-runs'
  ticker, has its sandbox reaped rather than deleted at once.
* Files are text to the tools (`sandbox_read`, `sandbox_write`); binary data moves with a
  command (`base64`).

## Run it

* [`examples/05_features/sandbox.py`](../examples/05_features/sandbox.py) — a `ReAct` analyst writing and running a
  script in its sandbox (Docker with `DOCKER_HOST`, else a provider of the example's own),
  then the provider governed without a harness (Way 2).
* Tests: `tests/integration/test_sandbox.py` (every adapter, calls at once, a pause, a lost
  sandbox, crashes, a timeout, governance, redaction, every ending, the reaper),
  `tests/unit/test_docker_sandbox.py` (the Engine API calls), and against the real daemon
  `tests/live/test_live_sandbox.py` and `tests/live/test_live_sandbox_data.py` (a data image,
  [`tests/live/sandbox-data`](../tests/live/sandbox-data/Dockerfile): code over a kept 50k-row
  result, an Excel workbook and a PNG chart, a missing module, two users at once).
