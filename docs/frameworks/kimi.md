# Kimi Code (over ACP)

The target is `kimi(...)`: Kimi Code's own agent (`kimi acp`), which the harness drives over the
[Agent Client Protocol](https://agentclientprotocol.com) (ACP), the open protocol editors use to
drive coding agents. Kimi keeps everything it does on its own: its tools, its sub-agents and
swarms, plan mode, goals, compaction. The harness adds memory, governance, durable runs,
records, budgets and its tools.

**Why ACP.** Moonshot archived the Python Kimi CLI (`kimi-cli`: its final release, 1.52.0, runs
nothing but a notice) and the Python `kimi-agent-sdk` that wrapped it. Their successor, Kimi
Code CLI (`@moonshot-ai/kimi-code`, TypeScript), offers no SDK of its own; its programmatic
surface is `kimi acp`, ACP over stdio. The harness speaks it through the official Python SDK
(`agent-client-protocol`). The adapter is a generic ACP client (`trellis.harness.adapters.acp`,
`ACPAgent`); Kimi Code is the one ACP agent this release configures, pins and tests.

**Install:**

```bash
pip install 'trellis-harness[kimi]'            # the ACP SDK and the MCP SDK
npm install -g @moonshot-ai/kimi-code@2.1.1    # the kimi program (Node 22.19 or later)
```

The `kimi` program is checked when a run starts: a version outside the tested range
(`>=2.1.1,<2.2`, [versioning.md](../versioning.md)) fails the run, saying which version answered
and how to install the tested one. Kimi's protocol surface (its tool names, its permission
requests, its usage log) is what the adapter reads, and nothing else pins it.

## Using Kimi Code

```python
from trellis import Harness, kimi

h = Harness()
agent = h.wrap(kimi(model="moonshot/kimi-k2", cwd="/srv/repo"), id="maintainer")

result = await agent.run("The build is red: find out why and fix it.", user="ada")
if result.interrupt:  # Kimi asked to run a command: Bash is irreversible, a person approves
    result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="ada")
```

The harness's tools — the MCP tools the Bifrost virtual key allows and, memory on, the memory
tools — reach Kimi as one more MCP server (below); your own MCP servers go in `mcp_servers=`.

`kimi(*, model=None, cwd=None, env=None, mcp_servers=(), mode=None, sandbox=None, skills=None,
command="kimi")`:

| Argument | What it is |
|---|---|
| `model` | the model's name on the gateway: with Bifrost serving the harness, Kimi's model calls go through Bifrost's OpenAI route with the virtual key (below). Without Bifrost, Kimi calls the model it is configured with: its login (`kimi login`) or `KIMI_MODEL_*` in `env` |
| `cwd` | where Kimi works (this process's working directory by default) |
| `env` | over this process's environment, for the `kimi` process (`KIMI_MODEL_BASE_URL`, `KIMI_MODEL_API_KEY`, `KIMI_LOOP_MAX_STEPS_PER_TURN`...) |
| `mcp_servers` | Kimi's own MCP servers, as ACP takes them: `{"name", "command", "args", "env"}` (stdio) or `{"type": "http", "name", "url", "headers"}`, beside the harness's |
| `mode` | Kimi's session mode: `"default"` or `"plan"`. `"auto"` and `"yolo"` are refused: they approve on their own, and governance would see no call |
| `sandbox` | `sandbox()`: Kimi's `Bash` and file writes go to the run's sandbox (below) |
| `skills` | a skill set, `**src.kimi()`: the run's skills in Kimi's own skills folder (below) |

## What is automatic

| | |
|---|---|
| Memory push | the context goes before the prompt, in the same `session/prompt` (ACP has no system message: Kimi keeps its own system prompt) |
| Harness tools | one more MCP server, `trellis` (`mcp__trellis__<tool>`), served for the attempt on `127.0.0.1` with a bearer token of its own, handed to Kimi in `session/new` beside its own servers; the tool hints narrow what it lists for the run; `read_file` pages a large result |
| Kimi's own tools | every call Kimi asks permission for (`Bash`, `Write`, `Edit`... in its default mode) is decided as a harness call is: the run's `before_tool` hooks, governance by its risk, a person when it asks. `Read`, `Glob`, `Grep`, `FetchURL` read; `Write`, `Edit` write; `Bash` is irreversible; the catalog overrides |
| Records | every call the main agent ran itself, with its result or error and its duration (from its `tool_call` updates); the harness's tools are recorded by the bridge |
| Sub-agents and swarms | each sub-agent Kimi starts (`Agent`: coder, explore, plan) and each member of a swarm (`AgentSwarm`, up to 128) is a child run of the run, from the call to its result. Kimi decides when (the model calls the tool): nothing to configure |
| Questions | Kimi's `AskUserQuestion` is the run's question (`Runtime.ask`): a durable pause, the answer one of its options |
| Spending | Kimi's own usage log (`usage.record` lines: every model request of the main agent and of every sub-agent) is read as the turn runs: tokens count toward the budget, and past it the turn is cancelled and the run ends `PARTIAL`. Dollars are not counted yet (a follow-up reads them from the gateway's log): a `max_cost` budget does not see Kimi's spending |
| Stalls | a model step that sends nothing for `STALL_SECONDS` (a tool's time not counted) ends the run with a `ModelTimeout` |
| Cancel and time limits | a cancelled run, or one past its `run_timeout`, ends the attempt: a harness tool still under way is cancelled with it (recorded `cancelled` or `timeout`), as an in-process tool is, then the `kimi` process ends. Kimi starts in a second or two per attempt, inside the run's time limit |
| Grounding, judges, tracing | as for every target |

## Approvals and pauses

A harness tool that asks, `trellis.current().ask` inside one, or a call of Kimi's own that asks,
pauses the run: the turn is cancelled (`session/cancel`) and the `kimi` process ends. The
session is in the run's journal; `agent.resume(...)` starts `kimi acp` again, resumes that
session (`session/resume`, else `session/load`) and tells Kimi to call the tool again — the
journal answers it, or runs the approved call once. After a deny, Kimi is told the call will not
run, with the reviewer's reason.

ACP's answer to a permission request carries no reason and no arguments, so:

* **a rejection** (governance, a hook's `Deny`) ends the turn and Kimi is told why in a prompt of
  its own (`Your call to Bash was not run: …`);
* **an edit** (a reviewer's, a hook's `Rewrite`) ends the turn and Kimi is told to call the tool
  again with the edited arguments: that call is approved when it comes.

Kimi keeps its sessions on its machine (`KIMI_CODE_HOME`, `~/.kimi-code` by default). A run
resumed where Kimi does not hold the session gets an `acp_session` warning and runs the prompt
again in a new session, against the journal: the harness calls already made are not made again;
Kimi's own tools run again.

## Native or ours: sandbox, skills, hooks

* **Sandbox:** `kimi(sandbox=sandbox())` gives Kimi ACP's file and terminal capabilities, served
  by the run's sandbox (made at its first use, kept across attempts, snapshotted while the run
  waits, deleted at its end — [sandbox.md](../sandbox.md)): Kimi's `Bash` runs there, its
  `Write` and `Edit` write there. **Its reads are not isolated:** Kimi Code 2.1.1 reads and
  searches the host's files itself (`Read` looks a file up on the host first; `Glob` and `Grep`
  never ask the client), and ACP cannot turn those tools off. What the harness can do it does: a
  sandboxed Kimi gets a home and a working directory of the run's own, both empty, and an
  environment of `PATH`, `LANG`, `LC_ALL`, `TZ` and your `env` only — no secret of this machine's
  — and each sandboxed run says on its stream that reads are not isolated (`acp_sandbox`). To
  isolate reads, run the harness itself in a container.
* **Skills:** `kimi(**src.kimi())` — the run's pinned skills ([skills.md](../skills.md)) are
  written into Kimi's own skills folder (`$KIMI_CODE_HOME/skills/<name>/SKILL.md` and its files)
  in a home of the run's own, before each attempt; Kimi lists them to its model and loads one
  with its `Skill` tool. Your Kimi configuration and login (`config.toml`, `credentials`) are
  linked into that home; your own skill folders (`.agents/skills`, `.kimi-code/skills` of the
  project) stay Kimi's to find.
* **Hooks:** the run's hooks (`h.wrap(hooks=[...])`: `before_tool`, `after_tool`,
  `on_run_start`, `on_run_end`, `on_error`, `on_output`) apply to the harness's tools and to every
  call Kimi asks permission for. Kimi's own hooks (`hooks` in its `config.toml`: shell commands
  on `PreToolUse`, `Stop`...) stay Kimi's. ACP shows no model call, so there is no model hook.

## What ACP does not show

* **Calls Kimi does not ask about.** ACP has no hook before a tool runs: a read Kimi runs without
  asking (its default mode asks about none) is governed by Kimi's own policy, then recorded.
* **A sub-agent's own calls.** Kimi shows a sub-agent's (and a swarm member's) calls only when it
  asks permission for one — those are governed; the others are not recorded, and the child run
  holds the sub-agent's task and answer.

## Models, swarms and limits

* **Through Bifrost.** With `BIFROST_URL` set and `model=` named (and no `KIMI_MODEL_BASE_URL`
  in `env`), `kimi` gets `KIMI_MODEL_PROVIDER_TYPE=openai`, Bifrost's URL, the virtual key, and
  the deny-all MCP scope, the run's dimensions and its `traceparent` as
  `KIMI_CODE_CUSTOM_HEADERS` lines beside your own. Elsewhere, said once in the log.
* **Swarms.** Kimi's `AgentSwarm` runs its members concurrently, at most
  `KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY` at once when you set it in `env`.
* **Framework options.** `framework_options=` takes `ACPAgent` fields: `env` and `mcp_servers`
  are added to the agent's, `cwd`, `mode` and `model` replace them. Kimi's own step limit is
  `{"env": {"KIMI_LOOP_MAX_STEPS_PER_TURN": "30"}}`: at it Kimi ends the turn without an answer.

## Streaming

`agent.stream(...)` yields Kimi's answer as text deltas, the harness tool calls and every call
the main agent ran as `TOOL_CALL_*` events, then `RUN_FINISHED`.

## Tests

The adapter's tests run on both sides of the real ACP SDK: a scripted ACP agent
(`tests/support/fake_acp_agent.py`) for every edge case, and Kimi Code itself
(`tests/integration/test_kimi.py`, and the Kimi column of the [feature matrix](../matrix.md)),
its model a planned OpenAI-compatible endpoint on the machine — no Moonshot key. They pin what
the adapter reads of Kimi: its tool names, its permission requests, its sub-agents' and swarms'
results, and the shape of its usage log.
