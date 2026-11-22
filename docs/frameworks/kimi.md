# Kimi Code (over ACP)

The target is `kimi(...)`: Kimi Code's own agent (`kimi acp`), which the harness drives over the
[Agent Client Protocol](https://agentclientprotocol.com) (ACP), the open protocol editors use to
drive coding agents. Kimi keeps everything it does on its own: its tools, its sub-agents and
plan mode, compaction (its goal mode and its swarms are the exceptions:
[below](#goal-mode-not-supported)). The harness adds memory, governance, durable runs,
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
npm install -g @moonshot-ai/kimi-code@2.6.2    # the kimi program (Node 22.19 or later)
sudo apt-get install -y ripgrep                 # rg, which Kimi's Glob runs (macOS: brew install ripgrep)
```

In a Docker image, add `ripgrep` where Kimi runs (the harness's image, not agent-runs'):

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends ripgrep \
    && rm -rf /var/lib/apt/lists/*
```

Kimi's `Glob` runs `rg`. Left to itself without it on the `PATH`, Kimi Code downloads it from
its own CDN at the first `Glob` (into `~/.kimi-code/bin`), or that `Glob` fails ("ripgrep (rg)
is not available ... and the automatic bootstrap failed"). The harness checks first: a run
whose Kimi has no `rg` on its `PATH` (this process's, or its `env`'s) ends at once, saying what
to install, and Kimi is not started (`Profile.needs`, `acp.missing`). ripgrep is MIT- or
Unlicense-licensed. Its PyPI wheel (`ripgrep`) is built for few platforms (Linux x86-64 with
glibc 2.39, macOS arm64), so the system package is the one to install.

The `kimi` program is checked when a run starts: a version outside the tested range
(`>=2.6.2,<2.8`, [versioning.md](../versioning.md)) fails the run, saying which version answered
and how to install the tested one. Kimi's protocol surface (its tool names, its permission
requests, its usage log) is what the adapter reads, and nothing else pins it.

## Using Kimi Code

```python
from trellis import Harness, kimi

h = Harness()
agent = h.wrap(kimi(model="anthropic/claude-sonnet-4-5", cwd="/srv/repo"), id="maintainer")

result = await agent.run("The build is red: find out why and fix it.", user="ada")
if result.interrupt:  # Kimi asked to run a command that writes: a person approves
    result = await agent.resume(result.interrupt.interrupt_id, "approve", reviewer="ada")
```

The harness's tools — the MCP tools the Bifrost virtual key allows and, memory on, the memory
tools — reach Kimi as one more MCP server (below); your own MCP servers go in `mcp_servers=`.

`kimi(*, model=None, cwd=None, env=None, mcp_servers=(), mode=None, sandbox=None, skills=None,
command="kimi", reads=())`:

| Argument | What it is |
|---|---|
| `model` | the model's name on the gateway: with Bifrost serving the harness, Kimi's model calls go through Bifrost's OpenAI route with the virtual key (below). Without Bifrost, Kimi calls the model it is configured with: its login (`kimi login`) or `KIMI_MODEL_*` in `env` |
| `cwd` | where Kimi works (this process's working directory by default) |
| `env` | over this process's environment, for the `kimi` process (`KIMI_MODEL_BASE_URL`, `KIMI_MODEL_API_KEY`, `KIMI_LOOP_MAX_STEPS_PER_TURN`...) |
| `mcp_servers` | Kimi's own MCP servers, as ACP takes them: `{"name", "command", "args", "env"}` (stdio) or `{"type": "http", "name", "url", "headers"}`, beside the harness's — over HTTP for Kimi to offer their tools to its model ([below](#models-swarms-and-limits)) |
| `mode` | Kimi's session mode for a session the run starts: `"default"` or `"plan"`. `"auto"` and `"yolo"` are refused: they approve on their own, and governance would see no call. Whatever is given, every session is set to ask ([below](#the-session-mode)) |
| `sandbox` | `sandbox()`: Kimi's `Bash` and file writes go to the run's sandbox (below). Without one Kimi's shell runs on this machine as the service's user, **without the harness's isolation**; `TRELLIS_HOST_EXECUTION=deny` refuses it at `h.wrap` ([governance.md](../governance.md#commands-on-the-host)) |
| `skills` | a skill set, `**src.kimi()`: the run's skills in Kimi's own skills folder (below) |
| `reads` | folders besides `cwd` the run may read (its shell's commands run unasked either way: [who decides what](#who-decides-what)) |

## What is automatic

| | |
|---|---|
| Memory push | the context goes before the prompt, in the same `session/prompt` (ACP has no system message: Kimi keeps its own system prompt), marked off from the question as `ReAct`'s is (`<memory_context>`, `history.ASKED_WITH`) |
| Harness tools | one more MCP server, `trellis` (`mcp__trellis__<tool>`), served for the `kimi` process on `127.0.0.1` with a bearer token of its own, each call made in the run being served, handed to Kimi in `session/new` beside its own servers; Kimi connects to it and lists its tools as the session opens, and a Kimi that has not listed them 10 seconds after (never connected, or connected and its listing failed) would answer without the run's tools, so its process is started again (a warning, `acp_tools`) and a second that does not either fails the run saying so; the tool hints narrow what it lists for the run, from 30 tools of the run's own (Kimi's built-ins are not counted); `read_file` pages a large result |
| Kimi's own tools | Kimi asks before the calls its own policy asks about (`Bash`, and `Write`, `Edit` outside a git working tree, in its default mode; its reads it runs). Each is decided as a harness call is — the run's `before_tool` hooks, governance by its risk and the catalog's rules — and then, as Kimi asked, a person is asked (once: a call governance asks about too, such as a tool of Kimi's own MCP servers that declares nothing, is one question), unless it is `Bash` (in the run's sandbox, or on this machine unless the deployment says `deny`: it runs announced, never the user's question) or only reads (a read tool). `Read`, `Glob`, `Grep` read; `Write`, `Edit` and `Bash` write; `FetchURL` and `WebSearch` are the catalog's alone ([who decides what](#who-decides-what)); the catalog overrides — a team that wants every shell command approved gives `Bash` the risk `irreversible` (or an `approve_when`) there. Two things in Kimi's own policy decide a call without asking, and ACP gives a client no way to be asked first. **A write inside a git working tree**: Kimi approves `Write` and `Edit` there itself (its `git-cwd-write-approve` policy, with no setting to turn it off), so those run on Kimi's approval and are recorded after; outside a git tree it asks. **An `allow` rule in Kimi's configuration** (`[permission]` in its `config.toml`, or a project's `.kimi-code/local.toml`; applied over ACP since Kimi Code 2.6): Kimi runs in a home of its user's own (`$XDG_DATA_HOME/trellis/kimi/<tenant and user>`, never shared with another user: its sessions, logs and caches), an overlay of the team's Kimi home (`KIMI_CODE_HOME`, else `~/.kimi-code`): the team's login, skills, agents, plugins, `AGENTS.md` and `mcp.json` are linked, and its `config.toml` is the team's as it is with an `ask` rule added after it for each `allow` rule (a `[[permission.rules]]` entry, or — the team's `rules` written inline — a `[[permission.ask]]` one, which Kimi reads too). Kimi reads `ask` rules first, so it asks and governance decides. A config whose whole `permission` is one inline table (`permission = { ... }`) takes no rule after it in any form (TOML): such a run is not started, and says which rules (Kimi's `ask` and `deny` rules only make it stricter, and stay). **A skill Kimi loads** (`Skill`, which Kimi's own policy runs unasked): the same config asks about it (an `ask` rule `Skill`), so each call is decided as a harness call is — a read, run at once and recorded, unless the catalog says more (`Skill` given the risk `irreversible` has a person asked first). A sandboxed run's Kimi has no config of the team's (nothing is linked into its home): its own config holds the same `ask` rules, so there too each `Skill` call is decided |
| Schedules: off | Kimi runs for an attempt — kept only while work in its background goes on ([below](#background-work)) — and ends after it, so a scheduled prompt (`CronCreate`) would never fire: Kimi's own switch is set for it, `KIMI_DISABLE_CRON=1` (the agent's own `env` cannot turn it back on), so a call says `Cron scheduling is disabled` and the model can tell the user, where Kimi scheduled it and answered as if it would run. `CronCreate`, `CronList` and `CronDelete` are not of the run's tool set (no hint offers them); nor are they offered to Kimi's model: ACP has no way to take a built-in off and Kimi no environment switch for it, so the harness sets Kimi's own `disabled = ["CronCreate", "CronList", "CronDelete"]` under `[tools]` in the user's copy of the team's `config.toml` (`acp.cron_hidden`; a team's own `[tools]` is kept as it is, its schedule tools then listed and refused) (pinned by `test_kimi.py`). A recurring task is the harness's `agent.schedule(...)` ([runs.md](../runs.md)) |
| Tower mode: off | Kimi's experimental tower mode (`KIMI_CODE_EXPERIMENTAL_TOWER`: `TowerInit`, `TowerSpawn`, `TowerMerge`...) starts each worker in Kimi's `auto` permission mode, which asks about none of its calls, so the harness could govern none of them. Kimi's own switch is set for it, `KIMI_CODE_EXPERIMENTAL_TOWER=0`, which Kimi reads over its `config.toml`'s; turned on in the agent's `env` (Kimi then offers `TowerInit`, `TowerStatus` and `TowerTeardown` even over ACP), it stays off and the run says why (`acp_env` warning, as for `KIMI_DISABLE_CRON=0`). A run's sub-agents work through `Agent`, whose calls Kimi asks about |
| Updates and usage reports: off | Kimi's own update (`KIMI_CODE_NO_AUTO_UPDATE=1`) and its usage reports to Moonshot (`KIMI_DISABLE_TELEMETRY=1`) are off: a server's Kimi stays the version the harness was tested with, and nothing of the tenants' use is sent out. Set otherwise in the agent's `env`, they stay off and the run says why (`acp_env` warning) |
| Records | every call the main agent ran itself, with its result or error and its duration (from its `tool_call` updates); the harness's tools are recorded by the bridge |
| Sub-agents | each sub-agent Kimi starts (`Agent`: coder, explore, plan) is a child run of the run, from the call to its result; so is each it goes on with (`Agent(resume=...)`), named by the type Kimi started it with (its log's `profile.bind`), its task the prompt it goes on with. One it runs in the background (`run_in_background`) is a child run until its task ends ([below](#background-work)). Kimi decides when (the model calls the tool): nothing to configure |
| Swarms: not over ACP | since Kimi Code 2.3.2 `AgentSwarm` runs only in Kimi's swarm mode, which only its user turns on, in its own terminal (`/swarm`). Kimi's ACP server has no mode, option or command for it, so no ACP client can. Kimi still offers the tool; called, it starts nobody and answers `Swarm mode is off...`. The run goes on with that answer; the child run opened for each member ends `ERROR` with Kimi's words. Several sub-agents at once are several `Agent` calls in one step, which Kimi runs together |
| Allowed for the conversation | Kimi offers "Approve for this session" on a call it asks about, so the person is offered the same choices as for a tool the harness runs: allow once, allow for this conversation, always allow (with a memory service), deny (an `irreversible` tool: once or deny) — one rule, as for every adapter ([governance.md](../governance.md#one-rule-every-adapter-and-every-way-a-tool-arrives)). "Always" is kept as for a harness tool and covers the tool in the user's later conversations. ACP tells the harness nothing of an MCP tool's annotations, so a tool of Kimi's own MCP servers says nothing until the catalog gives it a risk; the catalog's word decides before Kimi's own question. "For this conversation" answers Kimi with its session approval and covers the tool's later calls in the conversation: Kimi asks no more about the same command or file (its own session rule), and what it still asks about the tool the harness lets through without asking the person again, in this run and in the conversation's later runs. Calls Kimi's session rule covers are not asked about, so governance sees them only once they ran (they are recorded); a tool every call of which must be decided is given the risk `irreversible` in the catalog, which is never allowed beyond a call |
| Questions | Kimi's `AskUserQuestion` is the run's question (`Runtime.ask`: a durable pause), asked as `ask_user` asks (the payload's `source` names the run's agent and `AskUserQuestion`). The harness tells Kimi it takes forms (ACP's `elicitation/create`), so Kimi sends every question of the call (up to four) in one form: one question to the person, a field for each of Kimi's questions with its options and their descriptions, several picks where Kimi's question takes several. One question alone is a choice of its options, each with its description, the one labelled "(Recommended)" recommended. In a run nobody is known to read (not a chat front end's, or fired by a schedule) the form is declined: Kimi tells its model "User dismissed the question without answering.", and the run gets no note on when to ask. An answer is read as every adapter reads one: an option handed back (its label, case and spacing aside) is that option; any other answer in words ("SUP-04", "not SUP-04", "the second one", or words for a whole form) is never refused — Kimi's ACP server has no "Other", so the question is dismissed and Kimi told `The user answered your question "…" with: …` in the resumed turn (`ANSWERED`), and its model reads them. A client that takes no forms is asked only the first question by Kimi, and the rest are dropped |
| Plans | Kimi asks its user to review every plan it presents (`ExitPlanMode`): the plan, one option an approach (or `Approve`), `Revise` and `Reject and Exit`. A permission request in which the agent offers a choice of its own is the agent's question to its user: the run pauses on one question that shows the plan and every option, and the person's pick is Kimi's answer (words that pick none of them reach Kimi as a question's do: "make it faster"). One question a call; the resumed session is told to make the call again and reads the answer |
| Spending | Kimi's own usage log (`usage.record` lines: every model request of the main agent and of every sub-agent) is read as the turn runs: tokens count toward the budget, and past it the turn is cancelled and the run ends `PARTIAL`. Dollars are not counted yet (a follow-up reads them from the gateway's log): a `max_cost` budget does not see Kimi's spending |
| A message that starts with `/` | over ACP a prompt whose first block starts with `/` is a command of Kimi's own (`/compact`, `/status`, `/skill:<name>`), run with no model turn. A user's message is words for the model in every run: it never starts the prompt's first block. Kimi's commands are not reachable through a run |
| Kimi's plan | Kimi's todo list (`TodoList`) reaches an ACP client as its plan; each state of it is the run's `plan` event (a `CUSTOM` event named `plan`: `entries`, each with Kimi's `content`, `status` and `priority`), for a surface that shows what the agent is at. The `TodoList` call itself is recorded as any call |
| A turn Kimi could not finish | Kimi answers a prompt whose turn failed (its provider answered with an error, past Kimi's own retries: `KIMI_LOOP_MAX_RETRIES_PER_STEP`) as it answers a finished one, and says why only in its log. The run ends `ERROR` with Kimi's own error (`provider.api_error`, the provider's status), whether or not text came first |
| Stalls | a model step that sends nothing for `STALL_SECONDS` (a tool's time not counted) ends the run with a `ModelTimeout` |
| Cancel and time limits | a cancelled run, or one past its `run_timeout`, ends the attempt: a harness tool still under way is cancelled with it (recorded `cancelled` or `timeout`), as an in-process tool is, then the `kimi` process ends (unless work in its background keeps it: [below](#background-work)). Kimi starts in a second or two per attempt, inside the run's time limit |
| Grounding, judges, tracing | as for every target |

## Approvals and pauses

A harness tool that asks, `trellis.current().ask` inside one, or a call of Kimi's own that asks,
pauses the run: the turn is cancelled (`session/cancel`) and the `kimi` process ends (kept
while work in its background goes on: the resumed attempt is then sent into it). The session
is in the run's journal; `agent.resume(...)` starts `kimi acp` again, resumes that
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

## Who decides what

The run's user decides only the actions taken in their name; the rest is the deployment's
([governance.md](../governance.md#who-decides-what)):

| Kimi's | Decided by | What happens |
|---|---|---|
| `Read`, `Glob`, `Grep` in the workspace | nobody | they run, recorded |
| `Write`, `Edit` of the user's files | the user, under the one rule | asked: allow once, for this conversation, always (with memory), deny (Kimi's own approval inside a git working tree aside: below) |
| `Bash` in the run's sandbox (`kimi(sandbox=sandbox())`) | the deployment | runs unasked, announced (`tool_notice`) and recorded |
| `Bash` on the host (no sandbox) | the deployment | runs as the service's user, **without the harness's isolation**, unasked and announced; `TRELLIS_HOST_EXECUTION=deny` refuses it at `h.wrap`. To isolate it: `kimi(sandbox=sandbox())` |
| `FetchURL`, `WebSearch` | the administrator, in the tool catalog | Kimi's own policy runs them unasked, so a run's Kimi is configured to ask about them (an `ask` rule in its `config.toml`, as for `Skill`) and the catalog decides: refused (Kimi is told why) until it gives the tool a risk or an `approve_when`. A sandboxed run's Kimi has no config of the team's: its own holds the same `ask` rules, so the catalog decides there too (Kimi fetches from this machine's network either way) |
| MCP servers, the session mode | the deployment | `kimi(mcp_servers=)` or `h.wrap(framework_options=)`; a run's own `mcp_servers` that change them are refused, and `auto` and `yolo` are refused everywhere |

## Native or ours: sandbox, skills, hooks

* **Sandbox:** `kimi(sandbox=sandbox())` gives Kimi ACP's file and terminal capabilities, served
  by the run's sandbox (made at its first use, kept across attempts, snapshotted while the run
  waits, deleted at its end — [sandbox.md](../sandbox.md)): Kimi's `Bash` runs there, its
  `Write` and `Edit` write there. Without it, Kimi runs its commands in `cwd` on this machine,
  as the service's user, not through the harness's isolation (`TRELLIS_HOST_EXECUTION=deny`
  refuses that). **Its reads are not isolated:** Kimi Code reads and
  searches the host's files itself (`Read` looks a file up on the host first; `Glob`, `Grep`
  and `ReadMediaFile` never ask the client), and ACP cannot turn those tools off. What the
  harness can do it does: Kimi is configured to ask about `Glob`, `Grep` and `ReadMediaFile`,
  and one whose path is outside the run's folders (its working folder, `reads`, its skills) is
  refused, Kimi told why; a sandboxed Kimi gets a home and a working directory of the run's own, both empty, and an
  environment of `PATH`, `LANG`, `LC_ALL`, `TZ` and your `env` only — no secret of this machine's
  — and each sandboxed run says on its stream that reads are not isolated (`acp_sandbox`). To
  isolate reads, run the harness itself in a container.
* **Skills:** `kimi(**src.kimi())` — the run's pinned skills ([skills.md](../skills.md)) are
  written into Kimi's own skills folder (`$KIMI_CODE_HOME/skills/<name>/SKILL.md` and its files)
  in a home of the run's own, before each attempt; Kimi lists them to its model and loads one
  with its `Skill` tool. Your Kimi configuration and login (`config.toml`, `credentials`), your
  instructions for every session (`AGENTS.md`) and your plugins are linked into that home, and
  so are its sessions, which stay in your Kimi home: the conversation's next turn resumes the
  session this run ends in. Your own skills of that home (`skills/`) are linked beside the
  run's (a name the run has a skill of is the run's); your other skill folders
  (`~/.agents/skills`, `.agents/skills` and `.kimi-code/skills` of the project) stay Kimi's to
  find.
* **Hooks:** the run's hooks (`h.wrap(hooks=[...])`: `before_tool`, `after_tool`,
  `on_run_start`, `on_run_end`, `on_error`, `on_output`) apply to the harness's tools and to every
  call Kimi asks permission for. Kimi's own hooks (`hooks` in its `config.toml`: shell commands
  on `PreToolUse`, `Stop`...) stay Kimi's. ACP shows no model call, so there is no model hook.

## The session mode

Kimi's `config.toml` can start every session in a mode of its own: `default_permission_mode =
"auto"` or `"yolo"` (Kimi approves its own calls and asks about none) and `default_plan_mode =
true`. Kimi's ACP server still answers `default` for such a session, so nothing in the protocol
says the approvals are off. The harness governs only the calls Kimi asks about, so before the
first prompt of every session it opens — a new one, one a resume or a conversation's next turn
continues, a forked one — it sets the mode itself (`session/set_mode`) and reads Kimi's own log
of the session to see that it took:

* **The permission mode is always the one that asks** (Kimi's `manual`). A `config.toml` that
  says `auto` or `yolo` does not turn the approvals off: this is the one Kimi setting a run
  overrides, since a person's approval is the harness's to ask for.
* **Plan mode is Kimi's own, kept as the session has it.** `mode="plan"` starts a session in
  plan mode; so does Kimi's `default_plan_mode`. A session that is in plan mode when a run
  continues it stays in it, and one whose plan was approved stays out of it. Kimi refuses to
  enter the plan mode of a session already in it, so a session that plans and asks is set
  nothing; one its config starts both planning and approving on its own (`default_plan_mode`
  with `default_permission_mode`) is set to `default` and then `plan`, before any plan exists.
* **A session whose log does not confirm the mode gets no prompt**: the run fails saying which
  mode was set and what Kimi's log says. Kimi writes its log after it answers, later on a
  busy server: the line is waited for up to 30 s, as long as a prompt waits for its session,
  and no prompt is sent before it.

An ACP agent with no profile (`ACPAgent(profile=None)`) names its own modes and keeps the one
it starts in; only the `mode=` the target gives is set, on a session the run starts.

## What ACP does not show

* **Calls Kimi does not ask about.** ACP has no hook before a tool runs: a read Kimi runs without
  asking (its default mode asks about none) is governed by Kimi's own policy, then recorded.
* **A sub-agent's own calls.** Kimi shows a sub-agent's calls only when it
  asks permission for one — those are governed; the others are not recorded, and the child run
  holds the sub-agent's task and answer. Kimi's request for a sub-agent's call names the tool
  and one line that cuts a command at 50 characters, with no arguments. The harness decides it
  with what Kimi shows its own user of the call, read from Kimi's log of the session
  (`interaction.request`): a `Bash` command whole with the folder it runs in (`command`, `cwd`,
  `language`), a file's `path` and `content` for `Write`. So a reviewer reads the whole command,
  a catalog rule's `approve_when` sees it, and a sub-agent's command that only reads asks
  nobody, as the main agent's. These are Kimi's display fields, not the call's raw arguments:
  an argument Kimi does not show (a `Bash` `timeout`) is not in them.
* **Background work and Kimi's own turns.** Nothing of them reaches the client: the harness
  reads Kimi's log ([above](#background-work)).

## Goal mode: not supported

Kimi's goal mode (`CreateGoal`, `UpdateGoal`, `SetGoalBudget`) keeps working after a turn ends:
while a goal is active Kimi starts the next turn itself. Measured with Kimi 2.1.1 over ACP: the
prompt is answered when the first turn ends; Kimi then runs the goal's further turns on its
own, calls its model in them and asks permission for their calls, and sends the client none of
their text or tool updates (its ACP server forwards only the turn a prompt started).

Under the harness a run ends with its prompt's turn, and the agent's process with it unless
background work keeps it ([below](#background-work)), so an active goal stops there: the first
turn's work is done and recorded, the goal's later turns do not run. `CreateGoal` is a write, so
a person is asked before a goal starts. For work that goes on over many steps, give the run
the whole task (Kimi works through it in one turn), or schedule it (`agent.schedule(...)`,
[runs.md](../runs.md)).

## Background work

Kimi runs a sub-agent in the background (`Agent(run_in_background=true)`) and a command
(`Bash(run_in_background=true)`; a foreground one past its time limit, too): the call answers at
once with the task's id, and the task works on — in the `kimi` process, which a turn's end
would end. So a conversation's turn that ends with a task at work leaves the process to the
conversation (`bridges.acp_live`), and its next turn is served by it, with its own prompt; a turn
that ends with nothing at work ends it, as ever.

* **What ACP shows of it: nothing.** Kimi's ACP server forwards only the events of the turn the
  current prompt started: a background sub-agent's work, the turn Kimi runs by itself when a
  task ends (its notification, which Kimi's own context keeps: the next turn's model has it) and
  anything after a prompt's answer never reach the client. The harness reads Kimi's log of the
  session instead — the `agents/<id>/wire.jsonl` it reads usage from: each task's start and end
  (`task.started`, `task.terminated`, a sub-agent's last word in `subagent.completed`), the main
  agent's turns (`turn.prompt`, one whose `origin` is not the `user` being Kimi's own,
  `turn.ended`), whose a permission request is (`interaction.request`: ACP's names no agent),
  and each sub-agent's usage.
* **A task that ends while a turn waits for it answers that turn.** When a background task
  ends while the user's turn runs (the model waiting for it with `WaitFor`, say), Kimi steers
  the task's notification into that turn (`prompt.steered`), ends the turn right after its
  step with no answer, and answers in a turn of its own — which ACP does not forward. The
  harness waits for that turn (as long as a prompt waits for Kimi's own turns, 30 seconds at
  most) and takes its answer, read from Kimi's log, as the user's turn's answer, as `kimi -p`
  prints it; what it spends is that run's, counted once (the next turn counts only its own).
  Kimi asks about a call it makes in that turn in the user's run, which is still under way;
  the calls themselves are not on the run's stream (ACP shows none of them).
* **A background sub-agent keeps its child run.** The child run its `Agent` call started stays
  `RUNNING` from there — never `SUCCESS` with the launch's words — until Kimi's log says its task
  ended: `SUCCESS` with what the sub-agent said last, `ERROR` when it failed, `CANCELLED` when it
  was stopped in Kimi (`TaskStop`), and `ERROR`, saying so, when the process ends first (it
  closed, it died). It is held in the run store meanwhile, as Claude's is (`holding`; a killed
  server leaves it to agent-runs, `holder_lost`), and its attempts go through the pipeline.
* **Its calls are its run's.** Kimi asks about a sub-agent's call whenever it comes — turn or no
  turn: approvals are its session's — and the harness decides it in the sub-agent's run, as a
  call of Kimi's own is decided: your hooks, governance by risk and the catalog, a person when
  Kimi asks (unless it only reads); its harness tool calls run for that run. A call that asks a
  person pauses that run, and Kimi waits; the conversation's **next turn** asks the person
  first (`payload["subagent"]` names the run, with `background: true`), then sends its prompt.
  Approved the call runs, rejected it does not (ACP carries no reason to the sub-agent). An
  answer that reaches another server, where the session is not, ends the run `ERROR` and the
  turn says so (`acp_session`). Its tokens (its own log) are its run's, under the agent's
  `budget` and `run_timeout`; past them the run ends there.
* **No run, no call.** A call Kimi asks about while no run is under way — the main agent's, in the
  turn Kimi runs for a task's end — is refused, and the conversation's next run says how many
  were (`acp_session`). Its reads Kimi runs without asking, as always; what its own turns spend is
  on the next turn.
* **One prompt at a time.** A prompt sent while Kimi runs a turn of its own is queued behind it:
  Kimi answers it `end_turn` at once and runs it later, unseen. So a prompt waits until Kimi's log
  shows it idle (30 seconds at most; past that the process is replaced), and a prompt Kimi
  queued anyway fails the turn saying so — never an empty answer taken for its own.
* **ACP cannot stop one task** (`session/cancel` stops the main agent's turn, and its tasks go
  on): a background run that ends first — cancelled (with the turn that started it, or on its
  own), out of time, its budget spent — leaves its task at work, refused whatever it asks, until
  it ends or the process does.
* **Its life**, as Claude's: one process per conversation, in one server process, at most 8 over
  all its agents; it closes 5 minutes after its last activity when nothing works in it, 6 hours after its
  last turn whatever does, when a turn brings other settings (`framework_options`, other harness
  tools) or does not continue its session, and with the harness (`Harness.aclose`). Work that
  ends with it is said in the conversation's next run (`acp_session`); a restart loses it.
  When its time is up, its place is given to another conversation, or the harness closes, the
  session's own folder (`sessions/<workspace>/<session>`) leaves its user's Kimi home too — its
  login and config stay — and the conversation's next turn starts a new session, given its
  earlier turns by the harness (a session closed for other settings, or as the turn ends with
  nothing at work, stays on disk: the next turn resumes it). Such a session's folder stays as
  long as the agent keeps the session for the conversation's next turn: it leaves the Kimi home
  once the agent lets go of it — its place given to the 128 conversations whose turns are more
  recent, or dropped because another worker served a later turn — and with the harness
  (`Harness.aclose`). A paused turn's session stays: its run's resume continues it.
* **On another worker.** A turn whose session another worker's process still holds (a background
  run it handed on, not ended) forks it (`session/fork`) and goes on there in a process of its own.
* **Not kept:** a run with no `thread`, a sub-agent's run, a sandboxed run (its sandbox is the
  run's) and one given skills (a home of the run's own, removed with it): their process ends with
  the attempt, as before — a background sub-agent that ended by then ends its run with its result,
  one still at work was cut off (`ERROR`). Kimi's `keepAliveOnExit` (`[task]` in its config) is no
  way around it: it only leaves tasks unstopped as `kimi` exits, and a sub-agent's loop runs inside
  that process.

## Conversations

A conversation's turn continues the session the conversation's last turn ran in, in this
process (`session/resume`, else `session/load`, as a resume after a pause does: [runs.md](../runs.md#a-claude-or-acp-session-across-turns)).
ACP has no way to add a message to a turn under way, so `mode="steer"` waits as `"queue"`, said
once ([runs.md](../runs.md#a-message-to-a-busy-conversation-mode)).

## Models, swarms and limits

* **Through Bifrost.** With Bifrost the route (`LLM_GATEWAY=bifrost`) and `model=` named (and no `KIMI_MODEL_BASE_URL`
  in `env`), `kimi` gets `KIMI_MODEL_PROVIDER_TYPE=openai`, Bifrost's URL, the virtual key, and
  the deny-all MCP scope, the run's dimensions and its `traceparent` as
  `KIMI_CODE_CUSTOM_HEADERS` lines beside your own. Elsewhere, said once in the log.
* **Swarms.** Not over ACP ([above](#what-is-automatic)). `KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY` is still
  set to the harness's adaptive limit of child runs unless your `env` names it, for a Kimi
  whose swarm mode is on.
* **Your MCP servers: give them over HTTP.** Kimi 2.6.2 offers its model the tools of an MCP
  server over HTTP (`{"type": "http", "name", "url", "headers"}`) in its first request, but not
  those of a stdio server it starts itself: it starts the server, lists its tools
  (`tool_select.initialized`, as `initial`, `inline` tools) and leaves them out of every model
  request (`llm.tools_snapshot`), offering them only through its `select_tools`, which a model
  without Kimi's `dynamically_loaded_tools` capability is not given. Measured with a plain ACP
  client and under the harness alike (Haiku and a scripted model), for a Python MCP SDK server
  given in `session/new` (started at once, and a second late), in the Kimi home's `mcp.json`
  (the team's, linked into each user's home: started with the process) and in a project's
  `.kimi-code/mcp.json` (not read at all: Kimi's workspace trust gates it). The harness has no
  other way to hand Kimi a server, so run a stdio server behind HTTP (most MCP SDKs serve
  streamable HTTP) and give its URL; the harness's own tools reach Kimi that way
  (pinned by `test_kimi.py::test_kimi_offers_its_model_an_mcp_server_over_http_in_its_first_request`).
* **Framework options.** `framework_options=` takes `ACPAgent` fields: `env` and `mcp_servers`
  are added to the agent's, `cwd`, `mode` and `model` replace them. Kimi's own step limit is
  `{"env": {"KIMI_LOOP_MAX_STEPS_PER_TURN": "30"}}`: at it Kimi ends the turn without an answer
  (over ACP a finished turn; its own log says `loop.max_steps_exceeded`), and the run ends
  `PARTIAL` with what Kimi said before the limit, as Claude's turn limit does, and no model
  call more; having said nothing, Kimi is asked once, in the same session, to answer without
  tools, so the run does not end with no answer. A `step_limit` warning and the budget say
  so, its error `BUDGET_EXCEEDED` naming the limit, so an app can offer "continue".
  A loop on one call ends the same way: Kimi's own repeat breaker (`KIMI_CODE_REPEAT_BREAKER`,
  on by default: reminders to answer, then the turn ended at the twelfth identical call; its log
  says `repeat_breaker`) and the harness's refusal of a harness call repeated three times
  (`bridge.stalled`) are kept; a turn either ends with no answer is asked once more, told the
  turn stalled (`bridge.TURN_STALLED`), with a `stalled` warning.

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
