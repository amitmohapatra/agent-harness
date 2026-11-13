# Changelog

What changed in each release of `trellis-harness`. The framework versions each release was
tested with are in [docs/versioning.md](docs/versioning.md).

## 0.7.0 — any provider: one route strategy, an auth strategy, a model's profile (unreleased)

### Fixed: one question a call, run once, decided once — every adapter, whatever asks

`tests/integration/test_one_question.py` checks, for every adapter and every way a call can
need a person's approval (a tool that declares nothing, a catalog rule, a write, a hook's
`Ask`, an allowance for the conversation or always, the library's own gate: LangChain's and
Deep Agents' `interrupt_on`, OpenAI's `needs_approval` and an MCP server's
`require_approval`, Claude Code's `can_use_tool`, an ACP agent's permission request), on a
run, a stream, a resume on another harness, inside a sub-agent: one question a call, its
body run once when approved and never when rejected, one record of it (journal, memory) and
one count (budget), and its decision fed back once on the run that recorded it. What did
not hold:

- **The same call twice in a turn: the second took the first's answer.** A replayed call
  did not read past the answer it had read, so the next identical call took that answer
  for its own: a function target or a graph without a checkpointer, approving the first
  and rejecting the second, ran both. A framework that goes on from its session (Claude,
  Kimi over ACP) or its saved run (OpenAI's `needs_approval`) does not make the calls before
  the paused one again, but the attempt read the journal from its start: the call made
  again got the earlier call's record (approved, it did not run) or answer (Claude Code's
  question rejected for the second call ran it). The journal keeps what each call read
  (`reads`) and how far the calls before the pause read (`cursors`); a subagent paused in its
  work, which its framework starts again, reads its own again. (Not changed: a person's
  denial of a call stands for the run, so in a session the same call made again later in
  the turn is denied without asking.)
- **A subagent started again ran a built-in it had run.** Claude starts a subagent paused in
  its work again: it made its earlier built-in calls again, and they ran again on their
  recorded approval. A built-in that ran is recorded in the journal; made again, it is not
  run, and the subagent reads its result.
- **A decision on a call whose arguments nest was never counted.** The memory service refuses
  feedback whose metadata nests past two levels or passes 8 KiB: the decision on a list or
  object argument (an OpenAI shell command, any tool taking one) was lost. Each nested value
  is sent as its JSON text, cut to fit.
- **An OpenAI agent tool's own approval was fed back on the calling run.** The SDK's
  `needs_approval` inside `Agent.as_tool` names the agent tool's child run, which records the
  call, as governance's approvals of a helper's call do.

### Fixed: a graph's own tools, as LangGraph and Deep Agents have them

The LangGraph adapter was checked feature by feature against the installed libraries
(LangGraph 1.2, LangChain 1.4's `create_agent` and its middleware, Deep Agents 0.7): each
feature run through the harness with a scripted model, 76 tests in
`tests/integration/test_langgraph_library.py`, `test_create_agent_library.py`,
`test_deepagents_library.py`, `test_tool_updates.py` and `test_tool_errors_library.py`. Most
worked as the library gives them. Five did not:

- **A tool's error reaches the graph's own handling.** The harness records a failed call and
  handed the graph a failed result in the error's place, so nothing of the graph's that acts
  on a tool's error ever saw one: LangChain's `ToolRetryMiddleware` never tried again,
  `ToolErrorMiddleware` never worded the error, the `except` of a `wrap_tool_call` wrapper
  never ran. The call is recorded as before and the error is raised again as the tool raised
  it. An error nothing handles is, as before, the failed call's result the model reads (the
  run goes on). The tries a wrapper makes of one call are not counted as a model repeating a
  call (the stall guard stopped the third).
- **A tool's `Command` is the graph's again when its call is replayed.** A call answered
  from the journal gave the graph the message the model had read and dropped the state
  update the tool had returned with it: after a resume without a checkpointer, what a Deep
  Agent's sub-agent had written was gone (`task` returns its files as a `Command`), and so
  was any state a tool of the graph's own had set. The journal keeps what the tool returned
  with the call's record.
- **A list of `Command`s and messages returned by one tool is applied.** The harness gave the
  graph one message in its place, and no update was made.
- **A call that asks waits for the calls a `ToolNode` runs beside it.** The node stops at the
  first call that raises and leaves the others running: a write beside a call waiting for an
  approval ended after the run's pause was recorded, and the resumed run told the model its
  effect was unknown ("interrupted by a crash").
- **A Deep Agent with a `mode="fork"` sub-agent is wrapped.** `h.wrap` never returned: a fork
  runs the graphs it is one of, and reading the sub-agents' file tools went round.

Four more where the harness restricted what the library does:

- **An error nothing of the graph handles fails the run**, as it fails the graph without the
  harness (`ToolNode(handle_tool_errors=False)`, `create_agent`'s default,
  `ToolRetryMiddleware(on_failure="error")`, a wrapper that raises it again): the run ends
  `ERROR` with the tool's error, as any graph's error ends a run, and the call is recorded as
  failed. It was turned into a failed result the model read. A graph with the harness's
  middleware (`ReAct`, `middleware()`: `GovernTools`) keeps the harness's documented rule:
  the model reads the failure and the run goes on.
- **The graph's own text for a failed call is what the model reads**: a `ToolNode`'s
  `handle_tool_errors` text reached the model as `"<tool> failed: <text>"`. The run's
  `after_tool` hooks still have the last word.
- **What a node writes to LangGraph's `custom` stream mode is on the run's events**:
  `get_stream_writer()` (and a middleware's `runtime.stream_writer`) writes are `CUSTOM` events
  named `custom`, the write in `value` (AG-UI's `CUSTOM` event), from `agent.stream` and
  `agent.run` alike. `agent.run` reads the graph with `astream` (the `values` and `custom`
  modes), as `ainvoke` does.
- **A decision the graph makes again is not a model repeating itself**: the stall guard counts
  the same call made again only when it comes from another decision — for LangGraph, another
  task or another call id (`Runtime.deciding`). A node its `RetryPolicy` ran a third time had
  its harness call refused as a stall.

### Added: a background task of Deep Agents' async sub-agents, found again in later turns

Deep Agents' async sub-agent (an `AsyncSubAgent` in `ReAct(subagents=)`, or its
`AsyncSubAgentMiddleware` in a graph of your own) runs a task on a server of yours while the
chat goes on. Deep Agents keeps its list of tasks in the graph's state, and a graph the
harness runs on its run's thread starts each turn with a new state: the task started, and
the next turn's `check_async_task` was refused its id.

- **The list goes from a conversation's turn to its next one** (`adapters.langgraph.CARRIED`:
  the one place that names the channel), per user, thread and agent, a cancelled turn's too —
  for `ReAct`, for a graph built with `middleware()`, and for one with neither `middleware()`
  nor a checkpointer, whose list is read from the state each turn ends with. A graph with a
  checkpointer of its own keeps it in its thread, as before.
- **Across servers and restarts.** The list is kept in the process that ran the turn and in
  the file store (`TRELLIS_FILES_URL`), under `.carried/<tenant>/<user>/<conversation>/<agent>/`,
  which no listing of a user's files shows: the next turn has it after a restart, and on
  another server when the servers share the store (unset, it is a folder on the server's own
  disk). Deep Agents' record of each task and nothing else, at most 256 KiB for a
  conversation; a store that cannot be read or written is said in the log and the list stays
  in the process. Nothing is added to a run's record, to agent-runs or to the memory service.
- **Turns that overlap lose no task.** What a turn ends with is merged into what is kept,
  task by task, instead of replacing it: a turn approved after the conversation's next turn
  ended, or a worker's turn still running while another ends, each keep their tasks. A turn
  that waits for an approval has its list kept as it pauses: the task it started is listed
  by the next turn meanwhile, and is still found when the turn is cancelled at the approval.
- **The five task tools say what they do** where the harness says it of Deep Agents' own
  tools: a status check and the list are reads; a start, an update and a cancel are writes,
  announced and not asked about (the tool catalog still comes first). A write's result is
  declared the record of what was done, not a value that changes: with memory on, an earlier
  start in a later turn's history was marked "out of date: call start_async_task again",
  which invited a second task; it is left as it was said, and a status check's is still
  marked.

Not in this release: nothing tells the chat when a task ends (the model checks in a later
turn); cancelling a turn does not cancel its task; a graph with neither `middleware()` nor a
checkpointer loses the task of a turn cancelled after its start. See
[docs/frameworks/react.md](docs/frameworks/react.md#a-task-in-the-background).

### Fixed: the Claude Agent SDK's features through the harness, each tested on both paths

Driving the bundled CLI (2.1.283) against a scripted model showed what it sends, and where a
wrapped agent did not get what the SDK gives:

- **Subagents.** Claude Code runs a subagent in the background unless its call says
  otherwise. In a run with no conversation its child run ended at its launch, `SUCCESS` with
  the launch text, and its calls were recorded under the run: it now stays open until its
  task ends (`SUCCESS` with what it said; `ERROR` when it failed, was stopped, or the process
  ended first), and says so on the run's stream (`helper_finished`; a conversation's
  background subagent too, while the turn that started it is still answered). Two subagents
  of one type at work at once each record their own calls (by the task's id, `agent_id`,
  not the type's name). A subagent's forwarded text is no longer streamed as the run's.
- **Spending.** A result's cost is its session's running total and its usage the main
  agent's own. A query whose session ran a turn of its own, the last answer at the turn
  limit, and a conversation's turn in a process that resumed the session were charged the
  total again; each is charged what the cost grew by (the journal keeps the session's cost
  across attempts). A subagent's replies' tokens count toward the run's budget.
- **`max_budget_usd`** is Claude Code's cap on what one process spends: options with one run
  each turn in a process of its own again, so the cap stays a query's and a conversation
  goes on past it.
- **One call, one question.** A tool that says nothing of what it does asked a person
  twice for one call: governance's question, then on the resume Claude Code's own. The
  second is not asked — whatever made governance ask (the tool's risk, a catalog rule, a
  hook's `Ask`, a rule about the arguments a hook rewrote), the main agent's call or a
  subagent's.
- **Claude's own tools.** Its task list (`TaskCreate`, `TaskGet`, `TaskList`, `TaskUpdate`:
  `TodoWrite`'s successors), `EnterPlanMode` and `StructuredOutput` (`output_format`) asked
  a person on every call — a structured answer paused its own run. They run, as `TodoWrite`
  does.
- **Slash commands.** A turn the CLI answered itself (`/compact`: no reply of the model's
  in it) ended `ERROR` as a run with no answer; it ends `SUCCESS` with an empty answer, as
  any framework's run no model replied in. A blank reply of the model's still ends
  `ERROR` (`NO_ANSWER`).
- **Deadlines.** A native helper's child run (a Claude, Deep Agents or OpenAI Agents
  subagent) is recorded with the deadline of the run that started it, as a sub-agent's run
  is: agent-runs ends one a killed server left `RUNNING` by then.
- **Sessions.** Options with `session_id` failed when a pause resumed them (Claude Code
  takes it with `--resume` only to fork), and options with `continue_conversation` failed at
  once (the SDK asked the journal's store for the latest session): both work.

Tested through the harness on the query path and in a live session: every hook event of the
options' own, `can_use_tool` with its suggestions and permission updates, every option's
flag, the system prompt and thinking forms, structured output, partial messages, the session
options and the SDK's errors. What stays as it is:
[docs/frameworks/claude-agent-sdk.md](docs/frameworks/claude-agent-sdk.md#limits).

### Added: a conversation's Claude Code process stays between its turns, with its background work

A Claude Agent SDK turn ran in a CLI process of its own, and the next turn started another
that resumed the session. What Claude started in the background — a subagent with
`background`, a `Bash` command with `run_in_background` — belongs to its process: a
background subagent held the turn open until it finished, a background command ended with the
turn, and an approval ended both. A conversation's turn now leaves its process running
(`bridges.claude_live`: the SDK's `ClaudeSDKClient`) and the next turn is sent into it, so
that work goes on between the turns. A turn ends at the result of its own prompt
(`ResultMessage.origin` none or `human`), never at one the session injected for a finished
task; a pause ends the turn and keeps the process, and the resumed attempt is sent into it;
cancelling a turn stops that turn only (Claude Code's own interrupt stops its background work
too, so with any under way the turn ends at its next tool call instead). Each turn's hooks,
permission check and harness tools still run for that turn's run. A background subagent
keeps the child run its `Agent` call started: open until the SDK says its task ended
(`SUCCESS` with what it said last, `ERROR` for a failed task or a session that ended first,
`CANCELLED` for one stopped in Claude Code), its calls decided, run and recorded for that
run — hooks, governance, `can_use_tool`, harness tools — with or without a turn being
answered. A call of it that asks a person pauses that run and waits in Claude Code; the
question is asked in the conversation's next turn, before its prompt, and the answer goes
on to the run (nothing answers a child run directly today). Its limits are the agent's
`run_timeout` and `budget`; past them, or cancelled (`agent.cancel(child_run_id)`), the task
is stopped with the SDK's `stop_task`, that subagent only. The process's running cost is
shared at each result by the tokens each used, so a subagent's work between two turns is
charged to its run and the next turn its own. Any other call made between two turns (the
main agent's, in a turn the session injected) has no run and is refused, not run unchecked,
and the next run says so (`claude_session`). With no live process (a restart, another server, one closed for being idle, a
run with no `thread`) a turn runs as before, resuming the session from the journal, which
holds what was mirrored between the turns too. Limits and what is not kept:
[docs/frameworks/claude-agent-sdk.md](docs/frameworks/claude-agent-sdk.md#conversations).
`TaskStop` (a write) and six tools that only list or look up (reads) have a declared risk;
they asked a person on every call. Claude Code's own schedules (`CronCreate`,
`ScheduleWakeup`) stay off: a turn they fire has no run.

### Fixed: a Claude subagent is one child run when the run store takes time to answer

Claude Code says a subagent started two ways, its `Agent` call's `PreToolUse` hook and the
message that carries the call, and each started the helper by the call's id. With a run
store that takes time (agent-runs, over HTTP) the second came while the first was still
being recorded and started a second child run of the same subagent, which then ended at
once. A helper's start now takes its key one at a time (`bridges.helpers.Helpers.start`).

### Fixed: OpenAI Agents and ACP features that did not work through the harness

Each of the two libraries' own features was run through a wrapped agent with a scripted model
(`tests/integration/test_openai_agents_features.py`, `test_openai_agents_mcp.py`,
`test_acp.py`, `test_function_agent.py`); five did not work and do now:

- **An SDK approval inside an agent tool.** A `needs_approval` tool of an agent called as a
  tool (`Agent.as_tool`) paused the run, and every approval paused it again: the agent tool's
  unfinished call was journaled as done, so the resume replayed it and the run inside never
  continued. The call is now a pause to the bridge while the SDK holds its run waiting, its
  child run stays open, and the approval (or rejection) continues it, from another worker too.
- **The run's own `RunHooks`.** `framework_options={"hooks": ...}` was refused; the SDK's
  `RunHooks` are now called beside the harness's own (which still count every reply toward
  the budget). `starting_agent` and `input` stay the harness's.
- **An MCP server's resources.** The governed copy of an agent's MCP server answered
  `list_resources`, `list_resource_templates`, `read_resource` and `cached_tools` with the
  SDK base class's defaults; they are the server's own.
- **A shell tool's approval.** The SDK's approval of a call with no arguments of that name (a
  `ShellTool`'s commands) showed the person an empty call: it shows the call as the model
  made it.
- **One question for one call (ACP).** A call of an ACP agent's own that declares nothing (a
  tool of Kimi's own MCP servers, any tool of an agent with no profile) asked a person twice:
  governance's question, then the agent's. `bridge.permitted(..., framework=)` asks the
  framework's question only when no person was just asked about the call. An agent with no
  profile is named "the agent" in it, not Kimi Code.

Then, closing what was left:

- **The tools the SDK runs on this machine** (`ShellTool` with a local executor,
  `LocalShellTool`, `ApplyPatchTool`, `ComputerTool`, `CustomTool`) ran with no governance
  and no record: an `rm -rf` the model asked for ran unseen. Each call now goes through the
  bridge where the SDK hands it to the tool's code (`tools.convert.openai_local`): governed
  (one that declares nothing asks; the person sees the call), journaled, recorded, counted;
  reject and edit work; a call the tool's own `needs_approval` held is asked about once.
- **A `handoff(agent, ...)` object's agent** ran ungoverned: it now hands off to the governed
  copy of its agent (read from the SDK's `Handoff._agent_ref`, checked at `h.wrap`).
- **A tool's failure** the SDK formats is recorded as a failed call; **its own timeout** as
  that tool's call that timed out (it said "the run was cancelled"); **`failure_error_function
  =None`** (a function tool's, an MCP server's, an agent tool's) ends the run with the error, as the SDK does; an agent tool that fails under the default policy is a failed child run whose caller reads the SDK's default text (it read the harness's `check failed: ...`).
- **A call that pauses inside** (a question in a tool, an approval inside an agent tool)
  counted twice toward `max_tool_calls`: once now.
- **A `session_input_callback`** that put the earlier turns after the new input got the memory
  context marked into an earlier question: it goes into this run's.

### Fixed: a paused attempt's last words no longer stand for the run's answer in memory

A graph that paused with its own `interrupt()` after the model wrote a draft sent the draft
under the run's answer id (`conversation.answer_id`). The memory service keeps the first
message of an id it has seen, so the answer the resumed attempt gave was dropped and the
draft stood for it in the thread; the draft itself was stored a second time, named by the
resumed attempt's place. A paused attempt now names its last message by what it said
(`conversation.said_id`: who said what, and which time), like every other message that names
nothing else, so a message said again on a later attempt keeps its id and is stored once,
and the answer takes the answer id.

### Fixed: a person's approval of a helper's call counts in the memory service

A call a helper the framework spawned makes (a Deep Agents `task` sub-agent, ReAct's
helpers, an OpenAI agent called as a tool, a Claude subagent) is recorded under the helper's
run, but the decision on it was sent as feedback on the paused run, and the memory service
counts an approval only where the call it approves was recorded: those approvals reached
neither the tool's statistics nor its learned approval rule. The approval's interrupt now
names the helper in its payload (`{"native_helper": {"run_id", "name"}}`, beside a
sub-agent's), and the resume sends the decision on the helper's run
(`Feedback.agent_run_id`, the run the feedback is about).

### Fixed: an in-process run store's temporary folder goes with the store

`LocalRuns` (no `RUNS_URL`) kept a checkpoint's segments in a `trellis-runs-*` folder it made
in the temporary directory, and nothing removed it: one left behind for every harness
started without `RUNS_URL` (a run of this repository's tests left thousands). The folder is
removed when the store is closed, and when a store nobody closed is collected or the process
exits. A folder passed as `payload_root` is the caller's and stays.

### Changed: memory is asked for its whole budget unless the model's window is small

The context pushed for a question was budgeted at 5% of the model's window, between 2,000
and 8,000 tokens, and at 2,000 when the window was not known: a 128,000-token model got
6,400 and a 64,000-token one 3,200, where the memory service's accuracy is measured at
8,000. At 2,000 the service gives the profile 1,000 tokens, and a profile of ordinary size
no longer fits beside the rules that apply: it drops the whole `user` block. The budget is
8,000 now (`clients.memory.CONTEXT_TOKEN_MAX`), whether or not the window is known; only a
window known to be under 32,000 tokens gets less, a quarter of itself and never under 2,000
(`CONTEXT_TOKEN_MIN`, which replaces `CONTEXT_TOKEN_BUDGET`). The service packs what bears
on the question, so a request nothing answers stays a few hundred tokens. The front end's
context and state, which take a share of the same budget, have that share of the larger
one. A model whose window is not known and is in fact small can now be sent more than it
holds: say its window (`ReAct(context_window=)`, or a `context_window` on the target).

### Fixed: values of an earlier request are asked about when they come as memory entries

What the user asked for in an earlier chat reaches a new one as a memory entry
(`## Memories`), not only as a profile line. The note on using them (`agent.REMEMBERED`) came
only with a profile, so with entries alone the model wrote with the earlier values and said so
after the write: the approval was shown without a word of what had been filled in, and a
person who denied it never learned of it. The part of the note on the values of an earlier
request (`agent.EARLIER_VALUES`) now comes with memory entries too; the note sent with a
profile is the same text as before. In the asking benchmark ("Create a low-stock alert for
SKU-00031 too", the earlier alert an entry): asked first in 0 of 3 runs before, 3 of 3 after;
the 76 cases that were there, whose notes did not change, 218 and 221 of 228 over three runs
each. Three more cases of the same kind already passed and are kept (a costly reorder, and
the item itself left out, as a profile line and as an entry).

### Fixed: what a ReAct run is told and sent in the cases a closing run still got wrong

One change set, each part with its cause, a rule that knows no case, and generated tests
that read the words the model is sent.

- **Handing a task to a helper asks no one.** `task` was left out of the run's tool set as
  machinery when the agent has no sub-agents of its own, so its own spec, which declares
  nothing, was read, and a tool that says nothing asks a person: every task handed to the
  general-purpose helper paused for an approval of nothing, on every backend. It carries
  the helper rule's risk now (a read); each call the helper makes is governed on its own, by
  its own name. 8 of 16 generated cells (4 backends, 2 helpers, 2 risks) failed before.
- **The reply a call will send is reserved out of the model's own window.** `FitWindow` read
  `max_tokens` from the request's model; for a model given by name that is the routed
  wrapper, which has none, so nothing was reserved: 950,000 tokens could be sent where the
  provider takes 872,000. The reply is the call's own setting, else the `max_tokens` of the
  chat model that sends the call. It is taken out of the model's own window, which `ReAct`
  keeps beside a `context_window` given: given 32,000 for a model replying up to 64,000,
  the budget was zero. 51 of 60 generated models failed before.
- **A tool call cut at the output limit is said to be cut, and is made again.** The model
  was told an argument was missing, which it could not see was the limit's doing, and wrote
  the same long reply again; a call the provider could not parse at all was taken for a
  reply with no answer, and the model was made to answer with no tools. The refusal says the
  reply reached the output limit before the call's arguments were complete. The reply is a
  model call and counts as a step: a limit that did not count it would never end a run
  whose every reply is cut.
- **A file a command leaves at any depth is known to be the user's or not.** Only files
  directly in the working directory were looked at after a command, so one written a folder
  down (`reports/q3.csv`, or a folder inside `outputs/`) was neither, and an answer said it
  was delivered. Every file under the working directory is looked at (hidden folders left
  out, 5,000 at most), and a new one that is not directly in `outputs/` is named by its
  path. The command tool's description says what cannot be seen: a file written outside the
  working directory.
- **A window no one says is learned from the server's own refusal.** With no
  `context_window`, no profile in the library and a route whose model list names no window,
  nothing was fitted, and the first request over the window ended the run with the server's
  refusal. The refusal names the window in each server's wording the harness already reads
  (OpenAI's and vLLM's, Anthropic's, llama.cpp's): the request is fitted to it and sent
  again, and the window is the model's from then on. Clearing old results and the long-task
  tools stay off for such a model (they are set when the agent is built): give
  `context_window` to turn them on. No setting.
- **A page read of a result is cut as a page, never kept as a table.** A `read_file` page
  cut to fit a small window was kept as a result of its own, named as a table whose rows
  were the numbered lines ("2  SKU-00501"): a join against it matched nothing and the turn
  ran to its step limit. It is cut to its first whole lines and says where to read on in
  the file it read; nothing of it is kept.
- **After a summarization every result the run keeps is still named.** A summary is in the
  summarizer's own words and may name none of the kept results (0 of 56 in one run, whose
  model said it had lost track of them). The request after a summary names each result the
  run keeps that no remaining message names, by its table and its path, once.
- **Benchmark: a computed answer in a "cannot" case scores only against a stated figure.**
  A request written as cannot-be-done-here that a run computed all the same was scored
  right whatever it said. It is right only when the case states the figure and the answer
  holds it, and unverified where the case states none: counted neither way, in its own
  column. `cantdo-02` states what its line of Python prints, from the data.

### Fixed: a cleared result's mark names the result's own columns, and a mark is never cut as a result

Two changes of this release met badly. A cleared result's mark names the table a query
computes over (its columns), and a table of 5 rows or more read whole is kept, its message
ending with a note. The mark was built from the message's words, note and all: a JSON result
with a note after it is no JSON, so it was read as CSV and its "columns" were its data
(`columns: store_S20, category_knitwear, … planned_208, found_190`), about 2,250 characters
where the mark had 132. In a small window that long mark was then cut to fit like a result,
and the cut kept the mark's own text as the result to read on from: "the rest" of it was
the mark, and a query of that path found no rows.

- **A mark is told from the result's rows**, never from the message about it: the table the
  run already keeps, the one a query of that path reads. A cleared stub names the kept
  result's columns the same way; of a file that is Deep Agents' own, nothing is said that is
  not known.
- **A kept result's mark names the kept result's own path**, and a mark is never read as a
  result wherever the graph wrote it. Deep Agents' summarization writes the cleared request
  back as the graph's state and saves its messages as files, so the mark came to stand at
  the call's path: a query there read the mark's one line (15 failed queries in 3 live runs
  of `long-flow-02` with a backend, once the mark's columns were right and the model
  followed it). The kept path is read whatever the graph does with its messages; a result
  the run does not keep, whose only copy the graph replaced, is said to be gone.
- **A mark is no result.** A cleared result's line and a stub are shortened as marks when
  the window has no room (the path and the table's name stay, what is left out is counted);
  nothing of a mark is kept, and none is cut as rows are.
- **A result that begins with another's rows is itself.** A longer page or a wider date
  range starts with the rows of an earlier result of the same tool; it was taken for a copy
  of that one, and a count over 20 rows answered 10. A copy is the kept rows followed by
  their note, nothing else.

Seen in the closing acceptance run: 23 of 600 case runs carried such a mark (0 of 600
before), every run of `long-flow-02`, and one of them spent five of its twelve steps reading
marks. The generated tests now read the words the model is sent, not only what a query
returns: for a result of 1 to 9,000 rows as CSV, a JSON array or a JSON object, in each
state (shown, kept, cleared, cut, saved by Deep Agents, that stub cleared), the columns
named are the result's, the path and the table named give every row, and no request holds
a mark of a mark — on every backend. 25 of the 90 end-to-end situations failed before
(every cleared JSON result, and every result of a crowded step).

### Fixed: a model LangChain has no profile of replies up to its own output limit, and a cut answer says so

For a model newer than the installed LangChain package, `ChatAnthropic` falls back to 4,096
output tokens. A reply that thinks first then stops mid-call or mid-answer: 74 of 22,385
saved model calls ended at exactly 4,096 tokens, 41 of them a tool call cut before its
arguments, and one final answer went out as a success stopping at "| S20 | £".

- The route's record of the model names its output limit beside its window (Anthropic's
  `max_tokens`, a server's `max_output_tokens` or `max_completion_tokens`): asked once for
  both, with the same 5 seconds and the same kept answer, and used where LangChain knows
  none of the model and filled in its own fallback. A model LangChain knows keeps its
  profile's limit; a chat model object, or a call's own `max_tokens`, is the caller's; a
  route that names none leaves the library's fallback. No setting.
- The route's limit is used up to the share of the window the library gives the models it
  knows (its largest `max_output_tokens` of `max_input_tokens`, read from the profiles it is
  installed with: 64,000 of 200,000 today). Input and reply must fit the window together: a
  route reporting 128,000 of a 200,000 window would have left input 72,000, and one
  reporting a reply as large as its window none. The three models this applies to today
  report 128,000 of 1,000,000 and get it whole. A route that names no window to bound the
  limit by leaves the library's fallback.
- An answer whose reply reached the output limit is still the run's answer, as it is, and
  the run says so: the `output_limit` warning a reply cut before any answer already raised,
  on the stream and in the log. It is not continued.

### Fixed: a cleared result the run already keeps as a table, and a stubbed one, are queried by their call's path

A cleared result's mark names its call's path (`/large_tool_results/<call id>`) and the query
over its rows. Two results asked for by that path were still not read as their rows:

- **a table read whole** (30 rows or more: the run keeps it, and its message carries the
  note after the rows). The query was given the message as it stood, note and all: a JSON
  result became one row of thousands of columns, a CSV's figures became text
  (`sum(VARCHAR)`). It is now the table the run already keeps of it.
- **a result too large to show.** The run keeps it under a path of its own and shows a stub
  naming it; asked by the call's path, neither reader found it. Both now follow the stub to
  the kept file.

A generated test holds a result of 1 to 9,000 rows, as CSV or JSON, in each state — shown
whole, kept, cleared, saved by Deep Agents — and queries it by the path named for it: 36 of
its 144 situations failed before.

### Fixed: a held result is queryable wherever it lives; the query hands its rows to the sandbox; a file not kept is said at once

`ReAct(backend=sandbox())` answered differently from plain `ReAct` on the same question
(`long-data-03`: 105 SKUs under a 40% margin in 2 of 3 runs, where the exact query gives 104).
The cause was one cell of a wider gap, closed as four rules that hold for any task
([sandbox.md](docs/sandbox.md), [react.md](docs/frameworks/react.md)):

- **A result the run holds is queryable exactly, wherever it lives.** With a backend whose
  files are outside the graph's state (a sandbox, a disk, a store) the run now keeps a large
  result, as without a backend: Deep Agents kept the only copy where the harness cannot read
  it, so no table was named and `query_result` was not offered. A result cleared from the
  window, and one cut to fit a small window, now name their table too when they hold rows, and
  the query reads them (`middleware._paged`, `_readers`, `_kept_read`, `_rows_said`). A
  generated test crosses five backends with four ways a result is held: 13 of its 20 cells
  failed before.
- **The query hands its rows to the sandbox.** With a sandbox, a query's result of more than
  one row is kept by the run and is a CSV file there; the answer names it, and both tools'
  descriptions say which does what (the query: exact counts, sums, filters, rankings, joins;
  code: a chart, a formatted workbook, a script, starting from that file).
- **Same output with or without a sandbox.** Live, three runs: `long-data-03` went from 1 of
  3 right to 3 of 3 (104 each time, by `query_result`, no command); the other seven files and
  code cases stayed or became right every time.
- **A file for the user is kept, or the model knows at once that it was not, whichever tool
  wrote it.** A file written by its path (`write_file`, `sandbox_write`) outside the kept
  folder is told so by its path; a command that left a new file directly in the working
  directory is told that file is not the user's. The chart case kept its file in 3 of 3 runs
  (1 of 3 before). The model's command is told in Deep Agents' own entry for it
  (`SandboxBackend.aexecute_with_offload`), so a file tool's parsed output never carries the
  note. This costs one listing in the sandbox after each command of the model's.

### Fixed: at a small window a tool the question needed was dropped in silence

When a request does not fit, `FitWindow` narrows the tools. Its last step kept only the tools
the run had already used whenever the latest turn alone filled the window, so a tool the model
had not called yet was gone when it reached for it, and nothing said so: on `long-multi-02` at
8,000 tokens the model answered "none of my tools hold supplier delivery data" in 4 of 6 runs,
of a tool that had been dropped a step before to save 89 tokens of a request still 2,000 over.
The rule now, at every window size:

- a tool the narrowing does not keep is still offered by its **signature** (its name and
  arguments, the description left out) while that fits — always when the tool hints chose it,
  and for the first 8 besides the kept ones whatever the room. Only a request the model's
  server itself refused gives those up;
- a tool held back altogether is **named** after the instructions, with what finds one
  (`tool_search`, where the run has it, which is never held itself). The line's room is set
  aside, so a fitted request still fits.

### Fixed: a `ReAct` sub-agent that names no tools has the run's tools

Deep Agents gives a sub-agent that names no `tools` the main agent's. `ReAct`'s tools are not
built into its graph: they are offered per model call (`HarnessTools`), so such a sub-agent,
and Deep Agents' general-purpose helper, was offered the seven file tools only. A helper asked
for a stock level searched files for it and handed the question back, and the parent made every
call itself. A declarative sub-agent that names no `tools`, and the general-purpose helper, are
now given the run's tools, governed as the parent's are (`react._native`); one that names its
own `tools` has exactly those. Over three live runs each, the two sub-agent cases stayed right
3 of 3: `advflow-02` went from 67, 39 and 68 model calls to 42 each time (24 s to about 16 s),
`advflow-03` from 13, 7 and 10 to 9 each time (about 13 s to about 10 s).

### Changed: a handful of rows is a table too, and two figures are computed by the query

From two 8-row summaries a model divided £1,123,571.70 by 11,978 units by hand and gave £93.79
for £93.80 (`planning-03`: wrong in 4 of 9 runs, and in some run of every arm): a table under 30
rows was read as it was, with no `query_result` to compute with.

- **A table of 5 rows or more that is read whole is kept as a table** (`results.TABLE_ROWS`,
  was 30), so the model is offered `query_result` and told to compute with it.
- **Its note names dividing, and the query runs on `sql` alone** (`results.QUERY`,
  `tools.query`): `SELECT 1250.50 / 37` computes with the figures it names, no file
  needed (before, a query without `file_path` or `results` was refused).
- Measured with example figures that are no case's (`SELECT 1250.50 / 37`): `planning-03`
  right in 8 of 9 runs (5 of 9 before) and `long-flow-02` in 5 of 6 (4 of 6). Every wrong run
  is one where the model answered without calling the query; every run that called it is
  right. An earlier 9 of 9 was measured with the case's own division as the example and does
  not count. It costs a model call where the model now computes.

### Fixed: with memory off, an out-of-date value of the turn before was reused

With memory off, `ReAct` replays the conversation's earlier turns from the run store, each
result of a live tool marked out of date. Memory's replay also says so in the run's context,
with the question ("for a value the question needs, call its tool again; never answer with an
earlier one": `snapshots.note`); the run store's replay did not (`Agent.recalled` dropped the
list of out-of-date tools). The mark sits on the old result, and a model asked for a new figure
that only uses the old one as an input took it as it stood: on `freshness-03` the on-hand count
of the turn before went into the next turn's reorder formula in 3 of 3 runs, and on
`freshness-04` in 3 of 3. The note is now said with the question on that path too, and only
when a replayed result is out of date: the four freshness cases went from 20 of 27 turns right
to 26 of 27 over three runs.

### Changed: `ReAct` alone switches to Deep Agents' machinery by itself; asked for, it is Deep Agents from the first step

One `ReAct()`, two ways to run it, said by what it already takes
([react.md](docs/frameworks/react.md#two-ways-to-run-it)):

- **`ReAct` alone** is Deep Agents' graph underneath with its machinery held: an ordinary task
  is sent what `create_agent` with `middleware()` is sent (the same tools and instructions, a
  large result kept by the run, the window fitted by `FitWindow`). Once the conversation is
  past half the model's window — the run's own behaviour, where old tool results start being
  cleared — the model is offered planning (`write_todos`), Deep Agents' scratch file tools and
  its general-purpose `task` helper, for the rest of the thread.
- **`ReAct` asked for a Deep Agents capability** (`backend=`, `subagents=`, `skills=`,
  `memory_files=`, `permissions=`) offers Deep Agents' tools, planning and `read_file` from the
  first call (before: its scratch tools and helper came with the third model call), unless the
  window is known to be under 32,000 tokens.
- **The switch no longer reads the request's wording or counts model calls.** "Step by step",
  "a plan" and a list of three items matched no request of the long set (0 of 45 turns) and one
  of the benchmark's 256 cases, which asked for no steps. The switch after three model calls
  changed the tools in the middle of a run, so a provider's prompt cache was rewritten: on the
  12 pilot cases cached input fell from 456,000 to 337,000 tokens and the cost rose a third,
  and on 57 long and complex cases it fired in 26 runs and no model call used a tool it
  offered. `automatic.LONG_AFTER`, `STEP_ITEMS`, `calls_made` and `asks_for_steps` are removed;
  `automatic.long_task(request, window)` is the conversation's size against `LONG_AT`.

### Fixed: kept tables of one tool's calls made side by side

- **Each table says what its call asked** (`middleware._table`): two results of one tool
  called in one reply were numbered by which answered first, so the model took
  `sales_summary_2` for its second call and computed a profit from swapped columns (the priced
  pilot's `aggregate-04`, on three of four frameworks: a wrong result, one step more over
  80,000 tokens). They are now `sales_summary_revenue` and `sales_summary_units`: the tool, and
  the arguments that tell the reply's calls of it apart. Calls one after another are numbered
  as before.
- **Deep Agents' stub of a result it saved names its table** (`middleware._stubs_tabled`):
  `query_result` reads the saved file as a table, but Deep Agents' own stub does not say which,
  and a model took `result` for it in another file's query (`long-data-02` on `ReAct` with a
  backend: a failed query, a step more). The request's copy of the stub now says the table and
  its columns before the preview, as the harness's own stubs do.

### Fixed: a model LangChain has no profile of had no context window

A model name's window came only from the table of profiles bundled in the installed LangChain
package. A model released after that package, a local model, or any name on a provider other
than OpenAI and Anthropic had none: `ReAct` ran with clearing old tool results and the
long-task tools off, `FitWindow` stepped aside, and the memory context's and the replayed
conversation's budgets were the fixed ones (the priced pilot ran every arm so: the routed
model was newer than the table).

- **The route is asked when the table names no window** (`ModelGateway.window`,
  `gateways.routed_chat_model`): Anthropic's Models API through Anthropic's own client
  (`max_input_tokens`), an OpenAI-compatible endpoint's `/models` (the field its server names:
  `max_input_tokens`, `context_length`, `context_window`, `max_model_len`,
  `max_context_length`), with the route's auth, within 5 seconds, once per route and model.
  No table of windows is kept here.
- **A route that says none leaves the window unknown**, as before (said once): a model list
  without the model or without a window, an endpoint that does not answer (asked again by the
  next build). `context_window=` wins over all of it.

### Fixed: ReAct on Deep Agents on a small window

On a 6,000-token window (adversarial `advconv-05`, `react --variant deep`, memory off) the
summary requests Deep Agents sent were up to 23,156 tokens, 13 of 23 over the window; after a
summary the model answered the conversation's first question again in place of the run's. Five
causes, all the harness's; Deep Agents' summarization is left as it is:

- **The earlier turns replayed with memory off are sized for the request**
  (`conversation.earlier(budget=)`, `Agent.conversation_budget`), as memory's window is: each
  turn's question and answer from the newest back, then the newest turns whole while they fit.
  Before, the newest 10 turns went whole whatever the window, and Deep Agents summarized all
  of them in one request each turn (on `create_agent`, `FitWindow` left them out).
- **The task kept ahead of a summary is the run's question** (`ModelHooks`), not the first
  user message of the state, which with earlier turns replayed was the oldest turn's.
- **The readers of a kept result stay offered when `FitWindow` narrows the tools**
  (`query_result`, `read_file`): the preview says to call them.
- **A request refused after a summarization is fitted again with the latest turn's results
  cut** (Deep Agents keeps a `read_file` page whole): before, the same request was summarized
  again and refused again.
- **The model's tokens per counted token are learned where it is called**, of the request it
  was sent (`ModelHooks`), not of the request before a summarization made it smaller.

### Fixed: a kept table queried under its note's name reads its own rows

A kept table's note names the table to query (`FROM sku_catalog_2`). Inside a query, the
results a Deep Agents graph saved in its state joined the query and were named first (a
`ChainMap` iterates its last map first), so they took that name: the footwear note's table
read another turn's denim rows (ReAct on Deep Agents, adversarial `advconv-05`; the
`deepagents` adapter too). `query.tables` now names the conversation's earlier turns' results
first, then the run's own, then the graph's, which join only while a query runs. Adapters with
no graph files (OpenAI Agents, Claude, Kimi, `create_agent`) keep the names they had.

And a kept table `FitWindow` cuts to fit the window keeps its note (where it is kept, the table
to query) whole after the cut, its rest read from the same kept copy: the note was at the end
and was cut away, so the model guessed `FROM sku_catalog` and read the earlier turn's knitwear
rows for outerwear (`deepagents` adapter, 6,000 tokens).

### Added: the model is told what day it is

Memories, fetch times and earlier conversations carry dates, and nothing told the model the
day to measure them against (only `schedule_task`'s description did). The harness now says it
in the instructions, one line after them: `It is now Wed 2026-11-11 (UTC; the user's zone is
not known).`, in the user's zone when their app sends one. To the day and from the run's start,
so every model call of a run reads the same line and a prompt cache loses its prefix once a
day at most; the date only, no instruction with it.

- Where: `ReAct`, a `create_agent` or `create_deep_agent` graph (with the harness's middleware
  or without), and OpenAI Agents (after each agent's instructions). Claude Code and Kimi Code
  say the date themselves, so the harness adds none there. A hand-built `StateGraph` and a
  function call their model in their own code and get none.
- One wording of when it is (`snapshots.said`): `schedule_task` keeps its day and hour, from
  the same function. A run's start and its seconds read the one clock (`snapshots.now`).

### Fixed: a failed `query_result` names the closest tables, and stays short

A failed query's error listed every table the run keeps with its columns. With twenty saved
results it was about 1,000 characters, `FitWindow` cut its middle on a small window, and the
table the query meant went with it. The error now names the five tables closest to those the
query reads (DuckDB's own `get_table_names`, ranked with `difflib`; `result` is the queried
file's), each with its columns, then how many more there are and the query that lists them all
(`SELECT table_name FROM information_schema.tables`): about 500 characters however many
results the run keeps.

### Fixed: only a call that returned a value is marked out of date

A live tool's result from an earlier turn is marked out of date, so the model fetches the value
again. The mark went on every earlier tool message of the tool, also one of a call with no
result: in a thread a graph keeps, a call a newer message stopped read "[slow_count's result
from an earlier turn: out of date ... call slow_count again for the current value]" before
"slow_count has no result: its run stopped ...", and a failed call's error the same; in
memory's replay a failed call's error was marked too, and its tool counted among those whose
values are out of date. A call that failed or was stopped returned no value: its result is
sent as it was (`middleware._earlier`: the tool message's `status`; `snapshots.out_of_date`:
the run record's status).

### Added: "Always allow", kept in the memory service

With a memory service, a write's approval offers a fourth choice, **always allow** (`always`,
`remember="always"`; trellis-contracts 0.7.1): the tool's later calls asked of the same person
or role run without asking in every conversation of that user until the permission is revoked.

- Kept in the memory service's tool permissions (`PUT /v1/tool-permissions`, trellis-memory
  0.6.5) with the harness's service key, once, in the background (spooled when the service is
  down); read once per attempt when a call first asks, so a revoke counts from the next run.
- Never for an `irreversible` tool (neither offered nor honoured), nor where the tool's
  `approve_when` or a `before_tool` hook's `Ask` decides the call; a framework's own tools keep
  their own flow.
- Offered only where a pick is kept: an attempt that could not read the permissions, or a key
  that cannot grant (an issuer's token), offers the three choices. Permissions that cannot be
  read fail closed: the call asks, and a warning is logged once.
- **A hook's `Ask` beats every allowance:** a `before_tool` hook's question is asked whatever
  was allowed for this conversation, always, or by the run's schedule (as a PreToolUse "ask"
  in Claude Code). Before, an approval for the conversation or a schedule's covered it.
- Without a memory service nothing changes: three choices.

### Changed: a write asks, once a conversation, as in Claude Code

A tool the harness runs that says it writes now asks the person before it runs (it ran,
announced). The person picks, as in Claude Code: **allow once**, **allow for this
conversation** or **deny**. An `irreversible` tool offers allow once and deny only: it asks at
every call.

- **"For this conversation"** covers the tool's later calls in every later run of the same
  thread for the same user — a resumed run, another worker, a child run — never another
  thread, user or tool. It is read from the run store's history of answered interrupts
  (agent-runs' `resolutions`; `LocalRuns` keeps the same): no new setting, no contract change.
  `remember="run"` is that choice; before, it reached the rest of the run only, and covered an
  `irreversible` tool too.
- **The choices travel with the approval**: its interrupt's `options` (`once`, `run`,
  `reject`), and the question says them for plain chat. AG-UI takes the picked value as the
  resume's `payload`, A2A the choice in words, `agent.resume(id, "answer", answer="run")` too.
- **What does not ask:** a write the run keeps to itself — the schedule tools, the memory
  tools, the sandbox, a sub-agent's call, Deep Agents' files and `execute` — is announced as
  before. A framework's own tools (Claude Code's and Kimi's built-ins, OpenAI hosted tools)
  keep their own permission flow. The catalog's `risk` and `approve_when` are unchanged: a rule
  that does not hold still lets the call run, announced.
- **A schedule set in chat** names the writes its runs will call (`schedule_task(tools=)`);
  its confirmation allows them for its runs (kept in the schedule's metadata) or leaves them to
  ask at each run. A write it does not cover pauses in the chat's thread. A schedule made from
  code allows nothing.
- An OpenAI agent holds a write at its own gate (`needs_approval`), so the approved call
  continues the paused run.
- A call denied earlier in a run stays denied when a later approval in the run allows its tool.
- Way 2: `governed(fn, gov, side_effects="write")` and `Governance.check` ask about a write
  too (`on_ask`).

**Upgrading.** Nothing fails at startup. What changes: a write tool now asks once per
conversation.

| What you see | The fix |
|---|---|
| a `write` tool that ran now pauses (`Approve <tool>? <tool> is write. Allow once, allow for this conversation, or deny.`) | nothing: the person allows it once or for the conversation; an app renders the interrupt's `options` |
| an `irreversible` tool approved with `remember="run"` asks again | nothing: it asks at every call |
| a scheduled run's write pauses | name the tool in `schedule_task(tools=)` so the person allows it when confirming; a schedule from code: answer the pause |
| a tool that only reads is declared `write` | `@tool(side_effects="read")`, or its `risk` in the catalog |

### Changed: a tool that does not say what it does asks a person

A risky call must never run without approval. A tool that declared nothing was taken as
`write` and ran, only announced: in a live test a purge, a bulk price change and a read of
another account's data all ran. Now, as in Claude, ChatGPT or MCP, a tool that says nothing
needs the person's permission — one behaviour for every way a tool comes in. Nothing is
refused: no tool fails at startup.

- **A tool with no declared side effects asks at each call** (`<tool> has not declared its
  side effects.`): `@tool` alone, `tool(fn)`, a bare function in `tools=[...]` or given to
  `agent.tools.use`, `governed(fn, gov)`, a source of your own, a LangChain tool of your graph
  with no `metadata["side_effects"]`, an OpenAI Agents `FunctionTool`, a built-in of Claude
  Code or Kimi the harness has no word on, `Governance.check(tool, args)` given none. Its risk
  is none (`unknown`), and governance asks about a risk it does not know (it announced and
  ran). Declared tools are as before: `read` runs, `write` runs and is announced,
  `irreversible` asks.
- **An MCP tool's annotations are read with the protocol's defaults** (`readOnlyHint` false,
  `destructiveHint` true): a tool is `irreversible` unless it says it only reads (`read`) or
  says `destructiveHint: false` (`write`). A tool with no annotations is `irreversible`: each
  call waits for approval. Before, anything not marked destructive was `write`.
- **The tools that say nothing are named once**: when an agent's tools are listed, one warning
  (`trellis.tools`) lists those that declare no side effects, with what to add.
- The catalog is told the same (`destructiveHint: true`, never a `side_effects` of the
  harness's) for an MCP tool that says nothing, so an administrator's risk there stays: it is
  the place to say what a tool you cannot annotate does. Any other tool that declares nothing
  is published saying nothing of it: the catalog keeps no risk (`null`, trellis-memory 0.6.6)
  and the tool's own rule decides, so it asks with allow once, for this conversation, deny
  (always allow with a memory service). A catalog entry with no risk leaves the tool's own
  declared risk; before, it was read as one (`write`, then `irreversible`), and an undeclared
  tool asked every call with no "for this conversation".
- A built-in the harness knows only by name (the graph's state tools, Deep Agents' file tools
  and `execute`, declared a write) is the framework's own: a tool of yours that shares the name
  (`delete`, `write_file`...) says what it does itself, as any other.

**Upgrading.** Nothing fails at startup. What changes: a tool that declared nothing now pauses.

| What you see | The fix |
|---|---|
| a tool of yours that ran now pauses (`<tool> has not declared its side effects.`), and a warning names it | `@tool(side_effects="read")` (or `"write"`, `"irreversible"`) on the function; `tool(fn, side_effects=...)` for one given bare; `governed(fn, gov, side_effects=...)` |
| an MCP tool that ran now pauses (`<server>-<tool> is irreversible.`) | your server: annotate the tool (`readOnlyHint: true`, or `destructiveHint: false` for a change that can be undone); a server you do not own: its risk in the catalog, `put_catalog([{"name": "<server>-<tool>", "side_effects": "read"}])` |
| a framework's own tool that ran now pauses | a LangChain tool: `tool.metadata = {"side_effects": "read"}`; any other: its risk in the catalog (as above), or build the tool with the harness (`@tool(side_effects=...)`) |

### `query_result` over complex data

- JSON rows held at any depth, each array an object holds (`result_<key>`), JSON Lines, markdown
  tables, a one-column CSV, a quoted delimiter in a header, column names in any script.
- `NULL`, `N/A`, `-` among numbers are missing values; times with a zone are compared and
  written in UTC; money and percentages stay text, and a failed sum says how to convert them.
- A PDF, an image or binary data is refused, pointing to the sandbox; HTML or XML is queried as
  its lines, the answer saying so. A workbook is a table a sheet, under its header, merged
  cells filled.
- A text's typed tables are kept as Parquet by its content: the next query of it (this run, a
  later turn, a past result) reads them, not the text. Five queries of a 10 MB result: 3.1 s →
  0.8 s (CSV), 5.0 s → 1.3 s (JSON).

### A journal is read from its segments only

- A run's checkpoint lists its journal's segments (`trellis_segments`); that is the only
  format. A checkpoint holding the journal inline, or naming it as a run artifact
  (`journal_ref`), is refused (`HarnessError`). No migration: dev databases are reset.
  `LocalRuns` keeps the same format in its folder.

### Past results by reference

- `query_result(results=["msg_..."])` loads past tool results (the ids an earlier
  conversation's steps name in the memory context) as `t1`, `t2`… beside the run's own, read
  through the memory service's existing message read and its audience rules; a new chat is
  offered `query_result` once its context names one. "Which tool did you use last time?" is
  answered from those steps. Live history eval: `explain-08` and `hist-long-01` now pass.

### `query_result` needs DuckDB

- DuckDB is a required dependency; the SQLite fallback is gone, so the model sees one dialect
  and the query suites run once ([ADR 0007](docs/adr/0007-query-engine.md)). The `query` extra
  stays, empty. `trellis doctor` fails its "query engine" check when `duckdb` cannot be imported.

### Benchmark

- `--scripted` drives Kimi in the complex suite (`--adapter kimi`): the scripted model is served
  as a chat completions endpoint Kimi calls, keyed by the message it answers (a resumed Kimi is
  told to repeat the paused call, which a script in order answered with the next turn), and
  every request Kimi sent is kept, so `Turn.sent` is checked on Kimi as on the LangChain
  adapters. Before, a scripted Kimi run hung (Kimi was given the scripted model's text as a
  model name). Adversarial and break-and-fix: 58 and 70 cases, every strict score right, 1 of
  400 checks failed (`advconv-05`'s `fits 6000`: Kimi keeps its own context, larger than the
  case's window). Kimi is given the suite's step limit in its own slot
  (`KIMI_LOOP_MAX_STEPS_PER_TURN`), as the other adapters are.
- With a real model the Claude and Kimi CLIs call it themselves: what they send is not seen, so
  the checks of what was sent (`has`, `lacks`, `nonblank`, `fits`, `repeats`, `answered`) are
  not made for them. Before, a `has` failed and a `lacks` passed on nothing (Kimi, Haiku,
  adversarial: 48 of 52 failed checks were these).
- A history set for the complex suite (`--cases history`): earlier chats recalled, compared,
  trended, merged and explained, and how an answer was got (memory or tool, source and read
  time, after a resume); 10 new cases and 12 existing ones of those kinds.

### Credentials for tools

- `Harness(credentials=async (run, target) -> headers | None)`: the run's user's headers on each
  call of an `mcp(...)` server over HTTP, an `openapi(...)` operation and an `a2a(...)` agent,
  over their fixed `headers=`, asked once per run and target (a scheduled run asks again);
  `NeedsAuth(url)` pauses with "Connect your <name>"; a failing hook fails that call. With a
  hook, MCP connections are per server, tenant, user and headers, closed when idle. The token is
  on the request only. No hook: unchanged (`trellis.harness.credentials`, docs/tools.md).

### Schedules that run unattended

- Chat schedules pause themselves after 3 failed runs in a row (`ERROR`, `TIMEOUT`); that
  run's error says so in the thread, its record and its `run.finished` webhook. One untouched
  for 90 days asks at its next run "Still need this?" (Keep, recommended, or Cancel), paused
  until answered. No settings; `trellis_failures` and `trellis_touched` in the metadata.
  agent-runs #26 adds one run at a time and a stable per-schedule jitter. Docs: a user who left
  (`on_run_start`); `benchmark.load.threads` (1,000 thread streams, 100 fires).

### When and how to ask

- One note with every memory context, for every adapter (`agent.ASKING`): ask only when the
  model cannot find out itself, the choice is the user's and a wrong guess costs; otherwise act
  and say the assumption, or the remembered rule and its date; values filled from an earlier
  one are said, or asked about first when costly, old or in conflict; a superseded rule is asked
  about. The app's own tool when one fits, else the question tool, else the reply.
- `ask_user(..., multiple=False, recommended=None, why=None)`: several picks, one option
  recommended (`Option.recommended`, trellis-contracts 0.6.5) only with a why; a choice also
  takes another answer in words; plain chat reads the options numbered.
- Claude's `AskUserQuestion` is the run's question, as Kimi's is.
- A question's recommended option comes first wherever the model listed it: in the choice,
  the app's own control and the numbered text.
- The note asks "which one" when several things fit the user's words (it had made the model
  pick one or answer for each, against the app's own "ask which one is meant": break-and-fix
  clarify 17/21 → 21/21); how to use remembered rules, preferences and values
  (`agent.REMEMBERED`, said with their date) comes only with a context that has a profile, next
  to the asking note: no tokens on the other turns, and the dated cases 28/36 → 36/36.
- The schedule tools keep only what is theirs: the general asking words are the note's; the
  "update it" choice is recommended by the flag, not by its label.
- `schedule_task(in_minutes=)` says the minute it counted from (the tool's text says only the
  hour, for the cache): the model no longer "corrects" 03:53 for "in 2 hours" at 01:53 to 03:00.
- `ask_user` is offered by itself to a run a person is known to read: one a chat front end
  started (`serve_chat`; `Runtime.attended`), not one its schedule fired, on a framework with
  no question tool of its own (Claude Code and Kimi Code keep theirs: `Adapter.asks`), never
  on a graph with its tools bound. `without=["ask"]` turns it off (an eighth name). A run
  started from code (`run`, `stream`), queued, fired, called over A2A or by another run gets
  it only when its agent declares it (`tools=[ask_user()]`, as before): nobody is known to
  answer a pause there. A tool of the agent's named `ask_user` wins; one declared is offered
  once. `serve_chat` records a run's front end even when the app sent nothing with the
  message.
- `ask_user(fields=[{name, label, kind, options, multiple, default, why, required}])`: several
  values in one question, one answer (an object: what the person gave over what was filled
  in). No contract change: the `ask_user` component with `props.kind == "form"`, an object
  `expects` with `ui_schema`, a numbered list in plain chat. A single question is the
  one-field case; `recommended` with `why` on a number or a date shows the value filled in.
- Values a request leaves out that an earlier one had are confirmed once before they are used
  again, shown filled in (`agent.REMEMBERED`): they were filled and said. A rule, preference
  or value said more than six months ago, or one a tool shows changed, is confirmed the same
  way.
- Where the app declared tools of its own, `ask_user` says a value one of them asks for is
  asked with it (the app's date picker used in 1 of 8 runs before, 13 of 14 after).
- The asking benchmark (`benchmark/cost/asking.py`): `--ask auto|declared|none` (was
  `--no-ask-tool`), `--variant`; cases for an ambiguous target (14 products, two), a
  remembered value that is partial, old or changed elsewhere, a recommendation the data
  supports, an answer in words or outside the options, several values missing at once, a run
  nobody reads, and risky actions (cannot be undone, money, many rows, people outside, an
  instruction in a tool's result, an earlier yes to something else): each waits on its
  approval whatever the model does; a tool that declares no side effects is measured apart.

### Schedules set in chat

- Every agent has `schedule_task`, `list_schedules`, `update_schedule` and `cancel_schedule`
  (`without=["schedules"]` turns them off; `h.schedule_tools()` for a graph built with its
  tools bound): "every Monday at 8 check stock and tell me if it's under 5". A fired run is
  the same agent, user and thread, so its reply is saved into that chat; with a condition it
  answers `NO_REPORT` when the condition does not hold, and a condition that still holds is
  reported once. Limits: `TRELLIS_SCHEDULE_MIN_INTERVAL` (300 s), `TRELLIS_SCHEDULE_MAX_PER_USER`
  (20), per user `Harness(schedule_limits=)` ([docs/runs.md](docs/runs.md#schedules-users-set-in-chat)).
- Schedules are in the user's zone without asking it: AG-UI `forwardedProps.timezone` (the
  browser's), else UTC; `schedule_task` says the day and hour in it, so "tomorrow at 9" has a
  date, and takes `in_minutes` ("in 2 hours"). It is told when to schedule and when not, to
  check a condition once now and say its value, to say what its tools cannot watch and offer
  what they can; a fired run is offered no schedule tool (one made a new schedule from "remind
  me daily"). The tools ask with buttons (`Runtime.ask(options=)`) for a schedule like one the
  user has (update it, recommended, keep both, cancel; the same again is "already set"),
  `cancel_schedule(id="all")`, several that fit (`ids`) and a vague ask (`choices`). The eval
  set: `python -m benchmark.cost.schedules` (61 cases).
- AG-UI `GET {path}/threads/{thread_id}/events`: every run of a chat thread, scheduled ones
  included, live and after `Last-Event-ID`, fed by one poller per process.
- `RunStore.schedules` gains `list`, `get`, `update` and `delete` (`LocalRuns` has them); a
  `LocalRuns` fire takes the thread the schedule's metadata names. **Requires** agent-runs with
  `RunSummary.thread_id` and `schedules.list(on_behalf_of=)`.

### Docs

- Docs name the arguments the code takes: a run's time limit is `run_timeout=` (scenarios.md,
  runs.md), Langfuse's address is `LANGFUSE_BASE_URL` (README), the governance example binds a
  tenant id it names (there is no `h.tenant()`), and blocks/evaluation.md no longer points at the
  retired `eval_sample` setting.
- reliability.md: a scheduled run carries its time limit and agent version (agent-runs copies
  them into each fired run), and a `ReAct` model client makes one attempt only on the `bifrost`
  route; on the other routes the client keeps its own default retries.

### Background agents moved to their own branch

- Background sub-agents (ADR 0008), a run's inbox and a person's pause (ADR 0009) and the
  transcript of a run or sub-agent were reverted from `main` (#160, #166, #167) and continue on
  the `background-agents` branch, with the matching agent-runs and agent-contracts work, until
  they are ready to land together. `main` keeps every generic fix merged meanwhile (#162, #165,
  #168) and requires `trellis-contracts>=0.6.3,<0.8`: 0.7.0, the contracts without the run inbox,
  is admitted.

### Claude Code and Kimi Code ask as they do natively

- A `Bash` command that only reads (`ls`, `cat`, `wc`, `which`, Python that opens files to
  read) runs without a pause, as their `Read`, `Glob` and `Grep` do: the read-only check moved
  from the benchmark into `trellis.harness.governance.shell` (one implementation).
- Any other call follows the framework's own permission check, then a person: Claude Code's
  `PreToolUse` hook applies your hooks and governance, then leaves the call to Claude Code's
  mode and rules (`allowed_tools`, its settings: your allowlist, as it is); what it would ask
  its user, its `can_use_tool` asks the harness (your own callback, else a person). Kimi
  asks over ACP: after hooks and governance, a person is asked, as Kimi asked (`Write` and
  `Edit` outside a git working tree included, where they ran announced before).
- A file Claude Code or Kimi Code writes in `outputs/` of its working folder is the user's, as
  a sandbox's `outputs/` is (`sandbox.delivered`, `Adapter.workdir`): kept in the file store on
  success, announced with the `file` event, then removed from the folder. Each is told so
  (`sandbox.DELIVERS_HERE`: Claude after its system prompt, Kimi before its prompt).
- `Bash`'s risk is `write` for both (it was `irreversible`: every command asked): a team that
  wants every shell command approved gives it the risk `irreversible`, or an `approve_when`,
  in the tool catalog. `bridge.asked` asks a person about a call its framework asks about.

Model access is provider-neutral ([ADR 0004](docs/adr/0004-provider-neutral-models.md)): a
model name goes to any OpenAI-compatible endpoint, to Bifrost, or to the provider's own SDK,
chosen by name from one registry, and the memory service is told the agent's endpoint so its
own model work follows the harness's configuration.

### Breaking

- `BIFROST_URL` and `BIFROST_VIRTUAL_KEY` are gone, with no alias: `LLM_BASE_URL` (the
  endpoint's `/v1` root) and `LLM_API_KEY` (its bearer key; Bifrost's virtual key), with
  `LLM_GATEWAY=bifrost` for Bifrost. `Settings(bifrost_url=, bifrost_virtual_key=)` are
  `Settings(llm_base_url=, llm_api_key=, llm_gateway=, llm_auth=)`. An endpoint without
  `LLM_GATEWAY` is `openai-compatible`: model names as given, models only.
- `trellis.harness.gateways`: `ProviderKey` is `ProviderSDK`, `GatewayKind` is each route's
  `name` (its `LLM_GATEWAY` value), `bifrost_wins` is the route's `announce`; `GATEWAYS` maps
  names to routes and `model_gateway` picks from `registry()`.
- The memory service is sent the agent's whole model endpoint (`trellis.contracts.Endpoint`)
  where it took a virtual key (`MemoryScope.register_endpoint`, was `register_model_key`),
  needing trellis-memory with endpoint registration; trellis-contracts 0.6.3 and bifrost-sdk 0.4.

- A conversation is its tenant's, its user's and its thread's (`identity.conversation_key`, a
  digest of the three): the memory service's thread (its window, the OpenAI Agents
  `MemorySession`, `h.add_document(thread=)`), a graph's own checkpointer `thread_id`, the turn
  a busy conversation waits for in process, `start`'s default `concurrency_key`
  (`thread:<conversation>`) and the trace's `session.id` were keyed by the bare thread id, so
  two users (or tenants) giving the same `thread=` shared them. A conversation kept under a
  bare thread id before this release is not continued: its next run starts a new one. Read a
  graph's own state with `thread_id=await h.conversation(user=, thread=)`. AG-UI answers an
  interrupt only for the user whose run it is (`404` for another user's thread named alike).

### Changed

- The context's line about earlier tool values that are out of date now asks the model to
  call a tool again only for a value the question needs, and still never to answer with an
  earlier one (`snapshots.NOTE`). It said "call them again; never answer from the earlier
  ones", and on a turn that needed none of them ("Put that in a PDF for finance") ReAct
  re-fetched every input and often dropped why it could not make the file. Haiku, three
  repeats: that turn 1/3 -> 3/3 (4 of 10 across every ReAct run on main); the freshness
  cases 27/27 turns before and after.
- The tool hints narrow what the model is offered only from 30 tools in the run
  (`agent.NARROW_FROM`). Below that, every tool is offered and the hints' tools section
  still says which tools fit. Narrowing a set of 20 hid a tool the question needed (a name
  looked up before its id) in 22-26% of turns. Those turns were wrong 2-3 times as often, on
  every adapter.
- A stall no longer ends the run with no answer. A stall is the same call with the same
  arguments in `max_repeats` steps, or every call failing in 3 steps in a row. The model is
  now told so and called once more with no tool, as `StepLimit`'s last step is. The user
  gets an answer that says what went wrong, for example that the tool timed out. The run
  still stops (`ModelError`) when that call gives no answer.

### Fixed

- With a sandbox, a kept result's rows are `query_result`'s to count, sum, filter, rank and
  group, and code is for what a query cannot make. The stub's code note gave the same job to
  the sandbox ("to count, sum, filter, rank or group its rows, run code over that file"), so a
  model computed in its own Python: 105 SKUs under a 40% margin where every query said 104
  (`ReAct` with a sandbox, `long-data-03`). The note now says the file is there for a chart, a
  formatted workbook or a script, read from the file and never typed into a command; a table
  read whole says so too (its file is in the sandbox as well: 1,528 values were typed into one
  command).
- `ReAct` given a Deep Agents capability has one owner of a small window. Under 32,000
  tokens (`automatic.MIN_WINDOW`: no room for Deep Agents' tools from the first step) its
  summarization is no longer placed either, and `FitWindow` fits the request as on `ReAct`
  alone. With both, the summarization saved a second copy of each table the run kept and spent
  model calls on summaries: the small-window cases were 6 of 8 right with a backend against
  8 of 8 alone. Haiku, a backend, after: `long-data-02` at 16,000 tokens right in 12 s (out
  of time at 420 s), `long-multi-02` at 8,000 right in 24 s (wrong, 125 s), the 14-turn chat
  at 6,000 right in 69 s (113 s). From 32,000 tokens up nothing changes.
- `query_result` no longer runs a run out of its time on a small window with a backend. A
  table read whole is kept and sent with its note; Deep Agents then saved that whole message
  as a file of its own, and the query loaded the copy as one more table: a JSON result
  followed by the note read as "CSV" whose header was the JSON line, 3,902 columns, and
  typing them in one statement ran for minutes with no interrupt stopping it (`ReAct` with a
  backend, a 16,000-token window, `long-data-02`: 420 s, the run's limit). A graph's copy of
  a result the run keeps is now that result (one table; asked by the copy's path, the kept
  rows), and typing counts 64 columns a statement with the load's time looked at between two,
  so `LOAD_SECONDS` holds for any table. The same case now ends in 69 s.

- A file meant for the user that a sandbox command wrote outside the kept folder is said at
  once. Only `outputs/` of the working directory is kept; a model copied its export to
  `/outputs/` at the root, nothing was kept, and its answer said the file was saved (`ReAct`
  with a sandbox, `bf2-mixed-04`). A command that names an `outputs` folder is now told with
  its result what the kept folder holds, with its whole path, or that it holds no file yet
  (`SandboxSource.outputs_said`: Deep Agents' `execute` and `sandbox_exec`); a command that
  names none costs nothing more.
- `FitWindow` gives the tools back once it leaves the older turns out. A long chat on a small
  window was fitted in order: the tools narrowed, to none past `FIT_KEEP`, then the older
  turns left out — and the tools stayed out, though the room the turns took was free. Every
  turn after that offered the model no tool (`create_agent` with `middleware()`, a 6,000-token
  window, Haiku, `advconv-05`: from the fourth turn on, 567 tokens a request, "I only have a
  file-reading tool"). The tools the narrowing dropped are now offered again as far as they
  fit: the same case on Haiku fetched the figure again on its last turn and passed every check
  (0 of 16 failed; 1 of 16 and the answer before).
- `FitWindow` leaves the older turns out before the tools a question needs, and starts the
  latest turn at the user's question. Narrowing kept 8 tools counting the ones it must keep,
  so on Deep Agents (8 state tools) it kept none of the question's; and once the results were
  cut the request fitted with the older turns kept and the catalog tool gone (the `deepagents`
  adapter, 6,000 tokens, `advconv-05`: 0 of 12 catalog questions answered). Now 8 are kept
  besides those it must keep, and the tools are narrowed past them only after the older turns
  are left out. A note the step limit or the budget appends to the call is no longer taken for
  the question: a request over the window kept the note alone, and the model answered that it
  had looked nothing up.
- `FitWindow` leaves the older turns out in layers: first their tool calls and results, the
  oldest turn's first and only as far as it must, each such turn then its question and answer
  (what memory's window gives of an older turn); whole only when those do not fit either. They
  went all at once: with the tools kept for the question (above), a 14-turn chat on a
  6,000-token window lost the first turn's answer with every other, where all 14 questions and
  answers were a few hundred tokens (the scripted `deepagents` run of `advconv-05`, its last
  turn: 54 messages cut to 2; now 52, the oldest turn's call and result left out).
- A LangGraph graph that keeps its own thread (a checkpointer: `create_agent`, with or
  without `middleware()`, a hand-built graph) can be continued after a run stopped mid-call
  (`mode="interrupt"`, a time limit) or left paused on a call while the user wrote again. The
  thread held the model's call with no result, and Anthropic refused every later turn of it
  (400: "`tool_use` ids were found without `tool_result` blocks"); the break-it set on Haiku,
  `advexp-01` and `advfile-02`, ended two turns each ERROR. The next turn now answers such a
  call first with a failed result saying it has none (`langgraph.STOPPED_CALL`); a Deep Agent
  answers it itself (its `PatchToolCallsMiddleware`) and is left to. The complex suite's
  `answered` probe checks it.
- OpenAI Agents: a write the person allowed for the conversation no longer asks again for the
  other calls of the same model reply. Calls governance holds at the SDK's gate
  (`needs_approval`) wait there together, and a resumed `RunState` keeps each one's pending
  approval without asking its gate again: the person was asked about every one of them, in the
  SDK's bare shape ("Approve <tool>?", no choices). Governance now decides each of them when
  the run resumes, one by one, after the answer to the one before: a call allowed for the
  conversation runs, and one that still asks (an `irreversible` tool) asks with governance's
  question and choices. Live (Haiku, 46 runs each): 14 runs asked again before — every run
  with parallel writes allowed for the conversation, streamed and after a handoff too — and 0
  after.
- A question a tool asks (`ask_user`, a schedule tool's "update it?", any `Runtime.ask` in a
  tool, Claude's `AskUserQuestion`) is asked once on every adapter (#203). OpenAI Agents and a
  graph with no checkpointer run a resumed run again from its input, and Claude and Kimi tell
  their model to repeat the paused call; a real model words that call differently, the journal
  knew a call only by its content, and the person was asked again on every resume. The call a
  pause came from within is now kept as the model made it, with its place among the run's
  tool calls, and the call of that tool made there when the run resumes runs as it
  (`Replay.made`): the answer is for the arguments the person saw. The model is told which
  call that is, with its arguments (`RESUMED_ASKED`: a resumed Claude or Kimi session, which
  was told only to repeat "the call with no result"; an OpenAI agent run again from its
  input, which was told nothing). Approvals stay exact. Haiku, the four rows of #203
  (`benchmark.cost.schedules`), runs that paused on a tool's question: Claude 7 of 7 right
  (#203: three of the rows failed), OpenAI Agents 6 of 7 (4 of 7 before its model was told).
- A Kimi call that asks permission carries its arguments when the model streamed them: Kimi
  asks before it starts the call, so its arguments were only in the call's streamed updates,
  which the adapter dropped, and a pause on `Bash` showed `{}` (no command) to the person and
  to governance. On Haiku, the break-and-fix suite could approve none of 8 such pauses.
- A framework's built-in tools no longer count toward the tool hints' narrowing threshold
  (`NARROW_FROM`, 30): only the run's own tools do (`Runtime.own_tools`). Claude Code's and
  Kimi's built-ins, and Deep Agents' `task` and file tools, are tagged `builtin`. Counted with
  them, a Claude or Kimi run of 14 tools of its own narrowed, and the tool a question needed
  (`find_sku`) was never listed for the query: 4 missed turns in the break-and-fix gate.
- A run never ends `SUCCESS` with no answer. A reply that reached the model's output limit
  before it answered (no text, no tool call) ended LangChain's loop, and the run succeeded
  with an empty answer: two LangGraph turns at 4,096 tokens in the break-and-fix gate. In
  `ReAct`, `middleware()` and any `create_agent` or Deep Agents graph the model is now asked
  once more, told so (`budget.CUT_OFF`) and offered no tool (an `output_limit` warning).
  Claude Code and Kimi recover such a reply themselves. A run that still ends with no answer,
  on any adapter but a function, ends `ERROR` `NO_ANSWER` saying why.
- Kimi no longer schedules prompts that never fire. `kimi acp` runs for one attempt and ends
  with it, so a `CronCreate` schedule could not run, yet Kimi scheduled it and answered as if
  it would (found by #121; the break-and-fix gate paused on one). Kimi's own switch,
  `KIMI_DISABLE_CRON=1`, is now set in its environment (`acp.KIMI_ENV`, whatever the agent's
  own `env` says), as Claude's cron tools are disallowed: the call says scheduling is off. The
  cron tools left the run's tool set (`KIMI_OUTLIVES_THE_TURN`). ACP cannot take a built-in off
  Kimi's list, so Kimi still lists them.
- A `create_agent` or Deep Agents graph wrapped without `middleware()` computes over large
  results too: its harness tools' large results are kept, and the model is offered `read_file`
  and `query_result` once the run keeps one, as with `middleware()`. Before, it was offered
  neither, and summed rows by hand: 5 analytics turns in the break-and-fix gate, two of them
  past the 4,096-token output limit. The harness's narrowing copy offers them as schemas, and
  the graph's tool node answers them. A hand-built `StateGraph` still binds its own tools.
- A helper's own time limit (`h.wrap(run_timeout=)`) holds when it runs as a child of a run
  with a longer limit: before, the parent's remaining time replaced it, so a helper wrapped
  with 0.2 s ran its full 2 s (found by the adversarial matrix row X09).
- A Kimi run looping on one call ends with one answer (ADV-9): Kimi's own repeat breaker
  (`KIMI_CODE_REPEAT_BREAKER`: the same call 12 times in a row) ended the turn with no answer,
  the reason only in its log (`stopReason` `repeat_breaker`). When a turn ends with no answer
  after a loop (the bridge's refusal, `bridge.stalled`, or Kimi's breaker on its own tools),
  Kimi is asked once more in the same session, as at its step limit, told the turn stalled
  (`bridge.TURN_STALLED`, for this turn only); a `stalled` warning says so.
- A Claude Agent SDK run at its `max_turns`, and a Kimi run at its step limit, end with one
  answer, as `ReAct` and OpenAI Agents do (#117). Claude's query ended `error_max_turns` (and the
  SDK raised the CLI's exit after it): its session is resumed once (Claude's own `resume`), told
  the limit is reached, for one turn with no tool. Kimi ended the turn with no answer, the
  reason only in its own log (`loop.max_steps_exceeded`): it is asked once more in the same
  session. A `step_limit` warning says so on the stream. Both are told the limit is for this
  turn only (`budget.TURN_LIMIT`): the session keeps the message, and told ReAct's "This run has
  reached its limit", a model answered the conversation's next turn "I've reached the
  tool-call limit" without calling a tool.
- When memory holds nothing for a question (or only part of it) and the run offers
  `memory_search`, the note that says so now also tells the model to search the user's
  earlier conversations first (`kinds=["message"]`) before saying a detail is missing
  (`agent.EARLIER_CHATS`). Told only to offer to look it up, every adapter asked the user
  again for what they had said in an earlier chat ("the cardigan in that same colour, but in
  cotton"). Haiku, ReAct, three repeats: that turn went from 0/3 to 3/3 right, and the turns
  the note reaches went from 15/18 to 18/18, at about half a model call more each.
- `ReAct` with memory off and a known window replays each earlier turn whole — its tool calls and
  their results, not only its question and answer — so a follow-up has the figures behind an
  earlier answer (the vendor read, the 503 behind an explanation); clearing and `FitWindow`
  fit them to the window, and a result past its tool's age is marked out of date as memory's
  window marks it. Kept per conversation in this process (ADV-7).
- A large result a conversation's turn saved stays readable in its later turns: a follow-up's
  `read_file` or `query_result` on the path its history names found "File not found", so every
  what-if needed the whole fetch again. Kept per conversation in the process that saved them
  (`Agent.results`, bounded); a path no longer kept says so and how to fetch it again (ADV-6).
- A worker that dies (a `BaseException` that is not an error) in a harness tool an ACP agent
  called ends the run's attempt, as an in-process tool's does. It ended only the MCP server's
  own task group, and the run went on as if the call had failed (the `X15-kimi-…` matrix cells,
  "the worker did not die").
- An ACP agent's own usage counts toward the run's budget. The adapter said it read ACP's
  `usage` on a prompt's answer, and its test asserted nothing: it was never read, so an ACP
  agent other than Kimi (whose usage log is read as the turn runs) spent with no count and no
  cap. It is counted now when the harness keeps no usage log of the agent.
- The Claude Agent SDK loads the harness's tools from the first step. Claude Code defers every
  MCP server's tools behind its `ToolSearch` unless the server's config says `alwaysLoad`, so
  each turn's model saw only the tool names and spent a step loading them (a model that called
  one unloaded got a validation error and called it again). The harness's `trellis` server now
  says `alwaysLoad`: which tools a run offers is already the harness's choice (the tool hints,
  memory's `tool_search`). A team's own servers keep the CLI's deferral.
- A Deep Agent of yours on a backend outside its state (a disk, a store, a sandbox) had its
  file tools run ungoverned: a `write_file` changed a real file with no journal, record or
  approval. They are governed now, reads and writes, as `ReAct(backend=)` has them
  (`middleware.declare_files`, marked when the graph is wrapped).
- An ACP agent (Kimi Code) gets the memory context marked off from the question, as `ReAct`'s
  is (`history.ASKED_WITH`, moved there from `middleware` so an adapter without LangChain uses
  it). ACP's prompt has no other place for it: the context went as a bare first block, and Kimi
  on a small model read it as the user's message ("your message doesn't include a question").
- A Claude Agent SDK run no longer offers Claude Code's schedules. Its CLI runs for one attempt
  and ends with it, so a scheduled prompt (`CronCreate`) or a wake-up (`ScheduleWakeup`) never
  fires; offered, Claude answered "set this up every Monday" with a schedule nothing kept. They
  join the query's `disallowed_tools` (`adapters.claude.OUTLIVES_THE_QUERY`), after the team's.
- The memory context no longer breaks a provider's prompt cache from one turn to the next.
  A `create_agent` graph (`ReAct`'s, Deep Agents', yours) sent the context, new for every
  question, as a system message ahead of the history, so the next turn shared nothing past
  the instructions with this one. The model now gets it as the first part of the turn's
  question, after the history; the graph's state is unchanged. With a prompt-caching
  middleware that marks the request (LangChain's `AnthropicPromptCachingMiddleware`), the
  answer ending the history is marked too (`middleware.history_marked`, audit K1). The
  context is marked off from the question (`middleware.ASKED_WITH`), so a question after a
  long context does not read as one more saved note.
- Anthropic's prompt cache is on for every Anthropic chat model `ReAct` is given, on
  LangChain's loop and on Deep Agents'. A model name (routed to the provider's own SDK) is not
  a `ChatAnthropic`, so Deep Agents' `AnthropicPromptCachingMiddleware` skipped it, and plain
  `ReAct` had no caching middleware at all: nothing was cached. `middleware.PromptCache` runs
  LangChain's middleware (one given to `ReAct(middleware=)`, its settings kept) on the chat
  model the call is made with, in Deep Agents' own one's place, and marks the answer that
  ends the history. Any other provider's request is unchanged. The small hosted model by name,
  18 multi-turn cases: cached share of input 0 → 0.87, uncached input −87%, cost −62%.
- Anthropic's API given as an OpenAI-compatible endpoint (`LLM_BASE_URL=https://api.anthropic.com/v1/`)
  is called with Anthropic's own client (`ChatAnthropic`, the same endpoint and key), so its
  prompt cache is on: the OpenAI compatibility layer drops prompt caching, as Anthropic's docs
  say. `LLM_GATEWAY=openai-compatible` keeps the OpenAI-compatible client; `trellis doctor`
  says which client is used.
- A live tool's value from an earlier turn reads as out of date in every adapter, not only
  where memory's window replays the conversation. A `create_agent` or Deep Agents graph keeps
  its own thread (its checkpointer), so its earlier results are marked in the request it sends
  (`snapshots.EARLIER`; the graph's state is unchanged). The OpenAI Agents memory session
  marks them as memory's window does. In both, the tool stays offered.
- A model looping on one call is stopped on every adapter: the tool bridge does not run the
  same call a third time in a row and tells the model to answer with what it has (a `stalled`
  warning); a graph without `middleware()`, a Deep Agent and the Claude CLI ran all 30 (ADV-9).
  OpenAI Agents offers no tool on the turn after (its `FunctionTool.is_enabled`), so the
  model's reply is the run's answer: it looped on to the SDK's `max_turns` (10) and answered
  about the limit, or ended `ERROR` when that last call made no answer.

- A run at its step limit ends with one answer: `ReAct` at `max_steps` (LangChain's
  `ModelCallLimitMiddleware` counting) and OpenAI Agents at `max_turns` (the SDK's own error
  handler) call the model once more, told the limit is reached and offered no tool, and its
  answer — what it found, what is left — is the run's: not LangChain's bare "Model call limits
  exceeded" nor `MaxTurnsExceeded` with no answer. One extra model call at most, through #76's
  last-answer path.
- A run's first scratch write (Deep Agents' `write_file` or `edit_file` on the graph's state)
  says the file is the run's own scratch copy, which the user cannot open or download
  (`middleware.SCRATCH_NOTE`). Asked for a chart, a model saved an SVG at `/tmp/…` in the
  graph's state and told the user to open it (break-and-fix round 1, Deep Agents).
- A conversation is no longer lost when memory fails mid-chat. Where memory replays the
  conversation (its window) and its context call fails (an outage, the circuit open), the
  earlier turns come from the run store, as with memory off; before, the model was sent the
  question alone and answered "this conversation has no earlier list" (break-and-fix round 1).
- `query_result` adds money up exactly, on both engines: a column of decimals as written (at most
  15 digits) is `DECIMAL` — DuckDB's own `DECIMAL(p, s)`; on SQLite a `REAL` whose `sum`,
  `total` and `avg` (a window's too) add up in Python's `decimal` — and a number in an answer
  is written as SQLite writes a `REAL` (15 significant digits). 6,000 two-decimal refunds
  summed to `32266.149999999998` (ADV-2).
- OpenAI Agents: a call to a tool the model made up is an error the model reads and the run goes
  on (the SDK's `tool_not_found_behavior="return_error_to_model"`), not a `ModelBehaviorError`
  that ended the run with no answer (ADV-8).
- A blank message (empty, or whitespace only) opens no run: `run`, `stream` and `start` raise
  `ConfigurationError`, AG-UI answers `422`, an A2A task ends `rejected`. It was sent to the
  model and kept in the thread, where a provider that refuses a blank user message (Anthropic)
  would fail every later turn (ADV-1).
- `StallGuard`: three steps in a row in which every call failed stop the run only when a failure
  among them repeats (the same tool, the same error). A model correcting itself — a wrong type,
  then a missing argument, then a made-up tool name, then the right call — goes on to its
  answer, bounded by `max_steps` (ADV-5).

- A kept large result's note says its path is the run's own working copy, which the user
  cannot open or download (`results.PRIVATE`). Asked for an Excel export, a model told the
  user the rows were "saved as CSV at /large_tool_results/…, open it in Excel".
- A table of 30 rows or more that fits the context is read whole and also kept, so `query_result`
  counts, sums, ranks and joins over it exactly (`tools.results.tabled`). Before, only a
  result over 80,000 characters was kept, so a small model added up hundreds of rows by hand
  and got them wrong.
- A Docker sandbox's anonymous volumes (an image's `VOLUME`) are removed with it: before, each
  run of such an image left one volume behind (50 runs, 50 volumes; now none).

- A tool call whose output is over 256 KiB of JSON is recorded with the memory service again:
  the service refuses a larger recorded output (422), so the call's record was lost and the
  service never learned from it. The output is recorded cut to fit, saying so
  (`clients.memory.recorded_output`).

### Changed

- The note the model gets when memory holds nothing (or only part) for the question says
  what an honest answer does: name the detail it does not have, and offer to look it up or
  ask the user for it, instead of a bare "say you do not know" (`agent.ABSTAIN_NOTES`).

### Added

- `query_result(save_as="csv" | "xlsx", file_name=)`: a query's every row kept as a file for the
  user (`Runtime.save_file`), in a run with no tool that runs code (`tools.export.runs_code`);
  each adapter offers the run's own variant (about 50 tokens more, and none where
  `query_result` is not offered). Excel needs the new `xlsx` extra (openpyxl).
- `query_result(file_path="/files/<run>/<name>")` loads a file kept for the user from the store
  as the table `result` (Excel too), beside the run's own results: one query compares an earlier
  file with today's data, and its answer says where each table's rows come from and when they
  were fetched.
- Files kept for the user are read back by later runs, within the run's tenant and user: each
  file has a record beside it (`.<name>.json`: the run, what made it, the query and the results
  it read with their fetch times, the run's tool calls); a run of a chat with files is told of
  them as it starts (one line each); `read_file` reads `/files/<run>/<name>` (CSV and Excel as
  rows, with how it was made) and `/files` lists every chat's. A deleted file is said to be gone.
- A failed sandbox command says why when the sandbox knows (`sandbox.base.hint`): a command
  killed (exit 137) or out of memory names the memory limit it went past; a missing Python module
  says whether the sandbox can install it (no network: use the standard library). In
  `sandbox_exec`'s result as `hint`, at the end of Deep Agents' `execute` output.
- A sandbox's files for the user: the command tool says that files written in `outputs/` are
  given to the user. When the run succeeds, each file directly in `outputs/` is kept in the file
  store (`sandbox.delivered`: named in the journal, announced, served over AG-UI) before the
  sandbox is deleted.

- The sandbox's environment card: `sandbox_exec`, and Deep Agents' `execute` under
  `ReAct(backend=sandbox())`, are described with the image's Python version and installed
  distributions, and with no network, that nothing else can be installed
  (`SandboxSource.card`). The card is read once per process in a sandbox made for it alone.
  [sandbox.md](docs/sandbox.md) has an image recipe that installs from your index
  (`PIP_INDEX_URL`, `UV_INDEX_URL`, a build secret for credentials).

- `trellis.ask_user()`: an opt-in tool the model asks the person with — free text, a choice
  (`options`), a `date`, a `date_range` or a `number` — through the run's own pause, the answer
  its result. The interrupt names the `ask_user` component (`props`: `kind`, `options`) for an
  app's own picker, carries the schema and widget hints for any other UI, and its question says
  the kind of answer for plain chat ([interrupts.md](docs/interrupts.md#asking)).
- AG-UI activity: an `ACTIVITY_SNAPSHOT` after each tool call that ends groups the run's calls
  into one line, `used 2 tools · ran 1 command` ([surfaces.md](docs/surfaces.md)).
- AG-UI parity with CopilotKit-style clients (`frontend.py`): the request's `tools` are offered
  to the model beside the agent's own (one of the agent's by the same name wins, with a
  warning); a call streams its `TOOL_CALL_*` events and pauses the run (an interrupt whose
  reason is `frontend_tool`, its `toolCallId` the call's), and the client's result — a `tool`
  message with that `toolCallId`, or a `resume` entry — is the call's output. `context` and the
  `state` reach the model in the pushed context, together at most half of its token budget;
  the run reads and sets the state (`Runtime.state`), each change streamed as `STATE_DELTA`
  (a JSON Patch; `STATE_SNAPSHOT` where there was no object to patch), and a resume reads the
  state the client sends. `forwardedProps.mode` (AG-UI) and the message's `metadata.mode`
  (A2A) are the run's `mode` (queue, interrupt, steer). A call's paused `TOOL_CALL_RESULT` is
  no longer sent on AG-UI: the call has no result until the run resumes. New dependency:
  `jsonpatch`.
- MCP elicitation: an `mcp(...)` server that asks the person in the middle of a call pauses
  the run as `ask` does — one field of a few values a choice (a dropdown), any other form a
  form with the server's schema (a `format: date` field a date picker), a link a choice — and
  the answer goes back to the server when the run resumes, on both MCP protocol eras
  (`tools/elicit.py`; [tools.md](docs/tools.md#a-server-that-asks-mcp-elicitation)).
- MCP resources: an `mcp(...)` server that has resources gets `<server>-read_resource`, its
  description the map of what there is (resources and templates, at most 20, with the server's
  `lastModified`); it reads one by its URI (a binary one described) or, given none, lists them
  all as rows. A large resource is kept, paged and queried as any large result (`tools/resources.py`;
  [tools.md](docs/tools.md#a-servers-resources-mcp-resources)).
- A file store for the files runs make for their users (`trellis.harness.files`, on fsspec):
  `TRELLIS_FILES_URL` (`gs://`, `s3://`, `az://` with `gcsfs`, `s3fs`, `adlfs`; `file://`),
  else the XDG data folder. `Runtime.save_file(name, data, source=)` keeps a file under
  `<tenant>/<user>/<conversation>/<run>/`, names it in the run's journal and announces it (the
  `file` event); `serve_chat` serves it to the run's own user at
  `GET {path}/runs/{run_id}/files/{name}`: a 5-minute signed link where the store signs, else
  streamed. Stored only: never parsed or remembered. New dependency: `fsspec`.

- Forget by chat: `memory_forget` of what the user said ("forget my address"), which the memory
  service refuses an agent, asks a person through the approval path (the run pauses as any
  call that asks does) and, approved, forgets it as the user; rejected, the model reads why.
  A memory the agent wrote is forgotten at once, as before. Needs the memory service to name
  the memory it refuses (`details.memory_id`); an older one's refusal stays the error it was.
- Where a tool's result came from: `tool(response_format="content_and_artifact")` (LangChain's)
  returns `(content, artifact)`, the artifact a `Citation(name, date, link)` or a list of them
  (also a LangChain tool's own artifact); the model reads the content alone and the call's
  outcome keeps the citations in `metadata["sources"]`, so the run's trajectory has them.
- The turn written to memory carries its run record (`Said.facts`, under
  `custom_metadata["run_record"]`, plain values): each tool result its call's status, seconds,
  sources (the trajectory's outcome of the same call), when it was fetched (`fetched`, UTC)
  and its tool's `max_age`; the turn's last message the run's status, seconds and approvals —
  what the memory service's steps list of the turn shows, with its evidence line for "why" and
  "how sure" (trellis-memory 0.6.2). The fetch time is said there only: a tool result written
  to memory is no longer stamped `[snapshot of <tool> from <UTC time>]`, and a replayed result
  past its tool's age is marked out of date from its run record.
- A `ReAct` agent with a known window sizes the conversation the memory service replays
  (`conversation_token_budget`, trellis-memory 0.6): what one request may hold (95% of the
  window, as `FitWindow` fits it) less its instructions, offered tools and question
  (LangChain's `count_tokens_approximately`), the memory context's budget and 1,024 tokens for
  the reply (`Agent.conversation_budget`). Tool-result clearing (`ThisTurnsResults`, LangChain's
  `ClearToolUsesEdit`) clears only the turn under way's older results, never the replayed
  turns' that memory already sized.
- The cost benchmark's `--suite explain`: two-turn conversations, memory on, whose second turn
  asks about the first answer (its steps, sources, tools, time, approvals...) or is an ordinary
  follow-up, scored strictly against turn 1's own journal record (exact tool names, source
  names and dates, the step count, seconds within a tolerance), with the made-up claims and
  leaked template tags counted, every model request kept, and `--render` checking what a
  llama.cpp chat template shows the model (`benchmark/cost/explain.py`).
- What a finished run spent is on its end: every `RUN_FINISHED` carries `data.spent` (tokens,
  cached tokens, dollars, model calls, tool calls — its sub-agents' runs and earlier attempts
  included), which agent-runs' event log keeps (ADV-11).

- Large results in `ReAct` without a backend and in a `create_agent` graph with
  `middleware()`: kept in the run and previewed, paged with `read_file`, as on OpenAI Agents
  and Claude (`tools.results`); a result `FitWindow` cuts to fit a small window is kept too,
  its cut naming the `read_file` call that reads on from there.
- `query_result` runs on DuckDB when the new `query` extra is installed (in `all`), and on
  Python's own SQLite otherwise, chosen by what is installed ([ADR 0007](docs/adr/0007-query-engine.md)):
  one contract on both (the description names the engine; the same tables, column typing,
  `percentile_cont(x, p)` and `julianday(d)`, and limits), DuckDB locked after loading (no
  file, network, extension or setting; a 512 MB `memory_limit`). Loading a run's 4 results
  of 50,000 rows takes 0.37 s (was 1.01 s), of 500,000 rows 2.5 s (was 9.5 s); 8 calls at
  once over 50,000 rows take 1.8 s (was 7.2 s), over 500,000 rows 12 s (was 89 s)
  (`benchmark/query_engines/results/harness.md`).
  Both engines: a number is only what SQL writes as one (`1_000`, `0x10` stay text), a nested
  JSON value is its compact JSON text, a boolean answers `true`/`false`, a value a SQLite
  query builds is at most 64 MB, and a query past its time is interrupted by a watcher
  thread on either engine.
- `query_result`: read-only SQL (Python's own SQLite, in memory, no dependency) over every row
  of the run's saved results — each a table named by its tool, so one query joins them — with
  formulas, window functions and statistics (`stddev`, `median`, `percentile_cont`, `corr`,
  `regr_*`), offered with `read_file` on every adapter, bounded in time, input, sort memory and
  answer (`tools.query`). A tool declared `paged=True` is fetched to its last page in one call
  (`tools.pages`). A run with a sandbox finds every kept result as a file there too.
- Freshness: a tool's data is live by default; `tool(max_age=)`, `tool(freshness="static")`
  (or `"live"`), a LangChain tool's `metadata`, an MCP tool's `_meta` say otherwise
  (`tools.base.data_of`). Tool results replayed from memory's thread past the tool's age
  (their fetch time is in their run record) are marked out of
  date, the context says so, and the tool stays offered (`trellis.harness.snapshots`).
- `benchmark/cost`: the `aggregate` and `freshness` classes over 10,000 SKUs.
- Tool chains (`trellis.harness.tools.links`): which tool's output is another's argument, read
  from the schemas the tools publish (a function's return annotation, now its
  `ToolSpec.output_schema`; an MCP tool's `outputSchema`), said at the end of each linked
  tool's description (`Chains: find_sku → sku_id → get_stock.`) in every framework, and kept
  whole when the tool hints narrow what is offered ([tools.md](docs/tools.md#tool-chains)).
  `benchmark/cost --suite classes --classes chain` measures two- to four-call chains.
- `docs/research/`: the improvement backlog, the evaluation protocol every change must pass to
  be kept, the scenario coverage audit and the self-explanation research.
- Conversations across runs ([ADR 0005](docs/adr/0005-conversations-across-runs.md)):
  - with memory off, a run of a conversation is given its earlier turns from the run store
    (each run's question and answer) where the framework keeps no thread of its own —
    messages for `ReAct`, graphs and OpenAI Agents, `agent.history` for a function, the
    context's text for Claude and ACP;
  - `run(mode=)` / `stream(mode=)`: `"queue"` (default, as before), `"interrupt"` (cancel the
    conversation's working run and continue), `"steer"` (give the message to the working run
    at its next step: `HarnessTools.before_model`, OpenAI Agents `call_model_input_filter`,
    Claude `PostToolUse`; a function, ACP or a graph without the harness's middleware waits as
    `"queue"`, said once — `Adapter.steers`);
  - one id per message (`conversation.message_id`): the AG-UI text message and
    `TOOL_CALL_RESULT` ids, and each message's `custom_metadata` in the memory service
    (`message_id`, `run_id`, `tool_call_id`), stable across a resume;
  - a Claude Agent SDK or ACP conversation continues its last turn's session in this process
    (`agent.sessions`).
  - `HarnessTools` (in `ReAct` and `middleware()`) has a `before_model` hook: one more
    LangGraph step per model call, so a `recursion_limit` set to the bare minimum needs one
    more.
- `h.conversation(user=, thread=)`: the key a user's conversation keeps its state under.
- `OpenAICompatible`, `Bifrost(OpenAICompatible)` and `ProviderSDK` routes, selected by
  `LLM_GATEWAY` from one registry that packages extend under the entry-point group
  `trellis.gateways`. Bifrost's features ask `gateway.has(Capability.X)`; new capabilities
  `SKILLS` and `ANTHROPIC_ROUTE` (the Claude Agent SDK's routed environment).
- `LLM_AUTH`: bifrost-sdk's auth strategies by name — `bearer`, `google` (a refreshed Google
  token), `aws` (SigV4) — on every model client, the judge and `doctor`'s probe.
- Optional extras `trellis-harness[anthropic]`, `[gemini]`, `[vertex]`, `[bedrock]` (the last
  two with bifrost-sdk's `[google]` and `[aws]` login strategies); `vertex:` and `bedrock:`
  model names for LangChain's `init_chat_model`.
- `trellis doctor` names the route, its auth and where model profiles come from, checks the
  declared route against the endpoint (a `bifrost` route must answer MCP), and refuses
  settings that fit no route with the reason.
- A conformance suite over every registered route, auth strategy and profile against stubs of
  OpenAI's chat completions, the Anthropic, Gemini, Ollama and Bifrost endpoints that speak it,
  a Google token refresh and an AWS SigV4 signature recomputed from the specification.

### Removed

- `automatic.ToolSelection`, `react.defaults_on` and the `benchmark.cost` variant
  `main+selector`: a harness fallback to LangChain's `LLMToolSelectorMiddleware` when memory
  gave no hints, which ADR 0003 §6 rules out (one owner of tool narrowing: memory's hints). A
  team adds `LLMToolSelectorMiddleware` through `ReAct(middleware=)` itself.

### Changed

- A model call's HTTP status is retryable exactly when it is in bifrost-sdk's `RETRYABLE`
  (408, 409, 425, 429, 500, 502, 503, 504), the set the gateway client and the memory service
  use; the harness's own `middleware.RETRYABLE_STATUS` (408, 409, 429 and every 5xx) is gone.
  425 is now retryable; 501, 505 and the other 5xx are not.

### Removed (one clearing pass, one read-back tool)

- `read_result`, `READ_RESULT`, `READ_LIMIT`, `FIT_CLEARED` and `FitWindow`'s clearing of older
  results: LangChain's `ClearToolUsesEdit` is the one clearing pass, and its placeholder names
  `read_file` with the cleared call's id as its path (`/large_tool_results/<call id>`, Deep
  Agents' own), paged from the graph's state with the limits every kept result uses (100 lines,
  20,000 characters). `FitWindow` cuts the longest results instead (each kept whole, its cut
  naming the `read_file` call that reads on). Before, `FitWindow` cleared results to a
  placeholder naming `read_result`, a tool that request did not offer.
- `FitWindow(turns=False)` (ReAct on Deep Agents: a summarization after it) leaves tool results
  to Deep Agents too: its overflow path offloads a result whole to its backend. Before, both cut
  the same result, so a sandbox held it twice (the harness's whole copy, written in for code,
  and Deep Agents' offload of the cut text).

## 0.6.0 — ReAct with Deep Agents on demand; `middleware()` (unreleased)

`ReAct` is LangChain's `create_agent` with the harness's part on top, and Deep Agents'
`create_deep_agent`, every Deep Agents feature native, when a Deep Agents capability is asked
for — and that part is exported: `trellis.middleware()` gives a `create_agent` or
`create_deep_agent` graph of your own what `ReAct` adds.

### Added

- `ReAct` hands wide tasks to helpers at once: the `delegate` tool, offered by itself to a
  request of several independent items of one kind on a large model (never to steps in order,
  a request to change something, a small model or a helper). Its jobs take Deep Agents' `task`
  fields (`description`, and `subagent_type` or `role`); each is a child run that only reads
  (a write is refused, never asked about), its answers merged for the model to check; a
  failed, timed-out, refused or skipped job is reported so; identical jobs run once; the
  budget is shared and stops the fan-out early; cancelling the run cancels them; with
  `RUNS_URL` the jobs queue in agent-runs for any worker. See
  [docs/frameworks/react.md](docs/frameworks/react.md).
- Child runs at once, adaptively, for every engine (`trellis.harness.admission`): a parent's
  sub-agent calls, nested `agent.run`s and `delegate` jobs are admitted under an AIMD limit (1
  to 16, from 4) that rises while the container's CPU and memory (cgroup limits), the
  provider's pace and the gateway's 429s stay healthy and halves when they do not; Kimi Code
  gets it as `KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY`. The live task tree: `helper_queued`,
  `helper_started`, `helper_finished` (status, seconds, tokens, cost) and `delegate` events
  on the run's stream, every framework's helpers included. See
  [docs/runs.md](docs/runs.md#child-runs-at-once-and-the-task-tree).
- A sub-agent's child run works at most its own time limit too, not only what is left of its
  parent's.
- Kimi Code as a target: `kimi(model=..., cwd=..., env=..., mcp_servers=..., mode=...,
  sandbox=..., skills=...)` drives Kimi Code's own agent (`kimi acp`, `@moonshot-ai/kimi-code`
  2.1.x) over the Agent Client Protocol, through the official SDK (`pip install
  'trellis-harness[kimi]'`). Moonshot archived the Python Kimi CLI and its Python agent SDK,
  which is why Kimi is driven through ACP. The adapter is a generic ACP client
  (`trellis.harness.adapters.acp`, `ACPAgent`); Kimi is the one agent configured, pinned and
  tested. Everything Kimi does on its own (its tools, sub-agents and swarms, plan mode,
  compaction) passes through: the harness's tools reach it as an MCP server, every call it asks
  permission for is governed (a person's approval a durable pause, resumed in Kimi's session),
  each sub-agent and swarm member is a child run, its usage log counts toward the budget
  (tokens; dollars are a follow-up), its model goes through Bifrost with the run's headers. The
  `kimi` program's tested range is checked at each run (`compat.BINARIES`); a version outside it
  fails the run. See [docs/frameworks/kimi.md](docs/frameworks/kimi.md), including what ACP
  does not show (calls Kimi does not ask about, a sub-agent's own calls) and the sandbox's
  limit (Kimi's reads stay on the host).
- `src.kimi()`: a skill set in Kimi Code's own skills folder, written per run.
- `tools.convert.mcp`: the harness's tools as an MCP server (the official SDK), served per
  attempt on the loopback interface with a bearer token.
- Requires `trellis-contracts>=0.6.2` (its `acp` error source).

### Removed

- `ReAct(tool_guidance=, answer_check=)` and `middleware(tool_guidance=, answer_check=)`, with
  the middleware they turned on, `ToolGuidance` and `AnswerCheck` (`TOOL_GUIDANCE`,
  `ANSWER_CHECK`, `WRITE_CHECK`, `NO_RECORD` among them), the benchmark's `main+guidance`
  and `main+check` variants, and the live tool-choice test's variants (it runs `ReAct` as
  built; `TRELLIS_LIVE_VARIANTS` is gone). Measured on Granite with memory off, fresh and learned, the
  answer check lowered tool coverage by about 11 points and added about two minutes a run; the
  tool guidance added about 50 seconds a run and, without memory, dropped correct answers from
  0.73 to 0.25 — neither with a significant gain. Each old argument is refused by name with a
  `TypeError` saying what to do instead: how to choose among tools goes in `system_prompt` and
  the tools' descriptions; answers are measured with `h.evaluate` (`tool_coverage`). A run its
  tool hints narrowed is no longer widened to every tool before its answer (only `tool_search`
  adds tools). `FitWindow(optional=)` and its `guidance` step, which only dropped the tool
  guidance, go too (refused the same way). See
  [docs/adr/0002-tool-choice.md](docs/adr/0002-tool-choice.md#amendment-2026-10-28-the-guards-removed).
- `ReAct`'s `delegate` tool (`trellis.harness.delegate`), with what only it used: the wide-task
  and model-tier routing that offered it (`automatic.Delegation`, `wide`, `model_tier`,
  `SMALL_PARAMS`, `SMALL_FAMILIES`), the read-only helper runs (`Runtime.delegated`, the
  `delegated` run metadata, the read-only toolbox and the bridge's `HelperOnlyReads` refusal),
  its agent-runs queueing of jobs, the `delegate` stream event and the `delegate_8` overhead
  case. Deep Agents has helpers natively: its `task` tool, offered as before (a sub-agent of
  its own, else the general-purpose helper on a long task); the model calls it several times in
  one step for helpers at once, and LangGraph runs those calls at once. Kept for every engine:
  the adaptive limit of child runs and the live task tree
  ([docs/runs.md](docs/runs.md#child-runs-at-once-and-the-task-tree)), Kimi Code's
  `KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY`, and a sub-agent's own time limit.

### Breaking

- `ReAct` takes Deep Agents' capabilities: given `backend=` (file tools, large-result eviction,
  summarization; `sandbox()` for `execute`), a `SubAgent`/`CompiledSubAgent` sub-agent
  (`task`), `skills=`, `memory_files=` or `permissions=`, it is `create_deep_agent` (prompt
  caching, its long-task scratch files and helper); otherwise `create_agent` with
  `middleware()` and dangling-call repair — the base measured as good or better on every class
  of task (see Changed). The new signature, names following Deep Agents': `ReAct(system_prompt,
  model, *, tools, subagents, skills, memory_files, permissions, backend, response_format,
  state_schema, context_schema, cache, max_steps=12, max_repeats=3, call_timeout,
  context_window, effort, tool_guidance=None, answer_check=None, middleware)`
  (`memory_files` is Deep Agents' `memory`; its `interrupt_on`, `name` and `debug` are not
  exposed).
- `ReAct(tools=)` refuses `sandbox()` (code runs in `backend=sandbox()`) and agents
  (`a2a(url)` and wrapped agents go in `subagents=`); `max_repeats` is 2 or more, as
  `StallGuard` always required.
- `tool_guidance` and `answer_check` default to `None`, automatic — which, measured on a small
  model, keeps both off (see Changed); `True` turns either on, `False` keeps it off.
- One name per concern (the API audit of 2026-10-27). No alias is kept: each old usage fails
  with a `TypeError` (`AttributeError` for a method) naming what to use instead.
  - `h.evaluate(agent, dataset, judges, ...)` and `evaluate(target, dataset, judges, ...)`: the
    third argument is `judges` (was `evaluators`), the word of `h.wrap(judges=)`,
    `without={"judges"}` and `judge()`. Positional calls are unchanged.
  - `h.wrap(judges=[grounding()])` is refused: every wrapped run already gets the grounding
    check on a sampled share, so it would check the same runs twice.
  - `ReAct(checkpointer=, store=)` removed: a `ReAct` checkpoints into its run and its store is
    the memory service; a graph with its own is `create_agent(..., middleware=middleware())`.
  - `ReAct(capabilities=)`, `middleware(capabilities=)` and `ToolGuidance(capabilities=)`
    removed (and `middleware.CAPABILITIES`): what the assistant can and cannot do goes in
    `system_prompt`. `ToolGuidance(text)` takes a text only.
  - `h.governance_middleware()` removed: a wrapped graph's own tools are governed already, and
    `middleware()` has the same `GovernTools`.
  - `ReAct(middleware=middleware())` is refused saying that `ReAct` already has it; the refusal
    of a middleware named as the harness's says "is the harness's" (ADR 0003 point 4 says so:
    your middleware is added, one named as the harness's — `TodoListMiddleware` among them — is
    refused).
  - `serve_chat` without `identity=` reads the trusted `X-Trellis-Identity` header, as
    `serve_a2a` does (its tenant, when named, the key's): a request without it is `401`, never
    `anonymous`. `identity=` stays the override. `IdentityRefused` and the header's parsing
    (`trusted_user`) moved to `trellis.harness.identity`, shared by both surfaces.
  - `serve_a2a` keeps the same rule: a JSON-RPC call without the trusted header, or with one
    refused, is `401` (a problem document) before the SDK reads it, never run as `anonymous`
    (`trellis.harness.a2a.identity.ANONYMOUS` removed); the card stays public, and
    `identity=` stays the override. The harness's own `a2a(url)` calls send the header.
  - `h.feedback` takes a person's verdict only — `confirm`, `reject`, `correct` — and refuses
    `approve` and `edit`, the system's record of a decision on a paused action
    (`agent.resume`). `VERDICT_SCORES` lost their entries.
- `ReAct(middleware=)` with a Deep Agents middleware's name replaces Deep Agents' own (as
  `create_deep_agent` does); one named as the harness's is still refused.
- One sandbox entry: `sandbox()`, given where each engine takes it. No alias is kept.
  - `docker_sandbox()` removed (from `trellis`, where reading the name says what to use, and
    from `trellis.harness.sandbox.backend`):
    `ReAct(backend=sandbox())` takes the sandbox itself, and a `create_deep_agent` of your own
    takes `**sandbox().deep_agent()` (`{"backend": SandboxBackend(...)}`).
  - `sandbox()` is no longer a tool source itself: its three tools are `sandbox().tools()`
    (`agent.tools.use(sandbox().tools())`, the same object every time), for an engine with no
    sandbox of its own (a function, a LangGraph graph). A bare `sandbox()` given as a tool is
    refused, naming `.tools()`. Same per-run sandbox as before: made at the first call,
    journaled, paused, deleted at the run's end, reaped.
- The model hooks are removed: a model call is each framework's own (LangChain and Deep Agents
  middleware — `ReAct(middleware=[...])` —, the OpenAI Agents SDK's `AgentHooks`, the Claude
  Agent SDK's hooks), which run under `h.wrap` as without it.
  - `Hooks.before_model` and `Hooks.after_model` removed, with `ModelCall` (`trellis.ModelCall`
    says what to use). A hook class that still defines either is refused where it is given
    (`h.wrap(hooks=)`, `governed(hooks=)`): `ConfigurationError` naming the framework's own
    slot, never silently skipped. `Hooks` keeps `on_run_start`, `on_run_end`, `on_error`,
    `before_tool` (`Deny`/`Ask`/`Rewrite`), `after_tool` and `on_output`.
  - `middleware.ModelHooks(*hooks)` and `hooks.openai_agents.ModelHooks(*hooks)` take no hooks
    of yours: what they do for the harness stays — the `chat` span, `call_timeout`, the pinned
    stored prompt and a failed call's `on_error("model", ...)`; the OpenAI run's stall watch,
    hosted tool calls and budget accounting (nothing internal relied on the public hooks).
  - The feature matrix's `F71m` row (model hooks) is gone.
- Removed, Deep Agents now owning them: `automatic.JournalFiles`, `automatic.FilesState`,
  `automatic.ScratchFiles` (`automatic.LongTaskTools` offers Deep Agents' own scratch tools and
  general-purpose helper on a long task); `ModelHooks(routed=)` (it sees a routed model itself).

### Added

- Skill sources (`trellis.skills`): skills kept in a store and edited at any time, per tenant,
  workspace, user and agent, fed to each framework's own skills slot.
  - Sources: `skills.directory(path)` (`SKILL.md` folders; an allow-list `tenants:`/`users:`),
    `skills.bifrost()` (the gateway's skills repository), `skills.langfuse()` (prompts tagged
    `skill`), a store of your own (the `SkillSource` protocol: `list`, `load`, `read_file`,
    optional `changed_since`), `skills.combine(...)`. A name kept at a more specific level wins
    (agent, user, workspace, tenant, the platform-wide sources), then the source given first; a
    disabled winner hides it.
  - Slots: `ReAct(skills=src)` and `create_deep_agent(**src.deep_agent())` (Deep Agents' own
    `SkillsMiddleware`, the skills a `/skills/` route of the backend); `trellis.deep_agent(
    sandbox(), src)` merges a sandbox and skills into its one `backend=` (a `CompositeBackend`);
    `ClaudeAgentOptions(**src.claude())` (a local plugin written per run, removed after;
    `await src.claude_dir(...)` without a harness); `src.openai()` (the OpenAI Agents SDK's
    `Skills` capability of a `SandboxAgent`); `src.tools()` (`list_skills`, `load_skill`) where
    an engine has no slot — a function, LangGraph, a plain OpenAI `Agent`.
  - Per run: the run's scope; the versions pinned in its journal (`repository.journaled`, the
    stored prompts' pin), so a resume uses them and an edit applies from the next run; a deleted
    skill and a source that is down fail soft (`skills_unavailable`; the last listing stands);
    a description, a body and a file capped, a bad name or a path out of its skill skipped.
  - `repository` has again what every versioned source shares (`Kept`, `front_matter`,
    `digest`, `inside`), and `telemetry.Langfuse` reads prompts (`prompts`, `prompt`).
- Each framework page says where the sandbox, skills and hooks go on that engine, and that
  `budget=` is the one cross-engine token, dollar and tool-call cap while a framework's own
  turn limit (`recursion_limit`, `max_turns`) goes in `framework_options`.
- Chat attachments: on `serve_chat`, the user message's file parts — AG-UI 1.0 media parts
  (`document`, `image`, `audio`, `video`) whose `source` is `{"type": "file", "value":
  "<document id>"}`, a document already uploaded to the memory service — are the run's
  `document_ids`, as `agent.run(document_ids=)` takes them. Each is read first as the caller
  (`RunMemory.document`, `GET /v1/documents/{id}`), so the memory service's own read check
  decides: one it refuses or does not know is a `404` and nothing runs. Inline (`data`) and URL
  sources are a `422`: the surface uploads and fetches nothing. A text-only message runs as
  before. A2A file parts are not attachments yet.
- `trellis.middleware(*, max_steps, max_repeats, call_timeout, context_window, tool_guidance,
  answer_check, system_prompt)`: the harness's part of `ReAct` — the run's tools
  per model call narrowed per turn (the memory and gateway MCP tools among them), the graph's
  own tools governed, the stall guard, the step limit then a final answer, the budget's stop,
  planning on a long task, older results cleared and read back, the tool guidance and answer
  check, and `ModelHooks` (hooks, spans, `call_timeout`, a stored prompt pinned per run).
  `h.wrap` says once when a `create_agent` or Deep Agents graph is wrapped without it.
- `trellis.docker_sandbox()`: the run's sandbox (the lifecycle of `sandbox()`: made at the first
  command, journaled, paused with the run, deleted at its end, reaped) as a Deep Agents backend
  — `ReAct(backend=docker_sandbox())` or `create_deep_agent(backend=...)`; `execute` governed.
- `ReAct(subagents=)` takes Deep Agents' `SubAgent` and `CompiledSubAgent` dicts (a `ReAct` graph
  as the `runnable`) beside wrapped agents and `a2a(url)`: each `task` call a child run; a
  declarative sub-agent gets the loop's guards and its model name routed, and a tool it names in
  `interrupt_on` asks a person (its declared risk, as the catalog says).
- On a backend outside the graph's state (a disk, a store, a sandbox) Deep Agents' file tools
  are governed and published as any tool (`middleware.OUTSIDE_STATE`).
- `RunCheckpointer` keeps a LangGraph `DeltaChannel`'s history (its last snapshot and the writes
  since, replayed in the order they were applied) beside the latest checkpoint, so a Deep
  Agents graph's `messages` and `files` come back whole on a resume anywhere.
- `FitWindow` (in `middleware()`, so `ReAct`'s too): a run never fails because a request does
  not fit the window. Over it, the request is made to fit step by step, only as far as needed —
  the optional guidance, the tools narrowed (the used, the hinted, the relevant kept), their
  descriptions compacted, older results cleared, older turns, and last the memory context's
  lowest-ranked items (never a rule, the standing instructions or the learned plan) — each step
  said on the run (`context_fit`); a request that fits is sent unchanged. A request the
  model's server still refuses as too large (its tokenizer counts more than the estimate) is
  fitted again by the server's count and sent once more; the model's tokens per counted token
  are learned from what its replies say their requests were, so a request that fits by the
  model's own count is sent as it is, and a run's narrowed tools stay the same call after call
  while they fit. Only a user message
  that alone cannot fit fails: `ContextTooLarge` (`CONTEXT_TOO_LARGE`) naming the window and the
  size.
- With tool hints, each call is also offered the tools the memory context names (a learned
  skill's steps), beside the hinted, the used, the memory tools and what `tool_search` found.
- `automatic.ToolSelection`: LangChain's `LLMToolSelectorMiddleware` used only with no tool hints
  and over 20 tools — measured (`benchmark.cost` variant `main+selector`), not a default.
- `LIVE_SHARD=i/n` (`make live-shard`): the live suite in shards packed by the measured
  durations (`tests/live/durations.json`, `make live-durations`), each under a job's limit on a
  CPU model.
- Python 3.13 beside 3.12: CI's gates (lint, types, tests, examples, docs, the matrix, the
  overhead benchmark) and the weekly canary run on both (`python-version: ["3.12", "3.13"]`);
  the 3.13 classifier.
- `benchmark.cost` variants `deep` (`ReAct` on Deep Agents) and `deep-<piece>`, `pieces`,
  `main+guidance`, `main+check`, `main+selector`, `main@16k`; `--suite classes` (ten tasks per
  class) and `--memory`; `TRELLIS_LIVE_VARIANTS` picks the live tool-choice test's variants.

### Changed

- A run's tools are offered in the order first offered (the first call's by name, those offered
  later after them, a tool offered again in its old place), not re-sorted by name each call: the
  set only grows at its end, so a provider's prompt cache (whose prefix the tools open) is broken
  once per growth, not reordered.
- Deep Agents' summarization stands after the harness's middleware that narrow the tools, so
  its budget counts what the model is sent.
- The base was chosen by one Granite batch, each base with `middleware()` and `FitWindow`, ten
  different tasks per class (simple, multi-step, wide, a ~100k-character result, 50 and 100
  tools, 50 tools with memory on): `create_agent` equalled the best on simple tasks and was as
  good or better on every other class — on the large result it completed 10/10 runs where
  `create_deep_agent` completed 5/10 and Deep Agents' pieces on `create_agent` 6/10 —
  [docs/frameworks/react.md](docs/frameworks/react.md#the-base). The guidance and the check are
  off by default: the guidance raised coverage (0.89 to 1.00) but made the model call changes
  on 4 of 16 items that asked for none, the check added a model call per answer for no gain,
  and both broke the context's rule
  ([docs/frameworks/react.md](docs/frameworks/react.md#tool-choice)).

### Fixed
- Saving and resuming runs, for long runs and runs whose worker crashed
  ([docs/runs.md](docs/runs.md#continuing-where-a-run-stopped)):
  - A `create_agent` or `create_deep_agent` graph built with `middleware()` and compiled
    without a checkpointer kept no checkpoint: a resume or a crash ran it again from its input,
    the model asked again for every step, and a write the model planned anew (other arguments)
    ran again — seen with a worker process killed after a write. `h.wrap` now runs it as a copy
    with `RunCheckpointer()`: it continues where it stopped. A checkpointer of its own, or
    `checkpointer=False`, stays the graph's.
  - After a pause or a crash, a checkpointed graph's later call identical to an earlier step's
    (the same tool, the same arguments) replayed the earlier step's record: it was not run, and
    an approval it needed was taken from the earlier call's. The journal now keeps each call's
    record, and each answer a call read, with the step it was made in (`Journal.steps`); an
    attempt continuing from a checkpoint passes over the steps before it.
  - A long run's journal grew with every call: a `ReAct` or `middleware()` run drops a step's
    records once its checkpoint moves past it, and every run's trajectory keeps each output as
    the event stream previews it (2000 characters).
  - A graph with a durable checkpointer of its own, after a crash, started the run's turn again
    on top of the unfinished one (the question twice, a tool call left unanswered): the run's
    checkpoints are marked with its id (`trellis_run_id` metadata) and the next attempt
    continues from the latest.
  - A graph checkpoint in the journal that cannot be read back ends the run `ERROR` with
    `CHECKPOINT_UNREADABLE`, not retried and nothing run again
    (`middleware.CheckpointUnreadable`).
- A tool call a model step repeats — the same tool with the same arguments, in any key order
  (seen live: a small model on `ReAct` emitting `style_stock(style='ZZ999999')` 14 times in one
  message) — ran every time: 14 calls, 14 results back to the model, and for a write 14 side
  effects. It now runs once: each repeat runs nothing (no hooks, governance, approval, budget
  charge or journal entry) and the model reads, under the repeat's own call id, that its result
  is the first call's (`Same call as <id> above ...`); its `TOOL_CALL_RESULT` names that call
  (`same_as`). One owner, `tools.bridge` (`firsts`, `same`), fed by whoever reads the step: the
  LangGraph tool node's calls (`HarnessTools`, and `governed_call` for every tool node: `ReAct`,
  `middleware()`, `create_agent`, Deep Agents, a graph's own `ToolNode`) and an OpenAI Agents
  reply (the harness's `RunHooks`; its tools and the agent's own governed `FunctionTool`s).
  Claude and Kimi run a step's calls themselves over MCP, which carries no call id: unchanged.
  The same call in a later step runs again (`StallGuard`'s `max_repeats`); a step whose only
  run call failed still counts toward the error streak. Matrix row `F25d`.
- The tool-choice scenario's answer checks (`examples/06_scenarios/tool_choice.py`) no longer
  fail a right answer on its wording: `plain_answer` reads "no" from the same `CANNOT` words
  as `says_cannot` ("I don't have the capability to …"), and `says_no_record` also reads
  "does not have data", "doesn't exist" and "no information".

- AG-UI: a resume sent right after its pause could end with `RUN_ERROR` ("the run ended
  without a result"): the paused attempt, still finishing (its end hooks, its records) after
  announcing the pause, found the run's buffer reopened by the resume and said the run ended.
  Each attempt now answers for its own end only, and a stream closes when its attempt returned
  (after `RUN_FINISHED`, once the turn is written to memory and the end hooks ran): the next
  message's context holds the turn, and a resume finds the paused attempt done.
- A governed framework tool call inside a sub-agent whose call id equals its parent's `task`
  call id (ids are unique per conversation only) was let through ungoverned; the marker is now
  the tool's name and the call's id.
- A tenant's first runs were offered the toolbox's alphabetically first tools: the hints were
  asked before the tools just published were indexed, every hint a zero. A worker now publishes
  its agents' tools before its first claim, a toolbox's first hints wait (at most 5 s) until
  the catalog says tool search finds them, and hints whose best confidence is under 0.5 narrow
  nothing (every tool offered, with `tool_search`).
- Under Code Mode the narrowing dropped the meta-tools, cutting a run off from every tool behind
  them. The hints are asked about those tools too, and when one is hinted the meta-tools stay.
- The cost benchmark's `picks` check scored the first store an answer named, so a right
  answer that gave the other store's figures first ("S12 has 3 weeks, S40 0.2: S40 first")
  failed. It scores the store the answer concludes on. Benchmark only.
- A Code Mode script's nested calls were recorded after the run's own calls, numbered by the
  memory service in the order the log was read: a tool the run called after the script came
  before them in a learned procedure. They are now recorded in the order the gateway logged
  them, each at the step of the script that made it (the last one started by then). A script
  calling one tool twice with the same arguments is recorded once, as any call repeated at one
  step is.

## 0.5.0 — one breaking release (unreleased)

One API, each setting in its one place, no deprecated aliases: the environment says where the
services are; `Harness()` connects to them; the builder (`ReAct`, or your framework's own) says
what the agent is; `h.wrap` who it is and how its runs go; `agent.run` this call. Needs
`trellis-memory` 0.5 (the memory service's prompt-form `memory_ids`).

### Breaking: removed, with what replaces it

- `h.wrap(tools=, mcp=, skills=, timeout=)`: tools are the builder's — `ReAct(tools=[...],
  subagents=[...])`, or your framework's own tools, governed as it runs them (a graph adds the
  harness's with `HarnessTools()`; a function target calls `agent.tools.call(tool, ...)`);
  `h.wrap(run_timeout=)`.
- `Harness(governance=, prompts=, skills=, hooks=, judges=)`: hooks and judges are
  `h.wrap(hooks=, judges=)`; governance is the memory service's catalog; stored prompts are the
  gateway's.
- `h.tools`, `h.prompt_messages`, `h.model_headers` (the gateway's headers are sent for every
  routed model call; Claude's automatically), `h.resolve`, `h.built_for`, and `h.key`,
  `h.tenant`, `h.governance`, `h.toolbox`, `h.score` (internal now).
- `agent.as_tool()` (a wrapped agent in `ReAct(subagents=)`), `remote()` (`a2a(url)` is an
  async callable; `RemoteAgent` under it), `RunHandle.cancel` (`agent.cancel(handle.id)`),
  `RunHandle.run_id` (`handle.id`), `handle.result(timeout=)` (`wait=`),
  `agent.schedule(cron, input, on_behalf_of=)` (`schedule(cron, user=, input=)`).
- The skills API (`skills`, `skills_dir`, `Skill`, the gateway's Skills Repository source) and
  the prompt sources (`prompts_dir`, `Prompt`, Langfuse prompts): `stored("name", **values)` and
  `h.prompt(...)` read the gateway's Prompt Repository, the one prompt store.
- Virtual MCP selection (`mcp("slug")`): the key's MCP allow-list, in Bifrost.
- `ReAct(system=, prompt=, prompt_vars=, model_timeout=, output=)`: `system_prompt=` (a string
  or `stored(...)`), `call_timeout=`, `response_format=`; `ReAct(checkpointer=)` is for a graph
  run on its own (`h.wrap` refuses it).
- `tool/mcp/openapi/a2a/sandbox(timeout=)`: `call_timeout=`; `sandbox(provider, spec)`:
  `sandbox(provider, image=, spec=)`, Docker when no provider is given.
- `without=` names `memory_push`, `memory_pull`, `records`, `hints`, `skills`, `planning`,
  `files`, `react_auto`: the six names are `memory`, `tool_search`, `mcp`, `code_mode`,
  `grounding`, `judges`.
- Environment: `PROMPTS_DIR`, `SKILLS_DIR`, `TRELLIS_AGENT_VERSION`,
  `TRELLIS_WORKER_CONCURRENCY`, `SANDBOX`, `SANDBOX_IMAGE`, `TRELLIS_JUDGE_MODEL`,
  `TRELLIS_EVAL_SAMPLE`, `TRELLIS_GROUNDING_SAMPLE`, `TRELLIS_JUDGE_SAMPLE`, `LANGFUSE_HOST`;
  `Settings` has seven fields (`otlp_*` and `langfuse_*` are gone: their own SDKs read their
  variables). A fixed tenth of runs is grounded; each judge's share is `llm_judge(sample=)`.

### Added

- `stored(...)`, `llm_judge` exported from `trellis`; `h.wrap(description=)`;
  `run`/`stream`/`start`/`schedule(budget=)` (a run's own budget, kept with its record);
  `agent.tools.use(*sources)`; `a2a(url)(message, user=, tenant=)` outside a run.
- A Claude in-process MCP server's tool is governed by its annotations (`destructiveHint`:
  asks), as the run's tool set lists it.
- A function target as a sub-agent counts as a write unless its tools are declared.

### Changed

- Memory's window as the conversation itself: where memory owns a thread and the framework
  keeps no store (`ReAct`, a LangGraph graph without a checkpointer), the thread's recent turns
  arrive as LangChain's own messages — `HumanMessage`/`AIMessage` with `tool_calls`/
  `ToolMessage` — after the one memory system message, not as text inside it; an OpenAI agent's
  memory session gives the SDK the same turns as input items, its tool calls and results
  included. Each run's question, tool calls, results
  and answer are written back before the run returns (queued past 5 s or when memory is down),
  so the next turn reads them wherever it runs. `trellis.harness.history` holds the policy and
  the messages (`Said`, `from_window`). Needs the memory service's `window_format="messages"`.
- Redaction masks credentials in traces only. Span attributes still lose secret-named fields,
  bearer tokens, JWTs, `sk-` keys and key-like blobs; e-mail addresses are no longer masked
  (personal data is the domain's to define). The run's event stream (`agent.stream`, AG-UI,
  A2A task updates and pushes), its answer, the memory service's transcript, tool records and
  spool, and the journal now carry values as they are, so memory search and recall see the
  real data.

### Fixed

- OpenAI Agents: the tools of every agent reachable through `handoffs` (recursively, on copies)
  are governed — hooks, the catalog's rules, the journal; a `Handoff` object whose agent is
  hidden is said once. The harness's `RunHooks` are always passed, so a token budget is enforced
  without any hooks.
- Claude Agent SDK: built-in tools are governed by the SDK's own `PreToolUse` hook, so
  `allowed_tools`, `acceptEdits` and `bypassPermissions` no longer bypass governance; your
  `can_use_tool` is asked by that hook.
- History owner: one policy (`trellis.harness.history`) with a registry of RAM-only stores. A
  LangGraph `InMemorySaver`/`MemorySaver` or an OpenAI `SQLiteSession(":memory:")` no longer
  turns memory's window off (warned once); OpenAI `conversation_id`/`previous_response_id`/
  `auto_previous_response_id` count as the framework keeping the thread.
- An `approve_when` rule that cannot be evaluated asks a person (never allows), is logged once
  per rule, and is recorded on the run as an `APPROVAL_RULE_INVALID` warning
  (`ApprovalRuleError`).

### Added

- `Hooks.on_output(item) -> item`: mask what a user is shown. It runs, in order (the harness's
  hooks, then the agent's), on the answer `agent.run`/`agent.resume`/`RunHandle.result` return
  (`OutputKind.ANSWER`) and every event of `agent.stream`, `agent.events` and `serve_chat`'s
  AG-UI stream (`OutputKind.EVENT`); what it returns is sent. It never touches memory, the
  journal, A2A, traces or what the model and tools see. A hook that raises is logged and the
  item is withheld (`WITHHELD`), never sent as it was; with no hook overriding it nothing is
  wrapped.
- Native slots backed by our services (`trellis.harness.bridges`), with no new settings and
  each framework's defaults kept without `MEMORY_URL`: the memory service as a LangGraph/Deep
  Agents `BaseStore` (only when the graph has none; its own is kept, said once), as an OpenAI
  Agents `Session` (only when the run has none), and as a Claude `UserPromptSubmit` context
  hook. Every turn is still recorded to memory by the harness. `without={"memory"}` turns them
  off.
- Durable runs: Claude's session kept in the run's journal as the SDK's `SessionStore`
  (progress saved before each step), so a resume on another worker continues it; the OpenAI
  `RunState` of a pause resumes on any worker.
- Native helpers as child runs (lineage, turns in memory, budget, governance): a Deep Agents
  `task`, an OpenAI `Agent.as_tool` call, a Claude `agents=` subagent.
- P4: LangGraph `context=` (a `context_schema`) and `durability` pass through
  `framework_options`. P7: Claude's usage and cost are read from its `ResultMessage`.

- Learned approval suggestions are accepted automatically when the tool catalog refreshes
  (the memory service's accept endpoint), except for a tool at the highest risk, which waits for
  a person; each acceptance is logged and `trellis doctor` lists the accepted ones.

- `python -m benchmark.thread`: who keeps a conversation's history across runs, the framework's
  own thread or the memory service's window, measured on recall and input tokens per turn.
  Measured on Granite through Bifrost with the memory service: recall 1.00 vs 0.92, 202 vs 214
  tokens a turn — the owners stay as they are (ADR 0003 §6).
- Without Bifrost, model names go to any OpenAI-compatible endpoint the OpenAI SDK's own
  `OPENAI_BASE_URL` / `OPENAI_API_KEY` name (LiteLLM's proxy, Portkey, vLLM, Ollama, OpenAI):
  `ReAct(model="name")` and `llm_judge(model="name")` work there, with the client's own retries.
  Exactly one gateway serves names — Bifrost, else that endpoint, else none (a name is then an
  error naming the three) — and each declares its capabilities (`mcp`, `code_mode`, `cost`,
  `fallbacks`, `prompts`, `skills`; `trellis.harness.gateways`): what the deployment's gateway
  cannot do is off, and `trellis doctor` lists it.
- Approval rules have one place, the tool catalog (`risk`, `approve_when`), documented as such
  (governance.md "One place for approval rules"): hooks are for what a rule cannot say (a
  refusal, a rewrite, outside data, who reviews and where), and the examples that asked by an
  amount in a hook now set the catalog's rule. A framework's own approval gate still works,
  asked once, and the warning at `h.wrap` says its rule belongs in the catalog.
- One `tools=[...]` for everything an agent calls. `mcp(...)` is the entry for every MCP source:
  `mcp("https://host/mcp")` (streamable HTTP, `headers=` for its auth), `mcp(stdio("npx",
  "@mcp/files"))` (a local server the harness starts), or `mcp("finance-tools")` (a Bifrost
  Virtual MCP). A server of the agent's own is reached with the official MCP SDK (the new `mcp`
  extra), one kept connection per server, and its tools get the same `<server>-<tool>` names,
  annotations-to-risk, governance, journal, retries, paging and records as the gateway's. With a
  gateway and no `mcp(...)` entry the agent still has every MCP tool its key allows; once it
  names any, it has exactly those. A server reached both directly and through the gateway is an
  error. A wrapped agent in `tools=[...]` is a sub-agent. The same holds in `h.tools(...)` for
  every framework.
- Every gateway call says whose run it is and joins its trace: Bifrost dimension headers
  (`x-bf-dim-tenant`, `-agent`, `-run`, and a hash of the user, never the id) and the W3C
  `traceparent`, on `ReAct`'s calls, the judge's (`x-bf-dim-purpose: judge`) and
  `h.model_headers()` (read per request). The OpenAI Agents SDK's own trace export is left as
  set, said once.

- A framework's own tools are governed: an OpenAI Agents agent's `FunctionTool`s by `h.wrap`,
  on a copy of the agent; a `create_agent` or Deep Agents graph's tools through
  `h.governance_middleware()` (`ReAct` has it) — decided by risk (the catalog's, else the
  tool's declared `side_effects` or annotations, else `write`), asked when governance asks,
  journaled, timed and traced. The graph's state tools are let through.

- One approval per call: a tool held by the framework's own approval gate
  (`HumanInTheLoopMiddleware`, Deep Agents' `interrupt_on`, an OpenAI Agents `needs_approval`
  tool) and by governance is asked once — governance counts the framework's approval, kept in
  the run's journal; a hook's `Deny` still wins. `h.wrap` logs each tool both gate.

- Budgets, per agent and opt-in (`h.wrap(budget=Budget(max_tokens=, max_cost=, max_tool_calls=))`;
  none given, no cap): counted on the run's journal — so a resume goes on from what was spent
  and a sub-agent's spending counts toward its parent's. At a limit no new tool call starts, a
  `ReAct` gives its best answer with what it has, and the run ends `PARTIAL`, its result's error
  `BUDGET_EXCEEDED`. Dollars are what the gateway says each call cost; a tenant's or a team's
  budget is the gateway's.

- `ReAct(output=)` takes a JSON schema as well as a pydantic model; `ReAct(effort=)` sends a fixed
  `reasoning_effort` with every call of a model name; `ReAct(settings=)` takes the gateway's
  address and key in code. A read tool's retry waits what an error's `Retry-After` says when
  there is time left, else gives up. Cached input tokens are on the `chat` span.

- `ReAct` plans and keeps scratch files on a long task — LangChain's `TodoListMiddleware` and
  Deep Agents' file tools — once the model has made three calls or the request asks for steps,
  on a known window of 32,000 tokens or more; a shorter task is sent what it was before.
  `without=` turns each off: `planning`, `files` ([react.md](docs/frameworks/react.md#the-middleware)).

- `make bench-cost` (`python -m benchmark.cost`): the cost and accuracy of a `ReAct` agent over
  tool catalogues of 10, 50, 100 and 500 tools, against whatever model is configured — input,
  output and cached tokens, model calls, wall time, `tool_coverage` and task success per run,
  dollars from a price table with sources (`benchmark/cost/prices.py`), means and 95%
  intervals per variant as JSON and markdown ([evaluation.md](docs/evaluation.md#cost-benchmark)).

### Fixed

- `handle.result()` waits out agent-runs' request budget (429, its `Retry-After`) and polls
  again, instead of failing the wait.
- Every piece of the harness's own client-side state is bounded, so a long-lived process
  serving many tenants, tools and threads does not grow: governance and toolboxes per tenant,
  the catalog rules kept fresh and the published-once set (4,096 each), prompts and skills
  kept, an in-memory checkpointer's threads outside a run (10,000), the `h.tools` builds
  awaiting a wrap, and the names a once-only warning remembers (256) — each the most recently
  used, the rest read, sent or said again. The write spool stops at 64 MB (a write given up
  past it is lost, said once) so a long outage cannot fill the disk.
- A long-lived process without `RUNS_URL` no longer grows with every run: `LocalRuns` kept
  every finished run forever (about 4 kB each; the 30-minute load check grew 130 MB over 38,758
  runs). It now keeps the last 1,000 finished runs for an hour each, with their artifacts
  (`LocalRuns(keep_finished=, keep_finished_seconds=)`); paused, queued and running runs are
  always kept. agent-runs (`RUNS_URL`) stays the durable store for production.
- A `ReAct` run's files — a large result saved by `FilesystemMiddleware`, a summarized history
  — were lost on a resume (Deep Agents keeps them in a `DeltaChannel`, which needs every
  earlier checkpoint, and the run keeps only the latest): they are now in a channel the
  journal keeps whole.

### Changed

- Deprecated, working for one release with a warning naming the replacement:
  `h.wrap(mcp=[...])` and `h.tools(mcp=[...])` (`mcp("name")` in `tools=[...]`; `mcp=[]` is
  `without={"mcp"}`) and `agent.as_tool()` (the agent itself in `tools=[...]`; a sub-agent's
  risk, when its tools do not say it, is set in the tool catalog).
- `make bench-load` measures the production path too (`--queued MINUTES`): runs queued in
  agent-runs, claimed by worker processes, with the memory service on or as the key registry
  only, each process's resident memory slope reported (the harness workers, the memory API
  and worker, agent-runs, its ticker).
- The deprecated `TRELLIS_GROUNDING_SAMPLE` and `TRELLIS_JUDGE_SAMPLE` (and the matching
  `Settings` fields) are read as `TRELLIS_EVAL_SAMPLE` taking the larger of the two, so setting
  one to 0 no longer turns the other off; turn one part off with `without={"grounding"}` or
  `without={"judges"}`.
- The cost benchmark records each model call's seconds (p50 and p95 per cell) and the CPU
  seconds per run of its own process and of each `--watch name=pid` one (the model server, the
  gateway), with their peak memory. `make bench-gate` checks a report against the stored
  baseline (non-inferiority: success within 0.15, input tokens within 10% on the runs both did), run by
  `.github/workflows/bench.yml` where a gateway is configured; `make bench-load` measures many
  runs at once on one and two worker processes and checks for a memory leak.
- The retry loops (read-only tool retries, background memory writes, A2A push
  notifications) run on `tenacity`, now a direct dependency; the attempts, waits and
  `Retry-After` handling are unchanged.
- The judge asks the gateway's judge alias `trellis-judge` (routed by the Bifrost admin) unless a
  judge names its model; it never falls back to the judged agent's model. An alias the gateway
  does not route turns the judges off, said once. Bifrost's semantic cache is not used for
  judges.

- `without=` has fewer names: `memory`, `tool_search` (the tool hints and memory's `tool_search`
  tool), `mcp`, `code_mode`, `skills`, `judges`, `grounding`, `planning`, `files`, and the
  umbrella `react_auto` (`planning` and `files`). `memory` now also turns the narrowing off.
  `memory_push`, `memory_pull`, `records` and `hints` still work for a release, with a warning.

- Configuration: each setting in one layer. The environment holds endpoints and keys
  (`TRELLIS_JUDGE_VIRTUAL_KEY` kept: evaluation's own budget), one `TRELLIS_EVAL_SAMPLE` for
  grounding and judges, `TRELLIS_SPOOL_DIR` (default: the XDG state folder, except in a
  container or serverless runtime, said once) and `SANDBOX`; the tools' own variables are read
  under their own names — the OpenTelemetry SDK's exporter reads `OTEL_EXPORTER_OTLP_*` itself
  (`..._TRACES_ENDPOINT` honoured), `service.name` is only a default under `OTEL_SERVICE_NAME`
  (fixed: it overrode it), the agents' version is `OTEL_RESOURCE_ATTRIBUTES`' `service.version`,
  Langfuse is `LANGFUSE_BASE_URL` and its keys (scores no longer read from OTLP headers), and the
  Docker daemon is found as docker-py finds it (`DOCKER_HOST`, the docker context, the socket).
  Deprecated, read for one release with a warning naming the replacement: `PROMPTS_DIR`,
  `SKILLS_DIR`, `TRELLIS_AGENT_VERSION`, `TRELLIS_WORKER_CONCURRENCY`, `SANDBOX_IMAGE`,
  `TRELLIS_JUDGE_MODEL`, `TRELLIS_GROUNDING_SAMPLE`, `TRELLIS_JUDGE_SAMPLE`, `LANGFUSE_HOST`.
  A startup log line names what the harness is connected to, and `trellis doctor` prints every
  setting with its source, checks each service and key, the spool and the Docker daemon, and
  suggests what to set. Each prompt and skill source in use is logged.

- One owner per concern (the architecture review, ADR 0003): model retries, fallbacks and prompt
  caching are the gateway's — `ReAct`'s client makes one attempt, where it made three (set the
  provider's `network_config.max_retries` in Bifrost); a model's window is `context_window=`, else
  its chat model's LangChain profile — when neither says, clearing old tool results is off (it
  assumed 128k before), said once; a run's dollars are the gateway's cost.

- A function tool's description is its docstring without the sections (`Args:`, `Returns:`,
  `Raises:`...), every other paragraph kept (was: the first paragraph only), and each parameter
  is described by the docstring's `Args:` or `:param x:` entry unless its annotation says.
- The Code Mode meta-tools and `read_skill_file` describe every parameter.

- The memory context is always asked for with the run's own tools, so an agent with any number
  of tools gets the skills it learned; the tool hints still come from five tools (now the
  memory service's threshold), and `without={"hints"}` keeps the learned skills while offering
  every tool.
- The framework extras pin the minor range they were tested with: `langgraph>=1.2,<1.3`,
  `langchain>=1.4,<1.5`, `langchain-core>=1.6,<1.7`, `langchain-openai>=1.6,<1.7`,
  `deepagents>=0.7.19,<0.8`, `openai-agents>=0.22.3,<0.23`, `claude-agent-sdk>=0.2.160,<0.3`,
  `a2a-sdk>=1.1,<1.2` ([ADR 0001](docs/adr/0001-version-policy.md)).
- `trellis-contracts>=0.6.1,<0.7` (schedules carry their run options from 0.6.1).
- The documentation: a "Start here" README, `docs/api.md`, `docs/architecture.md` (the five
  repositories, then the harness inside; was `ARCHITECTURE.md`), `docs/flows.md` (sequence
  diagrams), `docs/configuration.md` (every setting with its default and an example),
  `docs/troubleshooting.md`, `docs/versioning.md`, `docs/adr/`. `docs/blocks/contracts.md` is
  merged into `docs/blocks/mixing.md`.
- The examples are numbered, simple to complex (`examples/01_start` ... `06_scenarios`), every
  one runs offline on a scripted model, memory service and gateway, and `make examples` runs
  them in parallel. Run one with `python -m examples.<group>.<name>`.

### Added

- Tool choice. `ReAct(tool_guidance=True)` (`ToolGuidance`) adds how to choose among tools to
  every call's instructions — every part of a request addressed, the most specific tool over a
  general one, current figures from tools rather than memory, a change made by calling its
  tool and never claimed without it — naming the run's tools that change data;
  `ReAct(answer_check=True)` (`AnswerCheck`) checks the draft answer against the request once
  before it is final (one more model call; the draft is not streamed), so a skipped part or a
  change only claimed in text makes the model call the tools still needed, and a run narrowed
  by tool hints is offered every tool from the check on; when every tool result is empty or not
  found (or memory is `INSUFFICIENT`) the check asks for a plain "no record"; and the first
  change of a run is checked against the rules in the instructions and memory context before
  it runs or is sent for approval. `ReAct(capabilities={...})` tells the model what it can and
  cannot do beyond its tools, for a plain yes or no. All LangChain middleware, opt-in.
- `trellis.harness.tools.lint.lint(specs)`: what makes a tool set hard to choose among (missing
  or short descriptions, undescribed parameters, names the OpenAI API refuses, look-alike
  names and descriptions, 20+ or 128+ tools at once); the toolbox logs each finding as a
  warning when its tools change.
- `tool_coverage()`: an evaluator scoring each item by the share of its `needed_tools` (the
  item's metadata) the run called, naming the missing ones and any `avoid_tools` called; it also
  scores a run that paused for approval, the call it paused on counting as made.
- `examples/06_scenarios/tool_choice.py` and `tests/live/test_live_tool_choice.py`: a tool set
  reproducing the four typical tool-choice failures, measured with and without the switches.

- Langfuse is set up with its own three names alone: with `LANGFUSE_PUBLIC_KEY` and
  `LANGFUSE_SECRET_KEY` set (at `LANGFUSE_HOST`, else Langfuse Cloud) and
  `OTEL_EXPORTER_OTLP_ENDPOINT` unset, `Settings.from_env()` derives the OTLP export as
  Langfuse's SDK does — `<host>/api/public/otel`, `Authorization: Basic base64(pk:sk)`,
  `x-langfuse-ingestion-version: 4` — so traces, scores, datasets and prompts reach one project
  from one set of credentials. A set `OTEL_EXPORTER_OTLP_ENDPOINT` still wins (a collector keeps
  working), a header in `OTEL_EXPORTER_OTLP_HEADERS` wins over the derived one, and the explicit
  `OTEL_*` form for Langfuse works as before.
- `h.model_headers(prompt=)` is pinned per run, as `ReAct(prompt=)` is: every run of the
  harness pins each stored prompt handed out at its start (journaled, so a resume keeps it; a
  `prompt` event; `trellis.prompt.*` on the run's agent span), and the headers — a mapping read
  at each request — select the version the run executing pinned (outside a run, the one
  resolved when they were made). An OpenAI client (`AsyncOpenAI(default_headers=...)`, the
  OpenAI Agents SDK) reads them per request; a LangChain chat model, which copies its headers,
  selects the run's version through `ModelHooks()`; the Claude Code CLI's
  `ANTHROPIC_CUSTOM_HEADERS` get it as the CLI starts. A prompt a run cannot pin is a
  `prompt_unavailable` warning. With `prompt=` it returns that `Mapping[str, str]`, not a `dict`
  ([gateway.md](docs/gateway.md#prompts)).
- `serve_chat`: `POST {path}/runs/{run_id}/cancel` stops a chat run (the run's own user only),
  as A2A's `tasks/cancel` and `agent.cancel` do.
- From the memory service, with no harness code: `memory_search` with `kinds: ["message"]`
  finds what was said in this conversation and in the same user's earlier ones
  ([memory.md](docs/memory.md#pull)), and every run is offered the skills its agent learned
  from its successful runs, across all of the agent's users, in its memory context
  ([skills.md](docs/skills.md#learned-skills)). The feature matrix's memory-pull cells send
  message searches through every adapter and way; live tests cover both end to end against the
  running service.

- A warning, once per framework, when the installed version is outside the tested range
  (`trellis.harness.compat`).
- `LocalRuns.schedules.fire(schedule_id, at=None)`: fire a schedule now, as agent-runs'
  `schedules.fire` does.
- `make docs-check` (links, anchors, snippets against the real API) and `make examples-live`.
- A weekly canary workflow: the latest framework releases, unpinned, through `make test` and
  `make matrix`; it never blocks a pull request.
- Large results on OpenAI Agents and Claude, automatically, as `ReAct` and Deep Agents already
  had them: a harness tool's result over 80,000 characters is kept in the run's journal (across
  a pause, on any worker), the model reads Deep Agents' head-and-tail preview naming
  `/large_tool_results/<id>` and pages the rest with `read_file` — offered once there is
  something to read on OpenAI Agents, listed on the `trellis` server from the start on Claude
  ([tools.md](docs/tools.md#large-results)). Deep Agents is not needed. Closes matrix gap G8
  (F26); Way 2 is n.a. by design (your code gets the whole result).
- Tool hints narrow a `create_agent` or Deep Agents graph's model calls, automatically: its
  tools are bound when it is built, so the harness runs a copy of it whose model calls go
  through one more `awrap_model_call` handler, offering only the tools the run offers — the
  hinted ones, the memory tools, those already used, what a `tool_search` found — and leaving
  out a part turned off after the graph was built (`without=`), as `ModelHooks` does. Your
  graph object is unchanged; its own tools (Deep Agents' `ls`, `task`, ...) are always offered.
  A hand-written `StateGraph` binds its model's tools in its own code and is not narrowed
  ([tools.md](docs/tools.md#tool-hints-what-the-model-is-offered)). Closes matrix gap G12
  (F28).

## 0.4.0 — 2026-09-30 and since

One distribution with extras (`langgraph`, `deepagents`, `react`, `openai-agents`,
`claude-agent-sdk`, `agui`, `a2a`, `otel`); the earlier harness, its integrations and Temporal
are gone. Since the 0.4.0 version was set:

### Added

- `ReAct(...)` is LangChain's `create_agent` with Deep Agents' context middleware and the
  harness's (`trellis.harness.middleware`: `HarnessTools`, `ModelHooks`, `StepLimit`,
  `StallGuard`, `ReadTools`, `read_result`, `RunCheckpointer`), usable in any `create_agent` or
  Deep Agents graph.
- `framework_options=`: the framework's own run options, on `h.wrap`, `run`, `stream`, `start`
  and `schedule`; schedules take every per-run option `start` takes.
- Trajectory evaluators: `called(...)`, `tool_sequence(...)`, `EvalCase.trajectory`.
- Prompt and skill sources: code, `PROMPTS_DIR` / `SKILLS_DIR`, Langfuse and the gateway, one
  name, pinned per run.
- People in the loop: labelled options with several picks, forms (`form=`, `ui_schema=`), your
  own screens (`component=`, `props=`), comments, `remember="run"`, a graph's `interrupt()`
  with fields; `trellis.testing.Reviewer`.
- Run events in agent-runs' event log (any replica streams any run), queue order (`priority`,
  `concurrency_key`), admission (`429`), JSON logs and metrics.
- Sub-agents (`agent.as_tool()`), sandboxes (`sandbox()`), hooks (`Hooks`, `Deny`, `Ask`,
  `Rewrite`), the run time limit and deadline, cancel, the agent's version.

### Removed

- `tool(approval=fn)`: an approval rule in code is a `before_tool` hook.
- `tool(external=True)` and `resume(result=)`: a result from outside is an `ask` in the tool.
- Notifiers (Slack, SMTP): agent-runs' `run.paused` webhook tells people.
- `h.serve_inbox` and its page: an inbox is `h.inbox` plus `agent.resume`.
- The harness's own ReAct loop: `ReAct` is a `create_agent` factory.

### Fixed

- The bugs the feature matrix found (BUG-2 to BUG-10): cut-short tool calls end on the event
  stream, A2A `tasks/cancel` stops the run, JSON text is redacted as what it holds,
  `without={"mcp"}` does not ask the gateway, the LangChain middleware offers only the run's
  tools, a worker's stop out of working time is told in time, and others.
- Tool calls whose arguments the model could not produce as JSON are an error the model reads
  (OpenAI Agents, LangChain).
- Claude Agent SDK: a session resumed after a person answered about a tool call is told which
  call (`mcp__trellis__<tool>`, or the built-in's name) to make again. It was told to repeat
  "your last tool call" if it had "no result yet", but the session holds that call's result as
  "Waiting for a person's approval.": the live model took it for a failure and called other
  tools until the CLI's turn limit. A pause that names no call (an `ask`) keeps the old wording;
  after a deny the session is told the call will not run.
- Grounding (`grounding()`, the sampled check on a run) is no verdict and no score against a
  context that held no evidence, as the memory service records none: it was 0.0 on the trace,
  every claim "unsupported" against nothing, one the run's own tools returned too.

## 0.3.0 — 2026-09-28

The harness core: run events, one interrupt mechanism, MCP and memory tools, AG-UI.

## 0.2.0 — 2026-09-28

Renamed to `trellis-harness`: the package is `trellis.harness`.

## 0.1.0 — 2026-09-16

The first harness: memory, tools and approvals around an agent you built.
