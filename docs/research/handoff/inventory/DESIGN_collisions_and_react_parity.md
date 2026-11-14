# Design: library-vs-harness collisions, and ReAct parity ("best of all libraries")

Design only, 2026-11-14. No product code was changed and no model was called. Scratch space was
`/var/tmp/cons/inventory-tmp/design`. The only commands run were `grep`, `sed` and `git log`/`git diff`
over the trees. No Python was imported.

## 0. How to read this

**The governing rule** (from the brief): prefer the library's own implementation. The harness adds
governance, recording and memory around it. A library feature is replaced only where its version
cannot be governed, recorded or made multi-tenant safe, or where it is measurably worse for accuracy.

**Decisions**
- **(A) As is:** the library's feature works unchanged through the harness. The harness integrates
  it: governed, recorded, counted.
- **(B) Replaced:** the harness's feature replaces the library's entirely. A reason is given.
- **(C) Adapter:** an adapter or bridge maps one onto the other. Usually the library's interface or
  implementation is kept, and the harness supplies the execution or the policy.

**Evidence tags**
- **[V]**: I verified it in this session, by reading the code or running grep/git in the tree named.
- **[Inv]**: taken from one of the five inventories. Their authors probed it or read the code; the
  inventory's own file:line references are kept.
- **[Inf]**: my inference, not verified.

**Path prefixes**
- `R:` = `/var/tmp/cons/reactbg/src/trellis/harness/`. This is the react-background branch, HEAD
  `983923b`. It moved past the `3d892eb` the inventories read; every `R:` line below was re-read at
  `983923b`.
- `F:` = `/var/tmp/cons/final-agent-harness/src/trellis/harness/` (`d2bf754`). The Claude, Kimi and
  OpenAI adapter lines come from there, as in their inventories.
- `K:` = Kimi `main.mjs`, as in kimi.md.

**Jobs and what their trees hold now** [V]

| Job | Worktree | What it holds (relevant to this design) |
|---|---|---|
| **langchain-fixes** | `/var/tmp/cons/lcfix` | **Committed** `071663b`: streams through `astream_events(v3)`, so the graph's transformers run. This fixes PII on the wire and list-content text streaming (inventory G1, G2). **Uncommitted** (`middleware.py`, `adapters/langgraph.py`): a call to the graph's own tools runs as often as the library runs it, so only harness tools are deduped (DECISIONS C2); the call's own ToolMessage is kept (`_made`). TEST_PLAN makes it the owner of L38-L41. |
| **openai-fixes** | `/var/tmp/cons/oaifix` | **Committed**: only `cf8bc03`, the 0.24 upgrade. **Uncommitted**: an edit to `adapters/openai_agents.py` (+70/−19). Owner of L42-L45 and F3. |
| **bg-kimi** | `/var/tmp/cons/bgkimi` | **Committed** `310284e`, `7971933`: tower mode off, resumed sub-agents as child runs, background work across turns. **Not yet in the tree:** TEST_PLAN assigns L46/L47 to this job, but `grep set_session_mode\|currentModeId` finds nothing, and `"ExitPlanMode": "read"` is still at `acp.py:211`. |
| **bg-claude-da** | `/var/tmp/cons/bgclaude` | Claude workflows as child runs, with a shared run when several run at once (`b9278d6`). **No fix yet** for `verbatim_prompts`, `setting_sources` or `on_failure` (grep finds none). |
| **native-skills** | `/var/tmp/cons/nskills` | **Uncommitted**, 14 files. Claude's `Skill` slot is never narrowed away (`SKILLS_SLOT`). Kimi's run home links `AGENTS.md`, `plugins` and the user's own skills. The tools of OpenAI `SandboxAgent` capabilities are governed (`SANDBOX_TOOLS`: `load_skill`/`view_image` read, `exec_command`/`write_stdin` write). |
| **react-background** | `/var/tmp/cons/reactbg` | `BackgroundTasks`, built on Deep Agents' async-task schemas; Deep Agents 0.7.38 and langchain 1.5. `115348a` disallows Claude `Workflow`, `SendMessage` and `TeamCreate`/`TeamDelete`. |

**Merge conflict to settle** [V]/[Inv]: react-background's `115348a` disallows `Workflow`, but
bg-claude-da supports it, and TEST_PLAN L6 says workflows are "kept working (bg-claude-da)". When
the branches merge, bg-claude-da must win and `Workflow` must come off react-background's
disallowed list.

---

# Deliverable 1: collision resolution

## 1.1 Collision table, one decision per row

"Code change" names the place to change: the file and line where the fix goes. "Job" says whether
a running job's tree or brief already covers it.

| # | Concern | Library feature(s) | Harness feature | Decision | Why (governing rule) | Code change (file:line) | Job / covered? |
|---|---|---|---|---|---|---|---|
| 1a | Context | LangChain `SummarizationMiddleware` on a user's graph with `middleware()` (way B) | `FitWindow` (turns=True), `ThisTurnsResults` clearing, `_with_task` | **C** | Both are kept. When the graph has a summarization, `FitWindow` must stop dropping turns, so the window has one owner. The library summary must also keep the memory context and the question. [Inv lc G10] | `R:middleware.py:1762-1783` `_with_task`: also recognise LangChain's summary (`lc_source == "summarization"`, already read at `R:middleware.py:1950`). `R:middleware.py:2559` `FitWindow`: `turns=None` means decide from the graph (off when the compiled graph has a `*Summarization*.before_model` node), set where the graph copy is made (`R:adapters/langgraph.py` near the copy, inventory `:410`). | langchain-fixes (L39 "summarization drops context"). Not in its tree yet [V]. |
| 1b | Context | LangChain `ContextEditingMiddleware`/`ClearToolUsesEdit` given by the user | the harness's own `ContextEditingMiddleware(ThisTurnsResults)` | **C** (the user's wins) | The harness clearing is already the library class (`R:react.py:211-227`). Today a user's copy raises `AssertionError` in way B (probe1) and `ConfigurationError` in ReAct (`R:react.py:742`, `:781-786`). Prefer the library: the user's instance replaces the harness's, and `ReadTools` stays. | `R:react.py:177-251` `_middleware`: when the given list holds a `ContextEditingMiddleware`, skip `clearing` (`R:react.py:211-227`). Remove `"ContextEditingMiddleware"` from `SET_BY` (`R:react.py:742`). `middleware()` (`R:react.py:130-174`) gets a `given=` parameter, or `stacked()` logic, to do the same. | langchain-fixes (G6). Not in its tree [V]. |
| 1c | Context | Deep Agents summarization (85%), `_overflow_clip`, eviction | `FitWindow`, kept results, `ReadTools` | **C** (as built) | ReAct already gives the window one owner: Deep Agents' summarization is placed after narrowing, as `ReActSummarization` (`R:react.py:408-412`, `:794-803`), or `FitWindow` alone. **Measured** [V, react.md:207-208 and `R:react.py:389-390`]: on small windows (6k-16k), Deep Agents' summarization got 6 of 8 cases right against `FitWindow`'s 8 of 8. With `middleware()` on `create_deep_agent`, both act today (Deep Agents at 85%, `FitWindow` at 95%) [Inv da overlaps]. | The same `FitWindow` graph detection as 1a, for `middleware()` on a Deep Agents graph. | langchain-fixes. Not in tree. |
| 1d | Context | Claude CLI auto-compaction, `/compact`, `PreCompact` | none (the CLI owns its context) | **A** | Compaction is CLI-native and cannot be replaced from outside. The harness only records it: today `compact_boundary` is ignored [Inv claude C]. | Turn `SystemMessage compact_boundary` into a run warning event (P3) in `F:adapters/claude.py` message loop (`:360-390` region, inventory). | No job. P3. |
| 1e | Context | Kimi auto-compaction | none | **A** plus a filter | Kimi owns its context. Its notices arrive as `agent_message_chunk` and leak into the answer, stream and memory [Inv kimi row 84]. | Filter the "Compacting conversation context…"/"Compaction completed" chunks out of the answer in `F:adapters/acp.py:538-593` (`H:566-567`). | bg-kimi area; not covered. P2. |
| 1f | Context | OpenAI `OpenAIResponsesCompactionSession`, sandbox `Compaction` capability | none on the OpenAI adapter | **A**, once Responses is reachable | Both are Responses-only. On Bifrost they fail mid-run or silently do nothing [Inv oai #115, #131]. Prefer the library: the route must reach Responses (DECISIONS F3); otherwise refuse at `h.wrap`. | `F:adapters/openai_agents.py:1104` (`use_responses=False`): an opt-in Responses route; refuse `OpenAIResponsesCompactionSession` on chat. | openai-fixes (F3, L44). WIP. |
| 2a | Approvals | LangChain `HumanInTheLoopMiddleware` (batched) | catalog governance, `ask` | **C**, one question | Both stay, but a call gets one question: when HITL gates a call, HITL's decision is the approval, journaled and recorded, and governance does not ask again. Today a person is asked twice (probe5: 2 approvals) [Inv lc]. | `R:adapters/langgraph.py:712-766` (`is_hitl`/`hitl_response`): mark the gated call ids approved in the bridge before resume. `R:middleware.py:411-503` `governed_call` consults that mark. | langchain-fixes (L39 "asks twice"). Not in tree. |
| 2b | Approvals | HITL `interrupt_mode="per_call"`; `interrupt(response_schema=)` | interrupt mapping | **C** | It is the library's own shape, and the harness must map it, not break it [Inv lc G5, G7]. | `R:adapters/langgraph.py:712-720`: accept `{"type":"tool_approval"}` interrupts. Read `Interrupt.response_schema` into the question's `expects`, and re-ask on an invalid answer. | langchain-fixes (G5, G7). Not in tree. |
| 2c | Approvals | Deep Agents `interrupt_on` (main and sub-agent); `permissions` interrupt | catalog risk | **C**, pass through to HITL | ReAct flattens a sub-agent's `interrupt_on` into `irreversible` and drops `when`, `allowed_decisions` and `description` (`R:react.py:645-652`) [V]. It does not expose `interrupt_on` on the main agent at all (`R:react.py:263-283`) [V]. Prefer the library: keep the DA/HITL gate and apply the one-question rule of 2a. Batch edit must edit every call, not only the first [Inv da P2-7]. | `R:react.py:645-652`: stop converting and keep `spec["interrupt_on"]`. Add `interrupt_on=` to `ReAct` and pass it to `create_deep_agent` (`R:react.py:416-440`). `R:adapters/langgraph.py:752-768`: one decision per action. | langchain-fixes (L41 "interrupt_on flattened", "batch edit"). Not in tree. |
| 2d | Approvals | Claude `can_use_tool`, permission modes, `permission_prompt_tool_name` | governance `PreToolUse` hook plus a wrapped `can_use_tool` | **C** (kept), with one **B** | The harness callback wraps the team's own, so governance runs first [Inv claude A]. `permission_prompt_tool_name` stays refused (**B**): the SDK forbids it alongside `can_use_tool`. **Fails open:** an exception other than `Paused` falls through to the CLI's own permission flow (P1). | `F:adapters/claude.py:1326-1330` [V]: catch everything and deny with `continue_:false`; after DECISIONS F1, use `HookMatcher(on_failure="block")`. Plan approval and egress follow DECISIONS B1/B2: `F:adapters/claude.py:157` (ExitPlanMode, [V] "read") and `:151-152`, `:156`. | **No running job** [V]. Give it to bg-claude-da or a new claude-fixes job. |
| 2e | Approvals | Kimi permission modes, `default_permission_mode`/`default_plan_mode` in `config.toml`, plans (`ExitPlanMode`), `allow_always` | ACP `request_permission` → governance | **C** | P0: a config `auto`/`yolo` mode silently bypasses governance, because the harness sends `set_session_mode` only when `mode=` is given (`F:adapters/acp.py:913-916`, [V]). Plans are auto-approved (`F:adapters/acp.py:183-184`, [V]), against Kimi's own rule. Fixes: always set the mode and check `currentModeId`; `ExitPlanMode` asks a person, with every option shown (B3); map "allow for this chat" to `allow_always` (B4). | `F:adapters/acp.py:913` (always `set_session_mode`, on new and resumed sessions); `:183-184` (ExitPlanMode is not "read"; ask with options); `:1610-1618` (`_chosen` maps "remember" to `allow_always`). | bg-kimi per TEST_PLAN L46/L47. **Not in bgkimi's committed tree** [V]. |
| 2f | Approvals | OpenAI `needs_approval`, MCP `require_approval`, `HostedMCPTool.on_approval_request`, `on_approval` | governance held at the SDK gate | **A** (RunState is the checkpoint) plus **C** for edit and remember | Works today [Inv oai #64, #119-124]. "Allow for this chat" must stick for the SDK's own tools (B5). | `F:adapters/openai_agents.py:374-384`: pass `always_approve`/`always_reject`. | openai-fixes (B5). WIP. |
| 3a | Skills | Deep Agents `SkillsMiddleware` | `trellis.skills` sources, pinned per run | **C** (harness source → DA slot) | Deep Agents' own slot reads the run's pinned skills through a `/skills/` route (`R:react.py:378-380`, `R:skills.py:503-560`) [V]. **P0** on the direct path: `_SkillsBackend` subclasses `StateBackend` (`R:skills.py:543`, [V]), so `declare_files` decides "state" for the whole graph (`R:middleware.py:323-330`). Sandbox and disk writes then go ungoverned and unrecorded [Inv da P0]. The general-purpose helper gets no `SkillsMiddleware` [Inv da P2-5]. | `R:skills.py:543`: make the skills route a read-only non-`StateBackend` backend, or mark it so `declare_files` ignores it for routes outside `/skills/`. `R:react.py:358-363`: give the general-purpose spec `skills=` when ReAct has them. | langchain-fixes (L40). Not in tree. |
| 3b | Skills | Claude skills and plugins (`Skill` tool) | `ClaudeSkills`: a temporary plugin per run | **C** | Claude Code's own slot, fed the run's pinned skills [Inv claude A `plugins`/`skills`]. The tool hints must not narrow the slot away. | `F:adapters/claude.py` `_disallowed` (`:1258-1267`): `SKILLS_SLOT` is never narrowed. | **native-skills** (uncommitted diff) [V]. |
| 3c | Skills | Kimi skills (`$KIMI_CODE_HOME/skills`, `Skill` tool) | the run home's `skills/` folder | **C** | Kimi's own discovery. A run home must keep the user's own skills, `AGENTS.md` and plugins beside the pinned ones [Inv kimi P2-15]. | `F:adapters/acp.py` `LINKED`, and the skills folder written at H:1449-1456. | **native-skills** (diff links `AGENTS.md`, `plugins` and the user's skills) [V]. |
| 3d | Skills | OpenAI sandbox `Skills` capability (`load_skill`) | `TrellisSkills`/`TrellisSkillSource` | **C** | The SDK's capability, fed pinned skills. `load_skill` was ungoverned [Inv oai #127, #132]. | `F:adapters/openai_agents.py` capability copy (`_sandboxed`). | **native-skills** (diff) [V]. |
| 3e | Skills | none (an engine with no skills slot) | `list_skills`/`load_skill` tools (`R:skills.py:735` `SkillTools`) | **B**, only where no slot exists | Used only where the library has nothing. | none | done |
| 4a | Memory | Deep Agents `MemoryMiddleware` (`AGENTS.md`, edited with `edit_file`) | memory service: context push, memory tools, records | **A** (both run, independently) | Allowed together [Inv da overlaps]. On ReAct's default `JournalBackend`, edits are lost after the turn because the thread is the run (`R:adapters/langgraph.py:147`: `CARRIED = ("async_tasks",)`, [V]). | Carry Deep Agents' `files` across turns: see 9b. | langchain-fixes (L41). Not in tree. |
| 4b | Memory | Claude `CLAUDE.md`/rules, auto memory, `AgentDefinition.memory` | memory service (not used by Claude: `F:bridges/claude.py:17-19`) | **B** for auto memory; **C** for `CLAUDE.md` | Auto memory lives on one worker's disk and is shared by every tenant and user there [Inv claude P1-2]. It cannot be made multi-tenant safe or survive a move to another worker, so the memory service replaces it (DECISIONS A2). Project `CLAUDE.md` stays the library's (**C**): each tenant and user gets its own `CLAUDE_CONFIG_DIR`, and `setting_sources` is set explicitly. | `F:adapters/claude.py` options builder (`:1182-1255` env region): set `CLAUDE_CONFIG_DIR` per tenant+user and `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`. | **No running job** [V]. |
| 4c | Memory | Kimi `AGENTS.md` chain | memory context block in the prompt | **A** | Kimi's own instructions. The run home must link the global `AGENTS.md` [Inv kimi row 81]. | `LINKED` | **native-skills** [V]. |
| 4d | Memory | OpenAI sessions (`SQLiteSession` and the rest); sandbox `Memory` capability | `MemorySession`, injected when none is given; the memory service | **C** for sessions; **C** or refuse for sandbox Memory | A session object on `h.wrap` is shared by every user (**P0**, probe `p_sess.py`). Use a per-conversation factory. Sandbox `Memory` generation runs `gpt-5.x` straight to OpenAI, ungoverned (**P1**): route the phase models through Bifrost, or refuse `generate=` [Inv oai P0-1, P1-6]. | `F:adapters/openai_agents.py:845` (shared `framework_options`): refuse a session object, or accept a factory. Build `MemoryGenerateConfig` models from `bifrost_provider`. | openai-fixes (L42, L43). WIP. |
| 5a | Planning | LangChain `TodoListMiddleware` (`write_todos`) | `automatic.Planning`: a `TodoListMiddleware` subclass that holds the tool until a long task (`R:automatic.py:122-146`) | **A** (it *is* the library), plus **C** for a user's own | Not a duplicate: the library class plus a gate. The gate is **measured**: offered from the start, the prompt cache was rewritten on 6 of 12 long cases and cost rose by a third (`R:automatic.py:51-54`, [V]). A user's `TodoListMiddleware` collides in way B (AssertionError, probe1) and is refused in ReAct. Prefer the library: adopt the user's instance, as `prompt_cache()` does for caching (`R:middleware.py:1904-1909`). | `R:automatic.py:122-146`: `Planning(native=…)` wraps a given `TodoListMiddleware`. `R:react.py:233-251` and `middleware()`: pull it out of `given`. | langchain-fixes (G6). Not in tree. |
| 5b | Planning | Kimi `TodoList`, ACP `plan` updates, plan mode | none | **A**, plus record | Kimi's own. The harness ignores the `plan` updates [Inv kimi row 29]: surface them as a run event. Plan approval: 2e. | `F:adapters/acp.py` update handler (`H:1208-1229`): `plan` → `CUSTOM` event. | P3, no job. |
| 5c | Planning | Claude `TodoWrite`/`Task*`, `EnterPlanMode`/`ExitPlanMode` | none | **A** | Recorded as tool calls [Inv claude E]. Plan approval: 2d. | none | done |
| 6 | Caching | `AnthropicPromptCachingMiddleware` (langchain-anthropic); Deep Agents' lazy Anthropic, Bedrock and Fireworks caching; Claude and Kimi internal caching; OpenAI `prompt_cache_*` | `PromptCache` (`R:middleware.py:1859-1898`) | **C** (the library middleware wrapped) | `PromptCache` runs LangChain's middleware itself (`self.native.awrap_model_call`, [V]). It adds two things: resolving a routed model name to the provider's chat model, and marking the end of the replayed history. So it is not a duplicate. Bedrock and Fireworks caching pass through as Deep Agents' own (**A**). One gap: the memory `cache_control` breakpoint is lost for routed Anthropic names, because `PromptCache` restores the routed model before `MemoryMiddleware` runs [Inv da P2-9]. | `R:middleware.py:1886-1898`: hand inner middleware the resolved model (or resolve before `MemoryMiddleware`, which sits after it in the captured order). A user's `AnthropicPromptCachingMiddleware` in way B (`middleware()`) collides: adopt it, as ReAct already does (`R:react.py:366`). | langchain-fixes (G6 covers the way-B collision; the memory breakpoint is **not covered**). |
| 7a | Retries | `ToolRetryMiddleware`, LangGraph `RetryPolicy` | harness read retries (3 tries), at `R:middleware.py:1178-1206` | **C** (the library wins when present) | Already coordinated and tested: the harness backs off when the graph retries [Inv lc]. Keep ours where the graph has none (DECISIONS C6), because ours retries only **reads**, and its tries count as one call, one approval and no stall. `ToolRetryMiddleware` has no notion of read versus write. | none | done |
| 7b | Retries | `ModelRetryMiddleware` | worker re-attempt on a retryable `ModelError` (`pipeline.py:925-929`) | **C** | Retries compound today: a graph-level retry, then a whole new attempt [Inv lc overlap]. The library goes first. A `ModelError` that escapes a graph with `ModelRetryMiddleware` should not also be re-attempted by the worker, unless it is a gateway or transport failure the library did not see. | `R:pipeline.py:925-929`: skip the re-attempt when the graph carries a `ModelRetryMiddleware` node (detected at wrap). | No job. P2. |
| 7c | Retries | OpenAI MCP `max_retry_attempts` and `ModelSettings.retry`; Kimi `KIMI_LOOP_MAX_RETRIES_*`; Claude internal retries | read retries in `tools/base.py:298-334`; the Bifrost client sets `max_retries=0` | **A**, with **C** for MCP | MCP retries compound with the harness's read retries [Inv oai #163]. Skip ours for a server that retries itself. Bifrost `max_retries=0` is right: the gateway retries. | `F:adapters/openai_agents.py:480-628` (`GovernedServer`): mark calls as library-retried. | openai-fixes? Not listed; **not covered**. P3. |
| 8a | Limits | `ModelCallLimitMiddleware` | `step_limit` → `AnswerAtLimit(ModelCallLimitMiddleware)` (`R:middleware.py:2044-2075`) | **A**/**C** | `AnswerAtLimit` *is* the library class, plus one answer call. For ReAct, the harness's own agent, keep it: it is ReAct's defined behaviour. For `middleware()` on a user's graph, the user's own `ModelCallLimitMiddleware` wins (today: AssertionError), and its `exit_behavior` stands. | `R:react.py:240` and `middleware()`: drop `step_limit` when the given list holds a `ModelCallLimitMiddleware`, and say so once. | langchain-fixes (G6). |
| 8b | Limits | `ToolCallLimitMiddleware` | Budget's `max_tool_calls` (harness tools only, L12), `StallGuard` | **A** | Independent, no fight [Inv lc overlap]. Calls it blocks are not calls of the run. That is correct: they did not run. | none | done |
| 8c | Limits | Claude `max_turns`, OpenAI `max_turns`, Kimi `maxStepsPerTurn` | one extra answer call at the limit | **C** → per DECISIONS D1 | D1 recommends returning the library's result as `PARTIAL`, with no extra paid call, for Claude and OpenAI. **Kimi** does the same extra call today (`H:940-953`) and should follow D1 for consistency [Inf]. | `F:adapters/claude.py:503-549`, `F:adapters/openai_agents.py:855-893`, `F:adapters/acp.py:940-953`. | Pending decision D1. No job. |
| 8d | Limits | Claude `max_budget_usd`, OpenAI usage, Kimi `wire.jsonl` usage | `Budget` (tokens, dollars, tool calls, across engines) | **A** (counted) | The library cap stays. `Budget` is the cross-engine cap no library has (**harness-only, additive**). Decision-model spend (`AutoModeMiddleware`, `ModelRouterMiddleware` classifiers) is **not counted** [Inv lc G14]. | Count `langchain_core.decisions` usage, which today is LangSmith-only (`decisions/base.py:169-184`): wrap the decision model's call with the accounting callback (`R:accounting.py:58`). | No job. P2. |
| 9a | Checkpoints | LangGraph checkpointers (a user's own, durable or not) | `RunCheckpointer`, latest checkpoint only (`R:middleware.py:3171`) | **A** for a user's own; **C** for RunCheckpointer | A graph with its own checkpointer keeps it, time travel included [Inv lc]. `RunCheckpointer` implements LangGraph's `BaseCheckpointSaver` into the journal: library interface, harness storage. Keep it (DECISIONS C9). | none (see 9b, and 22 in §2) | done |
| 9b | Checkpoints | Deep Agents thread state (`files`, `todos`, memory edits, `/large_tool_results/`, `/conversation_history/`) | `CARRIED` (`R:adapters/langgraph.py:147`, async tasks only) | **C** | In ReAct the thread is the run, so Deep Agents state is lost between turns (probe REACT_TURN2: `File '/notes.md' not found`) [Inv da P1-2]. Prefer the library's semantics (StateBackend "persists across turns"): carry these channels as `async_tasks` already is. | `R:adapters/langgraph.py:147`: `CARRIED = ("async_tasks", "files", "todos")`. The carried store (`R:pipeline.py:552-567`, 256 KiB cap) must accept `JournalBackend`'s 1M-character cap; raise the cap or store files in the journal. | langchain-fixes (L41). Not in tree. |
| 9c | Sessions | Claude `session_store`; Kimi sessions; OpenAI `RunState`, sessions | `JournalSessions`; resume; RunState as checkpoint | **C** | Library sessions, harness storage [Inv]. Gaps: `JournalSessions` lacks `list_subkeys`, so sub-agent transcripts are not restored (P3). Kimi sessions are never closed (P3). | `F:bridges/claude.py:114-140` | No job. P3. |
| 10 | Streaming | LangGraph v3 `stream_events` and transformers (PII, ToolCall, Subagent, InternalCall); Claude `StreamEvent`; OpenAI `RunItemStreamEvent`; Kimi chunks | harness events (TEXT, TOOL_CALL_*, CUSTOM) | **A** (the library's transformers run first) | Prefer the library: the graph's transformers decide what reaches the wire. **Done** in lcfix `071663b` [V]. Claude token deltas (P3) and OpenAI item events (P2) are not surfaced. | Claude: `F:adapters/claude.py:360-372`. OpenAI: `F:adapters/openai_agents.py:1130-1134`. | langchain-fixes: **done** (committed). Others: no job. |
| 11 | PII | `PIIMiddleware` (input, output, tool results; redact, mask, hash, block) | `redaction.py`: credentials in spans only (`R:redaction.py:1-10`, [Inv]) | **A** for the model path, **C** for what the harness writes | They do different jobs; keep both. The library's redaction must also cover (1) the wire, now fixed by `071663b`; (2) the memory context, which the harness moves into the question **after** PII's `before_model` (`R:middleware.py:1785-1828` `with_question`); (3) the harness's own copies of the question in memory and the run store, written unredacted from the input [Inv lc G1]. | (2) Order: `PIIMiddleware` redacts state in `before_model`, while `with_question` merges the context at request time inside `ModelHooks.awrap_model_call` (`R:middleware.py:1658`), after every `before_model`. Make the merge visible to PII: either a `before_model` hook ordered before the user's middleware writes the merged question into state, or the merge happens in `prepare_input` (`R:adapters/langgraph.py:210-223`). Either must leave the request the model receives byte-identical (golden test). (3) `R:adapters/langgraph.py:866-894` (`added()`): take the question from the final state's (redacted) messages when the graph has a `PIIMiddleware`. | langchain-fixes: (1) done; (2) and (3) **not covered**. P1 (multi-tenant data). |
| 12 | Guardrails | OpenAI input, output and tool guardrails; LangChain `AutoModeMiddleware`, `OpenAIModerationMiddleware`, `ModelRefusalMiddleware`; Claude and Kimi hooks | governance (catalog, approvals), grounding, judges | **A** (guardrails run in series; governance stays the authority on side effects) plus **C** to record | Library guardrails pass through and a tripwire fails the run [Inv oai #13-14, #106-109]. Calls that `AutoModeMiddleware` blocks are **not recorded** and its classifier is not budgeted [Inv lc G14]. | Record AutoMode blocks: a `wrap_tool_call` observer, or read its blocked `ToolMessage` in `R:middleware.py:govern_tool_nodes` (`:1078-1156`). Budget: row 8d. | No job. P2. |
| 13 | Tracing | OpenAI SDK tracing (exports to OpenAI); LangSmith (LangChain callbacks, `TracePolicy`/`omit_payload`); Claude OTel `traceparent` | harness OTel run, tool and `chat` spans | **A** (both), with **C** for TracePolicy | Library tracing is the team's choice; the OpenAI default gets a loud warning (DECISIONS A4). The harness's `chat` span should honour `omit_payload` [Inv lc G17]. | `R:middleware.py:1666`, `:2006` (`_said`): skip payloads when the request carries `omit_payload`. | openai-fixes (A4 warning). G17: no job, P3. |
| 14 | Scheduling | Claude `CronCreate`/`CronDelete`/`CronList`/`ScheduleWakeup`, `RemoteTrigger`; Kimi `CronCreate`… | `schedules.py` tools (`schedule_task`, `list_schedules`, `update_schedule`, `cancel_schedule`) | **B** | A library schedule lives in a process that does not outlive the query or attempt. It is not tracked or tenant-scoped and has no run when it fires (DECISIONS E4; `KIMI_DISABLE_CRON`). The harness's schedules are per user, governed and delivered to the chat. **Gap:** Claude `RemoteTrigger` also outlives the run and is not disallowed [Inv claude P2-1]. | `F:adapters/claude.py:196` (`OUTLIVES_THE_QUERY`): add `RemoteTrigger`. | No job. P2. |
| 15a | Background | Deep Agents `AsyncSubAgentMiddleware` (needs an Agent Protocol server) | `BackgroundTasks`, with Deep Agents' own tool schemas and `async_tasks` state (`R:bridges/background_tasks.py`) | **A** when the user hosts a server (`graph_id`/`url`), **C** otherwise | With a remote sub-agent, Deep Agents' own middleware is used, and the harness adds child runs, cancel-with-turn and polling (`R:react.py:484-486`, `R:bridges/async_tasks.py`). Without a server, the harness runs the task but keeps Deep Agents' interface: the same tools and records (L4). Not a duplicate: the library has no in-process execution. | none | react-background (done). |
| 15b | Background | Claude background sub-agents, `Bash(run_in_background)`, `Monitor`, `Workflow` | live session plus child runs | **C** | The library's own, kept alive across turns and attributed to child runs [Inv claude H]. `Workflow`: see the merge conflict in §0. | none beyond the merge | bg-claude-da (done, WIP for several workflows). |
| 15c | Background | Kimi `Bash`/`Agent(run_in_background)`, goal-mode continuations | process kept while background work runs | **C** | The process is killed at the end of the attempt on final; bgkimi keeps it [Inv kimi P1-6]. Goal mode still does not work: its continuation turns are not forwarded over ACP (`K:311741`). | bgkimi. Goal mode: forward continuation turns as notified turns (P2). | bg-kimi (background done; goal mode not covered). |
| 15d | Background | OpenAI: none (Responses `background` is not polled) | — | — | Library limitation; left as is by user decision (L3). | none | — |
| 16 | Large results | Deep Agents eviction (`/large_tool_results/`), OpenAI `ToolOutputTrimmer`, Claude `MAX_MCP_OUTPUT_TOKENS` | kept results, `read_file`, `query_result` (SQL over rows) | **C** in ReAct (measured); **A** off the harness loop (DECISIONS C3) | **Measured** [V, react.md table "a ~100k-character result"]: lean A completed 10/10 against Deep Agents B's 5/10, with a third of B's input tokens. ReAct's capability mode keeps Deep Agents' eviction (one copy, readable by sandbox code). On a plain `create_agent`, `StateGraph` or OpenAI agent, the library's own result must be left as returned (C3). | lcfix WIP keeps the library message (`_made`). `R:middleware.py:702-719` (`_message`) and `:747-765`: apply the rewriting only in way B and ReAct. | langchain-fixes (WIP). OpenAI side: openai-fixes (C3, C4). |
| 17 | Tool choice | `LLMToolSelectorMiddleware`, `ProviderToolSearchMiddleware`, Claude `ToolSearch`, Kimi `select_tools` | tool hints, `tool_search`, `FitWindow` tool step | **A** (theirs compose, since narrowing only shrinks), with **C** fixes | **Measured** [V, react.md:432-434]: `LLMToolSelectorMiddleware` completed 6/10 (coverage 0.60), so the harness narrowing stays the default and the library's is opt-in. Two fixes. `FitWindow` compaction drops `extras["defer_loading"]` [Inv lc]. Claude `ToolSearch`/`WaitForMcpServers` ask a person on every call [Inv claude P1-4]. | `R:middleware.py:2963-2973` (`_compact`): keep `extras`. `F:adapters/claude.py:146-182`: add `ToolSearch`/`WaitForMcpServers` as `read`. | Claude part: **no job** (P1). FitWindow part: langchain-fixes? Not listed (P2). |
| 18 | Store | LangGraph `store` (`InjectedStore`, Deep Agents `StoreBackend`) | memory service as the store when the graph has none | **C** | A graph's own store is kept. With none, the memory service fills the library's slot (DECISIONS C8). | none | done |
| 19 | Elicitation | `langchain.mcp` elicitation interrupt (`mcp_elicitation`); Claude SDK `elicitation` (rejected by 0.2.160); Kimi `elicitation/create` (not advertised) | `tools/elicit.py`: MCP elicitation → `Runtime.ask` | **C** | Map each library's form onto the same `ask` [V `R:tools/elicit.py:1-15`]. `langchain.mcp`'s interrupt is answered raw today; Kimi drops questions 2-4 [Inv kimi P1-9]. | `R:adapters/langgraph.py:712-766`: recognise `type: mcp_elicitation` and reuse `elicit.py`'s mapping. `F:adapters/acp.py:677`: advertise `ClientCapabilities(elicitation=form)`. | Kimi: bg-kimi (L47), not in tree. LangChain: no job, P2. |
| 20 | Sub-agents | Deep Agents `task`, Claude `Agent`, Kimi `Agent`/`AgentSwarm`, OpenAI `as_tool`/handoffs | child runs | **C** | Library delegation; each call becomes a child run (budget, records, memory). Kimi sub-agent approvals arrive without arguments [Inv kimi P1-2], so read the `interaction.request` payload in the wire log, which bgkimi already reads. | `F:adapters/acp.py:1274-1275`, `:1633-1645` | bg-kimi (L47), not in tree. |
| 21 | Harness-added behaviour | (the library does none of these) | date line (C1), duplicate-call dedupe (C2), tool re-sort (C4), `tool_not_found` override (C5), `DELIVERS_HERE` (E1) | per DECISIONS | C1 stays. C2: only harness tools (lcfix WIP). C4, C5: openai-fixes. E1: no job. | as in DECISIONS | mixed |

## 1.2 How each library "works as is, and uses our middleware if it wants"

- **LangChain / LangGraph (ways A and C):** every library feature passes through, and the
  harness adds governance at the `ToolNode` (`R:middleware.py:1078-1156`), accounting, memory and
  records. Way B (`middleware()`) is opt-in. After rows 1b, 5a, 6 and 8a, a team's own
  `TodoListMiddleware`, `ContextEditingMiddleware`, `ModelCallLimitMiddleware` or
  `AnthropicPromptCachingMiddleware` takes precedence over the harness's piece of the same name,
  instead of crashing the build.
- **Deep Agents:** the same, plus Deep Agents' state-backed tools treated as the graph's scratch.
  The P0 in row 3a and the thread-state loss in row 9b are the blockers.
- **Claude, Kimi:** no LangChain middleware applies. The library's loop runs as is, and the
  harness is reached through hooks, the permission callback and ACP permission requests.
  "Our middleware" there means governance, memory context, child runs and budget. The P0s
  (2e Kimi mode; A1/A2 for Claude) and fail-open (2d) are the blockers.
- **OpenAI Agents:** the same, through the governed agent copy, `RunHooks` and
  `call_model_input_filter`. Rows 4d (shared session, P0) and 1f/F3 (Responses through the gateway)
  are the blockers.

---

# Deliverable 2: ReAct parity, "best of all libraries"

## 2.1 What ReAct is today (verified)

[V] `ReAct(...)` (`R:react.py:263-450`) is always `create_deep_agent`. It runs in one of two modes:
- **lean**: no Deep Agents capability was asked for. The scratch tools, `task` and planning are held
  until a long task: past 50% of a known window of at least 32k tokens (`R:automatic.py:49-88`).
- **capability**: `backend`, a sub-agent, `skills`, `memory_files` or `permissions` was given.
  Deep Agents' tools and planning come from the first call, and Deep Agents' summarization is
  placed after narrowing when the window is at least 32k (`R:react.py:391-412`).

The stack is `_middleware()` (`R:react.py:233-251`), then the user's middleware, then `ModelHooks`,
then `PromptCache`. The checkpointer is always `RunCheckpointer` (`R:react.py:437`), and the
thread is the run.

**Accuracy baseline.** The caller reports the 357-turn Haiku benchmark on main: **plain (lean)
352/357, backend (capability) 356/357**. I could not find that result file in `reactbg/benchmark/`
(`grep 357` finds only TEST_PLAN). It is taken as given, **not verified**. The harness can already
switch pieces off for a measurement: `react.defaults_off(...)` (`R:react.py:755-766`) and the
benchmark variants `main`, `deep`, `deep-<piece>` and `main-planning`
(`reactbg/benchmark/cost/variants.py:246-305`) [V].

**Offline guard against accuracy drift** [V]. `tests/integration/test_react_golden.py`
(invariant I2) asserts that a short task sends byte-identical requests, checked against
`golden/react_short.json`. **Rule used below:** a change that leaves every request ReAct sends
unchanged cannot change measured accuracy, and is marked "no change (offline proof: golden plus a
long-task snapshot)". A change that alters a request is marked "needs a Haiku measurement".

**Cheapest Haiku measurement** (proposal, [Inf]):
1. Request-diff triage, offline and free. Replay main's recorded Haiku replies through the new
   build, with a scripted model that answers each step with main's recorded reply, and diff the
   first request of each of the 357 turns.
2. Re-run with Haiku only the turns whose request changed. A paired comparison on those turns is
   enough (exact McNemar on the discordant pairs).
3. If no per-step replies were recorded, run the subset of the suite that exercises the feature,
   for example `--sets` holding the long-task or follow-up cases, with `--repeats 1` and the two
   variants `main` and `main-<x>` (`benchmark/cost/__main__.py:49-105`).

**One finding to measure first** [Inf]. Backend beats plain by 4 of 357 turns on Haiku, yet the
lean default was chosen on Granite 2B (react.md "The base"). Four discordant turns, all one way,
give a two-sided exact p ≈ 0.125: not significant. The paired per-turn results, if they are kept,
settle this for free. If capability mode is better on Haiku, the default should be revisited per
model class.

## 2.2 The union of capabilities and ReAct's status

Columns:
- **Has it**: libraries offering it. C = Claude, K = Kimi, O = OpenAI, DA = Deep Agents, LC = LangChain/LangGraph.
- **ReAct today**: the evidence.
- **Best way to give it to ReAct**: library class first.
- **Accuracy**: the expected effect on the 357-turn benchmark.

| # | Capability | Has it | ReAct today (evidence) | Best way to give it to ReAct (library first) | Accuracy |
|---|---|---|---|---|---|
| 1 | Function tools, governed | all | **yes**: `tools=` functions go to the harness toolbox, LangChain tools stay the graph's (`R:react.py:551-571`) [V] | — | — |
| 2 | MCP tools | all | **yes**: `mcp(...)`, the gateway's MCP tools (`R:react.py:293-303`) [V]. `langchain.mcp` annotations are ignored, so every call asks a person [Inv lc G8] | `side_effects_of` (`R:middleware.py:1212-1222`) should also read `metadata["mcp"]["tool"]["annotations"]` (`langchain.mcp.MCPAdapter`) | no change |
| 3 | MCP resources and prompts | C, O, K | **yes**: harness `tools/resources.py` [V, file present] | — | — |
| 4 | MCP elicitation | C, K, LC | **yes** for harness MCP (`R:tools/elicit.py`) [V]. A `langchain.mcp` elicitation interrupt is not mapped [Inv lc] | map `langchain.mcp.elicitation.MCPElicitationInterrupt` (row 19) | no change |
| 5 | File system tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) | C, K, O (sandbox), DA, LC (`FilesystemFileSearchMiddleware`) | **yes**: Deep Agents `FilesystemMiddleware`; lean holds them until a long task (`R:automatic.py:60`, `:203-221`) [V] | — | — |
| 6 | Files persist across turns | DA (StateBackend via checkpoints), C, K (disk) | **no**: thread = run, `CARRIED=("async_tasks",)` (`R:adapters/langgraph.py:147`) [V]; probe REACT_TURN2 [Inv] | carry Deep Agents' `files` and `todos` channels (row 9b) | **needs Haiku** on multi-turn cases in backend mode (lean never writes files before a long task). Expected: helps follow-ups [Inf] |
| 7 | Shell / code execution in a sandbox | C (Bash, sandbox), K (Bash), O (Shell, LocalShell, sandbox), DA (`execute` on sandbox backends), LC (`ShellToolMiddleware`) | **yes**: `backend=sandbox()` → `SandboxBackend` with `execute`, governed (`R:react.py:530-535`) [V]. Deep Agents partner sandboxes pass through `backend=`, untested [Inv da row 37] | partner sandboxes (Daytona, Modal, LangSmith) through `backend=` as is: add offline tests | no change |
| 8 | Provider/hosted tools: web search, web fetch, code interpreter (server-side) | C (WebSearch, WebFetch), K (WebSearch, FetchURL), O (WebSearchTool, CodeInterpreterTool, FileSearch…), LC (provider dict tools: Anthropic `web_search_*`/`code_execution_*`, OpenAI Responses `web_search`, `code_interpreter`) | **no**: a dict tool given to `ReAct(tools=)` is not a `BaseTool`, so `_split` sends it to the harness toolbox (`R:react.py:570-571`), and `as_source` raises "not a tool" (`R:tools/sources.py:559-567`) [V] | keep `dict` tools in `own` (`R:react.py:570`): `create_agent` binds them and `_offers` already keeps unknown dict tools (`R:middleware.py:1485`). Record server-side calls from `server_tool_use`/`*_tool_result` content blocks in `ModelHooks._replied` (`R:middleware.py:2019`), as the OpenAI adapter does for hosted calls (`F:adapters/openai_agents.py:1250-1309`). Through Bifrost chat completions, provider server tools may not pass: refuse clearly [Inf] | no change unless used (opt-in). One Haiku call with `web_search` proves the wiring (small cost per search) |
| 9 | Computer use | C (Chrome), O (`ComputerTool`), Anthropic `computer_*` tool | **no** | LangChain has no executor (langchain-anthropic only formats the tool) [Inf from source listing]. A minimal harness tool named `computer`, governed per action, like `F:tools/convert/openai_local.py:264-337`, executing in `sandbox()`. P3 | no change (opt-in) |
| 10 | Sub-agents (isolated context) | all | **yes**: Deep Agents `task` (declarative and compiled), wrapped agents, `a2a(url)`, each a child run (`R:react.py:574-610`) [V]. The general-purpose helper lacks skills and profile settings [Inv da P2-5] | `R:react.py:358-363`: pass `skills` and the profile's `general_purpose_subagent` | **needs Haiku** only if the helper's prompt changes (profile); skills alone: no change when `skills=` is unset |
| 11 | Parallel sub-agents / swarm (Kimi `AgentSwarm`, Claude `Workflow`) | K, C, LC (`Send`) | **partly**: several `task` calls in one step run at once (a read: react.md "Several calls in one step") [V doc]. No item-map tool | a `CompiledSubAgent` built with LangGraph `Send` (map-reduce) for a fan-out over items: a recipe, no harness code. P3 | no change |
| 12 | Background work | C, K, DA (async) | **yes**: `BackgroundTasks` with Deep Agents' tool schemas (`R:react.py:470-495`, `R:bridges/background_tasks.py`) [V] | — | (behaviour tested offline; "model uses background tools" needs Haiku: TEST_PLAN 5.16) |
| 13 | Handoffs (control passes to another agent, which answers the user) | O, LC (`langgraph-swarm`/`langgraph-supervisor`, `Command(goto=…, graph=Command.PARENT)`) | **no**. Also, `return_direct` gives a false NO_ANSWER [Inv lc G4] | the `langgraph-swarm` package is **not installed** [V: absent from site-packages], so a dependency decision is needed. Cheapest: a `CompiledSubAgent` whose answer is returned directly, once G4 is fixed (`R:adapters/langgraph.py:296-306`). P3 | no change (opt-in) |
| 14 | Workflows (deterministic graphs) | LC (StateGraph, functional API), C (`Workflow`) | **yes** as a sub-agent (`CompiledSubAgent`) or wrapped on its own (way C) [V] | — | — |
| 15 | Planning (todo list) | C (TodoWrite), K (TodoList), DA/LC (`TodoListMiddleware`) | **yes**: `Planning` is a `TodoListMiddleware` (`R:automatic.py:122-146`) [V]. Todos are lost between turns (thread = run) | carry `todos` (row 9b). Adopt a user's `TodoListMiddleware` (row 5a) | carrying todos: **needs Haiku** (backend multi-turn); adopting: no change |
| 16 | Plan mode (read-only exploration, then a plan the user approves) | C, K | **no** | no library implementation in LC/DA [Inf]. Compose library parts: `HumanInTheLoopMiddleware(interrupt_on={"write_todos": …})`, plus a governance mode that denies writes until the plan is approved (a harness catalog rule, minimal). Needs a user decision. P2 | no change (opt-in) |
| 17 | Skills | C, K, O (sandbox), DA | **yes**: Deep Agents `SkillsMiddleware` through the `/skills/` route (`R:react.py:378-380`) [V]. Learned skills come with the memory context | `pinned_skills` input (DA 0.7.23) for "use skill X now" (pass through, untested [Inv da row 64]): a test only | no change |
| 18 | Memory files (`AGENTS.md`/`CLAUDE.md`) | C, K, DA | **yes**: `memory_files=` → `MemoryMiddleware` [V `R:react.py:431`]. Edits are lost per turn on `JournalBackend` | row 9b; or document a disk/store backend | edits persisting: **needs Haiku** (backend only) |
| 19 | Long-term memory (learned, cross-session) | C (auto memory), O (sandbox Memory), harness | **yes**: memory service (push, pull tools, learned skills) [V doc] | — (the harness's own, multi-tenant: B over Claude and OpenAI) | — |
| 20 | Compaction: summarization | C, K, O (Responses), DA, LC | **yes**: Deep Agents summarization in capability mode at windows of 32k or more (`R:react.py:408-412`). In lean mode, `FitWindow` drops older turns instead (`turns=True`, `R:react.py:248`) [V] | in lean mode with window ≥ 32k, a library `SummarizationMiddleware` could replace `FitWindow`'s turn-dropping step | **needs Haiku**: `main` vs a new `main+summary` variant on long multi-turn cases at a forced `--window 32000` |
| 21 | Compaction: clear old tool results | LC (`ClearToolUsesEdit`), DA, K | **yes**: LangChain `ContextEditingMiddleware` with `ThisTurnsResults(ClearToolUsesEdit)` (`R:react.py:211-227`, `R:middleware.py:2416-2450`) [V] | — | — |
| 22 | Manual compaction (`/compact`, a `compact_conversation` tool) | C, K, DA (`SummarizationToolMiddleware`) | **no** by default. Through `middleware=` it is possible, but the tool declares nothing, so every call asks a person [Inv da row 72] | declare `compact_conversation` as `read` in `_declared` (`R:react.py:719-733`). Keep it opt-in (`create_summarization_tool_middleware`). P2 | no change (opt-in) |
| 23 | Never fails for size; overflow recovery | DA (`ContextOverflowError` fallback), K (2.6.1) | **yes**: `FitWindow` (refit on server refusal) [V `R:middleware.py:2559-2600`] | keep (measured, see 2.3) | — |
| 24 | Large tool results offloaded and paged | DA, C (CLI), O (trimmer) | **yes**: kept results, `read_file`, `query_result` (lean); Deep Agents eviction (capability) [V] | keep (measured) | — |
| 25 | Steering a running turn | C (after a tool call), K (TUI only) | **yes**: `HarnessTools` takes a steered message before each model call (`R:adapters/langgraph.py:191-195`) [V] | — | — |
| 26 | Ask the user (structured questions) | C (`AskUserQuestion`), K, LC (`interrupt`) | **yes**: `ask_user` (`R:tools/ask.py:248-250`) [V]. LangGraph `interrupt(response_schema=)` is ignored [Inv lc G7] | read `Interrupt.response_schema` (row 2b) | no change |
| 27 | Approvals / HITL | all | **yes** via the catalog. Deep Agents' `interrupt_on` is not exposed and sub-agent gates are flattened (`R:react.py:645-652`) [V] | expose `interrupt_on=` → Deep Agents HITL, with the one-question rule (rows 2a, 2c) | no change (approvals are not in the benchmark) [Inf] |
| 28 | "Allow for this chat" | C, K, O | **yes** (harness) [DECISIONS B4/B5] | — | — |
| 29 | Structured output | all | **yes**: `response_format` (`R:react.py:273`, `:434`) [V] | — | — |
| 30 | Guardrails (input, output, tool) | O, LC (`before_agent`/`after_agent` + `jump_to`, `AutoModeMiddleware`, `OpenAIModerationMiddleware`, `PIIMiddleware(block)`) | **opt-in** through `middleware=` [V react.md "Yours, opt-in"]; AutoMode blocks are not recorded | document an "input/output guardrail" recipe with `@before_agent`/`@after_agent` and `jump_to="end"`; record AutoMode blocks (row 12). P2 | no change (opt-in) |
| 31 | PII redaction | LC (`PIIMiddleware`), C (sandbox `RedactionRule`) | **opt-in**. Stream fixed (lcfix). The memory context and harness records are unredacted (row 11) | row 11 | no change |
| 32 | Moderation | O (via guardrails), LC (`OpenAIModerationMiddleware`) | **opt-in** (`middleware=`) [Inv lc untested] | an offline test | no change |
| 33 | Model fallback | C (`fallback_model`), LC (`ModelFallbackMiddleware`), Bifrost | **opt-in**: `middleware=[ModelFallbackMiddleware(...)]`, placed before `ModelHooks` and outside `PromptCache` (`R:react.py:791`, `:422-427`) [V]. The fallback model gets no gateway headers (accounting warns) [Inv lc] | accept a model **name** in `ModelFallbackMiddleware`'s list by routing it (`routed_chat_model`) | no change (opt-in) |
| 34 | Model routing | LC (`ModelRouterMiddleware`, beta), Bifrost | **opt-in**. The picked model loses gateway headers, and the classifier is not budgeted [Inv lc] | route picked names (as 33); budget the classifier (row 8d) | no change |
| 35 | Tool selection (many tools) | LC (`LLMToolSelectorMiddleware`), C/K (tool search), O (`ToolSearchTool`) | **yes**: tool hints, `tool_search`, `FitWindow` tool step [V] | keep ours (**measured**: selector 6/10 against FitWindow 8/10, react.md:432-434); the selector stays opt-in | — |
| 36 | Provider-side tool search (`defer_loading`) | LC (`ProviderToolSearchMiddleware`), O | **opt-in**. `FitWindow` drops `defer_loading` [Inv lc] | keep `extras` in `_compact` (`R:middleware.py:2963`) | no change |
| 37 | Tool emulation (testing) | LC (`LLMToolEmulator`) | **opt-in**; ungoverned by design [Inv lc] | — | — |
| 38 | Checkpoint rewind / time travel / fork | LC (`get_state_history`, `checkpoint_id` fork), C (`rewind_files`, `fork_session`), K (`session/fork`) | **no**: `RunCheckpointer` keeps only the latest checkpoint (`R:middleware.py:3171`) [V]. Blocked by design (C9) | keep N checkpoints per run in `RunCheckpointer`, for within-run rewind and fork by `checkpoint_id`; journal size is the cost. P2, needs a decision | no change |
| 39 | Sessions / conversation continuity | all | **yes**: the memory service replays the window; the thread is the run [V] | — | — |
| 40 | Slash commands | C, K | **no** (none in LC/DA) | **do not add** as text dispatch: it is the P0 Claude has (A1). A front end may map `/skill-name` to Deep Agents' `pinned_skills` input. P3 | no change |
| 41 | Hooks (lifecycle) | C (33 events), K (20), O (`RunHooks`/`AgentHooks`), LC (middleware hooks) | **yes**: `h.wrap(hooks=)` plus LangChain middleware through `middleware=` [V react.md "Where … hooks go"] | — | — |
| 42 | Image/file **input** | C (multimodal stream input: not reachable), K (image blocks: not reachable), O, LC | **yes** when the input is a list of messages (`R:adapters/langgraph.py:219-220` passes them through) [V]. A plain string input is text only | the surfaces (AG-UI, A2A) need to send content blocks [Inf]. P2 | no change |
| 43 | Image/file **tool output** | O (`ToolOutputImage`), LC (content blocks, `content_and_artifact`), DA (multimodal `read_file`) | **no**. Graph tools: content is JSON-dumped and the artifact dropped [Inv lc G3]; lcfix WIP keeps the library's message. Harness function tools: always text (`R:tools/convert/__init__.py:14-21` `text_of`) [V] | graph tools: lcfix `_made`. Harness tools: let `tools/convert/langchain.py` return content blocks when a function returns them | no change for text tools |
| 44 | Thinking / reasoning effort | C (`thinking`, `effort`), K, O (`reasoning`) | **yes**: `effort=` for a model name (`R:react.py:339-344`); a chat model's own thinking settings are used as given [V]. Reasoning is not streamed as answer text (lcfix) | — | — |
| 45 | Prompt caching | C, K (internal), O, LC (Anthropic, Bedrock) | **yes**: `PromptCache` wraps `AnthropicPromptCachingMiddleware` (`R:middleware.py:1859-1898`) [V] | fix the memory breakpoint (row 6) | memory breakpoint: no accuracy change (cost only) |
| 46 | Cost caps | C (`max_budget_usd`), O (usage), harness | **yes**: `Budget` → `Budgeted` (`R:automatic.py:97-118`) [V] | count decision-model spend (row 8d) | — |
| 47 | Step limit with a final answer | C/O/K (`max_turns`), LC (`ModelCallLimitMiddleware`) | **yes**: `AnswerAtLimit` (`R:middleware.py:2054`) [V] | keep (ReAct's own behaviour) | — |
| 48 | Loop/stall detection | K (repeat breaker) | **yes**: `StallGuard` (`R:middleware.py:2122`) [V] | keep: no library equivalent (`ToolCallLimitMiddleware` only counts) | — |
| 49 | Retries (tool, model) | LC (`ToolRetryMiddleware`, `ModelRetryMiddleware`), O, K | **yes**: harness read retries, plus `ModelRetryMiddleware` opt-in [V] | row 7b (no double retry) | no change |
| 50 | Dangling tool-call repair | DA (`PatchToolCallsMiddleware`) | **yes** (Deep Agents default) [Inv da row 79] | — | — |
| 51 | Refusal handling | LC/DA (`ModelRefusalMiddleware`) | **yes** (Deep Agents default stack) [Inv da] | — | — |
| 52 | Unsupported-content fallback | DA (`UnsupportedContentMiddleware`) | **yes** (Deep Agents default) | — | — |
| 53 | Self-evaluation / goal loop (Kimi goal mode, Claude `/loop`) | K, C, DA (`RubricMiddleware`) | **opt-in**: `RubricMiddleware` through `middleware=`, untested under ReAct [Inv da row 84] | `RubricMiddleware` (Deep Agents) as the library implementation: an offline test, and a documented recipe. P3 | **needs Haiku** if it is ever a default (it would add model calls) |
| 54 | Scheduling | C, K (cron) | **yes**: harness schedule tools (`R:schedules.py:1-25`) [V] | keep (B, row 14) | — |
| 55 | Model profiles (per-model prompt tuning) | DA (`HarnessProfile`: Opus, Sonnet, **Haiku 4.5**, Codex, Nemotron; `ProviderProfile`) | **no** for model names: `RoutedChatModel` makes the profile lookup miss [Inv da P1-3] | make `RoutedChatModel` expose provider and model, so `_harness_profile_for_model` and `apply_provider_profile` resolve (`R:gateways.py:688`, `:823`) | **needs Haiku: the most likely change to move the score.** Haiku 4.5's built-in profile changes the system prompt suffix. Measure plain and backend on all 357 turns (or a triage subset) |
| 56 | File permissions (allow/deny paths) | C (rules), K, DA (`permissions=`) | **yes**: `permissions=` [V `R:react.py:432`] | — | — |
| 57 | Tool call caps per tool | LC (`ToolCallLimitMiddleware`) | **opt-in** [V doc] | — | — |
| 58 | Notify the user mid-turn | C (`SendUserMessage`, `PushNotification`), K (`NotifyUser`) | **partly**: `get_stream_writer` → `CUSTOM` events; no model tool [Inv lc] | a model-callable "progress" tool that writes a `CUSTOM` event. P3 | no change (opt-in) |
| 59 | Tracing | O, LC (LangSmith) | **yes**: OTel spans; LangSmith through env, since callbacks run [Inf] | honour `omit_payload` (row 13) | — |
| 60 | Realtime / voice | O | **no**. LangChain has no realtime agent | out of scope for ReAct (not borrowable from LC) | — |
| 61 | Agent teams, cross-session messaging, worktrees | C, K (tower) | **no**, deliberately: disabled even in Claude (L7) and Kimi (tower off) | not borrowed. The background `update_async_task` covers "send a running helper new instructions" | — |

## 2.3 Harness code in ReAct that duplicates a library implementation

| Harness piece | Library counterpart now available | Is it a duplicate? | Recommendation | Measured reason |
|---|---|---|---|---|
| `FitWindow` (`R:middleware.py:2559`) | `SummarizationMiddleware` (LC, DA), `ContextEditingMiddleware`, Deep Agents `_overflow_clip`/`ContextOverflowError` fallback | **Partly.** Its tool narrowing, description compaction, refit on server refusal and memory-trim steps exist in no library. Only its "turns" step overlaps summarization. | **Keep `FitWindow`** as the overflow guard. In capability mode it already hands turns to Deep Agents' summarization (`turns=False`). **Measure** handing lean-mode turns to `SummarizationMiddleware` at windows of 32k or more. | Granite 8k: 8/10 with `FitWindow` against 7/10 without, 3 refused (react.md:424-430). Small windows: Deep Agents summarization 6/8 against `FitWindow` 8/8 (`R:react.py:389-390`). At ≥32k on Haiku: **needs a measurement**. |
| `ThisTurnsResults` | `ClearToolUsesEdit` | **No.** It is a subclass that scopes clearing to the current turn and names the `read_file` path. | Keep. | Reasoned (memory's replayed window is already sized), not measured: **needs a measurement** only if someone proposes plain `ClearToolUsesEdit`. |
| `Planning` | `TodoListMiddleware` | **No.** It is the library class plus a gate. | Keep the gate. Adopt a user's `TodoListMiddleware` (row 5a). | Cache rewrite on 6/12 long cases, cost up a third, when offered early (`R:automatic.py:51-54`). |
| `PromptCache` | `AnthropicPromptCachingMiddleware` | **No.** It wraps the library middleware and adds routed-name resolution and the history breakpoint. | Keep. Fix the memory breakpoint. | "Marking [the model's own answer] changed the answers after it" (react.md, The middleware): an accuracy observation, so own answers are not marked. |
| `BackgroundTasks` (`R:bridges/background_tasks.py`) | `AsyncSubAgentMiddleware` | **No.** It reuses Deep Agents' schemas and state. Deep Agents needs an Agent Protocol server (L4); the harness executes in-process. | Keep. Deep Agents' own is used when the user gives a remote sub-agent. | n/a (a capability, not accuracy) |
| `AnswerAtLimit` | `ModelCallLimitMiddleware` | **No.** It is a subclass that adds one answer call. | Keep in ReAct. In `middleware()`, the user's own wins (row 8a). | n/a |
| Harness read retries (`R:middleware.py:1178-1206`) | `ToolRetryMiddleware`, `RetryPolicy` | **Overlaps**, coordinated: ours stands down when the graph retries. | Keep (DECISIONS C6). Ours knows read from write; the library's does not. | n/a (deterministic, tested offline) |
| Kept results, `ReadTools`, `query_result` | Deep Agents `FilesystemMiddleware` eviction | **Overlaps.** | Keep in lean mode; capability mode already uses Deep Agents'. | Large-result class: lean 10/10 completed against Deep Agents 5/10, with a third of the tokens (react.md "The base"). |
| Tool hints / narrowing / `tool_search` | `LLMToolSelectorMiddleware`, `ProviderToolSearchMiddleware` | **Overlaps.** | Keep ours as the default; the library's are opt-in and compose. | Selector 6/10, coverage 0.60, against `FitWindow` 8/10 (react.md:432-434). |
| `StallGuard` | none (Kimi's repeat breaker is internal to Kimi) | no | Keep. | n/a |
| `JournalBackend` | Deep Agents `StateBackend` | **No.** It is a subclass that adds caps. | Keep. | n/a |
| ReAct's own general-purpose helper spec (`R:react.py:358-363`) | Deep Agents' auto-added general-purpose sub-agent (with `HarnessProfile.general_purpose_subagent` and skills) | **Duplicate**, and worse: no skills and no profile settings [Inv da P2-5]. | **Switch** to Deep Agents' own construction, then add the harness guards by name: let Deep Agents build it and inject `StallGuard`, the step limit, `ModelHooks` and `PromptCache` through `HarnessProfile.extra_middleware`, or patch the built spec. | Prompt is unchanged unless a profile applies, so it rides on the #55 measurement. |
| Sub-agent `interrupt_on` → `irreversible` (`R:react.py:645-652`) | Deep Agents and LangChain HITL `interrupt_on` (with `when`, `allowed_decisions`) | **Duplicate**, and lossy. | **Switch** to the library's (row 2c). | n/a |
| `ModelHooks` context merge `with_question` (`R:middleware.py:1785-1828`) | none. It moves the memory system message into the user turn. | no | Keep, but run it before user middleware so `PIIMiddleware` sees the context (row 11). | Reordering changes no request when no user middleware is given, so the golden test proves it offline. |

## 2.4 Ranked implementation plan

**Effort:** S = under half a day; M = 1-2 days; L = 3 or more days.

**Proof:**
- "offline" means provable with the scripted models and fakes, with the golden snapshot unchanged.
- "Haiku" means a paid measurement is needed. The cheapest form is given.

### P0: governance, data exposure (all offline)

| # | Item | Effort | Job | Proof |
|---|---|---|---|---|
| 1 | Deep Agents skills route makes sandbox and disk writes ungoverned (row 3a, `R:skills.py:543`, `R:middleware.py:323-330`) | M | langchain-fixes (L40) | offline: assert `results(events)` includes `write_file` with skills present (the existing test does not assert governance) |
| 2 | PII: memory context checked by `PIIMiddleware`; harness records the redacted question (row 11 parts 2-3) | M | langchain-fixes (extend L38) | offline (scripted model; the probe1 pattern) |
| 3 | Kimi: always `set_session_mode` and verify `currentModeId` (row 2e; `F:adapters/acp.py:913`) | S | bg-kimi (L46): **not in its tree yet** | offline: real Kimi 2.1.1 with the scripted endpoint and a config holding `default_permission_mode = "auto"` (the inventory probe) |
| 4 | Claude: `verbatim_prompts=True` for end-user text (A1); per-tenant+user `CLAUDE_CONFIG_DIR`, auto memory off (A2) (rows 4b, L29, L31) | M | **no job**: give it to bg-claude-da or open claude-fixes | offline: fake CLI checks argv and env. Real CLI behaviour: gateway live test (L37) |
| 5 | OpenAI: refuse a shared session object on `h.wrap` / accept a factory (row 4d) | S | openai-fixes (L42) | offline (`p_sess.py`) |

### P1: library features broken through the harness, and ReAct parity gaps that cost behaviour

| # | Item | Effort | Job | Proof |
|---|---|---|---|---|
| 6 | `middleware()` yields to a user's `TodoListMiddleware`, `ContextEditingMiddleware`, `ModelCallLimitMiddleware` or `AnthropicPromptCachingMiddleware` (rows 1b, 5a, 6, 8a). In ReAct, drop the corresponding `SET_BY` refusals where the library instance can stand in | M | langchain-fixes (G6) | offline (probe1 builds; golden unchanged when nothing is given) |
| 7 | One question per call: HITL decision = approval; per-call HITL; `response_schema`; ReAct `interrupt_on=` passed through, sub-agent gates kept, batch edit for every call (rows 2a-2c) | M | langchain-fixes (L39, L41) | offline (probe5: 1 approval, not 2) |
| 8 | ReAct thread state across turns: carry `files` and `todos` (row 9b) | M | langchain-fixes (L41) | offline for the mechanism (REACT_TURN2 reads the file). **Haiku**: backend variant on the multi-turn/follow-up subset only, since lean is untouched until a long task |
| 9 | Deep Agents model profiles for routed names (#55) | S-M | langchain-fixes (L41) | offline that the profile resolves (`profile_check.py`). **Haiku: full plain and backend run (357 turns), or request-diff triage first.** Every Haiku request gets the Haiku 4.5 suffix, so most turns change |
| 10 | Graph-tool artifacts and content blocks kept; library tool results left as returned outside harness loops (C2, C3; #43) | M | langchain-fixes (WIP `_made`) | offline. The golden test must stay identical: ReAct's own tools are harness tools |
| 11 | `langchain.mcp` annotations read as risk (#2) | S | langchain-fixes (G8) | offline |
| 12 | False NO_ANSWER: `return_direct`, entrypoints, non-message state (G4; unblocks #13) | M | langchain-fixes (G4) | offline (probe2, probe4) |
| 13 | Provider/hosted tools in ReAct: keep dict tools in `own`; record `server_tool_use` blocks (#8) | M | **no job**: react-background or langchain-fixes | offline (a scripted model emitting server-tool blocks). One Haiku call with `web_search` to see the real block shape |
| 14 | Claude: governance fails closed (`F:adapters/claude.py:1326-1330`); `ToolSearch`/`WaitForMcpServers` read; ExitPlanMode asks; WebFetch, WebSearch and Skill ask (B1, B2) | M | **no job** | offline (fake CLI) |
| 15 | Kimi: plan review asks with all options; sub-agent arguments from the wire log; elicitation form advertised; compaction notices filtered (rows 2e, 1e, 19, 20) | M | bg-kimi (L47): not in tree | offline (real Kimi with the scripted endpoint) |
| 16 | OpenAI: dict `run_config`; Responses through Bifrost or a clear refusal (F3); sandbox Memory on the gateway (rows 1f, 4d) | L | openai-fixes (L43-L45) | offline (probes). A real gateway is needed for Responses through Bifrost |
| 17 | Merge: `Workflow` stays supported (bg-claude-da wins over react-background's `115348a`) | S | react-background and bg-claude-da at merge | offline |

### P2: parity additions and smaller collisions

| # | Item | Effort | Job | Proof |
|---|---|---|---|---|
| 18 | `SummarizationMiddleware` with `middleware()`: `_with_task` knows LangChain's summary; `FitWindow` stops dropping turns when the graph summarizes (rows 1a, 1c) | M | langchain-fixes (G10) | offline (request snapshot). Accuracy: way B only, ReAct untouched |
| 19 | Lean-mode summarization at windows of 32k or more, as a candidate default (#20) | S to build the variant | react-background | **Haiku**: `main` vs `main+summary` on long multi-turn cases at `--window 32000`, `--repeats 1` |
| 20 | General-purpose helper built by Deep Agents (skills, profile), harness guards injected (2.3) | M | langchain-fixes | offline. Accuracy rides on #9 |
| 21 | `compact_conversation` declared `read`; manual compaction opt-in (#22) | S | langchain-fixes | offline |
| 22 | `RunCheckpointer` keeps N checkpoints: within-run rewind and fork (#38) | M | needs a user decision (C9) | offline |
| 23 | Plan mode for ReAct: read-only governance mode plus plan approval through HITL on `write_todos` (#16) | M | needs a user decision | offline |
| 24 | Decision-model spend budgeted; AutoMode blocks recorded (rows 8d, 12) | M | no job | offline |
| 25 | `FitWindow` keeps `defer_loading` extras (row 17) | S | langchain-fixes | offline |
| 26 | Memory `cache_control` for routed Anthropic names (row 6) | S | langchain-fixes | offline (request snapshot: the breakpoint is present). Cost only, no accuracy change |
| 27 | No double model retry: worker vs `ModelRetryMiddleware` (row 7b) | S | no job | offline |
| 28 | Claude `RemoteTrigger` disallowed (row 14); stale built-in risk table (L34) | S | no job | offline |
| 29 | Routed names in `ModelFallbackMiddleware` and `ModelRouterMiddleware` (#33, #34) | S | langchain-fixes | offline |
| 30 | Guardrail recipe (`@before_agent`/`@after_agent` + `jump_to`) documented and tested (#30) | S | langchain-fixes (docs) | offline |
| 31 | Kimi goal-mode continuations forwarded (row 15c) | M | bg-kimi | offline (real Kimi with the scripted endpoint) |

### P3: nice to have

Each item is offline unless noted.

- Computer-use executor (#9).
- Handoffs through `langgraph-swarm`, after a dependency decision (#13).
- A fan-out recipe with `Send` (#11).
- A "progress" notify tool (#58).
- `RubricMiddleware` test and recipe (#53). Haiku only if it is ever a default.
- Kimi `plan` updates as events (row 5b).
- Claude `compact_boundary` as an event (row 1d).
- Claude token deltas and OpenAI item events (row 10).
- `JournalSessions.list_subkeys`; closing Kimi sessions (row 9c).
- MCP retry compounding (row 7c).
- TracePolicy honoured (row 13).
- Partner-sandbox tests (#7).
- `pinned_skills` test (#17).
- `OpenAIModerationMiddleware` test (#32).

### What can be proven offline vs what needs Haiku

- **Offline (free), the default:**
  - all P0 items;
  - all governance, recording and streaming fixes;
  - every collision decision in §1.1;
  - every opt-in capability (provider tools, plan mode, compaction tool, guardrails, fallback and routing names).

  Proof: the golden snapshot (`test_react_golden.py`) stays byte-identical, and a long-task request
  snapshot is added.
- **Needs Haiku**, in priority order. Run request-diff triage first; it is free, given recorded
  replies.
  1. **#9 Deep Agents model profiles.** It changes the system prompt of every Haiku turn. Full
     plain and backend run.
  2. **#8 carried files and todos.** Backend variant, multi-turn subset.
  3. **#19 lean summarization at windows of 32k or more.** Long multi-turn subset at a forced window.
  4. **Re-check plain vs backend (352 vs 356).** Free if the paired per-turn results exist; if
     backend wins significantly on Haiku, revisit the lean default per model class.
  5. TEST_PLAN 5.16 items that already need Haiku: background tool use, steering, real Claude CLI.

---

## 3. Verified facts vs inferences: summary

**Verified in this session [V]:**
- ReAct's construction and stack.
- `Planning` is a `TodoListMiddleware` subclass.
- `PromptCache` wraps `AnthropicPromptCachingMiddleware`.
- `ThisTurnsResults` subclasses `ClearToolUsesEdit`.
- `AnswerAtLimit` subclasses `ModelCallLimitMiddleware`.
- `CARRIED = ("async_tasks",)`.
- `RunCheckpointer` keeps the latest checkpoint only.
- Sub-agent `interrupt_on` is flattened (`R:react.py:645-652`).
- Dict provider tools are refused (`R:react.py:570-571` → `R:tools/sources.py:559-567`).
- `_SkillsBackend(StateBackend)` (`R:skills.py:543`).
- Steering via `HarnessTools`.
- Harness elicitation, schedules and `ask_user` exist.
- The golden-request test exists.
- The benchmark variant mechanism exists.
- Branch contents: lcfix `071663b` plus WIP; native-skills WIP; bgkimi lacks `set_session_mode` and the ExitPlanMode fix; bgclaude lacks `verbatim_prompts`, `setting_sources` and `on_failure`; oaifix holds only the upgrade plus a WIP edit.
- Spot-checks of the final tree: `F:adapters/claude.py:1326-1330`, `:157`; `F:adapters/acp.py:183-184`, `:913-916`; `F:adapters/openai_agents.py:1104`.
- `langgraph-swarm`/`langgraph-supervisor` and `langchain-quickjs` are not installed.

**From the inventories [Inv]:** every probe result (probe1-7, p_sess, p_chat, REACT_TURN2 and the
Kimi probes) and every line reference into the Claude, Kimi and OpenAI adapters not listed above.

**Inferences [Inf]:**
- The 357-turn figures (given by the caller; the result file was not found).
- The significance estimate for 352 vs 356.
- That provider server tools may not pass through Bifrost chat completions.
- That LangSmith tracing works through env.
- That surfaces send only text input.
- That LangChain has no computer-use executor and no plan-mode implementation.
- That Kimi's extra answer call should follow D1.
- The request-diff triage method.
