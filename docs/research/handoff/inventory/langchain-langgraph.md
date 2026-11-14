# LangChain 1.5.0 / LangGraph 1.2.22: full feature inventory against the Trellis harness

Read-only audit, 2026-11-14. No code was changed and no paid model was called. All probes used scripted
fake chat models, ran offline, and had every provider key unset.

## Sources

- **Library source:** `/var/tmp/cons/reactbg/.venv/lib/python3.13/site-packages`:
  - langchain 1.5.0, langchain-core 1.8.0, langgraph 1.2.22;
  - langgraph-prebuilt 1.1.0, langgraph-checkpoint 4.3.2;
  - deepagents 0.7.38, langchain-anthropic 1.7.10, langchain-openai 1.6.6.
- **Harness:** `/var/tmp/cons/reactbg`, branch `claude/cool-maxwell-w5avo8-react-background` at 3d892eb, which pins `langchain>=1.5,<1.6`.
  - `/var/tmp/cons/final-agent-harness` still pins `langchain>=1.4,<1.5` and `langchain-core<1.7`. Its LangChain 1.5 support is therefore untested.
  - Every code path behind the gaps below is identical in both trees: `_message` text-only results, the `isinstance(chunk.content, str)` stream filter, `_last_ai_text`, and `is_hitl`.
- **Docs fetched (no blocked hosts):**
  - docs.langchain.com/oss/python/langchain/middleware/built-in
  - docs.langchain.com/oss/python/langgraph/streaming
  - docs.langchain.com/oss/python/releases/changelog
- **Existing tests:** 174 tests across 7 files pass offline in 16 s:
  - test_create_agent_library, test_hitl_middleware, test_langgraph_library, test_tool_errors_library, test_tool_updates, test_langgraph, test_middleware.
- **New probes:** `/var/tmp/cons/inventory-tmp/langchain/probe1.py` to `probe7.py`. Each result quoted below is their output.

## The four ways LangGraph is used

| Way | What the harness does to the graph |
|---|---|
| **A** `create_agent` wrapped without `trellis.middleware()` | Runs a *copy* whose async model node gets two more innermost `awrap_model_call` handlers, `answered_in_full` and `_narrow` (`middleware.py:1558-1588`, `adapters/langgraph.py:503-512`, logged once at `:480-491`). Every `ToolNode` is governed by a process-wide monkeypatch (`middleware.py:1078-1156`). |
| **B** `create_agent(..., middleware=trellis.middleware())` | Adds 9-11 middlewares (`react.py:130-174`, `:177-253`): Planning ("TodoListMiddleware"), HarnessTools, GovernTools, StallGuard, AnswerAtLimit ("ModelCallLimitMiddleware"), Budgeted, ContextEditingMiddleware (only with `context_window`), ReadTools, FitWindow, ModelHooks, and PromptCache ("AnthropicPromptCachingMiddleware"). Without a checkpointer it is run with `RunCheckpointer` (`adapters/langgraph.py:588-600`). |
| **C** hand-built `StateGraph` / functional API | Only the input/output adapter (`adapters/langgraph.py:210-326`), the ToolNode monkeypatch, the accounting callback (`accounting.py:58`), and store, thread and config injection. The model's tools are never narrowed. |
| **D** `ReAct(...)` | A Deep Agents `create_deep_agent` graph with all of B, `RunCheckpointer` and a routed model (`react.py:263-449`). User middleware is added before ModelHooks, and a name that collides with a harness default is refused with a clear error (`react.py:738-799`). |

Legend:
- **P/T/B/O:** passes through / transforms / blocks / overrides.
- **Gov/Rec:** whether calls are governed by the catalog and recorded (journal, records, events).
- **Status values:** works, caveat, broken, blocked-by-design, untested, not reachable.
  - "untested" means the feature reaches the library unchanged but no repo test covers it.
  - "probe OK" means my probe ran it successfully.

---

## 1. `create_agent` parameters (langchain/agents/factory.py:895-1925)

| Feature | What it does | Harness handling (file:line) | Gov/Rec | Tested | Status |
|---|---|---|---|---|---|
| `model` (str via `init_chat_model`, or `BaseChatModel`) | Chooses the chat model. | **P** in A/B/C. **T** in D: a name goes through `routed_chat_model` (`react.py:345`). ModelHooks adds gateway headers only for routed models (`middleware.py:1670-1675`). | Every call is accounted by a callback (`accounting.py:58`). | test_create_agent_library (every test) | works |
| `tools`: BaseTool or callable | The tools the ToolNode runs. | **T**. Every call goes through `governed_call` (`middleware.py:411-503`); the risk comes from `metadata.side_effects`, else `annotations`, else UNDECLARED, which asks (`:1212-1222`). See §2 for how results are rewritten. | yes | test_langgraph::test_a_create_agent_graph_runs_with_harness_tools | caveat (result rewriting, §2) |
| `tools`: dict (provider built-in tools, e.g. web_search) | Bound to the model as a schema; the provider executes it. | **P**. `_offers` keeps unknown names (`runtime.py:457-468`). The provider runs the call server-side, so the harness never sees it. | no (server side) | none | untested |
| `system_prompt` (str or SystemMessage) | The system message. | **T** in A/B/D: `_dated` appends "It is now <day> (UTC...)" to the system message on every call (`middleware.py:1742-1759`, from `_narrow` `:1517` and ModelHooks `:1663`). Probe: `"You count stock.\n\nIt is now Sat 2026-11-14 (UTC; the user's zone is not known)."` | n/a | none (probe) | caveat (library behaviour changed) |
| `middleware` | The middleware stack. | **P** in A and C. In B, any user middleware named ModelCallLimitMiddleware, TodoListMiddleware, ContextEditingMiddleware or AnthropicPromptCachingMiddleware makes `create_agent` raise **AssertionError "Please remove duplicate middleware instances."** (probe1). D refuses such names with a ConfigurationError (`react.py:769-799`). | n/a | none for collisions | **broken** (B) |
| `response_format`: ToolStrategy | Structured output through an artificial tool. | **P**. The answer is `structured_response` (`adapters/langgraph.py:302`). | n/a | test_create_agent_library::test_a_tool_strategy_answer_is_the_runs_answer_and_a_bad_one_is_asked_again | works |
| `response_format`: ProviderStrategy | Native JSON schema output. | **P** | n/a | ::test_a_provider_strategy_answer_is_read_from_the_models_own_structured_reply | works |
| `response_format`: AutoStrategy / raw schema / dict | Detects which strategy the model supports. | **P**. Probe4: `SUCCESS Count(sku='a', units=7)`. | n/a | none (probe OK) | untested |
| ToolStrategy `handle_errors`, `tool_message_content`, MultipleStructuredOutputsError | Retries a bad structured output and words the returned message. | **P** | n/a | the ToolStrategy test covers the retry | works (retry); untested (others) |
| `state_schema` | Extends AgentState. | **P**. B adds `trellis_harness_tools` (`middleware.py:198`). | n/a | ::test_a_middlewares_state_and_the_runs_context_reach_its_hooks_and_tools | works |
| `context_schema` and `context=` | Runtime context. | **P** through `framework_options={"context": ...}` (`adapters/langgraph.py:115, 450-461`). | n/a | test_langgraph::test_the_graphs_context_and_durability_pass_through | works |
| `checkpointer` | Persists the thread. | **T**. `thread_id` is forced to the conversation, or to the run id in run-scoped mode (`:413-430, 515-520`). Checkpoints get the `trellis_run_id` metadata (`:428-429`). A durable checkpointer is warned once (`:200-208`). In B without a checkpointer, `RunCheckpointer` is used, which keeps the latest checkpoint only (`middleware.py:3171`). | n/a | test_langgraph (several) | works / caveat (no history in B or D) |
| `store` | Long-term memory store. | **O**. With no store of its own, the graph gets `MemoryStore` (the memory service) while memory is on (`adapters/langgraph.py:464-476`). A store of its own is kept and logged once. | n/a | test_langgraph_library::test_a_graphs_own_store_is_what_its_tools_are_handed | works (by design) |
| `interrupt_before` / `interrupt_after` | Static breakpoints. | **P**. A run that reaches one ends SUCCESS, and a later run with `None` input continues. | n/a | test_langgraph_library::test_a_static_breakpoint_stops_the_graph_and_a_run_given_nothing_goes_on | works |
| `debug` | Verbose printing. | **P** | n/a | none | untested |
| `name` | Graph name; sets `AIMessage.name` and the `lc_agent_name` metadata. | **P**. `h.wrap(id=)` is separate. | n/a | none | untested |
| `cache` (BaseCache) | Node cache. `create_agent`'s own nodes declare no CachePolicy. | **P**. `graph.copy` keeps it (`adapters/langgraph.py:410`). | n/a | the CachePolicy test covers StateGraph only | untested (for create_agent) |
| `transformers` (StreamTransformer factories) | Stream projections for `stream_events(version="v3")`. | **Not driven.** The harness only calls `astream(stream_mode=[...], version="v2")` (`:236-242, 267-273`). The default ToolCallTransformer and SubagentTransformer, middleware transformers (PII, InternalCall) and user transformers never run. | n/a | none | **not reachable** |
| Default config `recursion_limit=9999`, `ls_integration` metadata | create_agent's own `with_config`. | **P**. `framework_options.recursion_limit` overrides it (`:422-427`). | n/a | the recursion_limit test covers StateGraph | works |

## 2. Agent loop, tools and the middleware API

| Feature | What it does | Harness handling | Gov/Rec | Tested | Status |
|---|---|---|---|---|---|
| `before_agent` / `after_agent` / `before_model` / `after_model` nodes | Graph nodes around the loop. | **P** | n/a | ::test_the_hooks_around_the_agent_and_the_model_run_in_the_librarys_order | works |
| `hook_config(can_jump_to)` / `jump_to` | Jumps to model, tools or end. | **P** | n/a | ::test_a_hook_that_jumps_to_the_end_ends_the_run_with_its_message | works |
| `wrap_model_call` / `awrap_model_call` | Wraps the model call. | **P**. Harness handlers are innermost in A (`middleware.py:1580`) and in list order in B. | n/a | test_create_agent_library, test_fit_window | works |
| `wrap_tool_call` / `awrap_tool_call` | Wraps the tool call. | **P**. Harness governance sits *inside* every wrapper, at `ToolNode._execute_tool_async` (`middleware.py:1155`). | yes | test_tool_errors_library::test_a_wrapper_of_yours_gets_the_tools_error_and_answers_in_its_place | works |
| Sync-only custom middleware (only `wrap_*`) | Library: raises NotImplementedError when the graph runs async. | The harness always runs async (`ainvoke`/`astream`), so a sync-only `wrap_tool_call` or `wrap_model_call` fails (`types.py:815-825`). | n/a | none | blocked-by-design (only the async path runs) |
| `@dynamic_prompt` | Builds the system prompt per call. | **T**: the date line is appended after it. Probe: `"DYNAMIC\n\nIt is now Sat 2026-11-14 ..."`. | n/a | ::test_the_hooks... (uses dynamic_prompt) | caveat |
| `ExtendedModelResponse` (Command from wrap_model_call) | Updates state beside the model reply. | **P** | n/a | ::test_a_model_wrapper_updates_the_state_beside_the_models_reply | works |
| Middleware `tools` (added per call) | Extra tools contributed by a middleware. | **P** and governed. | yes | ::test_a_tool_a_middleware_adds_per_call_is_offered_run_and_governed | works |
| `ModelRequest.override(tools=...)` (dynamic tools) | Changes the tools for one call. | **T**. `_offered` narrows the tools to what the run offers (`middleware.py:1473-1489`). In A, the harness readers are added as schemas (`:1508-1513`). | n/a | test_middleware, test_tool_choice | works / caveat |
| Middleware `state_schema` and private attributes | Extra state, kept out of the graph's input and output. | **P** | n/a | test_middleware::test_a_middleware_state_never_reaches_the_graphs_input_or_output | works |
| `TracePolicy` / `omit_payload` / `configure_trace_policy` | Strips payloads from LangSmith middleware spans (AutoMode, Router and Refusal set `omit_payload`). | Ignored by the harness's OTel spans. ModelHooks' `chat` span keeps the full request messages, masking credentials only (`middleware.py:1666, 2006`). | n/a | none | caveat |
| `InternalCallTransformer` / `lc_internal_call` metadata | Hides middleware-internal model calls from v3 streams. | Not used. Summarization, tool selection and the emulator also tag `nostream`, which v2 `messages` honours. Decision-model classifiers (AutoMode, Router) are not chat models. | n/a | none | works (via nostream) |
| Invalid tool calls (`_patch_invalid_tool_calls`) | Library: turns invalid calls into error ToolMessages. | **T** in B: `ModelHooks.aafter_model` answers them and jumps back to the model (`middleware.py:1630-1651`). `_unbroken` sends their arguments as `{}` (`:1456`). | n/a | test_langgraph_library::test_arguments_that_do_not_fit_a_tools_schema_are_told_to_the_model, matrix X05 | works (B changes behaviour) |
| Parallel tool calls (one Send per call in create_agent) | Calls run concurrently. | **T**. Writes are serialised in the model's order; when one call asks a person, the step's other calls finish before the pause (`middleware.py:671-683, 1418-1442`). | yes | test_tool_updates::test_a_write_beside_a_call_that_asks_ends_before_the_pause_however_it_is_scheduled | works |
| Two identical calls in one step | Library runs both. | **O**. The second does not run and returns "Same call as call_1_0 above (the same tool, the same arguments): see its result." (`middleware.py:449-453`, `:1335`). Probe1: `ran ['stock a']`. | yes | matrix F25d | caveat (library behaviour changed) |
| Tool returning a Command (state update, goto) | Updates state or routes. | **P**. The Command is serialised into the journal and replayed (`middleware.py:542-605`). | yes | test_tool_updates (10 tests), test_langgraph_library::test_a_tools_command_updates_the_graphs_state_and_is_a_call_of_the_run | works |
| `InjectedState` / `InjectedStore` / `ToolRuntime` | Hidden tool arguments. | **P**. The harness records only the model's arguments. | yes | ::test_a_tool_is_handed_the_state_and_the_runtime_and_the_run_records_neither | works |
| `response_format="content_and_artifact"` tool | ToolMessage.artifact carries data the model does not read. | **O**. `_answered` rebuilds a text-only ToolMessage from the outcome (`middleware.py:524, 702-719`). Probe1: harness `artifact= None`; the library alone gives `artifact= {'rows': [1, 2, 3]}`. Citations are read before being dropped (`:478`). | yes | none | **broken** |
| Tool returning content blocks (image/file, multimodal) | A list content reaches the model as multimodal input. | **O**. `text_of` JSON-dumps the list (`tools/convert/__init__.py:14-21`). Probe1: `'[{"type": "text", ...}, {"type": "image", "base64": "AAAA", ...}]'` arrives as a *string*, so the model never sees the image. | yes | none | **broken** |
| Graph's own tool result: a table of 5+ rows | Library returns the text unchanged. | **T** in A/B/D: a note about where the rows are kept and how to `query_result` them is appended (`tools/results.py:194-217`, via `_message`). Probe6: the table message ends with "...never offer it to them as a file." | yes | test_query_export | caveat (behaviour changed; the doc says this applies to harness tools only) |
| Graph's own tool result over 80,000 characters | Library returns it whole. | **O** in A/B/D: replaced by a "Tool result too large..." stub, with `read_file` and `query_result` added to the model's tools (`middleware.py:747-765, 1492-1518`). Probe6: `len=11473`; tools on the second call were `['table','big','read_file','query_result']`. | yes | matrix F26 | caveat (`langgraph.md` says "a result of a *harness* tool") |
| `return_direct=True` tool | Library ends the loop on the ToolMessage. | **Broken.** `extract` reads only `structured_response` or the last AI text (`adapters/langgraph.py:296-306, 921-925`); `_unanswered` then makes the run ERROR NO_ANSWER (`pipeline.py:898-922`). Probe2: `ERROR None code='NO_ANSWER'` in A and B; the library gives `('tool','direct a')`. | yes (call) | none | **broken** |
| Tool error: `handle_tool_errors` (ToolNode default) | Library decides whether an error fails the graph. | **P** in A and C: the error is re-raised for the graph to handle (`middleware.py:691-699`). **O** in B: GovernTools makes an unhandled error a message the model reads (`:366-368, 1159-1175`). | yes | test_tool_errors_library::test_an_error_nothing_of_the_graph_handles_fails_the_run_as_langgraph_does, ::test_under_the_harness_middleware_an_error_nothing_handles_is_read_by_the_model | works (B by design) |
| Harness retry of a read whose error may pass | Library does nothing. | **T**: 3 tries where the graph has no retry of its own (`middleware.py:1178-1206`). | yes | ::test_where_the_graph_does_not_retry_its_tool_the_harnesss_retries_stand | works (behaviour added) |
| Reply cut at the output limit | Library ends the loop (no tool call). | **T**: one more model call, offered no tools (`answered_in_full`, `middleware.py:2181-2206`), in A, B and D. | n/a | matrix/benchmark | works (behaviour added) |
| Stopped or dangling calls in a checkpointed thread | Library: the provider returns 400 on the next turn. | **T**: STOPPED_CALL ToolMessages are prepended to the next turn's input (`adapters/langgraph.py:640-663`). Skipped when PatchToolCallsMiddleware is present. | n/a | test_langgraph::test_a_call_a_newer_message_stopped_is_answered_in_the_kept_thread | works (behaviour added) |

## 3. Built-in middleware (langchain/agents/middleware/__init__.py, plus provider packages)

| Middleware | What it does | Harness handling | Gov/Rec | Tested | Status |
|---|---|---|---|---|---|
| SummarizationMiddleware (`trigger`/TriggerClause, `keep`, `summarizer=`) | `before_model`: RemoveMessage(ALL), then a summary HumanMessage (`lc_source=summarization`) and the kept tail (`summarization.py:479-517`). | **P**. Its model call is tagged `nostream` and accounted. Overlaps: (1) the memory-context SystemMessage and the question are summarised away. `_with_task` cannot restore them because LangChain's summary is the first state message (`middleware.py:1762-1783`). (2) `added()` finds no question, so the turn's transcript is only the answer (`adapters/langgraph.py:866-894`). (3) In B, FitWindow (turns=True) also drops older turns, giving two owners of the window. D uses `turns=False`. | n/a | ::test_summarization_middleware_replaces_older_messages_with_its_summary (A only; asserts neither context retention nor the transcript) | caveat |
| HumanInTheLoopMiddleware, `interrupt_mode="batched"` | Approve/edit/reject/respond per model turn. | **T**. Becomes a harness Interrupt; decisions are mapped by `hitl_response` (`adapters/langgraph.py:712-766`). Needs a checkpointer; B and D supply RunCheckpointer. | approvals recorded | test_hitl_middleware (11) | works |
| HITL `interrupt_mode="per_call"` (langchain 1.4.6+) | One `{"type":"tool_approval"}` interrupt per call, with a `response_schema`. | **Broken.** `is_hitl` expects `action_requests` (`:712-720`), so it is treated as the graph's own interrupt: the question is the raw dict, and `approve` resumes with a non-Decision value. Probe5: `ERROR ... 1 validation error for tagged-union[ApproveDecision,...]`. | | none | **broken** |
| HITL plus harness governance | Two approval systems. | A tool HITL gates that is also undeclared or `write` is approved twice. Probe5: `approvals asked: 2`. The doc says to declare such tools read/write (`langgraph.md` "Gate a tool in one place"). | | doc only | caveat (overlap) |
| ModelCallLimitMiddleware | Thread and run model-call caps; `exit_behavior` end or error. | **P** in A/C. In B/D it **collides** with AnswerAtLimit, which carries the same name (`middleware.py:2054-2075`). Probe1: AssertionError. In D it is refused, pointing to `max_steps=`. | n/a | ::test_model_call_limit_middleware_ends_the_run_as_it_is_told_to (A) | works (A) / **broken** (B) |
| ToolCallLimitMiddleware | Per-tool or global call caps. | **P**. Blocked calls are not calls of the run. They count toward neither the budget nor StallGuard. | blocked calls not recorded | ::test_tool_call_limit_middleware_blocks_the_calls_past_its_limit | works |
| ModelFallbackMiddleware | Tries the next model on error. | **P**. Caveat on order in B: placed *after* `trellis.middleware()` (innermost), PromptCache has already decided caching for the first model (`middleware.py:1886-1898`). Placed before it, it works. ModelHooks turns openai errors into ModelError first, which the fallback still catches (any Exception). | n/a | ::test_model_fallback_middleware_asks_the_next_model_with_the_same_tools (A) | works (A); untested (B order) |
| ModelRetryMiddleware | Retries a failed model call. | **P**. Overlap: a queued run is also re-attempted by the worker on a retryable ModelError (`pipeline.py:925-929`). | n/a | ::test_model_retry_middleware_asks_the_model_again | works |
| ModelRefusalMiddleware (1.4.9+) | Raises ModelRefusalError on `refusal` content blocks. | **P**. The run ends ERROR with that error. Streamed text may already have gone out (library note). | n/a | none | untested |
| ModelRouterMiddleware (beta, 1.5) | A decision model picks the model in `before_agent`. | **P**. Caveats: (1) the picked model is a plain LangChain model, so it gets no gateway headers and the "not routed" warning fires (`accounting.py:125-142`). (2) Placed after PromptCache in B, the Anthropic cache settings can reach a non-Anthropic model. (3) The classifier's spend goes to LangSmith run metadata only (`langchain_core/decisions/base.py:169-184`), not through chat callbacks, so it is **not counted in the Budget**. | n/a | none | untested / caveat |
| AutoModeMiddleware (beta, 1.5) | A decision model blocks risky tool calls (p ≥ 0.5). | **P**. It sits outside harness governance, so an allowed call can still be asked about by the harness (two gates), and a blocked call never reaches the bridge and is **not recorded**. Its classifier spend is not counted in the budget (as above). | blocked calls not recorded | none | untested / caveat |
| PIIMiddleware (`apply_to_input`, `apply_to_output`, `apply_to_tool_results`; redact/mask/hash/block) | Redacts state in before/after_model, and redacts the wire through `_PIIStreamTransformer` (`pii.py:50-68, 673-678`). | **Partly bypassed.** (1) The stream transformer never runs (v2 astream). Probe1: deltas `['write to ada@e', 'xample.com now']` while the answer is `'write to [REDACTED_EMAIL] now'`. TEXT_MESSAGE events, AG-UI and `agent.run` events carry the PII. (2) The memory context is moved into the question *after* PII's before_model (`with_question`, `middleware.py:1785-1828`), so the context is never checked. (3) The harness writes its own copy of the question to memory and the run store from the input, unredacted. (4) The harness's redaction masks credentials in spans only (`redaction.py:1-10`). | | ::test_pii_middleware_redacts_what_the_model_is_sent_and_blocks_as_it_says (input/redact/block only) | **broken** (output streaming) / caveat |
| TodoListMiddleware | `write_todos` tool, a system-prompt section, and rejection of parallel calls. | **P** in A/C. In B/D the harness's Planning *is* a TodoListMiddleware under that name; it holds the tool back until a long task and drops the system-prompt section (`automatic.py:122-147`). A user's TodoListMiddleware **collides** (probe1: AssertionError). | n/a | test_react_long_tasks, test_react (Planning); none for user + middleware() | works (A) / **broken** (B) |
| LLMToolSelectorMiddleware | A model picks the tools before the call. | **P**. Overlap: harness narrowing applies after it (innermost in A) and can only shrink the set further. Its model is tagged `nostream` and accounted. | n/a | ::test_llm_tool_selector_offers_the_model_the_tools_its_selector_picked | works |
| ToolRetryMiddleware | Retries with backoff; `on_failure`. | **P**. The harness detects its retries and does not retry on top (3 tool runs, not 9). The tries count as one call, one approval and no stall (`middleware.py:1178-1206`). | yes | test_tool_errors_library (6 tests) | works |
| ToolErrorMiddleware (1.3.14+) | Converts errors to messages. | **P**. The harness's own retries still apply underneath it. | yes | ::test_tool_error_middleware_words_the_error_the_model_reads | works |
| LLMToolEmulator | A model answers in the tool's place. | **P**. The tool never runs, so it is **not governed or recorded**. Its model is accounted. | no | ::test_llm_tool_emulator_answers_for_the_tool_which_does_not_run | works (by design) |
| ContextEditingMiddleware / ClearToolUsesEdit | Clears old tool results in the request. | **P** in A/C. In B/D with a `context_window`, the harness's own ContextEditingMiddleware(ThisTurnsResults) holds that name (`react.py:200-215`, `middleware.py:2416-2449`), so a user's **collides** (probe1). `ReadTools` adds `read_file` for cleared results. | n/a | test_react_context, test_fit_window (harness's own); none for the user's | works (A) / **broken** (B with window) |
| ShellToolMiddleware (Host/Docker/CodexSandbox policies, RedactionRule) | A persistent shell tool. | **P**. The tool is undeclared, so every call asks a person. The library restarts the session on resume. | yes | ::test_shell_tool_middlewares_shell_asks_then_runs_in_the_librarys_session (Host only) | works (Docker/Codex untested) |
| FilesystemFileSearchMiddleware | glob_search and grep_search. | **P**. Undeclared, so it asks. | yes | ::test_file_search_middlewares_tool_asks_then_searches_the_folder | works |
| ProviderToolSearchMiddleware | Defers tools behind provider-side tool search. | **P**. The server tool dict survives `_offered`. Caveat: FitWindow's compaction builds new schemas (`middleware.py:2912-2973`), which drops `extras["defer_loading"]` when the request is fitted. Provider-side searches are not recorded. | partial | none | untested |
| AnthropicPromptCachingMiddleware (langchain-anthropic) | Sets cache_control. | **O** in B/D: PromptCache wraps it under its name (`middleware.py:1859-1910`). A user's copy **collides** in B (probe1). D adopts the user's settings. | n/a | test_prompt_cache | works (D) / **broken** (B) |
| ClaudeBashTool, State/Filesystem ClaudeTextEditor, ClaudeMemory, StateFileSearch (langchain-anthropic) | Anthropic tool-use middlewares. | **P**. Undeclared tools ask. Claude memory overlaps with the harness memory tools. | yes | none | untested |
| OpenAIModerationMiddleware (langchain-openai) | Moderates input and output. | **P** | n/a | none | untested |
| Bedrock prompt caching (langchain-aws) | Caching on Bedrock. | **P**. PromptCache does not apply to it. | n/a | none | untested |
| Deep Agents FilesystemMiddleware / SubAgentMiddleware / RubricMiddleware / PatchToolCalls | Deep Agents pieces on create_agent. | Covered by the Deep Agents inventory. `task` is a child run (`middleware.py:1030-1059`). | yes | test_deepagents_library | out of scope (see deepagents.md) |

## 4. Runtime, store, streaming, langchain-core

| Feature | What it does | Harness handling | Gov/Rec | Tested | Status |
|---|---|---|---|---|---|
| `Runtime` (context, store, stream_writer) in hooks and tools | Library runtime handed to hooks and tools. | **P**. The store is MemoryStore when the graph has none. | | ::test_a_middlewares_state_and_the_runs_context_reach_its_hooks_and_tools | works |
| `get_stream_writer` / `custom` mode | Custom events. | **T** into CUSTOM events named `custom` (`adapters/langgraph.py:433-439`). A subgraph's writes are lost (no `subgraphs=True`; documented). | events | ::test_what_a_node_writes_to_the_custom_stream_is_a_custom_event_of_the_run | works / caveat |
| `messages` mode (token stream) | LLM tokens. | **T**: only `AIMessage` chunks whose content is a **str** become text deltas (`:276-283`). ChatAnthropic streams list content whenever tools, documents or thinking are in the payload (`langchain_anthropic/chat_models.py:2217-2222, 2430-2442`); so does OpenAI's Responses API. Probe1: blocks=True gives `deltas=[]` and blocks=False gives `['hello th','ere world']`. **An Anthropic agent with tools streams no text** (A, B, C, and D when routed to the Anthropic SDK). Text from sub-agents and inner graphs (non-nostream) interleaves into the answer's deltas. | events | test_langgraph::test_streaming_carries_text_and_tool_events (str content only) | **broken** (list content) |
| `values` mode | State snapshots. | Read internally for the answer and interrupts, not exposed. | | n/a | works |
| `updates`, `checkpoints`, `tasks`, `debug` modes | Per-node deltas, checkpoint and task events. | Not requested; documented in "Limits". | | none | blocked-by-design |
| `version="v2"` stream and invoke | Typed StreamParts; GraphOutput. | Used. | | all | works |
| `stream_events` / `astream_events(version="v3")` (LangGraph 1.2 typed projections, langchain 1.3) | Lifecycle, subgraph and messages projections. | Never called. | | none | not reachable |
| `subgraphs=True` | Namespaced subgraph output. | Not passed. Sub-graph model text arrives through callbacks; custom writes do not. | | test_langgraph_library::test_the_text_of_a_model_inside_a_sub_graph_is_streamed | caveat |
| Notification metadata (`add_`/`get_notification_metadata`, core 1.7.2+) | Marks human messages the user did not type. | **Used**: replay of notified turns (`adapters/langgraph.py:846-852`), transcript (`:897-906`), `typed()` (`middleware.py:1937-1945`). | | test_background (get_); add_ untested | works |
| Content blocks / `content_blocks` / refusal blocks | Standard message content. | Answer text uses `message.text` (`adapters/langgraph.py:915-918`). Tool results flatten blocks (above). | | partial | caveat |
| Usage metadata, cache_read / cache_creation | Token accounting. | Accounted by the callback and on spans (`middleware.py:2019-2036`). | budget | test_middleware::test_a_chat_span_has_the_models_name_and_usage | works |
| Sync model call inside a sync node (thread executor) | Library allows it. | The accounting callback's `on_chat_model_start` raises `RuntimeError('no running event loop')` (`runtime.py:549`), which LangChain logs and swallows. The stall watchdog and the "unrouted" notice are skipped for that call; usage is still counted at `on_llm_end`. Seen in probe4. | budget partial | none | caveat |
| `langchain_core.decisions` (BaseDecisionModel, Noul, Choice; 1.5) | Typed classifiers. | Their usage goes to LangSmith only, not to the harness budget. | no | none | caveat |
| `langchain.mcp` MCPAdapter (1.4): tool annotations | MCP tools as LangChain tools; annotations under `metadata["mcp"]["tool"]["annotations"]` (`langchain/mcp/tools.py:183-209`). | `side_effects_of` reads only `metadata["annotations"]` (`middleware.py:1212-1222`). A `readOnlyHint` is ignored, so every langchain.mcp tool is UNDECLARED and asks at each call. Its content_and_artifact results lose the artifact. | yes | none | **broken** (annotations) |
| `langchain.mcp` elicitation interrupt (`type: mcp_elicitation`) | A server's question becomes a LangGraph interrupt. | Treated as the graph's own interrupt; the answer is resumed raw. | | none (the harness's own MCP elicitation is matrix F18e) | untested |

## 5. LangGraph

| Feature | What it does | Harness handling | Gov/Rec | Tested | Status |
|---|---|---|---|---|---|
| StateGraph, reducers, `add_messages` | Graph state. | **P**. Input: a string becomes `{"messages":[user]}` and a list becomes messages, with a leading context SystemMessage (`adapters/langgraph.py:210-223`). | | test_langgraph | works |
| Input of a non-message graph | A dict without `messages` passes through. A **str is always wrapped** in `{"messages": ...}`, even for an entrypoint that takes `str`. | **T**. Probe4: `@entrypoint def summarize(topic: str)` received a dict. | | none | caveat |
| Answer of a graph that calls a model but does not end with AI text | Library returns the state. | **Broken.** ERROR NO_ANSWER (`pipeline.py:898-922`). Probe4: a StateGraph whose model node writes `summary` gives `ERROR ... no answer`; an `@entrypoint` that calls a model and returns a str gives `ERROR`. | | none | **broken** |
| `Overwrite` (bypass reducer) | Replaces a reduced value. | **P**. Probe7 `['only']`. | | none (probe OK) | untested |
| `Send` (map-reduce) | Parallel branches. | **P**. Calls are governed and cancelled together. | yes | test_langgraph_library::test_send_branches_run_side_by_side_and_their_calls_are_the_runs, ::test_a_run_cancelled_inside_send_branches_stops_every_branch | works |
| `Command(update, goto)` from a node or as input | Routing and updates. | **P** | | ::test_a_nodes_command_updates_the_state_and_goes_where_it_says, ::test_a_command_as_the_runs_input_is_the_graphs_as_ainvoke_takes_it | works |
| `Command(graph=Command.PARENT)` / `destinations` | Navigation from a subgraph. | **P**. Probe7 SUCCESS. | | none (probe OK) | untested |
| `interrupt(value)` | Pauses the graph. | **T** into a contracts Interrupt; the value fields are parsed by `Question.described`. A graph without a checkpointer is refused (`adapters/langgraph.py:317-321`). | asks recorded | ::test_a_graphs_own_interrupt_is_resumed_with_the_raw_answer, plus 5 more | works |
| `interrupt(..., response_schema=)` (1.2) | A typed answer, surfaced as JSON Schema and validated on resume. | **Ignored.** The harness never reads `Interrupt.response_schema` (grep: no hit in src). Probe4: `expects= None`. A wrong answer makes the **run ERROR** ("1 validation error for Decision"), where LangGraph allows the interrupt to be answered again. | | none | **broken** |
| Several interrupts in one step | Resumed by id. | **T**: asked one after another (`:322-345`). | | ::test_two_interrupts_raised_in_one_step_are_asked_one_after_the_other | works |
| Resume `Command(resume={id: value})` | Continues the graph. | **T** (`:328-345`). An InMemorySaver resumed in another process: harness pauses replay from the journal; the graph's own pauses fail (`:523-531`). | | ::test_a_lost_checkpoint_fails_a_graphs_own_pause | works / caveat |
| Subgraphs (interrupt inside) | Nested graphs. | **P** | | ::test_a_sub_graphs_interrupt_pauses_the_run_and_its_answer_reaches_the_sub_graph | works |
| Checkpointers: InMemorySaver | In-process persistence. | **P**. Memory's window stays on (logged once). | | ::test_a_checkpointer_kept_in_ram_leaves_memorys_window_on | works |
| Checkpointers: durable (Postgres) | Durable persistence. | **P**. Logged as duplicate storage (`:534-552`). Crash continuation uses the `trellis_run_id` mark (`:603-637`). | | test_restarts (in memory) | works; untested with Postgres |
| RunCheckpointer (B without a checkpointer, D) | Harness checkpointer. | Keeps the latest checkpoint only and is thread=run (`middleware.py:3171-3300`). Probe4: history length 1. | | test_middleware::test_the_checkpoint_is_the_latest_only_and_lives_in_memory_outside_a_run | blocked-by-design (no time travel, no cross-turn graph state except CARRIED) |
| Durability modes (`sync`/`async`/`exit`) | When checkpoints are written. | **P** through framework_options. **T**: run-scoped graphs default to `sync` in workers (`:450-461`). | | test_langgraph::test_a_workers_run_of_a_run_checkpointed_graph_saves_each_step_before_the_next | works |
| DeltaChannel (beta, 1.2) | Delta-stored channels. | Supported by RunCheckpointer (`middleware.py:3300-3360`). | | ::test_a_deep_agents_delta_channels_come_back_whole_from_the_latest_checkpoint | works |
| Stores (BaseStore / `get_store` / InjectedStore) | Long-term store. | **O** (MemoryStore) or kept. | | ::test_a_graphs_own_store_is_what_its_tools_are_handed | works |
| CachePolicy / node caching (`cache=`) | Caches node results. | **P** | | ::test_a_nodes_cache_policy_answers_the_same_input_from_the_cache | works |
| RetryPolicy | Retries a node. | **P**. The harness neither retries on top of it nor counts its reruns as a stall (`middleware.py:1196-1206`). | | ::test_a_nodes_retry_policy_runs_it_again_with_the_calls_it_makes, ::test_a_node_its_retry_policy_runs_again_and_again_is_no_model_repeating_itself | works |
| TimeoutPolicy / node `timeout=` (1.2) | Per-node time limit. | **P**; the run ends saying so. | | ::test_a_node_past_its_timeout_policy_ends_the_run_saying_so | works |
| Node and graph `error_handler=` (1.2) | Recovery after retries. | **P**. Probe7 SUCCESS. The harness's 3 read retries run before the handler when there is no RetryPolicy. | | none (probe OK) | untested |
| `defer=True` nodes | Run at the end of the superstep. | **P**. Probe7. | | none (probe OK) | untested |
| Functional API `@entrypoint` / `@task` | Workflows. | **P** (string input is wrapped; see above). | | ::test_an_entrypoint_and_its_tasks_run_and_pause_as_a_graph_does | works / caveat |
| `entrypoint.final`, `previous` | Return value vs saved value. | **P** | | none | untested |
| Recursion limit | Graph step cap. | **P**; the run ends saying so. | | ::test_a_graph_past_its_recursion_limit_ends_the_run_saying_so | works |
| Time travel: `get_state_history` / `update_state` / `checkpoint_id` fork | Replay and fork. | **P** for a graph with its own checkpointer: input `None` plus `configurable.checkpoint_id`. Read with `h.conversation()`. Not possible on RunCheckpointer graphs. | | ::test_a_checkpoint_of_the_thread_is_forked_and_run_on_from | works / blocked (B, D) |
| `thread_id` / `checkpoint_ns` / configurable | Thread addressing. | **O**: a user `thread_id` is refused (`:360-365`); other configurable keys merge. | | test_langgraph | works (by design) |
| Managed values (RemainingSteps, IsLastStep) | Step awareness. | **P** | | none | untested |
| v2 output coercion to Pydantic/dataclass state | Typed output. | `extract` requires a dict state: a Pydantic output state gives answer None, which becomes NO_ANSWER if a model was called. | | none | caveat |
| `RunControl` graceful shutdown (1.2) | Stop after a superstep. | Not used; harness cancellation is task cancellation. | | none | not reachable |
| Task status (`PregelTask.error/result`, `tasks` mode) | Inspect tasks. | Not exposed. | | none | not reachable |
| RemoteGraph | A graph on LangGraph Server. | **B**: `h.wrap(RemoteGraph)` raises ConfigurationError "cannot wrap RemoteGraph" (probe4). It may be used as a node inside a local graph (untested). | | none | not reachable (by design) |
| `graph.copy` / `with_config` | Graph variants. | The harness itself copies the graph (narrowing, store, checkpointer). | | indirect | works |
| `create_react_agent` (legacy prebuilt) | Old helper. | **P**. Its tool node is governed. | yes | ::test_create_react_agents_hooks_run_and_its_tools_are_governed | works |
| Direct `graph.invoke` / `ainvoke` outside the harness | Plain library use. | Not intercepted: no memory, records or traces. The ToolNode patch is a no-op outside a run. | no | documented | blocked-by-design |

---

## 6. Overlaps (two mechanisms that may fight)

| Pair | Finding |
|---|---|
| SummarizationMiddleware vs FitWindow / ThisTurnsResults clearing | Both shrink the context. In B, the user's summarization summarises state while FitWindow(turns=True) also drops older turns from the request. The summary removes the memory context and the question; ModelHooks' `_with_task` cannot restore them for LangChain's summary (only for Deep Agents'). The transcript then holds only the answer. ReAct avoids this (`turns=False`, `react.py:396-415`). |
| ContextEditingMiddleware vs the harness's clearing | Same name, so AssertionError in B when `context_window` is given. |
| HITL vs governance | Double approval for any gated tool not declared `read`/`write` (probe5: 2 approvals). Documented. `per_call` mode is broken. |
| ToolCallLimitMiddleware vs Budget, MAX_REPEATS and StallGuard | Independent. Blocked calls are counted by none of the harness limits. No fight observed. |
| ModelCallLimitMiddleware vs `max_steps` | Same name in B, so AssertionError. D refuses with a message. |
| ToolRetryMiddleware vs harness retries | Coordinated: the harness backs off when the graph retries (tested). ToolErrorMiddleware-only graphs get the harness's 3 tries underneath. |
| ModelRetryMiddleware vs worker re-attempts | Both may retry; a retryable ModelError can be retried in the graph and then again as a new attempt. |
| TodoListMiddleware vs Planning | Same name, so AssertionError in B. Planning also drops TodoList's system-prompt section and withholds the tool until a long task. |
| PIIMiddleware vs redaction | The harness's redaction covers credentials in spans only. PII's wire redaction (stream transformer) is skipped, its state redaction misses the memory context, and the harness's records of the question are unredacted. |
| AutoModeMiddleware vs governance | Two gates in series; calls it blocks leave no record. |
| LLMToolSelector / ProviderToolSearch vs narrowing / FitWindow | Narrowing composes (it only shrinks the set). FitWindow's compaction can drop `defer_loading`. |
| AnthropicPromptCachingMiddleware vs PromptCache | Same name, so AssertionError in B. D wraps the user's. |
| ModelRouter / ModelFallback vs routing and PromptCache | Order-sensitive: placed innermost, they switch the model after PromptCache decided caching. A routed choice loses the gateway headers. |
| LLMToolEmulator vs governance | Emulated calls bypass governance and records (by design, tested). |
| Harness dedupe of identical calls vs the library | The library runs both; the harness runs one. |

## 7. Where the harness changes library behaviour (the user's rule: it must not)

1. **The ToolNode class is monkeypatched process-wide**: `_execute_tool_async` and `_arun_one` (`middleware.py:1155-1156`). It is a no-op outside a run.
2. The graph's own tool results are rebuilt as **text-only** ToolMessages:
   - the artifact is dropped and content blocks are JSON-dumped;
   - tables of 5+ rows get a note appended;
   - results over 80k characters are replaced by a stub (`middleware.py:506-534, 702-719`).
3. **Identical calls in one step run once** (`middleware.py:449-453`).
4. The **date line is appended** to every system prompt, dynamic prompts included (`middleware.py:1742-1759`).
5. The memory context is **injected as a SystemMessage and then moved into the question** at request time (`adapters/langgraph.py:215`, `middleware.py:1785-1828`).
6. The model's tools are **narrowed per call** in A and B. `read_file` and `query_result` are added in A, B and D.
7. A reply cut at the output limit gets **an extra model call** (A, B, D).
8. In B: invalid tool calls do not end the loop, and an unhandled tool error is read by the model.
9. Reads are **retried up to 3 times** where the graph does not retry them itself.
10. **STOPPED_CALL messages** are prepended in checkpointed threads.
11. `thread_id` is forced, and `trellis_run_id` checkpoint metadata and `durability="sync"` (workers, run-scoped) are added.
12. A graph without a store **gets MemoryStore**.
13. B without a checkpointer gets a **latest-only checkpointer** (no history).
14. **A successful graph is reported ERROR NO_ANSWER** when its last message is not AI text: `return_direct`, non-message state, an entrypoint returning a str.
15. A string input is **always wrapped** in `{"messages": ...}`.
16. **Streaming:** non-str chunks are dropped, transformers and v3 streams are never used, and only messages, values and custom are read.
17. `Interrupt.response_schema` is ignored, and an invalid answer fails the run.
18. **Name collisions** stop user middleware from being combined with `trellis.middleware()`.
19. Deep Agents scratch writes get a note appended (`middleware.py:375-409`).

## 8. Gaps, P0 to P3

**P0**
- **G1 PII leaks on the stream.** `PIIMiddleware(apply_to_output=True)` redacts the final state, but the harness streams raw deltas (`adapters/langgraph.py:276-283`; the v3 transformer never runs). Probe1 deltas show `ada@e` + `xample.com`. This reaches end users through AG-UI and `agent.stream`. Untested.

**P1**
- **G2 No streamed text for list-content models.** Text deltas are dropped for list-content chunks (`:278-281`): ChatAnthropic with tools or thinking, OpenAI Responses, Bedrock. An Anthropic agent with tools streams no text (probe1 `deltas=[]`). The tests only use str content.
- **G3 Tool artifacts and multimodal results are lost** (`middleware.py:702-719`, `tools/convert/__init__.py:14`). Probe1: `artifact=None` and an image dumped as JSON text. This also hits langchain.mcp tools, which return content_and_artifact.
- **G4 False NO_ANSWER errors** (`adapters/langgraph.py:296-306`, `pipeline.py:898-922`). Affected: a `return_direct` tool (probe2), a StateGraph whose model node writes a non-message key (probe4), an `@entrypoint` that calls a model and returns a str (probe4), and Pydantic output states.
- **G5 HITL `interrupt_mode="per_call"` fails** on approve, with a ValidationError (probe5). `is_hitl` only knows the batched shape (`adapters/langgraph.py:712-720`).
- **G6 Name collisions in B.** ModelCallLimitMiddleware, TodoListMiddleware, ContextEditingMiddleware (with a window) and AnthropicPromptCachingMiddleware raise AssertionError alongside `trellis.middleware()` (probe1). `middleware()` has no `stacked()`-style check or docs, unlike ReAct.

**P2**
- **G7 `interrupt(response_schema=)` is ignored**: no `expects` is surfaced, and a mismatched answer fails the run instead of re-asking (probe4).
- **G8 langchain.mcp annotations are ignored** (`metadata["mcp"]["tool"]["annotations"]` vs `metadata["annotations"]`, `middleware.py:1212-1222`), so every call of a read-only MCP tool asks a person.
- **G9 The date line and the duplicate-call dedupe change library behaviour** silently in A (probe1). The dedupe is in the matrix, but the date line is not documented on langgraph.md.
- **G10 SummarizationMiddleware drops the memory context and the question**, and the turn's transcript loses its tool calls. `_with_task` handles only Deep Agents' summaries (`middleware.py:1762-1783`).
- **G11 Large and table result rewriting reaches the graph's own tools**, but `langgraph.md` ("Large results and compute") says it applies to harness tools (probe6).
- **G12 Stream transformers and `astream_events(version="v3")` are not reachable**: the ToolCallTransformer, SubagentTransformer and middleware transformers are inert.
- **G13 No tests** for AutoModeMiddleware, ModelRouterMiddleware, ModelRefusalMiddleware, ProviderToolSearchMiddleware, OpenAIModerationMiddleware, the Anthropic tool middlewares, Bedrock caching, user ContextEditing/TodoList in B, ModelFallback order in B, and `debug`/`name`/`cache` on create_agent.
- **G14 Decision-model spend is not counted**: AutoMode and ModelRouter classifiers fall outside the budget (`langchain_core/decisions/base.py:169-184`). AutoMode-blocked calls are not recorded.

**P3**
- **G15 A string input is always wrapped** into `{"messages":...}`, including for str-typed entrypoints (probe4).
- **G16 Sync model calls in sync nodes** raise in `UsageAccounting.on_chat_model_start` (`runtime.py:549`). That call gets no stall watchdog and no unrouted notice.
- **G17 TracePolicy / `omit_payload` is not honoured by the harness's OTel `chat` spans.**
- **G18 Untested pass-through features** (probes OK): `updates`/`debug`/`tasks`/`checkpoints` modes and `subgraphs=True` (not exposed, documented); RemoteGraph is unwrappable; RunControl and task status are not reachable; `Overwrite`, `defer`, `error_handler`, `Command.PARENT` and `entrypoint.final` pass through untested.
- **G19 RunCheckpointer (B/D) keeps the latest checkpoint only**, so time travel and graph state across turns are unavailable. This is by design but should be stated beside the time-travel docs.
- **G20 Sync-only custom middleware** fails because only the async path runs.

## 9. Counts by status

Each table row is counted once, at its worst status: a row marked "works (A) / **broken** (B)" counts as broken.

| Status | Count |
|---|---|
| works | 52 |
| caveat | 22 |
| broken | 14 |
| blocked-by-design | 5 |
| untested | 21 |
| not reachable | 5 |
| out of scope (Deep Agents, covered by its own inventory) | 1 |
| **total rows** | **120** |
