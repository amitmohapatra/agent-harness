# 0005. A conversation across runs: history where nothing keeps it, busy-conversation modes, one id per message, the framework's session continued

**Status:** Accepted (2026-10-30). Amends [ADR 0003](0003-automatic-features.md) point 6. The
ownership it sets stays as it is; this ADR fills the case where no owner exists.

## Context

ADR 0003 point 6 says who keeps a conversation's history across runs. When the target keeps a
thread of its own (a graph with its own checkpointer, an OpenAI Agents `session`), the
framework keeps it. Otherwise the memory service's window keeps it. `ReAct` checkpoints per run
(`thread_id` is the run id) on purpose. Each conversation is scoped by its tenant, its user and
its thread (`identity.conversation_key`). Reading the code at `main` showed four gaps:

* **Memory off.** In that case nobody owns the history. `Agent.window` is never asked, and
  `runtime.history` stays `None`. A follow-up to `ReAct`, to a graph without a checkpointer, to
  an OpenAI agent without a `session`, to a function, to Claude or to an ACP agent reached the
  model without the earlier turns. The run store already holds every turn's question and answer
  (`RunRecord.input`, `.output`), so the data exists.
* **A busy conversation can only queue.** `Turns` (the in-process lock per conversation) and
  `start()`'s `concurrency_key` make a second message wait. Nothing lets that message cancel
  the working run (the run can be cancelled, through `agent.cancel`). Nothing lets the message
  join the working run either, even though LangChain middleware (`before_model`), OpenAI
  Agents (`RunConfig.call_model_input_filter`) and Claude (`PostToolUse` hooks) all have a
  hook between steps.
* **Message ids differ by place.** The AG-UI text id is `{run}.{attempt}.m`, so it changes on
  every attempt. A tool result's id is `result_<call>`. The memory service stores messages
  under `source_message_id` values that depend on position and attempt, and holds no id the
  stream also uses.
* **Claude and ACP sessions end with the run.** The session lives in the run's journal
  (`Journal.session`, `.transcripts`: `bridges.claude.JournalSessions`, `acp.py`'s
  `session/resume` and `session/load`). It is resumed only within the run, after a pause. The
  conversation's next run starts a new session.

## Decision

1. **The owner stays the same, and the gap gets a reader, not a store.** With memory on,
   nothing changes. With memory off, and only where the framework keeps no thread of its own
   (`thread_store` is `None`, or is the memory thread the run no longer has) and no session
   continues (point 4), the run's history is the conversation's earlier turns from the run
   store (`conversation.earlier`). Those turns are each run's question and answer, from the
   same agent, tenant, user and thread: at most 10 turns, oldest first. A cancelled turn
   contributes its question alone. A target that takes messages gets them as its own: LangGraph
   messages, OpenAI Agents input items, and `agent.history` for a function. Claude and an ACP
   agent get them in the context's text. No new store is added, and none of these is a
   fallback chain: each target has exactly one source.
2. **`mode=` on `run` and `stream`** decides what a message to a conversation that is busy in
   this process does:
   * `queue` (the default) waits, exactly as before.
   * `interrupt` cancels the working run with `agent.cancel` (reason `INTERRUPTED`). The new
     run then takes the conversation's turn and continues from it.
   * `steer` gives the message to the working run at its framework's next step, and that run's
     answer becomes the new run's answer. The new run's record ends `SUCCESS` with that answer.
     Its stream carries the answer under that answer's id. The steer hooks are LangChain
     `before_model` on the harness's `HarnessTools` (so `ReAct` and graphs built with
     `middleware()`), OpenAI Agents `call_model_input_filter`, and the Claude Agent SDK's
     `PostToolUse` `additionalContext`. A function, an ACP agent, and a graph without the
     harness's middleware declare `Adapter.steers(target) == False`. On those, `steer` waits as
     `queue` does, and a warning says so once per agent. A message the working run never took
     (it ended first, or ended without an answer) runs as its own turn.

   `start()` queues as before, through its `concurrency_key`. A worker's run is not held in
   process, so `mode` has nothing to act on there.
3. **One id per message**, `msg_` followed by a digest of the conversation and the message's
   native name: `<run>:input` for the question, `<run>:answer` for the answer,
   `result:<call>` for a tool result, `call:<first call>` for an assistant message that calls
   tools, and otherwise the run, attempt and position. The same id appears on the AG-UI
   stream (the text message, a `TOOL_CALL_RESULT`'s `messageId`) and in the memory service
   (the message's `custom_metadata`: `message_id`, `run_id`, `tool_call_id`, one level deep,
   within the service's limit of two). Because the name does not depend on the attempt or on
   the content, the id survives a resume, a replay and a cleared tool result.
   The memory service de-duplicates by this id too: it is the message's `source_message_id`,
   so a resumed attempt that sends the first attempt's messages again (a checkpointed graph's
   whole state) has each stored once (`test_a_resumed_graph_run_stores_each_message_of_its_transcript_once`).
4. **A conversation continues the framework's own session.** When a turn of a Claude Agent
   SDK or ACP target ends (and does not pause), its session (the journal's `session` and, for
   Claude, its `transcripts`) is kept per conversation in this process (`Agent.sessions`,
   bounded at 128 conversations: a session's transcripts can be large). The conversation's next turn starts with it in its journal, through the
   same path a resume after a pause takes, but it sends its own prompt rather than the "goes on"
   prompt. That session then owns the history: point 1 reads nothing, and with memory on, memory's
   window is not rendered beside it (`Agent.window` returns `Window.NONE`). When the session cannot
   be resumed, the existing warning fires and a new session starts.

## Consequences

* With memory off, a follow-up now has its conversation on every adapter. With memory on,
  a continued session's turn no longer gets memory's window of the same turns beside it
  (`test_a_continued_session_with_memory_on_gets_no_memory_window_beside_it`); every other
  run receives what it did.
* The kept session is in-process, as `Turns` is. A turn that a different worker or process
  runs starts a new session with the run store's turns (memory off) or memory's window
  (memory on). The durable owner remains the framework's own durable store.
* `steer` adds one graph node, `HarnessTools.before_model`, which counts as one LangGraph step
  per model call. It does nothing when nothing was steered.
* Tests: `tests/integration/test_conversation.py` covers every adapter × mode, follow-up
  continuity with memory on and off, the session continued, id stability, and steering; the
  scoping tests are in `test_conversation_scope.py`.
