"""Native slots, the platform's services behind them.

Each framework keeps its own feature, and the harness puts the memory service and the run's
journal behind the framework's own extension interface — never a client of its own, never a
fallback chain:

* LangGraph and Deep Agents: a ``BaseStore`` (:mod:`.store`), the graph's long-term store,
  backed by the memory service — given to a graph compiled without a store of its own;
* the OpenAI Agents SDK: a ``Session`` (:mod:`.session`), the agent's conversation read from
  the memory service's thread — used when the run is given no session of its own;
* the Claude Agent SDK: a ``UserPromptSubmit`` hook (:mod:`.claude`) that hands Claude the
  memory context with the prompt, and a ``SessionStore`` that keeps Claude's session in the
  run's journal, so a resume continues the session on any machine;
* every framework: the helpers it spawns itself (:mod:`.helpers`) are child runs.

A bridge is on while memory is on for the run (``MEMORY_URL``, and not ``without={"memory"}``);
the journal's always. A framework given its own store or session keeps it (said once). Every
turn is recorded to the memory service by the harness, whatever the framework reads back: it is
the source of truth.
"""
