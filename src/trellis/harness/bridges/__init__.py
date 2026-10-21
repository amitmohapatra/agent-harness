"""Native slots, the platform's services behind them.

Each framework keeps its own feature, and the harness puts the memory service and the run's
journal behind the framework's own extension interface — never a client of its own, never a
fallback chain:

* LangGraph and Deep Agents: a ``BaseStore`` (:mod:`.store`), the graph's long-term store,
  backed by the memory service — given to a graph that has no store of its own;
* the OpenAI Agents SDK: a ``Session`` (:mod:`.session`), the agent's conversation, kept in
  the memory service's thread — used when the run is given no session of its own;
* the Claude Agent SDK: a ``UserPromptSubmit`` hook (:mod:`.claude`) that hands Claude the
  memory context with the prompt, and a ``SessionStore`` that keeps Claude's session in the
  run's journal, so a resume continues the session on any machine.

A bridge is on when what it stands for is: memory (``MEMORY_URL``, and not ``without=``); the
journal always. A framework given its own store, session or session store keeps it (said once,
in the log). A service that does not answer fails the way the harness's own calls to it fail:
the store's and the session's calls are the framework's, so their error is the framework's
error; the context hook is the memory push, already a warning when the service is down.
"""

from __future__ import annotations

import logging

log = logging.getLogger("trellis.harness")

#: What was said once already (``said``), by key.
_said: set[str] = set()


def said(key: str, message: str, *args: object, extra: dict[str, object]) -> bool:
    """Log ``message`` once per process for ``key`` (``True`` the first time), with ``extra``
    as its structured fields."""
    if key in _said:
        return False
    _said.add(key)
    log.warning(message, *args, extra={"bridge": key.split(":", 1)[0], **extra})
    return True
