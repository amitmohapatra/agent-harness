"""A2A, both ways.

* Calling: ``a2a(url)`` is a tool, a sub-agent (``ReAct(subagents=)``) and an async callable —
  any code, any framework, awaits it with a message and gets the remote agent's answer, each
  call a :class:`RemoteAgent` (``client``).
* Serving: ``agent.serve_a2a(app, url)`` publishes a wrapped agent (``server``): the card,
  JSON-RPC, task = run, a pause = ``input-required``, signed push notifications (``push``).
"""

from __future__ import annotations

from trellis.harness.a2a.client import InputRequired, RemoteAgent

__all__ = ["InputRequired", "RemoteAgent"]
