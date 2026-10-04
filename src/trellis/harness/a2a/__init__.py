"""A2A, both ways.

* Calling: ``remote(url, tenant=, user=)`` is a :class:`RemoteAgent` — any code, any framework,
  awaits it with a message and gets the remote agent's answer (``client``). The harness's
  ``a2a(url)`` tool is built on it.
* Serving: ``agent.serve_a2a(app, url)`` publishes a wrapped agent (``server``): the card,
  JSON-RPC, task = run, a pause = ``input-required``, signed push notifications (``push``).
"""

from __future__ import annotations

from trellis.harness.a2a.client import InputRequired, RemoteAgent, remote

__all__ = ["InputRequired", "RemoteAgent", "remote"]
