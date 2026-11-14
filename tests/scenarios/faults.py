"""Faults a scenario's harness process (``tests.scenarios.serve``) takes at its own boundary,
each named in the process's spec (``faults``) and switched from the test by a flag file, so a
test says when the fault begins and ends:

* ``partition``: while the flag file exists, no request of this process reaches agent-runs (a
  connection error, as a cut network gives);
* ``runs_full``: while the flag file exists, agent-runs answers this process's writes of run data
  (events, payloads, artifacts) ``507 Insufficient Storage``, as a store whose disk is full;
* ``skew``: this process's clock runs that many seconds ahead of the machine's (the time the
  harness reads: ``time.time`` and ``datetime.now`` in its modules).
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

#: the agent-runs writes a full disk refuses (the run's data; its state changes still pass)
FULL_PATHS = ("/events", "/payloads", "/artifacts")


def install(faults: dict[str, Any]) -> None:
    """The faults of a process's spec, installed before its harness is made."""
    if "skew" in faults:
        _skewed(float(faults["skew"]))
    partition = Path(faults["partition"]) if "partition" in faults else None
    full = Path(faults["runs_full"]) if "runs_full" in faults else None
    if partition is None and full is None:
        return
    runs = urlsplit(os.environ["RUNS_URL"]).netloc
    send = httpx.AsyncClient.send

    async def faulty(
        self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        if request.url.netloc.decode() == runs:
            if partition is not None and partition.exists():
                raise httpx.ConnectError("the network between this server and agent-runs is cut")
            if (
                full is not None
                and full.exists()
                and request.method == "POST"
                and request.url.path.endswith(FULL_PATHS)
            ):
                problem = {
                    "title": "Insufficient Storage",
                    "status": 507,
                    "code": "DEPENDENCY_UNAVAILABLE",
                    "detail": "no space left on device",
                    "retryable": False,
                }
                return httpx.Response(
                    507,
                    json=problem,
                    request=request,
                    headers={"content-type": "application/problem+json"},
                )
        return await send(self, request, **kwargs)

    httpx.AsyncClient.send = faulty  # type: ignore[method-assign]


def _skewed(seconds: float) -> None:
    """This process's clock ``seconds`` ahead, where the harness and the SDKs read it."""
    real = time.time
    time.time = lambda: real() + seconds  # type: ignore[assignment]

    class Skewed(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Any:
            return datetime.now(tz) + timedelta(seconds=seconds)

    for name, module in list(sys.modules.items()):
        if name.startswith("trellis.") and getattr(module, "datetime", None) is datetime:
            module.datetime = Skewed  # type: ignore[attr-defined]
