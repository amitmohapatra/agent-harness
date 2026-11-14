"""The test plan's invariants (``docs/testing/TEST_PLAN.md``, section 1: INV-1..INV-10), checked at
the teardown of every test that uses the :func:`invariants` fixture — against the real memory
service and agent-runs, from what the harness actually sent them and what they keep. A test that
passes its own assertions but breaks an invariant fails.

Use it from a suite's ``conftest.py``::

    from tests.support.invariants import invariants  # noqa: F401  (an autouse fixture)

Every exchange the test process has with the two services (``MEMORY_URL``, ``RUNS_URL``) is
recorded as it happens (:class:`Traffic`, by wrapping ``httpx.AsyncClient.send``); a process the
test starts records its own when ``SCEN_TRAFFIC`` names a file (:func:`record_in_process`), and the
fixture reads it back. At teardown:

* **INV-1** every run the test touched has ended, or waits on a person the test declared
  (``inv.waits(run_id)``): none ``RUNNING``/``QUEUED`` once the holds and leases of the test's
  processes had time to lapse (:attr:`Invariants.settle` seconds), none ``PAUSED`` undeclared.
* **INV-2** every user turn is in the memory thread once (no message id twice; every successful
  run's question there); every tool call that ran (``TOOL_CALL_RESULT``) has its tool record
  (``POST /v1/tools/invocations``) and no record is of a call that never ran.
* **INV-3** one question per call: no call asked about again after it was approved.
* **INV-4** an approved call ran once, an edited one once with the edited arguments, a rejected
  one never.
* **INV-5** nothing crosses a user or tenant boundary: no memory answer for one user's scope
  carries another user's record, no answer of either service another tenant's.
* **INV-6** what a run spent never goes down across its attempts, and its tool calls are the
  calls its events show.
* **INV-7** every request and answer validates against the services' committed OpenAPI documents
  (``TRELLIS_MEMORY_OPENAPI`` / ``TRELLIS_RUNS_OPENAPI``).
* **INV-8** no key (the environment's, or any credential header the harness sent) appears in a
  log record, a request body (events, memory records), an answer, or a log file of the test's.
* **INV-9** no process the test started outlives it, and nothing new is left in the temporary
  directory outside the test's own folder.
* **INV-10** each run's events are well formed: per attempt, ``RUN_STARTED`` first, numbered in
  order, one end; one ``TOOL_CALL_START`` per tool call across the run, each with its end.

A check that cannot apply (no memory, a run of another tenant's key) is left out, not failed.
``inv.allow("INV-n", why)`` waives one invariant for a test that breaks it on purpose (a fault
injected), with its reason — never to make a failure pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from tests.support.openapi import MEMORY_OPENAPI, RUNS_OPENAPI, OpenAPI

#: run statuses that need nobody
ENDED: Final = frozenset({"SUCCESS", "ERROR", "TIMEOUT", "CANCELLED", "PARTIAL", "REJECTED"})
#: environment variables holding keys (INV-8)
SECRET_VARIABLES: Final = (
    "TRELLIS_API_KEY",
    "TRELLIS_ADMIN_KEY",
    "TRELLIS_API_KEY_OTHER",
    "LLM_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)
CREDENTIAL_HEADERS: Final = ("authorization", "x-api-key", "x-trellis-api-key")
#: the memory service's own agent tools: the service logs their calls itself, the harness
#: records none (docs/memory.md, "What is recorded")
SERVICE_LOGGED: Final = frozenset(
    {"memory_search", "memory_remember", "memory_update", "memory_forget", "profile_edit",
     "tool_search"}
)


def _netloc(url: str | None) -> str | None:
    return urlsplit(url).netloc if url else None


@dataclass
class Exchange:
    service: str
    method: str
    path: str
    query: dict[str, list[str]]
    request: Any
    status: int | None
    response: Any
    #: the credential the request carried (never printed: used to read back as the caller)
    auth: dict[str, str] = field(default_factory=dict, repr=False)
    raw_request: str = field(default="", repr=False)
    raw_response: str = field(default="", repr=False)


def _json(content: bytes) -> Any:
    try:
        return json.loads(content) if content else None
    except (ValueError, UnicodeDecodeError):
        return None


class Traffic:
    """Every exchange with the memory service and agent-runs, kept and checked against their
    OpenAPI documents as it happens."""

    def __init__(self) -> None:
        self.memory_api = OpenAPI.load(MEMORY_OPENAPI) if MEMORY_OPENAPI.exists() else None
        self.runs_api = (
            OpenAPI.load(RUNS_OPENAPI, strict_query=True) if RUNS_OPENAPI.exists() else None
        )
        self.hosts = {
            n: s
            for n, s in (
                (_netloc(os.environ.get("MEMORY_URL")), "memory"),
                (_netloc(os.environ.get("RUNS_URL")), "runs"),
            )
            if n
        }
        self.exchanges: list[Exchange] = []
        self.violations: list[str] = []

    def service(self, request: httpx.Request) -> str | None:
        netloc = request.url.netloc
        return self.hosts.get(netloc.decode() if isinstance(netloc, bytes) else netloc)

    def check(self, request: httpx.Request, response: httpx.Response | None) -> None:
        service = self.service(request)
        if service is None or request.url.path.startswith("/health"):
            return
        api = self.memory_api if service == "memory" else self.runs_api
        read = response is not None and _read(response)
        body = request.content if isinstance(request.content, bytes) else b""
        media = request.headers.get("content-type", "")
        self.exchanges.append(
            Exchange(
                service=service,
                method=request.method,
                path=request.url.path,
                query=parse_qs(
                    request.url.query.decode()
                    if isinstance(request.url.query, bytes)
                    else request.url.query
                ),
                request=_json(body) if "json" in media else None,
                status=response.status_code if response is not None else None,
                response=_json(response.content) if read and response is not None else None,
                auth={k: v for k, v in request.headers.items() if k.lower() in CREDENTIAL_HEADERS},
                raw_request=body.decode(errors="replace") if "json" in media else "",
                raw_response=response.content.decode(errors="replace")
                if read and response is not None
                else "",
            )
        )
        if api is None:
            return
        self.violations += [f"INV-7 {service}: {v}" for v in api.request(request)]
        if response is not None and read:
            self.violations += [f"INV-7 {service}: {v}" for v in api.response(request, response)]


def _read(response: httpx.Response) -> bool:
    try:
        response.content  # noqa: B018
    except httpx.ResponseNotRead:
        return False
    return True


def recording(traffic: Traffic, sink: Path | None = None) -> Any:
    """``httpx.AsyncClient.send`` recording into ``traffic`` (and, given ``sink``, appending each
    exchange as a JSON line there: a process the test started)."""
    send = httpx.AsyncClient.send

    async def recorded(
        self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        response = await send(self, request, **kwargs)
        before = len(traffic.exchanges), len(traffic.violations)
        try:
            traffic.check(request, response)
        except Exception as exc:
            traffic.violations.append(
                f"INV-7 checking {request.method} {request.url.path}: {exc!r}"
            )
        if sink is not None:
            with sink.open("a") as out:
                for e in traffic.exchanges[before[0] :]:
                    out.write(json.dumps({"exchange": e.__dict__}, default=str) + "\n")
                for v in traffic.violations[before[1] :]:
                    out.write(json.dumps({"violation": v}) + "\n")
        return response

    return recorded


def record_in_process() -> None:
    """In a process a test started: its exchanges recorded into ``SCEN_TRAFFIC`` for the test's
    invariants to read."""
    sink = os.environ.get("SCEN_TRAFFIC")
    if sink:
        httpx.AsyncClient.send = recording(Traffic(), Path(sink))  # type: ignore[method-assign]


# --------------------------------------------------------------------------- processes, files
def _children(root: int) -> dict[int, str]:
    """Every live descendant process of ``root``: pid -> command line."""
    parents: dict[int, int] = {}
    commands: dict[int, str] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            state = stat.rsplit(")", 1)[1].split()[0]
            if state == "Z":
                continue
            parents[int(entry.name)] = int(stat.rsplit(")", 1)[1].split()[1])
            commands[int(entry.name)] = (
                (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:200]
            )
        except (OSError, IndexError, ValueError):
            continue
    found: dict[int, str] = {}
    frontier = [root]
    while frontier:
        pid = frontier.pop()
        for child, parent in parents.items():
            if parent == pid and child not in found:
                found[child] = commands.get(child, "")
                frontier.append(child)
    return found


# --------------------------------------------------------------------------- the checks
class Invariants:
    """What a test may say to its invariants: a run that waits on a person on purpose
    (:meth:`waits`), an invariant waived with its reason (:meth:`allow`), a file a process of the
    test's records its traffic in (:meth:`traffic_file`)."""

    #: how long the runs a test touched get to end once it is done (a killed server's holds lapse)
    settle: float = 45.0

    def __init__(self, tmp: Path) -> None:
        self.traffic = Traffic()
        self.tmp = tmp
        self.waiting: set[str] = set()
        self.waived: dict[str, str] = {}
        self.files: list[Path] = []
        self.secrets = {v for n in SECRET_VARIABLES if (v := os.environ.get(n, "").strip())}

    def waits(self, run_id: str) -> None:
        """``run_id`` is left waiting for a person on purpose (INV-1)."""
        self.waiting.add(run_id)

    def allow(self, invariant: str, why: str) -> None:
        """``invariant`` is broken on purpose by this test (a fault injected), for ``why``."""
        self.waived[invariant] = why

    def traffic_file(self, name: str) -> Path:
        """A file a process the test starts records its exchanges in (``SCEN_TRAFFIC``)."""
        path = self.tmp / f"{name}.traffic"
        self.files.append(path)
        return path

    # -- reading back -------------------------------------------------------------------
    def _load_files(self) -> None:
        for path in self.files:
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                item = json.loads(line)
                if "violation" in item:
                    self.traffic.violations.append(item["violation"])
                else:
                    self.traffic.exchanges.append(Exchange(**item["exchange"]))

    def runs(self) -> dict[str, Exchange]:
        """Every run the test's processes made or claimed, with the exchange that made it."""
        found: dict[str, Exchange] = {}
        for e in self.traffic.exchanges:
            if e.service != "runs" or not isinstance(e.response, dict) or (e.status or 500) >= 300:
                continue
            if (e.method, e.path) in (("POST", "/v1/runs"), ("POST", "/v1/runs/claim")):
                record = (
                    e.response.get("record", e.response) if isinstance(e.response, dict) else {}
                )
                run_id = record.get("run_id") if isinstance(record, dict) else None
                if run_id:
                    found.setdefault(run_id, e)
        return found

    async def _get(
        self, client: httpx.AsyncClient, base: str, path: str, e: Exchange, **params: Any
    ) -> Any:
        got = await client.get(base + path, headers=e.auth, params=params or None)
        return got.json() if got.status_code == 200 else None

    async def _events(
        self, client: httpx.AsyncClient, base: str, run_id: str, e: Exchange
    ) -> list[dict[str, Any]]:
        """A run's whole event log (500 a page)."""
        out: list[dict[str, Any]] = []
        after = 0
        while True:
            page = await self._get(
                client, base, f"/v1/runs/{run_id}/events", e, after=after, limit=500
            )
            if not page:
                return out
            out += page
            after = max(int(p.get("position", 0)) for p in page)
            if len(page) < 500:
                return out

    async def check(self) -> list[str]:
        self._load_files()
        out = list(dict.fromkeys(self.traffic.violations))
        runs = self.runs()
        runs_url = os.environ.get("RUNS_URL", "").rstrip("/")
        memory_url = os.environ.get("MEMORY_URL", "").rstrip("/")
        async with httpx.AsyncClient(timeout=30.0) as client:
            records = await self._settled(client, runs_url, runs) if runs_url else {}
            events = {r: await self._events(client, runs_url, r, runs[r]) for r in records}
            out += self._inv1(records)
            out += self._inv2_tools(events)
            out += await self._inv2_turns(client, memory_url, records)
            out += self._inv3_inv4(records, events)
            out += self._inv5()
            out += self._inv6(events)
            out += self._inv8()
            out += self._inv10(events)
        return [v for v in out if v.split(" ", 1)[0] not in self.waived]

    async def _settled(
        self, client: httpx.AsyncClient, base: str, runs: dict[str, Exchange]
    ) -> dict[str, dict[str, Any]]:
        deadline = time.monotonic() + self.settle
        while True:
            records = {}
            for run_id, e in runs.items():
                got = await self._get(client, base, f"/v1/runs/{run_id}", e)
                if got is not None:
                    records[run_id] = got
            busy = [r for r, rec in records.items() if rec.get("status") in ("RUNNING", "QUEUED")]
            if not busy or time.monotonic() >= deadline:
                return records
            await asyncio.sleep(1.0)

    def _inv1(self, records: dict[str, dict[str, Any]]) -> list[str]:
        out = []
        for run_id, rec in records.items():
            status = rec.get("status")
            if status in ("RUNNING", "QUEUED"):
                out.append(
                    f"INV-1 run {run_id} ({rec.get('agent_id')}) still {status} {self.settle:.0f} s after the test"
                )
            elif status == "PAUSED" and run_id not in self.waiting and not rec.get("parent_run_id"):
                out.append(
                    f"INV-1 run {run_id} ({rec.get('agent_id')}) left PAUSED with nobody to answer it"
                )
        return out

    def _inv2_tools(self, events: dict[str, list[dict[str, Any]]]) -> list[str]:
        recorded: dict[str, Counter[str]] = defaultdict(Counter)
        for e in self.traffic.exchanges:
            if (
                (e.method, e.path) == ("POST", "/v1/tools/invocations")
                and (e.status or 500) < 300
                and isinstance(e.request, dict)
            ):
                run = (e.request.get("scope") or {}).get("agent_run_id")
                if run:
                    recorded[run][_call_key(e.request.get("tool"), e.request.get("args"))] += 1
        out = []
        for run_id, evs in events.items():
            ran: Counter[str] = Counter()
            args: dict[str, Any] = {}
            for p in evs:
                ev = p.get("event", p)
                if ev.get("type") == "TOOL_CALL_ARGS":
                    args[ev.get("tool_call_id")] = (ev.get("data") or {}).get("args")
                if ev.get("type") == "TOOL_CALL_RESULT":
                    tool = (ev.get("data") or {}).get("tool")
                    ran[_call_key(tool, args.get(ev.get("tool_call_id")))] += 1
            if run_id not in recorded and not any(
                e.service == "memory" for e in self.traffic.exchanges
            ):
                continue  # memory off: nothing to record into
            got = recorded.get(run_id, Counter())
            missing = [k for k in ran if k not in got and k.split(":", 1)[0] not in SERVICE_LOGGED]
            ghost = [k for k in got if k not in ran]
            if missing:
                out.append(f"INV-2 run {run_id}: calls that ran have no tool record: {missing[:3]}")
            if ghost:
                out.append(
                    f"INV-2 run {run_id}: tool records of calls its events never show: {ghost[:3]}"
                )
        return out

    async def _inv2_turns(
        self, client: httpx.AsyncClient, base: str, records: dict[str, dict[str, Any]]
    ) -> list[str]:
        if not base:
            return []
        from trellis.memory import MemoryClient

        threads: dict[str, tuple[dict[str, Any], str]] = {}
        asked: dict[str, set[str]] = defaultdict(set)
        for e in self.traffic.exchanges:
            if (
                (e.method, e.path) == ("POST", "/v1/messages")
                and (e.status or 500) < 300
                and isinstance(e.request, dict)
            ):
                scope = e.request.get("scope") or {}
                thread, run = scope.get("thread_id"), scope.get("agent_run_id")
                bearer = next((v for k, v in e.auth.items() if k.lower() == "authorization"), "")
                if thread and bearer:
                    keep = {
                        k: scope[k]
                        for k in ("tenant_id", "user_id", "agent_id", "thread_id")
                        if scope.get(k)
                    }
                    threads.setdefault(thread, (keep, bearer.split(" ", 1)[-1]))
                    if run:
                        asked[thread].add(run)
        out = []
        for thread, (scope, key) in threads.items():
            async with MemoryClient(base, api_key=key, timeout=30.0) as memory:
                try:
                    messages = await memory.bind(**scope).history(limit=500)
                except Exception as exc:
                    out.append(f"INV-2 thread {thread}: its history could not be read: {exc!r}")
                    continue
            ids = Counter(
                (m.model_extra or {}).get("source_message_id")
                for m in messages
                if (m.model_extra or {}).get("source_message_id")
            )
            twice = [i for i, n in ids.items() if n > 1]
            if twice:
                out.append(f"INV-2 thread {thread}: messages kept more than once: {twice[:3]}")
            users = [m.content for m in messages if str(m.role).upper().endswith("USER")]
            for run_id in asked[thread]:
                rec = records.get(run_id)
                if rec is None or rec.get("status") != "SUCCESS" or rec.get("parent_run_id"):
                    continue
                question = rec.get("input")
                if (
                    isinstance(question, str)
                    and question.strip()
                    and not any(question.strip() in u for u in users)
                ):
                    out.append(
                        f"INV-2 thread {thread}: run {run_id}'s question is not in its memory thread"
                    )
        return out

    def _inv3_inv4(  # noqa: PLR0912 - one pass over the decisions, each checked where it is
        self, records: dict[str, dict[str, Any]], events: dict[str, list[dict[str, Any]]]
    ) -> list[str]:
        asked: dict[str, list[dict[str, Any]]] = defaultdict(list)
        decided: dict[str, dict[str, Any]] = {}
        for e in self.traffic.exchanges:
            if (
                e.service != "runs"
                or e.method != "POST"
                or (e.status or 500) >= 300
                or not isinstance(e.request, dict)
            ):
                continue
            parts = e.path.split("/")
            if len(parts) == 5 and parts[4] == "pause":
                interrupt = e.request.get("interrupt") or {}
                asked[parts[3]].append(interrupt)
            if len(parts) == 5 and parts[4] == "resume":
                decided[e.request.get("interrupt_id", "")] = e.request
        out = []
        for run_id, interrupts in asked.items():
            approved: set[str] = set()
            for interrupt in interrupts:
                call = interrupt.get("tool_call") or {}
                key = _call_key(call.get("tool"), call.get("args")) if call else None
                if key and key in approved:
                    out.append(
                        f"INV-3 run {run_id}: asked again about {call.get('tool')} after it was approved"
                    )
                decision = (decided.get(interrupt.get("interrupt_id", "")) or {}).get("decision")
                if key and decision in ("APPROVE", "EDIT"):
                    approved.add(key)
                if not call or decision not in ("APPROVE", "EDIT", "REJECT"):
                    continue
                if records.get(run_id, {}).get("status") != "SUCCESS":
                    continue
                starts = _starts(
                    events.get(run_id, []), call.get("tool"), call.get("idempotency_key")
                )
                if decision == "REJECT" and starts:
                    out.append(f"INV-4 run {run_id}: {call.get('tool')} ran though it was rejected")
                if decision in ("APPROVE", "EDIT") and len(starts) != 1:
                    out.append(
                        f"INV-4 run {run_id}: approved {call.get('tool')} ran {len(starts)} times"
                    )
                payload = (decided.get(interrupt.get("interrupt_id", "")) or {}).get("payload")
                if (
                    decision == "EDIT"
                    and len(starts) == 1
                    and isinstance(payload, dict)
                    and starts[0] is not None
                    and starts[0] != payload
                ):
                    out.append(
                        f"INV-4 run {run_id}: edited {call.get('tool')} ran with {starts[0]}, not {payload}"
                    )
        return out

    def _inv5(self) -> list[str]:
        out = []
        tenants = {
            (e.request.get("scope") or {}).get("tenant_id")
            for e in self.traffic.exchanges
            if e.service == "memory" and isinstance(e.request, dict)
        } - {None}
        for e in self.traffic.exchanges:
            if (e.status or 500) >= 300 or e.response is None:
                continue
            scope = e.request.get("scope") if isinstance(e.request, dict) else None
            user = (scope or {}).get("user_id") or (e.query.get("user_id") or [None])[0]
            tenant = (scope or {}).get("tenant_id")
            for key, value in _fields(e.response, ("user_id", "tenant_id")):
                if key == "user_id" and user and value and value != user and e.service == "memory":
                    out.append(
                        f"INV-5 memory {e.method} {e.path} for {user} answered a record of {value}"
                    )
                if key == "tenant_id" and value and tenant and value != tenant:
                    out.append(
                        f"INV-5 {e.service} {e.method} {e.path} for tenant {tenant} answered tenant {value}"
                    )
                if (
                    key == "tenant_id"
                    and value
                    and not tenant
                    and len(tenants) == 1
                    and value not in tenants
                    and e.service == "memory"
                ):
                    out.append(f"INV-5 memory {e.method} {e.path} answered tenant {value}")
        return list(dict.fromkeys(out))

    def _inv6(self, events: dict[str, list[dict[str, Any]]]) -> list[str]:
        out = []
        for run_id, evs in events.items():
            spent_seen: list[dict[str, Any]] = []
            starts: set[str] = set()
            for p in evs:
                ev = p.get("event", p)
                if ev.get("type") == "TOOL_CALL_START" and ev.get("tool_call_id"):
                    starts.add(ev["tool_call_id"])
                if ev.get("type") in ("RUN_FINISHED", "RUN_ERROR"):
                    spent = (ev.get("data") or {}).get("spent")
                    if isinstance(spent, dict):
                        spent_seen.append(spent)
            for a, b in itertools.pairwise(spent_seen):
                for k in ("tokens", "cost", "model_calls", "tool_calls"):
                    if float(b.get(k) or 0) + 1e-9 < float(a.get(k) or 0):
                        out.append(
                            f"INV-6 run {run_id}: spent {k} went down across attempts ({a.get(k)} -> {b.get(k)})"
                        )
        return out

    def _inv8(self) -> list[str]:
        secrets = set(self.secrets)
        for e in self.traffic.exchanges:
            for value in e.auth.values():
                token = value.split(" ", 1)[-1].strip()
                if len(token) >= 12:
                    secrets.add(token)
        if not secrets:
            return []
        out = []
        for e in self.traffic.exchanges:
            for what, text in (("request", e.raw_request), ("answer", e.raw_response)):
                if any(s in text for s in secrets):
                    out.append(f"INV-8 a key in the {what} of {e.service} {e.method} {e.path}")
        for path in self.tmp.rglob("*.log"):
            with contextlib.suppress(OSError):
                if any(s in path.read_text(errors="replace") for s in secrets):
                    out.append(f"INV-8 a key in the log file {path.name}")
        self._secrets = secrets
        return list(dict.fromkeys(out))

    def secrets_in(self, text: str) -> bool:
        return any(s in text for s in getattr(self, "_secrets", self.secrets))

    def _inv10(self, events: dict[str, list[dict[str, Any]]]) -> list[str]:
        out = []
        for run_id, evs in events.items():
            by_attempt: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for p in evs:
                ev = p.get("event", p)
                by_attempt[int(ev.get("attempt") or 1)].append(ev)
            starts: Counter[str] = Counter()
            ends: Counter[str] = Counter()
            for attempt, evs_a in sorted(by_attempt.items()):
                if evs_a[0].get("type") != "RUN_STARTED":
                    out.append(
                        f"INV-10 run {run_id} attempt {attempt}: begins with {evs_a[0].get('type')}"
                    )
                numbers = [int(ev.get("sequence") or 0) for ev in evs_a]
                if numbers != sorted(numbers) or len(set(numbers)) != len(numbers):
                    out.append(
                        f"INV-10 run {run_id} attempt {attempt}: events out of order {numbers[:12]}"
                    )
                finished = [ev for ev in evs_a if ev.get("type") in ("RUN_FINISHED", "RUN_ERROR")]
                if len(finished) > 1:
                    out.append(f"INV-10 run {run_id} attempt {attempt}: {len(finished)} endings")
                if finished and finished[-1] is not evs_a[-1]:
                    out.append(f"INV-10 run {run_id} attempt {attempt}: events after its end")
                for ev in evs_a:
                    if ev.get("type") == "TOOL_CALL_START":
                        starts[str(ev.get("tool_call_id"))] += 1
                    if ev.get("type") == "TOOL_CALL_END":
                        ends[str(ev.get("tool_call_id"))] += 1
            twice = [c for c, n in starts.items() if n > 1]
            if twice:
                out.append(f"INV-10 run {run_id}: tool calls started more than once: {twice[:3]}")
        return out


def _call_key(tool: Any, args: Any) -> str:
    return f"{tool}:{json.dumps(args, sort_keys=True, default=str)}"


def _starts(evs: list[dict[str, Any]], tool: Any, call_id: Any) -> list[Any]:
    """The arguments of each start of ``tool``'s call ``call_id`` in a run's events (by its id,
    else by its tool)."""
    args: dict[Any, Any] = {}
    started: list[Any] = []
    for p in evs:
        ev = p.get("event", p)
        if ev.get("type") == "TOOL_CALL_ARGS":
            args[ev.get("tool_call_id")] = (ev.get("data") or {}).get("args")
    for p in evs:
        ev = p.get("event", p)
        named = call_id is None or ev.get("tool_call_id") == call_id
        if (
            ev.get("type") == "TOOL_CALL_START"
            and (ev.get("data") or {}).get("tool") == tool
            and named
        ):
            started.append(args.get(ev.get("tool_call_id")))
    if not started and call_id is not None:  # a framework that names its calls otherwise
        return _starts(evs, tool, None)
    return started


def _fields(value: Any, names: tuple[str, ...]) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for k, v in value.items():
            if k in names and isinstance(v, str):
                yield k, v
            else:
                yield from _fields(v, names)
    elif isinstance(value, list):
        for v in value:
            yield from _fields(v, names)


@pytest.fixture(autouse=True)
def invariants(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> Iterator[Invariants]:
    """INV-1..INV-10 at the end of the test (see the module)."""
    inv = Invariants(tmp_path)
    _UNDER_WAY[:] = [inv]
    monkeypatch.setattr(httpx.AsyncClient, "send", recording(inv.traffic))
    before = set(_children(os.getpid()))
    temp = Path(tempfile.gettempdir())
    temp_before = {p.name for p in temp.iterdir()} if temp.is_dir() else set()
    yield inv
    violations = asyncio.run(inv.check()) if not _in_loop() else []
    # INV-8: the log records of the test
    if inv.secrets_in(caplog.text) and "INV-8" not in inv.waived:
        violations.append("INV-8 a key in a log record")
    # INV-9: processes the test started, and what it left in the temporary directory
    deadline = time.monotonic() + 10
    left = {}
    while time.monotonic() < deadline:
        left = {p: c for p, c in _children(os.getpid()).items() if p not in before}
        if not left:
            break
        time.sleep(0.5)
    if left and "INV-9" not in inv.waived:
        violations.append(
            f"INV-9 processes the test started outlive it: {sorted(set(left.values()))[:3]}"
        )
    if temp.is_dir() and "INV-9" not in inv.waived:
        base = str(request.config._tmp_path_factory.getbasetemp())  # type: ignore[attr-defined]
        new = [
            p for p in temp.iterdir() if p.name not in temp_before and not base.startswith(str(p))
        ]
        if new:
            violations.append(f"INV-9 left in {temp}: {sorted(p.name for p in new)[:5]}")
    _UNDER_WAY.clear()
    assert not violations, "invariants broken:\n" + "\n".join(dict.fromkeys(violations))


#: the invariants of the test under way (a helper that starts a process names its traffic file)
_UNDER_WAY: list[Invariants] = []


def current() -> Invariants | None:
    """The invariants of the test under way, if it has them."""
    return _UNDER_WAY[0] if _UNDER_WAY else None


def _in_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
