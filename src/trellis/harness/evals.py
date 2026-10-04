"""Evaluation: score what an agent answered, offline over a dataset or online on live runs.

An :class:`Evaluator` is any ``async (EvalCase) -> EvalScore | None``: it reads the case — the
input, the answer, what was expected, the memory context the run was given — and returns a
score, or ``None`` when it has nothing to say about this case. Built in:

* :func:`grounding` — the answer checked against the memory context the run was given (the
  memory service's ``/v1/verify`` with the run's ``bundle_id``: the same check as the sampled
  one every run gets);
* :func:`exact_match`, :func:`contains` — against the case's ``expected``;
* :func:`llm_judge` — a judge model scores the answer against plain-language criteria (strict
  JSON ``{score, reasoning}`` at temperature 0). Which model, and through which virtual key,
  is the deployment's (``TRELLIS_JUDGE_MODEL``, ``TRELLIS_JUDGE_VIRTUAL_KEY``), never the code's.

Langfuse is the system of record: every score goes on the run's trace (``POST
/api/public/scores``, and a ``score`` span); each run is an item of a Langfuse experiment, as
its SDK's experiment runner makes one — a dataset read from Langfuse gets each run's trace
linked to the dataset run (``POST /api/public/dataset-run-items``, Langfuse v3), and every
span of the run carries the experiment's attributes (``langfuse.experiment.*``, what Langfuse v4
builds experiments from: ``telemetry.Experiment``).

* **Offline** — ``await h.evaluate(agent, dataset, evaluators)``: each item is run through the
  normal pipeline (memory, tools, approvals), evaluated, and scored; :class:`EvalReport` says
  how each item went and the mean of each evaluator.
* **Online** — ``Harness(judges=[...])``: after a sampled successful run (``TRELLIS_JUDGE_SAMPLE``,
  by the run id) each judge runs in the background writes queue, never on the request path,
  and its score lands on the run's trace. A judge that fails is a warning, never a failed run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness import pipeline, telemetry
from trellis.harness.adapters.react import ReAct, _message, _Named, _unfenced

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.harness import Harness

log = logging.getLogger("trellis.evals")

#: Items evaluated at once by default.
CONCURRENCY: Final = 4
#: The user offline evaluation runs act for, unless ``user=`` names one (memory is scoped to it).
EVAL_USER: Final = "trellis-evaluate"
#: Who cancels an item's run that paused, so it does not wait in an inbox.
EVAL_REVIEWER: Final = "trellis:evaluate"
#: What the judge model is told: score the answer against the criteria, as strict JSON.
JUDGE_SYSTEM: Final = (
    "You are a strict evaluator. You grade an AI assistant's answer against the criteria you "
    "are given, and nothing else. Reply with only a JSON object, no prose and no code fence: "
    '{"score": <a number from 0 to 1, where 1 fully meets the criteria>, "reasoning": '
    '"<one or two sentences>"}'
)
#: What the judge is told when its reply was not that JSON object (it is asked once more).
JUDGE_RETRY: Final = (
    'That was not the JSON object asked for ({problem}). Reply with only {{"score": <0 to 1>, '
    '"reasoning": "<text>"}}.'
)
#: How much of each part of the case the judge reads (characters).
JUDGE_PART_CHARS: Final = 8000
_JSON_OBJECT: Final = re.compile(r"\{.*\}", re.S)

ItemStatus = Literal["success", "error", "interrupted", "cancelled"]


# --------------------------------------------------------------------------- the model


@dataclass(frozen=True, slots=True)
class EvalCase:
    """What an evaluator judges: one run's input and answer, and what it is compared with."""

    input: Any
    #: the agent's answer
    output: Any
    #: what the answer should be, when the dataset says
    expected: Any = None
    run_id: str | None = None
    #: the memory context pushed into the run (its ``bundle_id`` and rendered text), if any
    bundle_id: str | None = None
    context: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: the agent that answered (built-in evaluators reach the harness through it)
    agent: Agent | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class EvalScore:
    """One score: a number from 0 to 1, a bool (Langfuse ``BOOLEAN``) or a category (a string,
    ``CATEGORICAL``)."""

    name: str
    value: float | bool | str
    comment: str | None = None


Evaluator = Callable[[EvalCase], Awaitable[EvalScore | None]]


class EvalItem(BaseModel):
    """One dataset item: the input the agent is run on, what is expected, and its Langfuse id
    when it came from a Langfuse dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: Any
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


@dataclass(frozen=True, slots=True)
class EvalResult:
    """How one item went: the run, its answer, its scores (an evaluator that failed is in
    ``failed``, by name, with why)."""

    input: Any
    expected: Any
    output: Any
    status: ItemStatus
    run_id: str | None
    scores: list[EvalScore] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    trace_url: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluatorStats:
    """One evaluator over the report: the mean of its numeric and bool scores (``None`` with
    none), how many scores it gave, and how many times it failed."""

    mean: float | None
    count: int
    failures: int


class Summary(dict[str, EvaluatorStats]):
    """Each evaluator's :class:`EvaluatorStats`, by name; ``print(report.summary)`` is a
    table."""

    def __str__(self) -> str:
        lines = [f"{'evaluator':<24} {'mean':>6} {'count':>6} {'failures':>8}"]
        for name, stats in self.items():
            mean = "-" if stats.mean is None else f"{stats.mean:.3f}"
            lines.append(f"{name:<24} {mean:>6} {stats.count:>6} {stats.failures:>8}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class EvalReport:
    """An offline evaluation: every item in dataset order, and each evaluator's summary."""

    run_name: str
    items: list[EvalResult]
    summary: Summary
    #: the Langfuse dataset it ran, when it came from one
    dataset: str | None = None
    #: the experiment's id in Langfuse: the dataset run's (as Langfuse answered the first
    #: link), else one made for this evaluation — what every item's spans carry
    experiment_id: str | None = None
    #: where Langfuse shows the dataset run, when it is one
    dataset_run_url: str | None = None

    @property
    def statuses(self) -> dict[str, int]:
        """How many items ended each way (``success``, ``error``, ``interrupted``...)."""
        counted: dict[str, int] = {}
        for item in self.items:
            counted[item.status] = counted.get(item.status, 0) + 1
        return counted

    def __str__(self) -> str:
        ended = ", ".join(f"{n} {status}" for status, n in self.statuses.items())
        where = f" — {self.dataset_run_url}" if self.dataset_run_url else ""
        return f"{self.run_name}: {len(self.items)} items ({ended}){where}\n{self.summary}"


# --------------------------------------------------------------------------- evaluators


def name_of(evaluator: Evaluator) -> str:
    """The name an evaluator's scores and failures are reported under."""
    named = getattr(evaluator, "name", None)
    return named if isinstance(named, str) else getattr(evaluator, "__name__", "evaluator")


@dataclass(frozen=True, slots=True)
class grounding:
    """The share of the answer's claims the run's memory context supports (``/v1/verify``
    with its ``bundle_id``); no score without memory, without a pushed context, or for an
    answer with no checkable claim."""

    name: str = "grounding"

    async def __call__(self, case: EvalCase) -> EvalScore | None:
        agent = case.agent
        if agent is None or case.bundle_id is None or case.run_id is None:
            return None
        if not isinstance(case.output, str) or not case.output:
            return None
        record = await agent.harness.runs.get(case.run_id)
        memory = await agent.run_memory(agent._identity_of(record)) if record else None
        if memory is None:
            return None
        score = await memory.verify(case.output, case.bundle_id)
        return None if score is None else EvalScore(self.name, score)


@dataclass(frozen=True, slots=True)
class exact_match:
    """Whether the answer is the expected one (text compared trimmed, and case-blind unless
    ``case_sensitive``); no score without an ``expected``."""

    name: str = "exact_match"
    case_sensitive: bool = False

    async def __call__(self, case: EvalCase) -> EvalScore | None:
        if case.expected is None:
            return None
        if isinstance(case.output, str) and isinstance(case.expected, str):
            got, want = case.output.strip(), case.expected.strip()
            if not self.case_sensitive:
                got, want = got.casefold(), want.casefold()
            return EvalScore(self.name, got == want)
        return EvalScore(self.name, pipeline.jsonable(case.output) == case.expected)


@dataclass(frozen=True, slots=True)
class contains:
    """Whether the answer contains the expected text — every one of them, when ``expected`` is
    a list (case-blind unless ``case_sensitive``); no score without an ``expected``."""

    name: str = "contains"
    case_sensitive: bool = False

    async def __call__(self, case: EvalCase) -> EvalScore | None:
        if case.expected is None:
            return None
        wanted = case.expected if isinstance(case.expected, list) else [case.expected]
        text = case.output if isinstance(case.output, str) else json.dumps(case.output)
        if not self.case_sensitive:
            text = text.casefold()
        found = [str(w) if self.case_sensitive else str(w).casefold() for w in wanted]
        missing = [w for w, f in zip(wanted, found, strict=True) if f not in text]
        comment = f"missing: {', '.join(map(str, missing))}" if missing else None
        return EvalScore(self.name, not missing, comment)


@dataclass(frozen=True, slots=True)
class llm_judge:
    """A judge model scores the answer against ``criteria`` (0 to 1, with its reasoning as the
    comment). The model is ``TRELLIS_JUDGE_MODEL`` through ``BIFROST_URL`` with
    ``TRELLIS_JUDGE_VIRTUAL_KEY`` (else ``BIFROST_VIRTUAL_KEY``); with no judge model set, the
    judged agent's own model (logged once). A reply that is not the JSON asked for is asked
    once more; a second one is no score, with a warning."""

    criteria: str
    name: str = "llm_judge"

    async def __call__(self, case: EvalCase) -> EvalScore | None:
        if case.agent is None:
            raise ConfigurationError("llm_judge judges a harness agent's runs")
        model = judge_model(case.agent)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": _judge_prompt(self.criteria, case)},
        ]
        for attempt in range(2):
            message = _message(await model.complete(messages, temperature=0))
            content = message.get("content")
            score, problem = _verdict(content)
            if score is not None:
                return EvalScore(self.name, score[0], score[1] or None)
            if attempt == 0:
                messages.append({"role": "assistant", "content": str(content or "")})
                messages.append({"role": "user", "content": JUDGE_RETRY.format(problem=problem)})
        log.warning(
            "judge %s gave no score for run %s: its reply was not the JSON asked for twice",
            self.name,
            case.run_id,
        )
        return None


def judge_model(agent: Agent) -> Any:
    """The chat model the judge asks: ``TRELLIS_JUDGE_MODEL`` through the judge's gateway, else
    the judged agent's own model (logged once per harness)."""
    harness = agent.harness
    if harness._judge_model is not None:  # a stand-in, in tests
        return harness._judge_model
    model: Any = harness.settings.judge_model
    if model is None:
        target = agent.target
        model = target.model if isinstance(target, ReAct) else None
        if model is None:
            raise ConfigurationError(
                "llm_judge needs a model: set TRELLIS_JUDGE_MODEL (a Bifrost model name)"
            )
        if not harness._judge_shares_logged:
            harness._judge_shares_logged = True
            log.warning(
                "TRELLIS_JUDGE_MODEL is unset: the judge shares the model of agent %s (set a "
                "different, stronger model so it does not grade itself)",
                agent.id,
            )
        if not isinstance(model, str):
            return model
    if harness.judge_gateway is None:
        raise ConfigurationError("llm_judge with a model name needs BIFROST_URL")
    return _Named(harness.judge_gateway, model)


def _judge_prompt(criteria: str, case: EvalCase) -> str:
    parts = [("Criteria", criteria), ("Input", case.input)]
    if case.expected is not None:
        parts.append(("Expected answer", case.expected))
    if case.context:
        parts.append(("Context the assistant was given", case.context))
    parts.append(("Answer to grade", case.output))
    return "\n\n".join(f"## {title}\n{_part(value)}" for title, value in parts)


def _part(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:JUDGE_PART_CHARS]


def _verdict(content: Any) -> tuple[tuple[float, str] | None, str]:
    """``((score, reasoning), "")`` read from a judge's reply, or ``(None, what is wrong)``."""
    if not isinstance(content, str) or not content.strip():
        return None, "the reply was empty"
    found = _JSON_OBJECT.search(_unfenced(content))
    if found is None:
        return None, "no JSON object in it"
    try:
        verdict = json.loads(found.group(0))
    except ValueError:
        return None, "the JSON did not parse"
    score = verdict.get("score") if isinstance(verdict, dict) else None
    if isinstance(score, bool) or not isinstance(score, int | float) or not 0 <= score <= 1:
        return None, "score must be a number from 0 to 1"
    reasoning = verdict.get("reasoning")
    return (float(score), reasoning if isinstance(reasoning, str) else ""), ""


# --------------------------------------------------------------------------- running them


async def scored(
    harness: Harness, case: EvalCase, evaluators: Sequence[Evaluator]
) -> tuple[list[EvalScore], dict[str, str]]:
    """Run each evaluator on ``case`` and put each score on the run's trace; an evaluator that
    raises is a failure (logged), and a score Langfuse refuses a warning — never an exception."""
    scores: list[EvalScore] = []
    failed: dict[str, str] = {}
    for evaluator in evaluators:
        name = name_of(evaluator)
        try:
            score = await evaluator(case)
        except Exception as exc:
            failed[name] = f"{type(exc).__name__}: {exc}"
            log.warning("evaluator %s failed on run %s: %s", name, case.run_id, exc)
            continue
        if score is None:
            continue
        scores.append(score)
        if case.run_id is None:
            continue
        try:
            await harness.score(
                case.run_id,
                score.name,
                score.value,
                key=f"{case.run_id}:{score.name}",
                comment=score.comment,
            )
        except Exception as exc:
            log.warning("score %s of run %s was not posted: %s", score.name, case.run_id, exc)
    return scores, failed


def items_of(dataset: Sequence[Any]) -> list[EvalItem]:
    """A local dataset as items: :class:`EvalItem`\\ s, or mappings with ``input`` (and
    ``expected``, ``metadata``, ``id``)."""
    return [
        item if isinstance(item, EvalItem) else EvalItem.model_validate(item) for item in dataset
    ]


def _langfuse_item(item: Mapping[str, Any]) -> EvalItem:
    metadata = item.get("metadata")
    return EvalItem(
        input=item.get("input"),
        expected=item.get("expectedOutput"),
        metadata=metadata if isinstance(metadata, dict) else {},
        id=item.get("id"),
    )


def item_id_of(item: EvalItem) -> str:
    """The experiment item id: the dataset item's, else — as Langfuse's SDK makes it — the first
    16 hex characters of the SHA-256 of the serialized input."""
    if item.id:
        return item.id
    text = telemetry.serialized(item.input)
    return hashlib.sha256((text if text is not None else "null").encode()).hexdigest()[:16]


@dataclass(slots=True)
class _Run:
    """One evaluation, as its items share it: where its items go in Langfuse."""

    name: str
    agent_id: str
    langfuse: telemetry.Langfuse | None
    #: the Langfuse dataset (its ``id``, ``projectId``), when the items came from one
    dataset: dict[str, Any] | None
    description: str | None
    metadata: dict[str, Any]
    #: the experiment id when no dataset run says one (made once per evaluation, as the SDK
    #: makes its fallback)
    fallback_id: str = field(default_factory=lambda: os.urandom(8).hex())
    #: the dataset run's id, as Langfuse answered the first link
    dataset_run_id: str | None = None

    async def experiment(self, run_id: str, item: EvalItem) -> telemetry.Experiment:
        """The item's experiment: linked to the dataset run first (v3; a refusal is a warning),
        so the run's spans carry the dataset run's id."""
        linked: str | None = None
        if self.langfuse is not None and self.dataset is not None and item.id:
            try:
                linked = await self.langfuse.link(
                    run_id,
                    run_name=self.name,
                    item_id=item.id,
                    metadata=self.metadata,
                    description=self.description,
                )
            except Exception as exc:
                log.warning("run %s was not linked to dataset run %s: %s", run_id, self.name, exc)
        self.dataset_run_id = self.dataset_run_id or linked
        return telemetry.Experiment(
            id=linked or self.dataset_run_id or self.fallback_id,
            name=self.name,
            item_id=item_id_of(item),
            dataset_id=self.dataset.get("id") if self.dataset is not None else None,
            description=self.description,
            metadata=telemetry.flattened(self.metadata),
            expected_output=telemetry.serialized(item.expected),
            item_metadata=telemetry.flattened(item.metadata),
        )

    def url(self) -> str | None:
        dataset = self.dataset or {}
        project, dataset_id = dataset.get("projectId"), dataset.get("id")
        if self.langfuse is None or not (project and dataset_id and self.dataset_run_id):
            return None
        return self.langfuse.dataset_run_url(project, dataset_id, self.dataset_run_id)


async def evaluate(
    harness: Harness,
    agent: Agent,
    dataset: str | Sequence[Any],
    evaluators: Sequence[Evaluator],
    *,
    run_name: str | None = None,
    description: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    concurrency: int = CONCURRENCY,
    limit: int | None = None,
    user: str | None = None,
) -> EvalReport:
    """What ``Harness.evaluate`` does (see there)."""
    if concurrency < 1:
        raise ConfigurationError("evaluate runs at least one item at a time")
    langfuse = harness.scores
    found: dict[str, Any] | None = None
    if isinstance(dataset, str):
        if langfuse is None:
            raise ConfigurationError(
                f"the dataset {dataset!r} is read from Langfuse: set OTEL_EXPORTER_OTLP_ENDPOINT "
                "and OTEL_EXPORTER_OTLP_HEADERS to Langfuse's (or pass the items themselves)"
            )
        found, rows = await langfuse.dataset(dataset)
        items = [_langfuse_item(i) for i in rows]
    else:
        items = items_of(dataset)
    items = items[:limit] if limit is not None else items
    run = _Run(
        name=run_name or f"{agent.id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        agent_id=agent.id,
        langfuse=langfuse,
        dataset=found,
        description=description,
        metadata={"agent_id": agent.id, **(metadata or {})},
    )
    slots = asyncio.Semaphore(concurrency)

    async def one(item: EvalItem) -> EvalResult:
        async with slots:
            return await _item(harness, agent, item, evaluators, user=user or EVAL_USER, run=run)

    results = await asyncio.gather(*(one(item) for item in items))
    await harness.writes.drain()
    await telemetry.flush()
    return EvalReport(
        run_name=run.name,
        items=list(results),
        summary=_summary(results, evaluators),
        dataset=dataset if isinstance(dataset, str) else None,
        experiment_id=run.dataset_run_id or run.fallback_id,
        dataset_run_url=run.url(),
    )


async def _item(
    harness: Harness,
    agent: Agent,
    item: EvalItem,
    evaluators: Sequence[Evaluator],
    *,
    user: str,
    run: _Run,
) -> EvalResult:
    """Run one item through the pipeline, as an item of the experiment, and evaluate its
    answer; whatever goes wrong is the item's error, never the evaluation's."""
    pushed: list[Any] = []
    run_id: str | None = None
    try:
        identity = await agent._opened(item.input, user=user, thread=None, tenant=None)
        run_id = identity.run_id
        current = await run.experiment(run_id, item)
        with telemetry.experiment(current):
            result = await pipeline.attempt(agent, identity, item.input, observe=pushed.append)
        if result.status is RunStatus.PAUSED and result.interrupt is not None:
            await agent.resume(result.interrupt.interrupt_id, "cancel", reviewer=EVAL_REVIEWER)
    except Exception as exc:
        log.warning("evaluation item failed: %s", exc)
        return _result(harness, item, None, "error", run_id, error=f"{type(exc).__name__}: {exc}")
    if result.status is not RunStatus.SUCCESS:
        status: ItemStatus = (
            "interrupted"
            if result.status is RunStatus.PAUSED
            else "cancelled"
            if result.status is RunStatus.CANCELLED
            else "error"
        )
        error = result.error.message if result.error is not None else None
        return _result(harness, item, None, status, run_id, error=error)
    context = pushed[0] if pushed else None
    case = EvalCase(
        input=item.input,
        output=result.answer,
        expected=item.expected,
        run_id=run_id,
        bundle_id=getattr(context, "bundle_id", None),
        context=getattr(context, "rendered", None),
        metadata=item.metadata,
        agent=agent,
    )
    with telemetry.experiment(current):  # the score spans are the experiment's too
        scores, failed = await scored(harness, case, evaluators)
    return _result(harness, item, result.answer, "success", run_id, scores=scores, failed=failed)


def _result(
    harness: Harness,
    item: EvalItem,
    output: Any,
    status: ItemStatus,
    run_id: str | None,
    **fields: Any,
) -> EvalResult:
    url = harness.scores.trace_url(run_id) if harness.scores is not None and run_id else None
    return EvalResult(
        input=item.input,
        expected=item.expected,
        output=output,
        status=status,
        run_id=run_id,
        trace_url=url,
        **fields,
    )


def _summary(results: Sequence[EvalResult], evaluators: Sequence[Evaluator]) -> Summary:
    summary = Summary()
    for evaluator in evaluators:
        name = name_of(evaluator)
        given = [s for r in results for s in r.scores if s.name == name]
        numbers = [float(s.value) for s in given if not isinstance(s.value, str)]
        summary[name] = EvaluatorStats(
            mean=round(sum(numbers) / len(numbers), 4) if numbers else None,
            count=len(given),
            failures=sum(name in r.failed for r in results),
        )
    return summary


__all__ = [
    "EvalCase",
    "EvalItem",
    "EvalReport",
    "EvalResult",
    "EvalScore",
    "Evaluator",
    "EvaluatorStats",
    "Summary",
    "contains",
    "exact_match",
    "grounding",
    "llm_judge",
]
