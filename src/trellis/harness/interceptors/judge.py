"""The online judge, hung off the end of a turn (design §11).

The turn returns first. Judging happens on the writeback queue, which is what makes "sampled,
asynchronous, off the critical path" true rather than aspirational: a user waiting for an
answer never waits for an opinion about it.

What a verdict becomes, all three from one place so they cannot disagree:

* a **score on the trace** — a Langfuse score keyed by trace id (which works after the run's
  span has closed) and a short ``agent.judge`` span carrying the same numbers, so a plain OTLP
  backend sees it too;
* a **feedback record** with ``source="judge"``, through the Memory Service, where human
  feedback already lives — the two records meet in one table (design §7);
* nothing at all, when the judge abstained.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from trellis.contracts.context import AgentExecutionContext
from trellis.contracts.evaluation import JudgeVerdict
from trellis.contracts.events import AgentEvalEvent
from trellis.contracts.messages import AgentRequest, AgentResponse

from trellis.harness.evaluation.events import build_eval_event
from trellis.harness.evaluation.judge import ScopedJudge
from trellis.harness.interceptors.base import BaseInterceptor, Order
from trellis.harness.memory.writeback import WritebackQueue
from trellis.harness.telemetry import names as N

if TYPE_CHECKING:  # pragma: no cover
    from trellis.harness.runtime.agent_runtime import AgentRuntime
    from trellis.harness.telemetry.tracer import HarnessTracer

#: What the score is called on a trace. One name, so a dashboard built on it keeps working.
SCORE_NAME = "judge_score"
METHOD_SCORE_NAME = "judge_method"

#: The feedback the harness records for a verdict.
FeedbackWriter = Callable[..., Awaitable[Any]]


class JudgeInterceptor(BaseInterceptor):
    """Runs the configured judge after the turn, never during it."""

    name = "judge"
    order = Order.EVALUATION

    def __init__(
        self,
        judge: Any,
        *,
        queue: WritebackQueue | None = None,
        evaluation: Any = None,
        feedback: FeedbackWriter | None = None,
        tracer: HarnessTracer | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.judge = judge
        self.queue = queue or WritebackQueue()
        self.evaluation = evaluation
        self.feedback = feedback
        self.tracer = tracer
        self.threshold = threshold

    async def after(self, result: AgentResponse, runtime: AgentRuntime) -> AgentResponse:
        if not result.status.ok:
            # A failed turn has no answer to judge; its failure is already the record.
            return result
        event = build_eval_event(runtime, result)
        admits = getattr(self.judge, "admits", None)
        if callable(admits) and not admits(event):
            # Cheap path: an unsampled or over-budget run does no work at all, not even the
            # binding below.
            return result
        judge = self._bind(runtime)
        work = self._judge(judge, event, result, runtime.context)
        if self.queue.submit(work, name=f"judge:{runtime.run_id}") is None:
            # A sampled judgement is optional; a memory write is not. Dropping beats writing
            # an opinion inline on a turn that is already behind. The coroutine is closed
            # rather than abandoned: an un-awaited coroutine is a warning on somebody else's
            # line, and a dropped judgement should be quiet.
            work.close()
            runtime.logger.debug("judge queue saturated; skipping this run")
        return result

    # ------------------------------------------------------------------ internals
    def _bind(self, runtime: AgentRuntime) -> Any:
        """Give the judge this run's evidence, when it can use it."""
        if not isinstance(self.judge, ScopedJudge):
            return self.judge
        request: AgentRequest | None = runtime.state.get("request")
        return self.judge.bound(
            verifier=runtime.memory if runtime.memory.enabled else None,
            bundle=runtime.memory_context,
            question=request.query if request is not None else None,
        )

    async def _judge(
        self,
        judge: Any,
        event: AgentEvalEvent,
        result: AgentResponse,
        context: AgentExecutionContext,
    ) -> None:
        verdict = await judge.judge(event, response=result)
        if verdict is None:
            return
        self._span(verdict, event)
        await self._score(verdict, event)
        await self._record(verdict, event, context)

    def _span(self, verdict: JudgeVerdict, event: AgentEvalEvent) -> None:
        if self.tracer is None:
            return
        with self.tracer.span(
            N.JUDGE,
            kind=N.KIND_INTERNAL,
            category="agent",
            attributes={
                N.JUDGE_SCORE: verdict.score,
                N.JUDGE_METHOD: verdict.method.value,
                N.JUDGE_LABEL: verdict.label,
                N.JUDGE_MODEL: verdict.model,
                N.JUDGE_COST: verdict.cost_usd,
                N.AGENT_ID: event.agent_id,
                N.AGENT_RUN_ID: event.agent_run_id,
                N.TENANT_ID: event.tenant_id,
            },
        ):
            pass

    async def _score(self, verdict: JudgeVerdict, event: AgentEvalEvent) -> None:
        if self.evaluation is None:
            return
        try:
            await self.evaluation.score(
                SCORE_NAME,
                float(verdict.score),
                trace_id=event.trace_id,
                data_type="NUMERIC",
                comment=verdict.rationale,
                agent_id=event.agent_id,
                agent_run_id=event.agent_run_id,
                method=verdict.method.value,
            )
            await self.evaluation.score(
                METHOD_SCORE_NAME,
                verdict.method.value,
                trace_id=event.trace_id,
                data_type="CATEGORICAL",
                agent_id=event.agent_id,
            )
        except Exception:  # pragma: no cover - a score is never worth an exception
            pass

    async def _record(
        self, verdict: JudgeVerdict, event: AgentEvalEvent, context: AgentExecutionContext
    ) -> None:
        """The verdict as feedback, built by the contract itself so the judge and a human
        write the same shape of record."""
        if self.feedback is None:
            return
        record = verdict.as_feedback(event, threshold=self.threshold)
        await self.feedback(
            context,
            record.target_kind.value,
            record.target_id,
            record.verdict.value,
            score=record.score,
            comment=record.comment,
            reviewer=record.reviewer,
            source=record.source.value,
            feedback_id=record.feedback_id,
            metadata=dict(record.metadata),
        )


__all__ = ["METHOD_SCORE_NAME", "SCORE_NAME", "JudgeInterceptor"]
