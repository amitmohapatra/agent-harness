"""Langfuse evaluation + prompt providers (§29, §30, §31).

Scoring is out-of-band: the harness returns the agent's result first and scores afterwards.
DeepEval (or any other metric runner) stays responsible for *computing* metrics; this
provider is how a computed score gets back onto the Langfuse trace.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from universal_agent_contracts.context import AgentExecutionContext
from universal_agent_contracts.events import AgentEvalEvent

from universal_agent_harness.runtime.logging import get_logger

log = get_logger("universal_agent_harness.langfuse")


class LangfuseEvaluationProvider:
    """Pushes scores, dataset items and feedback into Langfuse."""

    name = "langfuse"

    def __init__(self, client: Any, *, strict: bool = False) -> None:
        self.client = client
        self.strict = strict

    async def score(
        self,
        name: str,
        value: float | str,
        /,
        *,
        context: AgentExecutionContext | None = None,
        comment: str | None = None,
        **metadata: Any,
    ) -> None:
        trace_id = metadata.pop("trace_id", None) or (context.trace_id if context else None)
        observation_id = metadata.pop("observation_id", None)
        data_type = metadata.pop("data_type", None) or (
            "NUMERIC" if isinstance(value, int | float) else "CATEGORICAL"
        )
        await self._run(
            lambda: self.client.create_score(
                name=name,
                value=value,
                trace_id=trace_id,
                observation_id=observation_id,
                comment=comment,
                data_type=data_type,
                metadata=metadata or None,
            )
        )

    async def submit_dataset_item(self, dataset: str, item: Mapping[str, Any]) -> None:
        payload = dict(item)
        await self._run(
            lambda: self.client.create_dataset_item(
                dataset_name=dataset,
                input=payload.get("input"),
                expected_output=payload.get("expected_output"),
                metadata=payload.get("metadata"),
            )
        )

    async def submit_feedback(self, feedback: Mapping[str, Any]) -> None:
        data = dict(feedback)
        await self.score(
            data.pop("name", "user_feedback"),
            data.pop("value", 1),
            comment=data.pop("comment", None),
            **data,
        )

    async def _run(self, fn: Any) -> None:
        """Langfuse's client is synchronous and buffered; keep it off the event loop."""
        try:
            await asyncio.to_thread(fn)
        except Exception as exc:
            if self.strict:
                raise
            log.warning("langfuse evaluation call failed", error=str(exc))


class LangfuseEvaluationSink:
    """An :class:`EvaluationSink` that turns harness eval events into Langfuse scores.

    Only cheap, objective facts are scored here — latency and success. Quality metrics come
    from an evaluator (DeepEval, an LLM judge, a human) and are submitted through
    :class:`LangfuseEvaluationProvider` when they are ready.
    """

    name = "langfuse"

    def __init__(self, provider: LangfuseEvaluationProvider) -> None:
        self.provider = provider

    async def emit(self, event: AgentEvalEvent) -> None:
        await self.provider.score(
            "agent_success",
            1.0 if event.status in ("SUCCESS", "PARTIAL") else 0.0,
            trace_id=event.trace_id,
            data_type="NUMERIC",
            agent_id=event.agent_id,
            agent_run_id=event.agent_run_id,
        )
        if event.latency_ms:
            await self.provider.score(
                "agent_latency_ms",
                float(event.latency_ms),
                trace_id=event.trace_id,
                data_type="NUMERIC",
                agent_id=event.agent_id,
            )


class LangfusePromptProvider:
    """Prompt management through Langfuse (§31). Entirely optional."""

    name = "langfuse"

    def __init__(self, client: Any) -> None:
        self.client = client

    async def get_prompt(self, name: str, /, *, version: str | None = None, **vars: Any) -> Any:
        label = vars.pop("label", None)

        def _fetch() -> Any:
            kwargs: dict[str, Any] = {}
            if version is not None:
                kwargs["version"] = int(version) if str(version).isdigit() else version
            if label is not None:
                kwargs["label"] = label
            return self.client.get_prompt(name, **kwargs)

        prompt = await asyncio.to_thread(_fetch)
        if vars and hasattr(prompt, "compile"):
            return prompt.compile(**vars)
        return prompt
