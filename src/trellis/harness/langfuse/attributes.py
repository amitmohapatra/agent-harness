"""Langfuse's OpenTelemetry attribute names (§22, §24).

Langfuse's own SDK exports these as ``LangfuseOtelSpanAttributes``; we import them when the
SDK is installed and fall back to the same literal strings when it is not, so ``otlp`` mode
works with nothing but an OTLP exporter pointed at Langfuse.

Setting these on the harness's *existing* OpenTelemetry spans is what keeps one trace
identity: Langfuse renders the same span tree an OTLP backend sees, rather than a parallel
one built from a second SDK.
"""

from __future__ import annotations

from typing import Final

_FALLBACK: Final[dict[str, str]] = {
    "TRACE_NAME": "langfuse.trace.name",
    "TRACE_SESSION_ID": "session.id",
    "TRACE_USER_ID": "user.id",
    "TRACE_TAGS": "langfuse.trace.tags",
    "TRACE_METADATA": "langfuse.trace.metadata",
    "TRACE_INPUT": "langfuse.trace.input",
    "TRACE_OUTPUT": "langfuse.trace.output",
    "OBSERVATION_TYPE": "langfuse.observation.type",
    "OBSERVATION_INPUT": "langfuse.observation.input",
    "OBSERVATION_OUTPUT": "langfuse.observation.output",
    "OBSERVATION_METADATA": "langfuse.observation.metadata",
    "OBSERVATION_LEVEL": "langfuse.observation.level",
    "OBSERVATION_STATUS_MESSAGE": "langfuse.observation.status_message",
    "OBSERVATION_MODEL": "langfuse.observation.model.name",
    "OBSERVATION_MODEL_PARAMETERS": "langfuse.observation.model.parameters",
    "OBSERVATION_USAGE_DETAILS": "langfuse.observation.usage_details",
    "OBSERVATION_COST_DETAILS": "langfuse.observation.cost_details",
    "OBSERVATION_PROMPT_NAME": "langfuse.observation.prompt.name",
    "OBSERVATION_PROMPT_VERSION": "langfuse.observation.prompt.version",
    "OBSERVATION_COMPLETION_START_TIME": "langfuse.observation.completion_start_time",
    "ENVIRONMENT": "langfuse.environment",
    "RELEASE": "langfuse.release",
    "VERSION": "langfuse.version",
}


def _resolve() -> tuple[dict[str, str], bool]:
    """Prefer the SDK's own constants; fall back to the documented literals."""
    try:
        from langfuse import LangfuseOtelSpanAttributes as sdk  # noqa: PLC0415
    except ImportError:  # pragma: no cover - otlp mode without the SDK installed
        return dict(_FALLBACK), False
    return {name: getattr(sdk, name, default) for name, default in _FALLBACK.items()}, True


_NAMES, HAS_SDK_ATTRIBUTES = _resolve()

TRACE_NAME: Final[str] = _NAMES["TRACE_NAME"]
TRACE_SESSION_ID: Final[str] = _NAMES["TRACE_SESSION_ID"]
TRACE_USER_ID: Final[str] = _NAMES["TRACE_USER_ID"]
TRACE_TAGS: Final[str] = _NAMES["TRACE_TAGS"]
TRACE_METADATA: Final[str] = _NAMES["TRACE_METADATA"]
TRACE_INPUT: Final[str] = _NAMES["TRACE_INPUT"]
TRACE_OUTPUT: Final[str] = _NAMES["TRACE_OUTPUT"]
OBSERVATION_TYPE: Final[str] = _NAMES["OBSERVATION_TYPE"]
OBSERVATION_INPUT: Final[str] = _NAMES["OBSERVATION_INPUT"]
OBSERVATION_OUTPUT: Final[str] = _NAMES["OBSERVATION_OUTPUT"]
OBSERVATION_METADATA: Final[str] = _NAMES["OBSERVATION_METADATA"]
OBSERVATION_LEVEL: Final[str] = _NAMES["OBSERVATION_LEVEL"]
OBSERVATION_STATUS_MESSAGE: Final[str] = _NAMES["OBSERVATION_STATUS_MESSAGE"]
OBSERVATION_MODEL: Final[str] = _NAMES["OBSERVATION_MODEL"]
OBSERVATION_MODEL_PARAMETERS: Final[str] = _NAMES["OBSERVATION_MODEL_PARAMETERS"]
OBSERVATION_USAGE_DETAILS: Final[str] = _NAMES["OBSERVATION_USAGE_DETAILS"]
OBSERVATION_COST_DETAILS: Final[str] = _NAMES["OBSERVATION_COST_DETAILS"]
OBSERVATION_PROMPT_NAME: Final[str] = _NAMES["OBSERVATION_PROMPT_NAME"]
OBSERVATION_PROMPT_VERSION: Final[str] = _NAMES["OBSERVATION_PROMPT_VERSION"]
OBSERVATION_COMPLETION_START_TIME: Final[str] = _NAMES["OBSERVATION_COMPLETION_START_TIME"]
ENVIRONMENT: Final[str] = _NAMES["ENVIRONMENT"]
RELEASE: Final[str] = _NAMES["RELEASE"]
VERSION: Final[str] = _NAMES["VERSION"]

#: Harness span name -> Langfuse observation type (§24/§25).
OBSERVATION_TYPES: Final[dict[str, str]] = {
    "agent.run": "agent",
    "agent.model.invoke": "generation",
    "agent.model.stream": "generation",
    "agent.tool.call": "tool",
    "agent.memory.retrieve": "retriever",
    "agent.memory.observe": "span",
    "agent.artifact.create": "span",
    "agent.policy.check": "guardrail",
}

#: Harness span kind -> Langfuse observation type, for spans not named above.
KIND_TYPES: Final[dict[str, str]] = {
    "agent": "agent",
    "generation": "generation",
    "tool": "tool",
    "retriever": "retriever",
    "internal": "span",
}


def observation_type(name: str, kind: str) -> str:
    return OBSERVATION_TYPES.get(name) or KIND_TYPES.get(kind, "span")
