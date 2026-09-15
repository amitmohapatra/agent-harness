"""Span names and attribute keys (§34). One place, so backends and tests agree."""

from __future__ import annotations

from typing import Final

# -- span names ---------------------------------------------------------------------
AGENT_RUN: Final = "agent.run"
MEMORY_RETRIEVE: Final = "agent.memory.retrieve"
MEMORY_OBSERVE: Final = "agent.memory.observe"
MODEL_INVOKE: Final = "agent.model.invoke"
MODEL_STREAM: Final = "agent.model.stream"
TOOL_CALL: Final = "agent.tool.call"
ARTIFACT_CREATE: Final = "agent.artifact.create"
POLICY_CHECK: Final = "agent.policy.check"

# -- span kinds (mapped onto backend-specific observation types) ---------------------
KIND_AGENT: Final = "agent"
KIND_MODEL: Final = "generation"
KIND_TOOL: Final = "tool"
KIND_RETRIEVAL: Final = "retriever"
KIND_INTERNAL: Final = "internal"

# -- attributes ---------------------------------------------------------------------
AGENT_ID: Final = "agent.id"
AGENT_RUN_ID: Final = "agent.run.id"
AGENT_PARENT_RUN_ID: Final = "agent.parent_run.id"
AGENT_GROUP_ID: Final = "agent.group.id"
AGENT_SKILL: Final = "agent.skill"
AGENT_VERSION: Final = "agent.version"
AGENT_FRAMEWORK: Final = "agent.framework"
AGENT_FRAMEWORK_VERSION: Final = "agent.framework.version"
HARNESS_VERSION: Final = "agent.harness.version"

TENANT_ID: Final = "tenant.id"
WORKSPACE_ID: Final = "workspace.id"
USER_ID: Final = "enduser.id"
THREAD_ID: Final = "thread.id"
SESSION_ID: Final = "session.id"
TURN_ID: Final = "turn.id"
TASK_ID: Final = "task.id"
WORK_ID: Final = "work.id"
REQUEST_ID: Final = "request.id"
CORRELATION_ID: Final = "correlation.id"

STATUS: Final = "status"
RETRY: Final = "retry.attempt"
ERROR_CODE: Final = "error.code"
ERROR_CATEGORY: Final = "error.category"
DURATION_MS: Final = "duration_ms"

# Model attributes follow the OpenTelemetry GenAI semantic conventions, which Langfuse and
# OTLP backends already understand.
MODEL_PROVIDER: Final = "gen_ai.system"
MODEL_NAME: Final = "gen_ai.request.model"
MODEL_RESPONSE_MODEL: Final = "gen_ai.response.model"
MODEL_PROFILE: Final = "gen_ai.request.profile"
MODEL_FINISH_REASON: Final = "gen_ai.response.finish_reasons"
MODEL_INPUT_TOKENS: Final = "gen_ai.usage.input_tokens"
MODEL_OUTPUT_TOKENS: Final = "gen_ai.usage.output_tokens"
MODEL_TOTAL_TOKENS: Final = "gen_ai.usage.total_tokens"
MODEL_COST: Final = "gen_ai.usage.cost"
MODEL_STREAMING: Final = "gen_ai.request.streaming"
MODEL_TTFT_MS: Final = "gen_ai.response.time_to_first_token_ms"
MODEL_FALLBACK: Final = "gen_ai.response.fallback_used"
PROMPT_ID: Final = "gen_ai.prompt.id"
PROMPT_VERSION: Final = "gen_ai.prompt.version"

TOOL_NAME: Final = "tool.name"
TOOL_VERSION: Final = "tool.version"
TOOL_SOURCE: Final = "tool.source"
TOOL_SERVER: Final = "tool.server"
TOOL_CACHED: Final = "tool.cached"
TOOL_IDEMPOTENCY_KEY: Final = "tool.idempotency_key"
TOOL_RESULT_REF: Final = "tool.result.ref"
TOOL_ARGS_SCHEMA: Final = "tool.args.schema"

MEMORY_QUERY_TYPE: Final = "memory.query_type"
MEMORY_EVIDENCE_STATUS: Final = "memory.evidence.status"
MEMORY_TOKEN_ESTIMATE: Final = "memory.token_estimate"
MEMORY_TOKEN_BUDGET: Final = "memory.token_budget"
MEMORY_CACHE_HIT: Final = "memory.cache_hit"
MEMORY_ITEM_COUNT: Final = "memory.item_count"
MEMORY_BUNDLE_ID: Final = "memory.bundle.id"
MEMORY_OBSERVATION_ID: Final = "memory.observation.id"
MEMORY_KIND: Final = "memory.kind"

ARTIFACT_ID: Final = "artifact.id"
ARTIFACT_TYPE: Final = "artifact.type"
ARTIFACT_SIZE: Final = "artifact.size_bytes"

#: Attributes that may carry payloads and are therefore capture-gated everywhere.
INPUT: Final = "input.value"
OUTPUT: Final = "output.value"
