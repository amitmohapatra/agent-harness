# The live suite with Bifrost optional, 2026-11-15

Owner's direction: Bifrost is optional and the harness calls models directly. A test that
needs only a real model runs at the model's own endpoint when no gateway is configured, and
through Bifrost when one is. Only a test of one of Bifrost's own features needs Bifrost.

Every run used `claude-haiku-5-5` only. Spend was counted from each response's usage at
`benchmark.cost.prices` (see [Spend](#spend)).

## How each test that skipped for the gateway is classed

`needs_gateway` (Bifrost's own features) and `needs_model` (a real model, through the gateway
or directly) are in `tests/live/conftest.py`. The Claude Code CLI only speaks Anthropic's API,
so its tests use `needs_claude_model`: the gateway's Anthropic route, or `ANTHROPIC_API_KEY`.

| test | class | why |
|---|---|---|
| `test_gateway.py::test_react_calls_a_tool_and_answers_in_the_schema` | (b) | `ReAct` with a model name on the deployment's route |
| `test_gateway.py::test_langgraph_with_a_model_pointed_at_the_endpoint` | (b) | a framework's own OpenAI client at `LLM_BASE_URL` |
| `test_gateway.py::test_openai_agents_with_a_model_pointed_at_the_endpoint` | (b) | the same, for the OpenAI Agents SDK |
| `test_gateway.py::test_the_gateway_lists_the_mcp_tools_the_key_allows` | (a) | MCP tools a virtual key allows |
| `test_live_tool_choice.py::test_tool_coverage_of_function_tools` ×4 | (b) | function tools and `ReAct` only, with no MCP |
| `test_live_tool_choice.py::test_mcp_tools_reach_the_model_as_their_servers_list_them` | (a) | MCP clients declared in the gateway, and Code Mode |
| `test_live_hooks.py::test_react_model_calls_go_through_its_middleware` | (b) | the middleware of `ReAct` around real model calls |
| `test_live_hooks.py::test_a_hook_denies_and_rewrites_mcp_calls_through_the_gateway` | (a) | the call it rewrites goes to a gateway MCP client |
| `test_live_claude.py::test_claude_runs_on_the_live_model_and_resumes_its_session` | (b) | the real CLI on Anthropic's API |
| `test_live_kimi.py::test_kimi_runs_on_the_live_model_and_resumes_its_session` | (b) | Kimi's own `KIMI_MODEL_*` names the endpoint |
| `test_live_evals.py::test_a_real_judge_model_grades_the_answers` | (b) | the judge on the deployment's route |
| `test_live_hitl.py::test_a_react_agent_reads_a_result_from_outside` | (b) | `ReAct` with a model name, and agent-runs |
| `test_live_code_mode.py` (1) | (a) | Code Mode |
| `test_live_frameworks.py::test_a_target_runs_with_memory_and_tools` ×6 | (a) | asserts that the toolbox holds exactly the one MCP tool its virtual key allows |
| `test_live_gateway_features.py` (10) | (a) | Code Mode, gateway tools left off completions, stored prompts, the identity header sent to a declared MCP client, Agent Mode, the gateway's `/mcp` with and without a key |

## How it ran

- **Bifrost:** the real gateway, v2.5.1, from the official image `maximhq/bifrost`. The npm
  wrapper `@maximhq/bifrost` 1.6.3 downloads its binary from `downloads.getmaxim.ai`, and this
  machine's network refuses that host. The config had the Anthropic provider (the key comes
  from an environment variable), the four `mcp_fixture.py` clients declared, and inference
  auth on. Current Bifrost closes `/api` until dashboard auth is configured, and even then it
  refuses a virtual key on `/api/mcp/*`, `/api/mcp-logs` and `/api/logs`. The suite and the
  harness read those routes with a virtual key or none, so `client.whitelisted_routes`
  listed them. That is an operator's setting, and this deployment needs it.
- **Memory service and agent-runs:** private instances in their own `livedirect_*` databases.
  Docker ran the sandbox images and the OpenTelemetry collector image.
- **Direct path:** with no gateway, `LLM_BASE_URL` was Anthropic's OpenAI-compatible API with
  the Anthropic key. The Claude CLI used `ANTHROPIC_API_KEY`. A local meter stood in front of
  Anthropic so that every response's usage was counted.

## Results

| run | tests | passed | failed | skipped |
|---|---|---|---|---|
| whole `tests/live` through Bifrost, run 1 | 156 | 149 | 7 | 0 |
| whole `tests/live` through Bifrost, run 2 (after the fixes below) | 156 | 155 | 1 | 0 |
| the (b) tests directly, with no gateway | 12 | 12 | 0 | — |

The one failure left is `test_every_code_mode_call_of_a_react_run_goes_through_the_bridge`. It
failed twice in a row and passed in run 1. On its last step the test forces a reply with no
tool. Haiku's reply was 89 thinking tokens of 91, with no text, so the harness ended the run
`ERROR` (no answer). The root cause is not established.

### What the live runs found

- **Product:** a `ReAct` whose answer is a schema could not answer after a stall.
  LangChain forces a tool call for a structured answer, so the "no tool" last call still
  offered every tool, and Haiku called the stalled tool again. The last call now offers only
  the schema's tool (`middleware._answer_now`), as the budget's last call already did.
- **Product:** with no gateway, `llm_judge` sent `temperature`, which Anthropic refuses for
  this model. It now follows the model's profile (`evals.LLMJudge`).
- **Tests out of date with the product.** None of these weakens an assertion:
  - `proof.recorded` read the journal at the end of the run. A checkpointed graph drops a
    step's records once the graph has passed that step, so the check now reads every record
    filed during the run.
  - `ModelError`'s code is `MODEL_ERROR`.
  - The Claude CLI's sessions are kept in the run's journal store, and its context goes with
    the prompt, not the system prompt.
  - The memory service's transcript also keeps the assistant's tool-call messages.
  - `sign` is a write, so it asks for approval before it waits for the signature.
  - Bifrost 2.5 adds an empty `required` to a tool's schema.
  - The sandbox leak check now uses the tenant actually tested and ignores sandboxes that were
    there before the test began.
  - The hooks MCP test now declares its own catalog instead of relying on an earlier test's.

## Spend

`claude-haiku-5-5` at $0.10 per million input tokens, $0.01 per million cached and $0.50 per
million output tokens. Usage came from each response, as the meter recorded it. The total for
every run here was about $0.30.
