# 0004. Model access is provider-neutral: three strategies (route, auth, profile); configure the harness only, memory follows

**Status:** Accepted (2026-10-30)

## Context

The harness reached models one of two ways: through Bifrost (`BIFROST_URL`,
`BIFROST_VIRTUAL_KEY`), or through a provider's own SDK with its own key. Everything else that
speaks OpenAI's chat completions — OpenAI itself, OpenRouter, a LiteLLM proxy, Ollama, vLLM,
llama.cpp, Anthropic's and Gemini's OpenAI-compatible endpoints, Bedrock's — was reachable only
by pretending it was Bifrost, or through `OPENAI_BASE_URL` with no say in how a request is
authenticated. Vertex and Bedrock sign requests with a login (a refreshed Google token, AWS
SigV4), not a key. The memory service did its own model work on a key the harness registered
(`BIFROST_VIRTUAL_KEY`), so it reached only Bifrost too, and chose a model's request shape by
matching its name against a regular expression.

## Decision

1. **Three strategies, each chosen by name from data, never by branching:**
   * **route** — where a model name goes and how it is named there (`ModelGateway`,
     `trellis.harness.gateways`): `openai-compatible` (`OpenAICompatible`: any endpoint, the
     name as given), `bifrost` (`Bifrost(OpenAICompatible)`: the name as `provider/model`, and
     its extras), `provider-sdk` (`ProviderSDK`: LangChain's `init_chat_model`, the provider's
     own SDK and key). `LLM_GATEWAY` names it; unset, `LLM_BASE_URL` means `openai-compatible`
     and no `LLM_BASE_URL` means `provider-sdk`. One registry holds them, and a package adds a
     route of its own under the entry-point group `trellis.gateways`.
   * **auth** — how the endpoint's requests are authenticated: bifrost-sdk's
     `AUTH_STRATEGIES` by `LLM_AUTH`'s name (`bearer` with `LLM_API_KEY`, `google`, `aws`),
     the same strategy on the harness's model clients, its judge and `doctor`'s probe.
   * **profile** — what a model accepts (`trellis.contracts.ModelProfile`: reasoning,
     temperature, output tokens, tool calling), read from LangChain's model profile, never from
     the model's name.
2. **Capabilities, not kinds.** Each route declares what it can do (`Capability`); a feature
   asks `gateway.has(Capability.X)` — the MCP tools, Code Mode, stored prompts, skills, cost,
   attribution, the Claude Agent SDK's Anthropic route are Bifrost's because Bifrost declares
   them. No feature asks which route it is.
3. **Configure the harness only; memory follows.** With `LLM_BASE_URL`, each agent tells the
   memory service its endpoint once per process (`trellis.contracts.Endpoint`: the base URL,
   the route and auth by name, the agent's model as the route names it, its profile, and the
   key only when it is a bearer key — a login sends no secret). The memory service resolves an
   agent's model endpoint in one ordered chain: the agent's registered endpoint, then its own
   `LLM_*` settings, then no model work. Its operator can refuse agents' endpoints
   (`models.llm.accept_agent_endpoints: false`), and it checks an endpoint answers before it
   keeps it (a `localhost` only the harness reaches is refused). The provider SDK route tells
   nothing: the providers' keys are their SDKs', never read by the harness.
4. **The names say what they are, with no aliases.** `LLM_BASE_URL`, `LLM_API_KEY`,
   `LLM_GATEWAY` and `LLM_AUTH` replace `BIFROST_URL` and `BIFROST_VIRTUAL_KEY` in the harness,
   the memory service and bifrost-sdk's docs; the old names are not read.
5. **One failover owner.** No fallback chain between routes: the gateway (Bifrost's virtual
   key's fallbacks) or LangChain's `ModelFallbackMiddleware` owns failover. LiteLLM is an
   endpoint, never a library dependency.

## Consequences

* Any OpenAI-compatible endpoint serves every framework's model names with no code; Vertex and
  Bedrock through their OpenAI-compatible endpoints with a login (`trellis-harness[vertex]`,
  `[bedrock]`), or natively through the provider SDK route (`[anthropic]`, `[gemini]`,
  `[vertex]`, `[bedrock]`).
* Off Bifrost, its capabilities are off and `trellis doctor` lists them; one asked for
  explicitly says which route has it (`missing()`). The Claude CLI, Kimi Code and the OpenAI
  Agents SDK are routed only by a route that declares it (Bifrost), and call their own
  configured endpoints otherwise.
* `trellis doctor` names the route, its auth and where profiles come from, and checks the
  declared route against the endpoint: a `bifrost` route must answer MCP at the endpoint's
  root, and an endpoint that does under `openai-compatible` is said to have more to give.
* A conformance suite (`tests/integration/test_gateway_conformance.py`) runs every registered
  route, auth strategy and profile against stubs of each wire format; a route or strategy
  added without a case fails it.
* A deployment that set `BIFROST_URL` sets `LLM_BASE_URL` and `LLM_GATEWAY=bifrost` (and
  `LLM_API_KEY` for `BIFROST_VIRTUAL_KEY`) at the upgrade.
