"""One ``ReAct`` answer from each provider whose key is set: through its OpenAI-compatible
endpoint (the ``openai-compatible`` route, ``LLM_BASE_URL`` and ``LLM_API_KEY`` as a
deployment sets them) and through its own SDK (the ``provider-sdk`` route).

Opt-in like every live test (``make test-live``), and collected only for the providers whose
key variable is set: a provider without a key is not a skipped test, it is not a test here.
The key is the provider's own (``OPENAI_API_KEY``, ...), read by this test alone; the harness
reads none of them. ``LLM_LIVE_<PROVIDER>_MODEL`` overrides the model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from trellis import Harness, ReAct, Settings
from trellis.contracts import RunStatus


@dataclass(frozen=True)
class Provider:
    name: str
    key_variable: str
    base_url: str
    model: str
    #: the name the provider SDK's route takes (``provider:model``), when its SDK is one the
    #: harness's extras install and it reads ``key_variable`` itself
    native: str | None

    @property
    def chosen_model(self) -> str:
        return os.environ.get(f"LLM_LIVE_{self.name.upper()}_MODEL") or self.model


KNOWN = (
    Provider("openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini", "openai"),
    Provider(
        "anthropic",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com/v1",
        "claude-haiku-4-5",
        "anthropic",
    ),
    Provider(
        "gemini",
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.5-flash",
        None,  # google-genai reads GOOGLE_API_KEY: the endpoint route only
    ),
)
WITH_KEYS = [provider for provider in KNOWN if os.environ.get(provider.key_variable)]


async def _answered(settings: Settings, model: str) -> str:
    graph = ReAct(system_prompt="Reply with exactly: OK", model=model, context_window=32_000)
    async with Harness(config=settings, memory=False, gateway=False) as h:
        result = await h.wrap(graph, id="live-provider").run("Say it.", user="live")
    assert result.status is RunStatus.SUCCESS, result.error
    return str(result.answer)


if WITH_KEYS:

    @pytest.mark.live
    @pytest.mark.parametrize("provider", WITH_KEYS, ids=lambda provider: provider.name)
    async def test_a_provider_answers_on_its_openai_compatible_endpoint(
        provider: Provider,
    ) -> None:
        settings = Settings(
            llm_base_url=provider.base_url, llm_api_key=os.environ[provider.key_variable]
        )
        assert "OK" in (await _answered(settings, provider.chosen_model)).upper()

    NATIVE = [provider for provider in WITH_KEYS if provider.native is not None]
    if NATIVE:

        @pytest.mark.live
        @pytest.mark.parametrize("provider", NATIVE, ids=lambda provider: provider.name)
        async def test_a_provider_answers_through_its_own_sdk(provider: Provider) -> None:
            model = f"{provider.native}:{provider.chosen_model}"
            assert "OK" in (await _answered(Settings(), model)).upper()
