"""Two gateway behaviours that are silent by default.

Both were found by pointing a real gateway at a reasoning model on a free-tier key — the
configuration a first-time user will most likely have — and neither was visible in a test
suite that only scripted well-formed responses.
"""

from __future__ import annotations

import pytest
from tests.support_gateway import FakeGateway, completion
from universal_agent_contracts.errors import ModelError

from universal_agent_harness import BifrostModelClient

# ``Retry-After`` parsing moved to the shared gateway client when this module stopped
# carrying its own copy; its spellings — seconds, HTTP date, absent, unparseable, and the
# delay-in-the-body form neither copy here ever handled — are covered in bifrost-sdk's
# tests. What is still the harness's own is below: the contract it puts on a reply, and
# the breaker it keeps around the client's retries.


async def test_an_exhausted_output_budget_is_an_error_not_a_silent_empty_answer() -> None:
    """A reasoning model can spend the whole budget thinking and return no text.

    The gateway answers 200 with content=null and finish_reason="length". Passing that back
    as a ModelResponse makes a graph node see an empty answer and continue as if the model
    had chosen to say nothing.
    """
    body = {
        "choices": [{"message": {"content": None}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 6, "completion_tokens": 64, "total_tokens": 70},
    }
    with FakeGateway([body]) as gateway:
        client = BifrostModelClient(gateway.url, model="gemini/gemini-3.6-flash")
        with pytest.raises(ModelError, match="output budget was exhausted"):
            await client.invoke("hello")
        await client.aclose()


async def test_a_tool_call_with_no_text_is_still_a_valid_answer() -> None:
    """The guard must not fire on the ordinary "called a tool, said nothing" response."""
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "length",
            }
        ]
    }
    with FakeGateway([body]) as gateway:
        client = BifrostModelClient(gateway.url, model="m")
        response = await client.invoke("hello")
        await client.aclose()
    assert response.tool_calls and response.tool_calls[0]["id"] == "c1"


async def test_a_normal_short_answer_is_untouched() -> None:
    with FakeGateway([completion("ok")]) as gateway:
        client = BifrostModelClient(gateway.url, model="m")
        response = await client.invoke("hello")
        await client.aclose()
    assert response.text == "ok"


async def test_a_rate_limit_does_not_open_the_circuit() -> None:
    """429 is backpressure, not brokenness.

    Counting rate limits toward the breaker turns "slow down" into "stop": on a real paced
    run, 17 rate limits opened the circuit and the following 62 calls failed instantly
    without a request ever leaving the process. The breaker exists for a gateway that is
    down, and a gateway answering 429 is emphatically up.
    """
    limited = (429, {"error": "rate limited"})
    with FakeGateway([limited] * 4 + [completion("recovered")]) as gateway:
        client = BifrostModelClient(
            gateway.url,
            model="m",
            max_retries=0,
            backoff_seconds=0.0,
            circuit_failure_threshold=2,
        )
        for _ in range(4):
            with pytest.raises(ModelError):
                await client.invoke("hi")
        # the circuit would be open by now if 429s counted; the next call must reach the
        # gateway rather than fail locally
        response = await client.invoke("hi")
        await client.aclose()

    assert response.text == "recovered"
    # five requests actually reached the gateway; an open circuit would have stopped the
    # fifth before it was sent
    assert len(gateway.requests) == 5


async def test_a_server_error_still_opens_the_circuit() -> None:
    """The breaker must still do its job for a gateway that is actually broken."""
    with FakeGateway([(500, {"error": "boom"})] * 12) as gateway:
        client = BifrostModelClient(
            gateway.url,
            model="m",
            max_retries=0,
            backoff_seconds=0.0,
            circuit_failure_threshold=2,
        )
        for _ in range(2):
            with pytest.raises(ModelError):
                await client.invoke("hi")
        with pytest.raises(ModelError, match="circuit"):
            await client.invoke("hi")
        await client.aclose()
