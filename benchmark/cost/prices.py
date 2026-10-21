"""What a model's tokens cost: US dollars per million tokens, each price with its source and
the date it was read. Cached input (a prompt-cache read) is priced apart from the rest of the
input, and so is a cache write where the provider charges one. The cost benchmark prices runs
with it (a run's own ``max_cost`` budget counts what the gateway says a call cost, not this).
``--price-as`` prices the same tokens at listed models' rates as well, marked as an estimate
(another model's tokenizer counts the same text differently).

A gateway model name (``provider/model``, ``openrouter/openai/model``) is looked up by its
last part. A model that is not here has no price (never a guessed one): its tokens are counted,
its dollars are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

ANTHROPIC: Final = "https://platform.claude.com/docs/en/about-claude/pricing (read 2026-10-21)"
#: openai.com is not reachable from the machine the table was made on: the prices are
#: LiteLLM's published price list (``model_prices_and_context_window_backup.json`` in the
#: ``litellm`` 1.106.1 wheel on PyPI, dated 2026-10-20), a secondary source.
OPENAI: Final = (
    "LiteLLM 1.106.1 model_prices_and_context_window_backup.json (2026-10-20; a secondary "
    "source: openai.com was not reachable), read 2026-10-21"
)
LOCAL: Final = "self-hosted on this machine: no per-token price (hardware and power not priced)"


@dataclass(frozen=True)
class Price:
    """Dollars per million tokens."""

    input: float
    cached_input: float
    output: float
    #: a 5-minute cache write, where the provider charges one (``None``: priced as input)
    cache_write: float | None
    source: str


PRICES: Final[dict[str, Price]] = {
    # Anthropic: base input, 5m cache write, cache hit, output
    "claude-opus-5-5": Price(4.00, 0.20, 20.00, 5.00, ANTHROPIC),
    "claude-sonnet-5-5": Price(2.00, 0.10, 10.00, 2.50, ANTHROPIC),
    "claude-haiku-5-5": Price(0.10, 0.01, 0.50, 0.125, ANTHROPIC),  # prompts up to 100k tokens
    # OpenAI: input, cached input, output (no cache-write charge)
    "gpt-6-sol": Price(2.00, 0.20, 10.00, None, OPENAI),
    "gpt-6-luna": Price(0.10, 0.01, 0.50, None, OPENAI),
    "gpt-5.6": Price(4.00, 0.40, 20.00, None, OPENAI),
    "gpt-4.1-nano": Price(0.10, 0.025, 0.40, None, OPENAI),
    # local models: tokens are measured, dollars are not charged
    "granite-3.3-2b": Price(0.0, 0.0, 0.0, None, LOCAL),
}


def price_of(model: str) -> Price | None:
    """The price of a model, by its gateway name's last part; ``None`` when it is not listed."""
    return PRICES.get(model.rsplit("/", 1)[-1])


def dollars(
    price: Price,
    *,
    input_tokens: float,
    output_tokens: float,
    cached: float = 0,
    written: float = 0,
) -> float:
    """What the tokens cost: ``input_tokens`` counts every prompt token, of which ``cached``
    were read from the cache and ``written`` written to it."""
    plain = max(input_tokens - cached - written, 0)
    write = price.input if price.cache_write is None else price.cache_write
    return (
        plain * price.input
        + cached * price.cached_input
        + written * write
        + output_tokens * price.output
    ) / 1_000_000


#: The models a local run is also priced as, by default (an estimate: see the module).
REFERENCE: Final = ("claude-sonnet-5-5", "gpt-6-sol")
