"""The price table the benchmark prices runs with: the harness's own (``trellis.harness.prices``,
each price with its source and date). A model that is not there is reported with its tokens and
no dollars; ``--price-as`` prices the same tokens at listed models' rates as well, marked as an
estimate (another model's tokenizer counts the same text differently)."""

from __future__ import annotations

from typing import Final

from trellis.harness.prices import PRICES, Price, dollars, price_of

#: The models a local run is also priced as, by default (an estimate: see the module).
REFERENCE: Final = ("claude-sonnet-5-5", "gpt-6-sol")

__all__ = ["PRICES", "REFERENCE", "Price", "dollars", "price_of"]
