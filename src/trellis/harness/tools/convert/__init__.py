"""Harness tools as each framework's own tool type, one module per format.

Every converted tool does the same thing when the framework calls it: hand the arguments to
the bridge (``tools.bridge.call``) and give the framework the result as text. Nothing else
differs between formats.
"""

from __future__ import annotations

import json
from typing import Any


def text_of(output: Any) -> str:
    """A tool result as the text a model reads."""
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, default=str)
    except (TypeError, ValueError):
        return str(output)
