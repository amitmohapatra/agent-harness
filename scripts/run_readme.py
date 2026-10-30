"""Run the README's walkthrough as it is written: its ``python`` blocks, from the
``<!-- walkthrough -->`` marker to ``<!-- /walkthrough -->``, as one program (top-level
``await`` allowed, as the README shows it).

    python scripts/run_readme.py           # offline: a scripted model answers (make examples)
    python scripts/run_readme.py --live    # the environment as it is: LLM_BASE_URL or the
                                           # provider's key (make examples-live)

Offline the walkthrough's model name is served by the examples' scripted model
(``examples._support.offline``) on the route a name takes without Bifrost; everything else —
the harness, ``ReAct``, governance, the pause and the resume — is real. It fails unless the run
ends ``SUCCESS`` after its approval.
"""

from __future__ import annotations

import ast
import asyncio
import re
import sys
from pathlib import Path
from typing import Any

from trellis.contracts import RunStatus
from trellis.harness import gateways

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
#: what the scripted model does, offline: check the stock, reorder, answer
SCRIPT: list[Any] = [
    ("stock", {"sku": "SKU-1"}),
    ("reorder", {"sku": "SKU-1", "qty": 20}),
    "SKU-1 was at 3 units: I ordered 20 more.",
]


def walkthrough(text: str) -> str:
    """The walkthrough's python blocks, in order, as one program."""
    found = re.search(r"<!-- walkthrough\b.*?-->(.*?)<!-- /walkthrough -->", text, re.S)
    if found is None:
        raise SystemExit("README.md has no <!-- walkthrough --> ... <!-- /walkthrough -->")
    blocks = re.findall(r"^```python\n(.*?)^```", found.group(1), re.S | re.M)
    if not blocks:
        raise SystemExit("the README's walkthrough has no python block")
    return "\n".join(blocks)


def offline() -> None:
    """The route a model name takes without Bifrost answers from the script."""
    from examples._support.offline import langchain_model  # noqa: PLC0415 - ROOT on the path

    model = langchain_model(SCRIPT)
    gateways.ProviderSDK.chat_model = lambda self, name, *, effort: model  # type: ignore[method-assign]


async def run(source: str) -> dict[str, Any]:
    code = compile(source, str(README), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    namespace: dict[str, Any] = {"__name__": "__readme__"}
    awaited = eval(code, namespace)
    if awaited is not None:
        await awaited
    return namespace


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(ROOT))
    if "--live" not in argv:
        offline()
    namespace = asyncio.run(run(walkthrough(README.read_text())))
    status = getattr(namespace.get("result"), "status", None)
    if status is not RunStatus.SUCCESS:
        print(f"the walkthrough ended {status}, not SUCCESS", file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
