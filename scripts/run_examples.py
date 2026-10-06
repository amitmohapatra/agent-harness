"""Run every example, several at once, and fail if any fails.

    python scripts/run_examples.py          # offline: the services' variables unset (make examples)
    python scripts/run_examples.py --live   # the environment as it is (make examples-live)
    python scripts/run_examples.py 02_way1  # only the examples whose path contains this

Each example is ``examples/<NN_group>/<name>.py``, run as ``python -m examples.<NN_group>.<name>``
from the repository root (``examples/_support`` is what they share). Offline, the variables
that name a service are removed from each example's environment, so every one runs on its
scripted model, memory service and gateway. Standard library only.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: the variables that name a service: unset offline
SERVICES = (
    "BIFROST_URL",
    "BIFROST_VIRTUAL_KEY",
    "MEMORY_URL",
    "RUNS_URL",
    "TRELLIS_API_KEY",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "SANDBOX",
)
#: the most one example may take, in seconds
TIMEOUT = 180


def examples(only: list[str]) -> list[Path]:
    found = sorted((ROOT / "examples").glob("[0-9][0-9]_*/*.py"))
    return [p for p in found if not only or any(o in str(p) for o in only)]


def module(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def run(path: Path, environ: dict[str, str]) -> tuple[Path, bool, float, str]:
    started = time.monotonic()
    command = [sys.executable, "-m", module(path)]
    try:
        done = subprocess.run(
            command,
            cwd=ROOT,
            env=environ,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
        ok, output = done.returncode == 0, done.stdout + done.stderr
    except subprocess.TimeoutExpired as exc:
        ok, output = False, f"timed out after {TIMEOUT}s\n{exc.stdout or ''}{exc.stderr or ''}"
    return path, ok, time.monotonic() - started, output


def main(argv: list[str]) -> int:
    live = "--live" in argv
    only = [a for a in argv if not a.startswith("--")]
    environ = dict(os.environ)
    if not live:
        for name in SERVICES:
            environ.pop(name, None)
    environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT / "src"), environ.get("PYTHONPATH", "")) if p
    )
    paths = examples(only)
    workers = int(os.environ.get("EXAMPLES_JOBS") or min(8, (os.cpu_count() or 2) * 2))
    started = time.monotonic()
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, ok, seconds, output in pool.map(lambda p: run(p, environ), paths):
            print(f"{'ok  ' if ok else 'FAIL'} {seconds:5.1f}s {path.relative_to(ROOT)}")  # noqa: T201
            if not ok:
                failed.append((path, output))
    for path, output in failed:
        print(f"\n===== {path.relative_to(ROOT)}\n{output[-4000:]}")  # noqa: T201
    total = time.monotonic() - started
    print(f"{len(paths) - len(failed)}/{len(paths)} examples passed in {total:.0f}s")  # noqa: T201
    return 1 if failed or not paths else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
