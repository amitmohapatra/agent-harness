"""The scenario suite's results per test-plan id: ``python scripts/scenarios/plan_report.py
<junit.xml>...`` reads the JUnit files a run wrote (each test carries its plan ids as the property
``plan``) and prints, per id, how many tests passed, failed (an assertion of the test's own),
broke an invariant (an error at teardown: ``invariants broken``), were skipped (with why) or are
expected to fail (``xfail``, pending a branch)."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def main(paths: list[str]) -> None:
    seen: dict[str, dict[str, str]] = defaultdict(dict)
    for path in paths:
        for case in ET.parse(path).getroot().iter("testcase"):
            name = f"{case.get('classname')}::{case.get('name')}"
            ids = next(
                (p.get("value", "") for p in case.iter("property") if p.get("name") == "plan"), ""
            )
            if case.find("failure") is not None:
                outcome = "fail"
            elif case.find("error") is not None:
                text = (case.find("error").get("message") or "") + (case.find("error").text or "")  # type: ignore[union-attr]
                outcome = "invariant" if "invariants broken" in text else "error"
            elif case.find("skipped") is not None:
                kind = case.find("skipped").get("type", "")  # type: ignore[union-attr]
                outcome = "xfail" if "xfail" in kind else "skip"
            else:
                outcome = "pass"
            for plan_id in (ids or "unplanned").split():
                previous = seen[plan_id].get(name)
                # a test with an error at teardown also has its call's outcome: the worse stays
                if previous in (None, "pass") or outcome in ("fail", "error", "invariant"):
                    seen[plan_id][name] = outcome
    sys.stdout.write(f"{'plan id':<10} pass fail inv  err  skip xfail\n")
    for plan_id in sorted(seen):
        counts = defaultdict(int)
        for outcome in seen[plan_id].values():
            counts[outcome] += 1
        sys.stdout.write(
            f"{plan_id:<10} {counts['pass']:>4} {counts['fail']:>4} {counts['invariant']:>4} "
            f"{counts['error']:>4} {counts['skip']:>4} {counts['xfail']:>5}\n"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
