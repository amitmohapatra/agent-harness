"""The ``reasoning`` class: 40 questions with exact answers, half ``dev`` (where a technique's
thresholds are tuned, and the only pool worked examples come from) and half ``test`` (where
it is measured):

* **maths word problems** — 16 of GSM8K's test set (Cobbe et al. 2021, "Training Verifiers to
  Solve Math Word Problems", https://github.com/openai/grade-school-math, MIT licence,
  copyright 2021 OpenAI: ``data/GSM8K_LICENSE``), drawn once with a fixed seed and kept in
  ``data/gsm8k_test_sample.jsonl`` (each with its index in the test set and its final
  answer);
* **dates** — 8 generated: a date some days after another, the weekday of a date;
* **multi-hop** — 8 generated: a chain of three relations between people's ages or counts,
  each a step from the one before;
* **logic** — 8 generated: an order of four people from clues that fix it, who is first,
  last or second.

The generated items are made with a fixed seed and their answers computed (``tests`` check
them again independently). A question asks for one line ``Answer: <answer>``; the answer is
scored by :func:`exact_answer` (a number equal to the expected one, or the expected text).
"""

from __future__ import annotations

import datetime
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any, Final

from trellis.harness.evals import EvalScore

#: Where GSM8K's sample is kept.
GSM8K: Final = Path(__file__).parent / "data" / "gsm8k_test_sample.jsonl"
#: The seed of the generated items.
SEED: Final = 4242
#: What the reasoning set's assistant is told.
SYSTEM: Final = (
    "You answer questions precisely. Work the answer out step by step, then end with one "
    "line: 'Answer: <the answer>' (a number with no units, a date as YYYY-MM-DD, or a name)."
)
#: The splits, every family alternating between them.
SPLITS: Final = ("dev", "test")
NAMES: Final = ("Ana", "Ben", "Cyril", "Dora", "Emil", "Farah", "Gus", "Hana")


def _item(
    question: str, answer: str, kind: str, *, family: str, n: int, source: str
) -> dict[str, Any]:
    return {
        "input": question,
        "metadata": {
            "needed_tools": [],
            "avoid_tools": [],
            "unanswerable": False,
            "no_record": None,
            "forbidden": [],
            "capability": None,
            "answer": answer,
            "answer_kind": kind,
            "family": family,
            "source": source,
            "split": SPLITS[n % 2],
            "task": f"reasoning: {family} {n}",
            "length": "reasoning",
            "mix": False,
        },
    }


def gsm8k() -> list[dict[str, Any]]:
    """GSM8K's sampled items, numbers their answers."""
    rows = [json.loads(line) for line in GSM8K.read_text().splitlines() if line.strip()]
    return [
        _item(
            r["question"],
            r["answer"],
            "number",
            family="maths",
            n=n,
            source=f"gsm8k test #{r['index']}",
        )
        for n, r in enumerate(rows)
    ]


def dates(rng: random.Random) -> list[dict[str, Any]]:
    found = []
    for n in range(8):
        day = datetime.date(2024, 1, 1) + datetime.timedelta(days=rng.randrange(700))
        if n % 2 == 0:
            later = rng.randrange(20, 200)
            question = f"What date is {later} days after {day.isoformat()}?"
            answer = (day + datetime.timedelta(days=later)).isoformat()
            kind = "date"
        else:
            question = f"Which day of the week is {day.isoformat()}?"
            answer, kind = day.strftime("%A"), "text"
        found.append(_item(question, answer, kind, family="dates", n=n, source="generated"))
    return found


def multi_hop(rng: random.Random) -> list[dict[str, Any]]:
    found = []
    for n in range(8):
        a, b, c = rng.sample(NAMES, 3)
        base = rng.randrange(4, 15)
        times, more, less = rng.randrange(2, 4), rng.randrange(2, 9), rng.randrange(1, 4)
        if n % 2 == 0:
            middle = base * times
            last = middle + more
            question = (
                f"{c} is {base} years old. {b} is {times} times as old as {c}. {a} is {more} "
                f"years older than {b}. How old will {a} be in {less} years?"
            )
            answer = last + less
        else:
            middle = base + more
            last = middle * times
            question = (
                f"{c} has {base} marbles. {b} has {more} more marbles than {c}. {a} has {times} "
                f"times as many marbles as {b} and gives {less} away. How many marbles does {a} "
                "have left?"
            )
            answer = last - less
        found.append(
            _item(question, str(answer), "number", family="multi-hop", n=n, source="generated")
        )
    return found


#: What a logic item asks, and the place in the order (tallest first) it asks for.
ASKED: Final = (
    ("Who is the tallest?", 0),
    ("Who is the shortest?", -1),
    ("Who is the second tallest?", 1),
)


def logic(rng: random.Random) -> list[dict[str, Any]]:
    found = []
    for n in range(8):
        order = rng.sample(NAMES, 4)  # tallest first
        clues = [f"{x} is taller than {y}." for x, y in itertools.pairwise(order)]
        rng.shuffle(clues)
        asked, place = ASKED[n % len(ASKED)]
        question = f"{' '.join(clues)} {asked}"
        found.append(_item(question, order[place], "text", family="logic", n=n, source="generated"))
    return found


def items() -> list[dict[str, Any]]:
    """The 40 items: GSM8K's, then the generated ones."""
    rng = random.Random(SEED)
    return [*gsm8k(), *dates(rng), *multi_hop(rng), *logic(rng)]


# --------------------------------------------------------------------------- scoring
ANSWER_LINE: Final = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)
NUMBER: Final = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def final_answer(output: str) -> str:
    """What an answer concludes: after its last ``Answer:``, else its last line — inside a
    ``<response>`` when the model wrote one (Granite's thinking)."""
    text = output.rsplit("<response>", 1)[-1].split("</response>", 1)[0]
    found = ANSWER_LINE.findall(text)
    if found:
        return found[-1].strip()
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1].strip() if lines else ""


def normalized(answer: str, kind: str) -> str:
    """An answer as it is compared: a number's value, else its words, case-blind."""
    if kind == "number":
        numbers = NUMBER.findall(answer)
        if not numbers:
            return ""
        value = float(numbers[-1].replace("$", "").replace(",", ""))
        return f"{value:g}"
    return " ".join(re.findall(r"[\w-]+", answer.casefold()))


def correct(output: str, expected: str, kind: str) -> bool:
    """Whether ``output`` concludes on ``expected``."""
    said = normalized(final_answer(output), kind)
    wanted = normalized(expected, kind)
    if kind == "number":
        return bool(said) and said == wanted
    return bool(wanted) and wanted in said


async def exact_answer(case: Any) -> EvalScore | None:
    """Of a reasoning item, whether the answer it concludes on is the expected one."""
    expected = case.metadata.get("answer")
    if expected is None or not isinstance(case.output, str):
        return None
    kind = case.metadata.get("answer_kind", "text")
    return EvalScore(
        "exact_answer", correct(case.output, expected, kind), final_answer(case.output)
    )
