"""The memory context sent with the question, after the history a provider's prompt cache keeps
(``middleware.with_question``, audit K1), and the end of that history marked for a cache that
keeps only what a request marks."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tests.support.chat_model import ScriptedChatModel
from trellis.harness.adapters.langgraph import CONTEXT_MESSAGE_ID
from trellis.harness.budget import SPENT
from trellis.harness.history import ASKED_WITH, with_question

MARK = {"type": "ephemeral", "ttl": "5m"}
CONTEXT = SystemMessage("## Facts\n- prefers email", id=CONTEXT_MESSAGE_ID)
TOLD = ASKED_WITH.format(context=CONTEXT.text)


def request(messages: list[Any], kept: list[Any] | None = None, **settings: Any) -> Any:
    return ModelRequest(
        model=ScriptedChatModel(turns=[]),
        messages=messages,
        system_message=SystemMessage("s"),
        tools=[],
        tool_choice=None,
        response_format=None,
        state={"messages": messages if kept is None else kept},
        runtime=None,  # type: ignore[arg-type]
        model_settings=settings,
    )


def chat(answer: AIMessage) -> list[Any]:
    return [
        CONTEXT,
        HumanMessage("a?", id="q1"),
        answer,
        HumanMessage("and b?", id="q2"),
    ]


def test_the_context_comes_with_the_question_and_the_history_is_left_as_it_was() -> None:
    given = chat(AIMessage("7 units.", id="a1"))
    sent = with_question(request(given)).messages
    assert sent[:2] == given[1:3]  # the history, unchanged and unmarked: nothing asked to cache
    assert sent[2].content == [
        {"type": "text", "text": TOLD},
        {"type": "text", "text": "and b?"},
    ]
    assert sent[2].id == "q2" and len(sent) == 3


def test_the_context_is_marked_off_from_the_question_it_comes_with() -> None:
    """Sent in one message, the context says where it ends and that the user's message
    follows: a model read a question after a long context as one more saved note
    (break-and-fix round 1: "I don't see a request, only saved notes")."""
    told: list[Any] = list(
        with_question(request(chat(AIMessage("7 units.", id="a1")))).messages[2].content
    )
    context = told[0]["text"]
    assert context.startswith("<memory_context") and context.rstrip().endswith("</memory_context>")
    assert "not the user's message" in context and told[-1]["text"] == "and b?"


def test_a_cache_that_keeps_only_what_is_marked_gets_the_end_of_the_history_marked() -> None:
    sent = with_question(request(chat(AIMessage("7 units.", id="a1")), cache_control=MARK))
    assert sent.messages[1].content == [{"type": "text", "text": "7 units.", "cache_control": MARK}]
    assert sent.messages[2].content[0] == {"type": "text", "text": TOLD}


def test_the_last_text_part_of_an_answer_in_parts_is_marked() -> None:
    parts: list[Any] = [
        {"type": "text", "text": "one"},
        {"type": "text", "text": "two"},
        {"type": "text"},
    ]
    sent = with_question(request(chat(AIMessage(parts, id="a1")), cache_control=MARK))
    assert sent.messages[1].content == [parts[0], {**parts[1], "cache_control": MARK}, parts[2]]


@pytest.mark.parametrize(
    "answer",
    [
        AIMessage("", id="a1"),  # nothing to mark
        AIMessage([{"type": "text", "text": "x", "cache_control": MARK}], id="a1"),  # marked
        AIMessage("", id="a1", tool_calls=[{"name": "t", "args": {}, "id": "c"}]),  # a call
    ],
    ids=["empty", "marked-already", "a-call"],
)
def test_an_answer_with_nothing_to_mark_or_marked_already_is_left_as_it_is(
    answer: AIMessage,
) -> None:
    sent = with_question(request(chat(answer), cache_control=MARK))
    assert sent.messages[1] is answer


def test_a_result_before_the_question_is_left_as_it_is() -> None:
    result = ToolMessage("7", tool_call_id="c", id="r1")
    given = [CONTEXT, HumanMessage("a?", id="q1"), result, HumanMessage("b?", id="q2")]
    assert with_question(request(given, cache_control=MARK)).messages[1] is result


def test_the_first_question_has_no_history_to_mark() -> None:
    given = [CONTEXT, HumanMessage([{"type": "text", "text": "a?"}], id="q1")]
    sent = with_question(request(given, cache_control=MARK)).messages
    assert sent == [
        HumanMessage([{"type": "text", "text": TOLD}, {"type": "text", "text": "a?"}], id="q1")
    ]


def test_a_spent_budgets_last_word_is_not_the_question() -> None:
    last = HumanMessage(SPENT)
    kept = [CONTEXT, HumanMessage("a?", id="q1"), AIMessage("", id="a1", tool_calls=[])]
    sent = with_question(request([*kept, last], kept=kept)).messages
    assert sent[0].content[0] == {"type": "text", "text": TOLD}
    assert sent[-1] is last


@pytest.mark.parametrize(
    "given",
    [
        [HumanMessage("a?", id="q1")],  # no context
        [HumanMessage("a?", id="q1"), CONTEXT],  # no question after it
    ],
    ids=["no-context", "no-question"],
)
def test_a_request_without_a_context_to_move_is_sent_as_it_is(given: list[Any]) -> None:
    given_request = request(given)
    assert with_question(given_request) is given_request
