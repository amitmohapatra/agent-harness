"""``named_in``: the tool names a text names as whole words, in one pass."""

from trellis.harness.agent import named_in


def test_whole_words_only() -> None:
    text = "- archive: lookup_order then refund_order (worked 90%)\n## Tools\n- mcp.search"
    names = ["lookup_order", "refund", "refund_order", "mcp.search", "order"]
    assert named_in(text, names) == {"lookup_order", "refund_order", "mcp.search"}
    assert named_in("", names) == set()


def test_a_name_at_the_end_of_a_sentence_is_still_the_name() -> None:
    """A name may hold a dot or a colon, and so does the punctuation after it: a skill's step
    "Call fetch_consent_message." names the tool, as "then: create_allocation:" does."""
    names = ["fetch_consent_message", "create_allocation", "mcp.search", "other"]
    text = "1. Call fetch_consent_message.\n2. then create_allocation: with the store.\nmcp.search."
    assert named_in(text, names) == {"fetch_consent_message", "create_allocation", "mcp.search"}
    assert named_in("fetch_consent_messages.", names) == set()  # a longer word is another name


def test_the_status_is_said_once_with_what_to_do_about_it() -> None:
    """The memory service ends its context with what it holds of the question; the harness's
    note says the same in words, with what to do: the service's line goes, the note stays."""
    from trellis.harness.agent import STATUS, told_once

    context = f"## Profile\n- Priya{STATUS}INCOMPLETE"
    assert told_once(context, "## Memory\nMemory covers only part ...") == "## Profile\n- Priya"
    assert told_once(context, None) == context
    assert told_once("## Profile\n- Priya", "## Memory\n...") == "## Profile\n- Priya"
    later = f"## Profile{STATUS}INSUFFICIENT\n\n## Tail\nx"
    assert told_once(later, "note") == "## Profile\n\n## Tail\nx"
