"""``named_in``: the tool names a text names as whole words, in one pass."""

from trellis.harness.agent import named_in


def test_whole_words_only() -> None:
    text = "- archive: lookup_order then refund_order (worked 90%)\n## Tools\n- mcp.search"
    names = ["lookup_order", "refund", "refund_order", "mcp.search", "order"]
    assert named_in(text, names) == {"lookup_order", "refund_order", "mcp.search"}
    assert named_in("", names) == set()
