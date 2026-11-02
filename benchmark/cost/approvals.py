"""What the suite approves when a run pauses to ask a person, as a user reading the request
would: a shell command that only reads (:func:`read_only`). Claude Code and Kimi Code run code
in a shell of their own (``Bash``), which governance asks a person about: the suite answered no
one, so a turn that would compute with a read of the data ended ``PAUSED`` (the gate: Kimi's
files/data 0/6). A command that writes, deletes, installs, reaches the network or cannot be read
for certain is left paused, and counts as a pause.

Read with the standard library only: ``shlex`` splits the command (its quoting, its operators),
``ast`` reads the Python it runs. Anything it does not know is not read-only."""

from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Mapping
from typing import Any, Final

from trellis.contracts import RunStatus

#: The shell tools of the frameworks that have one (Claude Code's and Kimi Code's).
SHELLS: Final = frozenset({"Bash"})
#: The most read-only commands one turn is resumed past (a loop is a pause).
APPROVALS_MAX: Final = 8
#: Programs that only read, each with the arguments that would make it write (any argument
#: starting so).
READERS: Final[Mapping[str, tuple[str, ...]]] = {
    "cat": (),
    "head": (),
    "tail": (),
    "ls": (),
    "wc": (),
    "grep": (),
    "egrep": (),
    "rg": (),
    "cut": (),
    "tr": (),
    "uniq": (),
    "nl": (),
    "column": (),
    "jq": (),
    "echo": (),
    "printf": (),
    "pwd": (),
    "date": (),
    "stat": (),
    "file": (),
    "du": (),
    "df": (),
    "diff": (),
    "cmp": (),
    "basename": (),
    "dirname": (),
    "realpath": (),
    "which": (),
    "true": (),
    "sort": ("-o", "--output"),
    "sed": ("-i", "--in-place"),
    "find": ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fls"),
    "awk": (),
}
PYTHONS: Final = frozenset({"python", "python3"})
#: The shell's operators a read-only command may hold: pipes and sequences (no redirect to a
#: file, no background job).
JOINS: Final = frozenset({"|", "||", "&&", ";"})
#: What a Python program may not call or import to stay read-only: writing, deleting, moving,
#: running other programs, the network, code built at run time.
PY_WRITES: Final = frozenset(
    {
        "write",
        "writelines",
        "write_text",
        "write_bytes",
        "to_csv",
        "to_excel",
        "to_parquet",
        "to_json",
        "to_pickle",
        "to_sql",
        "to_feather",
        "savefig",
        "save",
        "dump",
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "rmtree",
        "mkdir",
        "makedirs",
        "touch",
        "rename",
        "replace",
        "move",
        "copy",
        "copyfile",
        "copytree",
        "chmod",
        "chown",
        "truncate",
        "symlink",
        "link",
        "system",
        "popen",
        "run",
        "call",
        "check_call",
        "check_output",
        "Popen",
        "kill",
        "exec",
        "eval",
        "compile",
        "__import__",
    }
)
PY_MODULES: Final = frozenset(
    {"subprocess", "shutil", "socket", "requests", "urllib", "http", "httpx", "ftplib", "smtplib"}
)
#: A ``sed`` script command that writes a file (``w``, ``W``) or runs one (``e``).
_SED_WRITES: Final = re.compile(r"(?:^|[/;{}])\s*[wWe](?:\s|$)")
#: A heredoc feeding a program its input: the line that runs it, then its body.
_HEREDOC: Final = re.compile(
    r"\A(?P<head>[^\n]*?)<<-?\s*(?P<q>['\"]?)(?P<tag>\w+)(?P=q)[ \t]*\n"
    r"(?P<body>.*?)\n(?P=tag)\s*\Z",
    re.DOTALL,
)


async def approved_reads(agent: Any, result: Any, reviewer: str) -> tuple[Any, list[str]]:
    """``result`` — a run's — resumed approved, as often as it pauses on a command that only
    reads (:func:`read_only`, at most :data:`APPROVALS_MAX` times), and those commands; a pause
    on anything else is left as it is."""
    commands: list[str] = []
    for _ in range(APPROVALS_MAX):
        pending = result.interrupt if result.status is RunStatus.PAUSED else None
        call = pending.tool_call if pending is not None else None
        if pending is None or call is None or not read_only(call.tool, call.args):
            break
        commands.append(str(call.args["command"]))
        result = await agent.resume(pending.interrupt_id, "approve", reviewer=reviewer)
    return result, commands


def read_only(tool: str, args: Mapping[str, Any]) -> bool:
    """Whether a paused call is a shell command (:data:`SHELLS`) that only reads: each of its
    commands a program of :data:`READERS` with none of its writing arguments, or Python whose
    code writes nothing (:func:`python_reads`), joined by pipes and sequences only (an output
    may be joined to another, ``2>&1``, or thrown away, ``2>/dev/null``)."""
    command = args.get("command")
    if tool not in SHELLS or not isinstance(command, str) or not command.strip():
        return False
    heredoc = _HEREDOC.match(command.strip())
    if heredoc is not None:
        head = _words(heredoc["head"])
        return head is not None and _python(head, stdin=heredoc["body"])
    if "`" in command or "$(" in command:
        return False
    words = _words(command)
    if not words or command.strip().count("\n") != sum(w.count("\n") for w in words):
        return False  # a newline between commands (one inside a quoted argument is read)
    commands: list[list[str]] = [[]]
    for word in _joined(words):
        if word in JOINS:
            commands.append([])
        elif set(word) <= set("<>&|;()"):
            return False  # a redirect, a background job, a subshell
        else:
            commands[-1].append(word)
    return all(_reads(c) for c in commands)


#: Where an output may go and nothing is written: thrown away.
DISCARD: Final = "/dev/null"


def _joined(words: list[str]) -> list[str]:
    """``words`` less each ``N>&M`` (one output sent where another goes) and ``N>/dev/null``
    (an output thrown away): nothing written."""
    kept: list[str] = []
    rest = iter(enumerate(words))
    for n, word in rest:
        after = words[n + 1] if n + 1 < len(words) else None
        if (word == ">&" and after is not None and after.isdigit()) or (
            word in (">", ">>") and after == DISCARD
        ):
            if kept and kept[-1].isdigit():
                kept.pop()
            next(rest)
            continue
        kept.append(word)
    return kept


def python_reads(code: str) -> bool:
    """Whether Python ``code`` only reads: it parses, imports none of :data:`PY_MODULES`, calls
    none of :data:`PY_WRITES`, and opens files only to read (``open``'s mode, when given, is
    ``r`` or ``rb``)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            if any(n.split(".")[0] in PY_MODULES for n in names):
                return False
        elif isinstance(node, ast.Call):
            called = node.func
            name = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")
            if name in PY_WRITES:
                return False
            if name == "open" and not _reading(node):
                return False
    return True


def _reading(call: ast.Call) -> bool:
    mode = call.args[1] if len(call.args) > 1 else None
    mode = next((k.value for k in call.keywords if k.arg == "mode"), mode)
    if mode is None:
        return True
    return isinstance(mode, ast.Constant) and mode.value in ("r", "rb", "rt")


def _words(command: str) -> list[str] | None:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:  # unbalanced quotes
        return None


def _reads(words: list[str]) -> bool:
    if not words or "=" in words[0]:
        return False  # empty, or a variable set for it
    program, rest = words[0].rsplit("/", 1)[-1], words[1:]
    if program in PYTHONS:
        return _python(words)
    writes = READERS.get(program)
    if writes is None:
        return False
    if program == "awk" and any("system(" in w or ">" in w or "|" in w for w in rest):
        return False  # a print to a file or a program (a comparison too: read as a pause)
    if program == "sed" and any(_SED_WRITES.search(w) for w in rest if not w.startswith("-")):
        return False
    return not any(w.startswith(writes) for w in rest) if writes else True


def _python(words: list[str], *, stdin: str | None = None) -> bool:
    """``python3 -c CODE``, or ``python3 [-]`` reading ``stdin`` (a heredoc), whose code only
    reads; a script file is not read here, so it is not read-only."""
    if not words or words[0].rsplit("/", 1)[-1] not in PYTHONS:
        return False
    rest = words[1:]
    if stdin is not None:
        return rest in ([], ["-"]) and python_reads(stdin)
    return len(rest) == 2 and rest[0] == "-c" and python_reads(rest[1])
