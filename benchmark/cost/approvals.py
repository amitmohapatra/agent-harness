"""What the suite approves when a run pauses to ask a person, as a user reading the request
would: a shell command that only reads (:func:`read_only`), and one that writes only inside the
run's own working folder (:func:`approvable`: "save this file" there). Claude Code and Kimi
Code run code in a shell of their own (``Bash``), which governance asks a person about: the
suite answered no one, so a turn that would compute with a read of the data ended ``PAUSED``
(the gate: Kimi's files/data 0/6). A command that writes outside that folder, deletes or moves
anything, installs, reaches the network or cannot be read for certain is left paused, and
counts as a pause.

Read with the standard library only: ``shlex`` splits the command (its quoting, its operators),
``ast`` reads the Python it runs. Anything it does not know is not read-only."""

from __future__ import annotations

import ast
import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from trellis.contracts import RunStatus

#: The shell tools of the frameworks that have one (Claude Code's and Kimi Code's).
SHELLS: Final = frozenset({"Bash"})
#: The frameworks' own file tools that write one file (its ``file_path``, Kimi's ``path``).
EDITORS: Final = frozenset({"Write", "Edit", "MultiEdit"})
#: Programs that write only the paths they are given (``cp``: its last), with the arguments
#: they may take: inside the run's folder they are a user's "save this file".
WRITERS: Final[Mapping[str, frozenset[str]]] = {
    "mkdir": frozenset({"-p", "-v"}),
    "touch": frozenset(),
    "cp": frozenset({"-p", "-r", "-R", "-v"}),
}
#: Python calls that write a file at the path they are given (their first argument, or the
#: ``Path`` they are called on): inside the run's folder, a saved file.
PY_SAVES: Final = frozenset(
    {"to_csv", "to_excel", "to_parquet", "to_json", "to_pickle", "to_feather", "savefig",
     "save", "write_text", "write_bytes", "mkdir", "makedirs", "touch"}
)  # fmt: skip
#: Python calls that write to a file already opened (checked where it was opened).
PY_HANDLES: Final = frozenset({"write", "writelines", "dump"})
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


async def approved_reads(
    agent: Any, result: Any, reviewer: str, folder: str | Path | None = None
) -> tuple[Any, list[str]]:
    """``result`` — a run's — resumed approved, as often as it pauses on a call the suite
    approves (:func:`approvable` with the run's working ``folder``, at most
    :data:`APPROVALS_MAX` times), and those calls; a pause on anything else is left as it is."""
    commands: list[str] = []
    for _ in range(APPROVALS_MAX):
        pending = result.interrupt if result.status is RunStatus.PAUSED else None
        call = pending.tool_call if pending is not None else None
        if pending is None or call is None or not approvable(call.tool, call.args, folder):
            break
        commands.append(str(call.args.get("command") or f"{call.tool} {_target(call.args)}"))
        result = await agent.resume(pending.interrupt_id, "approve", reviewer=reviewer)
    return result, commands


def read_only(tool: str, args: Mapping[str, Any]) -> bool:
    """Whether a paused call is a shell command (:data:`SHELLS`) that only reads: each of its
    commands a program of :data:`READERS` with none of its writing arguments, or Python whose
    code writes nothing (:func:`python_reads`), joined by pipes and sequences only (an output
    may be joined to another, ``2>&1``, or thrown away, ``2>/dev/null``)."""
    return approvable(tool, args)


def approvable(tool: str, args: Mapping[str, Any], folder: str | Path | None = None) -> bool:
    """Whether the suite approves a paused call: it only reads (:func:`read_only`), or - given
    the run's own working ``folder`` - every path it writes resolves inside that folder: the
    framework's file tool (:data:`EDITORS`) on such a path, or a shell command whose writes
    (:data:`WRITERS`, a redirect to a file, Python's :data:`PY_SAVES` and ``open`` for writing,
    each on a path written out) all land there, after a ``cd`` into it if any. Deleting,
    moving, installing and the network stay pauses."""
    home = Path(folder).resolve() if folder is not None else None
    if tool in EDITORS:
        return home is not None and _inside(_target(args), home, home)
    command = args.get("command")
    if tool not in SHELLS or not isinstance(command, str) or not command.strip():
        return False
    heredoc = _HEREDOC.match(command.strip())
    text, stdin = (heredoc["head"], heredoc["body"]) if heredoc is not None else (command, None)
    commands = _commands(text, home)
    return commands is not None and _approved(commands, stdin, home)


#: One command of a shell line: its words, and the files it redirects its output to.
_Command = tuple[list[str], list[str]]


def _commands(text: str, home: Path | None) -> list[_Command] | None:
    """The line's commands, joined by pipes and sequences; ``None`` for what is not read
    here (a substitution, a newline between commands, a subshell, a background job, a
    redirect to a file when no folder may be written)."""
    if "`" in text or "$(" in text:
        return None
    words = _words(text)
    if not words or text.strip().count("\n") != sum(w.count("\n") for w in words):
        return None  # a newline between commands (one inside a quoted argument is read)
    commands: list[_Command] = [([], [])]
    joined = _joined(words)
    redirect = home is not None
    for n, word in enumerate(joined):
        if word in JOINS:
            commands.append(([], []))
        elif redirect and word in (">", ">>") and n + 1 < len(joined):
            commands[-1][1].append(joined[n + 1])  # a redirect to a file: a write to check
        elif set(word) <= set("<>&|;()"):
            return None  # a redirect, a background job, a subshell
        elif not (redirect and n and joined[n - 1] in (">", ">>")):
            commands[-1][0].append(word)
    return commands


def _approved(commands: list[_Command], stdin: str | None, home: Path | None) -> bool:
    """Whether every command reads, or writes only inside ``home`` from where the line's
    ``cd`` (into it, if any) left it; the last one runs ``stdin`` (a heredoc) if given."""
    at = home
    for n, (said, written) in enumerate(commands):
        if written and not all(_inside(w, at, home) for w in written):
            return False
        if said[:1] == ["cd"]:
            if len(said) != 2 or not _inside(said[1], at, home):
                return False
            at = Path(os.path.normpath(at / said[1])) if at is not None else None
            continue
        if stdin is not None and n == len(commands) - 1:
            ok = _python(said, stdin=stdin, at=at, home=home)
        else:
            ok = _reads(said, at=at, home=home) or _writes(said, at, home)
        if not ok:
            return False
    return True


def _target(args: Mapping[str, Any]) -> str:
    """The one file a framework's file tool writes."""
    return str(args.get("file_path") or args.get("path") or "")


def _inside(path: str, at: Path | None, home: Path | None) -> bool:
    """Whether ``path`` (relative: to ``at``) resolves inside ``home``."""
    if not path or at is None or home is None:
        return False
    where = Path(path).expanduser()
    where = Path(os.path.realpath(where if where.is_absolute() else at / where))
    return where == home or home in where.parents


def _writes(words: list[str], at: Path | None, home: Path | None) -> bool:
    """A program of :data:`WRITERS` whose every path (``cp``: its target) is inside ``home``."""
    if not words or home is None or words[0] not in WRITERS:
        return False
    flags = [w for w in words[1:] if w.startswith("-")]
    paths = [w for w in words[1:] if not w.startswith("-")]
    if not paths or any(f not in WRITERS[words[0]] for f in flags):
        return False
    written = paths[-1:] if words[0] == "cp" else paths
    return (words[0] != "cp" or len(paths) > 1) and all(_inside(p, at, home) for p in written)


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


def python_reads(code: str, *, at: Path | None = None, home: Path | None = None) -> bool:
    """Whether Python ``code`` only reads: it parses, imports none of :data:`PY_MODULES`, calls
    none of :data:`PY_WRITES`, and opens files only to read (``open``'s mode, when given, is
    ``r`` or ``rb``). Given the run's folder (``home``; ``at`` its working directory), a
    :data:`PY_SAVES` call, ``open`` for writing and a write to an opened file
    (:data:`PY_HANDLES`) are allowed too when the path is written out and inside it."""
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
            saved = home is not None and _inside(_saved_to(node) or "", at, home)
            if name in PY_WRITES and not (
                (name in PY_SAVES and saved) or (name in PY_HANDLES and home is not None)
            ):
                return False
            if name == "open" and not (_reading(node) or saved):
                return False
    return True


def _saved_to(call: ast.Call) -> str | None:
    """The path a call writes, when it is written out: its first argument, or the ``Path``
    it is called on (``Path("q3.csv").write_text(...)``)."""
    first = call.args[0] if call.args else None
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    on = call.func.value if isinstance(call.func, ast.Attribute) else None
    if isinstance(on, ast.Call) and getattr(on.func, "id", getattr(on.func, "attr", "")) == "Path":
        inner = on.args[0] if on.args else None
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
            return inner.value
    return None


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


def _reads(words: list[str], *, at: Path | None = None, home: Path | None = None) -> bool:
    if not words or "=" in words[0]:
        return False  # empty, or a variable set for it
    program, rest = words[0].rsplit("/", 1)[-1], words[1:]
    if program in PYTHONS:
        return _python(words, at=at, home=home)
    writes = READERS.get(program)
    if writes is None:
        return False
    if program == "awk" and any("system(" in w or ">" in w or "|" in w for w in rest):
        return False  # a print to a file or a program (a comparison too: read as a pause)
    if program == "sed" and any(_SED_WRITES.search(w) for w in rest if not w.startswith("-")):
        return False
    return not any(w.startswith(writes) for w in rest) if writes else True


def _python(
    words: list[str],
    *,
    stdin: str | None = None,
    at: Path | None = None,
    home: Path | None = None,
) -> bool:
    """``python3 -c CODE``, or ``python3 [-]`` reading ``stdin`` (a heredoc), whose code only
    reads (or saves inside ``home``: :func:`python_reads`); a script file is not read here,
    so it is not approved."""
    if not words or words[0].rsplit("/", 1)[-1] not in PYTHONS:
        return False
    rest = words[1:]
    if stdin is not None:
        return rest in ([], ["-"]) and python_reads(stdin, at=at, home=home)
    return len(rest) == 2 and rest[0] == "-c" and python_reads(rest[1], at=at, home=home)
