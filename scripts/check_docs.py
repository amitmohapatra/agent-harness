"""Check the Markdown in this repository: links resolve, and the code a reader may copy is real.

    python scripts/check_docs.py          # every check (make docs-check); exit 1 on any finding
    python scripts/check_docs.py --links  # relative links and anchors only

Links. Every relative link in every Markdown file the repository holds must name a file or
directory that exists, and a ``#fragment`` must name a heading of the target (GitHub's anchor
rules) or an ``<a id>``. External links (``http(s)://``, ``mailto:``) are not fetched.

Snippets. Every ``python`` block must parse (top-level ``await`` allowed, as in a notebook).
Every ``from M import N`` of the platform (``trellis``, ``bifrost_sdk``) or of a framework the
harness supports must import, and ``M`` must have ``N``. A call of a name the platform exports
(``Harness(...)``, ``ReAct(...)``, ``tool(...)``) must pass only keywords its signature takes.
A method called on an object (``agent.run(...)``) must be a method of the platform's classes
or of a framework class the docs name (``FOREIGN``), and its keywords must be ones a method of
that name takes. Every ``make <target>`` in a ``bash`` block must be a Makefile target, and
every ``examples/...``/``scripts/...`` file or ``python -m examples...`` module it runs must
exist.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
#: Markdown that is not this repository's documentation
EXCLUDED = (".claude/", ".venv/", "build/", "typings/")
#: the packages whose names the docs use: the platform's, then the frameworks'
PLATFORM = ("trellis", "bifrost_sdk")
FRAMEWORKS = (
    "langgraph",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "deepagents",
    "agents",
    "claude_agent_sdk",
    "a2a",
    "fastapi",
    "pydantic",
    "opentelemetry",
    "httpx",
)
#: framework and standard-library classes whose methods the docs call on objects
FOREIGN_CLASSES = (
    "agents:Runner",
    "agents:RunState",
    "agents:RunResult",
    "langgraph.pregel:Pregel",
    "langgraph.graph:StateGraph",
    "langchain_core.language_models:BaseChatModel",
    "fastapi:FastAPI",
    "fastapi:Request",
    "httpx:AsyncClient",
    "claude_agent_sdk:ClaudeSDKClient",
    "pydantic:BaseModel",
)
#: methods of builtins and the standard library the docs call (str, dict, list, asyncio...)
FOREIGN_CALLS = frozenset(
    {
        "append",
        "items",
        "keys",
        "values",
        "get",
        "join",
        "split",
        "format",
        "startswith",
        "endswith",
        "strip",
        "lower",
        "upper",
        "replace",
        "update",
        "setdefault",
        "pop",
        "add",
        "extend",
        "create_task",
        "gather",
        "run",
        "sleep",
        "wait_for",
        "timeout",
        "now",
        "isoformat",
        "today",
        "read_text",
        "write_text",
        "encode",
        "decode",
        "dumps",
        "loads",
        "getLogger",
        "basicConfig",
        "setFormatter",
        "StreamHandler",
        "info",
        "warning",
        "exception",
        "result",
        "cancel",
        "set",
        "wait",
        "is_set",
        "aiter_lines",
        "json",
        "post",
        "put",
        "delete",
        "patch",
        "head",
        "stream",
        "from_url",
        # the standard library and OpenTelemetry, as the docs use them
        "signature",
        "wraps",
        "get_tracer",
        "start_as_current_span",
        "get_span_context",
        # the reader's own objects in the snippets (their tools, their services)
        "order",
        "pay",
    }
)
EXAMPLE_MODULE = re.compile(r"python -m (examples(?:\.[\w]+)+)")
EXAMPLE_PATH = re.compile(r"\b((?:examples|scripts)/[\w./-]+\.(?:py|sh|mjs))")
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>```+|~~~+)(?P<info>[^\n]*)$")
LINK = re.compile(
    r"(?<!\!)\[(?:[^\]\[]|\[[^\]]*\])*\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\)"
)
IMAGE = re.compile(r"!\[[^\]]*\]\((?P<target>[^)\s]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(?P<text>.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


@dataclass(frozen=True)
class Block:
    path: str
    line: int
    lang: str
    code: str


# --------------------------------------------------------------------------- files
def markdown_files() -> list[Path]:
    """Tracked Markdown, plus new files not yet added (a check before a commit sees them)."""
    command = ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"]
    out = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return sorted(ROOT / p for p in set(out) if not p.startswith(EXCLUDED) and (ROOT / p).is_file())


def split(text: str) -> tuple[list[tuple[int, str]], list[Block]]:
    """The prose lines (numbered, inline code blanked) and the fenced blocks of a file."""
    prose: list[tuple[int, str]] = []
    blocks: list[Block] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = FENCE.match(lines[i])
        if match:
            fence, info, start = match.group("fence"), match.group("info").strip(), i
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                body.append(lines[i])
                i += 1
            indent = len(match.group("indent"))
            code = "\n".join(line[indent:] if line[:indent].isspace() else line for line in body)
            blocks.append(Block("", start + 1, (info.split() or [""])[0].lower(), code + "\n"))
            i += 1
            continue
        prose.append((i + 1, re.sub(r"`[^`]*`", "", lines[i])))
        i += 1
    return prose, blocks


# --------------------------------------------------------------------------- links
def slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # links keep their text
    text = re.sub(r"<[^>]+>", "", text)  # inline HTML
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)  # punctuation goes; letters, digits, _ and - stay
    return text.replace(" ", "-")


def anchors(path: Path, cache: dict[Path, set[str]]) -> set[str]:
    if path not in cache:
        raw = path.read_text(encoding="utf-8")
        prose, _ = split(raw)
        lines = raw.splitlines()
        seen: dict[str, int] = {}
        found: set[str] = set()
        for number, _line in prose:
            match = HEADING.match(lines[number - 1])
            if not match:
                continue
            base = slug(match.group("text"))
            count = seen.get(base, 0)
            seen[base] = count + 1
            found.add(base if count == 0 else f"{base}-{count}")
        found |= set(re.findall(r"<a\s+(?:name|id)=\"([^\"]+)\"", raw))
        cache[path] = found
    return cache[path]


def check_links(files: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    cache: dict[Path, set[str]] = {}
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        prose, _ = split(path.read_text(encoding="utf-8"))
        for number, line in prose:
            targets = [m.group("target").strip("<>") for m in LINK.finditer(line)]
            targets += [m.group("target") for m in IMAGE.finditer(line)]
            for target in targets:
                if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
                    continue  # http:, https:, mailto:, ...
                file_part, _, fragment = target.partition("#")
                resolved = (path.parent / unquote(file_part)).resolve() if file_part else path
                if file_part and not resolved.exists():
                    findings.append(Finding(rel, number, f"broken link: {target}"))
                    continue
                anchored = fragment and resolved.suffix == ".md" and resolved.is_file()
                if anchored and unquote(fragment).lower() not in anchors(resolved, cache):
                    findings.append(Finding(rel, number, f"no heading for anchor: {target}"))
    return findings


# --------------------------------------------------------------------------- the API surface
@dataclass
class Surface:
    #: every attribute of the platform's classes, and of the framework classes in FOREIGN
    attributes: set[str]
    #: the keywords each method name takes, across those classes (None: it takes **kwargs)
    keywords: dict[str, set[str] | None]
    #: the platform's own method names (their keywords are checked)
    platform_methods: set[str]


def _modules(package: str) -> Iterator[Any]:
    root = importlib.import_module(package)
    yield root
    for info in pkgutil.walk_packages(getattr(root, "__path__", []), prefix=f"{package}."):
        if ".__main__" in info.name or info.name.endswith("worker.__main__"):
            continue
        try:
            yield importlib.import_module(info.name)
        except Exception:
            continue


def _params(fn: Any) -> set[str] | None:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params}


def _add(surface: Surface, cls: type, *, platform: bool) -> None:
    names = {n for n in dir(cls) if not n.startswith("__")}
    if platform and cls.__name__.endswith("API") and "__call__" in vars(cls):
        # an API object called as an attribute (``scope.feedback(...)``: ``FeedbackAPI``)
        called = cls.__name__.removesuffix("API").lower()
        accepted = _params(vars(cls)["__call__"])
        if accepted is None or surface.keywords.get(called, set()) is None:
            surface.keywords[called] = None
        else:
            surface.keywords.setdefault(called, set()).update(accepted)
    surface.attributes |= names | set(getattr(cls, "model_fields", None) or {})
    for name in names:
        member = inspect.getattr_static(cls, name, None)
        fn = member.__func__ if isinstance(member, (classmethod, staticmethod)) else member
        if not callable(fn) or isinstance(fn, type):
            continue
        if platform:
            surface.platform_methods.add(name)
        accepted = _params(fn)
        if accepted is None or surface.keywords.get(name, set()) is None:
            surface.keywords[name] = None
        else:
            surface.keywords.setdefault(name, set()).update(accepted)


def surface() -> Surface:
    found = Surface(set(), {}, set())
    for package in ("trellis.harness", "trellis.runs", "trellis.memory", "trellis.contracts"):
        for module in _modules(package):
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__.startswith(("trellis.", "bifrost_sdk")):
                    _add(found, cls, platform=True)
            source = Path(getattr(module, "__file__", "") or "")
            if source.suffix == ".py":
                text = source.read_text(encoding="utf-8")
                found.attributes |= set(re.findall(r"self\.(\w+)\s*[:=]", text))
    for module in _modules("bifrost_sdk"):
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__.startswith("bifrost_sdk"):
                _add(found, cls, platform=True)
    for spec in FOREIGN_CLASSES:
        module_name, _, name = spec.partition(":")
        _add(found, getattr(importlib.import_module(module_name), name), platform=False)
    return found


# --------------------------------------------------------------------------- snippets
def _imported(node: ast.ImportFrom, where: tuple[str, int]) -> tuple[list[Finding], dict]:
    """The names an ``import from`` brings, checked; the platform's ones, for their calls."""
    findings: list[Finding] = []
    names: dict[str, Any] = {}
    module_name = node.module or ""
    if not module_name.startswith((*PLATFORM, *FRAMEWORKS)):
        return findings, names
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return [Finding(*where, f"cannot import {module_name}: {exc}")], names
    for alias in node.names:
        try:
            value = getattr(module, alias.name)
        except AttributeError:
            try:  # a submodule
                value = importlib.import_module(f"{module_name}.{alias.name}")
            except ImportError:
                findings.append(Finding(*where, f"{module_name} has no {alias.name}"))
                continue
        if module_name.startswith(PLATFORM):
            names[alias.asname or alias.name] = value
    return findings, names


def check_python(block: Block, api: Surface) -> list[Finding]:
    try:
        tree = compile(
            block.code,
            block.path,
            "exec",
            flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            dont_inherit=True,
        )
    except SyntaxError as exc:
        return [Finding(block.path, block.line + (exc.lineno or 0), f"python: {exc.msg}")]
    findings, platform = _imports(block, tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            findings += _call(block, node, platform, api)
    return findings


def _imports(block: Block, tree: ast.AST) -> tuple[list[Finding], dict[str, Any]]:
    """The snippet's imports, checked; the platform names they bring, for their calls."""
    findings: list[Finding] = []
    platform: dict[str, Any] = {}
    for node in ast.walk(tree):
        where = (block.path, block.line + getattr(node, "lineno", 0))
        if isinstance(node, ast.ImportFrom):
            found, names = _imported(node, where)
            findings += found
            platform |= names
        elif isinstance(node, ast.Import):
            for alias in (a for a in node.names if a.name.startswith(PLATFORM)):
                try:
                    importlib.import_module(alias.name)
                except ImportError as exc:
                    findings.append(Finding(*where, f"cannot import {alias.name}: {exc}"))
    return findings, platform


def _call(block: Block, node: ast.Call, platform: dict[str, Any], api: Surface) -> list[Finding]:
    """A call's name and keywords, against the platform's signatures."""
    where = (block.path, block.line + node.lineno)
    given = [k.arg for k in node.keywords if k.arg is not None]
    if isinstance(node.func, ast.Name) and node.func.id in platform:
        target = platform[node.func.id]
        name, accepted = node.func.id, _params(target) if callable(target) else None
    elif isinstance(node.func, ast.Attribute):
        name = node.func.attr
        if name not in api.attributes and name not in FOREIGN_CALLS:
            return [Finding(*where, f"no method {name}() in the API")]
        if name not in api.platform_methods or name in FOREIGN_CALLS:
            return []
        accepted = api.keywords.get(name)
    else:
        return []
    if accepted is None:
        return []
    return [Finding(*where, f"{name}() takes no {k}=") for k in given if k not in accepted]


def check_bash(block: Block, targets: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    for offset, line in enumerate(block.code.splitlines(), start=1):
        command = line.split("#", 1)[0]
        where = (block.path, block.line + offset)
        for piece in re.split(r"&&|\|\||;|\|", command):
            words = piece.split()
            if not words or words[0] != "make":
                continue
            for word in words[1:]:
                if word.startswith("-") or "=" in word or not re.fullmatch(r"[\w.-]+", word):
                    continue  # a flag, a variable
                if word not in targets:
                    findings.append(Finding(*where, f"no make target {word!r}"))
        for path in EXAMPLE_PATH.findall(command):
            if not (ROOT / path).is_file():
                findings.append(Finding(*where, f"no file {path}"))
        for name in EXAMPLE_MODULE.findall(command):
            if not (ROOT / Path(*name.split("."))).with_suffix(".py").is_file():
                findings.append(Finding(*where, f"no example module {name}"))
    return findings


def make_targets() -> set[str]:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Za-z0-9_.-]+):", text, re.MULTILINE))


def check_snippets(files: list[Path]) -> list[Finding]:
    api = surface()
    targets = make_targets()
    findings: list[Finding] = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        prose, blocks = split(path.read_text(encoding="utf-8"))
        for found in blocks:
            block = Block(rel, found.line, found.lang, found.code)
            if block.lang in ("python", "py"):
                findings += check_python(block, api)
            elif block.lang in ("bash", "sh", "shell", "console"):
                findings += check_bash(block, targets)
        raw = path.read_text(encoding="utf-8").splitlines()
        for number, _ in prose:  # example modules and files named in prose
            text = raw[number - 1]
            for name in EXAMPLE_MODULE.findall(text):
                if not (ROOT / Path(*name.split("."))).with_suffix(".py").is_file():
                    findings.append(Finding(rel, number, f"no example module {name}"))
    return findings


def main(argv: list[str]) -> int:
    files = markdown_files()
    findings = check_links(files)
    if "--links" not in argv:
        findings += check_snippets(files)
    for finding in findings:
        print(finding)  # noqa: T201 - the script's output
    checks = "links" if "--links" in argv else "links and snippets"
    print(f"{len(files)} Markdown files, {checks}: {len(findings)} problem(s)")  # noqa: T201
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
