"""Whether a call of a framework's own shell (Claude Code's and Kimi Code's ``Bash``) only
reads: :func:`read_only`. Claude Code and Kimi Code ask before they run a shell command; a
command that only reads (``ls``, ``cat``, ``wc``, ``which``, Python that opens files to read)
is one the harness lets run without asking a person, as their own read tools (``Read``,
``Glob``, ``Grep``) run (``adapters.claude``, ``adapters.acp``). Given a working ``folder``,
:func:`approvable` also passes a command whose every write lands inside it (the benchmark's
reviewer approves "save this file" there: ``benchmark.cost.approvals``). A command that writes,
deletes or moves anything, installs, reaches the network or cannot be read for certain is not
read-only: the framework's own permission check decides it, and governance on top.

Read with the standard library only: ``shlex`` splits the command (its quoting, its operators),
``ast`` reads the Python it runs. Read-only is what the text proves: Python may use only the
modules, functions, methods and built-in names listed here (:func:`python_reads`), each one that
only reads or computes; a program only the arguments known to write nothing. Anything else -
a name not listed, a module held under another name, code built at run time - is not read-only,
and pauses."""

from __future__ import annotations

import ast
import builtins
import os
import re
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

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
#: Programs that only read, each with the arguments that would make it write or run a program
#: (any argument starting so; a one-letter one also among other letters: ``sort -no``). ``sed``
#: and ``awk`` run a script: read by :func:`_sed` and :func:`_awk`.
READERS: Final[Mapping[str, tuple[str, ...]]] = {
    "cat": (),
    "head": (),
    "tail": (),
    "ls": (),
    "wc": (),
    "grep": (),
    "egrep": (),
    "cut": (),
    "tr": (),
    "nl": (),
    "column": (),
    "jq": (),
    "echo": (),
    "printf": (),
    "pwd": (),
    "stat": (),
    "du": (),
    "df": (),
    "diff": (),
    "cmp": (),
    "basename": (),
    "dirname": (),
    "realpath": (),
    "which": (),
    "true": (),
    "sort": ("-o", "--output", "--compress-program"),
    "uniq": (),
    "rg": ("--pre", "--hostname-bin", "-z", "--search-zip"),
    "file": ("-C", "--compile", "-z", "-Z", "--uncompress"),
    "date": ("-s", "--set"),
    "find": ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fls"),
    "sed": (),
    "awk": (),
}
PYTHONS: Final = frozenset({"python", "python3"})
#: The shell's operators a read-only command may hold: pipes and sequences (no redirect to a
#: file, no background job).
JOINS: Final = frozenset({"|", "||", "&&", ";"})
#: One ``sed`` command that prints, deletes or stops, at a line or a pattern (``2,10p``,
#: ``/denim/p``, ``$d``, ``5q``), or substitutes and prints (``s/a/b/g``): none writes a file
#: (``w``), runs a program (``e``) or reads its script from elsewhere.
_SED: Final = re.compile(
    r"\s*(?:(?:\d+|\$|/[^/;\n]*/)(?:,(?:\d+|\$|/[^/;\n]*/))?)?!?\s*"
    r"(?:[pdq=]|s([/|#,:@])(?:(?!\1)[^;\n])*\1(?:(?!\1)[^;\n])*\1[gpiI\d]*)\s*"
)


def _listed(names: str) -> frozenset[str]:
    return frozenset(names.split())


#: The modules Python may import and stay read-only, each with what may be taken from it: a
#: function or a class that only reads or computes, a constant. A module is only ever the head
#: of such a name (``os.path.join``), never a value of its own (assigned, passed, kept): so
#: nothing else in it is reached. :data:`PY_SAVES`, :data:`PY_HANDLES` and ``open`` among them
#: are checked where they are called.
PY_MODULES: Final[Mapping[str, frozenset[str]]] = {
    module: _listed(names)
    for module, names in {
        "ast": "literal_eval",
        "base64": "b64encode b64decode",
        "bisect": "bisect bisect_left bisect_right insort",
        "calendar": "monthrange isleap month_name month_abbr day_name day_abbr weekday",
        "collections": "Counter defaultdict OrderedDict deque ChainMap",
        "copy": "deepcopy",
        "csv": "reader DictReader writer DictWriter QUOTE_ALL QUOTE_MINIMAL QUOTE_NONNUMERIC "
        "QUOTE_NONE field_size_limit",
        "datetime": "datetime date time timedelta timezone UTC",
        "decimal": "Decimal ROUND_HALF_UP ROUND_HALF_EVEN ROUND_DOWN ROUND_UP getcontext "
        "InvalidOperation",
        "difflib": "SequenceMatcher get_close_matches",
        "fnmatch": "fnmatch fnmatchcase filter",
        "fractions": "Fraction",
        "functools": "reduce partial lru_cache cache cmp_to_key",
        "glob": "glob iglob",
        "gzip": "open",
        "hashlib": "md5 sha1 sha256 sha512",
        "heapq": "heappush heappop heapify nlargest nsmallest",
        "html": "escape unescape",
        "io": "StringIO BytesIO open",
        "itertools": "chain groupby islice product permutations combinations accumulate count "
        "cycle repeat starmap takewhile dropwhile zip_longest tee pairwise batched compress "
        "filterfalse combinations_with_replacement",
        "json": "load loads dumps dump JSONDecodeError",
        "math": "sqrt floor ceil log log10 log2 exp pow pi e inf nan tau isnan isinf isfinite "
        "isclose fabs fsum prod gcd lcm comb perm factorial sin cos tan asin acos atan atan2 "
        "degrees radians hypot trunc erf dist copysign fmod modf",
        "matplotlib": "use",
        "matplotlib.pyplot": "figure subplots subplot plot bar barh scatter hist pie boxplot "
        "title xlabel ylabel xticks yticks xlim ylim legend grid tight_layout savefig close show "
        "axhline axvline text annotate fill_between colorbar imshow gca gcf clf cla suptitle "
        "errorbar subplots_adjust xscale yscale step stackplot get_cmap margins",
        "numpy": "array arange linspace zeros ones full empty eye mean median sum std var min "
        "max abs sqrt log log10 log2 exp round floor ceil where unique sort argsort argmax "
        "argmin percentile quantile cumsum cumprod diff dot concatenate vstack hstack stack "
        "isnan isinf isfinite nan inf pi e nanmean nansum nanmax nanmin nanstd nanmedian "
        "corrcoef cov histogram polyfit polyval clip average prod asarray float64 int64 int32 "
        "float32 bool_ ndarray count_nonzero any all maximum minimum power sign interp meshgrid "
        "reshape transpose bincount digitize searchsorted set_printoptions isclose allclose "
        "array_equal loadtxt genfromtxt save",
        "numpy.linalg": "norm inv det solve eig lstsq",
        "numpy.random": "seed rand randn randint random choice normal uniform shuffle "
        "permutation default_rng",
        "openpyxl": "load_workbook Workbook",
        "openpyxl.styles": "Font PatternFill Alignment Border Side",
        "openpyxl.utils": "get_column_letter column_index_from_string",
        "operator": "itemgetter add sub mul truediv floordiv mod neg lt le eq ne gt ge and_ or_ "
        "not_ contains",
        "os": "listdir getcwd walk scandir stat lstat getenv sep linesep name cpu_count getpid "
        "makedirs mkdir",
        "os.path": "join exists isfile isdir basename dirname splitext getsize getmtime getctime "
        "abspath realpath expanduser relpath normpath split isabs commonpath sep",
        "pandas": "read_csv read_excel read_json read_table read_parquet read_fwf ExcelFile "
        "DataFrame Series Index MultiIndex Categorical Grouper concat merge to_datetime "
        "to_numeric to_timedelta pivot_table crosstab cut qcut isna notna isnull notnull "
        "date_range Timestamp Timedelta NA NaT unique get_dummies melt json_normalize "
        "set_option option_context",
        "pathlib": "Path PurePath",
        "platform": "python_version machine",
        "pprint": "pprint pformat",
        "random": "random randint choice choices sample shuffle uniform seed gauss randrange",
        "re": "search match fullmatch findall finditer sub subn split escape IGNORECASE "
        "MULTILINE DOTALL VERBOSE ASCII I M S X",
        "scipy": "",
        "scipy.stats": "pearsonr spearmanr linregress ttest_ind ttest_rel ttest_1samp zscore norm "
        "describe skew kurtosis percentileofscore chi2_contingency f_oneway mannwhitneyu "
        "shapiro iqr sem",
        "statistics": "mean median mode stdev pstdev variance pvariance quantiles fmean "
        "geometric_mean harmonic_mean median_low median_high multimode correlation "
        "linear_regression",
        "string": "ascii_letters ascii_lowercase ascii_uppercase digits hexdigits punctuation "
        "whitespace printable capwords",
        "sys": "version version_info argv platform stdin stdout stderr exit executable maxsize "
        "getsizeof byteorder",
        "tabulate": "tabulate",
        "textwrap": "wrap fill shorten dedent indent",
        "time": "time sleep strftime strptime localtime gmtime mktime perf_counter monotonic",
        "unicodedata": "normalize category name",
        "yaml": "safe_load safe_load_all safe_dump",
        "zoneinfo": "ZoneInfo",
    }.items()
}
#: The modules that draw (Matplotlib's figures): of use only to save a file, so only with the
#: run's folder.
PY_DRAWS: Final = frozenset({"matplotlib.pyplot"})
#: The built-in names Python may use. Not among them: what reads a name at run time or builds
#: code (``getattr``, ``setattr``, ``vars``, ``globals``, ``eval``, ``exec``, ``__import__``, ...).
#: ``open`` and ``type`` are checked where they are called.
PY_BUILTINS: Final = _listed(
    "abs all any ascii bin bool bytearray bytes callable chr complex dict dir divmod enumerate "
    "filter float format frozenset hasattr hash hex id int isinstance issubclass iter len list "
    "map max min next oct ord pow print range repr reversed round set slice sorted str sum "
    "tuple zip exit quit __name__ __file__ Exception ValueError KeyError IndexError TypeError "
    "FileNotFoundError OSError IOError StopIteration ZeroDivisionError AttributeError "
    "RuntimeError NameError ImportError ModuleNotFoundError UnicodeDecodeError LookupError "
    "KeyboardInterrupt NotImplementedError AssertionError PermissionError ArithmeticError "
    "EOFError"
)
#: Every built-in name, Python 2's too (``python`` may be one): a name of these that the code
#: sets is still the built-in until it is set.
_BUILTINS: Final = frozenset(dir(builtins)) | _listed(
    "execfile file reload raw_input apply buffer coerce intern unichr unicode long xrange cmp "
    "basestring reduce"
)
#: The attributes Python may read of a value (what a function returned, a variable, a text):
#: each only reads or computes, on whatever object the listed functions can return. One not
#: listed (a name starting ``_``, a column read as ``df.revenue``) is not known, so not read-only.
PY_METHODS: Final = _listed(
    # texts and bytes
    "lower upper title capitalize casefold swapcase strip lstrip rstrip split rsplit splitlines "
    "join startswith endswith find rfind rindex count format format_map zfill ljust rjust center "
    "isdigit isalpha isalnum isnumeric isdecimal isspace isupper islower istitle encode decode "
    "partition rpartition removeprefix removesuffix expandtabs translate maketrans hex "
    # lists, dicts, sets, counters
    "append extend insert pop sort reverse clear index items keys values get setdefault popitem "
    "update add discard union intersection difference symmetric_difference issubset issuperset "
    "isdisjoint fromkeys appendleft popleft most_common elements total from_iterable "
    # opened files, csv readers and writers (a write goes to a handle: checked where it opened)
    "read readline readlines seek tell close closed name mode encoding flush buffer isatty "
    "getvalue writerow writerows writeheader fieldnames line_num "
    # matches, numbers, errors
    "group groups groupdict start end span pattern string search match fullmatch findall "
    "finditer sub subn real imag numerator denominator is_integer bit_length as_integer_ratio "
    "quantize args errno strerror filename lineno msg hexdigest digest ratio major minor micro "
    # dates and times
    "now today utcnow fromisoformat fromtimestamp strptime strftime isoformat date time year "
    "month day hour minute second microsecond weekday isoweekday isocalendar timestamp "
    "total_seconds days seconds tzinfo astimezone utc combine min max week quarter dayofweek "
    "day_name month_name days_in_month normalize tz_localize tz_convert "
    # paths, read only
    "exists is_file is_dir is_absolute is_symlink iterdir glob rglob read_text read_bytes stat "
    "lstat stem suffix suffixes parent parents parts resolve absolute expanduser joinpath "
    "with_suffix with_name with_stem relative_to as_posix cwd home samefile st_size st_mtime "
    "st_ctime st_mode path "
    # tables and arrays (pandas, numpy)
    "shape size ndim dtype dtypes columns T empty loc iloc at iat str dt cat names levels axes "
    "nbytes itemsize flat head tail sample sum mean median std var sem nunique unique "
    "value_counts describe info memory_usage groupby pivot pivot_table melt stack unstack merge "
    "concat assign explode squeeze transpose sort_values sort_index reset_index set_index "
    "reindex droplevel swaplevel isna isnull notna notnull fillna dropna drop drop_duplicates "
    "duplicated ffill bfill interpolate astype round abs clip cumsum cumprod cummax cummin diff "
    "pct_change shift rolling expanding ewm resample nlargest nsmallest idxmax idxmin argmax "
    "argmin argsort quantile corr cov rank where mask isin between any all iterrows itertuples "
    "tolist to_list to_dict to_numpy to_frame to_period to_timestamp to_records to_series "
    "filter map applymap pipe select_dtypes equals first last nth ngroups get_group cumcount "
    "mul div truediv floordiv mod pow dot eq ne lt le gt ge prod contains len slice pad wrap "
    "repeat get_dummies from_dict from_records reshape flatten ravel "
    "item nonzero trace diagonal fill normal uniform integers choice random standard_normal "
    "permutation shuffle cdf pdf ppf sf statistic pvalue slope intercept rvalue stderr "
    "correlation parse sheet_names "
    # figures and workbooks
    "set_title set_xlabel set_ylabel set_xlim set_ylim set_xticks set_yticks set_xticklabels "
    "set_yticklabels legend grid tight_layout annotate text axhline axvline fill_between twinx "
    "tick_params suptitle set_size_inches autofmt_xdate bar barh scatter pie errorbar step "
    "imshow colorbar add_subplot subplots figure xaxis yaxis spines set_visible invert_yaxis "
    "invert_xaxis bar_label margins set_aspect subplots_adjust xlabel ylabel active sheetnames "
    "worksheets iter_rows iter_cols cell value max_row max_column min_row min_column rows row "
    "column coordinate create_sheet font alignment border number_format column_dimensions "
    "row_dimensions width height merge_cells freeze_panes bold column_letter"
)
#: The ``__names__`` Python may read: each a text.
PY_DUNDERS: Final = frozenset({"__version__", "__name__", "__file__", "__doc__"})
#: A ``Path``'s own writes and opens: the path is the ``Path`` they are called on.
PY_PATHS: Final = frozenset({"write_text", "write_bytes", "mkdir", "touch", "open"})
#: Methods that give a table as text, or write it to the file they are given first (``buf``):
#: read-only called with neither.
PY_TEXTS: Final = frozenset({"to_string", "to_markdown", "to_html", "to_latex"})
#: Readers that also fetch an address (``https://...``): read-only on a path written out
#: with no ``://`` in it, or an opened file.
PY_LOCAL: Final = frozenset(
    {"read_csv", "read_excel", "read_json", "read_table", "read_parquet", "read_fwf", "ExcelFile",
     "loadtxt", "genfromtxt"}
)  # fmt: skip
#: pandas' options, named by a text: only how a table is shown (``display.``) may be set.
PY_OPTIONS: Final = frozenset({"set_option", "option_context"})
#: Methods that call the function they are given - or, given a text, the method of that name
#: (pandas): read-only given a ``lambda``, a ``def`` of the code, a listed function, or the
#: name of one of :data:`PY_AGGREGATES`.
PY_APPLIES: Final = frozenset({"apply", "agg", "aggregate", "transform"})
PY_AGGREGATES: Final = _listed(
    "sum mean median min max count std var first last nunique size prod any all cumsum cumprod "
    "cummax cummin rank idxmax idxmin sem"
)
#: Arguments that name a module to load or a program to run (Matplotlib's): never read-only.
PY_KEYWORDS: Final = frozenset({"backend", "usetex"})
#: Methods that are a text's unless the code can hold what the module makes: a ``Path`` (its
#: ``replace`` moves a file).
PY_UNLESS: Final[Mapping[str, str]] = {"replace": "pathlib"}
#: What may be read of a value: :data:`PY_METHODS`, and the checked ones a value has.
_METHODS: Final = (
    PY_METHODS | (PY_SAVES - {"makedirs"}) | {"write", "writelines"} | PY_PATHS
    | PY_TEXTS | PY_APPLIES | PY_UNLESS.keys() | {"plot"}
)  # fmt: skip
#: A pandas option that only sets how a table is shown.
_DISPLAY: Final = re.compile(r"display\.[a-z_.]+")
#: The arguments ``open`` may be given by name.
_OPEN: Final = frozenset({"mode", "encoding", "errors", "newline", "buffering"})
#: A heredoc feeding a program its input: the line that runs it, then its body.
_HEREDOC: Final = re.compile(
    r"\A(?P<head>[^\n]*?)<<-?\s*(?P<q>['\"]?)(?P<tag>\w+)(?P=q)[ \t]*\n"
    r"(?P<body>.*?)\n(?P=tag)\s*\Z",
    re.DOTALL,
)


def read_only(tool: str, args: Mapping[str, Any]) -> bool:
    """Whether a call is a shell command (:data:`SHELLS`) that only reads: each of its
    commands a program of :data:`READERS` with none of its writing arguments, or Python whose
    code writes nothing (:func:`python_reads`), joined by pipes and sequences only (an output
    may be joined to another, ``2>&1``, or thrown away, ``2>/dev/null``)."""
    return approvable(tool, args)


def approvable(tool: str, args: Mapping[str, Any], folder: str | Path | None = None) -> bool:
    """Whether a call only reads (:func:`read_only`), or - given a working ``folder`` (the
    benchmark's reviewer's "save this file there") - every path it writes resolves inside it: the
    framework's file tool (:data:`EDITORS`) on such a path, or a shell command whose writes
    (:data:`WRITERS`, a redirect to a file, Python's :data:`PY_SAVES` and ``open`` for writing,
    each on a path written out) all land there, after a ``cd`` into it if any. Deleting,
    moving, installing and the network stay pauses."""
    home = Path(folder).resolve() if folder is not None else None
    if tool in EDITORS:
        return home is not None and _inside(file_of(args), home, home)
    command = args.get("command")
    if tool not in SHELLS or not isinstance(command, str) or not command.strip():
        return False
    heredoc = _HEREDOC.match(command.strip())
    text, stdin = (heredoc["head"], heredoc["body"]) if heredoc is not None else (command, None)
    if heredoc is not None and not _verbatim(heredoc["body"], heredoc["tag"], heredoc["q"]):
        return False
    commands = _commands(text, home)
    return commands is not None and _approved(commands, stdin, home)


def _verbatim(body: str, tag: str, quoted: str) -> bool:
    """Whether a heredoc's ``body`` reaches the program as it is written, all of it: no line
    of it is the ``tag`` (the shell ends the body there, and runs the rest as commands), and
    when the tag is not quoted, nothing in it is the shell's to expand first (``$(...)``, a
    variable, a backslash it removes)."""
    ends = any(line.strip() == tag for line in body.split("\n"))
    return not ends and (bool(quoted) or not set(body) & set("$`\\"))


#: One command of a shell line: its words, and the files it redirects its output to.
_Command = tuple[list[str], list[str]]


def _commands(text: str, home: Path | None) -> list[_Command] | None:
    """The line's commands, joined by pipes and sequences; ``None`` for what is not read
    here (a substitution or another expansion: :func:`_literal`, a newline between commands, a
    subshell, a background job, a redirect to a file when no folder may be written)."""
    if "`" in text or "$(" in text or not _literal(text):
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


def _literal(text: str) -> bool:
    """Whether the shell runs the words as they are written: outside single quotes no variable
    (``$HOME``; ``$_`` is the last command's last word), and outside any quotes no braces (the
    shell makes ``{-o,out.csv}`` two words) and no ``$'...'`` or ``$"..."`` (quotings ``shlex``
    reads another way)."""
    quote, chars = "", iter(enumerate(text))
    for n, char in chars:
        after = text[n + 1 : n + 2]
        if quote == "'":
            quote = "" if char == "'" else quote
        elif char == "\\":
            next(chars, None)  # the next one is itself
        elif char == quote:
            quote = ""
        elif char in "'\"" and not quote:
            quote = char
        elif (char == "{" and not quote) or (
            char == "$"
            and (
                after.isalnum()
                or after in tuple("_{[@*#?!$-")
                or (after in ("'", '"') and not quote)
            )
        ):
            return False
    return True


def _approved(commands: list[_Command], stdin: str | None, home: Path | None) -> bool:
    """Whether every command reads, or writes only inside ``home`` from where the line's
    ``cd`` (into it, if any) left it; the last one runs ``stdin`` (a heredoc) if given. With no
    ``home`` (only reads), a ``cd`` into any one folder writes nothing."""
    at = home
    for n, (said, written) in enumerate(commands):
        if written and not all(_inside(w, at, home) for w in written):
            return False
        if said[:1] == ["cd"]:
            if len(said) != 2 or (home is not None and not _inside(said[1], at, home)):
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


def file_of(args: Mapping[str, Any]) -> str:
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
    """Whether Python ``code`` only reads, proven from its text: it parses; it imports only
    :data:`PY_MODULES` and takes only their listed names, a module never held as a value; every
    other name it reads is one it set itself or of :data:`PY_BUILTINS`; every attribute it
    reads of a value is of :data:`PY_METHODS`; and each call that could write is one that here
    does not (:data:`_CHECKS`: ``open`` to read, a text's ``replace``, a reader of a local path).
    So the code reaches no function that writes, deletes, moves, runs a program or sends: by a
    name, an alias, a container, a string or an object's insides. Given the run's folder
    (``home``; ``at`` its working directory), a :data:`PY_SAVES` call, ``open`` for writing and
    a write to an opened file (:data:`PY_HANDLES`) are allowed too when the path is written out
    and inside it."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    names = _names(tree, home)
    return names is not None and all(_allowed(node, names, at, home) for node in ast.walk(tree))


@dataclass(frozen=True)
class _Names:
    """What the names of a piece of Python stand for."""

    #: The modules it may use (:data:`PY_MODULES`; :data:`PY_DRAWS` only with a folder).
    known: Mapping[str, frozenset[str]]
    #: The name a module goes by -> the module (``import numpy as np``, ``from os import path``).
    modules: Mapping[str, str]
    #: The name a module's function, class or constant goes by -> the module, and its name there.
    values: Mapping[str, tuple[str, str]]
    #: Every other name the code sets, and those of them only ever set by a plain ``def``.
    bound: frozenset[str]
    defined: frozenset[str]
    #: The expressions that are called, and those an attribute is read of (by ``id``).
    called: frozenset[int]
    owners: frozenset[int]
    #: The expressions only shown (printed, formatted, asked their ``__name__``).
    shown: frozenset[int]

    def module(self, node: ast.AST) -> str | None:
        """The module ``node`` names (``np``, ``os.path``), if it names one."""
        if isinstance(node, ast.Name):
            return self.modules.get(node.id)
        if isinstance(node, ast.Attribute) and (parent := self.module(node.value)) is not None:
            return name if (name := f"{parent}.{node.attr}") in self.known else None
        return None

    def value(self, node: ast.AST) -> tuple[str, str] | None:
        """The module's function, class or constant ``node`` names (``Path``, ``np.mean``)."""
        if isinstance(node, ast.Name):
            return self.values.get(node.id)
        if isinstance(node, ast.Attribute) and (module := self.module(node.value)) is not None:
            return (module, node.attr) if node.attr in self.known[module] else None
        return None


def _names(tree: ast.AST, home: Path | None) -> _Names | None:
    """What the code's names stand for; ``None`` when an import is not of :data:`PY_MODULES`
    (a module, or a name in it, not listed; ``import *``), or a name an import gives is also
    set another way or is a built-in's (so what it stands for is not certain)."""
    known = {m: v for m, v in PY_MODULES.items() if home is not None or m not in PY_DRAWS}
    bound, defined = _set(tree)
    modules: dict[str, str] = {}
    values: dict[str, tuple[str, str]] = {}
    for name, module, attr, imported in _imports(tree):
        target = module if attr is None else f"{module}.{attr}"
        if imported not in known or name in _BUILTINS or name in bound:
            return None
        if target in known:  # a module: under one name, one module
            if modules.setdefault(name, target) != target or name in values:
                return None
        elif (
            attr not in known[module]
            or name in modules
            or values.setdefault(name, (module, attr)) != (module, attr)
        ):
            return None
    return _Names(known, modules, values, bound, defined, *_places(tree, "print" not in bound))


def _imports(tree: ast.AST) -> list[tuple[str, str, str | None, str]]:
    """Each name the code's imports give: the name, the module it is or is taken from, the
    name taken (``None``: the module itself), and the module imported for it."""
    found: list[tuple[str, str, str | None, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:  # `import os.path` gives `os`; `import os.path as p`, os.path
                top = a.name.split(".")[0]
                found.append((a.asname or top, a.name if a.asname else top, None, a.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module and not node.level else ""
            found += [(a.asname or a.name, module, a.name, module) for a in node.names]
    return found


def _set(tree: ast.AST) -> tuple[frozenset[str], frozenset[str]]:
    """The names the code sets other than by an import, and those of them only ever set by a
    plain ``def`` (a decorator may give anything back)."""
    stored, defined = set[str](), set[str]()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            stored.add(node.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            (stored if node.decorator_list else defined).add(node.name)
        elif isinstance(node, ast.ClassDef | ast.ExceptHandler) and node.name:
            stored.add(node.name)
        elif isinstance(node, ast.arg):
            stored.add(node.arg)
    return frozenset(stored | defined), frozenset(defined - stored)


def _places(tree: ast.AST, printing: bool) -> tuple[frozenset[int], ...]:
    """The expressions that are called, those an attribute is read of, and those only shown:
    printed (``printing``: ``print`` is the built-in), formatted, asked their ``__name__``."""
    called, owners, shown = set[int](), set[int](), set[int]()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called.add(id(node.func))
            if printing and isinstance(node.func, ast.Name) and node.func.id == "print":
                shown.update(id(a) for a in node.args)
        elif isinstance(node, ast.Attribute):
            owners.add(id(node.value))
            if node.attr == "__name__":
                shown.add(id(node.value))
        elif isinstance(node, ast.FormattedValue):
            shown.add(id(node.value))
    return frozenset(called), frozenset(owners), frozenset(shown)


def _allowed(node: ast.AST, names: _Names, at: Path | None, home: Path | None) -> bool:
    """Whether one node of the code is read-only: a name, an attribute, a call (a ``match``
    reads attributes by a name the text does not show as one: not read here)."""
    if isinstance(node, ast.Name):
        return _name(node, names)
    if isinstance(node, ast.Attribute):
        return _attribute(node, names)
    if isinstance(node, ast.Call):
        return _calls(node, names, at, home)
    return not isinstance(node, ast.Match)


def _name(node: ast.Name, names: _Names) -> bool:
    """A name read: a module as the head of a listed name only; a listed function that is
    checked where called, ``open`` and ``type`` only called; else a name the code set (and no
    built-in's: until it is set, that is the built-in) or one of :data:`PY_BUILTINS`. No
    ``__name__`` but the two that are texts, read or set."""
    name = node.id
    if name.startswith("__") and name not in PY_BUILTINS:
        return False
    if not isinstance(node.ctx, ast.Load):
        return True
    if name in names.modules:
        return id(node) in names.owners
    if name in names.values:
        return id(node) in names.called or names.values[name][1] not in _CHECKS
    if name in ("open", "type"):
        return id(node) in names.called
    return name in PY_BUILTINS or (name in names.bound and name not in _BUILTINS)


def _attribute(node: ast.Attribute, names: _Names) -> bool:
    """An attribute read (or set): of a module, one listed for it, read only - a module in it
    again only the head of a name; of anything else, one of :data:`PY_METHODS`. One checked
    where called is only called."""
    module = names.module(node.value)
    if names.module(node) is not None:
        return id(node) in names.owners
    if module is None:
        listed = node.attr in _METHODS
    else:
        listed = node.attr in names.known[module] and isinstance(node.ctx, ast.Load)
    return node.attr in PY_DUNDERS or (
        listed and (node.attr not in _CHECKS or id(node) in names.called)
    )


def _calls(call: ast.Call, names: _Names, at: Path | None, home: Path | None) -> bool:
    """Whether a call is read-only where its name alone does not say so (:data:`_CHECKS`).
    Arguments spread from a dict (``**``) or naming a module to load (:data:`PY_KEYWORDS`) are
    not read; nor, for a checked name, arguments spread from a list (which is which is not
    written out)."""
    func = call.func
    if any(k.arg is None or k.arg in PY_KEYWORDS for k in call.keywords):
        return False
    if isinstance(func, ast.Attribute):
        name, method = func.attr, names.module(func.value) is None
    elif isinstance(func, ast.Name):
        name, method = names.values.get(func.id, ("", func.id))[1], False
    else:
        return True
    check = _CHECKS.get(name)
    if check is None or any(isinstance(a, ast.Starred) for a in call.args):
        return check is None
    saved = home is not None and _inside(_written(call, name, method, names) or "", at, home)
    return check(_Call(call, method, names, home is not None, saved))


@dataclass(frozen=True)
class _Call:
    """A call of a checked name."""

    call: ast.Call
    #: Whether it is called on a value (``text.replace``), not a module's or a bare function.
    method: bool
    names: _Names
    #: Whether the run has a folder it may write; and whether this call writes a path written
    #: out inside it.
    folder: bool
    saved: bool

    @property
    def args(self) -> list[ast.expr]:
        return self.call.args

    @property
    def keywords(self) -> list[str | None]:
        return [k.arg for k in self.call.keywords]


def _saves(c: _Call) -> bool:
    return c.saved


def _in_folder(c: _Call) -> bool:
    return c.folder


def _opens(c: _Call) -> bool:
    mode = _mode(c.call, c.method)
    return mode in ("r", "rb", "rt") or (mode is not None and c.saved)


def _replaces(c: _Call) -> bool:
    """A text's ``replace(old, new[, count])``; a ``Path``'s moves a file (:data:`PY_UNLESS`)."""
    used = [*c.names.modules.values(), *(module for module, _ in c.names.values.values())]
    makes = bool(set(PY_UNLESS.values()) & {module.split(".")[0] for module in used})
    return len(c.args) in (2, 3) and not c.keywords and not makes


def _prints(c: _Call) -> bool:
    return not c.args and "buf" not in c.keywords


def _fetches(c: _Call) -> bool:
    """A reader given a path written out that is no address, or a file opened by ``open``."""
    first = c.args[0] if c.args else None
    func = first.func if isinstance(first, ast.Call) else None
    if isinstance(func, ast.Name) and func.id == "open":
        return "open" not in c.names.bound
    return bool(_text(first)) and "://" not in _text(first)


def _displays(c: _Call) -> bool:
    """A pandas option of how a table is shown (pandas reads the name as a pattern)."""
    return not c.keywords and all(_DISPLAY.fullmatch(_text(a)) for a in c.args[::2])


def _applies(c: _Call) -> bool:
    given = c.args[:1] or [
        k.value.elts[-1] if isinstance(k.value, ast.Tuple) and k.value.elts else k.value
        for k in c.call.keywords  # a named aggregation: total=("revenue", "sum")
    ]
    return bool(given) and all(_function(g, c.names) for g in given)


def _renders(c: _Call) -> bool:
    """Matplotlib's ``use``: the backend is a module it loads by name - the built-in one."""
    return [_text(a) for a in c.args] == ["Agg"] and not c.keywords


def _empty(c: _Call) -> bool:
    """openpyxl's ``Workbook()``: one written as it is filled keeps files of its own."""
    return not c.args and not c.keywords


def _shows(c: _Call) -> bool:
    """``type(value)``: the class of a value can make another (a raw file's opens a file to
    write), so it is only shown."""
    return id(c.call) in c.names.shown and len(c.args) == 1


#: The names checked where they are called - so only ever called, never kept or passed on -
#: each with what makes the call read-only.
_CHECKS: Final[Mapping[str, Callable[[_Call], bool]]] = {
    **dict.fromkeys(PY_SAVES, _saves),
    **dict.fromkeys(PY_HANDLES | {"plot"}, _in_folder),
    **dict.fromkeys(PY_TEXTS, _prints),
    **dict.fromkeys(PY_LOCAL, _fetches),
    **dict.fromkeys(PY_OPTIONS, _displays),
    **dict.fromkeys(PY_APPLIES, _applies),
    **dict.fromkeys(PY_UNLESS, _replaces),
    "open": _opens,
    "use": _renders,
    "Workbook": _empty,
    "type": _shows,
}


def _text(node: ast.AST | None) -> str:
    """The text ``node`` writes out, or none."""
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else ""


def _function(node: ast.AST, names: _Names) -> bool:
    """What :data:`PY_APPLIES` may be given: a ``lambda``, a function the code defines, a
    listed one, the name of an aggregate - or a list or dict of those."""
    if isinstance(node, ast.List | ast.Tuple):
        return all(_function(e, names) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(_function(v, names) for v in node.values)
    if isinstance(node, ast.Name) and node.id not in names.values:
        return node.id in names.defined or (node.id in PY_BUILTINS and node.id not in names.bound)
    return (
        isinstance(node, ast.Lambda)
        or _text(node) in PY_AGGREGATES
        or names.value(node) is not None
    )


def _written(call: ast.Call, name: str, method: bool, names: _Names) -> str | None:
    """The path a call writes, when it is written out: its first argument - or, for a
    ``Path``'s own (:data:`PY_PATHS`), the one text that ``Path`` is made of
    (``Path("q3.csv").write_text(...)``: its argument is the content)."""
    written = call.args[0] if call.args else None
    if method and name in PY_PATHS:
        on = call.func.value if isinstance(call.func, ast.Attribute) else None
        if not (isinstance(on, ast.Call) and names.value(on.func) == ("pathlib", "Path")):
            return None
        written = on.args[0] if len(on.args) == 1 and not on.keywords else None
    return _text(written) or None


def _mode(call: ast.Call, method: bool) -> str | None:
    """The mode an ``open`` opens in (a ``Path``'s ``open``: its first argument, else the
    second): ``r`` when none is given, ``?`` when it is not written out; ``None`` when the call
    is given anything else but how the text is decoded (:data:`_OPEN`: no ``opener``)."""
    at = 0 if method else 1
    if len(call.args) > at + 1 or any(k.arg not in _OPEN for k in call.keywords):
        return None
    given = [k.value for k in call.keywords if k.arg == "mode"] + call.args[at:]
    return (_text(given[-1]) or "?") if given else "r"


def _words(command: str) -> list[str] | None:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""  # the shell starts a comment only at a word's start: `a#b` is a word
    try:
        return list(lexer)
    except ValueError:  # unbalanced quotes
        return None


def _reads(words: list[str], *, at: Path | None = None, home: Path | None = None) -> bool:
    if not words or "=" in words[0]:
        return False  # empty, or a variable set for it
    program, rest = _program(words[0]), words[1:]
    if program in PYTHONS:
        return _python(words, at=at, home=home)
    writes = READERS.get(program)
    if writes is None:
        return False
    if program in _SCRIPTED:
        return _SCRIPTED[program](rest)
    if program == "uniq" and len([w for w in rest if not w.startswith("-")]) > 1:
        return False  # its second file is the one it writes
    return not any(
        w.startswith(flag) or (len(flag) == 2 and w[:1] == "-" and w[1:2] != "-" and flag[1] in w)
        for w in rest  # `-o`, also among other letters (`sort -no out.csv`)
        for flag in writes
    )


def _program(word: str) -> str:
    """The program a command's first word runs: by its name, or its full path (``./cat`` is
    a file of the folder, not the program)."""
    return word.rsplit("/", 1)[-1] if word.startswith("/") else word


def _awk(rest: list[str]) -> bool:
    """An ``awk`` program that prints: no file or program it writes to or runs (``>``, ``|``,
    ``system``, ``getline``, a library loaded by ``@``), its only options a separator and
    variables (``-F``, ``-v``: none that reads the program from a file, ``-f``, or edits one in
    place, ``-i``)."""
    options = [w for n, w in enumerate(rest) if w.startswith("-") and rest[n - 1 : n] != ["-v"]]
    return all(w.startswith(("-F", "-v")) for w in options) and not any(
        mark in w for w in rest for mark in ("system", "getline", ">", "|", "@")
    )


def _sed(rest: list[str]) -> bool:
    """A ``sed`` that prints: its options ``-n``, ``-E``, ``-r`` and ``-e`` only (none edits in
    place, ``-i``, even among other letters: ``-ni``), its script made of :data:`_SED` commands."""
    if any(re.fullmatch(r"-[nEr]*[nEre]", w) is None for w in rest if w.startswith("-")):
        return False
    said = [w for w in rest if not w.startswith("-")]
    scripts = [rest[n + 1] for n, w in enumerate(rest[:-1]) if w[:1] == "-" and w[-1] == "e"]
    scripts = scripts or said[:1]
    return bool(scripts) and all(
        _SED.fullmatch(command) for script in scripts for command in script.split(";")
    )


#: The readers that run a script of their own: what it may hold.
_SCRIPTED: Final[Mapping[str, Callable[[list[str]], bool]]] = {"awk": _awk, "sed": _sed}


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
    if not words or _program(words[0]) not in PYTHONS:
        return False
    rest = words[1:]
    if stdin is not None:
        return rest in ([], ["-"]) and python_reads(stdin, at=at, home=home)
    return len(rest) == 2 and rest[0] == "-c" and python_reads(rest[1], at=at, home=home)
