"""Whether a call of a framework's own shell only reads (``governance.shell.read_only``): what
the harness lets Claude Code and Kimi Code run without asking a person; and, given a working
folder, whether its every write lands there (``approvable``: the benchmark's reviewer)."""

from __future__ import annotations

import pytest

from trellis.harness.governance.shell import approvable, read_only

READS = [
    "cat /data/sales.csv | head -50",
    "wc -l sales.csv && head -3 sales.csv",
    "grep -c denim sales.csv; sort -t, -k3 -n sales.csv | tail -5",
    "awk -F, '{s+=$4} END {print s}' sales.csv",
    "sed -n '2,10p' sales.csv",
    "find . -name '*.csv'",
    """python3 -c "import csv; print(sum(float(r['revenue']) for r in csv.DictReader(open('s.csv'))))\"""",
    "python3 - <<'EOF'\nimport pandas as pd\ndf = pd.read_csv('s.csv')\nprint(df.revenue.sum())\nEOF",
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
