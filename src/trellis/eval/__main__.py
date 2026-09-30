"""``python -m trellis.eval gate [options]`` — the regression gate CI runs."""

from __future__ import annotations

import sys

from trellis.eval.gate import main

COMMANDS = {"gate": main}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"usage: python -m trellis.eval {{{','.join(COMMANDS)}}} [options]")
        sys.exit(2)
    sys.exit(COMMANDS[sys.argv[1]](sys.argv[2:]))
