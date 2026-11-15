"""What a framework's own shell runs in place of a command the harness runs for it
(``sandbox.relay``): ``python -I -S _relayed.py <socket> <token>``. Standard library only (it
runs with no site packages). It hands the harness its one-time token over the relay's socket,
writes what the command writes to its own stdout and stderr as it comes, and exits with the
command's exit code; killed (the framework's timeout, a task stopped), it closes the socket,
and the harness kills the command."""

import os
import socket
import struct
import sys

#: A frame's header: the stream (1, 2; 0: the exit code) and the length of what follows.
HEADER = struct.Struct(">BI")
#: The exit when the harness ended the connection before the command's end.
LOST = 125


def main(path: str, token: str) -> int:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as relay:
        relay.connect(path)
        relay.sendall(token.encode() + b"\n")
        frames = relay.makefile("rb")
        while True:
            head = frames.read(HEADER.size)
            if len(head) < HEADER.size:
                return LOST
            fd, size = HEADER.unpack(head)
            data = frames.read(size)
            if fd == 0:
                return int(data)
            while data:
                data = data[os.write(fd, data) :]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
