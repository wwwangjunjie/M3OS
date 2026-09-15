"""Execute a command inside a fail-closed Landlock filesystem sandbox.

The report process gets read access only to explicitly listed runtime paths and
write access only to its disposable workspace.  Landlock restrictions are
inherited by Bash commands and all child/sub-agent processes.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import platform
import sys
from pathlib import Path


LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
LANDLOCK_RULE_PATH_BENEATH = 1

ACCESS_EXECUTE = 1 << 0
ACCESS_WRITE_FILE = 1 << 1
ACCESS_READ_FILE = 1 << 2
ACCESS_READ_DIR = 1 << 3
ACCESS_REMOVE_DIR = 1 << 4
ACCESS_REMOVE_FILE = 1 << 5
ACCESS_MAKE_CHAR = 1 << 6
ACCESS_MAKE_DIR = 1 << 7
ACCESS_MAKE_REG = 1 << 8
ACCESS_MAKE_SOCK = 1 << 9
ACCESS_MAKE_FIFO = 1 << 10
ACCESS_MAKE_BLOCK = 1 << 11
ACCESS_MAKE_SYM = 1 << 12
ACCESS_REFER = 1 << 13
ACCESS_TRUNCATE = 1 << 14

READ_ACCESS = ACCESS_EXECUTE | ACCESS_READ_FILE | ACCESS_READ_DIR
WRITE_ACCESS_V1 = (
    ACCESS_WRITE_FILE
    | ACCESS_REMOVE_DIR
    | ACCESS_REMOVE_FILE
    | ACCESS_MAKE_CHAR
    | ACCESS_MAKE_DIR
    | ACCESS_MAKE_REG
    | ACCESS_MAKE_SOCK
    | ACCESS_MAKE_FIFO
    | ACCESS_MAKE_BLOCK
    | ACCESS_MAKE_SYM
)

PR_SET_NO_NEW_PRIVS = 38
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


def _syscall(libc: ctypes.CDLL, number: int, *args: object) -> int:
    result = int(libc.syscall(number, *args))
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def _landlock_abi(libc: ctypes.CDLL) -> int:
    return _syscall(
        libc,
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.c_void_p(),
        ctypes.c_size_t(0),
        ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION),
    )


def _handled_access(abi: int) -> int:
    handled = READ_ACCESS | WRITE_ACCESS_V1
    if abi >= 2:
        handled |= ACCESS_REFER
    if abi >= 3:
        handled |= ACCESS_TRUNCATE
    return handled


def _add_path_rule(
    libc: ctypes.CDLL,
    ruleset_fd: int,
    path: Path,
    allowed_access: int,
) -> None:
    resolved = path.expanduser().resolve(strict=True)
    fd = os.open(resolved, os.O_PATH | os.O_CLOEXEC)
    try:
        attr = PathBeneathAttr(allowed_access=allowed_access, parent_fd=fd)
        _syscall(
            libc,
            SYS_LANDLOCK_ADD_RULE,
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(attr),
            ctypes.c_uint(0),
        )
    finally:
        os.close(fd)


def restrict_filesystem(read_paths: list[Path], write_paths: list[Path]) -> int:
    """Apply a read allowlist and a narrower write allowlist to this process."""
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("The Landlock report launcher currently supports x86_64 only.")

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        abi = _landlock_abi(libc)
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
            raise RuntimeError("Landlock is unavailable; refusing to run Kimi Code unsandboxed.") from exc
        raise

    handled = _handled_access(abi)
    ruleset_attr = RulesetAttr(handled_access_fs=handled)
    ruleset_fd = _syscall(
        libc,
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        ctypes.c_uint(0),
    )
    try:
        readonly_allowed = READ_ACCESS & handled
        writable_allowed = handled
        seen: set[Path] = set()
        for path in read_paths:
            resolved = path.expanduser().resolve(strict=True)
            if resolved in seen:
                continue
            _add_path_rule(libc, ruleset_fd, resolved, readonly_allowed)
            seen.add(resolved)
        for path in write_paths:
            resolved = path.expanduser().resolve(strict=True)
            _add_path_rule(libc, ruleset_fd, resolved, writable_allowed)

        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        _syscall(
            libc,
            SYS_LANDLOCK_RESTRICT_SELF,
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint(0),
        )
    finally:
        os.close(ruleset_fd)
    return abi


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read", action="append", default=[])
    parser.add_argument("--write", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if not args.write:
        parser.error("at least one --write path is required")

    restrict_filesystem(
        read_paths=[Path(item) for item in args.read],
        write_paths=[Path(item) for item in args.write],
    )
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    sys.exit(main())
