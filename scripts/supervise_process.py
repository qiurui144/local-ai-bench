#!/usr/bin/env python3
"""Restart a child command when it exits.

This is intentionally small and generic. It is used for local benchmark helper
services that must be process-recycled during long runs to release native GPU
runtime state.
"""

from __future__ import annotations

import argparse
import subprocess
import time


def log(msg: str) -> None:
    print(f"[supervisor {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="child")
    parser.add_argument("--restart-delay", type=float, default=2.0)
    parser.add_argument("--max-restarts", type=int, default=0, help="0 means unlimited")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("child command is required after --")

    restarts = 0
    while True:
        log(f"starting {args.name}: {subprocess.list2cmdline(cmd)}")
        proc = subprocess.Popen(cmd)
        rc = proc.wait()
        log(f"{args.name} exited rc={rc}")
        restarts += 1
        if args.max_restarts and restarts > args.max_restarts:
            return int(rc)
        time.sleep(max(0.0, args.restart_delay))


if __name__ == "__main__":
    raise SystemExit(main())
