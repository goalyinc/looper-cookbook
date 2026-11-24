"""A program of this machine, run to its end or its deadline.

``run_process`` is what the sandboxes that drive a local program share:
``LocalSandbox`` runs each command with it, and ``DockerSandbox`` the
``docker`` program. A provider reached over a network has no use for it.
"""

from __future__ import annotations

import asyncio
import math
import os
import signal
from collections.abc import Mapping, Sequence

from looper_cookbook.sandboxes.base import MAX_OUTPUT_BYTES, CommandResult

__all__ = ["run_process"]


async def run_process(
    argv: Sequence[str],
    *,
    timeout_s: float,
    stdin: str | None = None,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> CommandResult:
    """Runs a program of this machine to its end or its deadline.

    The program gets a process group of its own. At the deadline, or when
    the caller is cancelled, the whole group is killed, so nothing it started
    outlives it. A result past the deadline keeps what was printed before it.

    Args:
        argv: The program and its arguments.
        timeout_s: Seconds it may take.
        stdin: Text it reads on standard input.
        cwd: Its working directory; this process's without it.
        env: Its whole environment; this process's without it.
        max_output_bytes: The most kept of each output stream.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"timeout_s must be a positive number of seconds, not {timeout_s}")
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL if stdin is None else asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    stdout, stderr = bytearray(), bytearray()
    truncated = False

    async def read(stream: asyncio.StreamReader, kept: bytearray) -> None:
        nonlocal truncated
        # Read to the end even when full, so the program never blocks on a full pipe.
        while chunk := await stream.read(65536):
            room = max_output_bytes - len(kept)
            truncated = truncated or len(chunk) > room
            kept.extend(chunk[: max(room, 0)])

    async def write() -> None:
        if stdin is None:
            return
        assert process.stdin is not None
        try:
            process.stdin.write(stdin.encode())
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # The program stopped reading: it exited, or closed its input.
            pass
        finally:
            process.stdin.close()

    exit_code: int | None = None
    try:
        async with asyncio.timeout(timeout_s):
            await asyncio.gather(
                read(process.stdout, stdout), read(process.stderr, stderr), write()
            )
            exit_code = await process.wait()
    except TimeoutError:
        pass
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
    return CommandResult(
        exit_code=exit_code,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
        truncated=truncated,
    )
