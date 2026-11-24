"""A sandbox on this machine, with no isolation.

``LocalSandbox`` runs each command as a process of this machine, in a
temporary directory of its own. It starts at once and needs nothing
installed, so it is the sandbox to write an environment against, and to run
code you trust.

**It isolates nothing.** A command runs as you, and can read and change
whatever you can. Run what a model writes in a ``DockerSandbox`` instead.

A process a command leaves running in the background outlives the sandbox:
nothing tells it from your own processes once its command has returned. A
Docker sandbox ends such a process with its container.

A command does not get this process's environment variables, only its
``PATH``: an API key among them would be readable to anything the model
runs, and from there to its observations and your logs.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
import tempfile

from looper_cookbook.sandboxes.base import CommandResult, Sandbox, SandboxError
from looper_cookbook.sandboxes.process import run_process

__all__ = ["LocalSandbox"]


class LocalSandbox(Sandbox):
    """Commands as processes of this machine, in a temporary directory.

    Attributes:
        directory: The sandbox's working directory, removed when it closes.
    """

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Config:
        """A local sandbox has no settings: every one is the same."""

        async def build(self) -> LocalSandbox:
            # Made in a thread: a slow disk must not hold the event loop.
            directory = await asyncio.to_thread(tempfile.mkdtemp, prefix="looper-sandbox-")
            return LocalSandbox(directory)

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self._closed = False

    async def run(
        self, command: str, *, timeout_s: float, stdin: str | None = None
    ) -> CommandResult:
        if self._closed:
            raise SandboxError("the sandbox is closed, so it runs no command", retryable=False)
        return await run_process(
            ("/bin/sh", "-c", command),
            timeout_s=timeout_s,
            stdin=stdin,
            cwd=self.directory,
            env={"PATH": os.environ.get("PATH", os.defpath), "HOME": self.directory},
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(shutil.rmtree, self.directory, ignore_errors=True)
