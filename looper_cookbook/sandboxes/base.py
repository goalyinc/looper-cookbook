"""The sandbox contract.

A ``Sandbox`` runs shell commands in a working directory of its own, until
it is closed. An environment runs its tools' commands in one, and a
verifier its checks. A new provider is a subclass with ``run`` and
``close``, in a module of this package: ``local.py`` and ``docker.py`` are
here, and a hosted sandbox service or a cluster is a module each.

Files go through ``run`` too. ``write_file`` pipes a file's text to ``cat``
inside the sandbox, so a provider needs no file transfer of its own, and one
that has a faster way overrides it.

Two kinds of failure stay apart. A command that fails, or runs out of time,
is a ``CommandResult``: a model or a grader reads it. A sandbox that cannot
run commands at all raises ``SandboxError``, whose ``retryable`` says
whether a new sandbox may do better.
"""

from __future__ import annotations

import abc
import dataclasses
import shlex
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

__all__ = ["MAX_OUTPUT_BYTES", "CommandResult", "Sandbox", "SandboxError"]

# The most a sandbox keeps of each of a command's output streams.
MAX_OUTPUT_BYTES = 1 << 20
# A file's write is a short command; this is its deadline.
_WRITE_TIMEOUT_S = 60.0


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class CommandResult:
    """What one command did.

    Attributes:
        exit_code: The command's exit status; ``None`` when it ran out of
            time and was killed.
        stdout: What it wrote to standard output, as text.
        stderr: What it wrote to standard error, as text.
        truncated: A stream was longer than the sandbox keeps, and lost its
            end.
    """

    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    truncated: bool = False

    @property
    def timed_out(self) -> bool:
        return self.exit_code is None


class SandboxError(RuntimeError):
    """A sandbox that could not start, or could not run a command at all.

    Attributes:
        retryable: Whether the same work may succeed in a new sandbox. A
            container that died is retryable; an image that does not exist,
            or a Docker daemon that is not running, is not.
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class Sandbox(abc.ABC):
    """A place that runs shell commands, with a working directory of its own."""

    @runtime_checkable
    class Config(Protocol):
        async def build(self) -> Sandbox:
            """Starts a new sandbox. Whoever builds it closes it."""
            ...

    @abc.abstractmethod
    async def run(
        self, command: str, *, timeout_s: float, stdin: str | None = None
    ) -> CommandResult:
        """Runs a shell command in the working directory, to its end or its
        deadline.

        Args:
            command: A shell command line, run by ``sh -c``.
            timeout_s: Seconds the command may take. Past them it is killed,
                with what it started, and the result has no exit code.
            stdin: Text the command reads on standard input; none without it.

        Raises:
            SandboxError: The sandbox could not run the command at all.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Stops the sandbox and removes its files. A second call does nothing."""

    async def write_file(self, path: str, text: str) -> None:
        """Writes ``text`` to ``path``, making its directory.

        Args:
            path: The file's path; a relative one is under the working
                directory.
            text: The file's whole content.

        Raises:
            SandboxError: The file could not be written.
        """
        target = shlex.quote(path)
        command = f'mkdir -p -- "$(dirname -- {target})" && cat > {target}'
        result = await self.run(command, timeout_s=_WRITE_TIMEOUT_S, stdin=text)
        if result.exit_code != 0:
            why = result.stderr.strip() or f"no answer in {_WRITE_TIMEOUT_S:g} s"
            raise SandboxError(f"could not write {path} in the sandbox: {why}", retryable=False)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
