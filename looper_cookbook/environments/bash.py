"""A shell in a sandbox.

``BashEnvironment`` offers one tool, ``bash``, and runs each command in its
sandbox. It is the environment of an agent that works in a terminal: it
writes files, runs programs and reads what they print.

Each command runs in a new shell, in the sandbox's working directory. Files
stay between commands; a ``cd`` and a shell variable do not. A command
returns once everything it started has closed its output, so a process left
running in the background, a server to test, has its output sent to a file.
The tool's description tells the model both.

The model reads a command's output as a terminal shows it, standard error
mixed into standard output in the order written. A long output keeps its
start and its end around a note of how much was left out: a build's or a
test run's errors are at the end. A status line follows a command that
failed or ran out of time.
"""

from __future__ import annotations

import dataclasses
import math
import shlex
from collections.abc import Sequence

from looper_cookbook.environments.base import Action, Environment, Observation, Tool, tool
from looper_cookbook.sandboxes.base import Sandbox

__all__ = ["BashEnvironment"]

_TOOLS = (
    tool(
        "bash",
        "Run a command in a bash shell and return what it prints. Each command runs in a new "
        "shell in the same working directory: files stay between commands, but a cd or a "
        "variable does not. To leave a process running in the background, send its output "
        "to a file, as in `server > server.log 2>&1 &`.",
        {"command": {"type": "string", "description": "The command line to run."}},
        required=("command",),
    ),
)


class BashEnvironment(Environment):
    """One tool, ``bash``, over a sandbox of its own."""

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Config:
        """A shell's sandbox and limits.

        Attributes:
            sandbox: Builds the sandbox the commands run in.
            timeout_s: Seconds one command may take before it is killed.
            max_output_chars: The most of a command's output the model
                reads. A longer one keeps its first and last halves.
        """

        sandbox: Sandbox.Config
        timeout_s: float = 60.0
        max_output_chars: int = 10_000

        def __post_init__(self) -> None:
            if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
                raise ValueError(f"timeout_s must be a positive number, not {self.timeout_s}")
            if self.max_output_chars < 2:
                raise ValueError(
                    f"max_output_chars must be at least 2, not {self.max_output_chars}"
                )

        async def build(self) -> BashEnvironment:
            return BashEnvironment(self, await self.sandbox.build())

    def __init__(self, config: Config, sandbox: Sandbox) -> None:
        self._config = config
        self._sandbox = sandbox

    @property
    def tools(self) -> Sequence[Tool]:
        return _TOOLS

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    async def execute(self, action: Action) -> Observation:
        # Standard error joins standard output, in the order the command wrote them.
        command = f"bash -c {shlex.quote(action.arguments['command'])} 2>&1"
        result = await self._sandbox.run(command, timeout_s=self._config.timeout_s)
        output = _limited(result.stdout, self._config.max_output_chars)
        if result.timed_out:
            status = f"[timed out after {self._config.timeout_s:g} s and was killed]"
        elif result.exit_code != 0:
            status = f"[exit code {result.exit_code}]"
        else:
            status = "" if output else "[no output]"
        if output and status and not output.endswith("\n"):
            output += "\n"
        return Observation(
            content=output + status,
            metrics={
                "failed": float(result.exit_code != 0),
                "timed_out": float(result.timed_out),
            },
        )


def _limited(text: str, limit: int) -> str:
    """``text`` whole when it fits ``limit`` characters, else its first and
    last halves around a note of how much is left out."""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n[... {omitted} characters left out ...]\n{text[-half:]}"
