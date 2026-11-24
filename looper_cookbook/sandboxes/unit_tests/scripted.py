"""A sandbox that runs nothing, so the unit tests of what runs in one need
no process.

It records each command it is given and answers with the next of the results
it was built with. What a real sandbox does with a command is the
integration tests' to show (``tests/integration/sandboxes/``).
"""

from __future__ import annotations

import dataclasses

from looper_cookbook.sandboxes.base import CommandResult, Sandbox

OK = CommandResult(exit_code=0)


class ScriptedSandbox(Sandbox):
    """Answers each command with the next of ``results``.

    Attributes:
        commands: Every call so far: its command, its deadline and its input.
        closes: How many times it was closed.
    """

    def __init__(self, *results: CommandResult) -> None:
        self._results = list(results)
        self.commands: list[tuple[str, float, str | None]] = []
        self.closes = 0

    async def run(
        self, command: str, *, timeout_s: float, stdin: str | None = None
    ) -> CommandResult:
        self.commands.append((command, timeout_s, stdin))
        return self._results.pop(0)

    async def close(self) -> None:
        self.closes += 1


@dataclasses.dataclass(frozen=True)
class Prebuilt:
    """A sandbox's config that builds the sandbox a test already holds."""

    sandbox: ScriptedSandbox

    async def build(self) -> ScriptedSandbox:
        return self.sandbox
