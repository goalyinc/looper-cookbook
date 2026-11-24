"""A program run against hidden tests.

``CodeTestsVerifier`` scores a programming problem: the reward is 1 when the
program in the model's reply passes every test, else 0. It is the reward of
the ``code_rl`` recipe, which trains on DeepCoder's problems.

**The program is the reply's last fenced code block.** A reply without one
scores 0, and counts as not ``answered``.

**The tests run in a sandbox of the verifier's own,** started for the check
and closed after it. The model never acted there, so nothing it did during
the episode can touch the tests or their verdict. ``CodingEnvironment``
shows the model a problem's example tests; the tests that decide the reward
are this verifier's, and stay here. Outputs are compared in this process,
by ``run_tests``.

An episode that was cut off is never verified, so no sandbox starts for it.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Sequence

from looper_cookbook.environments.base import Environment
from looper_cookbook.environments.coding import CodeTest, run_tests
from looper_cookbook.message_formatters.base import Message
from looper_cookbook.sandboxes.base import Sandbox
from looper_cookbook.verifiers.base import Score, Verifier, last_reply

__all__ = ["CodeTestsVerifier", "last_code_block"]

# A fenced block: its opening line may name a language.
_BLOCK = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


def last_code_block(text: str) -> str | None:
    """The content of the text's last fenced code block; ``None`` when it has none."""
    blocks = _BLOCK.findall(text)
    return blocks[-1] if blocks else None


class CodeTestsVerifier(Verifier):
    """Reward 1 when the reply's program passes every test, else 0."""

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Config:
        """A problem's hidden tests, and where they run.

        Attributes:
            sandbox: Builds the sandbox the program runs in, one per check.
            tests: The tests that decide the reward. The model never sees
                them.
            timeout_s: Seconds the program may take on one test.
            python: The command that runs Python in the sandbox.
            prelude: Code written before the program: the imports its
                problem set lets a solution assume.
        """

        sandbox: Sandbox.Config
        tests: Sequence[CodeTest]
        timeout_s: float = 6.0
        python: str = "python3"
        prelude: str = ""

        def __post_init__(self) -> None:
            if not self.tests:
                raise ValueError(
                    "tests is empty, so every program would pass: list the tests that "
                    "decide the problem's reward"
                )
            if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
                raise ValueError(f"timeout_s must be a positive number, not {self.timeout_s}")

        async def build(self) -> CodeTestsVerifier:
            return CodeTestsVerifier(self)

    def __init__(self, config: Config) -> None:
        self._config = config

    async def verify(self, messages: Sequence[Message], environment: Environment | None) -> Score:
        """Runs the reply's program against the tests, in a new sandbox.

        Raises:
            SandboxError: The sandbox could not start, or could not run the
                program at all.
        """
        code = last_code_block(last_reply(messages))
        if code is None:
            return Score(reward=0.0, metrics={"answered": 0.0})
        async with await self._config.sandbox.build() as sandbox:
            report = await run_tests(
                sandbox,
                code,
                self._config.tests,
                timeout_s=self._config.timeout_s,
                python=self._config.python,
                prelude=self._config.prelude,
            )
        return Score(reward=float(report.all_passed), metrics={"answered": 1.0})
