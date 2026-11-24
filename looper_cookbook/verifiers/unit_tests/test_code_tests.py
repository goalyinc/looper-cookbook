"""``CodeTestsVerifier``: the program it finds, the sandbox it runs it in,
and the score."""

from __future__ import annotations

import dataclasses

import pytest

from looper_cookbook.environments.coding import CodeTest
from looper_cookbook.sandboxes.base import CommandResult, SandboxError
from looper_cookbook.sandboxes.unit_tests.scripted import OK, ScriptedSandbox
from looper_cookbook.verifiers.base import Score
from looper_cookbook.verifiers.code_tests import CodeTestsVerifier, last_code_block

TESTS = (CodeTest(input="1 2\n", output="3\n"), CodeTest(input="5 5\n", output="10\n"))
SUM = "print(sum(map(int, input().split())))\n"


@dataclasses.dataclass
class Sandboxes:
    """A sandbox's config that hands out the sandboxes a test wrote, in order."""

    made: list[ScriptedSandbox]
    built: int = 0

    async def build(self) -> ScriptedSandbox:
        self.built += 1
        return self.made[self.built - 1]


def answer(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": "Add them."}, {"role": "assistant", "content": content}]


def printed(text: str) -> CommandResult:
    return CommandResult(exit_code=0, stdout=text)


@pytest.mark.parametrize(
    "text,code",
    [
        ("```python\nprint(1)\n```", "print(1)\n"),
        ("```\nprint(1)\n```", "print(1)\n"),
        # The last block is the program: an earlier one was a draft.
        ("```py\nold()\n```\nBetter:\n```python\nnew()\n```\nDone.", "new()\n"),
        ("print(1)", None),
        ("```python\nprint(1)", None),
    ],
)
def test_the_program_is_the_last_fenced_block(text: str, code: str | None) -> None:
    assert last_code_block(text) == code


async def test_a_program_that_passes_every_test_scores_one_in_a_sandbox_of_its_own() -> None:
    sandbox = ScriptedSandbox(OK, printed("3\n"), printed("10\n"))
    sandboxes = Sandboxes([sandbox])
    verifier = await CodeTestsVerifier.Config(
        sandbox=sandboxes, tests=TESTS, timeout_s=4, python="python3.12"
    ).build()
    # Building a verifier starts nothing: an episode cut off never verifies.
    assert sandboxes.built == 0
    score = await verifier.verify(answer(f"Here it is.\n```python\n{SUM}```"), None)
    assert score == Score(reward=1.0, metrics={"answered": 1.0})
    written, first, second = sandbox.commands
    assert written[2] == SUM
    assert first == ("python3.12 solution.py", 4, "1 2\n")
    assert second == ("python3.12 solution.py", 4, "5 5\n")
    assert sandbox.closes == 1


async def test_a_program_that_fails_a_test_scores_zero() -> None:
    sandbox = ScriptedSandbox(OK, printed("3\n"), printed("11\n"))
    verifier = await CodeTestsVerifier.Config(sandbox=Sandboxes([sandbox]), tests=TESTS).build()
    score = await verifier.verify(answer("```python\nprint(3)\n```"), None)
    assert score == Score(reward=0.0, metrics={"answered": 1.0})
    assert sandbox.closes == 1


async def test_a_reply_without_a_program_scores_zero_and_starts_no_sandbox() -> None:
    sandboxes = Sandboxes([])
    verifier = await CodeTestsVerifier.Config(sandbox=sandboxes, tests=TESTS).build()
    score = await verifier.verify(answer("I would add the two numbers."), None)
    assert score == Score(reward=0.0, metrics={"answered": 0.0})
    assert sandboxes.built == 0


async def test_a_sandbox_that_cannot_run_the_program_raises_and_is_closed() -> None:
    class Broken(ScriptedSandbox):
        async def run(self, command, *, timeout_s, stdin=None):
            raise SandboxError("the container is gone", retryable=True)

    sandbox = Broken()
    verifier = await CodeTestsVerifier.Config(sandbox=Sandboxes([sandbox]), tests=TESTS).build()
    with pytest.raises(SandboxError, match="the container is gone") as raised:
        await verifier.verify(answer("```python\nprint(3)\n```"), None)
    assert raised.value.retryable
    assert sandbox.closes == 1


def test_a_config_refuses_no_tests_and_no_time() -> None:
    sandboxes = Sandboxes([])
    with pytest.raises(ValueError, match="tests is empty, so every program would pass"):
        CodeTestsVerifier.Config(sandbox=sandboxes, tests=())
    with pytest.raises(ValueError, match="timeout_s must be a positive number, not 0"):
        CodeTestsVerifier.Config(sandbox=sandboxes, tests=TESTS, timeout_s=0)


async def test_the_verifier_writes_its_prelude_before_the_program() -> None:
    sandbox = ScriptedSandbox(OK, printed("3\n"), printed("10\n"))
    verifier = await CodeTestsVerifier.Config(
        sandbox=Sandboxes([sandbox]), tests=TESTS, prelude="import math\n"
    ).build()
    assert (await verifier.verify(answer(f"```python\n{SUM}```"), None)).reward == 1.0
    assert sandbox.commands[0][2] == "import math\n" + SUM
