"""The example environments over a real shell and a real Python, in a local
sandbox.

``sandboxes`` runs each test in a local sandbox, and in a Docker container
too when ``LOOPER_TEST_DOCKER=1``: an environment behaves the same in
either.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from looper_cookbook.environments.base import Action
from looper_cookbook.environments.bash import BashEnvironment
from looper_cookbook.environments.coding import (
    CheckReport,
    CodeTest,
    CodingEnvironment,
    Failure,
    run_tests,
)
from looper_cookbook.sandboxes.base import Sandbox
from looper_cookbook.sandboxes.docker import DockerSandbox
from looper_cookbook.sandboxes.local import LocalSandbox

IMAGE = "python:3.12-slim"
DOCKER = pytest.param(
    "docker",
    marks=pytest.mark.skipif(
        os.environ.get("LOOPER_TEST_DOCKER") != "1",
        reason="set LOOPER_TEST_DOCKER=1 to run in real Docker containers",
    ),
)
SUM = "import sys\nprint(sum(int(word) for word in sys.stdin.read().split()))\n"
SUM_TESTS = (
    CodeTest(input="1 2\n", output="3\n"),
    CodeTest(input="10\n20\n30\n", output="60"),
)


@pytest.fixture(params=["local", DOCKER])
def sandbox(request: pytest.FixtureRequest) -> tuple[Sandbox.Config, str]:
    """A sandbox's config, and the command that runs Python in it."""
    if request.param == "docker":
        return DockerSandbox.Config(image=IMAGE), "python3"
    # This interpreter: a machine's own python3 may be older, or missing.
    return LocalSandbox.Config(), sys.executable


def bash(command: str) -> Action:
    return Action(name="bash", arguments={"command": command})


def check(code: str) -> Action:
    return Action(name="check_solution", arguments={"code": code})


async def test_bash_shows_a_commands_output_as_a_terminal_does(
    sandbox: tuple[Sandbox.Config, str],
) -> None:
    async with await BashEnvironment.Config(sandbox=sandbox[0], timeout_s=5).build() as shell:
        mixed = await shell.step(bash("echo one; echo two >&2; echo three"))
        assert mixed.content == "one\ntwo\nthree\n"
        assert mixed.metrics == {"failed": 0.0, "timed_out": 0.0}
        # Bash, not sh: a model writes what it knows.
        assert (await shell.step(bash("[[ abc == a* ]] && echo bash"))).content == "bash\n"
        failed = await shell.step(bash("ls /no/such/place"))
        assert failed.content.endswith("\n[exit code 2]") and failed.metrics["failed"] == 1.0
        assert (await shell.step(bash("true"))).content == "[no output]"


async def test_bash_keeps_files_between_commands_and_nothing_of_the_shell(
    sandbox: tuple[Sandbox.Config, str],
) -> None:
    async with await BashEnvironment.Config(sandbox=sandbox[0], timeout_s=5).build() as shell:
        await shell.step(bash("mkdir work && cd work && echo 42 > answer && export SEEN=1"))
        assert (await shell.step(bash("cat work/answer; echo ${SEEN:-unset}"))).content == (
            "42\nunset\n"
        )


async def test_bash_kills_a_command_at_its_deadline_and_runs_the_next(
    sandbox: tuple[Sandbox.Config, str],
) -> None:
    async with await BashEnvironment.Config(sandbox=sandbox[0], timeout_s=1).build() as shell:
        late = await shell.step(bash("echo started; sleep 30; echo finished"))
        assert late.content == "started\n[timed out after 1 s and was killed]"
        assert late.metrics == {"failed": 1.0, "timed_out": 1.0}
        assert (await shell.step(bash("echo still here"))).content == "still here\n"


async def test_bash_leaves_a_background_process_running_for_the_next_command(
    sandbox: tuple[Sandbox.Config, str],
) -> None:
    async with await BashEnvironment.Config(sandbox=sandbox[0], timeout_s=5).build() as shell:
        # Its output goes to a file, so the command does not wait for it.
        started = await shell.step(bash("sleep 60 > sleep.log 2>&1 & echo $!"))
        pid = int(started.content)
        try:
            alive = await shell.step(bash(f"kill -0 {pid} && echo alive"))
            assert alive.content == "alive\n"
        finally:
            # A local sandbox does not end it, so the test does.
            await shell.step(bash(f"kill {pid}"))


async def test_bash_cuts_a_long_output_to_its_ends(sandbox: tuple[Sandbox.Config, str]) -> None:
    config = BashEnvironment.Config(sandbox=sandbox[0], timeout_s=5, max_output_chars=20)
    async with await config.build() as shell:
        observation = await shell.step(bash("seq 1 1000"))
        assert observation.content == (
            "1\n2\n3\n4\n5\n\n[... 3873 characters left out ...]\n\n999\n1000\n"
        )


async def test_check_solution_passes_a_right_program(sandbox: tuple[Sandbox.Config, str]) -> None:
    config = CodingEnvironment.Config(
        sandbox=sandbox[0], tests=SUM_TESTS, timeout_s=5, python=sandbox[1]
    )
    async with await config.build() as environment:
        observation = await environment.step(check(SUM))
        assert json.loads(observation.content) == {"passed": True, "tests_passed": 2, "tests": 2}
        assert observation.metrics == {"passed": 1.0}


@pytest.mark.parametrize(
    "code,passed,got,error",
    [
        ("print(3)", 1, "3", ""),
        ("print('3'); raise SystemExit(4)", 0, "3", "exit code 4"),
        ("import sys\nprint(3)\nsys.stderr.write('note\\n')\n1 / 0", 0, "3", "ZeroDivisionError"),
        ("print(", 0, "", "SyntaxError"),
        ("print(3, flush=True)\nwhile True: pass", 0, "3", "timed out after 1 s"),
    ],
)
async def test_check_solution_shows_the_first_test_a_program_fails(
    sandbox: tuple[Sandbox.Config, str], code: str, passed: int, got: str, error: str
) -> None:
    config = CodingEnvironment.Config(
        sandbox=sandbox[0], tests=SUM_TESTS, timeout_s=1, python=sandbox[1]
    )
    async with await config.build() as environment:
        reply = json.loads((await environment.step(check(code))).content)
        test = SUM_TESTS[passed]
        assert reply["passed"] is False and reply["tests_passed"] == passed
        failure = reply["first_failure"]
        assert (failure["input"], failure["expected"], failure["got"]) == (
            test.input,
            test.output,
            got,
        )
        assert error in failure["error"] and (error or not failure["error"])
        # A program that failed or hung leaves the environment able to check the next.
        assert json.loads((await environment.step(check(SUM))).content)["passed"] is True


async def test_run_tests_calls_a_function_as_leetcode_and_as_a_module_define_it(
    sandbox: tuple[Sandbox.Config, str],
) -> None:
    tests = (
        CodeTest(input="[3, 1, 2]\n2", output="[1, 2]", function="smallest"),
        CodeTest(input='["b", "a"]\n1', output='["a"]', function="smallest"),
    )
    method = "class Solution:\n    def smallest(self, xs, k):\n        return sorted(xs)[:k]\n"
    function = "def smallest(xs, k):\n    print('sorting', xs)\n    return sorted(xs)[:k]\n"
    wrong = "def smallest(xs, k):\n    return xs[:k]\n"
    async with await sandbox[0].build() as built:
        for code in (method, function):
            report = await run_tests(built, code, tests, timeout_s=5, python=sandbox[1])
            assert report == CheckReport(passed=2, total=2)
        report = await run_tests(built, wrong, tests, timeout_s=5, python=sandbox[1])
        assert report == CheckReport(
            passed=0,
            total=2,
            failure=Failure(input="[3, 1, 2]\n2", expected="[1, 2]", got="[3, 1]", error=""),
        )
        missing = await run_tests(built, "x = 1\n", tests, timeout_s=5, python=sandbox[1])
        assert missing.failure is not None and "AttributeError" in missing.failure.error


async def test_a_program_cannot_read_the_output_a_test_expects(
    sandbox: tuple[Sandbox.Config, str],
) -> None:
    # It lists every file and variable it can reach; the expected output is in none.
    spy = (
        "import os, sys\n"
        "print(sys.argv, sorted(os.listdir('.')), sorted(os.environ), sys.stdin.read())\n"
    )
    secret = "expected-only-in-this-process"
    async with await sandbox[0].build() as built:
        report = await run_tests(
            built, spy, (CodeTest(input="in", output=secret),), timeout_s=5, python=sandbox[1]
        )
    assert report.failure is not None and secret not in report.failure.got
    assert "solution.py" in report.failure.got
