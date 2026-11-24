"""The coding environment: how outputs compare, which commands a check
runs, and what its report says."""

from __future__ import annotations

import json
import shlex

import pytest

from looper_cookbook.environments.base import Action, Observation
from looper_cookbook.environments.coding import (
    CheckReport,
    CodeTest,
    CodingEnvironment,
    Failure,
    outputs_match,
    run_tests,
)
from looper_cookbook.sandboxes.base import CommandResult, SandboxError
from looper_cookbook.sandboxes.unit_tests.scripted import OK, Prebuilt, ScriptedSandbox

SUM = "print(sum(map(int, input().split())))"
TESTS = (CodeTest(input="1 2\n", output="3\n"), CodeTest(input="5 5\n", output="10\n"))


def printed(text: str) -> CommandResult:
    return CommandResult(exit_code=0, stdout=text)


@pytest.mark.parametrize(
    "got,expected",
    [
        ("3\n", "3"),
        # Tokens compare in order, wherever the lines break.
        ("1 2\n3\n", "1\n2 3"),
        ("  YES  \n", "YES"),
        # Numbers that are not both integers are equal within 1e-6.
        ("0.5", "0.50000"),
        ("0.3333333", "0.33333333333"),
        ("1e3", "1000.0"),
        ("2", "2.0"),
        ("", ""),
    ],
)
def test_outputs_match_as_a_judge_reads_them(got: str, expected: str) -> None:
    assert outputs_match(got, expected)


@pytest.mark.parametrize(
    "got,expected",
    [
        ("3", "4"),
        ("1 2", "2 1"),
        ("1 2", "1 2 3"),
        ("yes", "YES"),
        # An integer must be exact, however large.
        ("1000000007", "1000000008"),
        ("0.5", "0.51"),
        ("abc", "1.0"),
        ("", "0"),
    ],
)
def test_outputs_differ_when_a_token_does(got: str, expected: str) -> None:
    assert not outputs_match(got, expected)


async def test_a_check_writes_the_solution_then_runs_each_test_on_its_input() -> None:
    sandbox = ScriptedSandbox(OK, printed("3\n"), printed("10\n"))
    report = await run_tests(sandbox, SUM, TESTS, timeout_s=4, python="python3.12")
    assert report == CheckReport(passed=2, total=2)
    assert report.all_passed
    written, first, second = sandbox.commands
    assert written[0].endswith("cat > solution.py") and written[2] == SUM
    assert first == ("python3.12 solution.py", 4, "1 2\n")
    assert second == ("python3.12 solution.py", 4, "5 5\n")


async def test_a_check_stops_at_the_first_test_that_fails() -> None:
    tests = (*TESTS, CodeTest(input="9 9\n", output="18\n"))
    sandbox = ScriptedSandbox(OK, printed("3\n"), printed("11\n"))
    report = await run_tests(sandbox, SUM, tests, timeout_s=4)
    assert report == CheckReport(
        passed=1,
        total=3,
        failure=Failure(input="5 5\n", expected="10\n", got="11", error=""),
    )
    assert not report.all_passed
    # The third test never ran.
    assert len(sandbox.commands) == 3


@pytest.mark.parametrize(
    "result,failure",
    [
        # A right output from a program that then crashed is a failure.
        (
            CommandResult(exit_code=1, stdout="3\n", stderr="Traceback\nValueError: x\n"),
            Failure(input="1 2\n", expected="3\n", got="3", error="Traceback\nValueError: x"),
        ),
        (
            CommandResult(exit_code=3),
            Failure(input="1 2\n", expected="3\n", got="", error="exit code 3"),
        ),
        (
            CommandResult(exit_code=None, stdout="3\n"),
            Failure(input="1 2\n", expected="3\n", got="3", error="timed out after 4 s"),
        ),
        # An output that was cut cannot match, and the failure says why.
        (
            CommandResult(exit_code=0, stdout="3 3 3", truncated=True),
            Failure(
                input="1 2\n",
                expected="3\n",
                got="3 3 3",
                error="printed more than the 1 MiB a sandbox keeps of an output",
            ),
        ),
    ],
)
async def test_a_failure_says_what_went_wrong_beyond_the_output(
    result: CommandResult, failure: Failure
) -> None:
    report = await run_tests(ScriptedSandbox(OK, result), SUM, TESTS, timeout_s=4)
    assert report == CheckReport(passed=0, total=2, failure=failure)


async def test_a_failure_keeps_the_start_of_each_part_and_the_end_of_its_error() -> None:
    test = CodeTest(input="i" * 1_500, output="o" * 1_500)
    result = CommandResult(exit_code=1, stdout="g" * 1_500, stderr="s" * 1_500 + "Error: last")
    report = await run_tests(ScriptedSandbox(OK, result), SUM, (test,), timeout_s=4)
    assert report.failure == Failure(
        input="i" * 1_000,
        expected="o" * 1_000,
        got="g" * 1_000,
        error="s" * 989 + "Error: last",
    )


async def test_a_function_test_calls_the_function_with_a_json_argument_a_line() -> None:
    test = CodeTest(input="[1, 2]\n3", output="[4, 5]", function="add; rm -rf /")
    # Whatever the solution printed comes before the result's own line.
    sandbox = ScriptedSandbox(OK, printed("debug\n\n[4, 5]\n"))
    assert await run_tests(sandbox, "def add(xs, k): ...", (test,), timeout_s=4) == CheckReport(
        passed=1, total=1
    )
    command, _, stdin = sandbox.commands[1]
    program, flag, script, function = shlex.split(command)
    assert (program, flag, function, stdin) == ("python3", "-c", "add; rm -rf /", "[1, 2]\n3")
    assert "import solution" in script and "sys.argv[1]" in script


@pytest.mark.parametrize(
    "stdout,expected",
    [
        ("\n[4, 6]\n", "[4, 5]"),
        # A result is the JSON value, not its text.
        ('\n"4"\n', "4"),
        ("\nnot json\n", "[4, 5]"),
        ("", "[4, 5]"),
        ("\n[4, 5]\n", "not json"),
    ],
)
async def test_a_function_test_fails_on_another_value(stdout: str, expected: str) -> None:
    test = CodeTest(input="[1, 2]\n3", output=expected, function="add")
    report = await run_tests(ScriptedSandbox(OK, printed(stdout)), "", (test,), timeout_s=4)
    assert report.passed == 0 and report.failure is not None


async def test_a_function_test_passes_on_an_equal_value_written_another_way() -> None:
    test = CodeTest(input="2", output='{"a": 1.0, "b": [true]}', function="f")
    sandbox = ScriptedSandbox(OK, printed('\n{"b": [true], "a": 1}\n'))
    assert (await run_tests(sandbox, "", (test,), timeout_s=4)).all_passed


async def test_check_solution_replies_with_the_report_as_json() -> None:
    sandbox = ScriptedSandbox(OK, printed("3\n"), printed("10\n"), OK, printed("3\n"), printed("9"))
    environment = await CodingEnvironment.Config(
        sandbox=Prebuilt(sandbox), tests=TESTS, timeout_s=4, python="python3.12"
    ).build()
    solved = await environment.step(Action(name="check_solution", arguments={"code": SUM}))
    assert solved == Observation(
        content='{"passed": true, "tests_passed": 2, "tests": 2}', metrics={"passed": 1.0}
    )
    wrong = await environment.step(Action(name="check_solution", arguments={"code": "print(9)"}))
    assert json.loads(wrong.content) == {
        "passed": False,
        "tests_passed": 1,
        "tests": 2,
        "first_failure": {"input": "5 5\n", "expected": "10\n", "got": "9", "error": ""},
    }
    assert wrong.metrics == {"passed": 0.0}
    assert sandbox.commands[1] == ("python3.12 solution.py", 4, "1 2\n")


async def test_a_wrong_call_reaches_no_sandbox() -> None:
    sandbox = ScriptedSandbox()
    environment = await CodingEnvironment.Config(sandbox=Prebuilt(sandbox), tests=TESTS).build()
    observation = await environment.step(Action(name="check_solution", arguments={"code": 3}))
    assert observation == Observation(
        content="Error: check_solution's argument code must be of type string, not 3.",
        metrics={"invalid_action": 1.0},
    )
    assert sandbox.commands == []


async def test_a_sandbox_that_cannot_run_the_solution_raises() -> None:
    class Broken(ScriptedSandbox):
        async def run(self, command, *, timeout_s, stdin=None):
            raise SandboxError("the container is gone", retryable=True)

    environment = await CodingEnvironment.Config(sandbox=Prebuilt(Broken()), tests=TESTS).build()
    with pytest.raises(SandboxError, match="the container is gone") as raised:
        await environment.step(Action(name="check_solution", arguments={"code": SUM}))
    assert raised.value.retryable


async def test_closing_it_closes_its_sandbox() -> None:
    sandbox = ScriptedSandbox()
    async with await CodingEnvironment.Config(sandbox=Prebuilt(sandbox), tests=TESTS).build():
        pass
    assert sandbox.closes == 1


def test_a_config_refuses_no_tests_and_no_time() -> None:
    sandbox = Prebuilt(ScriptedSandbox())
    with pytest.raises(ValueError, match="tests is empty, so check_solution would have nothing"):
        CodingEnvironment.Config(sandbox=sandbox, tests=())
    with pytest.raises(ValueError, match="timeout_s must be a positive number, not -1"):
        CodingEnvironment.Config(sandbox=sandbox, tests=TESTS, timeout_s=-1)


async def test_a_prelude_is_written_before_the_solution() -> None:
    prelude = "from typing import *\n"
    sandbox = ScriptedSandbox(OK, printed("3\n"), printed("10\n"))
    assert (await run_tests(sandbox, SUM, TESTS, timeout_s=4, prelude=prelude)).all_passed
    assert sandbox.commands[0][2] == prelude + SUM
    # The tool writes its config's prelude the same way.
    sandbox = ScriptedSandbox(OK, printed("3\n"), printed("10\n"))
    environment = await CodingEnvironment.Config(
        sandbox=Prebuilt(sandbox), tests=TESTS, prelude=prelude
    ).build()
    await environment.step(Action(name="check_solution", arguments={"code": SUM}))
    assert sandbox.commands[0][2] == prelude + SUM
