"""Python solutions checked against a problem's tests.

``CodingEnvironment`` is the environment of a programming problem, the one
the ``code_rl`` recipe trains with on DeepCoder's problems. It offers one
tool, ``check_solution``: the model sends a whole program, the environment
runs it against tests in its sandbox, and the reply says how many passed
and shows the first that failed.

**Two kinds of test,** as DeepCoder's problems have:

- a program test gives ``input`` on standard input and compares what the
  program prints with ``output``;
- a function test, with ``function`` set, calls that function with the
  arguments ``input`` lists, a JSON value a line, and compares what it
  returns with the JSON value in ``output``. The function is a method of
  the code's ``Solution`` class when it defines one, as LeetCode's problems
  do, else a function of the module.

**The comparison happens here, never in the sandbox.** The sandbox gets the
program and a test's input, and returns what the program printed. The
expected output stays in this process, so nothing the program does can read
it or change the verdict.

**``check_solution`` shows a failing test whole:** its input, the output
expected and the output got. So an environment's config lists only tests
its model may read, such as a problem's examples. The tests that decide a
reward stay with the verifier, which runs them with ``run_tests`` in a
sandbox of its own.

A check stops at the first failing test. A test passes when the program
exits with status 0 inside ``timeout_s`` and its output matches.

**A prelude** is code written before every solution: the imports a problem
set's solutions may assume. LeetCode's problems are written with ``List``
and ``defaultdict`` in scope, and LiveCodeBench's runner supplies them.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import shlex
from collections.abc import Sequence

from looper_cookbook.environments.base import Action, Environment, Observation, Tool, tool
from looper_cookbook.sandboxes.base import MAX_OUTPUT_BYTES, CommandResult, Sandbox

__all__ = [
    "TOOLS",
    "CheckReport",
    "CodeTest",
    "CodingEnvironment",
    "Failure",
    "outputs_match",
    "run_tests",
]

# The environment's one tool: public, so a recipe can count its prompt's tokens before any is built.
TOOLS = (
    tool(
        "check_solution",
        "Run your Python solution against the problem's example tests. The reply says how "
        "many passed and shows the first that failed. Use it before you give your final answer.",
        {"code": {"type": "string", "description": "The whole Python program."}},
        required=("code",),
    ),
)
_SOLUTION = "solution.py"
# Calls the function a test names, with a JSON argument a line of stdin, and
# prints its result as JSON on a last line of its own.
_CALL = """\
import json, sys
import solution
arguments = [json.loads(line) for line in sys.stdin.read().splitlines() if line.strip()]
owner = solution.Solution() if hasattr(solution, "Solution") else solution
result = getattr(owner, sys.argv[1])(*arguments)
print()
print(json.dumps(result))
"""
_INTEGER = re.compile(r"[+-]?\d+")
# The most of each part of a failing test a report keeps, in characters.
_SHOWN_CHARS = 1_000


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class CodeTest:
    """One test of a programming problem.

    Attributes:
        input: What the program reads on standard input. For a function
            test, the call's arguments, a JSON value a line.
        output: What the program must print. For a function test, the JSON
            value the call must return.
        function: The function a function test calls; ``None`` for a
            program test.
    """

    input: str
    output: str
    function: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Failure:
    """The first test a solution failed, each part cut to its first 1,000
    characters, and the error to its last.

    Attributes:
        input: The test's input.
        expected: The output the test expects.
        got: What the program printed, or the function returned.
        error: Why it failed beyond a wrong output: the end of what it
            wrote to standard error, or that it ran out of time. Empty for
            a wrong output alone.
    """

    input: str
    expected: str
    got: str
    error: str


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class CheckReport:
    """How a solution did on a list of tests.

    Attributes:
        passed: The tests it passed before the first it failed.
        total: The tests in the list.
        failure: The first test it failed; ``None`` when it passed them all.
    """

    passed: int
    total: int
    failure: Failure | None = None

    @property
    def all_passed(self) -> bool:
        return self.failure is None


def outputs_match(got: str, expected: str) -> bool:
    """Whether a program's output answers a test, as a judge's checker reads
    it: the same tokens in the same order, wherever the lines break.

    Two tokens are the same when they are equal as text, or when both are
    numbers within 1e-6 of each other and not both integers: an integer
    must be exact.
    """
    got_tokens, expected_tokens = got.split(), expected.split()
    return len(got_tokens) == len(expected_tokens) and all(
        map(_tokens_match, got_tokens, expected_tokens)
    )


def _tokens_match(got: str, expected: str) -> bool:
    if got == expected:
        return True
    if _INTEGER.fullmatch(got) and _INTEGER.fullmatch(expected):
        return False
    try:
        return math.isclose(float(got), float(expected), rel_tol=1e-6, abs_tol=1e-6)
    except ValueError:
        return False


def _results_match(got: str, expected: str) -> bool:
    """Whether a function's result, the last line it printed, is the JSON
    value a test expects."""
    lines = got.splitlines()
    try:
        return bool(lines) and json.loads(lines[-1]) == json.loads(expected)
    except json.JSONDecodeError:
        return False


async def run_tests(
    sandbox: Sandbox,
    code: str,
    tests: Sequence[CodeTest],
    *,
    timeout_s: float,
    python: str = "python3",
    prelude: str = "",
) -> CheckReport:
    """Runs a Python solution against ``tests`` in ``sandbox``, in order,
    and stops at the first it fails.

    Args:
        sandbox: Where the solution runs. It is written there as
            ``solution.py``, replacing the last one checked.
        code: The whole program.
        tests: The tests to run.
        timeout_s: Seconds the solution may take on one test.
        python: The command that runs Python in the sandbox.
        prelude: Code written before the solution, such as the imports its
            problem set lets a solution assume.

    Raises:
        SandboxError: The sandbox could not run the solution at all.
    """
    await sandbox.write_file(_SOLUTION, prelude + code)
    for index, test in enumerate(tests):
        if test.function is None:
            command = f"{python} {_SOLUTION}"
            matches = outputs_match
        else:
            command = f"{python} -c {shlex.quote(_CALL)} {shlex.quote(test.function)}"
            matches = _results_match
        result = await sandbox.run(command, timeout_s=timeout_s, stdin=test.input)
        if result.exit_code != 0 or not matches(result.stdout, test.output):
            failure = Failure(
                input=test.input[:_SHOWN_CHARS],
                expected=test.output[:_SHOWN_CHARS],
                got=result.stdout.strip()[:_SHOWN_CHARS],
                error=_error(result, timeout_s),
            )
            return CheckReport(passed=index, total=len(tests), failure=failure)
    return CheckReport(passed=len(tests), total=len(tests))


def _error(result: CommandResult, timeout_s: float) -> str:
    if result.timed_out:
        return f"timed out after {timeout_s:g} s"
    if result.exit_code != 0:
        # A traceback names its error at its end.
        return result.stderr.strip()[-_SHOWN_CHARS:] or f"exit code {result.exit_code}"
    if result.truncated:
        return f"printed more than the {MAX_OUTPUT_BYTES >> 20} MiB a sandbox keeps of an output"
    return ""


class CodingEnvironment(Environment):
    """One tool, ``check_solution``, over a sandbox of its own."""

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Config:
        """A problem's sandbox and the tests its model may run.

        Attributes:
            sandbox: Builds the sandbox solutions run in.
            tests: The tests ``check_solution`` runs and may show whole.
                Never a test that decides the reward.
            timeout_s: Seconds a solution may take on one test.
            python: The command that runs Python in the sandbox.
            prelude: Code written before every solution: the imports its
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
                    "tests is empty, so check_solution would have nothing to run: list the "
                    "problem's example tests, or give the model no environment"
                )
            if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
                raise ValueError(f"timeout_s must be a positive number, not {self.timeout_s}")

        async def build(self) -> CodingEnvironment:
            return CodingEnvironment(self, await self.sandbox.build())

    def __init__(self, config: Config, sandbox: Sandbox) -> None:
        self._config = config
        self._sandbox = sandbox

    @property
    def tools(self) -> Sequence[Tool]:
        return TOOLS

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    async def execute(self, action: Action) -> Observation:
        report = await run_tests(
            self._sandbox,
            action.arguments["code"],
            self._config.tests,
            timeout_s=self._config.timeout_s,
            python=self._config.python,
            prelude=self._config.prelude,
        )
        reply: dict[str, object] = {
            "passed": report.all_passed,
            "tests_passed": report.passed,
            "tests": report.total,
        }
        if report.failure is not None:
            reply["first_failure"] = dataclasses.asdict(report.failure)
        return Observation(content=json.dumps(reply), metrics={"passed": float(report.all_passed)})
