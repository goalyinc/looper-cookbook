"""DeepCoder's problems as tasks.

DeepCoder (Luo et al., 2025) trains a coding model with RL on competitive
programming problems. Its dataset, ``agentica-org/DeepCoder-Preview-Dataset``
(MIT), has four subsets. The recipe trains on their train splits and tests
on their test splits:

- ``primeintellect``, train, 16,252 problems: a list of
  ``{"type": "stdin_stdout", "input", "output"}``;
- ``taco``, train, 7,436: ``{"inputs", "outputs"}``, with ``"fn_name"`` for
  a problem that asks for a function;
- ``lcbv5``, train and test, 599 and 279: a list of
  ``{"input", "output", "testtype"}``, ``"stdin"`` or ``"functional"``,
  with the function's name in the row's ``metadata``;
- ``codeforces``, test, 408: a list of ``{"input", "output"}``.

``code_tests`` reads all four as ``CodeTest``s: a program test, which gives
standard input and expects standard output, or a function test, which calls
a function with JSON arguments and expects a JSON result.

**A task** is one problem as an ``AgentSession`` attempts it:

- the question, under LiveCodeBench's prompt;
- a ``CodingEnvironment`` whose ``check_solution`` runs the problem's first
  ``public_tests`` tests. The tool shows a failing test whole, so those are
  the tests the model may read;
- a ``CodeTestsVerifier`` that runs every test the problem keeps, in a
  sandbox of its own: the reward is 1 when the final program passes them
  all, else 0.

**The tests a problem keeps** are its first ``max_tests`` among those of at
most ``max_test_chars`` characters, input and output together. Some
problems carry hundreds of megabytes of tests, and one whose expected
output is over the 1 MiB a sandbox keeps of an output could never pass. A
problem left with no test is dropped. Leaving out the largest tests lets a
solution too slow for them pass.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any, cast

from datasets import load_dataset

from looper_cookbook.environments.coding import CodeTest, CodingEnvironment
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.sandboxes.base import Sandbox
from looper_cookbook.verifiers.code_tests import CodeTestsVerifier

__all__ = [
    "LIVE_CODE_BENCH_IMPORTS",
    "SUBSETS",
    "Problem",
    "code_tests",
    "load",
    "question",
    "task",
]

DATASET = "agentica-org/DeepCoder-Preview-Dataset"
# The dataset as of 2025-04-09: a later edit changes no recorded run.
REVISION = "177913a7bd43791646ef6a43645caa3c871ab3db"
# The dataset's subsets in each of its splits.
SUBSETS: Mapping[str, tuple[str, ...]] = {
    "train": ("primeintellect", "taco", "lcbv5"),
    "test": ("codeforces", "lcbv5"),
}

# What LiveCodeBench's runner writes before every solution, so one written
# for LeetCode, with List or defaultdict unimported, runs as it does there.
LIVE_CODE_BENCH_IMPORTS = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
    "from random import *\nfrom statistics import *\nfrom itertools import *\n"
    "from functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\n"
    "from json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\n"
    "import datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\n"
    "import random\nimport statistics\nimport itertools\nimport functools\nimport operator\n"
    "import io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"
)

# LiveCodeBench's prompt, which every problem is asked under.
_EXPERT = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)
_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem and "
    "enclose your code within delimiters."
)
_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not "
    "directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm and "
    "writes output to STDOUT."
)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Problem:
    """One programming problem.

    Attributes:
        question: The problem as the model is asked it.
        tests: The tests it keeps, the first of them the ones a model may
            read.
    """

    question: str
    tests: tuple[CodeTest, ...]


def code_tests(raw: Any, metadata: Mapping[str, Any] | None = None) -> list[CodeTest]:
    """A row's tests, in whichever of the subsets' shapes they come.

    Args:
        raw: The row's ``tests``, as JSON text or already read.
        metadata: The row's ``metadata``, which names an ``lcbv5``
            problem's function.
    """
    tests = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(tests, Mapping):
        function = tests.get("fn_name")
        pairs = zip(tests.get("inputs", ()), tests.get("outputs", ()), strict=False)
        return [_taco_test(given, expected, function) for given, expected in pairs]
    named = (metadata or {}).get("func_name")
    result = []
    for test in tests or ():
        kind = test.get("testtype") or test.get("type") or "stdin"
        if kind != "functional":
            result.append(CodeTest(input=str(test["input"]), output=str(test["output"])))
        elif named:
            # The arguments are JSON already, a value a line.
            result.append(
                CodeTest(input=str(test["input"]), output=str(test["output"]), function=named)
            )
    return result


def _taco_test(given: Any, expected: Any, function: str | None) -> CodeTest:
    """One of TACO's tests. A program's input and output are text, or a list
    of lines; a call's arguments are a list, and its result is in a list of
    one."""
    if function is None:
        return CodeTest(input=_lines(given), output=_lines(expected))
    arguments = given if isinstance(given, list) else [given]
    result = expected[0] if isinstance(expected, list) and len(expected) == 1 else expected
    return CodeTest(
        input="\n".join(json.dumps(argument) for argument in arguments),
        output=json.dumps(result),
        function=function,
    )


def _lines(value: Any) -> str:
    if isinstance(value, list):
        return "".join(f"{line}\n" for line in value)
    return str(value)


def question(problem: str, starter_code: str | None = None) -> str:
    """A problem under LiveCodeBench's prompt: the statement, then how to
    answer, with its starter code when it has one."""
    text = f"{_EXPERT}\n\n{problem}"
    if starter_code and starter_code.strip():
        text += f"### Format: {_WITH_STARTER_CODE}\n```python\n{starter_code}\n```\n\n"
    else:
        text += f"### Format: {_WITHOUT_STARTER_CODE}\n```python\n# YOUR CODE HERE\n```\n\n"
    return text + "### Answer: (use the provided format with backticks)\n\n"


def problem(row: Mapping[str, Any], *, max_tests: int, max_test_chars: int) -> Problem | None:
    """A row of any subset as a problem; ``None`` for one with no statement
    or no test it can keep."""
    statement = row.get("problem")
    if not isinstance(statement, str) or not statement.strip():
        return None
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    tests = [
        test
        for test in code_tests(row.get("tests"), metadata)
        if len(test.input) + len(test.output) <= max_test_chars
    ][:max_tests]
    if not tests:
        return None
    return Problem(question=question(statement, row.get("starter_code")), tests=tuple(tests))


def load(
    split: str, *, limit: int | None = None, max_tests: int = 32, max_test_chars: int = 65_536
) -> list[Problem]:
    """A split's problems, subset after subset.

    Args:
        split: ``"train"`` or ``"test"``.
        limit: The most problems read of each subset. With one, the subset
            is streamed, so a trial reads its first rows and downloads
            little; without, each subset is downloaded whole into Hugging
            Face's cache, 8 GB for both splits.
        max_tests: The most tests a problem keeps.
        max_test_chars: The longest test kept, its input and its expected
            output together.
    """
    problems = []
    for subset in SUBSETS[split]:
        rows = load_dataset(
            DATASET, name=subset, split=split, revision=REVISION, streaming=limit is not None
        )
        kept = 0
        for row in rows:
            made = problem(
                cast(Mapping[str, Any], row), max_tests=max_tests, max_test_chars=max_test_chars
            )
            if made is None:
                continue
            problems.append(made)
            kept += 1
            if limit is not None and kept >= limit:
                break
    return problems


def task(
    made: Problem,
    *,
    sandbox: Sandbox.Config,
    python: str = "python3",
    public_tests: int = 2,
    timeout_s: float = 6.0,
    max_turns: int = 2,
    max_episode_tokens: int,
) -> AgentSession.Config:
    """A problem as a task: the question, a tool that checks a solution on
    the problem's first tests, and a verifier that scores the final program
    on all of them.

    Args:
        made: The problem.
        sandbox: Builds each sandbox: one for an attempt's tool, and
            another for its verifier.
        python: The command that runs Python in the sandbox.
        public_tests: The tests ``check_solution`` runs and may show.
        timeout_s: Seconds a solution may take on one test.
        max_turns: The most turns of an attempt: with 2, one check and the
            answer.
        max_episode_tokens: An attempt's token budget: its prompt, every
            turn and every tool reply together. A turn's budget is not the
            task's: it is the ``max_tokens`` of the run's sampling params.
    """
    shared: dict[str, Any] = {
        "sandbox": sandbox,
        "timeout_s": timeout_s,
        "python": python,
        "prelude": LIVE_CODE_BENCH_IMPORTS,
    }
    return AgentSession.Config(
        messages=[{"role": "user", "content": made.question}],
        environment=CodingEnvironment.Config(tests=made.tests[:public_tests], **shared),
        verifier=CodeTestsVerifier.Config(tests=made.tests, **shared),
        max_turns=max_turns,
        max_episode_tokens=max_episode_tokens,
    )
