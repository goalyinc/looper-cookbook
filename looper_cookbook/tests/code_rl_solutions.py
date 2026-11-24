"""DeepCoder's own solutions, checked as the ``code_rl`` recipe checks a
model's. Run by hand::

    python -m looper_cookbook.tests.code_rl_solutions --problems 200

The dataset's ``taco`` and ``primeintellect`` subsets carry solutions. If
the recipe reads a problem's tests right, and the code check runs them
right, the problem's own solution passes them. So this reads each subset's
first problems as the recipe does, runs each one's first solution against
the tests the recipe keeps, in a sandbox, and prints the share that pass, by
subset and by kind of test, with the first failures.

Not every failure is the check's: some solutions are Python 2, and some
take longer than a test's deadline on a busy machine. A share well under
the dataset's own would mean the recipe misreads a format, which is what
this is for. Run it when ``recipes/code_rl/tasks.py`` or
``environments/coding.py`` changes, and say what it showed in the PR.

It streams the rows it reads, so it downloads little. The solutions are the
dataset's, not ours: by default they run in Docker containers.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import sys

from datasets import load_dataset

from looper_cookbook.environments.coding import CheckReport, run_tests
from looper_cookbook.recipes.code_rl import tasks
from looper_cookbook.sandboxes.base import Sandbox
from looper_cookbook.sandboxes.docker import DockerSandbox
from looper_cookbook.sandboxes.local import LocalSandbox
from looper_cookbook.verifiers.code_tests import last_code_block

SUBSETS = ("taco", "primeintellect")


async def check(
    sandbox: Sandbox.Config, python: str, code: str, problem: tasks.Problem, timeout_s: float
) -> CheckReport:
    async with await sandbox.build() as built:
        return await run_tests(
            built,
            code,
            problem.tests,
            timeout_s=timeout_s,
            python=python,
            prelude=tasks.LIVE_CODE_BENCH_IMPORTS,
        )


async def main(args: argparse.Namespace) -> int:
    if args.sandbox == "local":
        sandbox: Sandbox.Config = LocalSandbox.Config()
        python = sys.executable
    else:
        sandbox = DockerSandbox.Config(image=args.image, memory_mb=1024)
        python = "python3"
    slots = asyncio.Semaphore(args.at_once)
    counts: collections.Counter[tuple[str, str, bool]] = collections.Counter()
    failures: list[str] = []

    async def one(subset: str, number: int, code: str, problem: tasks.Problem) -> None:
        async with slots:
            report = await check(sandbox, python, code, problem, args.timeout_s)
        kind = "function" if problem.tests[0].function else "program"
        counts[subset, kind, report.all_passed] += 1
        if report.failure is not None:
            failure = report.failure
            failures.append(
                f"{subset} #{number} ({kind}), test {report.passed + 1} of {report.total}: "
                f"expected {failure.expected[:60]!r}, got {failure.got[:60]!r}, "
                f"error {failure.error[-160:]!r}"
            )

    checks = []
    for subset in SUBSETS:
        rows = load_dataset(
            tasks.DATASET, name=subset, split="train", revision=tasks.REVISION, streaming=True
        )
        read = 0
        for number, row in enumerate(rows):
            problem = tasks.problem(
                row, max_tests=args.max_tests, max_test_chars=args.max_test_chars
            )
            solutions = row.get("solutions") or ()
            if problem is None or not solutions:
                continue
            # PrimeIntellect's solutions are fenced, as a model's answer is; TACO's are bare.
            code = last_code_block(solutions[0]) or solutions[0]
            checks.append(one(subset, number, code, problem))
            read += 1
            if read >= args.problems:
                break
    await asyncio.gather(*checks)
    for subset in SUBSETS:
        for kind in ("program", "function"):
            passed, failed = counts[subset, kind, True], counts[subset, kind, False]
            if passed + failed:
                share = passed / (passed + failed)
                print(f"{subset}, {kind} tests: {passed} of {passed + failed} pass ({share:.1%})")
    for line in failures[: args.show]:
        print(line)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepCoder's solutions against its tests.")
    parser.add_argument("--problems", type=int, default=100, help="problems of each subset")
    parser.add_argument("--sandbox", choices=("docker", "local"), default="docker")
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--max-tests", type=int, default=32)
    parser.add_argument("--max-test-chars", type=int, default=65_536)
    parser.add_argument("--timeout-s", type=float, default=6.0)
    parser.add_argument("--at-once", type=int, default=32, help="sandboxes running at once")
    parser.add_argument("--show", type=int, default=12, help="failures printed")
    sys.exit(asyncio.run(main(parser.parse_args())))
