"""DeepCoder's problems over real processes: each kind of test run by the
code check, and a whole episode of the recipe's task. The rows are the
dataset's own, cut to a few tests; the turns are written here, so no server
and no download is needed."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any

from looper_cookbook.environments.coding import CheckReport, run_tests
from looper_cookbook.episode_sessions.base import EpisodeSession, StopReason
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter
from looper_cookbook.message_formatters.unit_tests import tokenizers
from looper_cookbook.recipes.code_rl import tasks
from looper_cookbook.sandboxes.local import LocalSandbox
from looper_sdk import FinishReason, SampleSequence, SamplingParams

# An lcbv5 row: LeetCode's starter code, which names List and imports nothing.
SENIORS = {
    "problem": "Return the number of passengers who are strictly more than 60 years old.\n",
    "starter_code": (
        "class Solution:\n    def countSeniors(self, details: List[str]) -> int:\n        "
    ),
    "metadata": {"func_name": "countSeniors"},
    "tests": json.dumps(
        [
            {
                "input": '["7868190130M7522", "5303914400F9211", "9273338290F4010"]',
                "output": "2",
                "testtype": "functional",
            },
            {
                "input": '["1313579440F2036", "2921522980M5644"]',
                "output": "0",
                "testtype": "functional",
            },
            {"input": '["3988132605O4995"]', "output": "0", "testtype": "functional"},
        ]
    ),
}
SENIORS_SOLUTION = (
    "class Solution:\n"
    "    def countSeniors(self, details: List[str]) -> int:\n"
    "        return sum(int(detail[11:13]) > 60 for detail in details)\n"
)
# A TACO row that asks for a function, with the dataset's own first solution.
ANAGRAM = {
    "problem": "Complete the function to return true if the two arguments are anagrams.\n",
    "tests": json.dumps(
        {
            "fn_name": "is_anagram",
            "inputs": [["foefet", "toffee"], ["Buckethead", "DeathCubeK"], ["apple", "pale"]],
            "outputs": [[True], [True], [False]],
        }
    ),
}
ANAGRAM_SOLUTION = (
    "def is_anagram(test, original):\n\treturn sorted(original.lower()) == sorted(test.lower())\n"
)
# A Codeforces row: one program test.
MIN_MAX = {
    "problem": "Output the minimum of x and y, followed by the maximum.\n",
    "tests": '[{"input": "3\\n1 9\\n8 4\\n3 3", "output": "1 9\\n4 8\\n3 3"}]',
}
MIN_MAX_SOLUTION = (
    "for _ in range(int(input())):\n"
    "    x, y = map(int, input().split())\n"
    "    print(min(x, y), max(x, y))\n"
)


def problem(row: Mapping[str, Any]) -> tasks.Problem:
    made = tasks.problem(row, max_tests=32, max_test_chars=65_536)
    assert made is not None
    return made


async def check(
    row: Mapping[str, Any], code: str, prelude: str = tasks.LIVE_CODE_BENCH_IMPORTS
) -> CheckReport:
    async with await LocalSandbox.Config().build() as sandbox:
        return await run_tests(
            sandbox, code, problem(row).tests, timeout_s=10, python=sys.executable, prelude=prelude
        )


async def test_each_kind_of_deepcoder_test_passes_its_solution() -> None:
    assert (await check(SENIORS, SENIORS_SOLUTION)).all_passed
    assert (await check(ANAGRAM, ANAGRAM_SOLUTION)).all_passed
    assert (await check(MIN_MAX, MIN_MAX_SOLUTION)).all_passed


async def test_a_wrong_solution_fails_the_test_it_gets_wrong() -> None:
    report = await check(ANAGRAM, "def is_anagram(test, original):\n    return True\n")
    assert (report.passed, report.total) == (2, 3)
    assert report.failure is not None and report.failure.expected == "false"


async def test_leetcodes_starter_code_needs_the_prelude_live_code_bench_supplies() -> None:
    report = await check(SENIORS, SENIORS_SOLUTION, prelude="")
    assert report.passed == 0 and report.failure is not None
    assert "NameError: name 'List' is not defined" in report.failure.error


async def test_an_episode_checks_on_the_public_tests_and_is_scored_on_them_all() -> None:
    tokenizer = tokenizers.qwen()
    runtime = EpisodeSession.Runtime(
        formatter=Qwen3_5MessageFormatter(tokenizer, enable_thinking=False),
        params=SamplingParams(max_tokens=512),
    )

    def sampled(text: str) -> SampleSequence:
        tokens = tokenizer(text + "<|im_end|>")["input_ids"]
        return SampleSequence(
            token_ids=tokens, logprobs=[-1.0] * len(tokens), finish_reason=FinishReason.STOP
        )

    def call(code: str) -> str:
        return (
            f"<tool_call>\n<function=check_solution>\n<parameter=code>\n{code}\n</parameter>\n"
            "</function>\n</tool_call>"
        )

    # Right on the first two tests, which are all the tool runs, and wrong on the third.
    partly = "def is_anagram(test, original):\n    return True\n"
    task = tasks.task(
        problem(ANAGRAM),
        sandbox=LocalSandbox.Config(),
        python=sys.executable,
        public_tests=2,
        max_episode_tokens=16_384,
    )
    rewards = {}
    for name, answer in (("partly", partly), ("right", ANAGRAM_SOLUTION)):
        async with await task.build(runtime) as session:
            await session.start()
            assert await session.observe(sampled(call(partly))) is not None
            assert await session.observe(sampled(f"```python\n{answer}```")) is None
            episode = await session.finish()
        assert episode.stop_reason is StopReason.ANSWER
        assert json.loads(episode.messages[2]["content"]) == {
            "passed": True,
            "tests_passed": 2,
            "tests": 2,
        }
        rewards[name] = episode.reward
    # The tool's verdict was the same; the hidden third test tells the answers apart.
    assert rewards == {"partly": 0.0, "right": 1.0}
