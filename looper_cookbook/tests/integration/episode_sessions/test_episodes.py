"""Whole episodes over real processes: an agent session, an environment in a
real sandbox and a verifier in another.

The turns are written here, as a model would sample them, so no server is
needed. What is real is everything a turn leads to: the shell and the Python
that run the model's calls, and the sandbox the verifier's hidden tests run
in.
"""

from __future__ import annotations

import json
import os
import sys

from looper_cookbook.environments.bash import BashEnvironment
from looper_cookbook.environments.coding import CodeTest, CodingEnvironment
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.episode_sessions.base import EpisodeSession, StopReason
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter
from looper_cookbook.message_formatters.unit_tests import tokenizers
from looper_cookbook.sandboxes.local import LocalSandbox
from looper_cookbook.verifiers.code_tests import CodeTestsVerifier
from looper_cookbook.verifiers.final_number import FinalNumber
from looper_sdk import FinishReason, SampleSequence, SamplingParams

TOKENIZER = tokenizers.qwen()
FORMATTER = Qwen3_5MessageFormatter(TOKENIZER, enable_thinking=False)
RUNTIME = EpisodeSession.Runtime(formatter=FORMATTER, params=SamplingParams(max_tokens=256))
EXAMPLES = (CodeTest(input="1 2\n", output="3\n"),)
HIDDEN = (CodeTest(input="1 2\n", output="3\n"), CodeTest(input="10 20 30\n", output="60\n"))
TWO = "a, b = map(int, input().split())\nprint(a + b)\n"
ANY = "print(sum(map(int, input().split())))\n"


def sampled(text: str) -> SampleSequence:
    tokens = TOKENIZER(text + "<|im_end|>")["input_ids"]
    return SampleSequence(
        token_ids=tokens, logprobs=[-1.0] * len(tokens), finish_reason=FinishReason.STOP
    )


def call(name: str, argument: str, value: str) -> str:
    return (
        f"<tool_call>\n<function={name}>\n<parameter={argument}>\n{value}\n</parameter>\n"
        "</function>\n</tool_call>"
    )


def coding_task() -> AgentSession.Config:
    return AgentSession.Config(
        messages=[{"role": "user", "content": "Print the sum of the numbers on standard input."}],
        environment=CodingEnvironment.Config(
            sandbox=LocalSandbox.Config(), tests=EXAMPLES, python=sys.executable
        ),
        verifier=CodeTestsVerifier.Config(
            sandbox=LocalSandbox.Config(), tests=HIDDEN, python=sys.executable
        ),
        max_turns=3,
        max_episode_tokens=8192,
    )


async def test_a_program_checked_on_the_examples_is_scored_on_the_hidden_tests() -> None:
    async with await coding_task().build(RUNTIME) as session:
        await session.start()
        # It passes the one example the model may see, and not the hidden test of three numbers.
        assert await session.observe(sampled(call("check_solution", "code", TWO))) is not None
        assert await session.observe(sampled(f"```python\n{TWO}```")) is None
        episode = await session.finish()
    assert json.loads(episode.messages[2]["content"]) == {
        "passed": True,
        "tests_passed": 1,
        "tests": 1,
    }
    assert (episode.stop_reason, episode.reward) == (StopReason.ANSWER, 0.0)
    assert episode.metrics == {
        "passed": 1.0,
        "answered": 1.0,
        "turns": 2.0,
        "sampled_tokens": float(sum(len(turn.sequence.token_ids) for turn in episode.turns)),
        "truncated": 0.0,
    }


async def test_a_program_that_passes_the_hidden_tests_scores_one() -> None:
    async with await coding_task().build(RUNTIME) as session:
        await session.start()
        assert await session.observe(sampled(f"Here it is.\n```python\n{ANY}```")) is None
        episode = await session.finish()
    assert (episode.stop_reason, episode.reward) == (StopReason.ANSWER, 1.0)
    assert len(episode.turns) == 1


async def test_a_shells_files_are_there_for_the_next_turn_and_gone_after_the_loop() -> None:
    config = AgentSession.Config(
        messages=[{"role": "user", "content": "Work out 6 times 7 in the shell."}],
        environment=BashEnvironment.Config(sandbox=LocalSandbox.Config()),
        verifier=FinalNumber.Config(answer="42"),
        max_turns=3,
        max_episode_tokens=8192,
    )
    async with await config.build(RUNTIME) as session:
        await session.start()
        await session.observe(sampled(call("bash", "command", "echo $((6 * 7)) > n.txt; pwd")))
        await session.observe(sampled(call("bash", "command", "cat n.txt; cat missing.txt")))
        await session.observe(sampled("#### 42"))
        episode = await session.finish()
        assert session.environment is not None and session.environment.sandbox is not None
        directory = episode.messages[2]["content"].strip()
        assert os.path.isdir(directory)
    assert episode.messages[4]["content"] == (
        "42\ncat: missing.txt: No such file or directory\n[exit code 1]"
    )
    assert episode.reward == 1.0
    assert episode.metrics["failed"] == 1.0
    # Closing the session closed its environment, and that its sandbox.
    assert not os.path.exists(directory)
