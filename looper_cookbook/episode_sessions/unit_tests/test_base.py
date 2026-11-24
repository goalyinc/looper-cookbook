"""The episode session contract: an episode as its datum, the fields every
task has, what a session of one's own writes, sessions run at once, and what
a list of episodes comes to in numbers."""

from __future__ import annotations

import asyncio
import dataclasses
import math
from collections.abc import Sequence
from typing import Any, cast

import pytest

from looper_cookbook.episode_sessions.base import (
    Episode,
    EpisodeSession,
    StopReason,
    Turn,
    episode_statistics,
    evaluate,
    rollout,
)
from looper_cookbook.message_formatters.base import Message
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter
from looper_cookbook.message_formatters.unit_tests import tokenizers
from looper_cookbook.verifiers.base import Verifier
from looper_cookbook.verifiers.final_number import FinalNumber
from looper_sdk import Datum, FinishReason, SampleSequence, SamplingClient, SamplingParams

# The sessions here sample nothing, so they are handed no client.
NO_CLIENT = cast(SamplingClient, None)
RUNTIME = EpisodeSession.Runtime(
    formatter=Qwen3_5MessageFormatter(tokenizers.qwen()), params=SamplingParams(max_tokens=8)
)
QUESTION: Sequence[Message] = ({"role": "user", "content": "What is 6 times 7?"},)
RIGHT = FinalNumber.Config(answer="42")


def sampled(*tokens: int) -> SampleSequence:
    logprobs = [-0.5 * (index + 1) for index in range(len(tokens))]
    return SampleSequence(token_ids=tokens, logprobs=logprobs, finish_reason=FinishReason.STOP)


EPISODE = Episode(
    turns=(
        Turn(context=(1, 2, 3), sequence=sampled(10, 11)),
        Turn(context=(4,), sequence=sampled(12, 13, 14)),
    ),
    reward=0.5,
    stop_reason=StopReason.ANSWER,
    messages=({"role": "user", "content": "q"},),
    metrics={"turns": 2.0},
)


def test_an_episode_is_one_datum_with_weight_on_what_was_sampled() -> None:
    assert EPISODE.datum("step-1-task-0", 4) == Datum(
        tokens=(1, 2, 3, 10, 11, 4, 12, 13, 14),
        weights=(0, 0, 0, 1, 1, 0, 1, 1, 1),
        columns={"logprobs": (0.0, 0.0, 0.0, -0.5, -1.0, 0.0, -0.5, -1.0, -1.5)},
        reward=0.5,
        group_id="step-1-task-0",
        group_size=4,
    )


class Replayed(EpisodeSession):
    """A session of one's own: it returns an episode it was given."""

    def __init__(self, episode: Episode) -> None:
        self._episode = episode
        self.closes = 0

    async def run(self, sampling_client: SamplingClient) -> Episode:
        return self._episode

    async def close(self) -> None:
        self.closes += 1


async def test_a_session_of_ones_own_writes_run_and_close() -> None:
    session = Replayed(EPISODE)
    with pytest.raises(RuntimeError, match="boom"):
        async with session as entered:
            assert entered is session
            raise RuntimeError("boom")
    assert session.closes == 1


def test_a_session_without_run_or_close_is_not_one() -> None:
    class Unfinished(EpisodeSession):
        async def close(self) -> None:
            return None

    with pytest.raises(TypeError, match="abstract"):
        Unfinished()  # pyright: ignore[reportAbstractUsage]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Scripted(EpisodeSession.Config):
    """A session's own config: every task's fields, and its own. Its session
    does what a test wrote: it waits, fails or returns an episode, and notes
    each of its closes in ``closed``."""

    messages: Sequence[Message] = QUESTION
    verifier: Verifier.Config = RIGHT
    max_episode_tokens: int = 64
    reward: float = 0.0
    fails: bool = False
    waits: bool = False
    closed: list[None] = dataclasses.field(default_factory=list)

    async def build(self, runtime: EpisodeSession.Runtime) -> EpisodeSession:
        return _Scripted(self)


class _Scripted(EpisodeSession):
    def __init__(self, task: Scripted) -> None:
        self._task = task

    async def run(self, sampling_client: SamplingClient) -> Episode:
        if self._task.waits:
            await asyncio.Event().wait()
        # Let the sessions started with this one reach their own run first.
        await asyncio.sleep(0)
        if self._task.fails:
            raise RuntimeError("the sandbox is gone")
        return dataclasses.replace(EPISODE, reward=self._task.reward)

    async def close(self) -> None:
        self._task.closed.append(None)


def test_a_config_is_a_task_only_through_a_session_that_builds_it() -> None:
    with pytest.raises(TypeError, match="abstract"):
        EpisodeSession.Config(  # pyright: ignore[reportAbstractUsage]
            messages=QUESTION, verifier=RIGHT, max_episode_tokens=64
        )


def test_a_sessions_own_config_holds_every_tasks_fields_beside_its_own() -> None:
    task = Scripted(reward=1.0)
    assert isinstance(task, EpisodeSession.Config)
    assert (task.messages, task.verifier, task.max_episode_tokens) == (QUESTION, RIGHT, 64)
    # A task has no environment and one turn unless it says so.
    assert (task.environment, task.max_turns) == (None, 1)
    assert task.reward == 1.0


@pytest.mark.parametrize(
    "settings,why",
    [
        ({"messages": []}, "messages is empty"),
        ({"verifier": object()}, "verifier must be a verifier's Config"),
        ({"environment": object()}, "environment must be an environment's Config"),
        ({"max_turns": 0}, "max_turns must be at least 1, not 0"),
        ({"max_episode_tokens": 1}, "max_episode_tokens must hold a prompt and a turn"),
    ],
)
def test_a_config_refuses_what_is_not_a_task(settings: dict[str, Any], why: str) -> None:
    # The contract's own check, so a session's own config refuses the same.
    with pytest.raises(ValueError, match=why):
        Scripted(**settings)


async def test_attempts_run_at_once_and_return_in_their_configs_order() -> None:
    tasks = [Scripted(reward=1.0), Scripted(reward=2.0), Scripted(reward=3.0)]
    # A config given twice is attempted twice: that is a group.
    episodes = await rollout([*tasks, tasks[0]], RUNTIME, NO_CLIENT)
    assert [episode.reward for episode in episodes] == [1.0, 2.0, 3.0, 1.0]
    assert [len(task.closed) for task in tasks] == [2, 1, 1]


async def test_one_failed_attempt_cancels_and_closes_the_rest() -> None:
    tasks = [Scripted(waits=True), Scripted(fails=True), Scripted(waits=True)]
    with pytest.raises(RuntimeError, match="the sandbox is gone"):
        await rollout(tasks, RUNTIME, NO_CLIENT)
    # The two that would have waited forever were cancelled, and every session closed.
    assert [len(task.closed) for task in tasks] == [1, 1, 1]


async def test_an_evaluation_attempts_each_task_once_and_names_its_metrics() -> None:
    tasks = [Scripted(reward=1.0), Scripted(reward=0.0)]
    assert await evaluate(tasks, RUNTIME, NO_CLIENT) == {
        "eval/episodes": 2.0,
        "eval/reward": 0.5,
        "eval/reward_stderr": pytest.approx(0.5),
        "eval/turns": 2.0,
        "eval/turns_stderr": 0.0,
        "eval/stop/answer": 1.0,
        "eval/stop/answer_stderr": 0.0,
        "eval/stop/max_turns": 0.0,
        "eval/stop/max_turns_stderr": 0.0,
        "eval/stop/max_turn_tokens": 0.0,
        "eval/stop/max_turn_tokens_stderr": 0.0,
        "eval/stop/max_episode_tokens": 0.0,
        "eval/stop/max_episode_tokens_stderr": 0.0,
    }
    # Each session was built, run once and closed.
    assert [len(task.closed) for task in tasks] == [1, 1]


async def test_an_evaluation_needs_a_task() -> None:
    with pytest.raises(ValueError, match="tasks is empty: an evaluation needs a task"):
        await evaluate([], RUNTIME, NO_CLIENT)


def scored(reward: float, **metrics: float) -> Episode:
    """An episode of no turn that answered, with a reward and metrics."""
    return Episode(
        turns=(), reward=reward, stop_reason=StopReason.ANSWER, messages=(), metrics=metrics
    )


def test_statistics_are_each_metrics_mean_and_its_standard_error() -> None:
    episodes = [scored(1.0, turns=1.0), scored(0.0, turns=3.0), scored(1.0, turns=2.0, cut=1.0)]
    statistics = episode_statistics(episodes)
    assert statistics["reward"] == pytest.approx((2 / 3, math.sqrt(1 / 3) / math.sqrt(3)))
    assert statistics["turns"] == pytest.approx((2.0, 1.0 / math.sqrt(3)))
    # Over the one episode that reports it, where nothing estimates an error.
    assert statistics["cut"] == (1.0, 0.0)


def test_statistics_hold_the_share_each_stop_reason_ended() -> None:
    answered = scored(1.0)
    episodes = [
        answered,
        answered,
        dataclasses.replace(answered, reward=0.0, stop_reason=StopReason.MAX_TURN_TOKENS),
        dataclasses.replace(answered, reward=0.0, stop_reason=StopReason.MAX_EPISODE_TOKENS),
    ]
    statistics = episode_statistics(episodes)
    # Every reason has its share, so a log's columns are the same at every step.
    assert {name: mean for name, (mean, _) in statistics.items() if name != "reward"} == {
        "stop/answer": 0.5,
        "stop/max_turns": 0.0,
        "stop/max_turn_tokens": 0.25,
        "stop/max_episode_tokens": 0.25,
    }
    assert statistics["stop/max_turn_tokens"][1] == pytest.approx(0.25)
