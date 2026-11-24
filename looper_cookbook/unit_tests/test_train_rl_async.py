"""The asynchronous RL loop's parts that need no server
(looper_cookbook/train_rl_async.py): how many groups may be out, the order
tasks are read in, and the group queue, driven with sessions a test
releases. The loop itself runs against a real server
(``tests/e2e/test_train_rl_async.py``)."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Sequence
from typing import cast

import pytest

from looper_cookbook import train_rl, train_rl_async
from looper_cookbook.datasets.tasks import TaskDataset
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.episode_sessions.base import Episode, EpisodeSession, StopReason, Turn
from looper_cookbook.message_formatters.base import Message
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter
from looper_cookbook.message_formatters.unit_tests import tokenizers
from looper_cookbook.verifiers.base import Verifier
from looper_cookbook.verifiers.final_number import FinalNumber
from looper_sdk import FinishReason, SampleSequence, SamplingClient, SamplingParams, Version

# The sessions here sample nothing, so they are handed no client.
NO_CLIENT = cast(SamplingClient, None)
SAMPLED = SamplingParams(max_tokens=4)
RUNTIME = EpisodeSession.Runtime(
    formatter=Qwen3_5MessageFormatter(tokenizers.qwen()), params=SAMPLED
)
QUESTION: Sequence[Message] = ({"role": "user", "content": "What is 6 times 7?"},)
RIGHT = FinalNumber.Config(answer="42")
GROUP_SIZE = 2
# Seven tasks in batches of three: a pass's batches hold 3, 3 and 1.
DATASET = TaskDataset(
    [
        AgentSession.Config(
            messages=[{"role": "user", "content": letter}], verifier=RIGHT, max_episode_tokens=64
        )
        for letter in "abcdefg"
    ],
    batch_size=3,
)


def episode(reward: float, step: int) -> Episode:
    """An episode of one turn, sampled at the run's ``step``."""
    version = Version(run_id="run-1", epoch=0, step=step)
    sequence = SampleSequence(
        token_ids=(7, 8), logprobs=(-0.5, -1.0), finish_reason=FinishReason.STOP, version=version
    )
    return Episode(
        turns=(Turn(context=(1, 2, 3), sequence=sequence),),
        reward=reward,
        stop_reason=StopReason.ANSWER,
        messages=(),
        metrics={},
    )


class Stage:
    """What a test decides of one task's sessions: when they finish, the
    version that sampled them and whether they fail. It counts the sessions
    built and closed."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.version = 0
        self.fails = False
        self.built = 0
        self.closed = 0


@dataclasses.dataclass(frozen=True, kw_only=True)
class Staged(EpisodeSession.Config):
    """A task whose sessions finish when a test releases them."""

    messages: Sequence[Message] = QUESTION
    verifier: Verifier.Config = RIGHT
    max_episode_tokens: int = 64
    reward: float
    stage: Stage

    async def build(self, runtime: EpisodeSession.Runtime) -> EpisodeSession:
        self.stage.built += 1
        return _Staged(self)


class _Staged(EpisodeSession):
    def __init__(self, task: Staged) -> None:
        self._task = task

    async def run(self, sampling_client: SamplingClient) -> Episode:
        await self._task.stage.release.wait()
        if self._task.stage.fails:
            raise RuntimeError("the sandbox is gone")
        return episode(self._task.reward, self._task.stage.version)

    async def close(self) -> None:
        self._task.stage.closed += 1


def staged(count: int) -> list[Staged]:
    """Tasks told apart by their rewards: 1.0, 2.0 and so on."""
    return [Staged(reward=float(index + 1), stage=Stage()) for index in range(count)]


def queue_of(tasks: Sequence[Staged]) -> train_rl_async.RolloutQueue:
    return train_rl_async.RolloutQueue(iter(tasks), RUNTIME, NO_CLIENT, group_size=GROUP_SIZE)


def rewards(groups: Sequence[Sequence[Episode]]) -> list[list[float]]:
    return [[episode.reward for episode in group] for group in groups]


async def settled() -> None:
    """Lets the sessions that were started run until each waits."""
    for _ in range(8):
        await asyncio.sleep(0)


def test_room_is_the_groups_of_the_steps_a_group_started_now_can_train_in() -> None:
    def room(step: int, lag: int, steps: int = 10) -> int:
        return train_rl_async.room(DATASET, step, steps=steps, max_policy_lag=lag)

    # No lag is one batch: the step's own.
    assert [room(step, lag=0) for step in range(4)] == [3, 3, 1, 3]
    assert (room(0, lag=1), room(1, lag=1), room(1, lag=2)) == (6, 4, 7)
    # Nothing starts for a step past the run's last.
    assert (room(8, lag=3), room(9, lag=3), room(10, lag=3)) == (4, 3, 0)


def test_tasks_are_read_in_their_batches_order_from_the_runs_step() -> None:
    tasks = train_rl_async.tasks_from(DATASET, 2)
    read = [next(tasks) for _ in range(7)]
    # The last batch of one pass, then the next pass's.
    assert read == [*DATASET.batch(2), *DATASET.batch(3), *DATASET.batch(4)]


class Short:
    """A dataset whose batches run out after its first."""

    def __len__(self) -> int:
        return 1

    def batch(self, index: int) -> Sequence[EpisodeSession.Config]:
        return DATASET.batch(0) if index == 0 else []


def test_an_empty_batch_is_refused_where_it_is_read() -> None:
    tasks = train_rl_async.tasks_from(Short(), 0)
    assert [next(tasks) for _ in range(3)] == DATASET.batch(0)
    with pytest.raises(ValueError, match="the batch of step 2 is empty"):
        next(tasks)
    with pytest.raises(ValueError, match="the batch of step 2 is empty"):
        train_rl_async.room(Short(), 0, steps=4, max_policy_lag=1)


def test_lag_is_how_far_each_groups_oldest_turn_trails_the_step() -> None:
    groups = [[episode(1.0, 3), episode(0.0, 2)], [episode(1.0, 3), episode(1.0, 3)]]
    assert train_rl_async.lag_metrics(groups, 3) == {"policy_lag": 0.5, "policy_lag_max": 1.0}


def test_the_config_adds_the_lag_to_the_rl_loops_settings() -> None:
    config = train_rl_async.Config(params=SAMPLED)
    assert isinstance(config, train_rl.Config)
    assert (config.max_policy_lag, config.group_size) == (1, 8)
    with pytest.raises(ValueError, match="max_policy_lag must be 0 or above, not -1"):
        train_rl_async.Config(params=SAMPLED, max_policy_lag=-1)
    # The RL loop's own checks still hold.
    with pytest.raises(ValueError, match="group_size must be at least 2"):
        train_rl_async.Config(params=SAMPLED, group_size=1)


async def test_admit_starts_the_next_tasks_until_its_room_is_taken() -> None:
    tasks = staged(3)
    queue = queue_of(tasks)
    queue.admit(2)
    await settled()
    assert [task.stage.built for task in tasks] == [GROUP_SIZE, GROUP_SIZE, 0]
    # Room that is taken starts nothing.
    queue.admit(2)
    await settled()
    assert [task.stage.built for task in tasks] == [GROUP_SIZE, GROUP_SIZE, 0]
    await queue.close()
    assert [task.stage.closed for task in tasks] == [GROUP_SIZE, GROUP_SIZE, 0]


async def test_a_step_takes_the_first_groups_to_finish_oldest_version_first() -> None:
    slow, old, new, late = tasks = staged(4)
    new.stage.version = 1
    queue = queue_of(tasks)
    queue.admit(4)
    for task in (new, late, old):
        task.stage.release.set()
    groups, stale = await queue.take(3, step=1, max_policy_lag=1)
    # The older version first and, within it, in the order the groups started.
    assert rewards(groups) == [[2.0, 2.0], [4.0, 4.0], [3.0, 3.0]]
    assert stale == 0
    # The group still in flight is left to finish.
    assert slow.stage.closed == 0
    await queue.close()
    assert slow.stage.closed == GROUP_SIZE


async def test_a_group_past_the_lag_is_dropped_and_another_task_starts_in_its_place() -> None:
    aged, fresh, replacement = tasks = staged(3)
    fresh.stage.version = 2
    queue = queue_of(tasks)
    queue.admit(2)
    taking = asyncio.ensure_future(queue.take(1, step=2, max_policy_lag=1))
    aged.stage.release.set()
    await settled()
    # Sampled two steps back with a lag of one: dropped, and its place refilled.
    assert not taking.done()
    assert replacement.stage.built == GROUP_SIZE
    fresh.stage.release.set()
    groups, stale = await taking
    assert rewards(groups) == [[2.0, 2.0]]
    assert stale == 1
    await queue.close()


async def test_a_finished_group_that_waits_ages_too() -> None:
    first, second, third, fourth = tasks = staged(4)
    third.stage.version = 1
    queue = queue_of(tasks)
    queue.admit(2)
    first.stage.release.set()
    second.stage.release.set()
    await settled()
    groups, stale = await queue.take(1, step=0, max_policy_lag=0)
    assert (rewards(groups), stale) == ([[1.0, 1.0]], 0)
    # The second group finished in time, and waits through a step that did not take it.
    queue.admit(2)
    third.stage.release.set()
    groups, stale = await queue.take(1, step=1, max_policy_lag=0)
    assert (rewards(groups), stale) == ([[3.0, 3.0]], 1)
    # The dropped group's place went to the next task.
    assert fourth.stage.built == GROUP_SIZE
    await queue.close()


async def test_an_attempts_error_raises_from_take_and_close_cancels_the_rest() -> None:
    failing, waiting = tasks = staged(2)
    failing.stage.fails = True
    queue = queue_of(tasks)
    queue.admit(2)
    failing.stage.release.set()
    with pytest.raises(RuntimeError, match="the sandbox is gone"):
        await queue.take(2, step=0, max_policy_lag=0)
    await queue.close()
    # No session is left running: the failed group's and the waiting one's are closed.
    assert [task.stage.closed for task in tasks] == [GROUP_SIZE, GROUP_SIZE]
    assert waiting.stage.built == GROUP_SIZE


async def test_take_refuses_to_wait_for_more_groups_than_admit_made_room_for() -> None:
    queue = queue_of(staged(1))
    queue.admit(1)
    with pytest.raises(RuntimeError, match="take waits for 2 groups, and admit made room for 1"):
        await queue.take(2, step=0, max_policy_lag=0)
    await queue.close()
