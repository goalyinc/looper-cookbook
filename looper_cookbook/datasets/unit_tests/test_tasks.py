"""``TaskDataset`` (looper_cookbook/datasets/tasks.py): its batches and
passes, and what it refuses."""

from __future__ import annotations

import pytest

from looper_cookbook.datasets.tasks import TaskDataset
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.verifiers.final_number import FinalNumber

# Tasks that differ by what they ask: a dataset never builds what it batches.
TASKS = [
    AgentSession.Config(
        messages=[{"role": "user", "content": letter}],
        verifier=FinalNumber.Config(answer="42"),
        max_episode_tokens=64,
    )
    for letter in "abcdefg"
]


def test_a_pass_holds_every_task_once_in_an_order_of_its_own() -> None:
    dataset = TaskDataset(TASKS, batch_size=3, seed=1)
    assert len(dataset) == 3
    first = [dataset.batch(index) for index in range(3)]
    second = [dataset.batch(index) for index in range(3, 6)]
    # The last batch of a pass holds what is left.
    assert [len(batch) for batch in first] == [3, 3, 1]
    place = {id(task): index for index, task in enumerate(TASKS)}
    for batches in (first, second):
        assert sorted(place[id(task)] for batch in batches for task in batch) == list(range(7))
    assert first != second
    # The step alone decides the batch: a run called again reads the same one.
    assert dataset.batch(4) == second[1]
    other = TaskDataset(TASKS, batch_size=3, seed=2)
    assert [other.batch(index) for index in range(3)] != first


def test_a_dataset_refuses_what_cannot_be_batched() -> None:
    with pytest.raises(ValueError, match="tasks is empty"):
        TaskDataset([], batch_size=2)
    with pytest.raises(ValueError, match="batch_size must be at least 1, not 0"):
        TaskDataset(TASKS, batch_size=0)
    with pytest.raises(IndexError, match="0 or above, not -1"):
        TaskDataset(TASKS, batch_size=1).batch(-1)
