"""Tasks in batches: the cookbook's ``RLDataset``.

``TaskDataset`` holds a list of tasks, each an episode session's config, and
gives the RL loop a batch of them for each step of a run. It builds none of
them: a task is data until the loop builds a session for an attempt.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

from looper_cookbook.datasets.base import batch_positions
from looper_cookbook.episode_sessions.base import EpisodeSession

__all__ = ["TaskDataset"]


@dataclasses.dataclass(frozen=True, slots=True)
class TaskDataset:
    """Tasks in batches, over as many passes as a run has steps for.

    Each pass is in an order of its own, which ``seed`` and the pass fix, so
    a run called again reads the same batch at the same step. A pass's last
    batch holds what is left.

    Attributes:
        tasks: The tasks, each an episode session's config.
        batch_size: The tasks of one step.
        seed: Fixes each pass's order.
    """

    tasks: Sequence[EpisodeSession.Config]
    _: dataclasses.KW_ONLY
    batch_size: int
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("tasks is empty: a step needs a task")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, not {self.batch_size}")

    def __len__(self) -> int:
        return math.ceil(len(self.tasks) / self.batch_size)

    def batch(self, index: int) -> list[EpisodeSession.Config]:
        positions = batch_positions(
            len(self.tasks), batch_size=self.batch_size, seed=self.seed, index=index
        )
        return [self.tasks[position] for position in positions]
