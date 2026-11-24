"""The dataset contracts.

A dataset is a run's data in batches, one per optimizer step. A loop reads
``dataset.batch(k)`` for the run's step ``k``, counted from 0.

**The run's step decides the data.** A batch depends on its index alone, so
a loop called again on the same run, after an error or a ``RunRolledBack``,
reads the batch of the step the run stands at, with no bookkeeping by its
caller.

**A dataset never ends.** ``len(dataset)`` is the batches of one pass over
the data, and any index from 0 up names a batch: past one pass, the next
pass's. A caller trains several passes by asking its loop for that many
steps.

``SupervisedDataset`` is the SFT loop's: a batch is a step's datums.
``RLDataset`` is the RL loop's: a batch is a step's tasks, each the config
of an episode session. Each is a signature alone, so any class with these
two methods is one.

``batch_positions`` is the order the cookbook's own datasets batch in: each
pass shuffled in an order of its own, which a seed and the pass fix.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Protocol

from looper_cookbook.episode_sessions.base import EpisodeSession
from looper_sdk import Datum

__all__ = ["RLDataset", "SupervisedDataset", "batch_positions"]


class SupervisedDataset(Protocol):
    """A run's datums, a batch per step."""

    def __len__(self) -> int:
        """The batches of one pass over the data."""
        ...

    def batch(self, index: int) -> Sequence[Datum]:
        """The datums of the run's step ``index``, counted from 0. Any index
        from 0 up names a batch: past one pass, the next pass's."""
        ...


class RLDataset(Protocol):
    """A run's tasks, a batch per step."""

    def __len__(self) -> int:
        """The batches of one pass over the tasks."""
        ...

    def batch(self, index: int) -> Sequence[EpisodeSession.Config]:
        """The tasks of the run's step ``index``, counted from 0. Any index
        from 0 up names a batch: past one pass, the next pass's."""
        ...


def batch_positions(count: int, *, batch_size: int, seed: int, index: int) -> list[int]:
    """The positions, in a list of ``count`` items, of the items batch
    ``index`` holds.

    A pass is ``ceil(count / batch_size)`` batches that hold every position
    once, in an order of the pass's own, which ``seed`` and the pass fix. Its
    last batch holds what is left. ``index`` counts batches from 0 through
    every pass, so the same index always names the same positions.
    """
    if count < 1 or batch_size < 1:
        raise ValueError(
            f"a pass needs an item and a batch size of at least 1, not {count} items in "
            f"batches of {batch_size}"
        )
    if index < 0:
        raise IndexError(f"a batch's index is a run's step, 0 or above, not {index}")
    this_pass, place = divmod(index, math.ceil(count / batch_size))
    order = list(range(count))
    random.Random(f"{seed}:{this_pass}").shuffle(order)
    start = place * batch_size
    return order[start : start + batch_size]
