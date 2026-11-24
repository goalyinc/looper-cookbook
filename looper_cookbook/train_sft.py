"""The explicit SFT loop, written directly over ``TrainingClient``.

One optimizer step per batch. The run's step ``k``, counted from 0, trains
``dataset.batch(k)``. A step is its batch's ``forward_backward`` calls,
whose gradients accumulate, then an ``optim_step`` that names its learning
rate. A run's optimizer keeps no schedule, so the loop computes each rate
from the run's step.

**The run's step decides the data,** and the rate. Called again on the same
run, after an error or a ``RunRolledBack``, the loop trains the batch of the
step the run stands at, at that step's rate, with no bookkeeping by its
caller.

The loop submits ahead. Awaiting a submit returns once the service has
accepted and ordered the operation; awaiting its ``result()`` returns what it
computed. So the loop submits a step, then prepares and submits the next
ones while Looper trains, and awaits a step's results only once
``steps_ahead`` later steps are accepted. The GPUs never wait for
the client to read a batch or for a round trip.

What it leaves to its caller: the run, created with its ``TrainingConfig``;
the dataset, as ``datasets/`` makes one of a file of conversations; and the
save, two lines after ``train`` returns. An error raises as its operation's
result is awaited, and up to ``steps_ahead`` later steps are accepted by
then. They still train unless the run is deleted or its session closes.
"""

from __future__ import annotations

import collections
import dataclasses
import math
from collections.abc import Callable, Iterator, Sequence

from looper_cookbook.datasets.base import SupervisedDataset
from looper_sdk import (
    CrossEntropy,
    Datum,
    ForwardBackwardResult,
    Operation,
    StepResult,
    TrainingClient,
)

__all__ = ["SCHEDULES", "Config", "learning_rate", "train"]

SCHEDULES = ("constant", "linear", "cosine")


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
    """How the loop steps a run.

    Attributes:
        learning_rate: The rate of the run's first step, which the schedule
            starts from.
        schedule: How the rate falls over the run's steps. ``"linear"`` and
            ``"cosine"`` fall to 0, which the step after the last would
            train at; ``"constant"`` keeps the first rate.
        steps_ahead: Steps accepted beyond the one whose results the loop
            awaits. 0 awaits each step before it prepares the next batch, so
            the GPUs idle meanwhile. More overlaps the client's work
            with training, and leaves that many steps accepted when an error
            raises.
        max_call_tokens: The most tokens one ``forward_backward`` carries. A
            larger batch goes over several calls, whose gradients accumulate
            into the one step. A call's payload holds 16 MiB,
            about 7 bytes a token.
    """

    learning_rate: float
    schedule: str = "linear"
    steps_ahead: int = 1
    max_call_tokens: int = 1_000_000

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError(
                f"learning_rate must be finite and at least 0, not {self.learning_rate}"
            )
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, not {self.schedule!r}")
        if self.steps_ahead < 0:
            raise ValueError(f"steps_ahead must be at least 0, not {self.steps_ahead}")
        if self.max_call_tokens < 1:
            raise ValueError(f"max_call_tokens must be at least 1, not {self.max_call_tokens}")


def learning_rate(config: Config, step: int, steps: int) -> float:
    """The rate of the run's step ``step``, counted from 0, of ``steps``."""
    progress = step / steps
    if config.schedule == "linear":
        return config.learning_rate * (1 - progress)
    if config.schedule == "cosine":
        return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
    return config.learning_rate


async def train(
    training_client: TrainingClient,
    dataset: SupervisedDataset,
    config: Config,
    *,
    steps: int,
    on_step: Callable[[StepResult], None] | None = None,
) -> list[StepResult]:
    """Trains ``training_client``'s run up to its step ``steps``, one step
    per batch of ``dataset``.

    Args:
        training_client: The run's client, with no step of its own in
            flight. The loop starts at the run's current step, so a run
            that has trained takes only the steps left, each with its own
            batch, at the rate its schedule gives it.
        dataset: The datums, a batch per step. Each batch is read as its
            step is submitted.
        config: The schedule, and how far ahead the loop submits.
        steps: The step the run ends at, and the schedule's length.
        on_step: Called with each step's result, in step order.

    Returns:
        The results of the steps it trained, in order.
    """
    first = (await training_client.status()).step
    accepted: collections.deque[_Step] = collections.deque()
    results = []
    for step in range(first, steps):
        batch = dataset.batch(step)
        if not batch:
            raise ValueError(f"the batch of step {step + 1} is empty: a step needs a target")
        # Accepted in this order: the batch's calls, then the step that applies them.
        backwards = [
            await training_client.forward_backward(call, CrossEntropy())
            for call in _calls(batch, config.max_call_tokens)
        ]
        rate = learning_rate(config, step, steps)
        accepted.append(
            _Step(step + 1, backwards, await training_client.optim_step(learning_rate=rate))
        )
        if len(accepted) > config.steps_ahead:
            results.append(await _finish(accepted.popleft(), on_step))
    while accepted:
        results.append(await _finish(accepted.popleft(), on_step))
    return results


@dataclasses.dataclass(frozen=True, slots=True)
class _Step:
    """A step the service accepted: its number once applied, and its operations."""

    number: int
    backwards: list[Operation[ForwardBackwardResult]]
    applied: Operation[StepResult]


async def _finish(step: _Step, on_step: Callable[[StepResult], None] | None) -> StepResult:
    # The calls first: a failed call's error says why, where its step says only that one failed.
    for backward in step.backwards:
        await backward.result()
    result = await step.applied.result()
    if result.step != step.number:
        raise RuntimeError(
            f"the run took its step {result.step} where the loop expected step {step.number}: "
            "another writer stepped it, or it rolled back; the rates named since then "
            "belong to other steps"
        )
    if on_step is not None:
        on_step(result)
    return result


def _calls(batch: Sequence[Datum], max_tokens: int) -> Iterator[list[Datum]]:
    """``batch`` in order, as calls of at most ``max_tokens`` tokens each; a
    longer datum is a call of its own."""
    call: list[Datum] = []
    size = 0
    for datum in batch:
        if call and size + len(datum.tokens) > max_tokens:
            yield call
            call, size = [], 0
        call.append(datum)
        size += len(datum.tokens)
    if call:
        yield call
