"""The asynchronous RL loop: sampling does not wait for training.

``train_rl`` takes turns. It samples a batch, trains it, publishes, and only
then samples again: the sampler waits while the trainer works, and the
trainer waits for a whole batch. This loop overlaps the two:

1. **Attempt, ahead of training.** The loop keeps the groups of the next
   ``max_policy_lag + 1`` steps in flight or waiting to train. A group is
   ``group_size`` episode sessions of one task: the next task of the
   dataset's batches, read in order. Its turns are sampled from whatever
   weights the sampler serves while it runs.
2. **Train what finished first.** A step trains as many groups as the
   dataset's batch of that step holds: the first that finish, oldest
   version first. Then ``optim_step``, ``publish_weights`` and, when one is
   due, ``save_state``, as ``train_rl`` submits them.
3. **Evaluate,** when due: one attempt at each held-out task, on the weights
   just published, while the next groups go on sampling.

**Where the rollouts run.** In this process, on its one event loop: there is
no worker and no thread. ``RolloutQueue.admit`` starts each group as a task
of its own and returns at once. A task goes on whenever the loop awaits
something else: a step's calls, a publish, an evaluation. So groups sample
while the run trains because the work is elsewhere, on the sampler's GPUs
and the trainer's, and this process only waits for their replies. ``admit``
is called where room appears: before the first step, after each publish, and
when a stale group is dropped.

**A group trains on weights at most ``max_policy_lag`` steps newer than
those that sampled it.** A group's version is its oldest turn's,
and its lag is the run's step it trains at, less that version. The loss
reads the sampler's log probabilities against the trained policy's in its
clipped ratio, which is what corrects the gap.

**The bound is kept in two places.**

- *Where a group starts.* The loop starts a group only while fewer are in
  flight or waiting than the next ``max_policy_lag + 1`` steps train. So a
  group that starts now, trained in the order it started, trains within
  the bound.
- *Where a group is read.* Groups finish out of order, so a slow one can be
  overtaken. One that has aged past the bound when a step reads it is
  dropped, counted in that step's ``groups_stale``, and the dataset's next
  task starts in its place.

``max_policy_lag=0`` is ``train_rl``'s loop: one batch in flight, trained
whole, in the batch's order.

**What the overlap costs.**

- *The run's step no longer decides the data.* A step trains the groups
  that finish first. Called again on the same run, the loop reads on from
  the batch of the step the run stands at, so across the two calls a task
  may train twice, or wait for the dataset's next pass.
- *Slow episodes train less often.* An episode is as old as its first
  turn, so a group that runs longer than ``max_policy_lag + 1`` steps take
  is always dropped. Size the lag from the longest episode, counted in
  steps, and watch ``groups_stale``.
- *More runs at once.* Up to ``max_policy_lag + 1`` batches of sessions are
  in flight together, each with its sandbox. A sampler holds the requests
  past its ``max_concurrent_sequences`` in its queue.

An attempt's error raises when the loop next reads groups. The groups in
flight are cancelled first, and their sessions closed.

What the loop leaves to its caller is what ``train_rl`` leaves: the run, its
sampler, the tasks, held-out ones too, and the last save.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import statistics
from collections.abc import Callable, Iterator, Sequence

from looper_cookbook import train_rl
from looper_cookbook.datasets.base import RLDataset
from looper_cookbook.episode_sessions.base import Episode, EpisodeSession, evaluate, rollout
from looper_cookbook.message_formatters.base import MessageFormatter
from looper_cookbook.train_rl import StepReport
from looper_sdk import SamplingClient, TrainingClient, Version

__all__ = ["Config", "RolloutQueue", "lag_metrics", "room", "tasks_from", "train"]


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config(train_rl.Config):
    """``train_rl``'s settings, and how far sampling may run ahead of training.

    Attributes:
        max_policy_lag: The most steps a group's oldest turn may trail the
            step that trains it. The loop keeps the groups of this many
            steps beyond the next one in flight, and drops a group that
            finishes older. 0 samples one batch at a time, as ``train_rl``
            does.
    """

    max_policy_lag: int = 1

    def __post_init__(self) -> None:
        # Named, not super(): a dataclass with slots is rebuilt, and the bare form breaks.
        train_rl.Config.__post_init__(self)
        if self.max_policy_lag < 0:
            raise ValueError(f"max_policy_lag must be 0 or above, not {self.max_policy_lag}")


def _batch(dataset: RLDataset, index: int) -> Sequence[EpisodeSession.Config]:
    """The tasks of the run's step ``index``, which a dataset may not leave empty."""
    tasks = dataset.batch(index)
    if not tasks:
        raise ValueError(f"the batch of step {index + 1} is empty: a step needs a task")
    return tasks


def tasks_from(dataset: RLDataset, first: int) -> Iterator[EpisodeSession.Config]:
    """The dataset's tasks in the order its batches hold them, from the
    batch of step ``first`` on, pass after pass. It never ends."""
    for index in itertools.count(first):
        yield from _batch(dataset, index)


def room(dataset: RLDataset, step: int, *, steps: int, max_policy_lag: int) -> int:
    """How many groups may be in flight or waiting while the run stands at
    ``step``: those the steps from ``step`` to ``step + max_policy_lag``
    train, and none of a step past the run's last.

    A group that starts now is sampled at ``step`` or later. With fewer
    groups than this ahead of it, trained in the order it started, it
    trains by step ``step + max_policy_lag``: within the bound.
    """
    last = min(step + max_policy_lag + 1, steps)
    return sum(len(_batch(dataset, index)) for index in range(step, last))


def lag_metrics(groups: Sequence[Sequence[Episode]], step: int) -> dict[str, float]:
    """How far the groups trail the weights that train them, the run's step
    ``step``: ``policy_lag``, the mean over the groups of the steps between
    a group's oldest turn and ``step``, and ``policy_lag_max``, the most."""
    lags = [step - train_rl.oldest(group).step for group in groups]
    return {"policy_lag": statistics.fmean(lags), "policy_lag_max": float(max(lags))}


@dataclasses.dataclass(frozen=True, slots=True)
class _Finished:
    """A group that finished: its place in the order groups started, its
    episodes, and the oldest version that sampled one of its turns."""

    started: int
    episodes: list[Episode]
    version: Version


class RolloutQueue:
    """The groups in flight, and the finished ones waiting to train.

    ``admit`` starts groups, each ``group_size`` sessions of the next task,
    until a given number are in flight or waiting. ``take`` waits for the
    groups a step trains, and drops the stale ones. ``close`` cancels what
    is still in flight.

    Its tasks are an iterator that never ends, as ``tasks_from`` is.
    """

    def __init__(
        self,
        tasks: Iterator[EpisodeSession.Config],
        runtime: EpisodeSession.Runtime,
        sampling_client: SamplingClient,
        *,
        group_size: int,
    ) -> None:
        self._tasks = tasks
        self._runtime = runtime
        self._sampling_client = sampling_client
        self._group_size = group_size
        self._room = 0
        self._started = itertools.count()
        # Each group in flight, with its place in the order groups started.
        self._flights: dict[asyncio.Future[list[Episode]], int] = {}
        self._ready: list[_Finished] = []

    def admit(self, room: int) -> None:
        """Starts groups until ``room`` are in flight or waiting. The queue
        keeps that many: ``take`` starts another task for each group it
        drops."""
        self._room = room
        while len(self._flights) + len(self._ready) < self._room:
            attempts = [next(self._tasks)] * self._group_size
            flight = asyncio.ensure_future(
                rollout(attempts, self._runtime, self._sampling_client)
            )
            self._flights[flight] = next(self._started)

    async def take(
        self, count: int, *, step: int, max_policy_lag: int
    ) -> tuple[list[list[Episode]], int]:
        """Waits for ``count`` groups to train at the run's step ``step``.

        A finished group whose oldest turn trails ``step`` by more than
        ``max_policy_lag`` steps is dropped, and another task starts in its
        place. ``admit`` must have made room for ``count`` groups.

        Returns:
            The groups, oldest version first and, within one version, in
            the order they started; and how many groups were dropped.

        Raises:
            RuntimeError: ``admit`` made room for fewer than ``count`` groups.
            Exception: An attempt's error, as ``rollout`` raises it.
        """
        if count > self._room:
            raise RuntimeError(
                f"take waits for {count} groups, and admit made room for {self._room}"
            )
        stale = 0
        while True:
            fresh = [group for group in self._ready if step - group.version.step <= max_policy_lag]
            stale += len(self._ready) - len(fresh)
            self._ready = fresh
            self.admit(self._room)
            if len(self._ready) >= count:
                self._ready.sort(
                    key=lambda group: (group.version.epoch, group.version.step, group.started)
                )
                taken, self._ready = self._ready[:count], self._ready[count:]
                return [group.episodes for group in taken], stale
            done, _ = await asyncio.wait(self._flights, return_when=asyncio.FIRST_COMPLETED)
            for flight in done:
                started = self._flights.pop(flight)
                episodes = flight.result()
                self._ready.append(_Finished(started, episodes, train_rl.oldest(episodes)))

    async def close(self) -> None:
        """Cancels the groups in flight, and returns once each has closed
        its sessions. Finished groups that never trained are dropped."""
        for flight in self._flights:
            flight.cancel()
        await asyncio.gather(*self._flights, return_exceptions=True)
        self._flights.clear()
        self._ready.clear()


async def train(
    training_client: TrainingClient,
    sampling_client: SamplingClient,
    dataset: RLDataset,
    config: Config,
    *,
    formatter: MessageFormatter,
    steps: int,
    eval_tasks: Sequence[EpisodeSession.Config] = (),
    on_step: Callable[[StepReport], None] | None = None,
) -> list[StepReport]:
    """Trains ``training_client``'s run up to its step ``steps``, on episodes
    ``sampling_client`` samples while the run trains.

    Args:
        training_client: The run's client, with no step of its own in
            flight. The loop starts at the run's current step.
        sampling_client: The run's sampler, created by its training client.
            The loop publishes to it before its first sample and after each
            step.
        dataset: The tasks. The loop reads its batches in order, from the
            batch of the run's step, and a step trains as many groups as
            its batch holds.
        config: How episodes are sampled and trained, how far sampling may
            run ahead, and when the loop evaluates and saves.
        formatter: The model's message formatter, which every session
            writes its prompts and reads its turns with.
        steps: The step the run ends at.
        eval_tasks: Held-out tasks. The loop attempts each once on the
            run's sampler when an evaluation is due: before the first step,
            every ``config.eval_every`` steps and after the last. None of
            their episodes trains.
        on_step: Called with each report, in order.

    Returns:
        The reports, as ``train_rl.train`` returns them. A step's metrics
        also hold ``policy_lag`` and ``policy_lag_max``, of the groups it
        trained, and ``groups_stale``, the finished groups dropped since
        the step before. Its ``version`` is the oldest that sampled one of
        its turns: at most ``max_policy_lag`` steps before the step's own.
    """
    first = (await training_client.status()).step
    reports: list[StepReport] = []
    if first >= steps:
        return reports

    def report(made: StepReport) -> None:
        if on_step is not None:
            on_step(made)
        reports.append(made)

    def ahead(step: int) -> int:
        return room(dataset, step, steps=steps, max_policy_lag=config.max_policy_lag)

    runtime = EpisodeSession.Runtime(formatter=formatter, params=config.params)
    measuring = train_rl.eval_runtime(config, formatter)
    # The run's weights as they are, so the first groups sample them.
    await (await training_client.publish_weights()).result()
    queue = RolloutQueue(
        tasks_from(dataset, first), runtime, sampling_client, group_size=config.group_size
    )
    try:
        # Sampling starts before the baseline is measured, and runs beside it.
        queue.admit(ahead(first))
        if eval_tasks:
            measured = await evaluate(eval_tasks, measuring, sampling_client)
            report(StepReport(step=first, result=None, version=None, metrics=measured))
        for step in range(first, steps):
            count = len(_batch(dataset, step))
            groups, stale = await queue.take(count, step=step, max_policy_lag=config.max_policy_lag)
            result, checkpoint = await train_rl.train_step(
                training_client, groups, config, step=step
            )
            # The sampler serves the step: start the groups it makes room for, then evaluate.
            queue.admit(ahead(step + 1))
            metrics = {
                **train_rl.step_metrics(groups),
                **lag_metrics(groups, step),
                "groups_stale": float(stale),
            }
            due = config.eval_every is not None and (step + 1) % config.eval_every == 0
            if eval_tasks and (due or step + 1 == steps):
                metrics |= await evaluate(eval_tasks, measuring, sampling_client)
            report(
                StepReport(
                    step=step + 1,
                    result=result,
                    version=train_rl.oldest([episode for group in groups for episode in group]),
                    metrics=metrics,
                    checkpoint=checkpoint,
                )
            )
    finally:
        await queue.close()
    return reports
