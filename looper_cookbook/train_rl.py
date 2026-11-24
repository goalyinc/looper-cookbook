"""The explicit RL loop, written directly over ``TrainingClient`` and the run's
``SamplingClient``.

One optimizer step per batch of tasks. The run's step ``k``, counted from 0,
trains ``dataset.batch(k)``:

1. **Attempt.** ``group_size`` episode sessions of each task, all at once,
   sampled from the run's sampler. Each session's verifier scores it.
2. **Train.** Each task's episodes go as one group to ``forward_backward``
   under the config's loss, ``Grpo`` by default: the trainer forms each
   episode's advantage from its group's rewards. Then ``optim_step``,
   ``publish_weights`` and, when one is due, ``save_state``, submitted back
   to back.
3. **Evaluate,** when due: one attempt at each held-out task, on the weights
   just published.

The loop publishes once before its first attempt, so the sampler serves the
run's weights as they are, and awaits each publish before it samples again.
So every turn is sampled from the weights of the step before the one it
trains: on-policy, which each ``StepReport.version`` records.

**The run's step decides the data.** Called again on the same run, after an
error or a ``RunRolledBack``, the loop trains the batch of the step the run
stands at, with no bookkeeping by its caller.

What it leaves to its caller: the run, created with its ``TrainingConfig``;
the run's sampler, from ``training_client.create_sampling_client(...)``; the
tasks, held-out ones too; the last save; and whatever the run writes. Each
step trains at the run's own rate, set in its ``TrainingConfig.optimizer``.
An error raises as its operation's result is awaited.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Mapping, Sequence

from looper_cookbook.datasets.base import RLDataset
from looper_cookbook.episode_sessions.base import (
    Episode,
    EpisodeSession,
    episode_statistics,
    evaluate,
    rollout,
)
from looper_cookbook.message_formatters.base import MessageFormatter
from looper_sdk import (
    Datum,
    Grpo,
    SamplingClient,
    SamplingParams,
    StateCheckpoint,
    StepResult,
    TrainingClient,
    Version,
)

__all__ = [
    "Config",
    "StepReport",
    "calls",
    "eval_runtime",
    "oldest",
    "step_metrics",
    "train",
    "train_step",
]


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
    """How the loop samples, trains, evaluates and saves.

    Attributes:
        params: How each turn is sampled. ``max_tokens`` is one turn's
            token budget. An episode's is its task's,
            ``EpisodeSession.Config.max_episode_tokens``.
            The loss reads the sampler's log probabilities as the policy's
            own, which they are only at temperature 1 with no truncation,
            so any other setting is refused.
        group_size: The attempts at each task, compared within their group:
            an episode's advantage is its reward less its group's mean. A
            group whose rewards are all equal trains nothing.
        loss: What the trainer trains the groups with, from the episodes'
            rewards and the log probabilities they were sampled at:
            ``Grpo()`` or ``Dapo()``.
        max_call_tokens: The most tokens one ``forward_backward`` carries. A
            larger batch goes over several calls, whose gradients accumulate
            into the one step; a group is never split between calls. A
            call's payload holds 16 MiB, about 11 bytes a
            token with its log probability.
        eval_params: How an evaluation's turns are sampled. ``None`` samples
            them greedily: ``params`` at temperature 0, which measures the
            model's best answer. Nothing an evaluation samples trains, so
            any setting is allowed here.
        eval_every: Steps between evaluations of the loop's ``eval_tasks``.
            With such tasks the loop also evaluates before its first step
            and after its last; ``None`` evaluates only then.
        save_every: Steps between saves of the run's state, each named for
            the run and its step. ``None`` saves nothing: the last save is
            the caller's.
    """

    params: SamplingParams
    group_size: int = 8
    loss: Grpo = Grpo()
    max_call_tokens: int = 1_000_000
    eval_params: SamplingParams | None = None
    eval_every: int | None = None
    save_every: int | None = None

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError(
                f"group_size must be at least 2, not {self.group_size}: an episode's "
                "advantage is its reward against the rest of its group"
            )
        if self.params.temperature != 1.0 or self.params.top_p != 1.0:
            raise ValueError(
                f"params samples at temperature {self.params.temperature} and top_p "
                f"{self.params.top_p}: the loss reads the sampler's log probabilities as "
                "the policy's own, which they are only at temperature 1 and top_p 1"
            )
        if self.max_call_tokens < 1:
            raise ValueError(f"max_call_tokens must be at least 1, not {self.max_call_tokens}")
        for name in ("eval_every", "save_every"):
            every = getattr(self, name)
            if every is not None and every < 1:
                raise ValueError(f"{name} must be at least 1 step, or None, not {every}")


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class StepReport:
    """What the loop reports of one step of the run.

    Attributes:
        step: The run's step the report is of: the steps it had taken.
        result: What the run reported of the step. ``None`` in the loop's
            first report when it has ``eval_tasks``: that one is of the step
            the loop started at, evaluated before it trained anything.
        version: The oldest version that sampled one of the step's turns:
            the step before it, since the loop awaits each publish before
            it samples. ``None`` beside a ``result`` of ``None``.
        metrics: The step's numbers, by name. Of its episodes: ``reward``
            and each episode metric, as means, with ``truncated`` the share
            cut off; ``stop/<reason>``, the share each ``StopReason`` ended,
            where ``stop/max_turn_tokens`` and ``stop/max_episode_tokens``
            say which token budget cut them off; ``episodes`` and
            ``groups``, their counts; and
            ``groups_trained``, the groups whose rewards differ. Then the
            evaluation's, named ``eval/...``, at the steps that ran one.
        checkpoint: The state saved at the step; ``None`` when none was.
    """

    step: int
    result: StepResult | None
    version: Version | None
    metrics: Mapping[str, float]
    checkpoint: StateCheckpoint | None = None


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
    """Trains ``training_client``'s run up to its step ``steps``, one step
    per batch of tasks, on episodes ``sampling_client`` samples.

    Args:
        training_client: The run's client, with no step of its own in
            flight. The loop starts at the run's current step.
        sampling_client: The run's sampler, created by its training client.
            The loop publishes to it before its first sample and after each
            step.
        dataset: The tasks, a batch per step.
        config: How episodes are sampled and trained, and when the loop
            evaluates and saves.
        formatter: The model's message formatter, which every session
            writes its prompts and reads its turns with.
        steps: The step the run ends at.
        eval_tasks: Held-out tasks. The loop attempts each once on the
            run's sampler when an evaluation is due: before the first step,
            every ``config.eval_every`` steps and after the last. None of
            their episodes trains.
        on_step: Called with each report, in order.

    Returns:
        The reports, in order: with ``eval_tasks``, one of the step the loop
        started at, then one per step it trained.
    """
    first = (await training_client.status()).step
    reports: list[StepReport] = []
    if first >= steps:
        return reports

    def report(made: StepReport) -> None:
        if on_step is not None:
            on_step(made)
        reports.append(made)

    runtime = EpisodeSession.Runtime(formatter=formatter, params=config.params)
    # The run's weights as they are, so the first batch samples them.
    await (await training_client.publish_weights()).result()
    measuring = eval_runtime(config, formatter)
    if eval_tasks:
        measured = await evaluate(eval_tasks, measuring, sampling_client)
        report(StepReport(step=first, result=None, version=None, metrics=measured))
    for step in range(first, steps):
        tasks = dataset.batch(step)
        if not tasks:
            raise ValueError(f"the batch of step {step + 1} is empty: a step needs a task")
        # Every session in flight at once: the sampler batches their turns on its GPUs.
        attempts = [task for task in tasks for _ in range(config.group_size)]
        episodes = await rollout(attempts, runtime, sampling_client)
        groups = [
            episodes[start : start + config.group_size]
            for start in range(0, len(episodes), config.group_size)
        ]
        result, checkpoint = await train_step(training_client, groups, config, step=step)
        metrics = step_metrics(groups)
        due = config.eval_every is not None and (step + 1) % config.eval_every == 0
        if eval_tasks and (due or step + 1 == steps):
            metrics |= await evaluate(eval_tasks, measuring, sampling_client)
        report(
            StepReport(
                step=step + 1,
                result=result,
                version=oldest(episodes),
                metrics=metrics,
                checkpoint=checkpoint,
            )
        )
    return reports


async def train_step(
    training_client: TrainingClient,
    groups: Sequence[Sequence[Episode]],
    config: Config,
    *,
    step: int,
) -> tuple[StepResult, StateCheckpoint | None]:
    """One optimizer step of the run on ``groups``, published to its sampler.

    Each group goes whole to ``forward_backward`` under the config's loss.
    Then ``optim_step``, ``publish_weights`` and, when one is due,
    ``save_state``, submitted back to back. It returns once the sampler
    serves the step, so whatever samples next samples the weights it
    trained.

    Args:
        training_client: The run's client.
        groups: The episodes to train, a group per task.
        config: The loss, the size of a call, and when a save is due.
        step: The run's step before this one, counted from 0.

    Returns:
        What the run reported of the step, and the state saved at it or
        ``None``.

    Raises:
        RuntimeError: The run took another step than ``step + 1``.
    """
    data = [
        [episode.datum(f"step-{step + 1}-task-{index}", len(group)) for episode in group]
        for index, group in enumerate(groups)
    ]
    backward = [
        await training_client.forward_backward(call, config.loss)
        for call in calls(data, config.max_call_tokens)
    ]
    applied = await training_client.optim_step()
    published = await training_client.publish_weights()
    saving = None
    if config.save_every is not None and (step + 1) % config.save_every == 0:
        # After the step: a save is refused while gradients are pending.
        saving = await training_client.save_state(f"{training_client.run_id}-step-{step + 1}")
    # The calls first: a failed call's error says why, where its step says only that one failed.
    for operation in backward:
        await operation.result()
    result = await applied.result()
    if result.step != step + 1:
        raise RuntimeError(
            f"the run took its step {result.step} where the loop expected step {step + 1}: "
            "another writer stepped it, or it rolled back"
        )
    await published.result()
    return result, await saving.result() if saving is not None else None


def calls(groups: Sequence[Sequence[Datum]], max_tokens: int) -> Iterator[list[Datum]]:
    """The groups' datums as ``forward_backward`` calls of at most
    ``max_tokens`` tokens each, in order, every group whole within one call:
    the trainer forms a group's advantages from the call that carries it."""
    call: list[Datum] = []
    size = 0
    for index, group in enumerate(groups):
        tokens = sum(len(datum.tokens) for datum in group)
        if tokens > max_tokens:
            raise ValueError(
                f"the group of task {index} holds {tokens} tokens, over max_call_tokens "
                f"({max_tokens}), and a group goes whole in one call: raise max_call_tokens, "
                "or lower group_size or the tasks' max_episode_tokens"
            )
        if call and size + tokens > max_tokens:
            yield call
            call, size = [], 0
        call += group
        size += tokens
    if call:
        yield call


def step_metrics(groups: Sequence[Sequence[Episode]]) -> dict[str, float]:
    """A step's numbers from its episodes: the mean reward, the mean of
    each episode metric and the share each stop reason ended, how many
    episodes and groups it had, and how many groups' rewards differ, which
    are the ones that train."""
    episodes = [episode for group in groups for episode in group]
    metrics = {name: mean for name, (mean, _) in episode_statistics(episodes).items()}
    metrics["episodes"] = float(len(episodes))
    metrics["groups"] = float(len(groups))
    metrics["groups_trained"] = float(
        sum(len({episode.reward for episode in group}) > 1 for group in groups)
    )
    return metrics


def oldest(episodes: Sequence[Episode]) -> Version:
    """The oldest version that sampled one of the episodes' turns."""
    versions = [turn.sequence.version for episode in episodes for turn in episode.turns]
    if any(version is None for version in versions):
        raise ValueError("a turn names no version: it came from a sampler of no run")
    return min(
        (version for version in versions if version is not None),
        key=lambda version: (version.epoch, version.step),
    )


def eval_runtime(config: Config, formatter: MessageFormatter) -> EpisodeSession.Runtime:
    """What an evaluation's sessions are lent: the formatter, and the
    config's ``eval_params`` or, without them, its ``params`` at temperature
    0."""
    params = config.eval_params or dataclasses.replace(config.params, temperature=0.0)
    return EpisodeSession.Runtime(formatter=formatter, params=params)
