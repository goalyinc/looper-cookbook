"""The episode session contract.

An ``EpisodeSession`` is one attempt at one task. It is built for the
attempt, runs once, and returns an ``Episode``: what the model read and
wrote, and how it scored.

**Three levels, each with its own word.** A *training loop* (``train_rl``)
takes one optimizer step per batch of tasks. A *rollout* runs sessions on a
sampler: a group of them for each task of a step, or one for each held-out
task of an evaluation. A *session* is one attempt, and what happens inside
it is its own. ``AgentSession`` (``agent_session.py``) is the cookbook's:
one model that takes turns and calls tools until it answers.

**A task is a session's config.** A config is data, and ``build(runtime)``
makes a fresh session for one attempt. So a dataset holds configs, a group
is several sessions of one config, and an evaluation runs the same configs
again. ``Runtime`` is what every task of a run shares: the model's message
formatter, and how a turn is sampled.

**Every session's task has the same fields.** ``EpisodeSession.Config``
defines them: the messages the model is asked, the verifier that scores an
attempt, the environment its actions run in, and its two limits,
``max_turns`` and ``max_episode_tokens``. A session's own config is a
subclass that writes ``build``, so a task reads the same whichever session
attempts it.

**A session samples from the client it is handed.** ``run`` takes the
sampling client, so whoever runs the sessions decides which weights they
sample, and a session's own steps can be tested with no client.

**Every session returns an ``Episode``,** whatever happened inside it: its
sampled turns, each with the tokens the model read before it, and its
reward. ``episode.datum(...)`` is the episode as training data, so a
session of your own trains as the cookbook's does.

**A session of your own** is a subclass with ``run`` and ``close``. Its
config subclasses ``EpisodeSession.Config``, builds it, and adds the fields
only that session needs.

``rollout`` runs sessions on a sampling client: many at once, each built
from its task's config and closed when it ends. The RL loops roll out a
step's groups with it.

``episode_statistics`` is what a list of episodes comes to in numbers: the
mean and its standard error of their rewards and metrics, and the share
each stop reason ended. A step's report and an evaluation's metrics are
both made from it.

``evaluate`` measures a model: it rolls out one attempt at each of a list
of tasks and returns their statistics as ``eval/`` metrics. It trains
nothing, so it runs on any sampling client: a run's, between its steps, or
a saved checkpoint's.
"""

from __future__ import annotations

import abc
import asyncio
import dataclasses
import enum
import math
import statistics
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Self

from looper_cookbook.environments.base import Environment
from looper_cookbook.message_formatters.base import Message, MessageFormatter
from looper_cookbook.verifiers.base import Verifier
from looper_sdk import Datum, SampleSequence, SamplingClient, SamplingParams

__all__ = [
    "Episode",
    "EpisodeSession",
    "StopReason",
    "Turn",
    "episode_statistics",
    "evaluate",
    "rollout",
]


class StopReason(enum.StrEnum):
    """Why an episode ended.

    An episode has two token budgets, and each has its own reason. A turn's
    is the sampler's ``max_tokens``. An episode's covers all of its turns
    and what the model read before and between them.

    ``ANSWER``: the model gave its answer. ``MAX_TURNS``: the task's turn
    limit ended it. ``MAX_TURN_TOKENS``: a turn was cut off at the turn's
    token budget. ``MAX_EPISODE_TOKENS``: the episode's token budget ran
    out, within a turn or between two.
    """

    ANSWER = "answer"
    MAX_TURNS = "max_turns"
    MAX_TURN_TOKENS = "max_turn_tokens"
    MAX_EPISODE_TOKENS = "max_episode_tokens"


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Turn:
    """One sampled turn of an episode, with what the model read before it.

    Attributes:
        context: The tokens the model read since its last turn: the first
            prompt, or what was added after the turn before.
        sequence: The turn the sampler returned after them.
    """

    context: Sequence[int]
    sequence: SampleSequence


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Episode:
    """One finished attempt at a task.

    Attributes:
        turns: Its sampled turns, in order. Their contexts and sequences,
            end to end, are every token of the episode.
        reward: Its score, higher for a better attempt.
        stop_reason: Why it ended.
        messages: Its conversation: the task's messages, each sampled turn
            as a message, and whatever answered it.
        metrics: Counts for a run's log, such as its turns and its sampled
            tokens.
    """

    turns: Sequence[Turn]
    reward: float
    stop_reason: StopReason
    messages: Sequence[Message]
    metrics: Mapping[str, float]

    def datum(self, group_id: str, group_size: int) -> Datum:
        """The episode as its one datum: every token in order, weight
        1 on each sampled token and 0 on the rest, the sampler's log
        probabilities, and the episode's reward.

        Args:
            group_id: The group the episode is compared within.
            group_size: How many episodes the group has.
        """
        tokens: list[int] = []
        weights: list[int] = []
        logprobs: list[float] = []
        for turn in self.turns:
            sampled = turn.sequence.token_ids
            tokens += [*turn.context, *sampled]
            weights += [0] * len(turn.context) + [1] * len(sampled)
            logprobs += [0.0] * len(turn.context) + list(turn.sequence.logprobs)
        return Datum(
            tokens=tokens,
            weights=weights,
            columns={"logprobs": logprobs},
            reward=self.reward,
            group_id=group_id,
            group_size=group_size,
        )


class EpisodeSession(abc.ABC):
    """One attempt at one task, run once."""

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Runtime:
        """What whoever runs the sessions lends each one.

        Attributes:
            formatter: The model's message formatter. A session writes its
                prompts and reads its sampled turns with it.
            params: How each turn is drawn. ``max_tokens`` is one turn's
                token budget. An episode's is its task's
                ``max_episode_tokens``.
        """

        formatter: MessageFormatter
        params: SamplingParams

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Config(abc.ABC):
        """One task, and how it is attempted: the fields every session's
        task has.

        A session's own config subclasses this, writes ``build``, and adds
        the fields only that session needs. The session keeps the task's
        limits: it samples at most ``max_turns`` turns, and its episode
        holds at most ``max_episode_tokens`` tokens.

        Attributes:
            messages: What the model is asked: the conversation its first
                turn follows.
            verifier: Builds the verifier that scores an attempt.
            environment: Builds the environment an attempt's actions run
                in; ``None`` for a task with none, such as one the model
                answers in one turn.
            max_turns: The most turns an attempt samples.
            max_episode_tokens: The episode's token budget: every token the
                model reads and writes in one attempt, together. One turn's
                budget is the runtime's ``params.max_tokens``. The run's
                and the sampler's ``max_seq_len`` must hold the episode's.
        """

        messages: Sequence[Message]
        verifier: Verifier.Config
        environment: Environment.Config | None = None
        max_turns: int = 1
        max_episode_tokens: int

        def __post_init__(self) -> None:
            if not self.messages:
                raise ValueError("messages is empty: a task asks the model something")
            if not isinstance(self.verifier, Verifier.Config):
                raise ValueError(
                    "verifier must be a verifier's Config, such as FinalNumber.Config(...): "
                    "each attempt builds its own verifier from it"
                )
            if self.environment is not None and not isinstance(
                self.environment, Environment.Config
            ):
                raise ValueError(
                    "environment must be an environment's Config, such as "
                    "BashEnvironment.Config(...): each attempt builds its own from it"
                )
            if self.max_turns < 1:
                raise ValueError(f"max_turns must be at least 1, not {self.max_turns}")
            if self.max_episode_tokens < 2:
                raise ValueError(
                    f"max_episode_tokens must hold a prompt and a turn, not "
                    f"{self.max_episode_tokens}"
                )

        @abc.abstractmethod
        async def build(self, runtime: EpisodeSession.Runtime) -> EpisodeSession:
            """A fresh session for one attempt. Whoever builds it closes it."""

    @abc.abstractmethod
    async def run(self, sampling_client: SamplingClient) -> Episode:
        """Runs the attempt on ``sampling_client``'s samples, and returns it
        scored. A session runs once."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Releases what the session built. A second call does nothing."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()


async def rollout(
    configs: Sequence[EpisodeSession.Config],
    runtime: EpisodeSession.Runtime,
    sampling_client: SamplingClient,
) -> list[Episode]:
    """One attempt at each of ``configs``, all at once: each session is built,
    run on ``sampling_client`` and closed. The episodes are in the configs'
    order; a config given twice is attempted twice.

    When one attempt fails, the rest are cancelled and closed, and its error
    raises: no session is left sampling, and no sandbox is left running.
    """

    async def attempt(config: EpisodeSession.Config) -> Episode:
        async with await config.build(runtime) as session:
            return await session.run(sampling_client)

    attempts = [asyncio.ensure_future(attempt(config)) for config in configs]
    try:
        return list(await asyncio.gather(*attempts))
    except BaseException:
        for task in attempts:
            task.cancel()
        # Each cancelled session closes what it built before this returns.
        await asyncio.gather(*attempts, return_exceptions=True)
        raise


def episode_statistics(episodes: Sequence[Episode]) -> dict[str, tuple[float, float]]:
    """The mean and its standard error of the episodes' ``reward``, of each
    of their metrics over the episodes that report it, and of
    ``stop/<reason>`` for every ``StopReason``: the share of the episodes
    that ended for it.

    The shares tell an episode's two token budgets apart, where a session's
    own ``truncated`` counts a turn cut off by either:
    ``stop/max_turn_tokens`` is the share a turn's budget cut off, and
    ``stop/max_episode_tokens`` the share whose episode's budget ran out.

    A standard error is 0 for a single value, where nothing estimates it.
    """
    values: dict[str, list[float]] = {"reward": [episode.reward for episode in episodes]}
    for episode in episodes:
        for name, value in episode.metrics.items():
            values.setdefault(name, []).append(value)
    for reason in StopReason:
        values[f"stop/{reason.value}"] = [
            float(episode.stop_reason is reason) for episode in episodes
        ]
    return {
        name: (
            statistics.fmean(seen),
            statistics.stdev(seen) / math.sqrt(len(seen)) if len(seen) > 1 else 0.0,
        )
        for name, seen in values.items()
    }


async def evaluate(
    tasks: Sequence[EpisodeSession.Config],
    runtime: EpisodeSession.Runtime,
    sampling_client: SamplingClient,
) -> dict[str, float]:
    """One attempt at each of ``tasks`` on ``sampling_client``, as metrics.

    Nothing it samples trains. ``runtime.params`` says how its turns are
    sampled: at temperature 0 it measures the model's best answer.

    Returns:
        ``eval/episodes``, how many were attempted; and for the reward,
        each of the episodes' metrics and each ``stop/<reason>``, its mean
        as ``eval/<name>`` and that mean's standard error as
        ``eval/<name>_stderr``.
    """
    if not tasks:
        raise ValueError("tasks is empty: an evaluation needs a task to attempt")
    episodes = await rollout(tasks, runtime, sampling_client)
    metrics = {"eval/episodes": float(len(episodes))}
    for name, (mean, error) in episode_statistics(episodes).items():
        metrics[f"eval/{name}"] = mean
        metrics[f"eval/{name}_stderr"] = error
    return metrics
