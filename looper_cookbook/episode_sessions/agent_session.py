"""The agent session: an episode session of one model that takes turns and
calls tools.

``AgentSession`` is the cookbook's ``EpisodeSession``. It owns the conversation
and its tokens, says what to sample next, hands each tool call of a sampled
turn to the task's environment, keeps the budgets, and asks the task's
verifier for the score.

**Its config is the task,** with the fields every session's task has
(``EpisodeSession.Config``): the messages the model is asked, its
verifier's config, its environment's config if it has one, and its limits.

**Turn after turn:**

1. the first prompt is the task's messages, with the environment's tools;
2. a turn is sampled after the prompt, with no more tokens than the episode
   has left;
3. a turn that calls no tool is the model's answer, and ends the episode.
   Otherwise each call runs in the environment, in order, and its
   observation is a tool message;
4. the next prompt is every token so far, then those messages. A sampled
   turn is never written again, so an episode is one sequence;
5. the verifier scores the conversation, with the environment still open.

**A single-turn problem is the same session with no environment.** Its
first turn calls no tool, so it is the answer.

**It samples nothing itself.** ``start`` and ``observe`` each return the
next ``SampleRequest``, and ``finish`` the ``Episode``. ``run`` is those
three around a sampling client. A unit test drives them with turns it
wrote, with no sampler.

**Two token budgets, kept apart.** A turn's is the runtime's
``params.max_tokens``: the most tokens the sampler returns for one turn. An
episode's is the task's ``max_episode_tokens``: its first prompt, every
sampled turn and every tool reply together. A turn's request carries the
smaller of the turn's budget and what the episode has left. The sampler's
``max_seq_len`` is neither: it is the service's limit, and must hold the
episode's budget.

**What ends an episode** is its ``StopReason``: ``ANSWER``, a turn that
called no tool; ``MAX_TURNS``, the last turn the task allows called tools,
which ran; ``MAX_TURN_TOKENS``, a turn cut off at the turn's budget; and
``MAX_EPISODE_TOKENS``, the episode's budget ran out: a turn was cut off at
what the episode had left, or no token was left for another turn after a
turn's tools replied. An episode whose last turn was cut off, by either
budget, is not verified: it scores 0. An answer cut off in its reasoning
may hold the right number by accident.

**Its metrics** are its observations' metrics summed, then its verifier's,
then its own ``turns``, ``sampled_tokens`` and ``truncated``, 1 when its
last turn was cut off by either budget. Of two with one name, the later in
that order is kept.

**Failures.** A call the model got wrong is an observation, and the episode
goes on. A sandbox's, a verifier's or the sampler's error raises from the
step it happened in; the caller closes the session.

**A harness of its own** is a subclass that overrides ``respond``, with a
``Config`` of its own that builds it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from looper_cookbook.environments.base import Action, Environment, Tool
from looper_cookbook.episode_sessions.base import Episode, EpisodeSession, StopReason, Turn
from looper_cookbook.message_formatters.base import Message, call_arguments
from looper_cookbook.verifiers.base import Score, Verifier
from looper_sdk import FinishReason, SampleRequest, SampleSequence, SamplingClient

__all__ = ["AgentSession"]


class AgentSession(EpisodeSession):
    """One model's attempt at a task: its turns, its actions and its score.

    The runtime's ``params.max_tokens`` is one turn's token budget, and the
    task's ``max_episode_tokens`` the whole episode's. Each request carries
    the smaller of the turn's budget and what the episode has left, with
    the formatter's stop tokens added.

    Attributes:
        environment: The episode's environment, from ``start`` on; ``None``
            for a task without one.
    """

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Config(EpisodeSession.Config):
        """An agent session's task: every session's fields
        (``EpisodeSession.Config``), and none of its own.

        The session reads them its way. ``environment`` is where the model's
        tool calls run, and ``max_episode_tokens`` counts the first prompt,
        every sampled turn and every tool reply together.
        """

        async def build(self, runtime: EpisodeSession.Runtime) -> AgentSession:
            """A fresh session for one attempt. Whoever builds it closes it."""
            return AgentSession(self, runtime, await self.verifier.build())

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # AgentSession.Config builds an AgentSession: inherited, it would build no subclass.
        if "Config" not in vars(cls):
            raise TypeError(
                f"{cls.__name__} needs a Config of its own, or its tasks build a plain "
                f"AgentSession: give it `class Config(AgentSession.Config)` whose build returns "
                f"{cls.__name__}(self, runtime, await self.verifier.build())"
            )

    def __init__(self, config: Config, runtime: EpisodeSession.Runtime, verifier: Verifier) -> None:
        self._config = config
        self._formatter = runtime.formatter
        self._params = runtime.params
        self._verifier = verifier
        self.environment: Environment | None = None
        self._messages: list[Message] = list(config.messages)
        self._turns: list[Turn] = []
        self._metrics: dict[str, float] = {}
        # Every token of the episode so far, and those the model has read since its last turn.
        self._transcript: list[int] = []
        self._context: list[int] = []
        # Whether the last request's limit was what the episode had left, not the turn's budget.
        self._episode_bound = False
        self._started = False
        self._stop_reason: StopReason | None = None
        self._episode: Episode | None = None
        self._closed = False

    async def run(self, sampling_client: SamplingClient) -> Episode:
        """Runs the episode on ``sampling_client``'s samples, and scores it.

        The session stays open, so its environment can still be read: close
        it when done, or build it in an ``async with``.

        Raises:
            SandboxError: The environment's or the verifier's sandbox failed.
            LooperError: A sample failed.
        """
        request = await self.start()
        while request is not None:
            operation = await sampling_client.sample(request.prompt, params=request.params)
            (sequence,) = (await operation.result()).sequences
            request = await self.observe(sequence)
        return await self.finish()

    async def start(self) -> SampleRequest:
        """Builds the task's environment and returns the first turn's request.

        Raises:
            ValueError: The first prompt leaves no token for a turn.
        """
        if self._started or self._closed:
            raise RuntimeError("a session runs one episode: build another from the task's config")
        self._started = True
        if self._config.environment is not None:
            self.environment = await self._config.environment.build()
        prompt = self._formatter.build_generation_prompt(self._messages, self._tools())
        if len(prompt) >= self._config.max_episode_tokens:
            raise ValueError(
                f"the task's first prompt has {len(prompt)} tokens, and max_episode_tokens is "
                f"{self._config.max_episode_tokens}: no token is left for a turn. Raise it, "
                "with the run's and the sampler's max_seq_len, or shorten the task"
            )
        self._transcript = list(prompt)
        self._context = list(prompt)
        return self._request()

    async def observe(self, sequence: SampleSequence) -> SampleRequest | None:
        """Takes the turn sampled for the last request. Returns the next
        turn's request, or ``None`` when the episode has ended.

        Raises:
            SandboxError: The environment could not run one of the turn's
                actions at all.
        """
        if not self._started or self._stop_reason is not None or self._closed:
            raise RuntimeError("observe takes a turn sampled for start's or observe's request")
        self._turns.append(Turn(context=tuple(self._context), sequence=sequence))
        self._transcript += sequence.token_ids
        reply, _ = self._formatter.parse_response(sequence.token_ids, self._tools())
        self._messages.append(reply)
        if sequence.finish_reason is FinishReason.LENGTH:
            # Cut off at the limit its request carried: name the budget that set it.
            self._stop_reason = (
                StopReason.MAX_EPISODE_TOKENS if self._episode_bound else StopReason.MAX_TURN_TOKENS
            )
            return None
        replies = list(await self.respond(reply))
        if not replies:
            self._stop_reason = StopReason.ANSWER
            return None
        self._messages += replies
        if len(self._turns) >= self._config.max_turns:
            self._stop_reason = StopReason.MAX_TURNS
            return None
        prompt = self._formatter.continue_prompt(self._transcript, replies)
        if len(prompt) >= self._config.max_episode_tokens:
            self._stop_reason = StopReason.MAX_EPISODE_TOKENS
            return None
        self._context = prompt[len(self._transcript) :]
        self._transcript = prompt
        return self._request()

    async def respond(self, reply: Message) -> Sequence[Message]:
        """The messages that answer a sampled turn; none when the turn is
        the model's answer, which ends the episode.

        Each of the turn's tool calls runs in the environment, in order, and
        its observation is a tool message. A harness of its own overrides
        this to word a reply its own way.
        """
        if self.environment is None:
            return []
        replies = []
        for call in reply.get("tool_calls") or ():
            name, arguments = call_arguments(call)
            observation = await self.environment.step(Action(name=name, arguments=arguments))
            for metric, value in observation.metrics.items():
                self._metrics[metric] = self._metrics.get(metric, 0.0) + value
            replies.append({"role": "tool", "name": name, "content": observation.content})
        return replies

    async def finish(self) -> Episode:
        """Scores the ended episode and returns it. The environment is still
        open for the verifier, and stays open until ``close``. A second call
        returns the same episode, and verifies nothing again.

        Raises:
            SandboxError: The verifier's sandbox failed.
        """
        if self._episode is not None:
            return self._episode
        if self._stop_reason is None or self._closed:
            raise RuntimeError("finish scores an episode that observe has ended")
        truncated = self._turns[-1].sequence.finish_reason is FinishReason.LENGTH
        # A turn cut off, by either budget, is not an answer: it scores 0, and no verifier runs.
        score = Score(reward=0.0)
        if not truncated:
            score = await self._verifier.verify(tuple(self._messages), self.environment)
        sampled = sum(len(turn.sequence.token_ids) for turn in self._turns)
        self._episode = Episode(
            turns=tuple(self._turns),
            reward=score.reward,
            stop_reason=self._stop_reason,
            messages=tuple(self._messages),
            metrics={
                **self._metrics,
                **score.metrics,
                "turns": float(len(self._turns)),
                "sampled_tokens": float(sampled),
                "truncated": float(truncated),
            },
        )
        return self._episode

    async def close(self) -> None:
        """Closes the environment the session built. A second call does nothing."""
        if self._closed:
            return
        self._closed = True
        if self.environment is not None:
            await self.environment.close()

    def _tools(self) -> Sequence[Tool] | None:
        """The environment's tools, as a formatter takes them; none without one."""
        return self.environment.tools if self.environment is not None else None

    def _request(self) -> SampleRequest:
        """One turn after the transcript: at most the turn's budget, and no
        more than the episode has left."""
        left = self._config.max_episode_tokens - len(self._transcript)
        self._episode_bound = left < self._params.max_tokens
        stops = dict.fromkeys((*self._params.stop_token_ids, *self._formatter.stop_token_ids))
        params = dataclasses.replace(
            self._params,
            max_tokens=min(self._params.max_tokens, left),
            stop_token_ids=tuple(stops),
        )
        return SampleRequest(prompt=tuple(self._transcript), params=params)
