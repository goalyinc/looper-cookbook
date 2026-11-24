"""The verifier contract.

A ``Verifier`` scores one finished episode. It is one call: ``verify`` reads
the episode's conversation, and may read its environment, and returns a
``Score``. An episode session calls it once, when the episode has ended.

**A verifier reads; it does not act for the model.** It is lent the
episode's environment, still open, and may run its own checks in
``environment.sandbox``, on what the model left. It neither closes the
environment nor keeps it.

**The reference stays with the verifier,** in this process: the right
answer, or the tests that decide a reward. No prompt, tool or observation
carries it.

**A failure is never a reward.** A verifier that cannot do its check, with
a sandbox that will not start or a judge that does not answer, raises. It
returns a low score only for an episode it did check.

**One verifier serves one episode.** ``Config.build()`` makes it. A config
is data, so a task carries its verifier's config beside its messages.
"""

from __future__ import annotations

import abc
import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from looper_cookbook.environments.base import Environment
from looper_cookbook.message_formatters.base import Message

__all__ = ["Score", "Verifier", "last_reply"]


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Score:
    """What a verifier made of one episode.

    Attributes:
        reward: The episode's score, higher for a better one. Only its
            differences within a group train.
        metrics: Counts of what the verifier saw, for a run's log, such as
            whether the reply held an answer at all.
    """

    reward: float
    metrics: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.reward):
            raise ValueError(
                f"a reward must be a finite number, not {self.reward}: a verifier that "
                "cannot score an episode raises instead"
            )


class Verifier(abc.ABC):
    """What scores a finished episode."""

    @runtime_checkable
    class Config(Protocol):
        async def build(self) -> Verifier:
            """A verifier for one episode."""
            ...

    @abc.abstractmethod
    async def verify(self, messages: Sequence[Message], environment: Environment | None) -> Score:
        """Scores the episode whose conversation is ``messages``.

        Args:
            messages: The task's messages, each assistant turn the model
                sampled, and each tool's reply, in order.
            environment: The episode's environment, still open; ``None``
                when its task has none. Lent: the verifier does not close it.
        """


def last_reply(messages: Sequence[Message]) -> str:
    """The content of the conversation's last assistant message: the model's
    answer. Empty when the conversation has none."""
    for message in reversed(messages):
        if message["role"] == "assistant":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""
