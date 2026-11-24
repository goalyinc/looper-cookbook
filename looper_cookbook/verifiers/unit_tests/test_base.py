"""The verifier contract's values."""

from __future__ import annotations

import pytest

from looper_cookbook.verifiers.base import Score, Verifier, last_reply
from looper_cookbook.verifiers.final_number import FinalNumber


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), float("-inf")])
def test_a_reward_is_a_finite_number(reward: float) -> None:
    with pytest.raises(ValueError, match="a reward must be a finite number"):
        Score(reward=reward)


def test_a_score_has_no_metrics_unless_given() -> None:
    assert Score(reward=0.5) == Score(reward=0.5, metrics={})


def test_the_last_reply_is_the_last_assistant_messages_content() -> None:
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "first", "reasoning_content": "#### 9"},
        {"role": "tool", "content": "t"},
        {"role": "assistant", "content": "second"},
        {"role": "tool", "content": "after"},
    ]
    assert last_reply(messages) == "second"
    assert last_reply(messages[:1]) == ""
    # A turn that only called a tool has no text.
    assert last_reply([{"role": "assistant", "content": None, "tool_calls": []}]) == ""


def test_a_config_is_anything_that_builds_a_verifier() -> None:
    assert isinstance(FinalNumber.Config(answer="1"), Verifier.Config)
    assert not isinstance(object(), Verifier.Config)
