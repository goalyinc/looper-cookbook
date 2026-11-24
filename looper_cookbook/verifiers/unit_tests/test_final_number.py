"""``FinalNumber``: which reply holds the answer."""

from __future__ import annotations

from decimal import Decimal

import pytest

from looper_cookbook.verifiers.base import Score
from looper_cookbook.verifiers.final_number import FinalNumber, final_number


@pytest.mark.parametrize(
    "text,number",
    [
        ("#### 72", "72"),
        ("So 6 * 12 = 72.\n####72", "72"),
        ("#### $1,200.50 in all", "1200.50"),
        ("#### -0.5", "-0.5"),
        ("#### .5", "0.5"),
        # The last one counts: a reply may correct itself.
        ("#### 3\nNo.\n#### 4", "4"),
        ("The answer is 72.", None),
        ("#### seventy-two", None),
        ("", None),
    ],
)
def test_the_final_number_follows_the_last_marker(text: str, number: str | None) -> None:
    assert final_number(text) == (None if number is None else Decimal(number))


@pytest.mark.parametrize(
    "answer,reply,score",
    [
        ("72", "#### 72", Score(reward=1.0, metrics={"answered": 1.0})),
        # Numbers compare as numbers.
        ("1,200", "#### 1200.0", Score(reward=1.0, metrics={"answered": 1.0})),
        ("72", "#### 73", Score(reward=0.0, metrics={"answered": 1.0})),
        ("72", "It is 72.", Score(reward=0.0, metrics={"answered": 0.0})),
    ],
)
async def test_the_right_number_scores_one(answer: str, reply: str, score: Score) -> None:
    verifier = await FinalNumber.Config(answer=answer).build()
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": reply}]
    assert await verifier.verify(messages, None) == score


async def test_a_number_tried_in_the_reasoning_does_not_count() -> None:
    verifier = await FinalNumber.Config(answer="72").build()
    reply = {"role": "assistant", "reasoning_content": "#### 72", "content": "I am not sure."}
    assert (await verifier.verify([reply], None)).reward == 0.0


def test_an_answer_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(ValueError, match="answer is 'Paris', which is not a number"):
        FinalNumber.Config(answer="Paris")
