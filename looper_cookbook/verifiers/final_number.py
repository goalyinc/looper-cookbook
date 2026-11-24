"""A final number matched exactly.

``FinalNumber`` scores a problem whose answer is one number, as GSM8K's are.
The model is asked to end on ``#### <number>``, the way GSM8K writes its
answers, and the verifier reads the number after the reply's last ``####``.
Only the reply counts: a thinking model may try answers while it reasons.

Numbers compare as numbers, so ``1,200``, ``1200`` and ``1200.0`` are one
answer, and ``$`` before a number is skipped.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from decimal import Decimal

from looper_cookbook.environments.base import Environment
from looper_cookbook.message_formatters.base import Message
from looper_cookbook.verifiers.base import Score, Verifier, last_reply

__all__ = ["FinalNumber", "final_number"]

# The number after "####", with its sign, thousands commas and decimals.
_FINAL = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")


def final_number(text: str) -> Decimal | None:
    """The number after the text's last ``####``; ``None`` when it has none."""
    found = _FINAL.findall(text)
    return Decimal(found[-1].replace(",", "")) if found else None


class FinalNumber(Verifier):
    """Reward 1 when the reply's final number is the answer, else 0."""

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Config:
        """A problem's right answer.

        Attributes:
            answer: The number the reply must end on, as text: ``"72"``,
                ``"1,200"`` or ``"-0.5"``.
        """

        answer: str

        def __post_init__(self) -> None:
            if final_number(f"#### {self.answer}") is None:
                raise ValueError(
                    f"answer is {self.answer!r}, which is not a number: FinalNumber scores "
                    "problems whose answer is one number"
                )

        async def build(self) -> FinalNumber:
            answer = final_number(f"#### {self.answer}")
            assert answer is not None
            return FinalNumber(answer)

    def __init__(self, answer: Decimal) -> None:
        self._answer = answer

    async def verify(self, messages: Sequence[Message], environment: Environment | None) -> Score:
        given = final_number(last_reply(messages))
        return Score(
            reward=float(given == self._answer), metrics={"answered": float(given is not None)}
        )
