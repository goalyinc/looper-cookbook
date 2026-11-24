"""The SFT loop's pure parts (looper_cookbook/train_sft.py): its schedules
and how a batch splits into calls. The loop itself runs against a real
server."""

from __future__ import annotations

import math

import pytest

from looper_cookbook import train_sft
from looper_sdk import Datum


def datum(length: int) -> Datum:
    return Datum(tokens=range(length), weights=[0] + [1] * (length - 1))


def rates(schedule: str, steps: int = 4) -> list[float]:
    config = train_sft.Config(learning_rate=1.0, schedule=schedule)
    return [train_sft.learning_rate(config, step, steps) for step in range(steps)]


def test_each_schedule_starts_at_the_rate_and_falls_as_it_says() -> None:
    assert rates("constant") == [1.0, 1.0, 1.0, 1.0]
    assert rates("linear") == [1.0, 0.75, 0.5, 0.25]
    assert rates("cosine") == pytest.approx(
        [1.0, 0.5 * (1 + math.sqrt(0.5)), 0.5, 0.5 * (1 - math.sqrt(0.5))]
    )


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"learning_rate": -1e-4}, "learning_rate must be finite and at least 0"),
        ({"learning_rate": math.nan}, "learning_rate must be finite and at least 0"),
        ({"learning_rate": 1e-4, "schedule": "warmup"}, "schedule must be one of"),
        ({"learning_rate": 1e-4, "steps_ahead": -1}, "steps_ahead must be at least 0"),
        ({"learning_rate": 1e-4, "max_call_tokens": 0}, "max_call_tokens must be at least 1"),
    ],
)
def test_an_invalid_config_is_refused(fields: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        train_sft.Config(**fields)


def test_a_batch_splits_in_order_into_calls_within_the_limit() -> None:
    batch = [datum(4), datum(4), datum(9), datum(2), datum(2), datum(2)]
    calls = list(train_sft._calls(batch, 8))
    assert [[len(d.tokens) for d in call] for call in calls] == [[4, 4], [9], [2, 2, 2]]
    assert [d for call in calls for d in call] == batch
    assert list(train_sft._calls(batch, 100)) == [batch]
