"""The RL loop's pure parts (looper_cookbook/train_rl.py): its config, how an
evaluation samples, how a step's groups become calls, and what a step reports. The loop itself runs
against a real server (``tests/e2e/test_train_rl.py``)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from looper_cookbook import train_rl
from looper_cookbook.episode_sessions.base import Episode, StopReason, Turn
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter
from looper_cookbook.message_formatters.unit_tests import tokenizers
from looper_sdk import Dapo, Datum, FinishReason, SampleSequence, SamplingParams, Version

RUN = "run-1"
SAMPLED = SamplingParams(max_tokens=4)


def episode(reward: float, step: int | None = 0, epoch: int = 0, **metrics: float) -> Episode:
    """An episode of one turn, sampled at the run's ``step``."""
    version = None if step is None else Version(run_id=RUN, epoch=epoch, step=step)
    sequence = SampleSequence(
        token_ids=(7, 8), logprobs=(-0.5, -1.0), finish_reason=FinishReason.STOP, version=version
    )
    return Episode(
        turns=(Turn(context=(1, 2, 3), sequence=sequence),),
        reward=reward,
        stop_reason=StopReason.ANSWER,
        messages=(),
        metrics=metrics,
    )


def group(tokens: int, name: str) -> list[Datum]:
    """Two datums of ``tokens`` tokens between them."""
    half = tokens // 2
    return [
        Datum(tokens=range(half), weights=[0] + [1] * (half - 1), datum_id=f"{name}-{index}")
        for index in range(2)
    ]


@pytest.mark.parametrize(
    "settings,why",
    [
        ({"group_size": 1}, "group_size must be at least 2"),
        ({"params": SamplingParams(max_tokens=4, temperature=0.7)}, "only at temperature 1"),
        ({"params": SamplingParams(max_tokens=4, top_p=0.9)}, "only at temperature 1"),
        ({"max_call_tokens": 0}, "max_call_tokens must be at least 1"),
        ({"eval_every": 0}, "eval_every must be at least 1 step, or None, not 0"),
        ({"save_every": -2}, "save_every must be at least 1 step, or None, not -2"),
    ],
)
def test_a_config_refuses_what_the_loop_cannot_train(settings: dict[str, Any], why: str) -> None:
    with pytest.raises(ValueError, match=why):
        train_rl.Config(**{"params": SAMPLED, **settings})


def test_a_config_takes_either_loss() -> None:
    assert train_rl.Config(params=SAMPLED, loss=Dapo()).loss == Dapo()


def test_an_evaluation_samples_greedily_unless_the_config_says_how() -> None:
    formatter = Qwen3_5MessageFormatter(tokenizers.qwen())
    greedy = train_rl.eval_runtime(train_rl.Config(params=SAMPLED), formatter)
    assert greedy.formatter is formatter
    assert greedy.params == dataclasses.replace(SAMPLED, temperature=0.0)
    own = SamplingParams(max_tokens=16, temperature=0.3)
    config = train_rl.Config(params=SAMPLED, eval_params=own)
    assert train_rl.eval_runtime(config, formatter).params == own


def test_groups_go_whole_into_calls_of_at_most_the_limit() -> None:
    groups = [group(40, "a"), group(40, "b"), group(60, "c"), group(20, "d"), group(20, "e")]
    sent = list(train_rl.calls(groups, 100))
    assert [[datum.datum_id for datum in call] for call in sent] == [
        ["a-0", "a-1", "b-0", "b-1"],
        ["c-0", "c-1", "d-0", "d-1", "e-0", "e-1"],
    ]
    assert all(sum(len(datum.tokens) for datum in call) <= 100 for call in sent)
    # Everything in one call when it fits.
    assert len(list(train_rl.calls(groups, 1_000))) == 1


def test_a_group_larger_than_a_call_is_refused() -> None:
    with pytest.raises(ValueError, match="the group of task 1 holds 60 tokens, over max_call"):
        list(train_rl.calls([group(40, "a"), group(60, "b")], 50))


def test_a_step_reports_its_episodes_means_and_the_groups_that_train() -> None:
    cut = episode(0.0, turns=1.0, truncated=1.0)
    groups = [
        [episode(1.0, turns=1.0, answered=1.0), episode(0.0, turns=3.0, answered=1.0)],
        # Every attempt scored the same: nothing to compare, so it trains nothing.
        [
            dataclasses.replace(cut, stop_reason=StopReason.MAX_TURN_TOKENS),
            dataclasses.replace(cut, stop_reason=StopReason.MAX_EPISODE_TOKENS),
        ],
    ]
    assert train_rl.step_metrics(groups) == {
        "reward": 0.25,
        "turns": 1.5,
        # A metric is a mean over the episodes that report it.
        "answered": 1.0,
        "truncated": 1.0,
        # The share each reason ended: which budget cut off each of the two.
        "stop/answer": 0.5,
        "stop/max_turns": 0.0,
        "stop/max_turn_tokens": 0.25,
        "stop/max_episode_tokens": 0.25,
        "episodes": 4.0,
        "groups": 2.0,
        "groups_trained": 1.0,
    }


def test_a_steps_version_is_the_oldest_of_its_turns() -> None:
    assert train_rl.oldest([episode(1, step=4), episode(1, step=3)]) == Version(
        run_id=RUN, epoch=0, step=3
    )
    # A later epoch is newer, whatever its step.
    assert train_rl.oldest([episode(1, step=9), episode(1, step=2, epoch=1)]) == Version(
        run_id=RUN, epoch=0, step=9
    )
    with pytest.raises(ValueError, match="names no version"):
        train_rl.oldest([episode(1, step=None)])
