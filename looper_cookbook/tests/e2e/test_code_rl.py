"""The ``code_rl`` recipe against a real Looper server, at a size that
takes minutes.

It runs the recipe's own ``main`` on a few of DeepCoder's problems: a LoRA
run of Qwen3.5-9B, its sampler, two steps of sixteen episodes, and the
held-out problems before and after. The model's programs run in Docker
containers on this machine, so it needs Docker as well as a server.

It shows that the recipe runs end to end on a real model: episodes sampled
from its prompts, the tool called and answered, programs scored on hidden
tests, a step trained on the groups. It makes no claim that the model
learns in two steps; the recipe's README records a longer run.

    LOOPER_URL=https://api.looper.goaly.ai LOOPER_TEST_DOCKER=1 pytest -m gpu -s \\
        looper_cookbook/tests/e2e/test_code_rl.py
"""

from __future__ import annotations

import os

import pytest

from looper_cookbook.recipes.code_rl import train

pytestmark = pytest.mark.gpu

URL = os.environ.get("LOOPER_URL")
BATCH_SIZE = 4
GROUP_SIZE = 4
HELD_OUT = 4


async def test_the_recipe_trains_two_steps_and_measures_held_out_problems() -> None:
    if URL is None:
        pytest.skip("Set LOOPER_URL to the /v1 of a Looper server with GPUs behind it")
    if os.environ.get("LOOPER_TEST_DOCKER") != "1":
        pytest.skip("Set LOOPER_TEST_DOCKER=1: the model's programs run in Docker containers")
    flags = [
        f"--url={URL}",
        "--project=e2e",
        "--problems=8",
        f"--eval-problems={HELD_OUT}",
        "--steps=2",
        f"--batch-size={BATCH_SIZE}",
        f"--group-size={GROUP_SIZE}",
        "--max-turn-tokens=1024",
    ]
    reports = await train.main(train.arguments().parse_args(flags))

    assert [report.step for report in reports] == [0, 1, 2]
    baseline, *steps = reports
    # Both test subsets, each cut to HELD_OUT problems, before anything trained.
    assert baseline.result is None
    assert baseline.metrics["eval/episodes"] == 2 * HELD_OUT
    for report in steps:
        assert report.result is not None and report.result.step == report.step
        assert report.version is not None and report.version.step == report.step - 1
        metrics = report.metrics
        assert metrics["episodes"] == BATCH_SIZE * GROUP_SIZE
        assert 0 <= metrics["reward"] <= 1
        # An episode is at most a check and an answer.
        assert 1 <= metrics["turns"] <= 2
    assert 0 <= reports[-1].metrics["eval/reward"] <= 1
    # Across two steps and the held-out attempts, the model called the tool and it answered.
    assert any("passed" in report.metrics or "eval/passed" in report.metrics for report in reports)
