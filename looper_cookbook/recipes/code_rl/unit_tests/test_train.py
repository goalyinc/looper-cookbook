"""The recipe's own parts that need no server: which problems fit an
episode, the line a report prints as, and its flags
(looper_cookbook/recipes/code_rl/train.py)."""

from __future__ import annotations

import pytest

from looper_cookbook import train_rl
from looper_cookbook.environments.coding import TOOLS, CodeTest
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter
from looper_cookbook.message_formatters.unit_tests import tokenizers
from looper_cookbook.recipes.code_rl import tasks, train

FORMATTER = Qwen3_5MessageFormatter(tokenizers.qwen(), enable_thinking=False)
TESTS = (CodeTest(input="1", output="1"),)


def test_a_problem_fits_when_its_prompt_with_the_tool_leaves_room_for_a_turn() -> None:
    short = tasks.Problem(question="Echo.", tests=TESTS)
    long = tasks.Problem(question="Echo. " * 400, tests=TESTS)
    asked = [{"role": "user", "content": short.question}]
    prompt = len(FORMATTER.build_generation_prompt(asked, TOOLS))
    assert train.fitting([short, long], FORMATTER, room=prompt) == [short]
    assert train.fitting([short, long], FORMATTER, room=prompt - 1) == []


def test_a_report_prints_as_its_step_and_the_shown_metrics_it_holds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The baseline holds the held-out metrics alone, and a step between evaluations its own.
    held_out = {"eval/reward": 0.25, "eval/reward_stderr": 0.0625, "eval/episodes": 8.0}
    trained = {"reward": 0.5, "turns": 1.25, "stop/max_turn_tokens": 0.75, "episodes": 16.0}
    train.show(train_rl.StepReport(step=0, result=None, version=None, metrics=held_out))
    train.show(train_rl.StepReport(step=1, result=None, version=None, metrics=trained))
    assert capsys.readouterr().out == (
        "step 0: eval/reward 0.250, eval/reward_stderr 0.062\n"
        "step 1: reward 0.500, turns 1.250, stop/max_turn_tokens 0.750\n"
    )


def test_the_defaults_are_two_turns_in_docker_on_what_a_9b_sampler_holds() -> None:
    args = train.arguments().parse_args([])
    assert (args.sandbox, args.image) == ("docker", "python:3.12-slim")
    assert (args.max_turns, args.max_turn_tokens, args.max_episode_tokens) == (2, 2048, 8192)
    assert (args.batch_size, args.group_size, args.public_tests, args.max_tests) == (16, 8, 2, 32)
    assert args.problems is None and args.eval_problems == 64
