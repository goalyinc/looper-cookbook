"""The asynchronous RL loop against a real Looper server: a LoRA run of
Qwen3.5-9B whose sampler goes on sampling while each step trains.

One run of three steps at ``max_policy_lag=1`` shows the loop's contract as
the run reports it:

- **Sampling runs ahead of training.** The loop starts two steps' groups
  at once, so the second step trains groups sampled before the first
  step's weights were published: its ``policy_lag_max`` is 1.
- **No group trains past the bound.** Every step's oldest version is at
  most one step behind the weights the step trains.
- **A step trains as many groups as its batch holds,** whichever finished
  first.
- **The baseline and the evaluations are ``train_rl``'s:** before the
  first step, when due, and after the last.

It makes no claim that the model learns, or that the overlap saves time:
three steps of sixteen episodes show the mechanics.

It calls the ``/v1`` that ``$LOOPER_URL`` names, and takes a trainer and a
sampler on its GPUs for a few minutes. Opt-in, as every GPU test is; ``-s``
prints each report::

    LOOPER_URL=https://api.looper.goaly.ai pytest -m gpu -s \\
        looper_cookbook/tests/e2e/test_train_rl_async.py
"""

from __future__ import annotations

import json
import os
from importlib import resources

import pytest
from transformers import AutoTokenizer

from looper_cookbook import train_rl, train_rl_async
from looper_cookbook.datasets.tasks import TaskDataset
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.message_formatters.models import message_formatter
from looper_cookbook.verifiers.final_number import FinalNumber
from looper_sdk import AdamW, Lora, SamplingParams, ServiceClient, TrainingConfig

pytestmark = pytest.mark.gpu

URL = os.environ.get("LOOPER_URL")
MODEL = "Qwen/Qwen3.5-9B"
# Placement, the image's pull and the weights' load, for the run and for its sampler.
READY_S = 20 * 60
MAX_TURN_TOKENS = 256
BATCH_SIZE = 4
GROUP_SIZE = 4
HELD_OUT = 4
MAX_POLICY_LAG = 1
STEPS = 3
INSTRUCTION = "Solve it step by step, then give the final answer as a number after '####'."


def asked(problem: dict[str, str]) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{problem['question']}\n{INSTRUCTION}"}]


async def test_the_sampler_runs_ahead_of_training_within_the_lag() -> None:
    if URL is None:
        pytest.skip("Set LOOPER_URL to the /v1 of a Looper server with GPUs behind it")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    formatter = message_formatter(MODEL, tokenizer, enable_thinking=False)
    path = resources.files("looper_cookbook") / "example_data" / "math_problems.jsonl"
    problems = [json.loads(line) for line in path.read_text().splitlines()]
    prompts = [formatter.build_generation_prompt(asked(problem)) for problem in problems]
    episode_tokens = max(len(prompt) for prompt in prompts) + MAX_TURN_TOKENS
    tasks = [
        AgentSession.Config(
            messages=asked(problem),
            verifier=FinalNumber.Config(answer=str(problem["answer"])),
            max_episode_tokens=episode_tokens,
        )
        for problem in problems
    ]
    dataset = TaskDataset(tasks[:-HELD_OUT], batch_size=BATCH_SIZE, seed=0)
    loop_config = train_rl_async.Config(
        params=SamplingParams(max_tokens=MAX_TURN_TOKENS, temperature=1.0),
        group_size=GROUP_SIZE,
        max_policy_lag=MAX_POLICY_LAG,
        eval_every=2,
    )
    training = TrainingConfig(
        model=MODEL,
        kind=Lora(rank=32),
        tokens_per_step=BATCH_SIZE * GROUP_SIZE * episode_tokens,
        max_seq_len=episode_tokens,
        optimizer=AdamW(learning_rate=8e-5, betas=(0.9, 0.95), max_grad_norm=1.0),
    )

    def show(report: train_rl.StepReport) -> None:
        print(f"step {report.step}: version {report.version}")
        print(f"  {dict(report.metrics)}")

    config = ServiceClient.Config(base_url=URL, project="e2e", display_name="train-rl-async")
    async with await config.build() as service:
        training_client = await service.create_training_client(training, timeout_s=READY_S)
        # Room for the two batches the loop keeps in flight at this lag.
        sampling_client = await training_client.create_sampling_client(
            max_seq_len=episode_tokens,
            max_concurrent_sequences=(MAX_POLICY_LAG + 1) * BATCH_SIZE * GROUP_SIZE,
            timeout_s=READY_S,
        )
        reports = await train_rl_async.train(
            training_client,
            sampling_client,
            dataset,
            loop_config,
            formatter=formatter,
            steps=STEPS,
            eval_tasks=tasks[-HELD_OUT:],
            on_step=show,
        )
        status = await training_client.status()

    assert status.step == STEPS
    assert [report.step for report in reports] == [0, 1, 2, 3]
    baseline, one, two, three = reports

    # The baseline, measured before any step, while the first groups sampled.
    assert baseline.result is None and baseline.version is None
    assert baseline.metrics["eval/episodes"] == HELD_OUT
    assert "reward" not in baseline.metrics

    for report in (one, two, three):
        assert report.result is not None and report.result.step == report.step
        assert report.metrics["groups"] == BATCH_SIZE
        assert report.metrics["episodes"] == BATCH_SIZE * GROUP_SIZE
        assert 0 <= report.metrics["reward"] <= 1
        assert report.metrics["groups_stale"] >= 0
        # The step trained the weights of the step before it, on groups no older than the bound.
        assert report.version is not None
        lag = (report.step - 1) - report.version.step
        assert 0 <= lag <= MAX_POLICY_LAG
        assert report.metrics["policy_lag_max"] == lag
        assert 0 <= report.metrics["policy_lag"] <= lag

    # The first step's groups sampled the weights it trained.
    assert one.metrics["policy_lag_max"] == 0
    # The overlap: the second step trained groups that sampled while the first step trained.
    assert two.metrics["policy_lag_max"] == 1

    # Evaluated when due, every second step and after the last.
    assert "eval/reward" not in one.metrics
    for report in (two, three):
        assert 0 <= report.metrics["eval/reward"] <= 1
