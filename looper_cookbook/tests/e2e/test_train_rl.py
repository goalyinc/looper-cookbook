"""The RL loop against a real Looper server: a LoRA run of Qwen3.5-9B
trained on episodes its own sampler draws.

One run, trained by two calls of ``train_rl.train``, shows the loop's
contract as the run reports it:

- **Each step trains on the step before it.** A step's ``version`` is the
  step it sampled from.
- **An evaluation samples the weights just published.** The loop evaluates
  after it awaits a step's publish, and when a call returns, the sampler
  serves that call's last step.
- **The first report is the baseline,** of the step the call started at.
- **A save lands where ``save_every`` puts it,** named for the run and
  its step.
- **A second call goes on from the run's step,** and reads that step's
  batch.

It makes no claim that the model learns: three steps of sixteen episodes
show the mechanics, and a recipe's gate shows learning.

It calls the ``/v1`` that ``$LOOPER_URL`` names, and takes a trainer and a
sampler on its GPUs for a few minutes. Opt-in, as every GPU test is; ``-s``
prints each report::

    LOOPER_URL=https://api.looper.goaly.ai pytest -m gpu -s \\
        looper_cookbook/tests/e2e/test_train_rl.py
"""

from __future__ import annotations

import json
import os
from importlib import resources

import pytest
from transformers import AutoTokenizer

from looper_cookbook import train_rl
from looper_cookbook.datasets.tasks import TaskDataset
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.message_formatters.models import message_formatter
from looper_cookbook.verifiers.final_number import FinalNumber
from looper_sdk import (
    AdamW,
    Lora,
    SamplingParams,
    ServiceClient,
    TrainingConfig,
)

pytestmark = pytest.mark.gpu

URL = os.environ.get("LOOPER_URL")
MODEL = "Qwen/Qwen3.5-9B"
# Placement, the image's pull and the weights' load, for the run and for its sampler.
READY_S = 20 * 60
MAX_TURN_TOKENS = 256
BATCH_SIZE = 4
GROUP_SIZE = 4
HELD_OUT = 4
INSTRUCTION = "Solve it step by step, then give the final answer as a number after '####'."


def asked(problem: dict[str, str]) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{problem['question']}\n{INSTRUCTION}"}]


async def test_two_calls_train_one_run_and_report_each_step() -> None:
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
    greedy = SamplingParams(max_tokens=MAX_TURN_TOKENS, temperature=0.0)

    loop_config = train_rl.Config(
        params=SamplingParams(max_tokens=MAX_TURN_TOKENS, temperature=1.0),
        group_size=GROUP_SIZE,
        eval_every=2,
        save_every=2,
    )
    training = TrainingConfig(
        model=MODEL,
        kind=Lora(rank=32),
        tokens_per_step=BATCH_SIZE * GROUP_SIZE * episode_tokens,
        max_seq_len=episode_tokens,
        optimizer=AdamW(learning_rate=8e-5, betas=(0.9, 0.95), max_grad_norm=1.0),
    )

    def show(report: train_rl.StepReport) -> None:
        saved = report.checkpoint.name if report.checkpoint is not None else None
        print(f"step {report.step}: version {report.version}, saved {saved}")
        print(f"  {dict(report.metrics)}")

    config = ServiceClient.Config(base_url=URL, project="e2e", display_name="train-rl")
    async with await config.build() as service:
        training_client = await service.create_training_client(training, timeout_s=READY_S)
        sampler = await training_client.create_sampling_client(
            max_seq_len=episode_tokens,
            max_concurrent_sequences=BATCH_SIZE * GROUP_SIZE,
            timeout_s=READY_S,
        )

        async def served() -> int:
            """The step of the weights that answer a sample now."""
            operation = await sampler.sample(prompts[0], params=greedy)
            (sequence,) = (await operation.result(timeout_s=300)).sequences
            assert sequence.version is not None
            return sequence.version.step

        async def train(steps: int) -> list[train_rl.StepReport]:
            return await train_rl.train(
                training_client,
                sampler,
                dataset,
                loop_config,
                formatter=formatter,
                steps=steps,
                eval_tasks=tasks[-HELD_OUT:],
                on_step=show,
            )

        first = await train(2)
        after_first = await served()
        second = await train(3)
        after_second = await served()
        status = await training_client.status()

    assert status.step == 3
    assert [report.step for report in first] == [0, 1, 2]
    # A second call goes on from where the run stands.
    assert [report.step for report in second] == [2, 3]
    baseline, one, two = first
    again, three = second

    # The baseline, and the second call's, evaluated before either trained a step.
    for report in (baseline, again):
        assert report.result is None and report.version is None and report.checkpoint is None
        assert report.metrics["eval/episodes"] == HELD_OUT
        assert "reward" not in report.metrics

    for report in (one, two, three):
        assert report.result is not None and report.result.step == report.step
        # Sampled from the weights of the step before the one it trains.
        assert report.version is not None and report.version.step == report.step - 1
        assert report.metrics["episodes"] == BATCH_SIZE * GROUP_SIZE
        assert report.metrics["groups"] == BATCH_SIZE
        assert 0 <= report.metrics["groups_trained"] <= BATCH_SIZE
        assert 0 <= report.metrics["reward"] <= 1

    # Evaluated when due, every second step and after a call's last.
    assert "eval/reward" not in one.metrics
    for report in (two, three):
        assert 0 <= report.metrics["eval/reward"] <= 1
    # Each call's last evaluation came after its last publish: the sampler serves that step.
    assert (after_first, after_second) == (2, 3)

    # Saved at every second step, under the run's name and the step's.
    assert one.checkpoint is None and three.checkpoint is None
    assert two.checkpoint is not None
    assert two.checkpoint.name == f"{training_client.run_id}-step-2"
    assert two.checkpoint.version.step == 2
