"""The SFT loop against a real Looper server: a LoRA run of Qwen3.5-9B
trained on the bundled conversations.

One run, trained by two calls of ``train_sft.train``, shows the loop's
contract as the run reports it:

- **The run's step picks the batch.** Each step reports the targets and
  the tokens of ``dataset.batch(k)``, and no two of the batches here have
  the same counts.
- **Each step trains at the rate the schedule names for it.**
- **A second call goes on from the run's step,** with that step's batch,
  into the dataset's second pass.
- **Results come in step order,** with a step submitted ahead.

It makes no claim that the model learns: six steps of four conversations
show the mechanics, and a recipe's gate shows learning.

It calls the ``/v1`` that ``$LOOPER_URL`` names, and takes a trainer on its
GPUs for a few minutes. Opt-in, as every GPU test is; ``-s`` prints each
step::

    LOOPER_URL=https://api.looper.goaly.ai pytest -m gpu -s \\
        looper_cookbook/tests/e2e/test_train_sft.py
"""

from __future__ import annotations

import os
from importlib import resources

import pytest
from transformers import AutoTokenizer

from looper_cookbook import train_sft
from looper_cookbook.datasets.conversations import ChatDataset, read_conversations
from looper_cookbook.message_formatters.models import message_formatter
from looper_sdk import AdamW, Lora, ServiceClient, StepResult, TrainingConfig

pytestmark = pytest.mark.gpu

URL = os.environ.get("LOOPER_URL")
MODEL = "Qwen/Qwen3.5-9B"
# Placement, the image's pull and the weights' load.
READY_S = 20 * 60
MAX_SEQ_LEN = 512
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
# The first call's last step and the second's: four batches are a pass, so the second crosses it.
FIRST, SECOND = 3, 6


async def test_two_calls_train_one_run_on_the_batch_of_each_step() -> None:
    if URL is None:
        pytest.skip("Set LOOPER_URL to the /v1 of a Looper server with GPUs behind it")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    formatter = message_formatter(MODEL, tokenizer, enable_thinking=False)
    path = resources.files("looper_cookbook") / "example_data" / "conversations.jsonl"
    dataset = ChatDataset(
        read_conversations(str(path)),
        formatter,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        seed=0,
    )
    batches = [dataset.batch(step) for step in range(SECOND)]
    # What a step reports of its batch: its tokens, and the targets among them.
    counts = [
        (
            sum(len(datum.tokens) for datum in batch),
            sum(int(sum(datum.weights)) for datum in batch),
        )
        for batch in batches
    ]
    assert len(set(counts)) == SECOND, "two batches count the same, so a step could not tell them"
    loop_config = train_sft.Config(learning_rate=LEARNING_RATE, schedule="linear", steps_ahead=1)
    training = TrainingConfig(
        model=MODEL,
        kind=Lora(rank=32),
        tokens_per_step=max(MAX_SEQ_LEN, *(tokens for tokens, _ in counts)),
        max_seq_len=MAX_SEQ_LEN,
        optimizer=AdamW(learning_rate=LEARNING_RATE, betas=(0.9, 0.95), max_grad_norm=1.0),
    )
    reported: list[int] = []

    def show(result: StepResult) -> None:
        reported.append(result.step)
        loss = result.loss_sum / result.weighted_targets
        print(f"step {result.step}: lr {result.lr:.2e}, loss {loss:.4f}")

    config = ServiceClient.Config(base_url=URL, project="e2e", display_name="train-sft")
    async with await config.build() as service:
        training_client = await service.create_training_client(training, timeout_s=READY_S)
        first = await train_sft.train(
            training_client, dataset, loop_config, steps=FIRST, on_step=show
        )
        second = await train_sft.train(
            training_client, dataset, loop_config, steps=SECOND, on_step=show
        )
        status = await training_client.status()

    assert status.step == SECOND
    assert [result.step for result in first] == [1, 2, 3]
    # A second call goes on from where the run stands.
    assert [result.step for result in second] == [4, 5, 6]
    assert reported == [1, 2, 3, 4, 5, 6]
    for results, steps in ((first, FIRST), (second, SECOND)):
        for result in results:
            # The run's step k, counted from 0, trained dataset.batch(k).
            tokens, targets = counts[result.step - 1]
            assert result.weighted_targets == targets
            assert result.num_tokens == tokens
            # At the rate the call's schedule gives that step.
            rate = train_sft.learning_rate(loop_config, result.step - 1, steps)
            assert result.lr == pytest.approx(rate)
