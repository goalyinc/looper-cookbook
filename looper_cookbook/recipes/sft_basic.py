"""An SFT job from its data to its checkpoint, through the cookbook's SFT loop.

1. **The data.** By default GSM8K's train split: 7,473 grade-school math
   problems, each with a worked answer that ends in ``#### <number>`` (MIT),
   downloaded from Hugging Face at a pinned revision. With ``--data``, a
   JSONL file of conversations instead, each a line ``{"messages": [...]}``
   of chat messages:
   yours, or the bundled ``example_data/conversations.jsonl``.
2. **The dataset.** A ``ChatDataset`` puts the conversations in batches of
   ``--batch-size``, each pass in an order of its own. The model's message
   formatter writes each conversation as the model's own chat template
   does, one datum per assistant turn, and the script prints what the
   first datum trains (``looper_cookbook/datasets/``).
3. **The run.** A LoRA of rank ``--lora-rank``, or with ``--full`` every
   parameter, sized by ``--max-seq-len`` and the largest batch.
4. **The loop.** ``train_sft.train`` takes one optimizer step per batch,
   ``--epochs`` passes over the dataset, submitting ahead and decaying the
   learning rate itself (``looper_cookbook/train_sft.py``).
5. **The checkpoint.** It saves one and waits until it is uploaded.

In one pass, the default, a step's loss is computed before the run has
trained on that step's conversations, so a loss that falls over the run is
not a batch memorized. Nothing is held out and scored: the SDK has no
forward-only loss yet.

GSM8K's conversations carry no reasoning, so a Qwen model trains them in its
non-thinking mode: prompt the fine-tuned model through a formatter with
``enable_thinking=False`` too, as ``sampling_basic`` does. Pass ``--thinking``
for a file whose assistant turns carry ``reasoning_content``. Muse Glimmer
has no such mode, and the flag changes nothing for it.

    LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.sft_basic
    python -m looper_cookbook.recipes.sft_basic --data my.jsonl --epochs 2 --batch-size 16
    python -m looper_cookbook.recipes.sft_basic --full --learning-rate 1e-5

To check a deployment, train the bundled conversations: 8 steps, with no
dataset to download. ``--model``, ``--full`` and ``--max-seq-len`` pick the
profile to check.

    python -m looper_cookbook.recipes.sft_basic \\
        --data looper_cookbook/example_data/conversations.jsonl \\
        --batch-size 8 --epochs 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from looper_cookbook import train_sft
from looper_cookbook.datasets.conversations import ChatDataset, read_conversations
from looper_cookbook.message_formatters.models import message_formatter
from looper_sdk import (
    DEFAULT_BASE_URL,
    AdamW,
    Full,
    Lora,
    ServiceClient,
    StepResult,
    TrainingConfig,
)

GSM8K = "openai/gsm8k"
# The revision the README's numbers are from: a later edit of the dataset changes neither.
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
# The first step's rate when --learning-rate names none, near the profiles' own for each kind.
LEARNING_RATES = {"full": 1e-5, "lora": 5e-4}


def gsm8k_conversations() -> list[dict[str, Any]]:
    """GSM8K's train split as conversations, downloaded into Hugging Face's
    cache: each question, answered by its worked answer as the dataset has
    it, its ``<<48/2=24>>`` calculator notes and ``#### <number>`` line kept."""
    records = load_dataset(GSM8K, "main", split="train", revision=GSM8K_REVISION)
    # Naming a split loads that one Dataset, not the dict of every split.
    assert isinstance(records, Dataset)
    return [
        {
            "messages": [
                {"role": "user", "content": record["question"]},
                {"role": "assistant", "content": record["answer"]},
            ]
        }
        for record in records.to_list()
    ]


def chat_dataset(args: argparse.Namespace) -> ChatDataset:
    """The data's conversations in batches, a datum per assistant turn."""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    # A Qwen model's thinking mode; the fine-tuned model is prompted in the same one.
    formatter = message_formatter(args.model, tokenizer, enable_thinking=args.thinking)
    if args.data:
        conversations = read_conversations(args.data)
    else:
        conversations = gsm8k_conversations()
    if not conversations:
        raise SystemExit(f"{args.data} holds no conversation, so there is nothing to train")
    dataset = ChatDataset(
        conversations,
        formatter,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    # One pass holds every conversation once.
    data = [datum for index in range(len(dataset)) for datum in dataset.batch(index)]
    tokens = sum(len(datum.tokens) for datum in data)
    trained = sum(int(sum(datum.weights)) for datum in data)
    print(f"{len(conversations)} conversations: {len(data)} datums, {tokens} tokens,")
    print(f"of which {trained} train. The first step's first datum trains:")
    for text in formatter.targets(data[0]):
        print(f"  {text!r}")
    return dataset


def loss_over(steps: Sequence[StepResult]) -> str:
    """The loss per target over ``steps``, and the steps it covers."""
    loss = sum(step.loss_sum for step in steps) / sum(step.weighted_targets for step in steps)
    return f"{loss:.4f} over steps {steps[0].step} to {steps[-1].step}"


async def main(args: argparse.Namespace) -> None:
    dataset = chat_dataset(args)
    # A pass is len(dataset) steps, and the run's step k trains dataset.batch(k).
    total = args.epochs * len(dataset)
    kind = Full() if args.full else Lora(rank=args.lora_rank)
    peak = LEARNING_RATES["full" if args.full else "lora"]
    if args.learning_rate is not None:
        peak = args.learning_rate
    largest = max(sum(len(datum.tokens) for datum in dataset.batch(step)) for step in range(total))
    training = TrainingConfig(
        model=args.model,
        kind=kind,
        tokens_per_step=max(args.max_seq_len, largest),
        max_seq_len=args.max_seq_len,
        # The run's optimizer; each step names its own rate in place of this one.
        optimizer=AdamW(learning_rate=peak, betas=(0.9, 0.95), max_grad_norm=1.0),
    )
    loop_config = train_sft.Config(
        learning_rate=peak, schedule=args.schedule, steps_ahead=args.steps_ahead
    )

    def report(step: StepResult) -> None:
        loss = step.loss_sum / step.weighted_targets
        print(f"step {step.step}/{total}: lr {step.lr:.2e}, loss {loss:.4f}")
        if args.metrics:
            row = {"step": step.step, "lr": step.lr, "loss": loss, "grad_norm": step.grad_norm}
            with open(args.metrics, "a", encoding="utf-8") as metrics:
                metrics.write(json.dumps(row) + "\n")

    config = ServiceClient.Config(base_url=args.url, project=args.project, display_name="sft-basic")
    async with await config.build() as service:
        print("starting the run; the first one can take minutes")
        training_client = await service.create_training_client(training, timeout_s=20 * 60)
        steps = await train_sft.train(
            training_client, dataset, loop_config, steps=total, on_step=report
        )
        # A tenth of the run at each end: one step's loss is as noisy as its batch.
        window = max(1, len(steps) // 10)
        print(f"loss: {loss_over(steps[:window])}, {loss_over(steps[-window:])}")
        name = args.checkpoint or f"sft-basic-{uuid4().hex[:8]}"
        saved = await (await training_client.save_state(name)).result(timeout_s=300)
        # A LoRA's upload takes seconds; a 9B full checkpoint is 107.5 GB and takes minutes.
        ready = await service.wait_checkpoint(saved.id, timeout_s=15 * 60)
        print(f"checkpoint {name!r}: {ready.state.value}, {ready.size_bytes} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An SFT job: GSM8K, or your conversations.")
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--project", default="dev")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tokenizer", help="the tokenizer's name or directory; the model's")
    parser.add_argument("--data", help="a JSONL file of conversations; GSM8K without it")
    parser.add_argument("--thinking", action="store_true", help="the turns carry reasoning")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--full", action="store_true", help="train every parameter instead")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="a datum's longest length")
    parser.add_argument("--batch-size", type=int, default=64, help="conversations per step")
    parser.add_argument("--epochs", type=int, default=1, help="passes over the data")
    parser.add_argument("--seed", type=int, default=0, help="fixes each pass's order")
    parser.add_argument("--learning-rate", type=float, help="the first step's rate")
    parser.add_argument("--schedule", choices=train_sft.SCHEDULES, default="linear")
    parser.add_argument("--steps-ahead", type=int, default=1, help="steps submitted ahead")
    parser.add_argument("--checkpoint", help="the checkpoint's name; sft-basic-… by default")
    parser.add_argument("--metrics", help="a JSONL file to append each step's metrics to")
    asyncio.run(main(parser.parse_args()))
