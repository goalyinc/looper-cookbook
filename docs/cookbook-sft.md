# SFT with the cookbook

This page takes you from a file of your own examples to a fine-tuned model you can try, using the cookbook. It's the place to start if you want to run supervised fine-tuning (SFT) without writing the training loop yourself.

There are two ways to run it, and both use the same loop underneath:

- **A ready-made script,** `sft_basic`, which you run from the command line. You write no code, only your data file.
- **A short script of your own,** which you can change. Start here if you want to add things later, such as your own data loading.

This page covers:

- [Before you start](#before-you-start): what you need installed and set.
- [Get your data ready](#get-your-data-ready): the file format, and how to convert a spreadsheet into it.
- [Run the ready-made script](#run-the-ready-made-script): train with one command.
- [Run your own script](#run-your-own-script): the same training, as a script you can change.
- [Try the trained model](#try-the-trained-model): ask it questions, and compare it with the model before training.
- [Is it working?](#is-it-working): what to look at while it trains, and after.
- [Keep your program running](#keep-your-program-running): what happens if your program stops partway.
- [Next steps](#next-steps)

## Before you start

- **Set up the SDK and your key,** as in [Quickstart](../../server/docs/quickstart.md).
- **Install the cookbook.** It isn't on PyPI yet, so you install it from a clone of its repository, [goalyinc/looper-cookbook](https://github.com/goalyinc/looper-cookbook). [The cookbook](index.md#run-it) shows how.
- **Pick your settings.** Everything on this page uses the starting settings from [Choosing settings](../../server/docs/choosing-settings.md), which work for a first run.

## Get your data ready

Your data is a JSONL file, which is a text file with one JSON object per line. Each line is one conversation, written as `{"messages": [...]}`, where each message has a `role` and its `content`:

```json
{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Go to Settings, then Account, and click Reset password."}]}
```

- **The model learns to write the `assistant` messages.** The other messages are only context, so the model reads them but isn't trained to write them.
- **A conversation can have several turns.** Each assistant reply is trained after the conversation that came before it.
- **You can start with a `system` message,** such as instructions for the assistant. Put it first, and use the same one when you try the model.

If your examples are in a spreadsheet, save it as CSV and convert it with a few lines of Python. This one reads a file with `question` and `answer` columns:

```python
import csv
import json

with open("examples.csv", newline="", encoding="utf-8") as source, open(
    "my_conversations.jsonl", "w", encoding="utf-8"
) as out:
    for row in csv.DictReader(source):
        messages = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        out.write(json.dumps({"messages": messages}) + "\n")
```

Change the column names to match yours. Before you train, read a few lines of the output file to check they look right.

How much data you need depends on what you're teaching. A few hundred good examples can be enough to teach a format or a style. Teaching new knowledge or a harder skill takes more. It's usually better to start small, check the result, then add more.

## Run the ready-made script

From the root of your clone of the cookbook, with `LOOPER_API_KEY` set, run:

```bash
python -m looper_cookbook.recipes.sft_basic --data my_conversations.jsonl --project my-project
```

It trains a LoRA of `Qwen/Qwen3.5-9B` on your file for one pass, saves a checkpoint, and waits until the checkpoint is ready. Starting the run can take a few minutes while Looper finds GPUs. As it trains, it prints each step's loss. At the end, it prints the checkpoint's name, such as `sft-basic-1a2b3c4d`. Keep that name, because you'll use it to try the model.

The options you're most likely to change:

| Option | Default | What it sets |
|---|---|---|
| `--model` | `Qwen/Qwen3.5-9B` | The model to train. See [Available models](../../server/docs/models.md). |
| `--epochs` | 1 | How many times to go through your file. |
| `--batch-size` | 64 | How many conversations each training step uses. |
| `--learning-rate` | 5e-4 for LoRA | How big each update is. |
| `--lora-rank` | 32 | The size of the LoRA. |
| `--max-seq-len` | 2048 | The longest conversation it can train on, in tokens. |
| `--thinking` | Off | Turn this on if your assistant messages have `reasoning_content`. |
| `--checkpoint` | `sft-basic-` and a random id | The checkpoint's name. |

[Choosing settings](../../server/docs/choosing-settings.md) explains when to change each one. [Recipes](recipes.md) lists every option.

## Run your own script

This script does the same training as `sft_basic`, in about 50 lines you can read and change. Save it as `sft.py`, set `LOOPER_API_KEY` and `LOOPER_PROJECT`, and run it with `python sft.py`:

```python
import asyncio
import os
from uuid import uuid4

from transformers import AutoTokenizer

from looper_cookbook import train_sft
from looper_cookbook.datasets.conversations import ChatDataset, read_conversations
from looper_cookbook.message_formatters.models import message_formatter
from looper_sdk import AdamW, Lora, ServiceClient, TrainingConfig

MODEL = "Qwen/Qwen3.5-9B"
DATA = "my_conversations.jsonl"
LORA_RANK = 32
MAX_SEQ_LEN = 2048
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 5e-4

# Turn the conversations into training data, in batches.
tokenizer = AutoTokenizer.from_pretrained(MODEL)
formatter = message_formatter(MODEL, tokenizer, enable_thinking=False)
conversations = read_conversations(DATA)
dataset = ChatDataset(
    conversations, formatter, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, seed=0
)
steps = EPOCHS * len(dataset)
largest = max(sum(len(datum.tokens) for datum in dataset.batch(step)) for step in range(steps))
print(f"{len(conversations)} conversations, {steps} steps")

# The training run on Looper, and the cookbook's loop on your machine.
training = TrainingConfig(
    model=MODEL,
    kind=Lora(rank=LORA_RANK),
    tokens_per_step=max(MAX_SEQ_LEN, largest),
    max_seq_len=MAX_SEQ_LEN,
    optimizer=AdamW(learning_rate=LEARNING_RATE, betas=(0.9, 0.95), max_grad_norm=1.0),
)
loop_config = train_sft.Config(learning_rate=LEARNING_RATE, schedule="linear", steps_ahead=1)


def report(step):
    loss = step.loss_sum / step.weighted_targets
    print(f"step {step.step}/{steps}: loss {loss:.4f}")


async def main():
    config = ServiceClient.Config(
        project=os.environ["LOOPER_PROJECT"],
        display_name="sft",
    )
    async with await config.build() as service:
        print("starting the run, which can take a few minutes...")
        training_client = await service.create_training_client(training, timeout_s=20 * 60)
        await train_sft.train(training_client, dataset, loop_config, steps=steps, on_step=report)

        name = f"sft-{uuid4().hex[:8]}"
        saved = await (await training_client.save_state(name)).result(timeout_s=300)
        await service.wait_checkpoint(saved.id, timeout_s=15 * 60)
        print("saved the trained model as", name)


asyncio.run(main())
```

What each part does:

- **The data.** `read_conversations` reads your file. The model's **message formatter** writes each conversation in the model's own chat format, and `ChatDataset` splits them into batches, one per training step. `enable_thinking=False` turns off Qwen's thinking mode, since your answers have no reasoning in them. [Message formatting](message-formatting.md) explains the formatter.
- **`steps`** is how many training steps make `EPOCHS` passes through your file. `largest` is the biggest batch, in tokens, which sets the run's size.
- **`TrainingConfig`** describes the training run on Looper. [Choosing settings](../../server/docs/choosing-settings.md) explains each value.
- **`train_sft.Config`** sets up the cookbook's loop. `schedule="linear"` lowers the learning rate step by step, down to 0 at the end. `steps_ahead=1` keeps one step queued, so Looper's GPUs don't wait for your program.
- **`train_sft.train`** runs every step, and calls `report` after each one.
- **The save** stores the trained model as a checkpoint, under a random name, and waits until it's ready to use. The loop doesn't save for you.

[Supervised learning](supervised-learning.md) explains the dataset and the loop in full, including how to load data that isn't a file of conversations.

## Try the trained model

This script asks a model one question. Run it with your checkpoint's name, and then with the base model, to compare the answers before and after training:

```bash
python try.py sft-1a2b3c4d "How do I change my email address?"
python try.py base:Qwen/Qwen3.5-9B "How do I change my email address?"
```

Save it as `try.py`:

```python
import asyncio
import os
import sys

from transformers import AutoTokenizer

from looper_cookbook.message_formatters.models import message_formatter
from looper_sdk import SamplerConfig, SamplingParams, ServiceClient

MODEL = "Qwen/Qwen3.5-9B"
checkpoint, question = sys.argv[1], sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
formatter = message_formatter(MODEL, tokenizer, enable_thinking=False)


async def main():
    config = ServiceClient.Config(
        project=os.environ["LOOPER_PROJECT"],
    )
    async with await config.build() as service:
        sampler = await service.create_sampling_client(
            SamplerConfig(checkpoint_name=checkpoint, max_seq_len=2048, max_concurrent_sequences=4),
            timeout_s=20 * 60,
        )
        try:
            prompt = formatter.build_generation_prompt([{"role": "user", "content": question}])
            params = SamplingParams(
                max_tokens=512, temperature=0.7, stop_token_ids=formatter.stop_token_ids
            )
            result = await (await sampler.sample(prompt, params=params)).result(timeout_s=300)
        finally:
            await sampler.stop()
    message, _ = formatter.parse_response(result.sequences[0].token_ids)
    print(message["content"])


asyncio.run(main())
```

- **Use the same `MODEL` and formatter settings you trained with,** so the question is written in the same format the model learned.
- **If you trained with a `system` message,** put the same one before the question.
- **Starting a sampler takes a few minutes,** the same as starting a training run.

[Sampling](../../server/docs/sampling.md) explains every option, and [Checkpoint](../../server/docs/checkpoint.md) explains how checkpoints are kept.

## Is it working?

**While it trains, watch the loss.** The loss measures how far the model's predictions are from your examples. It should go down over the run, quickly at first and then more slowly. Each step's loss jumps around, because each batch is different, so look at the trend over many steps rather than at single steps. [Choosing settings](../../server/docs/choosing-settings.md#signs-to-watch-for) lists what to do if it doesn't fall, or if it falls too far.

**After it trains, read what it writes.** A low loss shows the model learned your examples, but not that it does well on new questions. The best check is to keep 20 or so questions out of your training file, ask them to the trained model and the base model, and compare the answers yourself.

There's no automatic score for this, because SFT answers usually can't be marked right or wrong by code. If your answers can be checked by code, such as a final number or a passing test, [Evaluation](evaluation.md) can score them for you.

## Keep your program running

Your program has to keep running for the whole training run. The training happens on Looper, but your program sends it each step, and it keeps the session open.

- **If your program stops,** for example because your laptop goes to sleep or a notebook's kernel restarts, Looper stops hearing from it. After 60 seconds, the session expires, and its training run is shut down. See [When a session expires](../../server/docs/session.md#when-a-session-expires).
- **A run can't be resumed from a checkpoint yet.** If the run is shut down partway, you start again from the beginning. Checkpoints you saved are kept, and you can still sample them.

So for a long run, run your script on a machine that stays on, such as a server or a cloud VM, rather than a laptop.

## Next steps

- [Choosing settings](../../server/docs/choosing-settings.md): when to change the model, the learning rate and the other settings.
- [Supervised learning](supervised-learning.md): the dataset and the loop in full.
- [Message formatting](message-formatting.md): how conversations become training data, including thinking and tools.
- [Checkpoint](../../server/docs/checkpoint.md): what you can do with the trained model.
