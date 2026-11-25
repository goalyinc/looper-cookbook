# RL with the cookbook

This page takes you from a file of your own problems to a model trained with reinforcement learning (RL), using the cookbook. It's the place to start if you want to run RL without writing the training loop yourself.

In RL, you don't give the model example answers. You give it problems, and a way to check its answers. The model writes several answers to each problem, each answer gets a score, called its **reward**, and the model learns to write answers like the ones that scored well. [How training works](../../server/docs/how-training-works.md) explains this in more detail.

There are two ways to run it, and both use the same loop underneath:

- **A ready-made script,** `rl_basic`, which you run from the command line. It works for problems whose answer is a number.
- **A short script of your own,** which you can change. Start here if your answers need to be checked another way, such as by running tests.

This page covers:

- [Before you start](#before-you-start): what you need installed and set.
- [Choose your problems](#choose-your-problems): which problems teach the model, and how to check yours.
- [Get your data ready](#get-your-data-ready): the file format.
- [Run the ready-made script](#run-the-ready-made-script): train with one command.
- [Run your own script](#run-your-own-script): the same training, as a script you can change.
- [Is it working?](#is-it-working): the numbers to watch while it trains.
- [Keep your program running](#keep-your-program-running): what happens if your program stops partway.
- [Next steps](#next-steps)

## Before you start

- **Set up the SDK and your key,** as in [Quickstart](../../server/docs/quickstart.md).
- **Install the cookbook.** It isn't on PyPI yet, so you install it from a clone of its repository, [goalyinc/looper-cookbook](https://github.com/goalyinc/looper-cookbook). [The cookbook](index.md#run-it) shows how.
- **Pick your settings.** Everything on this page uses the starting settings from [Choosing settings](../../server/docs/choosing-settings.md), which work for a first run.

## Choose your problems

This matters more than any setting. RL learns by comparing the answers the model writes to the same problem. Answers that scored better than the others become more likely, and answers that scored worse become less likely.

So a problem only teaches the model something when its answers get different scores:

- **A problem the model always gets right** teaches it nothing, because every answer scores the same.
- **A problem the model always gets wrong** teaches it nothing either, for the same reason.
- **A problem the model gets right some of the time** is where it learns. Problems it gets right about half the time teach the most.

Before you train, check how the model does on your problems. The script below measures the model on some held-out problems before it trains, so its first line of output tells you. If the model already gets nearly everything right, or nearly everything wrong, change your problems before you spend time training. Make them harder or easier, or mix in problems of different difficulty.

While it trains, the loop reports how many groups of answers had different scores (`groups_trained`). If that number stays near 0, the model isn't learning from your problems.

## Get your data ready

The ready-made script and the script below both take a JSONL file, which is a text file with one JSON object per line. Each line is one problem, with its `question` and its `answer`, which is a number:

```json
{"question": "A shop sells pens at 3 for $2. How much do 12 pens cost?", "answer": "8"}
```

The model is asked each question with an instruction to finish with `####` and the number. Its answer scores 1 if the number after `####` is right, and 0 if not.

If your answers aren't numbers, you need a different way to score them, called a **verifier**. [Verifiers](verifiers.md) explains how verifiers work, and how to write your own. [Using your verifier with the cookbook](verifiers.md#using-your-verifier-with-the-cookbook) shows where it goes in the script below.

Keep some problems out of training, so you can measure the model on problems it hasn't seen. 50 to 100 gives you a rough score, and more gives you a steadier one. Put them in a separate file for the ready-made script, or at the end of your file for the script below.

## Run the ready-made script

From the root of your clone of the cookbook, with `LOOPER_API_KEY` set, run:

```bash
python -m looper_cookbook.recipes.rl_basic --data my_problems.jsonl --eval-data my_held_out.jsonl --project my-project
```

It trains a LoRA of `Qwen/Qwen3.5-9B` for 50 steps, then saves a checkpoint. Starting the run and its sampler can take a few minutes while Looper finds GPUs. It measures the held-out problems before the first step and after the last, and prints the share it got right each time, such as `before: held out 0.420 ± 0.050; after: held out 0.610 ± 0.049`. At the end, it prints the checkpoint's name.

The options you're most likely to change:

| Option | Default | What it sets |
|---|---|---|
| `--model` | `Qwen/Qwen3.5-9B` | The model to train. See [Available models](../../server/docs/models.md). |
| `--steps` | 50 | How many training steps to run. |
| `--batch-size` | 32 | How many problems each step uses. |
| `--group-size` | 16 | How many answers the model writes to each problem. |
| `--max-turn-tokens` | 512 | The longest answer the model can write, in tokens. An answer that runs out of room scores 0. |
| `--learning-rate` | 8e-5 for LoRA | How big each update is. |
| `--eval-every` | Only before and after | How often to measure the held-out problems. |
| `--thinking` | Off | Let the model reason before it answers. |

[Choosing settings](../../server/docs/choosing-settings.md) explains when to change each one. [Recipes](recipes.md) lists every option.

## Run your own script

This script does the same training as `rl_basic`, and you can change any part of it. Save it as `rl.py`, set `LOOPER_API_KEY` and `LOOPER_PROJECT`, and run it with `python rl.py`. It holds out the last `HELD_OUT` problems in your file:

```python
import asyncio
import json
import os
from uuid import uuid4

from transformers import AutoTokenizer

from looper_cookbook import train_rl
from looper_cookbook.datasets.tasks import TaskDataset
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.message_formatters.models import message_formatter
from looper_cookbook.verifiers.final_number import FinalNumber
from looper_sdk import AdamW, Lora, SamplingParams, ServiceClient, TrainingConfig

MODEL = "Qwen/Qwen3.5-9B"
DATA = "my_problems.jsonl"
HELD_OUT = 50
LORA_RANK = 32
LEARNING_RATE = 8e-5
MAX_TURN_TOKENS = 512
BATCH_SIZE = 32
GROUP_SIZE = 16
STEPS = 50
INSTRUCTION = "Solve it step by step, then give the final answer as a number after '####'."

tokenizer = AutoTokenizer.from_pretrained(MODEL)
formatter = message_formatter(MODEL, tokenizer, enable_thinking=False)
with open(DATA, encoding="utf-8") as lines:
    problems = [json.loads(line) for line in lines if line.strip()]


def asked(problem):
    return [{"role": "user", "content": f"{problem['question']}\n{INSTRUCTION}"}]


# The most tokens one attempt can use: the longest question, and the longest answer.
episode_tokens = (
    max(len(formatter.build_generation_prompt(asked(problem))) for problem in problems)
    + MAX_TURN_TOKENS
)


def task(problem):
    return AgentSession.Config(
        messages=asked(problem),
        verifier=FinalNumber.Config(answer=str(problem["answer"])),
        max_episode_tokens=episode_tokens,
    )


tasks = [task(problem) for problem in problems]
train_tasks, held_out = tasks[:-HELD_OUT], tasks[-HELD_OUT:]
dataset = TaskDataset(train_tasks, batch_size=BATCH_SIZE, seed=0)
print(f"{len(train_tasks)} problems to train on, {len(held_out)} held out")

training = TrainingConfig(
    model=MODEL,
    kind=Lora(rank=LORA_RANK),
    tokens_per_step=BATCH_SIZE * GROUP_SIZE * episode_tokens,
    max_seq_len=episode_tokens,
    optimizer=AdamW(learning_rate=LEARNING_RATE, betas=(0.9, 0.95), max_grad_norm=1.0),
)
loop_config = train_rl.Config(
    params=SamplingParams(max_tokens=MAX_TURN_TOKENS, temperature=1.0),
    group_size=GROUP_SIZE,
    eval_every=10,
)


def report(step):
    if step.result is None:
        print(f"before training: held out {step.metrics['eval/reward']:.3f}")
        return
    line = (
        f"step {step.step}/{STEPS}: reward {step.metrics['reward']:.3f}, "
        f"{step.metrics['groups_trained']:.0f} of {step.metrics['groups']:.0f} groups trained"
    )
    if "eval/reward" in step.metrics:
        line += f", held out {step.metrics['eval/reward']:.3f}"
    print(line)


async def main():
    config = ServiceClient.Config(
        project=os.environ["LOOPER_PROJECT"],
        display_name="rl",
    )
    async with await config.build() as service:
        print("starting the run and its sampler, which can take a few minutes...")
        training_client = await service.create_training_client(training, timeout_s=20 * 60)
        sampler = await training_client.create_sampling_client(
            max_seq_len=episode_tokens,
            max_concurrent_sequences=min(192, BATCH_SIZE * GROUP_SIZE),
            timeout_s=20 * 60,
        )
        await train_rl.train(
            training_client,
            sampler,
            dataset,
            loop_config,
            formatter=formatter,
            steps=STEPS,
            eval_tasks=held_out,
            on_step=report,
        )

        name = f"rl-{uuid4().hex[:8]}"
        saved = await (await training_client.save_state(name)).result(timeout_s=300)
        await service.wait_checkpoint(saved.id, timeout_s=15 * 60)
        print("saved the trained model as", name)


asyncio.run(main())
```

What each part does:

- **The tasks.** Each problem becomes a **task**, which is the question the model is asked, together with the verifier that scores its answers. `FinalNumber.Config` holds that problem's right answer. [Verifiers](verifiers.md#where-a-verifier-goes-the-task) explains tasks.
- **`episode_tokens`** is the most tokens one attempt can use. The run and its sampler are sized to fit it.
- **`TaskDataset`** splits the training tasks into batches, one per step.
- **`TrainingConfig`** describes the training run on Looper. `tokens_per_step` is big enough to hold every answer in a step. [Choosing settings](../../server/docs/choosing-settings.md) explains each value.
- **`train_rl.Config`** sets up the cookbook's loop. RL must sample at `temperature=1.0`. `eval_every=10` measures the held-out problems every 10 steps, as well as before the first and after the last.
- **The sampler** writes the model's answers with the weights the run is training. It's created from the training client, so the loop can send it new weights after each step. `max_concurrent_sequences` is how many answers it writes at once, and the rest wait their turn. 192 fits every model's sampler for a LoRA run, so you can change `MODEL` without changing it. See [Sampling sizes](../../server/docs/models.md#sampling-sizes).
- **`train_rl.train`** runs every step. Each step, the model answers every problem in the batch `GROUP_SIZE` times, the verifiers score the answers, the run trains on the scores, and the new weights go to the sampler.
- **The save** stores the trained model as a checkpoint, and waits until it's ready to use. The loop doesn't save for you.

To use your own way of scoring, replace `FinalNumber.Config(...)` in `task` with your verifier's config. [Reinforcement learning](reinforcement-learning.md) explains the loop in full, and [The RL loop](../tutorials/302_rl_loop.ipynb) walks through it step by step.

To try the trained model, use `try.py` from [Try the trained model](cookbook-sft.md#try-the-trained-model), with the same instruction after your question.

## Is it working?

The script prints three numbers to watch:

- **The held-out score** (`held out`) is the share of the held-out problems the model gets right. This is the number that matters most, because the model never trains on these problems. Compare the score before training with the score at the end. The loop picks the model's single most likely answer when it measures, so the score doesn't jump around because of random sampling.
- **The reward** is the average score of the answers the model trained on in that step. It should rise over the run, but it jumps around from step to step, because each step has different problems. Look at the trend, not at single steps.
- **`groups_trained`** is how many problems in the step had answers with different scores. Only those teach the model anything. If it stays near 0, see [Choose your problems](#choose-your-problems).

With a small held-out set, the score can change by a few points by chance. [Reading the numbers](evaluation.md#reading-the-numbers) explains how to tell a real improvement from noise.

## Keep your program running

Your program has to keep running for the whole training run. The training happens on Looper, but your program sends it each step, and it keeps the session open.

- **If your program stops,** for example because your laptop goes to sleep or a notebook's kernel restarts, Looper stops hearing from it. After 60 seconds, the session expires, and its training run and sampler are shut down. See [When a session expires](../../server/docs/session.md#when-a-session-expires).
- **A run can't be resumed from a checkpoint yet.** If the run is shut down partway, you start again from the beginning. Checkpoints you saved are kept, and you can still sample them. To save along the way, set `save_every` in `train_rl.Config`.

So for a long run, run your script on a machine that stays on, such as a server or a cloud VM, rather than a laptop.

## Next steps

- [Choosing settings](../../server/docs/choosing-settings.md): when to change the model, the learning rate and the other settings.
- [Verifiers](verifiers.md): score answers your own way.
- [Reinforcement learning](reinforcement-learning.md): the loop in full, including `train_rl_async`, which samples while a step trains.
- [Evaluation](evaluation.md): measure a model on your problems, before or after training.
- If the model uses tools while it answers, such as running shell commands, see [Episode sessions](episode-sessions.md) and [Environments](environments.md).
