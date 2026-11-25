# The cookbook

The cookbook is ready-made code, built on the Looper SDK, that you would otherwise write yourself. It has loops that train a run on your conversations or on your rewards, a message formatter for each model, the pieces of an RL task, and recipes that run a whole job from data to checkpoint. Run them as they are, read them, and copy one when your training needs something else.

The cookbook is the `looper_cookbook` package, in its own repository, [goalyinc/looper-cookbook](https://github.com/goalyinc/looper-cookbook). It isn't on PyPI yet, so you install it from a clone.

## The cookbook or the SDK

You can use as much or as little of the cookbook as you like. You can run a loop as it is and swap in your own pieces, copy a loop and change it, or write your own loop with the SDK alone. [What you build](../../server/docs/what-you-build.md) explains what Looper, the SDK and the cookbook each do, and how to choose.

## What is in it

Each page on the left covers one part:

| Page | What it covers |
|---|---|
| [Message formatting](message-formatting.md) | One formatter for each model family. It writes conversations as prompts and datums, and reads sampled turns back as messages |
| [Supervised learning](supervised-learning.md) | `train_sft`, which trains a run on conversations, one step per batch of datums |
| [Reinforcement learning](reinforcement-learning.md) | `train_rl`, which trains a run on rewards, using tasks, groups of attempts and on-policy sampling. `train_rl_async` samples while a step trains |
| [Episode sessions](episode-sessions.md) | One attempt at one task. `AgentSession` lets a model take turns and call tools until it answers |
| [Environments](environments.md) | Where a model's tool calls run, such as a shell, a code checker, or one of your own |
| [Sandboxes](sandboxes.md) | Where an environment's commands run, either on your machine or in a Docker container |
| [Verifiers](verifiers.md) | What scores a finished attempt, for example by checking a final number, or by running a program against hidden tests |
| [Evaluation](evaluation.md) | Measuring a run on held-out tasks, before and after it trains |
| [Recipes](recipes.md) | Scripts that run whole jobs, from data to checkpoint. [Code RL](../looper_cookbook/recipes/code_rl/README.md) has a page of its own |

## How an RL task is put together

Supervised learning needs two of these pieces. A formatter makes the datums, and the SFT loop trains on them. RL needs more pieces, and you can replace each one without touching the others:

- **The RL loop** takes one optimizer step per batch of tasks.
- **A rollout** is the loop running sessions on the sampler, a group of them for each task.
- **An episode session** is one attempt at one task.
- **An environment** is where the model's tool calls run during an attempt, in a **sandbox**. A task the model answers in one turn needs neither.
- **A verifier** scores the attempt when it ends.
- **An evaluation** attempts each held-out task once, without training.

A task is just data. It holds the messages the model is asked, its verifier's config, and its environment's config if it has one.

```python
from looper_cookbook.environments.bash import BashEnvironment
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.sandboxes.docker import DockerSandbox
from looper_cookbook.verifiers.final_number import FinalNumber

task = AgentSession.Config(
    messages=[{"role": "user", "content": "Work out 2 to the power 100 in the shell. End on '#### ' and the number."}],
    environment=BashEnvironment.Config(sandbox=DockerSandbox.Config(image="python:3.12-slim")),
    verifier=FinalNumber.Config(answer=str(2**100)),
    max_turns=4,
    max_episode_tokens=4096,
)
```

## Run it

Install it from a clone of its repository, with the `recipes` extra, which adds `datasets` for the recipes. You then import the package as `looper_cookbook`:

```bash
git clone https://github.com/goalyinc/looper-cookbook.git
cd looper-cookbook
pip install -e ".[recipes]"
python -m looper_cookbook.recipes.sampling_basic --help
```

You need to set `LOOPER_API_KEY` to call Looper. The examples on these pages that only build prompts, datums and tasks run on your machine, without a key.

## Next steps

- [The tutorials](../tutorials/index.md) take you from a first sample to these loops.
- [The SDK reference](../../sdk/docs/api-reference.md) lists every client, value and error the loops use.
