# The cookbook

The cookbook is code written on top of the Looper SDK that you would otherwise write yourself: loops that train a run on your conversations or on your rewards, a message formatter for each model, the pieces of an RL task, and recipes that run a whole job from data to checkpoint. Run them as they are, read them, and copy one when your training needs something else.

The cookbook is the `looper_cookbook` package, in its own repository, [goalyinc/looper-cookbook](https://github.com/goalyinc/looper-cookbook). It is not on PyPI yet, so for now you install it from a clone.

## The cookbook or the SDK

| | The SDK, `goaly-looper` | The cookbook, `looper_cookbook` |
|---|---|---|
| What it is | The clients and values every call to Looper takes | Loops, data helpers and recipes written over the SDK |
| Use it when | You write your own loop | A loop here does what you need, or is a good start |
| How to get it | `pip install "goaly-looper==0.1.0b4"` | A clone of `goalyinc/looper-cookbook`, until it is on PyPI |

Everything here imports the SDK alone, plus `transformers` to tokenize on your machine. The loops make no call the SDK lacks, so whatever a loop does, your own code can do too.

## What is in it

Each page on the left covers one part:

| Page | What it covers |
|---|---|
| [Message formatting](message-formatting.md) | A formatter per model family. It writes conversations as prompts and datums, and reads sampled turns back as messages |
| [Supervised learning](supervised-learning.md) | `train_sft`: a run trained on conversations, one step per batch of datums |
| [Reinforcement learning](reinforcement-learning.md) | `train_rl`: a run trained on rewards, with tasks, groups of attempts and on-policy sampling. `train_rl_async` samples while a step trains |
| [Episode sessions](episode-sessions.md) | One attempt at one task. `AgentSession` lets a model take turns and call tools until it answers |
| [Environments](environments.md) | Where a model's tool calls run: a shell, a code checker, or one of your own |
| [Sandboxes](sandboxes.md) | Where an environment's commands run: your machine, or a Docker container |
| [Verifiers](verifiers.md) | What scores a finished attempt: a final number, or a program against hidden tests |
| [Evaluation](evaluation.md) | Measuring a run on held-out tasks, before and after it trains |
| [Recipes](recipes.md) | Whole jobs as scripts, from data to checkpoint. [Code RL](../looper_cookbook/recipes/code_rl/README.md) has a page of its own |

## How an RL task is put together

Supervised learning needs two of these pages: a formatter makes datums, and the SFT loop trains them. RL needs more, and each piece is replaced without the others:

- **The RL loop** takes one optimizer step per batch of tasks.
- **A rollout** is the loop running sessions on the sampler: a group of them for each task.
- **An episode session** is one attempt at one task.
- **An environment** is where the model's tool calls run during an attempt, in a **sandbox**. A task the model answers in one turn needs neither.
- **A verifier** scores the attempt when it ends.
- **An evaluation** attempts each held-out task once, and trains nothing.

A task is data: the messages the model is asked, its verifier's config, and its environment's config if it has one.

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

Install it from a clone of its repository, with its recipes' extra, which adds `datasets`. The package imports as `looper_cookbook`:

```bash
git clone https://github.com/goalyinc/looper-cookbook.git
cd looper-cookbook
pip install -e ".[recipes]"
python -m looper_cookbook.recipes.sampling_basic --help
```

Set `LOOPER_API_KEY` to call Looper. The pages' examples that build prompts, datums and tasks run on your machine with no key.

## Next steps

- [The tutorials](../tutorials/index.md): a path from a first sample to these loops.
- [The SDK reference](../../sdk/docs/api-reference.md): every client, value and error the loops use.
