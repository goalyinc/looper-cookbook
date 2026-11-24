# Reinforcement learning

`looper_cookbook.train_rl` trains a run on rewards. Each optimizer step, the run's own sampler attempts a batch of tasks, each attempt is scored, and the run trains on the scores and publishes its new weights back to the sampler.

The pieces, from the outside in:

| Piece | What it is | Page |
|---|---|---|
| The RL loop, `train_rl.train` | One optimizer step per batch of tasks | This page |
| A dataset | The tasks, a batch per step | This page |
| A rollout | The loop running sessions on the sampler: a group of them for each task of a step | [Episode sessions](episode-sessions.md#a-rollout-many-sessions-at-once) |
| An episode session | One attempt at one task | [Episode sessions](episode-sessions.md) |
| An environment | Where the model's tool calls run, for a task that has tools | [Environments](environments.md) |
| A verifier | What scores a finished attempt | [Verifiers](verifiers.md) |
| An evaluation | One attempt at each held-out task, which trains nothing | [Evaluation](evaluation.md) |

## Tasks and the dataset

A task is the config of an episode session: what the model is asked, and the verifier that scores an attempt. A dataset puts tasks in batches, one per step. `TaskDataset` is the cookbook's:

```python
from looper_cookbook.datasets.tasks import TaskDataset
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.verifiers.final_number import FinalNumber

problems = [
    ("What is 6 times 7?", "42"),
    ("What is 9 times 8?", "72"),
    ("What is 12 times 12?", "144"),
    ("What is 7 times 8?", "56"),
    ("What is 15 times 4?", "60"),
]


def task(question, answer):
    return AgentSession.Config(
        messages=[{"role": "user", "content": f"{question} End on '#### ' and the number."}],
        verifier=FinalNumber.Config(answer=answer),
        max_episode_tokens=1024,
    )


tasks = [task(question, answer) for question, answer in problems]
dataset = TaskDataset(tasks[:4], batch_size=2, seed=0)
print(len(dataset), "batches a pass")
for step in range(4):
    print(step, [chosen.messages[0]["content"][:19] for chosen in dataset.batch(step)])
```

```output
2 batches a pass
0 ['What is 12 times 12', 'What is 7 times 8? ']
1 ['What is 9 times 8? ', 'What is 6 times 7? ']
2 ['What is 7 times 8? ', 'What is 12 times 12']
3 ['What is 6 times 7? ', 'What is 9 times 8? ']
```

- **The run's step picks the batch.** Step `k`, counted from 0, trains `dataset.batch(k)`. So the loop called again on the same run, after an error or a rollback, goes on from where the run stands, with nothing for you to keep track of.
- **A dataset never ends.** Past one pass, an index names a batch of the next pass. Each pass is in an order of its own, which the seed and the pass fix. Train several passes by asking the loop for that many steps.
- **A dataset of your own** is anything with `__len__`, the batches of one pass, and `batch(index)`, the tasks of a step. That is the `RLDataset` contract, in `looper_cookbook.datasets.base`. See [below](#a-dataset-of-your-own).

A task here is one turn. A task with tools names an environment too, and the model then takes turns until it answers: see [Episode sessions](episode-sessions.md).

### A dataset of your own

This one keeps its tasks in the order given, easy to hard, where `TaskDataset` shuffles each pass:

```python
import math
from collections.abc import Sequence

from looper_cookbook.datasets.base import RLDataset
from looper_cookbook.episode_sessions.base import EpisodeSession


class InOrder:
    def __init__(self, tasks: Sequence[EpisodeSession.Config], batch_size: int) -> None:
        self.tasks, self.batch_size = tasks, batch_size

    def __len__(self) -> int:
        return math.ceil(len(self.tasks) / self.batch_size)

    def batch(self, index: int) -> Sequence[EpisodeSession.Config]:
        start = index % len(self) * self.batch_size
        return self.tasks[start : start + self.batch_size]


curriculum: RLDataset = InOrder(tasks[:4], batch_size=2)
for step in range(3):
    print(step, [chosen.messages[0]["content"][:19] for chosen in curriculum.batch(step)])
```

```output
0 ['What is 6 times 7? ', 'What is 9 times 8? ']
1 ['What is 12 times 12', 'What is 7 times 8? ']
2 ['What is 6 times 7? ', 'What is 9 times 8? ']
```

## The loop

`train` takes the run, the run's sampler, the dataset and the loop's config:

```python
from transformers import AutoTokenizer

from looper_cookbook import train_rl
from looper_cookbook.message_formatters.models import message_formatter
from looper_sdk import SamplingParams

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
formatter = message_formatter("Qwen/Qwen3.5-9B", tokenizer, enable_thinking=False)
sampling_client = await training_client.create_sampling_client(
    max_seq_len=1024, max_concurrent_sequences=16
)
config = train_rl.Config(params=SamplingParams(max_tokens=512), group_size=8)
reports = await train_rl.train(
    training_client, sampling_client, dataset, config, formatter=formatter, steps=20
)
for report in reports:
    print(report.step, report.metrics["reward"], report.version.step)
```

It first publishes the run's weights to the sampler, so the first batch samples them. Then, for each step:

1. It attempts every task of the step's batch `group_size` times, all at once, on the sampler. Each attempt is an episode session, scored by its verifier.
2. It sends each task's episodes to the trainer as one group, under `config.loss`.
3. It takes the optimizer step and publishes the new weights, and saves when one is due.
4. It waits for the publish, then evaluates when one is due, and reports the step.

## Groups

A group is the attempts at one task. `Grpo` compares each attempt's reward with the rest of its group: an attempt that scored above its group's mean is made more likely, and one below it less.

- **Only differences within a group train.** A group whose attempts all score the same trains nothing. A model that already solves a task every time learns nothing more from it, and neither does one that never solves it.
- **The trainer forms the advantages.** A datum carries its reward and its group, never an advantage.
- **A group goes whole.** A step too large for one call goes over several, and a group is never split between them.

## Every turn is on-policy

The loop publishes before its first sample and waits for each publish before it samples again. So every turn of a step is sampled from the weights of the step before it. Each report's `version` names the oldest weights that sampled one of its turns.

That is also why `Config` refuses a temperature or a `top_p` other than 1. The loss reads the sampler's log probabilities as the policy's own, which they are only when nothing reshapes the distribution.

## Sampling while a step trains

`train_rl` takes turns: the sampler waits while the trainer works, and the trainer waits for a whole batch. `train_rl_async` overlaps the two. It takes the same arguments, and its config adds one setting, `max_policy_lag`:

```python
from looper_cookbook import train_rl_async

config = train_rl_async.Config(
    params=SamplingParams(max_tokens=512), group_size=8, max_policy_lag=1
)
reports = await train_rl_async.train(
    training_client, sampling_client, dataset, config, formatter=formatter, steps=20
)
for report in reports:
    print(report.step, report.metrics["reward"], report.metrics["policy_lag"])
```

- **It samples ahead.** The loop keeps the groups of the next `max_policy_lag + 1` steps in flight or waiting to train. It reads tasks from the dataset's batches in order.
- **A step trains what finished first:** as many groups as its batch holds, oldest weights first. No step waits for one batch's slowest episode.
- **A group may train on weights newer than those that sampled it,** by at most `max_policy_lag` steps. The loss corrects for that gap with its clipped ratio. A group that finishes older is dropped, and the next task starts in its place.
- **`max_policy_lag=0` is `train_rl`'s loop.**

What changes against `train_rl`:

| | `train_rl` | `train_rl_async` |
|---|---|---|
| What a step trains | Its own batch, `dataset.batch(step)` | The first groups to finish |
| The weights that sampled a turn | The step before | Up to `max_policy_lag` steps older |
| A second call on the same run | Trains the batch of the run's step | Reads on from that batch. A task may train twice, or wait for the next pass |
| A slow episode | Its step waits for it | Its group is dropped once it is older than the lag |
| Sessions running at once | One batch | Up to `max_policy_lag + 1` batches, each with its sandbox |

A step's report holds three more metrics:

| Metric | What it is |
|---|---|
| `policy_lag`, `policy_lag_max` | How many steps the step's groups trail the weights that train them: the mean over the groups, and the most. A group is as old as its first turn |
| `groups_stale` | The finished groups dropped since the step before, for being older than `max_policy_lag` |

- **Size the lag from your longest episode,** counted in steps. An episode that runs longer than `max_policy_lag + 1` steps take never trains, so a lag that is too small trains short episodes more often than long ones. Watch `groups_stale`.
- **Size the sampler for the batches in flight:** `max_concurrent_sequences` of `max_policy_lag + 1` batches, times `group_size`. Requests past it wait in the sampler's queue.

## The config

`train_rl.Config` sets how the loop samples, trains, evaluates and saves:

| Field | Default | What it sets |
|---|---|---|
| `params` | none | How each turn is sampled. `max_tokens` is one turn's token budget. An episode's is its task's `max_episode_tokens`: see [two token budgets](episode-sessions.md#two-token-budgets). Temperature and `top_p` must be 1 |
| `group_size` | 8 | The attempts at each task. At least 2 |
| `loss` | `Grpo()` | What the trainer trains the groups with: `Grpo()` or `Dapo()` |
| `max_call_tokens` | 1,000,000 | The most tokens one `forward_backward` call carries. A larger step goes over several calls |
| `eval_params` | `None` | How an evaluation's turns are sampled. `None` samples greedily: `params` at temperature 0 |
| `eval_every` | `None` | Steps between evaluations of the loop's `eval_tasks`. With such tasks the loop also evaluates before its first step and after its last |
| `save_every` | `None` | Steps between saves of the run's state, each named `<run id>-step-<step>`. `None` saves nothing |

Each step trains at the run's own rate, set in its `TrainingConfig.optimizer`.

## What a step reports

`train` returns a `StepReport` per step, and calls `on_step` with each as it is made:

| Field | What it holds |
|---|---|
| `step` | The run's step the report is of |
| `result` | What the run reported of the step: its loss, its learning rate, its gradient norm |
| `version` | The oldest weights that sampled one of the step's turns: the step before |
| `metrics` | The step's numbers, by name |
| `checkpoint` | The state saved at the step, or `None` |

Its `metrics` are:

| Metric | What it is |
|---|---|
| `reward` | The mean reward of the step's episodes |
| `episodes`, `groups` | How many of each the step had |
| `groups_trained` | The groups whose rewards differ, which are the ones that train |
| `truncated` | The share of episodes whose last turn was cut off, by a turn's token budget or by the episode's |
| `stop/<reason>` | The share of episodes each [stop reason](episode-sessions.md#why-an-episode-ends) ended: `stop/answer`, `stop/max_turns`, `stop/max_turn_tokens` and `stop/max_episode_tokens`. The last two say which token budget ended them |
| `turns`, `sampled_tokens` | Means per episode |
| Each environment's and verifier's metric | Its mean over the episodes that report it, such as `answered` |
| The evaluation's metrics | At the steps that ran one, named `eval/…`, such as `eval/reward` |

With `eval_tasks`, the first report is the baseline: it is of the step the call started at, measured before anything trained. Its `result` and `version` are `None`.

## Sizing the run and its sampler

The loop checks no size: the service refuses what does not fit. Size three things when you create them:

- **The run's `max_seq_len`** holds a task's `max_episode_tokens`, the longest episode.
- **The run's `tokens_per_step`** holds a step: the batch size, times `group_size`, times the longest episode.
- **The sampler's `max_concurrent_sequences`** is how many turns it runs at once. A step sends the batch size times `group_size` requests at once, and the rest wait in the sampler's queue.

## What the loop leaves to you

- **The run,** created with its `TrainingConfig`: its model, a LoRA or every parameter, its sizes and its optimizer.
- **The run's sampler,** from [`TrainingClient.create_sampling_client`](../../sdk/docs/api-reference.md#trainingclientcreate_sampling_client).
- **The tasks,** held-out ones too.
- **The last save,** two lines after `train` returns.
- **What the run writes.** The loop writes no files. Record what you need from each report in `on_step`.
- **Recovery.** An error raises as its operation's result is awaited. Call `train` again on the same run to go on from the step it stands at.

`looper_cookbook/train_rl.py` is short enough to read, and meant to be copied and changed when your training needs something else.

## Next steps

- [The RL loop](../tutorials/302_rl_loop.ipynb) trains math problems this way, start to end.
- [Your first RL run](../tutorials/104_first_rl.ipynb) writes one step out with the SDK alone.
- [Evaluation](evaluation.md) measures the run on tasks it never trains.
- [Recipes](recipes.md): `rl_basic` is this loop as a script, on GSM8K.
