# Reinforcement learning

The cookbook's RL loops run reinforcement learning for you. You give them a list of tasks and a way to score answers, and they handle the rest. They generate answers, score them, train on the scores, and send the new weights back to the sampler. [How training works](../../server/docs/how-training-works.md) explains RL itself, and [What you build](../../server/docs/what-you-build.md) explains how to use parts of the cookbook and choose which parts to build yourself.

There are two loops:

- **`train_rl`** is synchronous, which means it finishes each round before starting the next. Start here.
- **`train_rl_async`** is asynchronous, which means it keeps generating answers while it trains, so training finishes sooner. It works the same way, with one extra setting.

This page covers:

- [Words used on this page](#words-used-on-this-page): task, attempt, group and the other terms the cookbook uses.
- [Tasks and the dataset](#tasks-and-the-dataset): write the tasks to train on, and group them into batches.
- [The loop](#the-loop): run `train_rl`, and what it does at each step.
- [Groups](#groups): how a task's attempts are compared, and when a group teaches nothing.
- [Every answer comes from the newest weights](#every-answer-comes-from-the-newest-weights): why `train_rl` waits for each publish, and why temperature must be 1.
- [Sampling while a step trains](#sampling-while-a-step-trains): `train_rl_async`, and how to choose `max_policy_lag`.
- [The config](#the-config): every setting of the loop.
- [What a step reports](#what-a-step-reports): the numbers each step gives you.
- [Sizing the run and its sampler](#sizing-the-run-and-its-sampler): how big to make them.
- [What the loop leaves to you](#what-the-loop-leaves-to-you): what you still do yourself.

## Words used on this page

The cookbook uses a few words for the parts of RL. Using the example of a model learning arithmetic:

| Word | What it means | In the example |
|---|---|---|
| **Task** | One thing to ask the model, plus how to score its answer | "What is 6 times 7?", scored by checking that the answer is 42 |
| **Attempt**, or **episode** | One try at a task, from the question to the model's final answer | One of the model's answers to "What is 6 times 7?" |
| **Group** | All the attempts at the same task in one step | The 8 answers the model wrote to "What is 6 times 7?" |
| **Turn** | One reply from the model within an attempt. Simple tasks have one turn; tasks with tools can have several | The model's reply "6 times 7 is 42" |
| **Batch** | The tasks trained on in one step | "What is 6 times 7?" and "What is 9 times 8?" |
| **Pass** | One trip through every task in the dataset | All the questions, each asked once |
| **Verifier** | The code that scores an attempt, giving its reward | Checks the final number, and gives 1 for 42 and 0 otherwise |

## Tasks and the dataset

A task is what the model is asked, and the verifier that scores its answers. You write one `AgentSession.Config` per task. A dataset then puts the tasks into batches, one batch per training step. The cookbook's dataset is `TaskDataset`.

The code below makes five arithmetic tasks. Each asks the model to end its answer with `####` and a number, and uses the cookbook's `FinalNumber` verifier to check that number. Then it puts four of the tasks into a dataset with two tasks per batch, and prints which tasks each of the first four steps gets:

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

Four tasks with two per batch make two batches per pass. Steps 0 and 1 are the first pass, and steps 2 and 3 are the second, in a new order.

- **The step number picks the batch.** Step 0 trains `dataset.batch(0)`, step 1 trains `dataset.batch(1)`, and so on. So if training stops, after an error or a rollback, you can call the loop again on the same run and it picks up with the right batch. You don't need to track where you were.
- **A dataset never runs out.** After one pass, it starts the next, in a new order. The `seed` fixes each pass's order, so the same seed gives the same order every time. To train for several passes, ask the loop for that many steps.
- **These tasks have one turn.** The model answers in a single reply. For tasks where the model uses tools, such as running shell commands, you also give the task an environment, and the model takes turns until it answers. See [Episode sessions](episode-sessions.md).

### A dataset of your own

`TaskDataset` shuffles the tasks in each pass. If you need something different, you can write your own dataset. It needs two methods:

- `__len__`: how many batches one pass has.
- `batch(index)`: the tasks for step `index`.

That's the `RLDataset` contract, from `looper_cookbook.datasets.base`. This one keeps the tasks in the order you give them, for example from easiest to hardest, instead of shuffling:

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

`train_rl.train` runs the training. You give it four things:

- **The training run,** as `training_client`. See [Create a training client](../../server/docs/training.md#create-a-training-client).
- **The run's sampler,** as `sampling_client`. Create it from the training client, so it uses the run's own weights.
- **The dataset** of tasks.
- **The config,** which sets how answers are written and how many attempts each task gets.

It also needs the model's message formatter, which writes the prompts in the model's chat format. See [Message formatting](message-formatting.md).

The code below trains for 20 steps, with 8 attempts per task, and prints the average reward after each step:

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

Before the first step, the loop publishes the run's weights to the sampler, so the sampler has a model to answer with. Then, for each step:

1. **Attempt.** The sampler attempts every task in the step's batch `group_size` times, all at the same time. Each attempt's verifier scores it.
2. **Train.** The loop sends each task's attempts to the trainer as one group, with the config's loss (`Grpo` by default).
3. **Update.** It takes the optimizer step and publishes the new weights to the sampler. If a save is due, it saves.
4. **Report.** It waits for the publish to finish, runs an evaluation if one is due, and reports the step.

## Groups

A group is all the attempts at one task. `Grpo` compares each attempt's reward with the average of its group. Attempts that scored above the average become more likely, and attempts below it become less likely.

- **A group only teaches when its rewards differ.** If every attempt in a group gets the same score, there's nothing to compare, and the group teaches nothing. So a task the model always gets right teaches it nothing more, and neither does a task it always gets wrong. The most useful tasks are ones the model sometimes gets right.
- **You never compute the comparison yourself.** Each attempt carries its reward and its group, and the trainer works out how much better or worse than average it was (its advantage).
- **A group is never split up.** If a step is too big for one `forward_backward` call, the loop splits it across several calls, but always keeps each group together in one call.

## Every answer comes from the newest weights

`train_rl` publishes before its first sample, and waits for each publish to finish before it samples again. So every answer in a step comes from the weights the previous step made. Each report's `version` says which weights wrote the step's answers, and if they differ, the oldest of them.

This is also why the loop only allows a `temperature` and `top_p` of 1. The loss uses each answer's `logprobs` as the model's true probability of writing it. Any other temperature or `top_p` reshapes the probabilities while sampling, so the `logprobs` would no longer match the model, and the loss would be wrong.

## Sampling while a step trains

`train_rl` takes turns. The sampler waits while the trainer trains, and the trainer waits until a whole batch of attempts is done. `train_rl_async` lets both work at the same time, so training finishes sooner. You call it the same way, and its config has one extra setting, `max_policy_lag`, which sets how many steps old an attempt's weights may be when it's trained on.

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

How it works:

- **It starts the next batches early.** While a step trains, the sampler is already attempting the tasks of the next `max_policy_lag + 1` steps. It takes tasks from the dataset's batches in order.
- **Each step trains the groups that finish first.** It takes as many groups as a batch holds, oldest first. No step waits for one slow attempt.
- **Groups can be slightly out of date.** A group may be trained on after the weights have moved on, by up to `max_policy_lag` steps. The loss's clipping accounts for that difference. A group that falls further behind is dropped, and a new task is started in its place.
- **`max_policy_lag=0` behaves like `train_rl`.**

How it differs from `train_rl`:

| | `train_rl` | `train_rl_async` |
|---|---|---|
| What a step trains on | Its own batch, `dataset.batch(step)` | Whichever groups finish first |
| Which weights wrote the answers | The previous step's | Up to `max_policy_lag` steps older |
| If you call the loop again on the same run | Picks up with the batch for the run's current step | Reads on from that batch. A task may be trained twice, or wait until the next pass |
| A slow attempt | Its step waits for it | Its group is dropped once it's older than the lag |
| Attempts running at once | One batch | Up to `max_policy_lag + 1` batches, each with its own sandbox |

Each report has three extra metrics:

| Metric | What it tells you |
|---|---|
| `policy_lag`, `policy_lag_max` | How many steps behind the current weights the step's groups were, on average and at most. A group's age counts from its first turn |
| `groups_stale` | How many finished groups were dropped since the previous step, for being older than `max_policy_lag` |

Choosing the settings:

- **Choose `max_policy_lag` from your longest attempts.** An attempt that takes longer than `max_policy_lag + 1` steps to finish is always dropped. If the lag is too small, long attempts never get trained on, and the model learns mostly from short ones. If `groups_stale` is often above 0, raise the lag.
- **Make the sampler big enough** to run `max_policy_lag + 1` batches at once. Set `max_concurrent_sequences` to the batch size, times `group_size`, times `max_policy_lag + 1`. Requests beyond that wait in the sampler's queue.

## The config

`train_rl.Config` sets how the loop samples, trains, evaluates and saves:

| Setting | Default | What it does |
|---|---|---|
| `params` | Required | How each turn is sampled, as `SamplingParams`. `max_tokens` is the longest one turn can be. An attempt's total limit is its task's `max_episode_tokens` (see [two token budgets](episode-sessions.md#two-token-budgets)). `temperature` and `top_p` must be 1 |
| `group_size` | 8 | How many attempts each task gets. At least 2 |
| `loss` | `Grpo()` | The loss to train with, either `Grpo()` or `Dapo()` |
| `max_call_tokens` | 1,000,000 | The most tokens one `forward_backward` call sends. A bigger step is split across several calls |
| `eval_params` | `None` | How evaluation answers are sampled. `None` uses `params` with temperature 0, which always picks the most likely token |
| `eval_every` | `None` | How many steps between evaluations of `eval_tasks`. If you give `eval_tasks`, the loop also evaluates before the first step and after the last |
| `save_every` | `None` | How many steps between saves of the run's state. Each save is named `<run id>-step-<step>`. `None` saves nothing |

There's no learning-rate setting. Each step uses the run's own rate, from its `TrainingConfig.optimizer`.

## What a step reports

`train` returns one `StepReport` per step. It also passes each report to your `on_step` function, if you give one, as soon as the step finishes:

| Field | What it holds |
|---|---|
| `step` | Which step of the run this report is for |
| `result` | What the trainer reported for the step, including its loss, learning rate and gradient size |
| `version` | The oldest weights that wrote one of the step's answers. In `train_rl`, that's the previous step's |
| `metrics` | The step's numbers, by name (see below) |
| `checkpoint` | The state saved at this step, or `None` |

The `metrics` are:

| Metric | What it tells you |
|---|---|
| `reward` | The average reward of the step's attempts. This is the main number to watch |
| `episodes`, `groups` | How many attempts and groups the step had |
| `groups_trained` | How many groups had different rewards, and so actually taught the model something |
| `truncated` | The share of attempts that were cut off because they hit a length limit |
| `stop/<reason>` | The share of attempts that ended each [way](episode-sessions.md#why-an-episode-ends), as `stop/answer` (the model answered), `stop/max_turns`, `stop/max_turn_tokens` and `stop/max_episode_tokens` (the last two mean a length limit ended it) |
| `turns`, `sampled_tokens` | The average number of turns and of tokens written per attempt |
| Each environment's and verifier's own metrics | Their average over the attempts that report them, such as `answered` |
| The evaluation's metrics | At steps that ran an evaluation, named `eval/…`, such as `eval/reward` |

If you give `eval_tasks`, the first report is a **baseline**, which is the model's evaluation before any training, at the step the call started from. Its `result` and `version` are `None`, because nothing has trained yet.

## Sizing the run and its sampler

The loop doesn't check sizes for you. If something doesn't fit, Looper refuses it with an error. Set three sizes when you create the run and the sampler:

- **The run's `max_seq_len`** must fit the longest attempt, which is your tasks' `max_episode_tokens`.
- **The run's `tokens_per_step`** must fit a whole step, which is the batch size, times `group_size`, times the longest attempt.
- **The sampler's `max_concurrent_sequences`** is how many turns it writes at once. Each step sends the batch size times `group_size` requests at once. Any beyond the sampler's limit wait in its queue.

## What the loop leaves to you

- **The run,** created with its `TrainingConfig`, which sets the model, LoRA or every parameter, its sizes and its optimizer.
- **The run's sampler,** from [`TrainingClient.create_sampling_client`](../../sdk/docs/api-reference.md#trainingclientcreate_sampling_client).
- **The tasks,** including held-out tasks for evaluation.
- **The final save,** after `train` returns. It takes two lines.
- **Recording results.** The loop writes no files. Save what you need from each report in `on_step`.
- **Recovering from errors.** An error is raised when the loop waits on the failed call's result. To carry on, call `train` again on the same run, and it picks up from the run's current step.

`looper_cookbook/train_rl.py` is short enough to read. It's meant to be copied and changed when your training needs something the loop doesn't do.

## Next steps

- [The RL loop](../tutorials/302_rl_loop.ipynb) trains on math problems this way, from start to finish.
- [Your first RL run](../tutorials/104_first_rl.ipynb) writes one step out by hand with the SDK.
- [Evaluation](evaluation.md) measures the model on tasks it never trains on.
- [Recipes](recipes.md) has `rl_basic`, which is this loop as a ready-to-run script, on the GSM8K math dataset.
