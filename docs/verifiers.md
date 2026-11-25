# Verifiers

A verifier scores the model's answers. In RL, the model writes several answers to each question, and the verifier gives each one a **reward**: a number that says how good it was. The model then learns to write more answers like the high-scoring ones. In the arithmetic example from [How training works](../../server/docs/how-training-works.md), the verifier is the part that checks whether an answer says 84, and gives it 1 if so and 0 if not.

This page shows where a verifier goes and what one looks like, then the two verifiers the cookbook comes with, and then how to write your own.

This page covers:

- [Where a verifier goes: the task](#where-a-verifier-goes-the-task): what a task is, how each question gets its own answer, and how tasks, attempts and groups fit together.
- [What a verifier looks like](#what-a-verifier-looks-like): the parts every verifier has, and what each one does.
- [The cookbook's verifiers](#the-cookbooks-verifiers): the two that come ready to use.
  - [A number: `FinalNumber`](#a-number-finalnumber): score answers that end in a number.
  - [A program: `CodeTestsVerifier`](#a-program-codetestsverifier): score code by running it against tests.
- [A verifier of your own](#a-verifier-of-your-own): write your own scoring.
- [Cut-off answers](#cut-off-answers): answers that hit a length limit, which score 0 without being checked.

## Where a verifier goes: the task

A **task** is one question for the model, together with how to score its answers for that particular question. A dataset of 1,000 questions is 1,000 tasks.

Each task has its own verifier settings, holding that question's right answer. Here's the 12 × 7 question as a task:

```python
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.verifiers.final_number import FinalNumber

task = AgentSession.Config(
    messages=[{"role": "user", "content": "What is 12 times 7? End on '#### ' and the number."}],
    verifier=FinalNumber.Config(answer="84"),
    max_episode_tokens=2048,
)
```

The task keeps the question and the answer in separate places:

- **`messages`:** the question. This is your data, and it's all the model sees.
- **`verifier`:** how to score the model's reply, including the right answer, 84. The model never sees this part. Here it's `FinalNumber`, the cookbook's verifier for answers that are a single number.
- **`max_episode_tokens`:** the most tokens the whole attempt can use. An answer that runs longer is [cut off](#cut-off-answers).

## What a verifier looks like

Every verifier extends the base class `Verifier`, from `looper_cookbook.verifiers.base`, and has the same parts. Here they are in `FinalNumber`, the verifier the task above uses. This is a simplified version of `looper_cookbook/verifiers/final_number.py`:

```python
class FinalNumber(Verifier):
    # The settings: the right answer, such as "84".
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Config:
        answer: str

        # Creates the verifier from the settings.
        async def build(self) -> "FinalNumber":
            return FinalNumber(Decimal(self.answer))

    def __init__(self, answer):
        self._answer = answer

    # The scoring: 1 if the number after the reply's last "####" is the right answer, 0 if not.
    async def verify(self, messages, environment) -> Score:
        given = final_number(last_reply(messages))
        return Score(
            reward=float(given == self._answer),
            metrics={"answered": float(given is not None)},
        )
```

`final_number` is a small helper in the same file: it finds the number after the last `####` in some text, or gives `None` if there isn't one.

### `Config`: the settings

`Config` holds the verifier's settings. For `FinalNumber`, that's one setting, `answer`. It's a class inside the verifier, so you write `FinalNumber.Config(answer="84")`, as the task above does.

Each question gets its own `Config`, with its own answer. A task holds the `Config`, not the verifier itself. A `Config` is just data, so you can create thousands of tasks cheaply.

`FinalNumber` makes its `Config` a frozen dataclass. That gives it a constructor for free, and keeps it from being changed after a task is created.

### `build`: creating the verifier

`build()` is the one method `Config` must have. It creates the verifier from the settings. `FinalNumber`'s turns the answer from text into a number, so it can be compared with the model's.

The cookbook calls `build()` at the start of each attempt, so every attempt gets its own verifier. It's `async` because some verifiers need to set something up first.

### `verify`: the scoring

`verify` is called once, when the attempt has finished. It gets two things:

- **`messages`:** the whole conversation, in order: the task's messages, each reply the model wrote, and each tool's response. In the example, that's the question and the model's one reply, such as `"12 times 7 is 84.\n#### 84"`.
- **`environment`:** the environment the model worked in, still open, for tasks where the model uses tools. The verifier can use it to check what the model did, for example by reading a file the model wrote. The example has no tools, so it's `None`, and `FinalNumber` ignores it.

`FinalNumber.verify` gets the model's reply, reads the number after its last `####`, and compares it with the right answer.

### `Score`: what `verify` returns

```text
Score(reward: float, metrics: Mapping[str, float] = {})
```

- **`reward`:** the attempt's score. Higher is better. It can be any finite number, but RL only learns from how the rewards in a group differ, so 1 for right and 0 for wrong is usually enough. `FinalNumber` gives 1 if the number is 84, and 0 if not. A reward that isn't finite, such as `nan`, raises a `ValueError`.
- **`metrics`:** optional numbers for your logs. They appear in the training loop's reports, and don't affect training. `FinalNumber` reports `answered`: 1 if the reply gave a number at all, even a wrong one, and 0 if not.

### `last_reply`: the model's answer

`last_reply(messages)`, also from `looper_cookbook.verifiers.base`, returns the text of the model's last reply, or an empty string if there isn't one. In the example, that's `"12 times 7 is 84.\n#### 84"`. Most verifiers only need to check the final answer, so this is usually the first thing `verify` calls.

### Rules every verifier follows

- **Check the model's work, but don't change it.** This only matters for tasks where the model uses tools. There, `verify` gets the model's `environment`, such as the sandbox where it ran commands and wrote files. The verifier can look around in it, for example by reading a file or running the model's program. But it shouldn't edit or delete anything the model left. The cookbook closes the environment when the attempt is over, so the verifier shouldn't close it itself, or hold on to it after `verify` returns.
- **The right answer stays in the verifier.** It belongs in the verifier's `Config`, like the 84 in the example, never in the prompt, a tool or anything else the model can see.
- **If it can't check, it raises an error.** For example, if its sandbox won't start, or a judge model doesn't respond. It never returns a low score for an attempt it couldn't check, or the model would be trained on a score that doesn't mean anything.
- **It starts nothing until `verify` is called.** The cookbook creates the verifier at the start of an attempt, and [cut-off attempts](#cut-off-answers) never call `verify`. So anything expensive, such as a sandbox, is started inside `verify`, not in `build` or `__init__`.

## The cookbook's verifiers

The cookbook comes with two verifiers, in `looper_cookbook.verifiers`:

| Verifier | Gives a reward of 1 when | Needs |
|---|---|---|
| `final_number.FinalNumber` | The number after the reply's last `####` is the right answer | Nothing |
| `code_tests.CodeTestsVerifier` | The code in the reply passes every test | A [sandbox](sandboxes.md) to run the code in |

### A number: `FinalNumber`

`FinalNumber` scores questions whose answer is a single number, such as math problems. Ask the model to end its reply with `####` and the number. That's the format of the GSM8K math dataset. The verifier reads the number after the last `####` in the reply.

The code below checks three replies to "What is 30 times 40?", where the right answer is 1,200:

```python
verifier = await FinalNumber.Config(answer="1,200").build()
question = {"role": "user", "content": "What is 30 times 40? End on '#### ' and the number."}
for reply in ("30 * 40 = 1200.\n#### 1200", "#### 120\nNo, that is wrong.\n#### 1200.0", "It is 1200."):
    score = await verifier.verify([question, {"role": "assistant", "content": reply}], None)
    print(score.reward, dict(score.metrics))
```

```output
1.0 {'answered': 1.0}
1.0 {'answered': 1.0}
0.0 {'answered': 0.0}
```

The first reply is right. The second first says 120, then corrects itself to 1200.0, and still scores 1. The third has the right number but no `####`, so it scores 0.

- **Numbers are compared as numbers,** so `1,200`, `1200` and `1200.0` all count as the same answer. A `$` before the number is ignored.
- **Only the last `####` counts,** so the model can correct itself, as the second reply does.
- **`answered` tells you whether the reply gave a number at all.** This separates wrong answers from replies in the wrong format. If `answered` is often 0, the model isn't following the format, rather than getting the math wrong.
- **Only the final reply is checked,** never the model's reasoning. A thinking model may try out several answers while it reasons, and those don't count.

### A program: `CodeTestsVerifier`

`CodeTestsVerifier` scores programming tasks. It takes the last code block in the reply as the model's program, runs it against the task's tests, and gives a reward of 1 only if it passes all of them.

The code below sets up two tests for "print the sum of the numbers on standard input," then checks three replies:

```python
from looper_cookbook.environments.coding import CodeTest
from looper_cookbook.sandboxes.docker import DockerSandbox
from looper_cookbook.verifiers.code_tests import CodeTestsVerifier

verifier = await CodeTestsVerifier.Config(
    sandbox=DockerSandbox.Config(image="python:3.12-slim", memory_mb=512),
    tests=[CodeTest(input="1 2\n", output="3\n"), CodeTest(input="10 20 30\n", output="60\n")],
    timeout_s=6,
).build()
ask = {"role": "user", "content": "Print the sum of the numbers on standard input."}
replies = (
    "```python\nprint(sum(map(int, input().split())))\n```",
    "```python\na, b = map(int, input().split())\nprint(a + b)\n```",
    "I would add the numbers.",
)
for reply in replies:
    score = await verifier.verify([ask, {"role": "assistant", "content": reply}], None)
    print(score.reward, dict(score.metrics))
```

```output
1.0 {'answered': 1.0}
0.0 {'answered': 1.0}
0.0 {'answered': 0.0}
```

The first program passes both tests. The second only handles two numbers, so it fails the second test and scores 0. The third reply has no code, so it scores 0 with `answered` 0.

- **The tests run in a fresh sandbox.** The verifier starts its own sandbox for the check, and closes it afterwards. The model never worked in it, so nothing the model did can affect the tests or their results.
- **The expected outputs never leave your program.** The sandbox only gets the program and each test's input. It sends back what the program printed, and the verifier compares that with the expected output on your side.
- **These are the hidden tests.** For tasks where the model can test its code while it works, a [`CodingEnvironment`](environments.md#check-a-program-codingenvironment) gives it example tests through a tool. The tests that decide the reward belong in the verifier, where the model can't see them.
- **A reply with no code block scores 0,** with `answered` 0, and no sandbox is started.
- **`prelude` is code added before the model's program,** such as imports. If the task also has a `CodingEnvironment`, give both the same `prelude`.

Tests can check a whole program's output, or a single function's return value. [Environments](environments.md#check-a-program-codingenvironment) shows both kinds.

## A verifier of your own

To score answers your own way, write a class that extends `Verifier`, with the parts described in [What a verifier looks like](#what-a-verifier-looks-like):

1. A `Config` class holding your settings, such as the right answer, with a `build` method that creates your verifier.
2. An `__init__` that keeps the config, so `verify` can use it.
3. A `verify` method that does the scoring and returns a `Score`.

Make `Config` a frozen dataclass, as `FinalNumber` does.

### A simple example: checking the reply

This verifier gives a reward of 1 if the model's reply names the right city. It only reads the reply, so it ignores `environment`:

```python
import dataclasses

from looper_cookbook.verifiers.base import Score, Verifier, last_reply


class NamesCity(Verifier):
    # The settings: here, just the right answer.
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Config:
        city: str

        # Creates the verifier from these settings.
        async def build(self) -> "NamesCity":
            return NamesCity(self)

    def __init__(self, config):
        self._config = config

    # The scoring: 1.0 if the reply names the city, 0.0 if not.
    async def verify(self, messages, environment) -> Score:
        answer = last_reply(messages).lower()
        return Score(reward=float(self._config.city.lower() in answer))
```

`last_reply(messages)` gives the text of the model's last message. Both of the cookbook's verifiers use it too.

### Using your verifier with the cookbook

Your verifier plugs into the cookbook's tasks and training loops in the same place as `FinalNumber`: the task's `verifier`. Everything else is ordinary cookbook code.

First, build one task per question, each with a `NamesCity.Config` holding that question's city:

```python
from looper_cookbook.datasets.tasks import TaskDataset
from looper_cookbook.episode_sessions.agent_session import AgentSession

capitals = [("France", "Paris"), ("Japan", "Tokyo"), ("Kenya", "Nairobi"), ("Peru", "Lima")]
tasks = [
    AgentSession.Config(
        messages=[{"role": "user", "content": f"What is the capital of {country}?"}],
        verifier=NamesCity.Config(city=city),
        max_episode_tokens=1024,
    )
    for country, city in capitals
]
dataset = TaskDataset(tasks, batch_size=2, seed=0)
```

Then train with one of the cookbook's loops, exactly as you would with a built-in verifier. The loop builds a `NamesCity` from each task's `Config` at the start of every attempt, and calls its `verify` when the attempt ends. This assumes you've already created `training_client`, `sampling_client` and `formatter`, as [Reinforcement learning](reinforcement-learning.md#the-loop) shows:

```python
from looper_cookbook import train_rl_async
from looper_sdk import SamplingParams

config = train_rl_async.Config(
    params=SamplingParams(max_tokens=512), group_size=8, max_policy_lag=1
)
reports = await train_rl_async.train(
    training_client, sampling_client, dataset, config, formatter=formatter, steps=20
)
```

Only the verifier is your code. The tasks, the dataset and the loop are the cookbook's. [What you build](../../server/docs/what-you-build.md#example-your-own-scoring-with-the-cookbooks-loop) walks through this same example step by step.

### Checking the model's work in its environment

For tasks with tools, a verifier can also look at what the model did. This one checks that the model wrote a file with the right contents, by reading the file in the sandbox where the model worked:

```python
import dataclasses

from looper_cookbook.environments.bash import BashEnvironment
from looper_cookbook.sandboxes.local import LocalSandbox
from looper_cookbook.verifiers.base import Score, Verifier


class FileHolds(Verifier):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Config:
        path: str
        text: str

        async def build(self) -> "FileHolds":
            return FileHolds(self)

    def __init__(self, config):
        self._config = config

    async def verify(self, messages, environment) -> Score:
        result = await environment.sandbox.run(f"cat {self._config.path}", timeout_s=10)
        return Score(reward=float(result.stdout.strip() == self._config.text))


verifier = await FileHolds.Config(path="answer.txt", text="42").build()
async with await BashEnvironment.Config(sandbox=LocalSandbox.Config()).build() as environment:
    print((await verifier.verify([], environment)).reward)
    await environment.sandbox.write_file("answer.txt", "42\n")
    print((await verifier.verify([], environment)).reward)
```

```output
0.0
1.0
```

The first check scores 0, because the file doesn't exist yet. The test code then writes the file itself, standing in for the model, and the second check scores 1.

Your verifier must follow the same [rules](#rules-every-verifier-follows) as the cookbook's.

## Cut-off answers

If an answer was cut off because it hit a length limit, the verifier never sees it, and it scores 0. The limit can be on one turn or on the whole attempt: see [two token budgets](episode-sessions.md#two-token-budgets).

This is because a cut-off answer might contain the right number by accident. Skipping it also means no sandbox is started for it.

## Next steps

- [Episode sessions](episode-sessions.md): how an attempt runs, and when its verifier is called.
- [Evaluation](evaluation.md): measure a model on held-out tasks with the same verifiers.
- [Reinforcement learning](reinforcement-learning.md): train on the rewards your verifiers give.
