# Verifiers

A verifier scores one finished episode. It is one call: `verify(messages, environment)` reads the episode's conversation, and may read its environment, and returns a `Score`. An [episode session](episode-sessions.md) calls it once, when its episode has ended.

A `Score` is the episode's `reward`, higher for a better attempt, and `metrics`, counts for your log. In RL only differences in reward within a group train, so a reward of 1 for right and 0 for wrong is enough.

The cookbook has two verifiers in `looper_cookbook.verifiers`:

| Verifier | Reward 1 when | Needs |
|---|---|---|
| `final_number.FinalNumber` | The number after the reply's last `####` is the answer | Nothing |
| `code_tests.CodeTestsVerifier` | The reply's last code block passes every hidden test | A [sandbox](sandboxes.md) |

A task names its verifier by a config, and each attempt builds its own from it:

```python
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.verifiers.final_number import FinalNumber

task = AgentSession.Config(
    messages=[{"role": "user", "content": "What is 6 times 7? End on '#### ' and the number."}],
    verifier=FinalNumber.Config(answer="42"),
    max_episode_tokens=2048,
)
```

## A number: `FinalNumber`

`FinalNumber` scores a problem whose answer is one number. Ask the model to end on `####` and the number, as GSM8K writes its answers, and the verifier reads the number after the reply's last `####`:

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

- **Numbers compare as numbers,** so `1,200`, `1200` and `1200.0` are one answer, and a `$` before a number is skipped.
- **The last `####` counts,** so a reply may correct itself.
- **`answered`** says whether the reply gave a number at all. It tells a wrong answer from a reply in the wrong form.
- **Only the reply counts,** never the model's reasoning: a thinking model may try answers while it reasons.

## A program: `CodeTestsVerifier`

`CodeTestsVerifier` scores a programming problem. It takes the reply's last fenced code block as the program and runs it against the problem's tests:

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

- **The tests run in a sandbox of the verifier's own,** started for the check and closed after it. The model never acted there, so nothing it did during the episode can touch the tests or their verdict.
- **The expected outputs stay in your process.** The sandbox gets the program and each test's input, and returns what the program printed.
- **These are the hidden tests.** A [`CodingEnvironment`](environments.md#check-a-program-codingenvironment) shows the model a problem's example tests through a tool. The tests that decide the reward belong here.
- **A reply without a code block scores 0,** with `answered` 0, and starts no sandbox.
- **`prelude`** is code written before the program, as in the environment: give both the same one.

A test is a program test or a function test, as on the [Environments](environments.md#check-a-program-codingenvironment) page.

## What a verifier is never asked

An episode whose last turn was cut off, by a turn's token budget or by [the episode's](episode-sessions.md#two-token-budgets), is not verified. It scores 0. An answer cut off in the middle may hold the right number by accident, so the session never hands it to a verifier, and no sandbox starts for it.

## A verifier of your own

Subclass `Verifier` and write `verify`. It gets the whole conversation, and the episode's environment, still open, or `None` for a task without one. This one checks a file the model was asked to write, where the model worked:

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

- **Read, and do not act for the model.** A verifier is lent the environment. It neither closes it nor keeps it.
- **Keep the reference in the verifier:** the right answer, or the tests. No prompt, tool or observation should carry it.
- **Raise when you cannot check.** A verifier whose sandbox will not start, or whose judge does not answer, raises. A failure is never a reward: return a low score only for an episode you did check.
- **Start nothing until `verify`.** A session builds its verifier with itself, and a cut-off episode never calls it.
- **`last_reply(messages)`** in `looper_cookbook.verifiers.base` gives the content of the conversation's last assistant message, which is what both verifiers here read.

## Next steps

- [Episode sessions](episode-sessions.md) call a task's verifier when its episode ends.
- [Evaluation](evaluation.md) reports a verifier's rewards over held-out tasks.
