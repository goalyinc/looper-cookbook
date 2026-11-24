# Episode sessions

An episode session is one attempt at one task. It is built for the attempt, runs once, and returns an `Episode`: what the model read and wrote, and how it scored.

The cookbook has a word for each level:

| Level | What it is |
|---|---|
| A training loop, `train_rl` | One optimizer step per batch of tasks |
| A rollout | Sessions run on a sampler: a group of them for each task of a step, or one for each held-out task of an evaluation |
| A session, `AgentSession` | One attempt at one task: a model takes turns and calls tools until it answers |

The [RL loop](reinforcement-learning.md) rolls sessions out for you, in training and in an [evaluation](evaluation.md). This page runs one by hand, so you can see what a session does.

## A task is a session's config

A task is data: what the model is asked, and how an attempt is scored. Every session's task has the same fields, which `EpisodeSession.Config` defines:

| Field | What it sets |
|---|---|
| `messages` | What the model is asked: the conversation its first turn follows |
| `verifier` | The config of the [verifier](verifiers.md) that scores an attempt |
| `environment` | The config of the [environment](environments.md) the model's tool calls run in. Leave it out for a task the model answers in one turn |
| `max_turns` | The most turns an attempt samples, 1 by default |
| `max_episode_tokens` | The episode's token budget: its first prompt, every sampled turn and every tool reply together. One turn's budget is not set here: see [two token budgets](#two-token-budgets) |

Each kind of session has a config of its own that holds these fields and builds it. `AgentSession.Config` is the cookbook's, and adds no field.

This task asks for a program. The model may check it against two example tests with a tool, and hidden tests decide the reward:

```python
from transformers import AutoTokenizer

from looper_cookbook.environments.coding import CodeTest, CodingEnvironment
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.episode_sessions.base import EpisodeSession
from looper_cookbook.message_formatters.models import message_formatter
from looper_cookbook.sandboxes.docker import DockerSandbox
from looper_cookbook.verifiers.code_tests import CodeTestsVerifier
from looper_sdk import SamplingParams

docker = DockerSandbox.Config(image="python:3.12-slim")
examples = [CodeTest(input="1 2\n", output="3\n"), CodeTest(input="10 20 30\n", output="60\n")]
task = AgentSession.Config(
    messages=[{"role": "user", "content": "Print the sum of the numbers on standard input."}],
    environment=CodingEnvironment.Config(sandbox=docker, tests=examples),
    verifier=CodeTestsVerifier.Config(
        sandbox=docker, tests=[CodeTest(input="4 5 6 7\n", output="22\n")]
    ),
    max_turns=4,
    max_episode_tokens=8192,
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
formatter = message_formatter("Qwen/Qwen3.5-9B", tokenizer, enable_thinking=False)
runtime = EpisodeSession.Runtime(formatter=formatter, params=SamplingParams(max_tokens=1024))
```

`EpisodeSession.Runtime` holds what every task of a run shares: the model's [message formatter](message-formatting.md), and how a turn is sampled. Its `max_tokens` is one turn's token budget. `task.build(runtime)` makes a fresh session for one attempt, so a task can be attempted as many times as you like.

## Run an episode

With a sampler, one call runs the whole episode:

```python
async with await task.build(runtime) as session:
    episode = await session.run(sampling_client)
```

`run` is three steps around the sampler: `start` returns the first turn's request, `observe` takes each sampled turn and returns the next request, and `finish` scores the episode. Here each turn is written by hand, so the page needs no server:

```python
from looper_sdk import FinishReason, SampleSequence


def turn(text):
    """A turn as a sampler returns it, written here by hand."""
    tokens = tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)
    return SampleSequence(
        token_ids=tokens, logprobs=[-1.0] * len(tokens), finish_reason=FinishReason.STOP
    )


program = "print(sum(map(int, input().split())))"
async with await task.build(runtime) as session:
    request = await session.start()
    print(len(request.prompt), request.params.max_tokens)
    request = await session.observe(
        turn(
            "<tool_call>\n<function=check_solution>\n<parameter=code>\n"
            f"{program}\n</parameter>\n</function>\n</tool_call>"
        )
    )
    print(repr(tokenizer.decode(request.prompt)[-149:]))
    request = await session.observe(turn(f"```python\n{program}\n```"))
    episode = await session.finish()
print(request, episode.stop_reason.value, episode.reward)
```

```output
315 1024
'<|im_start|>user\n<tool_response>\n{"passed": true, "tests_passed": 2, "tests": 2}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
None answer 1.0
```

- **`start` builds the environment** and returns a `SampleRequest`: the first prompt, with the environment's tools listed, and the sampling params for the turn.
- **A turn that calls a tool is answered.** Each call runs in the environment, in order, and the next prompt is every token so far and then the tools' replies. A sampled turn is never written again, so an episode is one token sequence.
- **A turn that calls no tool is the model's answer,** and ends the episode: `observe` returns `None`.
- **`finish` asks the verifier for the score,** with the environment still open. Here it ran the answer's program against the hidden test, in a container of its own.
- **Leaving the `async with` closes the session,** which closes its environment and that environment's sandbox.

A call the model gets wrong, such as a tool that does not exist, is answered with what was wrong, and the episode goes on. A sandbox that fails raises from the step it happened in.

## Two token budgets

An episode has several turns, so its tokens have two budgets. Each is set in its own place:

| Budget | Where you set it | What it counts |
|---|---|---|
| A turn's | `SamplingParams.max_tokens`, in the runtime's `params` | The tokens the sampler returns for one turn. Every task of a run shares it |
| An episode's | `max_episode_tokens`, in the task's config | The first prompt, every sampled turn and every tool reply together. Each task sets its own |

Each turn's request carries the smaller of the turn's budget and what the episode has left. `max_seq_len` is neither budget. It is the longest sequence your run and your sampler hold, and it must hold the episode's budget.

Here a turn is cut off under each. The first task keeps its 8,192 tokens, so the turn's 1,024 are the limit. The second has 1,000 tokens for its whole episode, and its first prompt took 315 of them:

```python
import dataclasses


def cut_off(text):
    """A turn the sampler cut off at its request's limit."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return SampleSequence(
        token_ids=tokens, logprobs=[-1.0] * len(tokens), finish_reason=FinishReason.LENGTH
    )


for episode_tokens in (8192, 1000):
    short = dataclasses.replace(task, max_episode_tokens=episode_tokens)
    async with await short.build(runtime) as session:
        request = await session.start()
        await session.observe(cut_off("```python\nprint(sum("))
        ended = await session.finish()
    print(request.params.max_tokens, ended.stop_reason.value, ended.reward)
```

```output
1024 max_turn_tokens 0.0
685 max_episode_tokens 0.0
```

## Why an episode ends

`episode.stop_reason` says why, and which budget when a budget ended it:

| Reason | What happened | Verified |
|---|---|---|
| `answer` | A turn called no tool, so it is the model's answer | Yes |
| `max_turns` | The last turn the task allows called tools. They ran | Yes |
| `max_turn_tokens` | A turn was cut off at the turn's budget | No: it scores 0 |
| `max_episode_tokens` | The episode's budget ran out. Either a turn was cut off at what the episode had left, or a turn's tools replied and no token was left for another turn | A turn cut off: no, it scores 0. No token left for another turn: yes |

A turn that was cut off, by either budget, is not an answer. A number or a program cut off in the middle may be right by accident, so the verifier never sees it, and the episode's `truncated` metric is 1.

## The episode

An `Episode` is what every session returns:

| Field | What it holds |
|---|---|
| `turns` | Each sampled turn: `context`, the tokens the model read since its last turn, and `sequence`, what the sampler returned |
| `reward` | The verifier's score |
| `stop_reason` | Why it ended |
| `messages` | The conversation: the task's messages, each sampled turn as a message, and each tool's reply |
| `metrics` | Counts for your log: the environment's, the verifier's, and `turns`, `sampled_tokens` and `truncated` |

`episode.datum(group_id, group_size)` is the episode as one datum for `forward_backward`: every token in order, with weight 1 on the sampled ones, their log probabilities, and the reward.

```python
print(dict(episode.metrics))
datum = episode.datum("step-1-task-0", 8)
print(len(datum.tokens), sum(datum.weights), datum.reward, datum.group_id)
print([message["role"] for message in episode.messages])
```

```output
{'passed': 1.0, 'answered': 1.0, 'turns': 2.0, 'sampled_tokens': 50.0, 'truncated': 0.0}
400 50.0 1.0 step-1-task-0
['user', 'assistant', 'tool', 'assistant']
```

## A task the model answers in one turn

Leave `environment` out. The first turn calls no tool, so it is the answer:

```python
from looper_cookbook.verifiers.final_number import FinalNumber

question = AgentSession.Config(
    messages=[{"role": "user", "content": "What is 6 times 7? End on '#### ' and the number."}],
    verifier=FinalNumber.Config(answer="42"),
    max_episode_tokens=2048,
)
async with await question.build(runtime) as session:
    await session.start()
    await session.observe(turn("6 times 7 is 42.\n#### 42"))
    answered = await session.finish()
print(answered.stop_reason.value, answered.reward, len(answered.turns))
```

```output
answer 1.0 1
```

## A rollout: many sessions at once

`rollout` attempts each of a list of tasks, all at once, and returns their episodes in the tasks' order. The RL loops use it, for a step's groups and for an evaluation. A task given several times is attempted that many times, which is how a group is made:

```python
from looper_cookbook.episode_sessions.base import rollout

group = await rollout([question] * 8, runtime, sampling_client)
print([episode.reward for episode in group])
```

When one attempt fails, the rest are cancelled and closed before its error raises, so no session is left sampling and no sandbox is left running.

## A harness of your own

To change how a turn is answered, subclass `AgentSession` and override `respond`, which returns the messages that answer a sampled turn. None means the turn was the model's answer. This one reminds a model that neither called a tool nor gave a program:

```python
import dataclasses


class Reminding(AgentSession):
    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Config(AgentSession.Config):
        async def build(self, runtime):
            return Reminding(self, runtime, await self.verifier.build())

    async def respond(self, reply):
        replies = await super().respond(reply)
        if replies or "```" in reply["content"]:
            return replies
        reminder = "Call check_solution, or give your program in a code block."
        return [{"role": "tool", "name": "check_solution", "content": reminder}]


reminding = Reminding.Config(
    messages=task.messages,
    environment=task.environment,
    verifier=task.verifier,
    max_turns=4,
    max_episode_tokens=8192,
)
async with await reminding.build(runtime) as session:
    await session.start()
    request = await session.observe(turn("I would add the numbers."))
    print(repr(tokenizer.decode(request.prompt)[-160:-42]))
```

```output
'<|im_start|>user\n<tool_response>\nCall check_solution, or give your program in a code block.\n</tool_response><|im_end|>'
```

- **The subclass has a `Config` of its own that builds it.** A subclass without one is refused when it is defined, since `AgentSession.Config` would build a plain `AgentSession`.
- **A reply is a tool message.** A prompt continues with tool replies only: a new user message would change how some formats write the turns before it.
- **The sampling, the tokens and the budgets stay `AgentSession`'s.**

## A session of your own

`AgentSession` is one kind of session. Another kind, such as a model that talks to a simulated user, is a subclass of `EpisodeSession` with two methods. Its config subclasses `EpisodeSession.Config`, so its task has the fields every task has, and adds what only your session needs:

```python
from looper_cookbook.episode_sessions.base import Episode


class MySession(EpisodeSession):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Config(EpisodeSession.Config):
        persona: str  # a field of its own: who the simulated user is

        async def build(self, runtime):
            return MySession(self, runtime)

    def __init__(self, config, runtime):
        self._config, self._runtime = config, runtime

    async def run(self, sampling_client) -> Episode:
        ...  # sample turns from sampling_client, and return an Episode built from them

    async def close(self) -> None:
        ...  # release what the session built; a second call does nothing
```

- **Read the task from your config:** `messages`, `verifier`, `environment`, `max_turns` and `max_episode_tokens` are checked when a config is made, as `AgentSession.Config`'s are.
- **Keep the task's limits.** Sample at most `max_turns` turns, and end the episode before it holds more than `max_episode_tokens` tokens.
- **Return an `Episode` whose `turns` hold every token in order,** each sampled turn after the tokens the model read before it. Its `datum` is then right, so your session trains through the RL loop as `AgentSession` does: put its configs in a [dataset](reinforcement-learning.md#tasks-and-the-dataset), or hand them to a loop as [held-out tasks](evaluation.md).

## Next steps

- [Reinforcement learning](reinforcement-learning.md) runs a group of sessions for every task and trains on their rewards.
- [Environments](environments.md) and [verifiers](verifiers.md) are what a task names.
- [Message formatting](message-formatting.md) explains how a prompt continues with tool replies.
