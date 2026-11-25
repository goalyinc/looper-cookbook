# Episode sessions

An episode session runs one attempt at one task. It asks the model for a turn, runs any tools the model calls, and repeats until the model gives its answer. Then it has the task's [verifier](verifiers.md) score the answer. It returns an `Episode`, which holds everything the model read and wrote, and the reward.

If you use the cookbook's [RL loops](reinforcement-learning.md) or its [evaluation](evaluation.md), they run sessions for you, and you only write the tasks. Read this page to see what happens inside an attempt, to set an attempt's limits, or to change how an attempt runs. The examples here run a session by hand, step by step.

This page covers:

- [Words used on this page](#words-used-on-this-page): task, attempt, turn and the other terms the cookbook uses.
- [A task is a session's config](#a-task-is-a-sessions-config): what you write for each task, with an example that runs through the page.
- [Run an episode](#run-an-episode): one attempt, turn by turn.
- [Two token budgets](#two-token-budgets): the limit on one turn, and the limit on the whole attempt.
- [Why an episode ends](#why-an-episode-ends): the reasons an attempt stops, and which ones are scored.
- [The episode](#the-episode): what a session returns, and how it becomes training data.
- [A task the model answers in one turn](#a-task-the-model-answers-in-one-turn): a task without tools.
- [A rollout: many sessions at once](#a-rollout-many-sessions-at-once): run many attempts together, as the RL loops do.
- [A harness of your own](#a-harness-of-your-own): change how the model's turns are answered.
- [A session of your own](#a-session-of-your-own): write a different kind of session.

## Words used on this page

| Word | What it means | In this page's example |
|---|---|---|
| **Task** | One question for the model, together with how to score its answers | "Print the sum of the numbers on standard input" |
| **Attempt**, or **episode** | One try at a task, from the question to the model's answer. One attempt gets one reward | One run of the model at the sum task |
| **Episode session** | The object that runs one attempt. The cookbook's is `AgentSession` | `await task.build(runtime)` |
| **Turn** | One reply from the model within an attempt. A turn either calls tools or gives the answer | The model calling `check_solution` with its program |
| **Tool call** | A turn asking for a tool to run, such as a shell command. The [environment](environments.md) runs it, and its reply is added to the conversation | `check_solution` running the program against example tests |
| **Group** | Several attempts at the same task, which the RL loop compares | 8 attempts at the sum task |
| **Rollout** | Running many sessions on a sampler at once, such as a group for each task in a training step, or one attempt for each task in an evaluation | `rollout([question] * 8, ...)` |

## A task is a session's config

A task is data. It says what the model is asked, and how an attempt is scored. You write one for each question. Every task has the same fields, which `EpisodeSession.Config` defines:

| Field | What it sets |
|---|---|
| `messages` | The question, as the conversation the model's first turn follows |
| `verifier` | The [verifier](verifiers.md) that scores the attempt, with this task's right answer or tests |
| `environment` | The [environment](environments.md) that runs the model's tool calls. Leave it out for a task the model answers in one turn, with no tools |
| `max_turns` | The most turns one attempt can take. 1 by default |
| `max_episode_tokens` | The token budget for the whole attempt: its first prompt, every turn and every tool reply together. Required. A single turn's budget is set elsewhere (see [two token budgets](#two-token-budgets)) |

Each kind of session has its own config class with these fields, which builds that session. `AgentSession.Config` is the cookbook's, and adds no fields of its own.

**The example on this page** asks the model for a program that prints the sum of the numbers on standard input. The model can test its program as it works, with a tool called `check_solution` that runs it against two example tests. A third test, which the model never sees, decides the reward:

- **`environment`** is a [`CodingEnvironment`](environments.md#check-a-program-codingenvironment) holding the two example tests, `1 2` and `10 20 30`. The model sees these through the tool.
- **`verifier`** is a [`CodeTestsVerifier`](verifiers.md#a-program-codetestsverifier) holding the hidden test, `4 5 6 7`. The model never sees this one.
- **Both run code in a Docker container.** They share the same `DockerSandbox` settings, but each starts its own container.
- **`max_turns=4`** lets the model test its program a few times before it answers.

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

All of this is your setup code. The classes you import are the cookbook's.

The last three lines set up the **runtime**, `EpisodeSession.Runtime`. It holds the things every task in a run shares. These are the model's [message formatter](message-formatting.md), which turns messages into the model's tokens, and the sampling settings for each turn. Its `max_tokens` is the most tokens one turn can use.

`task.build(runtime)` creates a new session for one attempt. You can build as many as you like from the same task, one for each attempt.

## Run an episode

With a sampler, one call runs the whole attempt:

```python
async with await task.build(runtime) as session:
    episode = await session.run(sampling_client)
```

Inside, `run` is three steps around the sampler:

1. **`start`** sets up the environment, and returns the request for the first turn.
2. **`observe`** takes each turn the sampler wrote, runs its tool calls, and returns the request for the next turn. It returns `None` when the turn is the model's answer.
3. **`finish`** has the verifier score the answer.

The code below runs these steps by hand, so you can see each one without a sampler. The `turn` helper stands in for the sampler. It turns text into a turn, as if the model had written it. The model plays two turns:

- **Turn 1:** the model calls `check_solution` with its program. The environment runs it against the two example tests, and both pass.
- **Turn 2:** the model gives the same program in a code block, and calls no tool. That's its answer, so the attempt ends, and the verifier runs the program against the hidden test.

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

The three lines of output show:

1. The first prompt is 315 tokens, and a turn can use up to 1,024.
2. The end of the second prompt: the tool's reply to turn 1 (both example tests passed), then the start of the model's next turn.
3. After turn 2 there's no next request (`None`). The attempt ended because the model answered (`answer`), and it passed the hidden test, so the reward is 1.

What each step does:

- **`start` sets up the environment** and returns a `SampleRequest`: the first prompt, with the environment's tools listed in it, and the sampling settings for the turn.
- **A turn that calls a tool gets a reply.** Each call runs in the environment, in order. The next prompt is every token so far, then the tools' replies. The model's turns are kept exactly as they were sampled, never re-encoded, so the whole attempt is one sequence of tokens that can be trained on directly.
- **A turn that calls no tool is the model's answer,** and ends the attempt: `observe` returns `None`.
- **`finish` asks the verifier for the score,** while the environment is still open. Here the verifier ran the answer's program against the hidden test, in a container of its own.
- **Leaving the `async with` closes the session,** which closes its environment and the environment's sandbox.

If the model makes a mistake in a tool call, such as calling a tool that doesn't exist, the reply says what was wrong, and the attempt goes on. If a sandbox itself fails, the step it failed in raises an error.

## Two token budgets

An attempt can have several turns, so there are two limits on its length, each set in a different place:

| Budget | Where you set it | What it counts |
|---|---|---|
| One turn's | `SamplingParams.max_tokens`, in the runtime's `params` | The tokens the model writes in one turn. Every task in a run shares it |
| The whole attempt's | `max_episode_tokens`, in the task | The first prompt, every turn and every tool reply together. Each task sets its own |

Each turn is allowed whichever is smaller: one turn's budget, or what's left of the attempt's budget.

`max_seq_len`, which you set on the training run and the sampler, is neither budget. It's the longest sequence they can hold, so it must be at least `max_episode_tokens`.

The code below cuts off a turn under each budget. The first task keeps its 8,192-token attempt budget, so the turn's own 1,024 tokens are the limit. The second task has only 1,000 tokens for the whole attempt, and its first prompt took 315 of them, which leaves 685 for the turn:

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

Both attempts score 0, because a cut-off turn isn't an answer, as the next section explains.

## Why an episode ends

`episode.stop_reason` says why an attempt ended, and which budget ended it, if one did:

| Reason | What happened | Scored by the verifier? |
|---|---|---|
| `answer` | A turn called no tool, so it's the model's answer | Yes |
| `max_turns` | The last turn the task allows called tools. The tools ran, and the attempt ended there | Yes |
| `max_turn_tokens` | A turn was cut off at one turn's budget | No, it scores 0 |
| `max_episode_tokens` | The attempt's budget ran out. Either a turn was cut off at what the attempt had left, or the tools replied and no tokens were left for another turn | No, it scores 0, if a turn was cut off. Yes, if no tokens were left for another turn |

A turn that was cut off, by either budget, isn't an answer. A number or a program cut off partway might be right by accident, so the verifier never sees it, and the episode's `truncated` metric is 1. [Verifiers: Cut-off answers](verifiers.md#cut-off-answers) explains the same rule from the verifier's side.

## The episode

Every session returns an `Episode`, whatever happened during the attempt:

| Field | What it holds |
|---|---|
| `turns` | Each of the model's turns, with `context`, the tokens the model read since its last turn, and `sequence`, what the sampler returned |
| `reward` | The verifier's score |
| `stop_reason` | Why it ended |
| `messages` | The conversation, made of the task's messages, each of the model's turns as a message, and each tool's reply |
| `metrics` | Numbers for your logs, from the environment and the verifier, plus `turns`, `sampled_tokens` and `truncated` |

`episode.datum(group_id, group_size)` turns the episode into training data, as one datum for `forward_backward`. It holds every token in order, with weight 1 on the tokens the model wrote, their log probabilities, and the reward. `group_id` and `group_size` say which group the attempt belongs to, so the loss can compare it with the other attempts at the same task. The RL loops do this for you. Here it is for the sum task's episode:

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

For a question with no tools, leave `environment` out. The model's first turn calls no tool, so it's the answer:

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

`rollout` runs one attempt at each task in a list, all at the same time, and returns their episodes in the same order as the tasks. To make a group, list the same task several times, and each copy is attempted once. The RL loops use `rollout` for each training step's groups, and for evaluations:

```python
from looper_cookbook.episode_sessions.base import rollout

group = await rollout([question] * 8, runtime, sampling_client)
print([episode.reward for episode in group])
```

If one attempt fails, the others are stopped and closed before its error is raised. So no session is left sampling, and no sandbox is left running.

## A harness of your own

The **harness** is the code that answers the model's turns. `AgentSession`'s harness runs each tool call in the environment, and replies with the result. To answer turns differently, subclass `AgentSession` and override `respond`. It takes the model's turn as a message, and returns the messages that reply to it. Returning no messages means the turn was the model's answer, and ends the attempt.

This example reminds the model when it neither called a tool nor gave a program, instead of ending the attempt:

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

- **The subclass needs its own `Config`, which builds it.** A subclass without one is refused as soon as it's defined, because `AgentSession.Config` would build a plain `AgentSession` instead.
- **A reply is a tool message.** A prompt can only continue with tool replies: a new user message would change how some chat formats write the turns before it.
- **Everything else stays `AgentSession`'s:** the sampling, the tokens and the budgets.

## A session of your own

`AgentSession` is one kind of session: one model that takes turns and calls tools until it answers. For a different kind, such as a model that talks to a simulated user, subclass `EpisodeSession` and write two methods, `run` and `close`. Its config subclasses `EpisodeSession.Config`, so its tasks have the same fields as every other task, plus any your session needs:

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

- **Read the task from your config.** `messages`, `verifier`, `environment`, `max_turns` and `max_episode_tokens` are checked when a config is created, the same as for `AgentSession.Config`.
- **Keep the task's limits.** Sample at most `max_turns` turns, and end the attempt before it holds more than `max_episode_tokens` tokens.
- **Return an `Episode` whose `turns` hold every token in order,** with each turn after the tokens the model read before it. Then its `datum` is right, and your session trains through the RL loops the same way `AgentSession` does: put its configs in a [dataset](reinforcement-learning.md#tasks-and-the-dataset), or give them to a loop as [held-out tasks](evaluation.md).

## Next steps

- [Reinforcement learning](reinforcement-learning.md): the RL loops run a group of sessions for every task, and train on their rewards.
- [Environments](environments.md) and [verifiers](verifiers.md): the tools and the scoring a task names.
- [Message formatting](message-formatting.md): how a prompt continues with tool replies.
