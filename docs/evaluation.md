# Evaluation

An evaluation measures a model and trains nothing. It attempts held-out tasks on a sampler, once each, and reports how the attempts scored. A training loop takes the held-out tasks, and outside a loop `evaluate` is one call on a sampling client.

## Held-out tasks

A held-out task is a task like any other, as on [Episode sessions](episode-sessions.md), that your run never trains:

```python
from transformers import AutoTokenizer

from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.message_formatters.models import message_formatter
from looper_cookbook.verifiers.final_number import FinalNumber
from looper_sdk import SamplingParams

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
formatter = message_formatter("Qwen/Qwen3.5-9B", tokenizer, enable_thinking=False)


def task(question, answer):
    return AgentSession.Config(
        messages=[{"role": "user", "content": f"{question} End on '#### ' and the number."}],
        verifier=FinalNumber.Config(answer=answer),
        max_episode_tokens=1024,
    )


held_out = [task("What is 15 times 4?", "60"), task("What is 13 times 3?", "39")]
```

Any task can be held out: one with an [environment](environments.md) and tools as well as a one-turn question.

## In a training loop

The [RL loop](reinforcement-learning.md) takes the held-out tasks as `eval_tasks`, and attempts them on the run's own sampler. Here `dataset` is the run's tasks, as on that page:

```python
from looper_cookbook import train_rl

config = train_rl.Config(params=SamplingParams(max_tokens=512), group_size=8, eval_every=10)
reports = await train_rl.train(
    training_client, sampling_client, dataset, config,
    formatter=formatter, steps=50, eval_tasks=held_out,
)
print(reports[0].metrics["eval/reward"], reports[-1].metrics["eval/reward"])
```

- **Before the first step.** The first report is the baseline: the weights the call started from, measured before anything trained.
- **Every `eval_every` steps,** and after the last step whatever `eval_every` is.
- **On the weights just published.** The loop waits for a step's publish before it evaluates, so `eval/reward` in a step's report is of that step's weights.
- **Greedily, unless you say otherwise.** An evaluation samples at temperature 0, with the loop's own token budget for a turn, so it measures the model's best answer and measures the same thing each time. Set the config's `eval_params` to sample another way.
- **On the sampler you already have.** An evaluation starts nothing and takes no GPU of its own. Its requests share the run's sampler.

So one run's reports hold the model before and after, measured the same way. `train_rl_async` takes `eval_tasks` too.

A report's evaluation metrics are:

| Metric | What it is |
|---|---|
| `eval/reward` | The mean reward of the attempts. With a verifier that gives 1 or 0, the share of tasks answered right |
| `eval/episodes` | How many tasks were attempted |
| `eval/<metric>` | The mean of each episode metric, such as `eval/truncated`, the share of attempts cut off, or `eval/answered` |
| `eval/stop/<reason>` | The share of attempts each [stop reason](episode-sessions.md#why-an-episode-ends) ended, such as `eval/stop/max_turn_tokens` |
| `eval/<metric>_stderr` | The standard error of each of those, `eval/reward_stderr` included |

## On any sampler

Outside a loop, `evaluate` is the same measurement as one call. It takes the tasks, what their sessions are lent, and a sampling client:

```python
from looper_cookbook.episode_sessions.base import EpisodeSession, evaluate

greedy = SamplingParams(max_tokens=512, temperature=0.0)
runtime = EpisodeSession.Runtime(formatter=formatter, params=greedy)
metrics = await evaluate(held_out, runtime, sampling_client)
print(metrics["eval/reward"], metrics["eval/reward_stderr"], metrics["eval/episodes"])
```

A saved checkpoint is measured the same way, on a sampler of the checkpoint:

```python
from looper_sdk import SamplerConfig

checkpoint_sampler = await service.create_sampling_client(
    SamplerConfig(
        checkpoint_name="rl-basic-1a2b3c4d", max_seq_len=1024, max_concurrent_sequences=16
    ),
    timeout_s=20 * 60,
)
print(await evaluate(held_out, runtime, checkpoint_sampler))
await checkpoint_sampler.stop()
```

Name `base:Qwen/Qwen3.5-9B` in place of a checkpoint to measure the base model. This sampler is a second one, with GPUs of its own.

## The attempts themselves

To read what the model wrote, and not only the means, run the attempts yourself. `rollout` returns each task's episode:

```python
from looper_cookbook.episode_sessions.base import rollout

episodes = await rollout(held_out, runtime, sampling_client)
for episode in episodes:
    print(episode.reward, episode.stop_reason.value, episode.messages[-1]["content"][-40:])
```

Any other measurement is a call on the sampling client and your own arithmetic. Record it from a loop's `on_step`, or between two calls of `train`.

## Reading the numbers

- **A mean has an error.** `eval/reward` over 100 tasks moves by a few points from one set of tasks to another. `eval/reward_stderr` says how much.
- **Compare a rise with its error.** A rise from the baseline to the last step means something when it is larger than about twice the error of the difference: the square root of the two standard errors' squares, summed.
- **Hold tasks out.** A step's own `reward` is of tasks the run is training on. A held-out reward is of tasks it has never seen.
- **Watch what cut attempts off.** An attempt cut off by a token budget scores 0, so a low reward with a high `eval/truncated` asks for a larger budget, not a better model. The stop shares say [which budget](episode-sessions.md#two-token-budgets). A high `eval/stop/max_turn_tokens` asks for a larger `max_tokens`, a turn's. A high `eval/stop/max_episode_tokens` asks for a larger `max_episode_tokens` on the tasks.

No recipe checks its own result yet: read the baseline and the last report it prints.

## Next steps

- [Reinforcement learning](reinforcement-learning.md) evaluates between its steps.
- [Verifiers](verifiers.md) are what score each attempt.
- [The RL loop](../tutorials/302_rl_loop.ipynb) trains with held-out tasks, start to end.
