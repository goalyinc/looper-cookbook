# Evaluation

An evaluation measures how well a model does on a set of tasks, without training it. The model makes one attempt at each task, a [verifier](verifiers.md) scores each attempt, and the evaluation reports the average score. For example, on 100 math questions scored 1 for right and 0 for wrong, a score of 0.62 means the model got 62 of them right.

You can evaluate in two places:

- **During RL training,** to see whether the model is getting better. The cookbook's RL loops do this for you, if you give them tasks to evaluate on.
- **On any sampler,** with one call. For example, to measure a base model, or a checkpoint you saved.

This page covers:

- [Held-out tasks](#held-out-tasks): the tasks you evaluate on.
- [In a training loop](#in-a-training-loop): evaluate as RL trains, and read the results.
- [On any sampler](#on-any-sampler): evaluate a base model, a checkpoint or a running sampler with one call.
- [The attempts themselves](#the-attempts-themselves): read what the model actually wrote.
- [Reading the numbers](#reading-the-numbers): tell a real improvement from noise.

Words used on this page:

| Word | Meaning |
|---|---|
| **Task** | One question for the model, with the verifier that scores its answers. See [Verifiers](verifiers.md#where-a-verifier-goes-the-task). |
| **Held-out task** | A task the model never trains on, kept apart to measure it. |
| **Attempt** | One try at a task, which gives one answer from the model and its reward. Also called an **episode**. |
| **Baseline** | The score before training starts, to compare the trained model against. |
| **Standard error** | How much an average score would move if you measured on a different set of tasks of the same kind. |

## Held-out tasks

A held-out task is a task like any other, as on [Episode sessions](episode-sessions.md), but your run never trains on it. It's kept apart so you measure how well the model answers questions it hasn't seen, not how well it remembers the ones it trained on.

These two arithmetic questions are the held-out tasks used throughout this page. Each has a `FinalNumber` verifier with its right answer:

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

Any task can be held out, including one where the model uses tools in an [environment](environments.md), not just a one-turn question.

## In a training loop

Give the [RL loop](reinforcement-learning.md) your held-out tasks as `eval_tasks`, and it evaluates the model as it trains, on the run's own sampler. Here `dataset` holds the tasks the run trains on, as on that page:

```python
from looper_cookbook import train_rl

config = train_rl.Config(params=SamplingParams(max_tokens=512), group_size=8, eval_every=10)
reports = await train_rl.train(
    training_client, sampling_client, dataset, config,
    formatter=formatter, steps=50, eval_tasks=held_out,
)
print(reports[0].metrics["eval/reward"], reports[-1].metrics["eval/reward"])
```

The last line prints the score before training and the score after the last step.

When and how the loop evaluates:

- **Before the first step.** The first report is the baseline, which measures the model before anything is trained.
- **Every `eval_every` steps,** and always after the last step.
- **On the newest weights.** The loop waits for a step's weights to reach the sampler before it evaluates. So the `eval/reward` in a step's report measures that step's weights.
- **With the model's most likely answer, unless you say otherwise.** An evaluation samples at temperature 0, with the loop's own token limit per turn. So it measures the model's best answer, the same way every time. To sample differently, set `eval_params` in the loop's config.
- **On the sampler you already have.** An evaluation doesn't start anything, or use GPUs of its own. It shares the run's sampler.

So one run's reports show the model before and after training, measured the same way. `train_rl_async` takes `eval_tasks` too.

Each report has these evaluation metrics:

| Metric | What it is |
|---|---|
| `eval/reward` | The average reward of the attempts. With a verifier that gives 1 or 0, this is the share of tasks answered right. |
| `eval/episodes` | How many tasks were attempted. |
| `eval/<metric>` | The average of each attempt's metric, such as `eval/truncated`, the share of attempts cut off, or `eval/answered`. |
| `eval/stop/<reason>` | The share of attempts that ended for each [stop reason](episode-sessions.md#why-an-episode-ends), such as `eval/stop/max_turn_tokens`. |
| `eval/<metric>_stderr` | The standard error of each of these, including `eval/reward_stderr`. |

## On any sampler

Outside a training loop, `evaluate` does the same measurement in one call. It takes the tasks, a runtime and a sampling client. The runtime holds what each attempt needs, which is the formatter and the sampling parameters. Here they're set to temperature 0, as the loop does:

```python
from looper_cookbook.episode_sessions.base import EpisodeSession, evaluate

greedy = SamplingParams(max_tokens=512, temperature=0.0)
runtime = EpisodeSession.Runtime(formatter=formatter, params=greedy)
metrics = await evaluate(held_out, runtime, sampling_client)
print(metrics["eval/reward"], metrics["eval/reward_stderr"], metrics["eval/episodes"])
```

To measure a checkpoint you saved, start a sampler for it, and evaluate on that sampler:

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

To measure the base model instead, use `base:Qwen/Qwen3.5-9B` in place of the checkpoint's name. Either way, this is a second sampler, with GPUs of its own, so stop it when you're done, as the last line does.

## The attempts themselves

The averages don't show what the model wrote. To read the answers themselves, run the attempts with `rollout`, which returns each task's episode:

```python
from looper_cookbook.episode_sessions.base import rollout

episodes = await rollout(held_out, runtime, sampling_client)
for episode in episodes:
    print(episode.reward, episode.stop_reason.value, episode.messages[-1]["content"][-40:])
```

Each line shows an attempt's reward, why it ended, and the end of the model's last message.

For any other measurement, call the sampling client yourself, and do your own arithmetic. You can record it from a loop's `on_step`, or between two calls of `train`.

## Reading the numbers

- **An average is never exact.** `eval/reward` over 100 tasks moves by a few points from one set of tasks to another. `eval/reward_stderr` says by how much.
- **Check that a rise is bigger than the noise.** A rise from the baseline to the last step means something when it's more than about twice the error of the difference. That error is the square root of the sum of the two standard errors squared.
- **Evaluate on held-out tasks.** A step's own `reward` is on tasks the run is training on. A held-out reward is on tasks it has never seen.
- **Check what cut attempts off.** An attempt cut off by a token limit scores 0. So a low reward with a high `eval/truncated` means the limit is too small, not that the model is bad. The stop shares say [which limit](episode-sessions.md#two-token-budgets). A high `eval/stop/max_turn_tokens` means raising `max_tokens`, the limit for one turn. A high `eval/stop/max_episode_tokens` means raising the tasks' `max_episode_tokens`.

No recipe checks its own result yet. Compare the baseline with the last report it prints.

## Next steps

- [Reinforcement learning](reinforcement-learning.md) evaluates between its steps.
- [Verifiers](verifiers.md) score each attempt.
- [The RL loop](../tutorials/302_rl_loop.ipynb) trains with held-out tasks, from start to end.
