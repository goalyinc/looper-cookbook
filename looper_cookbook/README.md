# looper_cookbook/

The cookbook's package. Everything that runs is here; the
[tutorials](../tutorials/README.md) are notebooks that document it.

## Usage

A loop trains a run you created, so these need a Looper server:
`training_client` and `sampling_client` are the SDK's clients for the run
and its sampler, and `formatter` the model's
[message formatter](message_formatters/README.md). A
[recipe](recipes/README.md) is the same calls as a whole program.

`train_sft` takes one optimizer step per batch of a
[dataset](datasets/README.md), at the rate its schedule gives the step:

```python
from looper_cookbook import train_sft

config = train_sft.Config(learning_rate=4e-5, schedule="linear")
steps = await train_sft.train(training_client, dataset, config, steps=3 * len(dataset))
print(steps[-1].loss_sum / steps[-1].weighted_targets)
```

`train_rl` takes one step per batch of tasks. It attempts each task
`group_size` times on the sampler, trains the groups, and publishes before
it samples again. Held-out tasks, `eval_tasks`, are attempted once each
before the first step and after the last, and train nothing:

```python
from looper_cookbook import train_rl
from looper_sdk import SamplingParams

config = train_rl.Config(params=SamplingParams(max_tokens=512), group_size=8)
reports = await train_rl.train(
    training_client, sampling_client, dataset, config,
    formatter=formatter, steps=20, eval_tasks=held_out,
)
print(reports[0].metrics["eval/reward"], reports[-1].metrics["eval/reward"])
```

`train_rl_async` takes the same arguments, and samples while a step trains.
`max_policy_lag` is how many steps the weights that sampled a group may
trail the weights that train it:

```python
from looper_cookbook import train_rl_async
from looper_sdk import SamplingParams

config = train_rl_async.Config(
    params=SamplingParams(max_tokens=512), group_size=8, max_policy_lag=1
)
reports = await train_rl_async.train(
    training_client, sampling_client, dataset, config, formatter=formatter, steps=20
)
print(reports[-1].metrics["policy_lag"], reports[-1].metrics["groups_stale"])
```

The [Supervised learning](../docs/supervised-learning.md),
[Reinforcement learning](../docs/reinforcement-learning.md) and
[Evaluation](../docs/evaluation.md) pages have the rest: creating a run and
its sampler, each config's settings, what a step reports, and when to pick
the asynchronous loop.

| Path | What it holds |
|---|---|
| [message_formatters/](message_formatters/README.md) | The message formatters, one per model family: Qwen3, Qwen3.5, Qwen3.8 and Muse Glimmer. Conversations of chat messages become generation prompts and a datum per assistant turn, the turn alone as targets; sampled turns parse back into messages, with their reasoning and tool calls; `targets` is the text a datum trains. `models.py` names each served model's formatter |
| [environments/](environments/README.md) | Where a model's actions run: the `Environment` contract, with `tools`, `step`, `sandbox` and `close`; and two example environments, `BashEnvironment`, a shell, and `CodingEnvironment`, which checks a Python program against a problem's tests |
| [sandboxes/](sandboxes/README.md) | Where commands run: the `Sandbox` contract, with `run` and `close`; `LocalSandbox`, processes of this machine, and `DockerSandbox`, a container. An environment that runs commands takes one as its argument |
| [verifiers/](verifiers/README.md) | What scores a finished episode: the `Verifier` contract, one call that returns a `Score`; `FinalNumber`, a final number matched exactly, and `CodeTestsVerifier`, a program run against hidden tests in a sandbox of its own |
| [episode_sessions/](episode_sessions/README.md) | One attempt at one task: the `EpisodeSession` contract, `run(sampling_client)` and `close`, built from a task's config, and the `Episode` every session returns, whose `datum(...)` is its training data. `AgentSession` is the cookbook's session, one model that takes turns and calls tools: each call of a sampled turn runs in the task's environment, the prompt is continued with the replies, and the task's verifier scores the result |
| [datasets/](datasets/README.md) | A run's data, a batch per optimizer step: the contracts, `SupervisedDataset` for the SFT loop and `RLDataset` for the RL loop, each `len(dataset)` and `batch(index)`. `ChatDataset` is conversations in batches, each a datum per assistant turn, and `read_conversations` reads a JSONL file of them. `TaskDataset` is tasks in batches. Each pass is in an order of its own, which a seed fixes |
| `train_sft.py` | The explicit SFT loop over `TrainingClient`. `train(training_client, dataset, config, steps=...)` takes one optimizer step per batch on a run its caller created: the run's step `k` trains `dataset.batch(k)`, at the rate its schedule gives that step, submitting `steps_ahead` steps beyond the one it awaits |
| `train_rl.py` | The explicit RL loop over `TrainingClient` and the run's `SamplingClient`. `train(training_client, sampling_client, dataset, config, formatter=..., steps=...)` takes one optimizer step per batch of tasks: it runs a group of episode sessions for each task, trains each group whole under `Grpo` or `Dapo`, publishes the step to the sampler and awaits the publish before it samples again. Given held-out tasks, it attempts them before its first step and when its config says. It saves when its config says. Its `dataset` is an `RLDataset`, of `datasets/` |
| `train_rl_async.py` | The asynchronous RL loop, with `train_rl`'s arguments: sampling does not wait for training. It keeps the groups of the next `max_policy_lag + 1` steps in flight or waiting, a step trains the first that finish, and a group that finishes older than the lag is dropped. `RolloutQueue` holds the groups in flight and the finished ones. `max_policy_lag=0` is `train_rl`'s loop |
| [recipes/](recipes/README.md) | The entry points, run as `python -m looper_cookbook.recipes.<name>` |
| [example_data/](example_data/README.md) | The recipes' example data, small enough to read |
| [tests/](tests/README.md) | The sandboxes', environments' and episode sessions' integration tests, the agent session's test against a real server, the formatters' check against each model's template, and the tutorials' runner and recorder; the unit tests sit beside the sources, in `unit_tests/` |

Rules:

- Every module here, the tests included, imports `looper_sdk` and nothing
  else of ours, and takes a tokenizer as an object. It runs wherever the
  SDK is installed.
- Prompts, datums and replies go through the model's message formatter, so
  a model is sampled as it was trained.
- A loop takes a run its caller created, and the RL loop the run's
  sampler, which the caller creates from the run's training client. It
  returns what the run reported. Creating the run and its sampler,
  preparing the data and saving are its caller's, as `recipes/sft_basic.py`
  and `recipes/rl_basic.py` show. It has no recovery: an error raises, and
  a `RunRolledBack` is its caller's to answer.
- A loop is a module here, short enough to read and meant to be copied
  and changed.
- A conversation its model's format cannot hold is refused, never trained
  approximately.
- An environment runs a model's actions and scores nothing. It imports no
  message formatter and nothing of the SDK, so one environment serves any
  model and any loop.
- Three things here are loops, one inside the next. A training loop,
  `train_sft` or `train_rl`, takes a step per batch. The RL loop runs a
  group of episode sessions for each task, with `rollout`. Inside one session, `AgentSession`
  is one model's turns. "Loop" alone means a training loop.
- An episode's tokens are only appended to. `AgentSession` continues a prompt
  with `continue_prompt` and never formats a sampled turn again, so an
  episode is one datum whose log probabilities are those its sampler
  recorded.
- A session samples from the client it is handed. `AgentSession`'s `start`
  and `observe` return what to sample, and `run` takes the sampling
  client, so its unit tests build no client and mock nothing of the SDK.
- A concept with more than one implementation is a package. Its `base.py`
  is the contract, and each other module one implementation, named for
  what it is: `sandboxes/docker.py` holds `DockerSandbox`.
- A loop and a contract import only contracts. A module at this package's
  top level, and each package's `base.py`, import of the cookbook only
  `base.py` modules and other top-level modules. Naming an implementation,
  a Docker sandbox or a model's formatter, is a recipe's, a tutorial's or
  a test's. CI checks it.
- A contract is a base class when it does work every implementation
  passes through: `Environment.step` checks a call before it runs, and
  `MessageFormatter` tokenizes. So is one that nests a `Config`, as
  `Verifier` and `EpisodeSession` do. One that is only a signature is a `Protocol`, as each
  `Config` is.
- What a task carries as data is built through its nested `Config`, with
  `await config.build()`: an episode session, its environment, the
  environment's sandbox and its verifier. A message
  formatter is constructed with its tokenizer, a live object a run has one
  of. A loop's `Config` holds its function's settings and builds nothing.
- A notebook never lives here: it is a tutorial, in `tutorials/`.
