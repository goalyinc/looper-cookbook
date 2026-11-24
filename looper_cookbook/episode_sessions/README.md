# episode_sessions/

Episode sessions: one attempt at one task each. A
session is built for the attempt from the task's config, runs once on a
sampling client, and returns an `Episode`: what the model read and wrote,
and how it scored.
[Episode sessions](../../docs/episode-sessions.md) is the page for users.

## Usage

A task is an `AgentSession.Config`: what the model is asked, and the verifier
that scores an attempt. Running one needs a Looper server: `sampling_client`
is the SDK's `SamplingClient`, and `formatter` the model's
[message formatter](../message_formatters/README.md).

```python
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.episode_sessions.base import EpisodeSession, evaluate, rollout
from looper_cookbook.verifiers.final_number import FinalNumber
from looper_sdk import SamplingParams

task = AgentSession.Config(
    messages=[{"role": "user", "content": "What is 6 times 7? End on '#### ' and the number."}],
    verifier=FinalNumber.Config(answer="42"),
    max_episode_tokens=2048,
)
runtime = EpisodeSession.Runtime(formatter=formatter, params=SamplingParams(max_tokens=256))

# One attempt.
async with await task.build(runtime) as session:
    episode = await session.run(sampling_client)
print(episode.reward, episode.stop_reason)

# Eight attempts at once, as the RL loop runs a group.
episodes = await rollout([task] * 8, runtime, sampling_client)

# One attempt at each held-out task, as `eval/` metrics. Nothing it samples trains.
metrics = await evaluate([task], runtime, sampling_client)
```

To let the model call tools, give the task an environment and more turns:
`environment=BashEnvironment.Config(sandbox=...)` and `max_turns=4`.

The [Episode sessions](../../docs/episode-sessions.md) page has the rest:
[an episode driven by hand](../../docs/episode-sessions.md#run-an-episode)
with `start`, `observe` and `finish`, which needs no server, the two token
budgets, and
[a session of your own](../../docs/episode-sessions.md#a-session-of-your-own),
whose config subclasses `EpisodeSession.Config`.

The cookbook has a word for each level:

| Level | What it is |
|---|---|
| A training loop, `train_rl` | One optimizer step per batch of tasks |
| A rollout | Sessions run on a sampler: a group of them for each task of a step, or one for each held-out task of an evaluation |
| A session, `AgentSession` | One attempt at one task: a model takes turns and calls tools until it answers |

| File | What it holds |
|---|---|
| `base.py` | `EpisodeSession`, the contract: `run(sampling_client)` and `close`, built through its `Config` with the `Runtime` every task of a run shares. `EpisodeSession.Config` is a task, with the fields every session's task has: its messages, verifier, environment, `max_turns` and `max_episode_tokens`. `Episode`, what every session returns, with `Turn` and `StopReason`; `episode.datum(...)` is the episode as one datum. `rollout` runs many sessions at once, as the RL loops do for a step's groups. `evaluate` rolls out one attempt at each of a list of tasks, and returns their statistics as `eval/` metrics. `episode_statistics` is a list of episodes in numbers: the mean and standard error of their rewards and metrics, and the share each stop reason ended |
| `agent_session.py` | `AgentSession`: its `Config` is the contract's, with no field of its own. `start` and `observe` say what to sample next, each tool call of a sampled turn runs in the environment, the prompt continues with the replies, and `finish` returns the episode, scored by the verifier |
| `unit_tests/` | The contract, and the agent session driven turn by turn with turns written by hand. Whole episodes over real processes are `tests/integration/episode_sessions/`'s, and episodes sampled from a model `tests/e2e/test_agent_session.py`'s |

An episode has several turns, so its tokens have three limits. Each has one
name:

| Limit | Its name | What it counts |
|---|---|---|
| A turn's budget | `SamplingParams.max_tokens`, in the session's `Runtime.params` | The tokens the sampler returns for one turn. Every task's turns share it |
| An episode's budget | `EpisodeSession.Config.max_episode_tokens` | The first prompt, every sampled turn and every tool reply together. Each task sets its own |
| The sampler's context | `max_seq_len`, on the run and its sampler | The longest sequence the service holds. It must hold the episode's budget |

An episode's `stop_reason` says which ended it: `answer`, `max_turns`,
`max_turn_tokens` when a turn was cut off at the turn's budget, and
`max_episode_tokens` when the episode's budget ran out, within a turn or
between two.

Rules:

- **A task is a session's config.** It is data, and `build(runtime)` makes
  a fresh session for one attempt. Whoever builds a session closes it.
- **Every task has the same fields,** which `EpisodeSession.Config`
  defines and checks. A session's own config subclasses it, writes
  `build`, and adds the fields only that session needs. A session keeps
  its task's limits, `max_turns` and `max_episode_tokens`.
- **A session runs once,** and samples from the client `run` is handed. So
  whoever runs the sessions decides which weights they sample.
- **Every session returns an `Episode`,** built from its `Turn`s: the
  tokens the model read, then the sequence it sampled. A session never
  writes a sampled turn again, so an episode is one datum whose log
  probabilities are those its sampler recorded.
- **A turn's budget and an episode's are two settings.** A turn's request
  carries the smaller of the turn's budget and what the episode has left.
  An episode whose last turn was cut off, by either, is not verified: it
  scores 0, and its `truncated` metric is 1.
- **A session imports contracts alone:** an environment, a verifier and a
  message formatter by their `base.py`. It runs any of them.
- **A harness of the agent session's kind is a subclass of `AgentSession`** that
  overrides `respond`, with a `Config` of its own that builds it. One
  without is refused when it is defined.
- **A new kind of session is a module here:** a subclass of
  `EpisodeSession` with `run` and `close`, its config, a subclass of
  `EpisodeSession.Config`, and its unit tests.
