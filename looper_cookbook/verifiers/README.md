# verifiers/

Verifiers: what scores a finished episode. A verifier is
one call, `verify(messages, environment)`, which returns a `Score`: the
episode's reward, and counts for the log. An
[episode session](../episode_sessions/README.md) calls it once, when its
episode has ended.
[Verifiers](../../docs/verifiers.md) is the page for users.

## Usage

`FinalNumber` gives 1 when the number after the reply's last `####` is the
answer:

```python
from looper_cookbook.verifiers.final_number import FinalNumber

verifier = await FinalNumber.Config(answer="42").build()
messages = [
    {"role": "user", "content": "What is 6 times 7? End on '#### ' and the number."},
    {"role": "assistant", "content": "6 times 7 is 42.\n#### 42"},
]
score = await verifier.verify(messages, None)
print(score.reward)  # 1.0
```

`CodeTestsVerifier` gives 1 when the reply's last code block passes every
test, in a sandbox it starts for the check. Here it needs Docker:

```python
from looper_cookbook.environments.coding import CodeTest
from looper_cookbook.sandboxes.docker import DockerSandbox
from looper_cookbook.verifiers.code_tests import CodeTestsVerifier

verifier = await CodeTestsVerifier.Config(
    sandbox=DockerSandbox.Config(image="python:3.12-slim"),
    tests=[CodeTest(input="1 2\n", output="3\n")],
).build()
messages = [
    {"role": "user", "content": "Print the sum of the numbers on standard input."},
    {"role": "assistant", "content": "```python\nprint(sum(map(int, input().split())))\n```"},
]
score = await verifier.verify(messages, None)
print(score.reward)  # 1.0
```

The [Verifiers](../../docs/verifiers.md) page has how each verifier reads a
reply, and
[a verifier of your own](../../docs/verifiers.md#a-verifier-of-your-own).

| File | What it holds |
|---|---|
| `base.py` | `Verifier`, the contract: `verify`, built through its `Config`. `Score`, a reward and its metrics. `last_reply` reads a conversation's last assistant message |
| `final_number.py` | `FinalNumber`: reward 1 when the number after the reply's last `####` is the answer, as GSM8K writes its answers |
| `code_tests.py` | `CodeTestsVerifier`: reward 1 when the reply's last code block passes a problem's hidden tests, in a sandbox of the verifier's own. The `code_rl` recipe's reward |
| `unit_tests/` | Each verifier's scores, over conversations written by hand and a sandbox that runs nothing. A verifier over real processes is `tests/integration/episode_sessions/`'s |

Rules:

- **A verifier reads; it does not act for the model.** It is lent the
  episode's environment, still open, and may run its own checks in
  `environment.sandbox`, where the model worked. It neither closes the
  environment nor keeps it.
- **The reference stays with the verifier,** in this process: the right
  answer, or the tests that decide a reward. No prompt, tool or
  observation carries it.
- **A failure is never a reward.** A verifier that cannot do its check
  raises, as a `SandboxError` when its sandbox fails. It returns a low
  score only for an episode it did check.
- **Building a verifier starts nothing.** A session builds its verifier
  with itself, and an episode that was cut off never calls it. So a sandbox
  starts inside `verify`, and is closed there.
- **Only the reply counts.** A verifier reads the last assistant message's
  content, never its reasoning: a model may try answers while it reasons.
- **A new verifier is a module here:** a subclass of `Verifier` with
  `verify`, and its unit tests.
