# environments/

Environments: where a model's actions run. An
environment offers tools, runs one tool call at a time and returns what the
model reads next. It knows nothing of models, tokens or rewards: an episode
session hands it a sampled turn's calls, and a verifier scores the episode.
[Environments](../../docs/environments.md) is the page for users.

## Usage

`BashEnvironment` offers one tool, `bash`. Its commands run in the sandbox
its config names:

```python
from looper_cookbook.environments.base import Action
from looper_cookbook.environments.bash import BashEnvironment
from looper_cookbook.sandboxes.local import LocalSandbox

environment = await BashEnvironment.Config(sandbox=LocalSandbox.Config()).build()
action = Action(name="bash", arguments={"command": "echo $((6 * 7))"})
observation = await environment.step(action)
print(observation.content)  # 42
await environment.close()
```

`CodingEnvironment` offers one tool, `check_solution`, which runs a Python
program against the tests its config lists:

```python
from looper_cookbook.environments.coding import CodeTest, CodingEnvironment

tests = [CodeTest(input="1 2\n", output="3\n")]
environment = await CodingEnvironment.Config(sandbox=LocalSandbox.Config(), tests=tests).build()
action = Action(name="check_solution", arguments={"code": "print(sum(map(int, input().split())))"})
observation = await environment.step(action)
print(observation.content)  # {"passed": true, "tests_passed": 1, "tests": 1}
await environment.close()
```

For what a model writes, pass `DockerSandbox.Config(image="python:3.12-slim")`
as the sandbox. The [Environments](../../docs/environments.md) page has the
rest: a model's tool calls read by a message formatter, the two kinds of
test, and
[an environment of your own](../../docs/environments.md#an-environment-of-your-own).

| File | What it holds |
|---|---|
| `base.py` | `Environment`, the contract: `tools`, `step`, `sandbox` and `close`, built through its `Config`. `Action`, one tool call, and `Observation`, its reply. `tool` writes a tool's JSON schema, and `refusal` says why a call fits none. A subclass writes `tools` and `execute` |
| `bash.py` | `BashEnvironment`: one tool, `bash`, for an agent in a terminal |
| `coding.py` | `CodingEnvironment`: one tool, `check_solution`, which runs a Python program against a problem's tests, as the `code_rl` recipe checks DeepCoder's. `run_tests` is the check itself, for a verifier's hidden tests |
| `unit_tests/` | Each environment over a sandbox that runs nothing (`sandboxes/unit_tests/scripted.py`). Real processes and containers are `tests/integration/environments/`'s |

Rules:

- **An environment scores nothing and samples nothing.** A reward is a
  verifier's, and a turn is an episode session's. So an
  environment here imports no message formatter and nothing of the SDK.
- **One environment serves one episode.** It is built through its
  `Config`. A config is data.
- **A sandbox is an argument of the environment that runs in one.** Its
  `Config` takes a [sandbox](../sandboxes/README.md)'s config, `build()`
  starts that sandbox, and `sandbox` names it. The environment owns it:
  the contract's `close` closes it. An environment that answers from this
  process has none, and writes `tools` and `execute` alone.
- **An environment names the `Sandbox` contract, never a provider.** So
  each one here runs locally, in Docker, and on a provider written later.
- **A model's mistake is an observation; the environment's own failure is
  an exception.** `step` answers a call that fits no tool without running
  it. A command that fails or runs out of time is a reply too. Only a
  sandbox that cannot run commands raises, as a `SandboxError`.
- **A reference never enters a sandbox.** `run_tests` sends a program and
  a test's input, and compares the output in this process. An
  environment's config lists only tests its model may read.
- **A new environment is a module here** with its unit tests over the
  scripted sandbox, a case in the integration tests, and its section of
  the page.
