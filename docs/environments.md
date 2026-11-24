# Environments

An environment is where a model's actions run. It offers tools, runs one tool call at a time, and returns what the model reads next. It knows nothing of models, tokens or rewards, so one environment serves any model, in training and in an evaluation alike.

An environment that runs commands runs them in a [sandbox](sandboxes.md): a Docker container, or a directory on your machine for code you trust. The cookbook has two example environments in `looper_cookbook.environments`: a shell, and a checker for programming problems.

An [episode session](episode-sessions.md) is what calls an environment during training: it asks the model for a turn, runs the turn's tool calls here, and continues the prompt with their replies. This page calls an environment by hand.

## A shell: `BashEnvironment`

`BashEnvironment` offers one tool, `bash`. It is the environment of an agent that works in a terminal. An environment is built from its config, which holds its sandbox's config:

```python
from looper_cookbook.environments.base import Action
from looper_cookbook.environments.bash import BashEnvironment
from looper_cookbook.sandboxes.docker import DockerSandbox

config = BashEnvironment.Config(
    sandbox=DockerSandbox.Config(image="python:3.12-slim"), timeout_s=30
)
environment = await config.build()
for command in ("echo 'print(6 * 7)' > answer.py", "python3 answer.py", "python3 missing.py"):
    observation = await environment.step(Action(name="bash", arguments={"command": command}))
    print(repr(observation.content))
```

```output
'[no output]'
'42\n'
"python3: can't open file '/workspace/missing.py': [Errno 2] No such file or directory\n[exit code 2]"
```

- **An `Action` is one tool call:** the tool's name and its arguments.
- **An `Observation` is the tool's reply:** `content`, the text the model reads, and `metrics`, counts for your log. `bash` counts the commands that `failed` and that `timed_out`.
- **Each command runs in a new shell,** in the sandbox's working directory. Files stay between commands. A `cd` and a shell variable do not, and the tool's description tells the model so.
- **A command returns once everything it started has closed its output.** A process left running in the background, such as a server to test, sends its output to a file: `server > server.log 2>&1 &`. The tool's description tells the model that too.
- **The model reads output as a terminal shows it,** standard error mixed into standard output. Output over `max_output_chars`, 10,000 by default, keeps its start and its end around a note of how much was left out. A status line follows a command that failed or ran past `timeout_s`.

## Run a model's tool calls

An environment's `tools` are JSON schemas, as a [message formatter](message-formatting.md#tools) lists them in a prompt. The formatter reads a sampled turn's calls back, each call is an action, and each observation is a tool message that continues the prompt:

```python
from transformers import AutoTokenizer

from looper_cookbook.message_formatters.base import call_arguments
from looper_cookbook.message_formatters.models import message_formatter

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
formatter = message_formatter("Qwen/Qwen3.5-9B", tokenizer, enable_thinking=False)

messages = [{"role": "user", "content": "Run answer.py and tell me what it prints."}]
prompt = formatter.build_generation_prompt(messages, environment.tools)

# `sampled` is the turn your sampler returned for the prompt.
reply, ended = formatter.parse_response(sampled, environment.tools)
print(reply["tool_calls"])
replies = []
for call in reply["tool_calls"]:
    name, arguments = call_arguments(call)
    observation = await environment.step(Action(name=name, arguments=arguments))
    replies.append({"role": "tool", "name": name, "content": observation.content})
# The next prompt: every token so far, as sampled, then the replies.
prompt = formatter.continue_prompt([*prompt, *sampled], replies)
print(repr(tokenizer.decode(prompt)[-104:]))
```

```output
[{'type': 'function', 'function': {'name': 'bash', 'arguments': {'command': 'python3 answer.py'}}}]
'<|im_start|>user\n<tool_response>\n42\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

**A call the model got wrong is answered, never raised.** `step` checks every action against the tools before it runs: the tool must exist, every required argument must be there, and each argument must have the type its tool declares. A call that fails the check is not run. Its observation says what was wrong, so the model can correct it, and counts as an `invalid_action`:

```python
observation = await environment.step(Action(name="bash", arguments={"cmd": "ls"}))
print(repr(observation.content), dict(observation.metrics))
await environment.close()
```

```output
'Error: bash was called without its required command.' {'invalid_action': 1.0}
```

Close an environment when its episode ends, which closes its sandbox. `async with` does it for you, as the next example shows.

## Check a program: `CodingEnvironment`

`CodingEnvironment` is the environment of a programming problem, the one the `code_rl` recipe trains with on DeepCoder's problems. It offers one tool, `check_solution`. The model sends a whole Python program, the environment runs it against the problem's example tests in its sandbox, and the reply says how many passed and shows the first that failed:

```python
from looper_cookbook.environments.coding import CodeTest, CodingEnvironment

examples = [
    CodeTest(input="1 2\n", output="3\n"),
    CodeTest(input="10 20 30\n", output="60\n"),
]
config = CodingEnvironment.Config(
    sandbox=DockerSandbox.Config(image="python:3.12-slim", memory_mb=512), tests=examples
)
async with await config.build() as environment:
    for code in (
        "a, b = map(int, input().split())\nprint(a + b)",
        "print(sum(map(int, input().split())))",
    ):
        observation = await environment.step(
            Action(name="check_solution", arguments={"code": code})
        )
        print(observation.content)
```

```output
{"passed": false, "tests_passed": 1, "tests": 2, "first_failure": {"input": "10 20 30\n", "expected": "60\n", "got": "", "error": "Traceback (most recent call last):\n  File \"/workspace/solution.py\", line 1, in <module>\n    a, b = map(int, input().split())\n    ^^^^\nValueError: too many values to unpack (expected 2)"}}
{"passed": true, "tests_passed": 2, "tests": 2}
```

A test is one of two kinds:

| Kind | `input` | `output` | Passes when |
|---|---|---|---|
| A program test | What the program reads on standard input | What it must print | The printed tokens equal the expected ones, in order, wherever the lines break. Numbers that are not both integers may differ by 1e-6 |
| A function test, with `function` set | The call's arguments, a JSON value a line | The JSON value it must return | The function returns an equal value. It is a method of the code's `Solution` class when the code defines one, as LeetCode's problems do, else a function of the module |

A test also fails when the program exits with a status other than 0, or runs past `timeout_s`, 6 seconds a test by default. A check stops at the first test that fails.

`prelude` is code written before every program, such as the imports a problem set lets its solutions assume: LeetCode's solutions use `List` and `defaultdict` without importing them.

**Give the tool only tests the model may read.** `check_solution` shows a failing test whole: its input, the output expected and the output got. So list a problem's examples in the environment's config, and keep the tests that decide a reward out of it. Run those with `run_tests`, in a sandbox the model never acted in, which is what a [`CodeTestsVerifier`](verifiers.md#a-program-codetestsverifier) does when an episode ends:

```python
from looper_cookbook.environments.coding import run_tests

hidden = [
    CodeTest(input="[3, 1, 2]\n2", output="[1, 2]", function="smallest"),
    CodeTest(input="[5, 4]\n1", output="[4]", function="smallest"),
]
code = "class Solution:\n    def smallest(self, numbers, k):\n        return sorted(numbers)[:k]\n"
async with await DockerSandbox.Config(image="python:3.12-slim").build() as sandbox:
    report = await run_tests(sandbox, code, hidden, timeout_s=6)
print(report.passed, report.total, report.all_passed)
```

```output
2 2 True
```

**A program cannot read the output a test expects.** The sandbox gets the program and a test's input, and returns what the program printed. The comparison runs in your process, where the expected output stays.

## An environment of your own

Subclass `Environment` and write two things: `tools`, the JSON schema of each action, made with `tool`; and `execute`, which runs one action and returns its observation. `step` has already checked the action, so `execute` sees only a tool you offer, with its required arguments, each of the type you declared. An environment needs no sandbox. This one answers from your process:

```python
import dataclasses

from looper_cookbook.environments.base import Action, Environment, Observation, tool


class Inventory(Environment):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Config:
        stock: dict[str, int]

        async def build(self) -> "Inventory":
            return Inventory(self.stock)

    def __init__(self, stock: dict[str, int]) -> None:
        self._stock = dict(stock)

    @property
    def tools(self):
        item = {"type": "string", "description": "The item's name."}
        return (tool("count", "How many of an item are in stock.", {"item": item}, required=("item",)),)

    async def execute(self, action: Action) -> Observation:
        return Observation(content=str(self._stock.get(action.arguments["item"], 0)))


async with await Inventory.Config(stock={"bolt": 40}).build() as environment:
    for arguments in ({"item": "bolt"}, {"item": 7}):
        observation = await environment.step(Action(name="count", arguments=arguments))
        print(repr(observation.content))
```

```output
'40'
"Error: count's argument item must be of type string, not 7."
```

- **Build it through its `Config`.** A config is data, so a task can carry its environment's config and build a fresh environment for every attempt. One environment serves one episode.
- **Hold a sandbox when you run commands.** Give the config a `sandbox` field, build the sandbox in `build`, and return it from the `sandbox` property, as `BashEnvironment` does. Closing the environment then closes its sandbox.
- **Answer a model's mistake, raise your own failure.** Return an observation for what the model can correct. Raise when the environment itself cannot work, so a failure is never read as a reply.

## Next steps

- [Sandboxes](sandboxes.md) are where an environment's commands run, and how to write one for another provider.
- [Episode sessions](episode-sessions.md) call an environment for you, turn after turn.
- [Verifiers](verifiers.md) score what a model did, in a sandbox of their own or in the environment's.
- [Message formatting](message-formatting.md) writes an environment's tools into a prompt, reads a model's calls back and continues a prompt with their replies.
