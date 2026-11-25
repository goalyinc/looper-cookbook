# Environments

An environment gives the model tools to use during an attempt, such as a shell to run commands in. When the model calls a tool, the environment runs it and returns the reply, which the model reads in its next turn. An environment knows nothing about models, tokens or rewards, so the same environment works with any model, in training and in evaluation.

You only need an environment for tasks where the model uses tools. For a question the model answers in one turn, such as "What is 12 times 7?", leave the task's `environment` out.

During training, an [episode session](episode-sessions.md) calls the environment for you. It asks the model for a turn, runs the turn's tool calls in the environment, and adds their replies to the conversation. This page calls environments by hand, so you can see what they do.

This page covers:

- [Words used on this page](#words-used-on-this-page): tool, action, observation and sandbox.
- [A shell: `BashEnvironment`](#a-shell-bashenvironment): let the model run shell commands.
- [Run a model's tool calls](#run-a-models-tool-calls): how a tool call goes from the model's turn to the environment and back.
- [Check a program: `CodingEnvironment`](#check-a-program-codingenvironment): let the model test its code against example tests.
- [An environment of your own](#an-environment-of-your-own): write your own tools.

## Words used on this page

| Word | What it means | Example |
|---|---|---|
| **Tool** | Something the model can call, described to it in the prompt by a name, a description and its arguments | `bash`, which takes a `command` |
| **Action** | One tool call, made of the tool's name and its arguments | `bash` with the command `python3 answer.py` |
| **Observation** | The tool's reply, which is the text the model reads next, plus numbers for your logs | `'42\n'`, what the command printed |
| **Sandbox** | Where an environment's commands run, either in a Docker container or, for code you trust, in a folder on your machine. See [Sandboxes](sandboxes.md) | A `python:3.12-slim` container |

The cookbook comes with two environments, in `looper_cookbook.environments`. `BashEnvironment` is a shell, and `CodingEnvironment` checks programs for programming problems.

## A shell: `BashEnvironment`

`BashEnvironment` gives the model one tool, `bash`, which runs a shell command. Use it for a model that works in a terminal.

You create an environment from its config, as with verifiers. The config holds the settings for its sandbox, where the commands run. The code below creates one in a Docker container, then runs three commands the way the model might. They write a small program, run it, and then try to run a file that doesn't exist:

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

The first command prints nothing, the second prints the program's output, and the third shows the error and the exit code, just as a terminal would.

- **An `Action` is one tool call,** with the tool's name and its arguments.
- **An `Observation` is the tool's reply.** It has `content`, the text the model reads, and `metrics`, numbers for your logs. `bash` counts the commands that `failed`, and the ones that `timed_out`.
- **Each command runs in a new shell,** in the sandbox's working folder. Files stay between commands, but a `cd` or a shell variable doesn't. The tool's description tells the model this.
- **A command returns once everything it started has finished writing output.** To leave a process running in the background, such as a server to test against, send its output to a file, as in `server > server.log 2>&1 &`. The tool's description tells the model this too.
- **The model sees output as a terminal shows it,** with errors mixed into the normal output. Output longer than `max_output_chars`, 10,000 characters by default, keeps its start and its end, with a note in between saying how much was left out. A status line is added after a command that failed, or that ran longer than `timeout_s`.

## Run a model's tool calls

This is what an [episode session](episode-sessions.md) does for you during an attempt, written out by hand:

1. **List the tools in the prompt.** An environment's `tools` are written as JSON schemas, which the [message formatter](message-formatting.md#tools) puts into the prompt.
2. **Read the model's tool calls.** After the sampler returns the model's turn, the formatter reads the tool calls out of it.
3. **Run each call.** Each call becomes an `Action`, and the environment runs it.
4. **Continue the prompt.** Each `Observation` becomes a tool message, and the formatter adds them to the prompt for the model's next turn.

Here, the model asked to run `answer.py`, from the example above:

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

The first line is the tool call the formatter read from the model's turn. The second is the end of the next prompt, with the tool's reply, `42`, followed by the start of the model's next turn.

### When the model makes a mistake

**A tool call the model got wrong gets a reply, not an error.** Before running an action, `step` checks it against the tools. The tool must exist, every required argument must be there, and each argument must have the type the tool expects. A call that fails the check isn't run. Its reply says what was wrong, so the model can fix it, and its metrics count it as an `invalid_action`. Here, the model wrote `cmd` instead of `command`:

```python
observation = await environment.step(Action(name="bash", arguments={"cmd": "ls"}))
print(repr(observation.content), dict(observation.metrics))
await environment.close()
```

```output
'Error: bash was called without its required command.' {'invalid_action': 1.0}
```

Close an environment when the attempt ends. That also closes its sandbox. `async with` closes it for you, as the next example shows.

## Check a program: `CodingEnvironment`

`CodingEnvironment` is for programming problems. It's the environment the `code_rl` recipe trains with, on DeepCoder's problems. It gives the model one tool, `check_solution`. The model sends a whole Python program, the environment runs it against the problem's example tests in its sandbox, and the reply says how many tests passed and shows the first one that failed.

The example below is the problem "print the sum of the numbers on standard input", with two example tests. The model's first program only handles two numbers, so it fails the second test. Its second program passes both:

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

Each test is one of two kinds:

| Kind | `input` | `output` | Passes when |
|---|---|---|---|
| A program test | What the program reads on standard input | What it must print | What it prints matches the expected output word for word, in order, ignoring where the lines break. Numbers that aren't both whole numbers may differ by up to 0.000001 |
| A function test, with `function` set | The function's arguments, one JSON value per line | The JSON value it must return | The function returns an equal value. If the code defines a `Solution` class, as LeetCode's problems do, the function is a method of it. Otherwise it's a function in the module |

A test also fails if the program exits with an error status, or runs longer than `timeout_s`, 6 seconds per test by default. A check stops at the first test that fails.

`prelude` is code added before every program, such as the imports a problem set lets its solutions assume. For example, LeetCode's solutions use `List` and `defaultdict` without importing them.

### Keep the reward's tests hidden

**Only give the tool tests the model is allowed to see.** `check_solution` shows a failing test in full, including its input, the expected output, and what the program printed. So put a problem's example tests in the environment's config, and keep the tests that decide the reward out of it.

Run those hidden tests with `run_tests`, in a sandbox the model never worked in. That's what a [`CodeTestsVerifier`](verifiers.md#a-program-codetestsverifier) does when an attempt ends. The example below runs two hidden function tests against a `Solution` class:

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

**A program can't read the output a test expects.** The sandbox only gets the program and the test's input, and sends back what the program printed. The comparison with the expected output happens in your program, where the expected output stays.

## An environment of your own

To give the model your own tools, subclass `Environment` and write two things:

- **`tools`:** the description of each tool, made with the `tool` helper.
- **`execute`:** runs one action and returns its `Observation`.

`step` has already checked the action before calling `execute`, so `execute` only ever sees one of your tools, with its required arguments, each of the right type.

An environment doesn't need a sandbox. This one answers from your own program. It gives the model a `count` tool that looks up how many of an item are in stock. The example first asks for bolts, then calls the tool with a number instead of a name, so the type check catches it:

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

- **Create it through its `Config`.** A config is just data, so a task can hold its environment's config, and a fresh environment is built for every attempt. One environment serves one attempt.
- **Hold a sandbox if you run commands.** Give the config a `sandbox` field, start the sandbox in `build`, and return it from the `sandbox` property, as `BashEnvironment` does. Closing the environment then closes its sandbox.
- **Reply to the model's mistakes, but raise your own failures.** Return an observation for anything the model can fix. Raise an error when the environment itself can't work, so a failure is never mistaken for a reply.

To use your environment, put its config in each task's `environment`, as in `environment=Inventory.Config(stock={"bolt": 40})`.

## Next steps

- [Sandboxes](sandboxes.md) explains where an environment's commands run, and how to write a sandbox for another provider.
- [Episode sessions](episode-sessions.md) call an environment for you, turn after turn.
- [Verifiers](verifiers.md) score what the model did, in a sandbox of their own or in the environment's.
- [Message formatting](message-formatting.md) explains how a formatter writes an environment's tools into a prompt, reads the model's tool calls back, and continues the prompt with their replies.
