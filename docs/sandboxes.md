# Sandboxes

A sandbox is a place to run shell commands, with a working folder of its own that lasts until the sandbox is closed. Model-written code runs in one, so it can't touch the rest of your machine. An [environment](environments.md) runs its tools' commands in a sandbox, and a [verifier](verifiers.md) can run its checks in one, such as running the model's program against hidden tests.

You usually don't call a sandbox yourself. You put a sandbox's config in an environment's or a verifier's config, and they start and close it for you. Read this page to choose a sandbox and its settings, or to write one for another provider.

This page covers:

- [Start a sandbox](#start-a-sandbox): run commands and write files in a sandbox.
- [The local sandbox](#the-local-sandbox): run commands on your machine, for code you trust.
- [The Docker sandbox](#the-docker-sandbox): run commands in a container, for code a model wrote.
- [When a sandbox fails](#when-a-sandbox-fails): a failed command, and a sandbox that can't run commands at all.
- [A sandbox of your own](#a-sandbox-of-your-own): run commands at another provider.

The cookbook comes with two sandboxes, in `looper_cookbook.sandboxes`:

| Sandbox | Where commands run | Use it for | Settings |
|---|---|---|---|
| `local.LocalSandbox` | As ordinary processes on your machine, in a temporary folder | Code you trust, and trying out an environment you're writing | None |
| `docker.DockerSandbox` | In a Docker container on your machine | Code the model wrote | `image`, `network`, `memory_mb`, `cpus`, `lifetime_s` |

## Start a sandbox

Create a sandbox from its config, run commands in it, and close it when you're done. This one starts a Docker container from the `python:3.12-slim` image, and runs a command that prints the Python version and the working folder:

```python
from looper_cookbook.sandboxes.docker import DockerSandbox

sandbox = await DockerSandbox.Config(image="python:3.12-slim").build()
result = await sandbox.run("python3 --version; pwd", timeout_s=30)
print(result.exit_code, repr(result.stdout))
```

```output
0 'Python 3.12.14\n/workspace\n'
```

`run(command, *, timeout_s, stdin=None)` runs one command and returns a `CommandResult`:

| Field | What it holds |
|---|---|
| `exit_code` | The command's exit status, which is 0 if it succeeded, or `None` if it ran longer than `timeout_s` and was stopped |
| `stdout`, `stderr` | What the command printed to its normal output and to its error output, as text |
| `truncated` | Whether either output was longer than 1 MiB, the most a sandbox keeps of each |
| `timed_out` | `True` when `exit_code` is `None` |

A command that runs longer than `timeout_s` is stopped, along with everything it started:

```python
late = await sandbox.run("sleep 60", timeout_s=1)
print(late.exit_code, late.timed_out)
```

```output
None True
```

`write_file(path, text)` writes a file inside the working folder. Files stay between commands, so a later command can read it:

```python
await sandbox.write_file("notes/answer.txt", "42\n")
result = await sandbox.run("cat notes/answer.txt", timeout_s=10)
print(repr(result.stdout))
await sandbox.close()
```

```output
'42\n'
```

Closing a Docker sandbox removes its container. To close a sandbox for you, even if your code fails partway, use `async with await config.build() as sandbox:`.

## The local sandbox

`LocalSandbox` runs each command as an ordinary process on your machine, in a temporary folder that's deleted when the sandbox closes. It starts straight away, and needs nothing installed:

```python
from looper_cookbook.sandboxes.local import LocalSandbox

async with await LocalSandbox.Config().build() as local:
    result = await local.run("echo $((6 * 7))", timeout_s=10)
    print(repr(result.stdout))
```

```output
'42\n'
```

- **It doesn't protect your machine.** A command runs as you, and can read and change anything you can. Use it for code you trust, and for trying out an environment you're writing. Run code a model wrote in a Docker sandbox.
- **Commands get your `PATH`, and none of your other environment variables.** So an API key you've set there can't end up in what a command prints.
- **A process left running in the background keeps running after the sandbox closes.** A Docker sandbox stops such processes when it removes its container.

## The Docker sandbox

`DockerSandbox` starts one container, runs each command in it, and removes the container when it closes. It runs the `docker` program, so it only needs Docker installed. It has these settings:

- **`network`:** the container has no network access, unless you set `network=True`.
- **`memory_mb` and `cpus`:** limit how much of your machine one sandbox can use. A command that goes over the memory limit is stopped.
- **`lifetime_s`:** the container removes itself this many seconds after it starts, an hour by default. That way, a run that crashed doesn't leave containers behind. Set it longer than your longest attempt.
- **`image`:** the Docker image to start from. It needs `sh`, `timeout`, `cat` and `mkdir`, which Debian's and Alpine's images have. If your machine doesn't have the image, Docker downloads it when the first sandbox starts.

## When a sandbox fails

There are two kinds of failure:

- **A command that fails** is a normal result. Read its `exit_code` and its `stderr`, as you would in a terminal.
- **A sandbox that can't run commands at all,** because its container stopped or Docker isn't running, raises a `SandboxError`. Its `retryable` says whether starting a new sandbox might work:

| Failure | `retryable` |
|---|---|
| The container was stopped, or its lifetime ended | `True` |
| The image doesn't exist, or Docker isn't running | `False` |
| The sandbox was already closed | `False` |

## A sandbox of your own

To run commands at another provider, such as a hosted sandbox service or your own cluster, subclass `Sandbox`. You write a config that starts one sandbox at your provider, and two methods: `run`, which runs one command, and `close`, which stops the sandbox:

```python
import dataclasses

from looper_cookbook.sandboxes.base import CommandResult, Sandbox


class ProviderSandbox(Sandbox):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Config:
        image: str

        async def build(self) -> "ProviderSandbox":
            ...  # start one sandbox at your provider, and return it

    async def run(
        self, command: str, *, timeout_s: float, stdin: str | None = None
    ) -> CommandResult:
        ...  # run `sh -c command` there, and return its exit code and its output

    async def close(self) -> None:
        ...  # stop it; a second call does nothing
```

Every environment relies on `run` to behave the same way, so yours must:

- **Run the command with `sh -c`,** in the sandbox's working folder, with `stdin` as its input.
- **Stop the command at `timeout_s`,** along with everything it started, and return a result with no exit code.
- **Return a failed command as a result,** with its exit code. Don't raise for it.
- **Raise `SandboxError` when the sandbox itself fails,** with `retryable` set if a new sandbox might work.

You don't need to write `write_file`, because it works through `run`. Override it if your provider has a faster way to send files.

Every environment and verifier can then use your sandbox. Pass its config wherever these pages pass `DockerSandbox.Config`.

## Next steps

- [Environments](environments.md) run the model's tool calls in a sandbox.
- [Verifiers](verifiers.md) can run their checks in one.
