# Sandboxes

A sandbox is where commands run. It runs shell commands in a working directory of its own, until it is closed. An [environment](environments.md) runs its tools' commands in one, and a [verifier](verifiers.md) can run its checks in one.

The cookbook has two sandboxes in `looper_cookbook.sandboxes`:

| Sandbox | Where commands run | Settings |
|---|---|---|
| `local.LocalSandbox` | As processes of your machine, in a temporary directory | None |
| `docker.DockerSandbox` | In one container of your machine's Docker daemon | `image`, `network`, `memory_mb`, `cpus`, `lifetime_s` |

## Start a sandbox

Build a sandbox from its config, run commands in it, and close it when you are done:

```python
from looper_cookbook.sandboxes.docker import DockerSandbox

sandbox = await DockerSandbox.Config(image="python:3.12-slim").build()
result = await sandbox.run("python3 --version; pwd", timeout_s=30)
print(result.exit_code, repr(result.stdout))
```

```output
0 'Python 3.12.14\n/workspace\n'
```

`run(command, *, timeout_s, stdin=None)` returns a `CommandResult`:

| Field | What it holds |
|---|---|
| `exit_code` | The command's exit status, or `None` when it ran past `timeout_s` and was killed |
| `stdout`, `stderr` | What it wrote to each stream, as text |
| `truncated` | Whether a stream was longer than the 1 MiB a sandbox keeps of each |
| `timed_out` | True when `exit_code` is `None` |

A command that runs past `timeout_s` is killed, with everything it started:

```python
late = await sandbox.run("sleep 60", timeout_s=1)
print(late.exit_code, late.timed_out)
```

```output
None True
```

`write_file(path, text)` writes a file under the working directory, and files stay between commands:

```python
await sandbox.write_file("notes/answer.txt", "42\n")
result = await sandbox.run("cat notes/answer.txt", timeout_s=10)
print(repr(result.stdout))
await sandbox.close()
```

```output
'42\n'
```

Closing a Docker sandbox removes its container. `async with await config.build() as sandbox:` closes it for you.

## The local sandbox

`LocalSandbox` runs each command as a process of your machine, in a temporary directory that is removed when it closes. It starts at once and needs nothing installed:

```python
from looper_cookbook.sandboxes.local import LocalSandbox

async with await LocalSandbox.Config().build() as local:
    result = await local.run("echo $((6 * 7))", timeout_s=10)
    print(repr(result.stdout))
```

```output
'42\n'
```

- **It isolates nothing.** A command runs as you, and can read and change whatever you can. Use it for code you trust and for writing an environment. Run what a model writes in a Docker sandbox.
- **A command gets your `PATH` and none of your other environment variables,** so an API key in them stays out of what it prints.
- **A process left running in the background outlives it.** A Docker sandbox ends such a process with its container.

## The Docker sandbox

`DockerSandbox` starts one container, runs each command in it, and removes it when it closes. It drives the `docker` program, so it needs Docker installed and nothing else.

- **No network.** The container has none unless you set `network=True`.
- **Limits.** `memory_mb` and `cpus` bound what one sandbox takes from your machine. A command that passes its memory is killed.
- **A lifetime.** The container removes itself `lifetime_s` seconds after it starts, an hour by default, so a run that crashed leaves none behind. Set it longer than your longest episode.
- **The image** needs `sh`, `timeout`, `cat` and `mkdir`, which Debian's and Alpine's images have. Docker pulls an image your machine lacks when the first sandbox starts.

## When a sandbox fails

A command that fails is a result: read its `exit_code` and its `stderr`. A sandbox that cannot run commands at all raises `SandboxError`, because its container died or Docker is not running. Its `retryable` says whether a new sandbox may do better:

| Failure | `retryable` |
|---|---|
| The container was killed, or its lifetime ended | True |
| The image does not exist, or Docker is not running | False |
| The sandbox is closed | False |

## A sandbox of your own

A sandbox is two methods. To run on another provider, such as a hosted sandbox service or your own cluster, subclass `Sandbox`:

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

`run` keeps four promises, which every environment relies on:

- It runs the command with `sh -c`, in the sandbox's working directory, with `stdin` as its standard input.
- At `timeout_s` it kills the command and everything the command started, and returns a result with no exit code.
- It returns a command's own failure as a result, with its exit code.
- It raises `SandboxError` when the sandbox itself failed, with `retryable` set when a new sandbox may do better.

`write_file` goes through `run`, so your sandbox needs no file transfer of its own. Override it if your provider has a faster one. Every environment then runs on your sandbox: pass its config wherever these pages pass `DockerSandbox.Config`.

## Next steps

- [Environments](environments.md) run a model's tool calls in a sandbox.
- [Verifiers](verifiers.md) can run their checks in one.
