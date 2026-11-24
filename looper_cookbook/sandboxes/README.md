# sandboxes/

Sandboxes: where commands run. A sandbox runs shell
commands in a working directory of its own, until it is closed. An
[environment](../environments/README.md) runs its tools' commands in one,
and a verifier its checks.
[Sandboxes](../../docs/sandboxes.md) is the page for users.

## Usage

`LocalSandbox` runs commands as processes of this machine. It needs nothing
installed and isolates nothing, so use it for code you trust:

```python
from looper_cookbook.sandboxes.local import LocalSandbox

sandbox = await LocalSandbox.Config().build()
await sandbox.write_file("answer.py", "print(6 * 7)")
result = await sandbox.run("python3 answer.py", timeout_s=10)
print(result.exit_code, result.stdout)  # 0 42
await sandbox.close()
```

`DockerSandbox` runs them in a container with no network. It needs Docker.
Run what a model writes here:

```python
from looper_cookbook.sandboxes.docker import DockerSandbox

sandbox = await DockerSandbox.Config(image="python:3.12-slim", memory_mb=512).build()
result = await sandbox.run("python3 --version", timeout_s=30)
print(result.stdout)  # Python 3.12.14
await sandbox.close()
```

The [Sandboxes](../../docs/sandboxes.md) page has every setting, what a
command returns when it fails or runs out of time, and
[a sandbox of your own](../../docs/sandboxes.md#a-sandbox-of-your-own).

| File | What it holds |
|---|---|
| `base.py` | `Sandbox`, the contract: `run` and `close`, with `write_file` through `run`, built through its `Config`. `CommandResult`, what a command did, and `SandboxError`, a sandbox that cannot run one |
| `local.py` | `LocalSandbox`: processes of this machine in a temporary directory, with no isolation |
| `docker.py` | `DockerSandbox`: one container of this machine's Docker daemon, through the `docker` program |
| `process.py` | `run_process`, a program of this machine run to its end or its deadline. The two sandboxes above share it; a provider reached over a network has no use for it |
| `unit_tests/` | The contract, and the Docker sandbox's commands with no Docker. `scripted.py` is a sandbox that runs nothing, for the unit tests of whatever runs in one. Real processes and containers are `tests/integration/sandboxes/`'s |

Rules:

- **A sandbox knows nothing of models, tools or environments.** It imports
  nothing else of the cookbook, so an environment, a verifier and your own
  code use it alike.
- **One sandbox serves whoever built it.** `Config.build()` starts a fresh
  one, and whoever builds it closes it. An environment built over a
  sandbox owns it and closes it.
- **A command's failure is a result; the sandbox's own is an exception.**
  A command that fails or runs out of time returns a `CommandResult`. A
  sandbox that cannot run commands raises `SandboxError`, whose `retryable`
  says whether a new sandbox may do better.
- **A sandbox keeps a deadline for everything a command started.** At
  `timeout_s` the command and its children are killed, and the result has
  no exit code.
- **A new provider is a module here:** a subclass of `Sandbox` with `run`
  and `close`, its unit tests, and its tests against the real provider in
  `tests/integration/sandboxes/`.
