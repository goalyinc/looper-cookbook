"""The ``docker`` commands a Docker sandbox runs, and how it reads their
results, with no Docker.

``run_process`` is replaced, so these tests start no program. A real
container is ``tests/integration/sandboxes/test_docker.py``'s.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from looper_cookbook.sandboxes.base import CommandResult, SandboxError
from looper_cookbook.sandboxes.docker import DockerSandbox

OK = CommandResult(exit_code=0)


class Docker:
    """Stands in for ``run_process``: records each call and, after ``seconds``,
    answers with the next of its results, or raises it."""

    def __init__(self, *results: CommandResult | BaseException, seconds: float = 0.0) -> None:
        self._results = list(results)
        self._seconds = seconds
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    async def __call__(self, argv: Sequence[str], **settings: Any) -> CommandResult:
        self.calls.append((tuple(argv), settings))
        await asyncio.sleep(self._seconds)
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch):
    def install(*results: CommandResult | BaseException, seconds: float = 0.0) -> Docker:
        fake = Docker(*results, seconds=seconds)
        monkeypatch.setattr("looper_cookbook.sandboxes.docker.run_process", fake)
        return fake

    return install


async def started(docker, *results: CommandResult | BaseException, seconds: float = 0.0):
    """A sandbox built through its config, and the stand-in that answers what
    the test does next with ``results``."""
    docker(OK)
    sandbox = await DockerSandbox.Config(image="python:3.12-slim").build()
    return sandbox, docker(*results, seconds=seconds)


async def test_a_sandbox_starts_a_container_without_network_that_removes_itself(docker) -> None:
    fake = docker(OK)
    sandbox = await DockerSandbox.Config(image="python:3.12-slim").build()
    assert sandbox.name.startswith("looper-sandbox-")
    assert fake.calls == [
        (
            ("docker", "run", "--detach", "--rm", "--init", "--name", sandbox.name)
            + ("--workdir", "/workspace", "--network", "none")
            + ("python:3.12-slim", "sleep", "3600"),
            {"timeout_s": 600.0, "stdin": None},
        )
    ]


async def test_its_config_sets_the_containers_network_limits_and_lifetime(docker) -> None:
    fake = docker(OK)
    config = DockerSandbox.Config(
        image="node:22", network=True, memory_mb=2048, cpus=1.5, lifetime_s=120
    )
    sandbox = await config.build()
    assert fake.calls[0][0] == (
        ("docker", "run", "--detach", "--rm", "--init", "--name", sandbox.name)
        + ("--workdir", "/workspace", "--memory", "2048m", "--cpus", "1.5")
        + ("node:22", "sleep", "120")
    )


@pytest.mark.parametrize(
    "result,why",
    [
        (
            CommandResult(exit_code=125, stderr="Unable to find image 'nope:1' locally\n"),
            "could not start a container of nope:1: Unable to find image 'nope:1' locally. "
            "Check that Docker is running",
        ),
        (CommandResult(exit_code=None), "could not start a container of nope:1: no answer in 600"),
    ],
)
async def test_a_container_that_does_not_start_is_an_error_and_is_removed(
    docker, result: CommandResult, why: str
) -> None:
    fake = docker(result, OK)
    with pytest.raises(SandboxError, match=why) as raised:
        await DockerSandbox.Config(image="nope:1").build()
    assert not raised.value.retryable
    # The daemon may have made the container before it failed.
    name = fake.calls[0][0][6]
    assert fake.calls[1][0] == ("docker", "rm", "--force", name)


async def test_a_start_that_is_cancelled_removes_what_it_may_have_started(docker) -> None:
    fake = docker(asyncio.CancelledError(), OK)
    with pytest.raises(asyncio.CancelledError):
        await DockerSandbox.Config(image="python:3.12-slim").build()
    assert fake.calls[1][0][:3] == ("docker", "rm", "--force")


async def test_a_machine_without_docker_says_what_to_install(docker) -> None:
    docker(FileNotFoundError("docker"), FileNotFoundError("docker"))
    with pytest.raises(SandboxError, match="the docker program is not on PATH") as raised:
        await DockerSandbox.Config(image="python:3.12-slim").build()
    assert not raised.value.retryable


async def test_a_command_runs_in_the_container_under_a_deadline_kept_inside_it(docker) -> None:
    sandbox, fake = await started(docker, CommandResult(exit_code=3, stdout="out", stderr="err"))
    result = await sandbox.run("echo out; exit 3", timeout_s=2.5)
    assert result == CommandResult(exit_code=3, stdout="out", stderr="err")
    assert fake.calls == [
        (
            ("docker", "exec", sandbox.name, "timeout", "-s", "KILL", "2.5")
            + ("sh", "-c", "echo out; exit 3"),
            # The client gets longer than the command, whose deadline the container keeps.
            {"timeout_s": 12.5, "stdin": None},
        )
    ]


async def test_a_command_with_input_keeps_its_standard_input_open(docker) -> None:
    sandbox, fake = await started(docker, OK)
    await sandbox.run("cat", timeout_s=6, stdin="1 2\n")
    argv, settings = fake.calls[0]
    assert argv[:4] == ("docker", "exec", "--interactive", sandbox.name)
    assert argv[4:8] == ("timeout", "-s", "KILL", "6") and settings["stdin"] == "1 2\n"


@pytest.mark.parametrize("status", [124, 137])
async def test_a_command_killed_at_its_deadline_has_no_exit_code(docker, status: int) -> None:
    killed = CommandResult(exit_code=status, stdout="so far\n")
    sandbox, _ = await started(docker, killed, seconds=0.06)
    result = await sandbox.run("sleep 9", timeout_s=0.05)
    assert result == CommandResult(exit_code=None, stdout="so far\n")
    assert result.timed_out


async def test_a_command_killed_before_its_deadline_keeps_its_status(docker) -> None:
    # Out of memory, or killed by its own program: not out of time.
    sandbox, _ = await started(docker, CommandResult(exit_code=137))
    result = await sandbox.run("python big.py", timeout_s=2)
    assert result.exit_code == 137 and not result.timed_out


@pytest.mark.parametrize(
    "stderr,retryable",
    [
        ("Error response from daemon: container abc is not running\n", True),
        ("Error response from daemon: No such container: looper-sandbox-0\n", True),
        ("Cannot connect to the Docker daemon at unix:///var/run/docker.sock.\n", False),
        ('OCI runtime exec failed: exec: "timeout": executable file not found\n', False),
    ],
)
async def test_dockers_own_failure_is_an_error_not_a_result(
    docker, stderr: str, retryable: bool
) -> None:
    sandbox, _ = await started(docker, CommandResult(exit_code=126, stderr=stderr))
    with pytest.raises(SandboxError, match="Docker could not run a command") as raised:
        await sandbox.run("ls", timeout_s=2)
    assert raised.value.retryable is retryable
    assert ("The container is gone" in str(raised.value)) is retryable


async def test_a_commands_own_failure_that_names_docker_is_a_result(docker) -> None:
    # Only how the message starts tells Docker's failure from a program's output.
    stderr = "grep: Error response from daemon: no such file\n"
    sandbox, _ = await started(docker, CommandResult(exit_code=2, stderr=stderr))
    result = await sandbox.run("grep x", timeout_s=2)
    assert result == CommandResult(exit_code=2, stderr=stderr)


async def test_a_client_that_never_returns_retires_the_container(docker) -> None:
    sandbox, fake = await started(docker, CommandResult(exit_code=None), OK)
    with pytest.raises(SandboxError, match="did not return 10 s after") as raised:
        await sandbox.run("ls", timeout_s=2)
    assert raised.value.retryable
    assert fake.calls[1][0] == ("docker", "rm", "--force", sandbox.name)
    with pytest.raises(SandboxError, match="the sandbox is closed"):
        await sandbox.run("ls", timeout_s=2)


async def test_closing_removes_the_container_once(docker) -> None:
    sandbox, fake = await started(docker, OK)
    await sandbox.close()
    await sandbox.close()
    assert fake.calls == [
        (("docker", "rm", "--force", sandbox.name), {"timeout_s": 30.0, "stdin": None})
    ]
    with pytest.raises(SandboxError, match="the sandbox is closed") as raised:
        await sandbox.run("ls", timeout_s=2)
    assert not raised.value.retryable


async def test_a_container_that_cannot_be_removed_is_logged_with_what_to_run(
    docker, caplog: pytest.LogCaptureFixture
) -> None:
    stuck = CommandResult(exit_code=1, stderr="Error response from daemon: removal in progress")
    sandbox, _ = await started(docker, stuck)
    await sandbox.close()
    assert f"could not remove the container {sandbox.name}: Error response from daemon" in (
        caplog.text
    )
    assert f"run `docker rm --force {sandbox.name}`" in caplog.text


async def test_a_container_already_gone_is_closed_quietly(
    docker, caplog: pytest.LogCaptureFixture
) -> None:
    sandbox, _ = await started(docker, CommandResult(exit_code=1, stderr="No such container: x"))
    await sandbox.close()
    assert caplog.text == ""


@pytest.mark.parametrize(
    "settings,why",
    [
        ({"image": " "}, "image must name a Docker image"),
        ({"image": "a", "memory_mb": 0}, "memory_mb must be at least 1 MiB, not 0"),
        ({"image": "a", "cpus": 0}, "cpus must be a positive number, not 0"),
        ({"image": "a", "cpus": float("nan")}, "cpus must be a positive number, not nan"),
        ({"image": "a", "lifetime_s": 0}, "lifetime_s must be at least 1 second, not 0"),
    ],
)
def test_a_config_refuses_what_docker_would(settings: dict[str, Any], why: str) -> None:
    with pytest.raises(ValueError, match=why):
        DockerSandbox.Config(**settings)
