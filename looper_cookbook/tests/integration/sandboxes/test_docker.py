"""A Docker sandbox over real containers: what the container is denied, what
a deadline kills, and what is left when it closes.

These start containers of ``python:3.12-slim``, so they run only with
``LOOPER_TEST_DOCKER=1``, on a machine with Docker.
"""

from __future__ import annotations

import os
import time

import pytest

from looper_cookbook.sandboxes.base import CommandResult, SandboxError
from looper_cookbook.sandboxes.docker import DockerSandbox
from looper_cookbook.sandboxes.process import run_process

IMAGE = "python:3.12-slim"

pytestmark = pytest.mark.skipif(
    os.environ.get("LOOPER_TEST_DOCKER") != "1",
    reason="set LOOPER_TEST_DOCKER=1 to run in real Docker containers",
)


async def containers(name: str) -> str:
    """The names of the containers called ``name``, running or not."""
    listed = await run_process(
        ("docker", "ps", "--all", "--filter", f"name=^{name}$", "--format", "{{.Names}}"),
        timeout_s=30,
    )
    return listed.stdout.strip()


@pytest.fixture
async def sandbox():
    async with await DockerSandbox.Config(image=IMAGE).build() as built:
        yield built


async def test_a_command_runs_in_the_container_with_its_input(sandbox: DockerSandbox) -> None:
    result = await sandbox.run("pwd; cat; echo err >&2; exit 3", timeout_s=20, stdin="in\n")
    assert result == CommandResult(exit_code=3, stdout="/workspace\nin\n", stderr="err\n")
    # Not this machine: the container's own user and files.
    assert (await sandbox.run("id -u; ls /workspace", timeout_s=20)).stdout == "0\n"


async def test_files_stay_between_commands(sandbox: DockerSandbox) -> None:
    await sandbox.write_file("pkg/it's a file.py", "print('$HOME')\n")
    result = await sandbox.run('python3 "pkg/it\'s a file.py"', timeout_s=20)
    assert result == CommandResult(exit_code=0, stdout="$HOME\n")


async def test_the_container_has_no_network_unless_asked(sandbox: DockerSandbox) -> None:
    probe = "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 3)\""
    refused = await sandbox.run(probe, timeout_s=20)
    assert refused.exit_code == 1 and "Network is unreachable" in refused.stderr
    assert (await sandbox.run("ls /sys/class/net", timeout_s=20)).stdout == "lo\n"
    async with await DockerSandbox.Config(image=IMAGE, network=True).build() as networked:
        interfaces = (await networked.run("ls /sys/class/net", timeout_s=20)).stdout.split()
        assert "lo" in interfaces and len(interfaces) > 1


async def test_a_command_past_its_deadline_is_killed_inside_the_container(
    sandbox: DockerSandbox,
) -> None:
    started = time.monotonic()
    result = await sandbox.run("echo before; sleep 300; echo after", timeout_s=1)
    assert time.monotonic() - started < 10
    assert result == CommandResult(exit_code=None, stdout="before\n")
    # One `sleep` is left, the container's own: the command's was killed with it.
    left = await sandbox.run("cat /proc/[0-9]*/comm | grep -c '^sleep$'", timeout_s=20)
    assert left.stdout == "1\n"
    assert (await sandbox.run("echo ok", timeout_s=20)).stdout == "ok\n"


async def test_a_command_that_passes_its_memory_is_killed_before_its_deadline() -> None:
    async with await DockerSandbox.Config(image=IMAGE, memory_mb=64).build() as small:
        result = await small.run('python3 -c "x = bytearray(512 << 20)"', timeout_s=60)
        assert result.exit_code == 137 and not result.timed_out
        assert (await small.run("echo ok", timeout_s=20)).stdout == "ok\n"


async def test_closing_removes_the_container() -> None:
    sandbox = await DockerSandbox.Config(image=IMAGE).build()
    assert await containers(sandbox.name) == sandbox.name
    await sandbox.close()
    await sandbox.close()
    assert await containers(sandbox.name) == ""
    with pytest.raises(SandboxError, match="the sandbox is closed"):
        await sandbox.run("true", timeout_s=20)


async def test_a_container_removes_itself_when_its_lifetime_ends() -> None:
    sandbox = await DockerSandbox.Config(image=IMAGE, lifetime_s=2).build()
    try:
        assert (await sandbox.run("echo ok", timeout_s=20)).stdout == "ok\n"
        for _ in range(60):
            if not await containers(sandbox.name):
                break
            time.sleep(0.25)
        assert await containers(sandbox.name) == ""
        with pytest.raises(SandboxError, match="The container is gone") as raised:
            await sandbox.run("true", timeout_s=20)
        assert raised.value.retryable
    finally:
        await sandbox.close()


async def test_an_image_that_does_not_exist_is_an_error_that_leaves_nothing() -> None:
    with pytest.raises(SandboxError, match="could not start a container of") as raised:
        await DockerSandbox.Config(image="looper-cookbook/no-such-image:0").build()
    assert not raised.value.retryable
    left = await run_process(
        ("docker", "ps", "--all", "--filter", "ancestor=looper-cookbook/no-such-image:0", "-q"),
        timeout_s=30,
    )
    assert left.stdout == ""
