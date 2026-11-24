"""A local sandbox over real processes: what a command gets, what comes
back, and what a deadline kills."""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from looper_cookbook.sandboxes.base import CommandResult, SandboxError
from looper_cookbook.sandboxes.local import LocalSandbox


@pytest.fixture
async def sandbox():
    async with await LocalSandbox.Config().build() as built:
        yield built


async def test_a_command_returns_its_streams_and_its_status(sandbox: LocalSandbox) -> None:
    result = await sandbox.run("echo out; echo err >&2; exit 3", timeout_s=10)
    assert result == CommandResult(exit_code=3, stdout="out\n", stderr="err\n")


async def test_a_command_reads_its_input(sandbox: LocalSandbox) -> None:
    result = await sandbox.run("tr a-z A-Z", timeout_s=10, stdin="hello\nworld\n")
    assert result == CommandResult(exit_code=0, stdout="HELLO\nWORLD\n")
    # Without input a command reads an empty one, and does not wait.
    assert (await sandbox.run("cat", timeout_s=10)).stdout == ""


async def test_a_command_that_never_reads_its_input_still_returns(sandbox: LocalSandbox) -> None:
    # More than a pipe holds, to a program that exits without reading it.
    result = await sandbox.run("echo done", timeout_s=10, stdin="x" * 4_000_000)
    assert result == CommandResult(exit_code=0, stdout="done\n")


async def test_commands_share_the_working_directory_and_no_other(sandbox: LocalSandbox) -> None:
    await sandbox.run("echo 7 > n.txt", timeout_s=10)
    result = await sandbox.run("pwd; cat n.txt", timeout_s=10)
    assert result.stdout == f"{os.path.realpath(sandbox.directory)}\n7\n"
    async with await LocalSandbox.Config().build() as other:
        assert (await other.run("ls", timeout_s=10)).stdout == ""


async def test_a_file_is_written_whole_with_its_directory(sandbox: LocalSandbox) -> None:
    text = "print('it''s \"quoted\"')\n$HOME `id`\n"
    await sandbox.write_file("src/pkg/it's a file.py", text)
    result = await sandbox.run('cat "src/pkg/it\'s a file.py"', timeout_s=10)
    assert result.stdout == text


async def test_a_file_that_cannot_be_written_raises(sandbox: LocalSandbox) -> None:
    await sandbox.run("touch taken", timeout_s=10)
    with pytest.raises(SandboxError, match="could not write taken/a.py in the sandbox"):
        await sandbox.write_file("taken/a.py", "")


async def test_a_command_past_its_deadline_is_killed_with_what_it_started(
    sandbox: LocalSandbox,
) -> None:
    started = time.monotonic()
    # The child holds the output open: only killing the whole group returns.
    command = "sleep 30 & echo $! > child; echo before; sleep 30; echo after"
    result = await sandbox.run(command, timeout_s=0.5)
    assert time.monotonic() - started < 10
    assert result == CommandResult(exit_code=None, stdout="before\n")
    assert result.timed_out
    # The child is gone too, once the system has collected it.
    alive = await sandbox.run("kill -0 $(cat child)", timeout_s=10)
    for _ in range(40):
        if alive.exit_code != 0:
            break
        await asyncio.sleep(0.05)
        alive = await sandbox.run("kill -0 $(cat child)", timeout_s=10)
    assert alive.exit_code != 0


async def test_a_cancelled_command_is_killed(sandbox: LocalSandbox) -> None:
    task = asyncio.create_task(sandbox.run("echo $$ > pid; sleep 30", timeout_s=30))
    for _ in range(100):
        if os.path.exists(os.path.join(sandbox.directory, "pid")):
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pid = (await sandbox.run("cat pid", timeout_s=10)).stdout.strip()
    alive = await sandbox.run(f"kill -0 {pid}", timeout_s=10)
    assert alive.exit_code != 0


async def test_a_command_gets_the_path_and_none_of_this_processs_secrets(
    sandbox: LocalSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOOPER_API_KEY", "sk-not-for-a-model")
    result = await sandbox.run("env", timeout_s=10)
    variables = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert "LOOPER_API_KEY" not in variables
    assert variables["PATH"] == os.environ["PATH"]
    assert variables["HOME"] == sandbox.directory


@pytest.mark.parametrize("timeout_s", [0, -1, float("nan"), float("inf")])
async def test_a_deadline_must_be_a_positive_number_of_seconds(
    sandbox: LocalSandbox, timeout_s: float
) -> None:
    with pytest.raises(ValueError, match="timeout_s must be a positive number of seconds"):
        await sandbox.run("true", timeout_s=timeout_s)


async def test_closing_removes_the_directory_and_refuses_more_commands() -> None:
    sandbox = await LocalSandbox.Config().build()
    await sandbox.run("mkdir -p deep/er && touch deep/er/file", timeout_s=10)
    await sandbox.close()
    await sandbox.close()
    assert not os.path.exists(sandbox.directory)
    with pytest.raises(SandboxError, match="the sandbox is closed") as raised:
        await sandbox.run("true", timeout_s=10)
    assert not raised.value.retryable
