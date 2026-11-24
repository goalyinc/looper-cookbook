"""What every sandbox shares: a command's result, a file written through
``run``, and the error a broken sandbox raises."""

from __future__ import annotations

import pytest

from looper_cookbook.sandboxes.base import CommandResult, Sandbox, SandboxError
from looper_cookbook.sandboxes.docker import DockerSandbox
from looper_cookbook.sandboxes.local import LocalSandbox
from looper_cookbook.sandboxes.unit_tests.scripted import OK, ScriptedSandbox


def test_a_result_without_an_exit_code_ran_out_of_time() -> None:
    assert CommandResult(exit_code=None).timed_out
    assert not CommandResult(exit_code=0).timed_out
    assert not CommandResult(exit_code=137).timed_out


def test_a_sandbox_error_says_whether_a_new_sandbox_may_do_better() -> None:
    assert SandboxError("the container died", retryable=True).retryable
    assert not SandboxError("no such image", retryable=False).retryable


async def test_write_file_pipes_the_text_to_cat_in_the_sandbox() -> None:
    sandbox = ScriptedSandbox(OK)
    await sandbox.write_file("src/it's here.py", "print(1)\n")
    assert sandbox.commands == [
        (
            "mkdir -p -- \"$(dirname -- 'src/it'\"'\"'s here.py')\" && "
            "cat > 'src/it'\"'\"'s here.py'",
            60.0,
            "print(1)\n",
        )
    ]


@pytest.mark.parametrize(
    "result,why",
    [
        (CommandResult(exit_code=1, stderr="cat: a.py: Permission denied\n"), "Permission denied"),
        (CommandResult(exit_code=None), "no answer in 60 s"),
    ],
)
async def test_a_file_that_cannot_be_written_is_the_sandboxs_error(
    result: CommandResult, why: str
) -> None:
    with pytest.raises(SandboxError, match=why) as raised:
        await ScriptedSandbox(result).write_file("a.py", "")
    assert not raised.value.retryable


def test_a_config_is_anything_that_builds_a_sandbox() -> None:
    assert isinstance(LocalSandbox.Config(), Sandbox.Config)
    assert isinstance(DockerSandbox.Config(image="python:3.12-slim"), Sandbox.Config)
    assert not isinstance(object(), Sandbox.Config)


async def test_leaving_its_block_closes_a_sandbox() -> None:
    sandbox = ScriptedSandbox()
    with pytest.raises(RuntimeError, match="boom"):
        async with sandbox as entered:
            assert entered is sandbox
            raise RuntimeError("boom")
    assert sandbox.closes == 1
