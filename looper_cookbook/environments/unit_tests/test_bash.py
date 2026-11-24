"""The bash environment's replies, over a sandbox that runs nothing."""

from __future__ import annotations

from typing import Any

import pytest

from looper_cookbook.environments.base import Action, Observation
from looper_cookbook.environments.bash import BashEnvironment
from looper_cookbook.sandboxes.base import CommandResult
from looper_cookbook.sandboxes.unit_tests.scripted import OK, Prebuilt, ScriptedSandbox


async def shell(sandbox: ScriptedSandbox, **settings: Any) -> BashEnvironment:
    return await BashEnvironment.Config(sandbox=Prebuilt(sandbox), **settings).build()


def bash(command: str) -> Action:
    return Action(name="bash", arguments={"command": command})


async def test_it_offers_one_tool_that_takes_a_command() -> None:
    (schema,) = (await shell(ScriptedSandbox())).tools
    assert schema["function"]["name"] == "bash"
    assert schema["function"]["parameters"] == {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The command line to run."}},
        "required": ["command"],
    }


async def test_a_command_runs_in_bash_with_its_streams_joined() -> None:
    sandbox = ScriptedSandbox(CommandResult(exit_code=0, stdout="it's 4\n"))
    environment = await shell(sandbox, timeout_s=30)
    observation = await environment.step(bash('echo "it\'s $((2 + 2))"'))
    assert observation == Observation(content="it's 4\n", metrics={"failed": 0.0, "timed_out": 0.0})
    assert sandbox.commands == [("bash -c 'echo \"it'\"'\"'s $((2 + 2))\"' 2>&1", 30, None)]
    # The sandbox is the environment's own, for a verifier to read.
    assert environment.sandbox is sandbox


@pytest.mark.parametrize(
    "result,content,metrics",
    [
        (OK, "[no output]", {"failed": 0.0, "timed_out": 0.0}),
        (
            CommandResult(exit_code=2, stdout="ls: cannot access 'x'\n"),
            "ls: cannot access 'x'\n[exit code 2]",
            {"failed": 1.0, "timed_out": 0.0},
        ),
        # A status line starts on a line of its own.
        (
            CommandResult(exit_code=1, stdout="partial"),
            "partial\n[exit code 1]",
            {"failed": 1.0, "timed_out": 0.0},
        ),
        (CommandResult(exit_code=1), "[exit code 1]", {"failed": 1.0, "timed_out": 0.0}),
        (
            CommandResult(exit_code=None, stdout="step 1\n"),
            "step 1\n[timed out after 60 s and was killed]",
            {"failed": 1.0, "timed_out": 1.0},
        ),
    ],
)
async def test_a_reply_ends_with_what_went_wrong(
    result: CommandResult, content: str, metrics: dict[str, float]
) -> None:
    environment = await shell(ScriptedSandbox(result))
    assert await environment.step(bash("ls x")) == Observation(content=content, metrics=metrics)


async def test_a_long_output_keeps_its_start_and_its_end() -> None:
    output = "a" * 6 + "b" * 30 + "c" * 6
    sandbox = ScriptedSandbox(
        CommandResult(exit_code=0, stdout=output), CommandResult(exit_code=0, stdout="a" * 12)
    )
    environment = await shell(sandbox, max_output_chars=12)
    observation = await environment.step(bash("make"))
    assert observation.content == "aaaaaa\n[... 30 characters left out ...]\ncccccc"
    # An output that just fits is whole.
    assert (await environment.step(bash("make"))).content == "a" * 12


async def test_a_wrong_call_reaches_no_sandbox() -> None:
    sandbox = ScriptedSandbox()
    environment = await shell(sandbox)
    observation = await environment.step(Action(name="bash", arguments={"cmd": "ls"}))
    assert observation == Observation(
        content="Error: bash was called without its required command.",
        metrics={"invalid_action": 1.0},
    )
    assert sandbox.commands == []


async def test_closing_it_closes_its_sandbox() -> None:
    sandbox = ScriptedSandbox()
    async with await shell(sandbox):
        pass
    assert sandbox.closes == 1


@pytest.mark.parametrize(
    "settings,why",
    [
        ({"timeout_s": 0}, "timeout_s must be a positive number, not 0"),
        ({"timeout_s": float("inf")}, "timeout_s must be a positive number, not inf"),
        ({"max_output_chars": 1}, "max_output_chars must be at least 2, not 1"),
    ],
)
def test_a_config_refuses_limits_that_bound_nothing(settings: dict[str, Any], why: str) -> None:
    with pytest.raises(ValueError, match=why):
        BashEnvironment.Config(sandbox=Prebuilt(ScriptedSandbox()), **settings)
