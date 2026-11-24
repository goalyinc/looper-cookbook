"""The environment contract: an action checked against the tools before it
runs, and the sandbox an environment owns."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from looper_cookbook.environments.base import (
    Action,
    Environment,
    Observation,
    Tool,
    refusal,
    tool,
)
from looper_cookbook.sandboxes.unit_tests.scripted import ScriptedSandbox

SEARCH = tool(
    "search",
    "Search the notes.",
    {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
        "score": {"type": "number"},
        "exact": {"type": "boolean"},
        "tags": {"type": "array"},
        "filters": {"type": "object"},
        "anything": {},
        "either": {"type": ["string", "integer"]},
    },
    required=("query",),
)
CLOCK = tool("clock", "The time.", {})


class Notes(Environment):
    """Records the actions that reach ``execute``."""

    def __init__(self) -> None:
        self.executed: list[Action] = []

    @property
    def tools(self) -> Sequence[Tool]:
        return (SEARCH, CLOCK)

    async def execute(self, action: Action) -> Observation:
        self.executed.append(action)
        return Observation(content="found", metrics={"hits": 2.0})


class NotesOnDisk(Notes):
    """The same over a sandbox, as an environment that runs commands is."""

    def __init__(self, sandbox: ScriptedSandbox) -> None:
        super().__init__()
        self._sandbox = sandbox

    @property
    def sandbox(self) -> ScriptedSandbox:
        return self._sandbox


def search(**arguments: Any) -> Action:
    return Action(name="search", arguments=arguments)


def test_tool_writes_the_schema_a_formatter_takes() -> None:
    assert tool("clock", "The time.", {"zone": {"type": "string"}}, required=("zone",)) == {
        "type": "function",
        "function": {
            "name": "clock",
            "description": "The time.",
            "parameters": {
                "type": "object",
                "properties": {"zone": {"type": "string"}},
                "required": ["zone"],
            },
        },
    }


@pytest.mark.parametrize(
    "action",
    [
        search(query="tides"),
        search(query="tides", limit=3, score=0.5, exact=True, tags=["a"], filters={"a": 1}),
        # An integer is a number, and a tuple an array.
        search(query="tides", score=2, tags=("a",)),
        # A parameter that declares no single type takes any value.
        search(query="tides", anything=[1], either=4),
        Action(name="clock", arguments={}),
    ],
)
def test_an_action_that_fits_a_tool_is_not_refused(action: Action) -> None:
    assert refusal(action, (SEARCH, CLOCK)) is None


@pytest.mark.parametrize(
    "action,why",
    [
        (
            Action(name="lookup", arguments={}),
            "Error: there is no tool named lookup. The tools are: search, clock.",
        ),
        (search(limit=3), "Error: search was called without its required query."),
        (
            Action(name="clock", arguments={"zone": "UTC"}),
            "Error: clock has no argument named zone. It takes: no arguments.",
        ),
        (
            search(query="tides", page=2),
            "Error: search has no argument named page. It takes: query, limit, score, exact, "
            "tags, filters, anything, either.",
        ),
        (search(query=7), "Error: search's argument query must be of type string, not 7."),
        (
            search(query="tides", limit="3"),
            'Error: search\'s argument limit must be of type integer, not "3".',
        ),
        (
            search(query="tides", limit=2.5),
            "Error: search's argument limit must be of type integer, not 2.5.",
        ),
        # Python counts a bool as an int; JSON does not.
        (
            search(query="tides", limit=True),
            "Error: search's argument limit must be of type integer, not true.",
        ),
        (
            search(query="tides", score=False),
            "Error: search's argument score must be of type number, not false.",
        ),
        (
            search(query="tides", exact=1),
            "Error: search's argument exact must be of type boolean, not 1.",
        ),
        (
            search(query="tides", tags={"a": 1}),
            'Error: search\'s argument tags must be of type array, not {"a": 1}.',
        ),
        (
            search(query="tides", filters=["a"]),
            'Error: search\'s argument filters must be of type object, not ["a"].',
        ),
    ],
)
def test_an_action_that_fits_no_tool_is_refused_in_words(action: Action, why: str) -> None:
    assert refusal(action, (SEARCH, CLOCK)) == why


def test_a_refusal_shows_only_the_start_of_a_long_value() -> None:
    why = refusal(search(query=["x" * 200]), (SEARCH,))
    shown = '["' + "x" * 78
    assert why == f"Error: search's argument query must be of type string, not {shown}."


async def test_step_runs_an_action_that_fits() -> None:
    notes = Notes()
    action = search(query="tides")
    assert await notes.step(action) == Observation(content="found", metrics={"hits": 2.0})
    assert notes.executed == [action]


async def test_step_answers_a_wrong_call_without_running_it() -> None:
    notes = Notes()
    observation = await notes.step(search(limit=3))
    assert observation == Observation(
        content="Error: search was called without its required query.",
        metrics={"invalid_action": 1.0},
    )
    assert notes.executed == []


async def test_leaving_its_block_closes_an_environments_sandbox() -> None:
    sandbox = ScriptedSandbox()
    notes = NotesOnDisk(sandbox)
    with pytest.raises(RuntimeError, match="boom"):
        async with notes as entered:
            assert entered is notes and notes.sandbox is sandbox
            raise RuntimeError("boom")
    assert sandbox.closes == 1


async def test_an_environment_that_needs_no_sandbox_has_none_to_close() -> None:
    async with Notes() as notes:
        assert notes.sandbox is None


def test_a_config_is_anything_that_builds_an_environment() -> None:
    class Config:
        async def build(self) -> Notes:
            return Notes()

    assert isinstance(Config(), Environment.Config)
    assert not isinstance(object(), Environment.Config)
