"""The environment contract.

An ``Environment`` is where a model's actions run. It offers tools, runs one
action at a time, and returns what the model reads next:

- ``tools``: the JSON schemas of its actions, as a message formatter lists
  them in a prompt;
- ``step(action)``: runs one tool call and returns its ``Observation``;
- ``close()``: releases what it holds.

It knows nothing of models, tokens, prompts or rewards. An episode session turns
a sampled turn's tool calls into actions and the observations into the
conversation's next messages, and a verifier scores the episode. So
one environment serves any model, any loop, and training and evaluation
alike.

**One environment serves one episode.** ``Config.build()`` makes a fresh
one, with whatever it runs in, and whoever builds it closes it. A config is
data, so a task can carry its environment's config and build one for every
attempt.

**An environment may run in a sandbox.** Its config then takes the
sandbox's config, ``build()`` starts the sandbox, and ``sandbox`` is the
one it runs in. The environment owns it: ``close()`` closes it. An
environment that answers from this process has none.

**A subclass writes ``tools`` and ``execute``.** ``step`` checks an action
against the tools first, so ``execute`` only sees a tool it offers, with the
arguments that tool requires, each of the type it declares.

**Two kinds of failure stay apart.** A call the model got wrong, an unknown
tool or a missing argument, is an observation that says so: the model reads
it and may try again. A place that cannot run actions at all raises, as a
sandbox raises ``SandboxError``.
"""

from __future__ import annotations

import abc
import dataclasses
import json
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable

from looper_cookbook.sandboxes.base import Sandbox

__all__ = ["Action", "Environment", "Observation", "Tool", "refusal", "tool"]

Tool = Mapping[str, Any]

# What each JSON schema type is in Python. A bool is no integer here, though it is one in Python.
_JSON_TYPES: Mapping[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (Mapping,),
}


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Action:
    """One thing a model asked its environment to do: a tool call.

    Attributes:
        name: The tool to call.
        arguments: The call's arguments by name, as a message formatter's
            ``parse_response`` read them.
    """

    name: str
    arguments: Mapping[str, Any]


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Observation:
    """What an action led to.

    Attributes:
        content: The text the model reads as the tool's reply.
        metrics: Counts of what happened, for a run's log: numbers that add
            up over an episode's actions, such as ``invalid_action``.
    """

    content: str
    metrics: Mapping[str, float] = dataclasses.field(default_factory=dict)


def tool(
    name: str,
    description: str,
    parameters: Mapping[str, Mapping[str, Any]],
    *,
    required: Sequence[str] = (),
) -> dict[str, Any]:
    """A tool's JSON schema, as the message formatters take it.

    Args:
        name: The tool's name, as the model calls it.
        description: What the tool does, as the model reads it.
        parameters: Each parameter's own JSON schema, by name.
        required: The parameters a call must give.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(parameters),
                "required": list(required),
            },
        },
    }


def refusal(action: Action, tools: Sequence[Tool]) -> str | None:
    """Why ``action`` fits none of ``tools``, in words its model can act on;
    ``None`` when it fits one.

    An action fits a tool it names when it gives every required argument, no
    argument the tool lacks, and each as the JSON type the tool declares.
    """
    schemas = {entry["function"]["name"]: entry["function"]["parameters"] for entry in tools}
    if action.name not in schemas:
        return f"Error: there is no tool named {action.name}. The tools are: {', '.join(schemas)}."
    schema = schemas[action.name]
    properties = schema.get("properties", {})
    missing = [name for name in schema.get("required", ()) if name not in action.arguments]
    if missing:
        return f"Error: {action.name} was called without its required {', '.join(missing)}."
    for name, value in action.arguments.items():
        if name not in properties:
            takes = ", ".join(properties) or "no arguments"
            return f"Error: {action.name} has no argument named {name}. It takes: {takes}."
        kind = properties[name].get("type")
        # A parameter that declares no single type takes any value.
        types = _JSON_TYPES.get(kind) if isinstance(kind, str) else None
        if types is None:
            continue
        if not isinstance(value, types) or (isinstance(value, bool) and kind != "boolean"):
            given = json.dumps(value, default=str)[:80]
            return f"Error: {action.name}'s argument {name} must be of type {kind}, not {given}."
    return None


class Environment(abc.ABC):
    """A place where a model's actions run, for one episode."""

    @runtime_checkable
    class Config(Protocol):
        async def build(self) -> Environment:
            """A fresh environment, with what it runs in. Whoever builds it closes it."""
            ...

    @property
    @abc.abstractmethod
    def tools(self) -> Sequence[Tool]:
        """The JSON schemas of the actions it offers, each made by ``tool``."""

    @property
    def sandbox(self) -> Sandbox | None:
        """The sandbox its commands run in; ``None`` when it needs none.

        A verifier lent the environment runs its own checks there, on what
        the model left.
        """
        return None

    @abc.abstractmethod
    async def execute(self, action: Action) -> Observation:
        """Runs an action that fits one of ``tools``. Callers use ``step``."""

    async def step(self, action: Action) -> Observation:
        """Runs one action and returns what the model reads next.

        An action that fits none of the tools is not run. Its observation
        says why, and counts as an ``invalid_action``.

        Raises:
            SandboxError: The environment's sandbox could not run the action
                at all.
        """
        problem = refusal(action, self.tools)
        if problem is not None:
            return Observation(content=problem, metrics={"invalid_action": 1.0})
        return await self.execute(action)

    async def close(self) -> None:
        """Closes its sandbox, if it has one. A second call does nothing."""
        sandbox = self.sandbox
        if sandbox is not None:
            await sandbox.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
