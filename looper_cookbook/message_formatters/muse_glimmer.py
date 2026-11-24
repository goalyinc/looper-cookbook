"""Muse Glimmer's message format:
``meta-models/Muse-Glimmer-30B``.

It is Muse Glimmer's chat template, the ATEM protocol, written out, from
`Muse-Glimmer-30B's chat_template.jinja
<https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/chat_template.jinja>`_:

- **A message** is ``<|start|>``, its role, ``<|message|>``, its content and
  ``<|eot|>``. The conversation starts with ``<|begin_of_text|>``.
- **An assistant turn is written to recipients.** Its reasoning goes
  ``to=self`` and ends with ``<|eom|>``, more of the turn following. Its
  answer goes ``to=user``, and a tool call ``to=`` the tool's name. The
  turn ends with ``<|eot|>``, which is what sampling stops on; the
  tokenizer's own EOS, ``<|end_of_text|>``, ends no turn.
- **The system turn** always exists. It carries the reasoning strength, the
  tools and the recipients the model may write to. A conversation without a
  system message gets a default one, with the model's knowledge cutoff and
  the current date.
- **Tools** are listed as JSON schemas, grouped by the namespace before a
  name's first dot. A call is an ``<atem:function_calls>`` block: a string
  argument as it is, a list or an object as JSON. A tool's reply names its
  tool, in a ``<tool_output>`` block.
- **Text before a turn's calls** goes ``to=user`` and ends with ``<|eom|>``,
  and the calls follow. The template writes this only when the text and the
  calls are two messages, the first with ``end_turn`` false, and drops the
  content of one message that has both. The formatter writes one message's
  text and calls as the template writes the two, and does not read
  ``recipient`` or ``end_turn``.
"""

from __future__ import annotations

import datetime
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from looper_cookbook.message_formatters.base import (
    Message,
    MessageFormatter,
    Tool,
    argument_value,
    call_arguments,
    parameter_schemas,
    text_content,
    tool_call,
    tool_replies,
)

__all__ = ["REASONING_STRENGTHS", "MuseGlimmerMessageFormatter"]

REASONING_STRENGTHS = ("low", "medium", "high", "xhigh")
KNOWLEDGE_CUTOFF = "2026-01-04"

_TOOLS_INTRO = (
    "In this environment you have access to a set of tools you can use to answer the user's "
    "question.\n\n"
    'You can invoke a function by writing a "<atem:function_calls>" block like the following:\n'
    '<atem:function_calls>\n<atem:invoke name="$FUNCTION_NAME">\n'
    '<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>\n...\n'
    "</atem:invoke>\n</atem:function_calls>\n\n"
    "String and scalar parameters should be specified as is, while lists and objects should "
    "use JSON format. Note that spaces for string values are not stripped. The output is not "
    "expected to be valid XML and is parsed with regular expressions.\n"
    "Here are the functions available in JSONSchema format:\n"
    "// Tool metadata\n"
)
_TOOLS_EXAMPLE = (
    "\n\nHere's an example of how to call a function in the tool set:\n"
    "(If the tool namespace is not specified, invoke the function directly as "
    "`example_function_name` rather than `example_tool_name.example_function_name`)\n\n"
    "to=example_tool_name.example_function_name\n\n"
    '<atem:function_calls>\n<atem:invoke name="example_tool_name.example_function_name">\n'
    '<atem:parameter name="example_parameter_1">value_1</atem:parameter>\n'
    '<atem:parameter name="example_parameter_2">This is the value for the second parameter\n'
    'that can span\n"multiple" lines\n</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)
_SEGMENT = re.compile(r" to=([^<\s]+)<\|message\|>(.*)", re.DOTALL)
_INVOKE = re.compile(
    r'<atem:function_calls>\n<atem:invoke name="([^"\n]+)">\n(.*)</atem:invoke>\n'
    r"</atem:function_calls>",
    re.DOTALL,
)
_PARAMETER = re.compile(r'<atem:parameter name="([^"\n]+)">(.*?)</atem:parameter>\n', re.DOTALL)


class MuseGlimmerMessageFormatter(MessageFormatter):
    """Muse Glimmer's format.

    Args:
        tokenizer: Muse Glimmer's tokenizer.
        reasoning_strength: How much the model reasons before it answers:
            ``"low"``, ``"medium"``, ``"high"`` or ``"xhigh"``. The system
            turn states it, unless a system message states its own. The
            model has no switch that turns reasoning off.
        current_date: The date the default system turn states, as
            ``YYYY-MM-DD``; None for the day the formatter is built. Name
            one to format the same tokens on any day.
    """

    end_of_turn = "<|eot|>"
    special_tokens = ("<|begin_of_text|>", "<|start|>", "<|message|>", "<|eom|>")

    def __init__(
        self,
        tokenizer: Any,
        *,
        reasoning_strength: str = "high",
        current_date: str | None = None,
    ) -> None:
        super().__init__(tokenizer)
        if reasoning_strength not in REASONING_STRENGTHS:
            raise ValueError(
                f"reasoning_strength is {reasoning_strength!r}; Muse Glimmer has "
                f"{', '.join(REASONING_STRENGTHS)}"
            )
        self.reasoning_strength = reasoning_strength
        self.current_date = current_date or datetime.date.today().isoformat()

    def format_prompt(
        self, messages: Sequence[Message], tools: Sequence[Tool] | None = None
    ) -> str:
        messages = list(messages)
        text = "<|begin_of_text|>"
        if not any(message["role"] == "system" for message in messages):
            default = (
                "You are a helpful AI assistant.\n"
                f"Knowledge cutoff: {KNOWLEDGE_CUTOFF}.\nCurrent date: {self.current_date}."
            )
            text += self._system(default, tools)
        for index, message in enumerate(messages):
            role = message["role"]
            if role == "system":
                text += self._system(_reasoning_strength_named(text_content(message)), tools)
            elif role == "user":
                text += f"<|start|>user<|message|>{text_content(message)}<|eot|>"
            elif role == "tool":
                text += _tool_reply(message, messages)
            elif role == "assistant":
                # Calls that another assistant message follows do not end the turn.
                more = index + 1 < len(messages) and messages[index + 1]["role"] == "assistant"
                text += self._assistant(message, end="<|eom|>" if more else "<|eot|>")
            else:
                raise ValueError(
                    f"a message's role is {role!r}; the format has system, user, assistant and tool"
                )
        return text + "<|start|>assistant"

    def format_response(self, message: Message) -> str:
        # The prompt wrote the turn's first <|start|>assistant.
        return self._assistant(message, end="<|eot|>").removeprefix("<|start|>assistant")

    def format_continuation(self, messages: Sequence[Message]) -> str:
        replies = tool_replies(messages)
        # A reply names its tool itself here: the calls it answers are tokens already.
        return "".join(_tool_reply(reply, replies) for reply in replies) + "<|start|>assistant"

    def parse_text(self, text: str, tools: Sequence[Tool] | None = None) -> dict[str, Any]:
        unread = {"role": "assistant", "content": text}
        reasoning: list[str] = []
        calls: list[dict[str, Any]] = []
        content: str | None = None
        # Each part of the turn names its recipient; all but the last end with <|eom|>.
        for segment in filter(None, text.split("<|eom|>")):
            found = _SEGMENT.fullmatch(segment.removeprefix("<|start|>assistant"))
            if found is None:
                return unread
            recipient, body = found.groups()
            if recipient == "self":
                reasoning.append(body)
            elif recipient == "user":
                # A message holds one text, before its calls: any other turn stays as written.
                if content is not None or calls:
                    return unread
                content = body
            else:
                call = _parse_call(body, tools)
                if call is None:
                    return unread
                calls.append(call)
        message: dict[str, Any] = {"role": "assistant"}
        if reasoning:
            message["reasoning_content"] = "\n".join(reasoning)
        message["content"] = content or ""
        if calls:
            message["tool_calls"] = calls
        return message

    def _system(self, text: str, tools: Sequence[Tool] | None) -> str:
        """A system turn: its text, then the reasoning strength, the tools and
        the recipients the model may write to."""
        parts = [text]
        if "reasoning strength" not in text.lower():
            parts.append(f"Reasoning strength: {self.reasoning_strength}.")
        functions = [tool.get("function", tool) for tool in tools or ()]
        namespaces = list(dict.fromkeys(function["name"].split(".")[0] for function in functions))
        if functions:
            parts.append(_tool_definitions(functions, namespaces))
        recipients = ['"self"', *(f'"{namespace}.*"' for namespace in namespaces), '"user"']
        parts.append(f"# Valid recipients: {', '.join(recipients)}.")
        return "<|start|>system<|message|>" + "\n\n".join(parts) + "<|eot|>"

    def _assistant(self, message: Message, *, end: str) -> str:
        """An assistant message: its reasoning, then its text to the user, then
        its tool calls.

        Args:
            end: The token after the last tool call.
        """
        if message.get("recipient") not in (None, "user") or message.get("end_turn") is False:
            raise ValueError(
                "this assistant message carries Muse Glimmer's own recipient or end_turn, "
                "which the formatter does not read: give a turn's text as its content and "
                "its calls as its tool_calls, in one message"
            )
        text = ""
        if message.get("reasoning_content"):
            text += f"<|start|>assistant to=self<|message|>{message['reasoning_content']}<|eom|>"
        content = text_content(message)
        calls = message.get("tool_calls") or ()
        if not calls:
            return text + f"<|start|>assistant to=user<|message|>{content}<|eot|>"
        # Text before the calls goes to the user, and the turn goes on.
        if content.strip():
            text += f"<|start|>assistant to=user<|message|>{content}<|eom|>"
        for position, call in enumerate(calls):
            name, arguments = call_arguments(call)
            text += f"<|start|>assistant to={name}<|message|>{_function_calls(name, arguments)}"
            text += end if position == len(calls) - 1 else "<|eom|>"
        return text


def _reasoning_strength_named(text: str) -> str:
    """A system message's text, with a reasoning effort it states named as
    the model names it, a reasoning strength."""
    for effort, strength in (
        ("Reasoning effort", "Reasoning strength"),
        ("Reasoning Effort", "Reasoning Strength"),
        ("reasoning effort", "reasoning strength"),
        ("REASONING EFFORT", "REASONING STRENGTH"),
    ):
        text = text.replace(effort, strength)
    return text


def _tool_definitions(functions: Sequence[Mapping[str, Any]], namespaces: Sequence[str]) -> str:
    """The tools' schemas, and how the model calls one, as the system turn has them."""
    for function in functions:
        missing = [key for key in ("description", "parameters") if key not in function]
        if missing:
            raise ValueError(
                f"the tool {function['name']} has no {' or '.join(missing)}, and Muse Glimmer's "
                "format lists both"
            )
    text = _TOOLS_INTRO
    for namespace in namespaces:
        text += f'{{"name": {_json(namespace)}, "description": ""}}\n'
    text += "// Function schemas"
    for function in functions:
        text += (
            f'\n{{"name": {_json(function["name"])}, '
            f'"description": {_json(function["description"])}, '
            f'"parameters": {_json(function["parameters"])}}}'
        )
    return text + _TOOLS_EXAMPLE


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _tool_name(message: Message, messages: Sequence[Message]) -> str:
    """The tool a reply is from: its ``name``, or the name of the call its
    ``tool_call_id`` answers."""
    if message.get("name"):
        return message["name"]
    call_id = message.get("tool_call_id")
    names = [
        call["function"]["name"]
        for earlier in messages
        for call in earlier.get("tool_calls") or ()
        if call_id is not None and call.get("id") == call_id
    ]
    if not names:
        raise ValueError(
            "a tool message does not say which tool it is from, and Muse Glimmer's format "
            "names it: set the message's name, or a tool_call_id that a call's id matches"
        )
    return names[-1]


def _tool_reply(message: Message, messages: Sequence[Message]) -> str:
    """A tool's reply, which names its tool, in a ``<tool_output>`` block."""
    name = _tool_name(message, messages)
    text = f'<|start|>tool {name}<|message|><tool_output name="{name}">\n'
    return text + f"{text_content(message)}\n</tool_output><|eot|>"


def _function_calls(name: str, arguments: Mapping[str, Any]) -> str:
    """One tool call as an ``<atem:function_calls>`` block."""
    text = f'<atem:function_calls>\n<atem:invoke name="{name}">\n'
    for key, value in arguments.items():
        text += f'<atem:parameter name="{key}">{_argument(value)}</atem:parameter>\n'
    return text + "</atem:invoke>\n</atem:function_calls>"


def _argument(value: Any) -> str:
    """A tool call's argument as the format writes it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return _json(value)
    return str(value)


def _parse_call(text: str, tools: Sequence[Tool] | None) -> dict[str, Any] | None:
    """The call an ``<atem:function_calls>`` block holds, or None when it is not one."""
    invoke = _INVOKE.fullmatch(text)
    if invoke is None:
        return None
    name, parameters = invoke.groups()
    if _PARAMETER.sub("", parameters):
        return None
    schemas = parameter_schemas(tools, name)
    arguments = {
        key: argument_value(value, schemas.get(key))
        for key, value in _PARAMETER.findall(parameters)
    }
    return tool_call(name, arguments)
