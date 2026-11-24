"""Qwen3.5's message format: ``Qwen/Qwen3.5-4B``,
``Qwen/Qwen3.5-9B`` and ``Qwen/Qwen3.5-35B-A3B``.

It is Qwen3.5's chat template written out, from `Qwen3.5-9B's
chat_template.jinja
<https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/chat_template.jinja>`_.
It keeps Qwen3's turns (``qwen3.py``) and changes these parts:

- **Content** loses the whitespace around it.
- **Reasoning.** Every assistant turn after the last user question has a
  ``<think>`` block, empty when the turn did not reason. The generation
  prompt opens the block itself, so the model starts inside it; with
  ``enable_thinking=False`` the prompt closes it empty.
- **Tools** come before the system message's own text. A call names its
  function and gives each argument in a ``<parameter=name>`` block: a string
  as it is, a list or an object as JSON.
- **A system message** is the first message or is refused.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from looper_cookbook.message_formatters.base import (
    Message,
    Tool,
    argument_value,
    call_arguments,
    parameter_schemas,
    tool_call,
)
from looper_cookbook.message_formatters.qwen3 import Qwen3MessageFormatter

__all__ = ["Qwen3_5MessageFormatter"]

_TOOLS_HOW = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT after\n- If there is "
    "no function call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>"
)
_FUNCTION = re.compile(r"\s*<function=([^>\n]+)>\n(.*)</function>\s*", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>\n?", re.DOTALL)


def tools_text(tools: Sequence[Tool]) -> str:
    """The tools' schemas, and how the model calls one, as the system turn has them."""
    listed = "".join("\n" + json.dumps(tool, ensure_ascii=False) for tool in tools)
    opening = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
    return f"{opening}{listed}\n</tools>{_TOOLS_HOW}"


class Qwen3_5MessageFormatter(Qwen3MessageFormatter):
    """Qwen3.5's format.

    Args:
        tokenizer: A Qwen3.5 model's tokenizer.
        enable_thinking: Whether the model is prompted to reason before it
            answers. On, the prompt opens a ``<think>`` block and a turn
            without reasoning closes it empty. Off, the prompt closes the
            block itself, and a turn that carries reasoning is refused. Train
            and sample a model in the same mode.
    """

    def _prefill(self) -> str:
        return "<think>\n" if self.enable_thinking else "<think>\n\n</think>\n\n"

    def _content(self, message: Message) -> str:
        return super()._content(message).strip()

    def _system(self, messages: list[Message], tools: Sequence[Tool] | None) -> str:
        system = self._content(messages[0]) if messages[0]["role"] == "system" else None
        if tools:
            text = tools_text(tools) + (f"\n\n{system}" if system else "")
            return f"<|im_start|>system\n{text}<|im_end|>\n"
        return "" if system is None else f"<|im_start|>system\n{system}<|im_end|>\n"

    def _later_system(self, content: str) -> str:
        raise ValueError("a system message comes first in the conversation, or not at all")

    def _keeps_thinking(self, after_query: bool) -> bool:
        """Whether an assistant turn is written with its ``<think>`` block."""
        return after_query

    def _argument(self, value: Any) -> str:
        """A tool call's argument as the format writes it."""
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _assistant(self, message: Message, *, after_query: bool, last: bool) -> str:
        reasoning, content = self._reasoning_and_content(message)
        body = content
        if self._keeps_thinking(after_query):
            body = f"<think>\n{reasoning.strip()}\n</think>\n\n{content}"
        for position, call in enumerate(message.get("tool_calls") or ()):
            name, arguments = call_arguments(call)
            # A blank line parts the first call from the content before it.
            body += "\n" if position else "\n\n" if content else ""
            body += f"<tool_call>\n<function={name}>\n"
            for key, value in arguments.items():
                body += f"<parameter={key}>\n{self._argument(value)}\n</parameter>\n"
            body += "</function>\n</tool_call>"
        return body

    def _parse_call(self, block: str, tools: Sequence[Tool] | None) -> dict[str, Any] | None:
        function = _FUNCTION.fullmatch(block)
        if function is None:
            return None
        name, parameters = function.groups()
        if _PARAMETER.sub("", parameters).strip():
            return None
        schemas = parameter_schemas(tools, name)
        arguments = {
            key: argument_value(value, schemas.get(key))
            for key, value in _PARAMETER.findall(parameters)
        }
        return tool_call(name, arguments)
