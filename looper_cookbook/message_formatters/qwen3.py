"""Qwen3's message format: the hybrid thinking
models, such as ``Qwen/Qwen3-8B``.

It is Qwen3's chat template written out, from `Qwen3-8B's tokenizer_config.json
<https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/tokenizer_config.json>`_:

- **Turns** are ChatML's: ``<|im_start|>role``, a newline, the content and
  ``<|im_end|>``.
- **Reasoning** is a ``<think>`` block that opens an assistant turn. Only a
  turn after the last user question keeps its block; an earlier turn is
  written without it.
- **Thinking mode.** With ``enable_thinking=False`` the generation prompt
  closes an empty ``<think>`` block, which tells the model to answer without
  reasoning.
- **Tools** are listed in the system turn as JSON schemas. A call is a JSON
  object in a ``<tool_call>`` block, and a tool's reply a ``<tool_response>``
  block in a user turn, a run of replies sharing one turn.

Qwen3.5's and Qwen3.8's formats are subclasses, in ``qwen3_5.py`` and
``qwen3_8.py``: they keep the turns and change the parts named above.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from looper_cookbook.message_formatters.base import (
    Message,
    MessageFormatter,
    Tool,
    reasoning_content,
    text_content,
    tool_call,
    tool_replies,
)

__all__ = ["Qwen3MessageFormatter"]

_TOOLS_OPEN = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
_TOOLS_CLOSE = (
    "\n</tools>\n\nFor each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)
_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class Qwen3MessageFormatter(MessageFormatter):
    """Qwen3's format.

    Args:
        tokenizer: A Qwen3 model's tokenizer.
        enable_thinking: Whether the model is prompted to reason before it
            answers. Off, the prompt closes an empty ``<think>`` block, and a
            turn that carries reasoning is refused. Train and sample a model
            in the same mode.
    """

    end_of_turn = "<|im_end|>"
    special_tokens = (
        "<|im_start|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
    )

    def __init__(self, tokenizer: Any, *, enable_thinking: bool = True) -> None:
        super().__init__(tokenizer)
        self.enable_thinking = enable_thinking

    def format_prompt(
        self, messages: Sequence[Message], tools: Sequence[Tool] | None = None
    ) -> str:
        messages = list(messages)
        last_query = self._last_query(messages)
        text = self._system(messages, tools)
        for index, message in enumerate(messages):
            role = message["role"]
            if role == "system":
                # The first one is in the system turn already.
                if index:
                    text += self._later_system(self._content(message))
            elif role == "user":
                text += f"<|im_start|>user\n{self._content(message)}<|im_end|>\n"
            elif role == "assistant":
                last = index == len(messages) - 1
                body = self._assistant(message, after_query=index > last_query, last=last)
                text += f"<|im_start|>assistant\n{body}<|im_end|>\n"
            elif role == "tool":
                text += self._tool_reply(messages, index)
            else:
                raise ValueError(
                    f"a message's role is {role!r}; the format has system, user, assistant and tool"
                )
        return text + "<|im_start|>assistant\n" + self._prefill()

    def format_response(self, message: Message) -> str:
        body = self._assistant(message, after_query=True, last=True)
        prefill = self._prefill()
        if not body.startswith(prefill):
            raise ValueError(
                "this assistant turn carries reasoning, and the formatter has thinking off: "
                "its prompt closes an empty <think> block, which tells the model not to "
                "reason. Build the formatter with enable_thinking=True, or drop the turn's "
                "reasoning"
            )
        return body.removeprefix(prefill) + "<|im_end|>"

    def format_continuation(self, messages: Sequence[Message]) -> str:
        replies = tool_replies(messages)
        # The template's newline after the sampled turn's <|im_end|>, then the replies' turn.
        text = "\n" + "".join(self._tool_reply(replies, index) for index in range(len(replies)))
        return text + "<|im_start|>assistant\n" + self._prefill()

    def parse_text(self, text: str, tools: Sequence[Tool] | None = None) -> dict[str, Any]:
        # The turn's whole text: what the prompt opened, then what the model wrote.
        body = self._prefill() + text
        reasoning = ""
        if "</think>" in body:
            # The first block is the turn's reasoning; whatever follows it is content.
            reasoning, _, body = body.partition("</think>")
            reasoning, body = reasoning.removeprefix("<think>").strip(), body.lstrip("\n")
        elif body.startswith("<think>"):
            # Cut off while it reasoned: the turn has no content yet.
            reasoning, body = body.removeprefix("<think>").strip(), ""
        content, calls = self._parse_calls(body, tools)
        message: dict[str, Any] = {"role": "assistant"}
        if reasoning:
            message["reasoning_content"] = reasoning
        message["content"] = content
        if calls:
            message["tool_calls"] = calls
        return message

    def _prefill(self) -> str:
        """What the generation prompt writes of the turn before the model does."""
        return "" if self.enable_thinking else "<think>\n\n</think>\n\n"

    def _content(self, message: Message) -> str:
        return text_content(message)

    def _tool_reply(self, messages: Sequence[Message], index: int) -> str:
        """The tool reply ``messages[index]``. A run of replies shares one user turn."""
        text = ""
        if index == 0 or messages[index - 1]["role"] != "tool":
            text += "<|im_start|>user"
        text += f"\n<tool_response>\n{self._content(messages[index])}\n</tool_response>"
        if index == len(messages) - 1 or messages[index + 1]["role"] != "tool":
            text += "<|im_end|>\n"
        return text

    def _system(self, messages: list[Message], tools: Sequence[Tool] | None) -> str:
        """The system turn: the first message when it is a system one, and the tools."""
        system = self._content(messages[0]) if messages[0]["role"] == "system" else None
        if tools:
            listed = "".join("\n" + json.dumps(tool, ensure_ascii=False) for tool in tools)
            text = "" if system is None else system + "\n\n"
            return f"<|im_start|>system\n{text}{_TOOLS_OPEN}{listed}{_TOOLS_CLOSE}<|im_end|>\n"
        return "" if system is None else f"<|im_start|>system\n{system}<|im_end|>\n"

    def _later_system(self, content: str) -> str:
        """A system message after the first: a turn of its own."""
        return f"<|im_start|>system\n{content}<|im_end|>\n"

    def _last_query(self, messages: list[Message]) -> int:
        """The index of the last user question: a user message that is not a
        tool's reply written as one."""
        for index in reversed(range(len(messages))):
            if messages[index]["role"] != "user":
                continue
            content = self._content(messages[index])
            if not (content.startswith("<tool_response>") and content.endswith("</tool_response>")):
                return index
        raise ValueError(
            "the conversation has no user message before the assistant turn, so there is no "
            "question for the turn to answer"
        )

    def _reasoning_and_content(self, message: Message) -> tuple[str, str]:
        """An assistant message's reasoning and its content. Reasoning written
        inline, as a ``<think>`` block in the content, is read out of it."""
        content = self._content(message)
        if isinstance(message.get("reasoning_content"), str):
            return reasoning_content(message), content
        if "</think>" in content:
            reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            return reasoning, content.split("</think>")[-1].lstrip("\n")
        return "", content

    def _assistant(self, message: Message, *, after_query: bool, last: bool) -> str:
        """An assistant turn between its ``<|im_start|>assistant`` line and its
        ``<|im_end|>``.

        Args:
            after_query: The turn comes after the last user question.
            last: It is the conversation's last message.
        """
        reasoning, content = self._reasoning_and_content(message)
        body = content
        # After the question, the last turn always has a block, and an earlier one only its own.
        if after_query and (last or reasoning):
            stripped = reasoning.strip("\n")
            body = f"<think>\n{stripped}\n</think>\n\n" + content.lstrip("\n")
        for position, call in enumerate(message.get("tool_calls") or ()):
            function = call.get("function", call)
            arguments = function.get("arguments") or {}
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            if position or content:
                body += "\n"
            body += f'<tool_call>\n{{"name": "{function["name"]}", "arguments": {arguments}}}\n'
            body += "</tool_call>"
        return body

    def _parse_calls(
        self, text: str, tools: Sequence[Tool] | None
    ) -> tuple[str, list[dict[str, Any]]]:
        """The content before a turn's tool calls, and the calls. When any of
        them cannot be read, the whole text is the content and there are none."""
        head, opened, rest = text.partition("<tool_call>")
        if not opened:
            return text, []
        blocks = _CALL.findall(opened + rest)
        if _CALL.sub("", opened + rest).strip():
            return text, []
        calls = [self._parse_call(block, tools) for block in blocks]
        if any(call is None for call in calls):
            return text, []
        return head.rstrip("\n"), [call for call in calls if call is not None]

    def _parse_call(self, block: str, tools: Sequence[Tool] | None) -> dict[str, Any] | None:
        """The call inside one ``<tool_call>`` block, or None when it is not one.
        Its arguments are JSON, which says their types itself."""
        try:
            call = json.loads(block)
        except json.JSONDecodeError:
            return None
        if not isinstance(call, Mapping) or not isinstance(call.get("name"), str):
            return None
        arguments = call.get("arguments", {})
        return tool_call(call["name"], arguments) if isinstance(arguments, Mapping) else None
