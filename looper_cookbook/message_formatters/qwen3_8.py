"""Qwen3.8's message format: ``Qwen/Qwen3.8-27B``.

It is Qwen3.8's chat template written out, from `Qwen3.8-27B's
chat_template.jinja
<https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja>`_.
It keeps Qwen3.5's format (``qwen3_5.py``) and changes these parts:

- **Reasoning effort.** With thinking on, ``xhigh`` and ``low`` each put an
  instruction at the start of the system turn, making one when the
  conversation has none. ``medium`` puts none.
- **Reasoning is kept in history.** Every assistant turn has its ``<think>``
  block, wherever it stands, unless ``preserve_thinking`` is off; then only
  the turns after the last user question do, as in Qwen3.5.
- **Reasoning comes from ``reasoning_content`` alone.** A ``<think>`` block
  written inline in a message's content stays content.
- **A tool call's argument** that is not a string is JSON: ``true`` and
  ``null``, where Qwen3.5 writes ``True`` and ``None``.
- **An empty system message** writes no system turn.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from looper_cookbook.message_formatters.base import Message, Tool, reasoning_content
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter, tools_text

__all__ = ["REASONING_EFFORTS", "Qwen3_8MessageFormatter"]

# Each effort, and the instruction it puts in the system turn.
REASONING_EFFORTS = {
    "xhigh": (
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate "
        "key assumptions, consider plausible alternatives, and prioritize correctness, "
        "consistency, and clarity in the final answer."
    ),
    "medium": "",
    "low": (
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
        "directly to the conclusion without unnecessary elaboration."
    ),
}


class Qwen3_8MessageFormatter(Qwen3_5MessageFormatter):
    """Qwen3.8's format.

    Args:
        tokenizer: A Qwen3.8 model's tokenizer.
        enable_thinking: Whether the model is prompted to reason before it
            answers, as in Qwen3.5. Off, no effort's instruction is written.
        reasoning_effort: How much the model is told to reason: ``"xhigh"``,
            ``"medium"`` or ``"low"``.
        preserve_thinking: Whether the turns before the last user question
            keep their ``<think>`` blocks.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        enable_thinking: bool = True,
        reasoning_effort: str = "xhigh",
        preserve_thinking: bool = True,
    ) -> None:
        super().__init__(tokenizer, enable_thinking=enable_thinking)
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort is {reasoning_effort!r}; Qwen3.8 has "
                f"{', '.join(REASONING_EFFORTS)}"
            )
        self.reasoning_effort = reasoning_effort
        self.preserve_thinking = preserve_thinking

    def _system(self, messages: list[Message], tools: Sequence[Tool] | None) -> str:
        instruction = REASONING_EFFORTS[self.reasoning_effort] if self.enable_thinking else ""
        system = self._content(messages[0]) if messages[0]["role"] == "system" else ""
        parts = [part for part in (instruction, tools_text(tools) if tools else "", system) if part]
        return "<|im_start|>system\n" + "\n\n".join(parts) + "<|im_end|>\n" if parts else ""

    def _reasoning_and_content(self, message: Message) -> tuple[str, str]:
        return reasoning_content(message), self._content(message)

    def _keeps_thinking(self, after_query: bool) -> bool:
        return self.preserve_thinking or after_query

    def _argument(self, value: Any) -> str:
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
