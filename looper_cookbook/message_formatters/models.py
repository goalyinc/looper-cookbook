"""The formatter of each model Looper serves.

A recipe names its model as its run does, by its Hugging Face name, and gets
the model's formatter here. A new model is a line in ``MESSAGE_FORMATTERS``,
and a new family a module beside this one. For a model that is not listed,
such as one of Qwen3's, build its family's formatter directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from looper_cookbook.message_formatters.base import MessageFormatter
from looper_cookbook.message_formatters.muse_glimmer import MuseGlimmerMessageFormatter
from looper_cookbook.message_formatters.qwen3 import Qwen3MessageFormatter
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter
from looper_cookbook.message_formatters.qwen3_8 import Qwen3_8MessageFormatter

__all__ = ["MESSAGE_FORMATTERS", "message_formatter"]

MESSAGE_FORMATTERS: Mapping[str, type[MessageFormatter]] = {
    "Qwen/Qwen3.5-4B": Qwen3_5MessageFormatter,
    "Qwen/Qwen3.5-9B": Qwen3_5MessageFormatter,
    "Qwen/Qwen3.5-35B-A3B": Qwen3_5MessageFormatter,
    "Qwen/Qwen3.8-27B": Qwen3_8MessageFormatter,
    "meta-models/Muse-Glimmer-30B": MuseGlimmerMessageFormatter,
}


def message_formatter(
    model: str, tokenizer: Any, *, enable_thinking: bool = True
) -> MessageFormatter:
    """The formatter of ``model``, over its tokenizer.

    Args:
        model: The model's Hugging Face name, as ``TrainingConfig.model`` names it.
        tokenizer: The model's Hugging Face tokenizer.
        enable_thinking: A Qwen model's thinking mode. Muse Glimmer has no
            such mode: it reasons as much as its reasoning strength says,
            whatever this is.
    """
    if model not in MESSAGE_FORMATTERS:
        raise ValueError(
            f"{model} has no message formatter here; the models are "
            f"{', '.join(MESSAGE_FORMATTERS)}. For another model of a family, build its "
            "formatter directly, as Qwen3MessageFormatter(tokenizer)"
        )
    formatter = MESSAGE_FORMATTERS[model]
    if issubclass(formatter, Qwen3MessageFormatter):
        return formatter(tokenizer, enable_thinking=enable_thinking)
    return formatter(tokenizer)
