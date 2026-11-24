"""Each model Looper serves, and its formatter."""

from __future__ import annotations

import pytest

from looper_cookbook.message_formatters.models import MESSAGE_FORMATTERS, message_formatter
from looper_cookbook.message_formatters.muse_glimmer import MuseGlimmerMessageFormatter
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter
from looper_cookbook.message_formatters.qwen3_8 import Qwen3_8MessageFormatter

QUESTION = [{"role": "user", "content": "What is 6 times 7?"}]


@pytest.mark.parametrize(
    ("model", "formatter"),
    [
        ("Qwen/Qwen3.5-4B", Qwen3_5MessageFormatter),
        ("Qwen/Qwen3.5-9B", Qwen3_5MessageFormatter),
        ("Qwen/Qwen3.5-35B-A3B", Qwen3_5MessageFormatter),
        ("Qwen/Qwen3.8-27B", Qwen3_8MessageFormatter),
    ],
)
def test_a_qwen_model_gets_its_formatter_in_the_mode_asked(
    qwen_tokenizer, model: str, formatter: type
) -> None:
    thinking = message_formatter(model, qwen_tokenizer)
    direct = message_formatter(model, qwen_tokenizer, enable_thinking=False)
    assert type(thinking) is type(direct) is formatter
    assert thinking.format_prompt(QUESTION).endswith("<|im_start|>assistant\n<think>\n")
    assert direct.format_prompt(QUESTION).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_muse_glimmer_has_no_thinking_mode(muse_tokenizer) -> None:
    model = "meta-models/Muse-Glimmer-30B"
    thinking = message_formatter(model, muse_tokenizer)
    direct = message_formatter(model, muse_tokenizer, enable_thinking=False)
    assert type(thinking) is type(direct) is MuseGlimmerMessageFormatter
    assert thinking.format_prompt(QUESTION) == direct.format_prompt(QUESTION)


def test_a_model_without_a_formatter_is_refused(qwen_tokenizer) -> None:
    with pytest.raises(ValueError, match="Qwen/Qwen3-8B has no message formatter here"):
        message_formatter("Qwen/Qwen3-8B", qwen_tokenizer)
    assert len(MESSAGE_FORMATTERS) == 5
