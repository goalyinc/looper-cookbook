"""What Qwen3.8's format changes of Qwen3.5's, token by token. The rest is
tested in ``test_qwen3_5.py``."""

from __future__ import annotations

import pytest

from looper_cookbook.message_formatters.qwen3_8 import REASONING_EFFORTS, Qwen3_8MessageFormatter
from looper_cookbook.message_formatters.unit_tests.tokenizers import pieces

START, END = "<|im_start|>", "<|im_end|>"
THINK, END_THINK = "<think>", "</think>"

QUESTION = {"role": "user", "content": "What is 6 times 7?"}
ASKED = [START, "user\nWhat is 6 times 7?", END, "\n"]
OPENED = [START, "assistant\n", THINK, "\n"]
WEATHER = {"type": "function", "function": {"name": "weather", "parameters": {}}}


@pytest.mark.parametrize("effort", ["xhigh", "low"])
def test_an_effort_puts_its_instruction_in_a_system_turn(qwen_tokenizer, effort: str) -> None:
    formatter = Qwen3_8MessageFormatter(qwen_tokenizer, reasoning_effort=effort)
    assert pieces(qwen_tokenizer, formatter.build_generation_prompt([QUESTION])) == [
        START,
        f"system\n{REASONING_EFFORTS[effort]}",
        END,
        "\n",
        *ASKED,
        *OPENED,
    ]
    assert REASONING_EFFORTS[effort].startswith(f"Reasoning effort is set to {effort}.")


@pytest.mark.parametrize(
    "settings",
    [{"reasoning_effort": "medium"}, {"reasoning_effort": "xhigh", "enable_thinking": False}],
)
def test_medium_effort_and_thinking_off_put_no_instruction(qwen_tokenizer, settings) -> None:
    formatter = Qwen3_8MessageFormatter(qwen_tokenizer, **settings)
    messages = [{"role": "system", "content": " "}, QUESTION]
    # An empty system message writes no system turn either.
    assert pieces(qwen_tokenizer, formatter.build_generation_prompt(messages))[:4] == ASKED


def test_the_instruction_comes_before_the_tools_and_the_system_message(qwen_tokenizer) -> None:
    formatter = Qwen3_8MessageFormatter(qwen_tokenizer, reasoning_effort="low")
    messages = [{"role": "system", "content": "Be brief."}, QUESTION]
    low = REASONING_EFFORTS["low"]
    system = qwen_tokenizer.decode(formatter.build_generation_prompt(messages)).split(END)[0]
    assert system == f"<|im_start|>system\n{low}\n\nBe brief."
    with_tools = formatter.build_generation_prompt(messages, [WEATHER])
    system = qwen_tokenizer.decode(with_tools).split(END)[0]
    assert system.startswith(f"<|im_start|>system\n{low}\n\n# Tools\n\nYou have access")
    assert system.endswith("</IMPORTANT>\n\nBe brief.")


def test_an_effort_qwen3_8_does_not_have_is_refused(qwen_tokenizer) -> None:
    with pytest.raises(ValueError, match="reasoning_effort is 'high'.*xhigh, medium, low"):
        Qwen3_8MessageFormatter(qwen_tokenizer, reasoning_effort="high")


CONVERSATION = [
    QUESTION,
    {"role": "assistant", "reasoning_content": "6 * 7 = 42.", "content": "42"},
    {"role": "user", "content": "And 6 times 8?"},
    {"role": "assistant", "content": "48"},
    {"role": "user", "content": "And 6 times 9?"},
]


def test_every_turn_keeps_its_think_block_in_history(qwen_tokenizer) -> None:
    formatter = Qwen3_8MessageFormatter(qwen_tokenizer, reasoning_effort="medium")
    assert pieces(qwen_tokenizer, formatter.build_generation_prompt(CONVERSATION)) == [
        *ASKED,
        START,
        "assistant\n",
        THINK,
        "\n6 * 7 = 42.\n",
        END_THINK,
        "\n\n42",
        END,
        "\n",
        START,
        "user\nAnd 6 times 8?",
        END,
        "\n",
        # A turn that did not reason keeps an empty block.
        START,
        "assistant\n",
        THINK,
        "\n\n",
        END_THINK,
        "\n\n48",
        END,
        "\n",
        START,
        "user\nAnd 6 times 9?",
        END,
        "\n",
        *OPENED,
    ]


def test_without_preserve_thinking_history_loses_its_blocks(qwen_tokenizer) -> None:
    formatter = Qwen3_8MessageFormatter(
        qwen_tokenizer, reasoning_effort="medium", preserve_thinking=False
    )
    prompt = pieces(qwen_tokenizer, formatter.build_generation_prompt(CONVERSATION))
    assert prompt[4:8] == [START, "assistant\n42", END, "\n"]
    assert prompt.count(THINK) == 1


def test_a_think_block_written_inline_stays_content(qwen_tokenizer) -> None:
    formatter = Qwen3_8MessageFormatter(qwen_tokenizer)
    inline = {"role": "assistant", "content": "<think>6 * 7 = 42.</think>42"}
    assert formatter.format_response(inline) == (
        "\n</think>\n\n<think>6 * 7 = 42.</think>42<|im_end|>"
    )


def test_an_argument_that_is_not_a_string_is_json(qwen_tokenizer) -> None:
    formatter = Qwen3_8MessageFormatter(qwen_tokenizer, enable_thinking=False)
    arguments = {"city": "Paris", "hourly": True, "units": None, "days": 2}
    turn = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "weather", "arguments": arguments}}
        ],
    }
    response = formatter.format_response(turn)
    assert response == (
        "<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n"
        "<parameter=hourly>\ntrue\n</parameter>\n<parameter=units>\nnull\n</parameter>\n"
        "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>"
    )
    properties = {
        "city": {"type": "string"},
        "hourly": {"type": "boolean"},
        "units": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "days": {"type": "integer"},
    }
    tool = {
        "type": "function",
        "function": {"name": "weather", "parameters": {"properties": properties}},
    }
    sampled = qwen_tokenizer(response)["input_ids"]
    assert formatter.parse_response(sampled, [tool]) == (turn, True)


def test_a_continued_prompt_is_the_formats_own(qwen_tokenizer) -> None:
    formatter = Qwen3_8MessageFormatter(qwen_tokenizer)
    call = {"type": "function", "function": {"name": "weather", "arguments": {"days": 2}}}
    turn = {"role": "assistant", "reasoning_content": "Look.", "content": "", "tool_calls": [call]}
    replies = [{"role": "tool", "content": "21C"}]
    prompt = formatter.build_generation_prompt([QUESTION])
    sampled = qwen_tokenizer(formatter.format_response(turn))["input_ids"]
    whole = formatter.build_generation_prompt([QUESTION, turn, *replies])
    assert formatter.continue_prompt(prompt + sampled, replies) == whole
