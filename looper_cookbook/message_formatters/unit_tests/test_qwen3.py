"""Qwen3's format, token by token. What
Qwen3.5's formatter inherits unchanged is tested with it, in ``test_qwen3_5.py``."""

from __future__ import annotations

import pytest

from looper_cookbook.message_formatters.qwen3 import Qwen3MessageFormatter
from looper_cookbook.message_formatters.unit_tests.tokenizers import pieces

START, END = "<|im_start|>", "<|im_end|>"
THINK, END_THINK = "<think>", "</think>"
CALL, END_CALL = "<tool_call>", "</tool_call>"

QUESTION = {"role": "user", "content": "What is 6 times 7?"}
ASKED = [START, "user\nWhat is 6 times 7?", END, "\n"]
WEATHER = {"type": "function", "function": {"name": "weather", "parameters": {}}}


def call(name: str, **arguments: object) -> dict[str, object]:
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.fixture
def thinking(qwen_tokenizer) -> Qwen3MessageFormatter:
    return Qwen3MessageFormatter(qwen_tokenizer)


@pytest.fixture
def direct(qwen_tokenizer) -> Qwen3MessageFormatter:
    return Qwen3MessageFormatter(qwen_tokenizer, enable_thinking=False)


def test_a_prompt_leaves_the_think_block_to_the_model(thinking, qwen_tokenizer) -> None:
    prompt = thinking.build_generation_prompt([QUESTION])
    assert pieces(qwen_tokenizer, prompt) == [*ASKED, START, "assistant\n"]


def test_a_prompt_with_thinking_off_closes_a_block_empty(direct, qwen_tokenizer) -> None:
    prompt = direct.build_generation_prompt([QUESTION])
    assert pieces(qwen_tokenizer, prompt)[len(ASKED) :] == [
        START,
        "assistant\n",
        THINK,
        "\n\n",
        END_THINK,
        "\n\n",
    ]


def test_content_keeps_the_whitespace_around_it(thinking, qwen_tokenizer) -> None:
    messages = [
        {"role": "system", "content": "Answer in one word. "},
        {"role": "user", "content": "  What is 6 times 7?"},
        {"role": "system", "content": "Now in two."},
    ]
    assert pieces(qwen_tokenizer, thinking.build_generation_prompt(messages)) == [
        START,
        "system\nAnswer in one word. ",
        END,
        "\n",
        START,
        "user\n  What is 6 times 7?",
        END,
        "\n",
        # A later system message is a turn of its own.
        START,
        "system\nNow in two.",
        END,
        "\n",
        START,
        "assistant\n",
    ]


def test_a_turn_trains_its_think_block_whether_or_not_it_reasoned(thinking, qwen_tokenizer) -> None:
    reasoned = {"role": "assistant", "reasoning_content": "\n6 * 7 = 42.\n", "content": "\n42"}
    plain = {"role": "assistant", "content": "42"}
    for turn, block in ((reasoned, "\n6 * 7 = 42.\n"), (plain, "\n\n")):
        (datum,) = thinking.build_supervised_examples([QUESTION, turn], max_seq_len=128)
        prompt = thinking.build_generation_prompt([QUESTION])
        assert list(datum.tokens[: len(prompt)]) == prompt
        assert pieces(qwen_tokenizer, datum.tokens[len(prompt) :]) == [
            THINK,
            block,
            END_THINK,
            "\n\n42",
            END,
        ]


def test_with_thinking_off_a_turn_trains_its_answer_alone(direct) -> None:
    (datum,) = direct.build_supervised_examples(
        [QUESTION, {"role": "assistant", "content": "42"}], max_seq_len=128
    )
    assert direct.targets(datum) == ["42<|im_end|>"]
    reasoned = {"role": "assistant", "reasoning_content": "6 * 7 = 42.", "content": "42"}
    with pytest.raises(ValueError, match="carries reasoning.*thinking off"):
        direct.build_supervised_examples([QUESTION, reasoned], max_seq_len=128)


def test_an_earlier_turn_after_the_question_keeps_a_block_only_if_it_reasoned(
    thinking, qwen_tokenizer
) -> None:
    messages = [
        QUESTION,
        {"role": "assistant", "reasoning_content": "Old.", "content": "42"},
        {"role": "user", "content": "Is it raining?"},
        {"role": "assistant", "tool_calls": [call("weather")]},
        {"role": "tool", "content": "rain"},
        {"role": "assistant", "reasoning_content": "Rain.", "tool_calls": [call("weather")]},
        {"role": "tool", "content": "rain"},
    ]
    prompt = qwen_tokenizer.decode(thinking.build_generation_prompt(messages))
    assert "Old." not in prompt
    assert prompt.count("<think>") == 1
    assert '<|im_start|>assistant\n<tool_call>\n{"name": "weather"' in prompt
    assert '<|im_start|>assistant\n<think>\nRain.\n</think>\n\n<tool_call>\n{"name"' in prompt


def test_the_system_turn_lists_tools_after_the_system_message(thinking, qwen_tokenizer) -> None:
    messages = [{"role": "system", "content": "Be brief."}, QUESTION]
    prompt = thinking.build_generation_prompt(messages, [WEATHER])
    system, rest = qwen_tokenizer.decode(prompt).split(END, 1)
    assert system.startswith(
        "<|im_start|>system\nBe brief.\n\n# Tools\n\nYou may call one or more functions"
    )
    assert (
        '<tools>\n{"type": "function", "function": {"name": "weather", "parameters": {}}}\n'
        "</tools>" in system
    )
    assert system.endswith('"arguments": <args-json-object>}\n</tool_call>')
    assert rest.startswith("\n<|im_start|>user\n")


def test_a_call_is_a_json_object_in_a_tool_call_block(direct, qwen_tokenizer) -> None:
    turn = {
        "role": "assistant",
        "content": "Let me look.",
        "tool_calls": [call("weather", city="Paris", days=2), call("convert", exact=True)],
    }
    response = qwen_tokenizer(direct.format_response(turn))["input_ids"]
    assert pieces(qwen_tokenizer, response) == [
        "Let me look.\n",
        CALL,
        '\n{"name": "weather", "arguments": {"city": "Paris", "days": 2}}\n',
        END_CALL,
        "\n",
        CALL,
        '\n{"name": "convert", "arguments": {"exact": true}}\n',
        END_CALL,
        END,
    ]
    assert direct.parse_response(response) == (turn, True)


def test_json_string_arguments_are_written_as_they_are(direct) -> None:
    as_string = {"function": {"name": "weather", "arguments": '{"city":"Paris"}'}}
    response = direct.format_response({"role": "assistant", "tool_calls": [as_string]})
    assert response == (
        '<tool_call>\n{"name": "weather", "arguments": {"city":"Paris"}}\n</tool_call><|im_end|>'
    )


def test_a_call_that_is_not_json_stays_in_the_content(direct, qwen_tokenizer) -> None:
    written = "<tool_call>\n{name: weather}\n</tool_call>"
    message, ended = direct.parse_response(qwen_tokenizer(written)["input_ids"])
    assert (message, ended) == ({"role": "assistant", "content": written}, False)


def test_a_sampled_turn_reads_back_with_its_reasoning(thinking, qwen_tokenizer) -> None:
    sampled = qwen_tokenizer("<think>\n6 * 7 = 42.\n</think>\n\n42<|im_end|>")["input_ids"]
    assert thinking.parse_response(sampled) == (
        {"role": "assistant", "reasoning_content": "6 * 7 = 42.", "content": "42"},
        True,
    )
    cut_off = qwen_tokenizer("<think>\n6 * 7 is")["input_ids"]
    assert thinking.parse_response(cut_off) == (
        {"role": "assistant", "reasoning_content": "6 * 7 is", "content": ""},
        False,
    )


def test_a_continued_prompt_is_the_formats_own_and_leaves_the_block_to_the_model(
    thinking, qwen_tokenizer
) -> None:
    turn = {
        "role": "assistant",
        "reasoning_content": "I should look.",
        "content": "",
        "tool_calls": [call("weather", city="Paris")],
    }
    replies = [{"role": "tool", "content": "21C"}]
    prompt = thinking.build_generation_prompt([QUESTION], [WEATHER])
    sampled = qwen_tokenizer(thinking.format_response(turn))["input_ids"]
    continued = thinking.continue_prompt(prompt + sampled, replies)
    assert continued == thinking.build_generation_prompt([QUESTION, turn, *replies], [WEATHER])
    assert pieces(qwen_tokenizer, continued)[-3:] == ["\n", START, "assistant\n"]
