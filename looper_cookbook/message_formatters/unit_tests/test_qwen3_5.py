"""Qwen3.5's format, token by token."""

from __future__ import annotations

import pytest

from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter
from looper_cookbook.message_formatters.unit_tests.tokenizers import pieces

START, END = "<|im_start|>", "<|im_end|>"
THINK, END_THINK = "<think>", "</think>"
CALL, END_CALL = "<tool_call>", "</tool_call>"
RESPONSE, END_RESPONSE = "<tool_response>", "</tool_response>"

QUESTION = {"role": "user", "content": "What is 6 times 7?"}
ASKED = [START, "user\nWhat is 6 times 7?", END, "\n"]
WEATHER = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "The weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
WEATHER_JSON = (
    '{"type": "function", "function": {"name": "weather", "description": "The weather in a '
    'city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}'
)


def call(name: str, **arguments: object) -> dict[str, object]:
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.fixture
def thinking(qwen_tokenizer) -> Qwen3_5MessageFormatter:
    return Qwen3_5MessageFormatter(qwen_tokenizer)


@pytest.fixture
def direct(qwen_tokenizer) -> Qwen3_5MessageFormatter:
    return Qwen3_5MessageFormatter(qwen_tokenizer, enable_thinking=False)


def test_a_prompt_opens_the_think_block_for_the_model(thinking, qwen_tokenizer) -> None:
    prompt = thinking.build_generation_prompt([QUESTION])
    assert pieces(qwen_tokenizer, prompt) == [*ASKED, START, "assistant\n", THINK, "\n"]


def test_a_prompt_with_thinking_off_closes_the_block_empty(direct, qwen_tokenizer) -> None:
    prompt = direct.build_generation_prompt([QUESTION])
    assert pieces(qwen_tokenizer, prompt) == [
        *ASKED,
        START,
        "assistant\n",
        THINK,
        "\n\n",
        END_THINK,
        "\n\n",
    ]


def test_content_loses_the_whitespace_around_it(direct, qwen_tokenizer) -> None:
    messages = [
        {"role": "system", "content": "\nAnswer in one word. "},
        {"role": "user", "content": "  What is 6 times 7?\n"},
    ]
    assert pieces(qwen_tokenizer, direct.build_generation_prompt(messages))[:8] == [
        START,
        "system\nAnswer in one word.",
        END,
        "\n",
        *ASKED,
    ]


def test_only_turns_after_the_last_question_keep_their_think_block(
    thinking, qwen_tokenizer
) -> None:
    messages = [
        QUESTION,
        {"role": "assistant", "reasoning_content": "6 * 7 = 42.", "content": "42"},
        {"role": "user", "content": "Is it raining in Paris?"},
        {"role": "assistant", "tool_calls": [call("weather", city="Paris")]},
        {"role": "tool", "content": "rain"},
    ]
    assert pieces(qwen_tokenizer, thinking.build_generation_prompt(messages)) == [
        *ASKED,
        # Before the last question: the reasoning is gone.
        START,
        "assistant\n42",
        END,
        "\n",
        START,
        "user\nIs it raining in Paris?",
        END,
        "\n",
        # After it: a block, empty when the turn did not reason.
        START,
        "assistant\n",
        THINK,
        "\n\n",
        END_THINK,
        "\n\n",
        CALL,
        "\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n",
        END_CALL,
        END,
        "\n",
        START,
        "user\n",
        RESPONSE,
        "\nrain\n",
        END_RESPONSE,
        END,
        "\n",
        START,
        "assistant\n",
        THINK,
        "\n",
    ]


def test_an_answer_trains_after_its_prompt_through_the_end_token(direct, qwen_tokenizer) -> None:
    conversation = [QUESTION, {"role": "assistant", "content": "42"}]
    (datum,) = direct.build_supervised_examples(conversation, max_seq_len=128)
    prompt = direct.build_generation_prompt([QUESTION])
    assert list(datum.tokens[: len(prompt)]) == prompt
    assert pieces(qwen_tokenizer, datum.tokens[len(prompt) :]) == ["42", END]
    assert list(datum.weights) == [0] * len(prompt) + [1] * 3
    assert direct.targets(datum) == ["42<|im_end|>"]


def test_a_turns_reasoning_trains_inside_the_block_the_prompt_opened(
    thinking, qwen_tokenizer
) -> None:
    conversation = [
        QUESTION,
        {"role": "assistant", "reasoning_content": " 6 * 7 = 42.\n", "content": "42"},
    ]
    (datum,) = thinking.build_supervised_examples(conversation, max_seq_len=128)
    prompt = thinking.build_generation_prompt([QUESTION])
    assert list(datum.tokens[: len(prompt)]) == prompt
    assert pieces(qwen_tokenizer, datum.tokens[len(prompt) :]) == [
        "6 * 7 = 42.\n",
        END_THINK,
        "\n\n42",
        END,
    ]


def test_a_turn_without_reasoning_closes_the_open_block_after_the_prompts_tokens(
    thinking, qwen_tokenizer
) -> None:
    conversation = [QUESTION, {"role": "assistant", "content": "42"}]
    (datum,) = thinking.build_supervised_examples(conversation, max_seq_len=128)
    prompt = thinking.build_generation_prompt([QUESTION])
    newline, two_newlines = (qwen_tokenizer.encode(text)[0] for text in ("\n", "\n\n"))
    # The prompt's newline and the turn's stay two tokens: the model is sampled after the first.
    assert list(datum.tokens[: len(prompt)]) == prompt
    assert list(datum.tokens[len(prompt) - 1 : len(prompt) + 1]) == [newline, newline]
    assert two_newlines != newline
    assert thinking.targets(datum) == ["\n</think>\n\n42<|im_end|>"]


def test_a_turn_with_reasoning_is_refused_with_thinking_off(direct) -> None:
    conversation = [
        QUESTION,
        {"role": "assistant", "reasoning_content": "6 * 7 = 42.", "content": "42"},
    ]
    with pytest.raises(ValueError, match=r"messages\[1\].*carries reasoning.*thinking off"):
        direct.build_supervised_examples(conversation, max_seq_len=128)


def test_reasoning_written_inline_is_read_out_of_the_content(thinking) -> None:
    inline = {"role": "assistant", "content": "<think>\n6 * 7 = 42.\n</think>\n\n42"}
    field = {"role": "assistant", "reasoning_content": "6 * 7 = 42.", "content": "42"}
    assert thinking.format_response(inline) == thinking.format_response(field)


def test_tools_come_before_the_system_message_in_the_system_turn(direct, qwen_tokenizer) -> None:
    messages = [{"role": "system", "content": "Be brief."}, QUESTION]
    prompt = direct.build_generation_prompt(messages, [WEATHER])
    # The instructions name <tool_call>, a token of its own, so the turn is read as text.
    system, rest = qwen_tokenizer.decode(prompt).split(END, 1)
    assert system.startswith(
        "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        f"{WEATHER_JSON}\n</tools>\n\nIf you choose to call a function ONLY reply"
    )
    assert system.endswith("\n</IMPORTANT>\n\nBe brief.")
    assert rest.startswith("\n<|im_start|>user\nWhat is 6 times 7?<|im_end|>\n")


def test_a_call_gives_each_argument_in_a_parameter_block(direct, qwen_tokenizer) -> None:
    turn = {
        "role": "assistant",
        "content": "Let me look.",
        "tool_calls": [
            call("weather", city="Paris", days=2, hourly=True, units=None),
            call("convert", scales=["F", "K"], note={"to": "größe"}),
        ],
    }
    response = qwen_tokenizer(direct.format_response(turn))["input_ids"]
    assert pieces(qwen_tokenizer, response) == [
        "Let me look.\n\n",
        CALL,
        "\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n"
        "</parameter>\n<parameter=hourly>\nTrue\n</parameter>\n<parameter=units>\nNone\n"
        "</parameter>\n</function>\n",
        END_CALL,
        "\n",
        CALL,
        '\n<function=convert>\n<parameter=scales>\n["F", "K"]\n</parameter>\n<parameter=note>\n'
        '{"to": "größe"}\n</parameter>\n</function>\n',
        END_CALL,
        END,
    ]


def test_a_call_with_json_string_arguments_formats_as_one_with_a_mapping(direct) -> None:
    as_string = {"function": {"name": "weather", "arguments": '{"city": "Paris"}'}}
    assert direct.format_response({"role": "assistant", "tool_calls": [as_string]}) == (
        direct.format_response({"role": "assistant", "tool_calls": [call("weather", city="Paris")]})
    )


def test_a_run_of_tool_replies_shares_one_user_turn(direct, qwen_tokenizer) -> None:
    messages = [
        QUESTION,
        {"role": "assistant", "tool_calls": [call("a"), call("b")]},
        {"role": "tool", "content": " first "},
        {"role": "tool", "content": "second"},
    ]
    prompt = pieces(qwen_tokenizer, direct.build_generation_prompt(messages))
    assert prompt[prompt.index(RESPONSE) - 2 :][:10] == [
        START,
        "user\n",
        RESPONSE,
        "\nfirst\n",
        END_RESPONSE,
        "\n",
        RESPONSE,
        "\nsecond\n",
        END_RESPONSE,
        END,
    ]


def test_a_sampled_turn_reads_back_with_its_reasoning(thinking, qwen_tokenizer) -> None:
    sampled = qwen_tokenizer("6 * 7 = 42.\n</think>\n\n42<|im_end|>")["input_ids"]
    assert thinking.parse_response(sampled) == (
        {"role": "assistant", "reasoning_content": "6 * 7 = 42.", "content": "42"},
        True,
    )


def test_a_turn_cut_off_while_it_reasons_has_no_content(thinking, direct, qwen_tokenizer) -> None:
    sampled = qwen_tokenizer("6 * 7 is")["input_ids"]
    assert thinking.parse_response(sampled) == (
        {"role": "assistant", "reasoning_content": "6 * 7 is", "content": ""},
        False,
    )
    # With thinking off the prompt closed the block, so the same text is the answer.
    assert direct.parse_response(sampled) == ({"role": "assistant", "content": "6 * 7 is"}, False)


def test_a_block_the_model_writes_after_its_reasoning_stays_in_the_content(
    thinking, direct, qwen_tokenizer
) -> None:
    sampled = qwen_tokenizer("One.\n</think>\n\n<think>Two.</think>42")["input_ids"]
    message, _ = thinking.parse_response(sampled)
    assert message == {
        "role": "assistant",
        "reasoning_content": "One.",
        "content": "<think>Two.</think>42",
    }
    again = qwen_tokenizer("<think>Two.</think>42")["input_ids"]
    assert direct.parse_response(again)[0] == {
        "role": "assistant",
        "content": "<think>Two.</think>42",
    }


FORECAST = {
    "type": "function",
    "function": {
        "name": "forecast",
        "description": "A city's forecast.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "zip": {"type": "string"},
                "days": {"type": "integer"},
                "hourly": {"type": "boolean"},
                "units": {"type": ["string", "null"]},
                "note": {"type": "object"},
            },
        },
    },
}


def test_sampled_calls_read_back_with_the_types_their_tool_declares(direct, qwen_tokenizer) -> None:
    arguments = {
        "city": "Paris",
        "zip": "75001",
        "days": 2,
        "hourly": True,
        "units": None,
        "note": {"to": "F"},
    }
    turn = {
        "role": "assistant",
        "content": "Let me look.",
        "tool_calls": [call("forecast", **arguments), call("convert")],
    }
    sampled = qwen_tokenizer(direct.format_response(turn))["input_ids"]
    # A string that spells a number stays a string, since its tool says it is one.
    assert direct.parse_response(sampled, [FORECAST]) == (turn, True)


def test_without_the_tools_every_argument_is_the_text_the_model_wrote(
    direct, qwen_tokenizer
) -> None:
    turn = {
        "role": "assistant",
        "content": "",
        "tool_calls": [call("forecast", zip="75001", days=2, hourly=True, note={"to": "F"})],
    }
    response = direct.format_response(turn)
    message, _ = direct.parse_response(qwen_tokenizer(response)["input_ids"])
    assert message["tool_calls"] == [
        call("forecast", zip="75001", days="2", hourly="True", note='{"to": "F"}')
    ]
    # Text is written back as it was read, so the next prompt keeps the model's own turn.
    assert direct.format_response(message) == response


@pytest.mark.parametrize(
    "written",
    [
        "<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</function>\n</tool_call>",
        "<tool_call>\n<function=weather>\n</function>\n</tool_call> and then some text",
        "<tool_call>\n<function=weather>\n",
    ],
)
def test_a_call_that_cannot_be_read_stays_in_the_content(direct, qwen_tokenizer, written) -> None:
    message, _ = direct.parse_response(qwen_tokenizer(f"Look.\n\n{written}")["input_ids"])
    assert message == {"role": "assistant", "content": f"Look.\n\n{written}"}


@pytest.mark.parametrize(
    ("messages", "why"),
    [
        ([{"role": "system", "content": "Be brief."}], "no user message"),
        ([QUESTION, {"role": "system", "content": "Be brief."}], "system message comes first"),
        ([{"role": "user", "content": [{"type": "image"}]}], "not text"),
        ([QUESTION, {"role": "narrator", "content": "Later."}], "role is 'narrator'"),
    ],
)
def test_a_conversation_the_format_cannot_hold_is_refused(direct, messages, why) -> None:
    with pytest.raises(ValueError, match=why):
        direct.build_generation_prompt(messages)


CALLED = {
    "role": "assistant",
    "reasoning_content": "I should look.",
    "content": "",
    "tool_calls": [call("weather", city="Paris"), call("weather", city="Rome")],
}
REPLIES = [{"role": "tool", "content": "21C"}, {"role": "tool", "content": " 25C "}]


def test_a_continued_prompt_keeps_the_sampled_tokens_and_adds_the_replies(
    thinking, qwen_tokenizer
) -> None:
    prompt = thinking.build_generation_prompt([QUESTION], [WEATHER])
    # A turn written as no formatter writes it: it stays as it was sampled.
    sampled = qwen_tokenizer("look\n</think>\n<tool_call>odd</tool_call><|im_end|>")["input_ids"]
    continued = thinking.continue_prompt(prompt + sampled, REPLIES)
    assert continued[: len(prompt) + len(sampled)] == prompt + sampled
    assert pieces(qwen_tokenizer, continued[len(prompt) + len(sampled) :]) == [
        "\n",
        START,
        "user\n",
        RESPONSE,
        "\n21C\n",
        END_RESPONSE,
        "\n",
        RESPONSE,
        "\n25C\n",
        END_RESPONSE,
        END,
        "\n",
        START,
        "assistant\n",
        THINK,
        "\n",
    ]


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_a_continued_prompt_is_the_formats_own_after_a_turn_written_its_way(
    qwen_tokenizer, enable_thinking: bool
) -> None:
    formatter = Qwen3_5MessageFormatter(qwen_tokenizer, enable_thinking=enable_thinking)
    turn = CALLED if enable_thinking else {**CALLED, "reasoning_content": ""}
    prompt = formatter.build_generation_prompt([QUESTION], [WEATHER])
    sampled = qwen_tokenizer(formatter.format_response(turn))["input_ids"]
    whole = formatter.build_generation_prompt([QUESTION, turn, *REPLIES], [WEATHER])
    assert formatter.continue_prompt(prompt + sampled, REPLIES) == whole


def test_a_transcript_the_sampler_left_open_gets_its_end_token_first(
    direct, qwen_tokenizer
) -> None:
    prompt = direct.build_generation_prompt([QUESTION])
    sampled = qwen_tokenizer("<tool_call>x</tool_call>")["input_ids"]
    closed = [*prompt, *sampled, direct.end_of_turn_id]
    assert direct.continue_prompt(prompt + sampled, REPLIES) == direct.continue_prompt(
        closed, REPLIES
    )


@pytest.mark.parametrize(
    "transcript,messages,why",
    [
        ([1, 2], [], "continued with at least one tool reply"),
        ([1, 2], [{"role": "user", "content": "And Rome?"}], "this message's role is 'user'"),
        ([1, 2], [*REPLIES, {"role": "assistant", "content": "x"}], "role is 'assistant'"),
        ([], REPLIES, "the transcript is empty"),
    ],
)
def test_only_tool_replies_continue_a_prompt(direct, transcript, messages, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        direct.continue_prompt(transcript, messages)
