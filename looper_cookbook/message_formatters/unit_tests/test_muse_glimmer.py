"""Muse Glimmer's format, token by token."""

from __future__ import annotations

import datetime

import pytest

from looper_cookbook.message_formatters.muse_glimmer import MuseGlimmerMessageFormatter
from looper_cookbook.message_formatters.unit_tests.tokenizers import pieces

BOS, START, MESSAGE, EOM, EOT = (
    "<|begin_of_text|>",
    "<|start|>",
    "<|message|>",
    "<|eom|>",
    "<|eot|>",
)
QUESTION = {"role": "user", "content": "What is 6 times 7?"}
ASKED = [START, "user", MESSAGE, "What is 6 times 7?", EOT]
DEFAULT_SYSTEM = (
    "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\nCurrent date: 2026-11-18."
    '\n\nReasoning strength: high.\n\n# Valid recipients: "self", "user".'
)
LOOKUP = {
    "type": "function",
    "function": {
        "name": "weather.lookup",
        "description": "The weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
CONVERT = {
    "type": "function",
    "function": {"name": "convert", "description": "Convert.", "parameters": {}},
}


def call(name: str, call_id: str | None = None, **arguments: object) -> dict[str, object]:
    made = {"type": "function", "function": {"name": name, "arguments": arguments}}
    return made if call_id is None else {**made, "id": call_id}


@pytest.fixture
def formatter(muse_tokenizer) -> MuseGlimmerMessageFormatter:
    return MuseGlimmerMessageFormatter(muse_tokenizer, current_date="2026-11-18")


def test_a_conversation_without_a_system_message_gets_the_default_one(
    formatter, muse_tokenizer
) -> None:
    prompt = formatter.build_generation_prompt([QUESTION])
    assert pieces(muse_tokenizer, prompt) == [
        BOS,
        START,
        "system",
        MESSAGE,
        DEFAULT_SYSTEM,
        EOT,
        *ASKED,
        START,
        "assistant",
    ]


def test_the_default_system_turn_states_the_day_the_formatter_was_built(muse_tokenizer) -> None:
    formatter = MuseGlimmerMessageFormatter(muse_tokenizer)
    today = datetime.date.today().isoformat()
    assert f"\nCurrent date: {today}.\n" in formatter.format_prompt([QUESTION])


@pytest.mark.parametrize(
    ("system", "written"),
    [
        (
            "Be brief.",
            'Be brief.\n\nReasoning strength: xhigh.\n\n# Valid recipients: "self", "user".',
        ),
        # A system message that states its own strength keeps it, named as the model names it.
        (
            "Be brief. Reasoning effort: low.",
            'Be brief. Reasoning strength: low.\n\n# Valid recipients: "self", "user".',
        ),
    ],
)
def test_a_system_message_gets_the_strength_and_the_recipients(
    muse_tokenizer, system: str, written: str
) -> None:
    formatter = MuseGlimmerMessageFormatter(muse_tokenizer, reasoning_strength="xhigh")
    prompt = formatter.build_generation_prompt([{"role": "system", "content": system}, QUESTION])
    assert pieces(muse_tokenizer, prompt)[:6] == [BOS, START, "system", MESSAGE, written, EOT]


def test_a_strength_muse_glimmer_does_not_have_is_refused(muse_tokenizer) -> None:
    with pytest.raises(ValueError, match="reasoning_strength is 'max'.*low, medium, high, xhigh"):
        MuseGlimmerMessageFormatter(muse_tokenizer, reasoning_strength="max")


def test_an_answer_trains_to_the_user_through_eot(formatter, muse_tokenizer) -> None:
    conversation = [QUESTION, {"role": "assistant", "content": "42"}]
    (datum,) = formatter.build_supervised_examples(conversation, max_seq_len=512)
    prompt = formatter.build_generation_prompt([QUESTION])
    assert list(datum.tokens[: len(prompt)]) == prompt
    assert pieces(muse_tokenizer, datum.tokens[len(prompt) :]) == [" to=user", MESSAGE, "42", EOT]
    assert list(datum.weights) == [0] * len(prompt) + [1] * (len(datum.tokens) - len(prompt))


def test_a_turns_reasoning_goes_to_self_before_its_answer(formatter, muse_tokenizer) -> None:
    turn = {"role": "assistant", "reasoning_content": "6 * 7 = 42.", "content": "42"}
    (datum,) = formatter.build_supervised_examples([QUESTION, turn], max_seq_len=512)
    assert formatter.targets(datum) == [
        " to=self<|message|>6 * 7 = 42.<|eom|><|start|>assistant to=user<|message|>42<|eot|>"
    ]
    sampled = [token for token, weight in zip(datum.tokens, datum.weights, strict=True) if weight]
    assert formatter.parse_response(sampled) == (turn, True)


def test_sampling_stops_on_eot_not_on_the_tokenizers_eos(formatter, muse_tokenizer) -> None:
    assert muse_tokenizer.eos_token == "<|end_of_text|>"
    assert formatter.stop_token_ids == (muse_tokenizer.convert_tokens_to_ids(EOT),)


def test_the_system_turn_lists_the_tools_and_their_recipients(formatter, muse_tokenizer) -> None:
    prompt = formatter.build_generation_prompt([QUESTION], [LOOKUP, CONVERT])
    system = pieces(muse_tokenizer, prompt)[4]
    assert system.startswith(
        "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\nCurrent date: "
        "2026-11-18.\n\nReasoning strength: high.\n\nIn this environment you have access to a "
        "set of tools"
    )
    assert (
        "Here are the functions available in JSONSchema format:\n// Tool metadata\n"
        '{"name": "weather", "description": ""}\n{"name": "convert", "description": ""}\n'
        '// Function schemas\n{"name": "weather.lookup", "description": "The weather in a city.", '
        '"parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}\n'
        '{"name": "convert", "description": "Convert.", "parameters": {}}\n\n'
        "Here's an example of how to call a function in the tool set:\n"
    ) in system
    # A tool's namespace, the name before its first dot, is a recipient.
    assert system.endswith(
        '</atem:function_calls>\n\n# Valid recipients: "self", "weather.*", "convert.*", "user".'
    )


def test_a_call_goes_to_its_tool_as_a_function_calls_block(formatter, muse_tokenizer) -> None:
    turn = {
        "role": "assistant",
        "reasoning_content": "Look it up.",
        "content": "",
        "tool_calls": [
            call("weather.lookup", city="Paris", days=2, hourly=True, units=None),
            call("convert", scales=["F", "K"], note={"to": "größe"}, text=" two\nlines "),
        ],
    }
    response = muse_tokenizer(formatter.format_response(turn))["input_ids"]
    assert pieces(muse_tokenizer, response) == [
        " to=self",
        MESSAGE,
        "Look it up.",
        EOM,
        START,
        "assistant to=weather.lookup",
        MESSAGE,
        '<atem:function_calls>\n<atem:invoke name="weather.lookup">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        '<atem:parameter name="days">2</atem:parameter>\n'
        '<atem:parameter name="hourly">true</atem:parameter>\n'
        '<atem:parameter name="units">null</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>",
        # Every call but the turn's last ends with <|eom|>.
        EOM,
        START,
        "assistant to=convert",
        MESSAGE,
        '<atem:function_calls>\n<atem:invoke name="convert">\n'
        '<atem:parameter name="scales">["F", "K"]</atem:parameter>\n'
        '<atem:parameter name="note">{"to": "größe"}</atem:parameter>\n'
        '<atem:parameter name="text"> two\nlines </atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>",
        EOT,
    ]
    typed = {
        "weather.lookup": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "hourly": {"type": "boolean"},
            "units": {"type": ["string", "null"]},
        },
        "convert": {
            "scales": {"type": "array"},
            "note": {"type": "object"},
            "text": {"type": "string"},
        },
    }
    tools = [
        {
            "type": "function",
            "function": {"name": name, "description": "", "parameters": {"properties": properties}},
        }
        for name, properties in typed.items()
    ]
    assert formatter.parse_response(response, tools) == (turn, True)
    # Without the tools, each argument is the text the model wrote.
    untyped, _ = formatter.parse_response(response)
    assert untyped["tool_calls"][0] == call(
        "weather.lookup", city="Paris", days="2", hourly="true", units="null"
    )


def test_text_before_a_call_goes_to_the_user_and_the_turn_goes_on(
    formatter, muse_tokenizer
) -> None:
    turn = {
        "role": "assistant",
        "content": "Let me look.",
        "tool_calls": [call("convert", celsius="11")],
    }
    response = muse_tokenizer(formatter.format_response(turn))["input_ids"]
    assert pieces(muse_tokenizer, response) == [
        " to=user",
        MESSAGE,
        "Let me look.",
        EOM,
        START,
        "assistant to=convert",
        MESSAGE,
        '<atem:function_calls>\n<atem:invoke name="convert">\n'
        '<atem:parameter name="celsius">11</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>",
        EOT,
    ]
    # What the model sampled reads back, and the next prompt is built from it.
    message, ended = formatter.parse_response(response)
    assert (message, ended) == (turn, True)
    reply = {"role": "tool", "name": "convert", "content": "51.8"}
    prompt = formatter.format_prompt([QUESTION, message, reply])
    assert muse_tokenizer.decode(response) in prompt


def test_a_tools_reply_names_its_tool(formatter, muse_tokenizer) -> None:
    messages = [
        QUESTION,
        {"role": "assistant", "tool_calls": [call("weather.lookup", "call_1", city="Paris")]},
        {"role": "tool", "tool_call_id": "call_1", "content": "rain"},
        {"role": "tool", "name": "convert", "content": "52"},
    ]
    prompt = pieces(muse_tokenizer, formatter.build_generation_prompt(messages))
    assert prompt[prompt.index("assistant to=weather.lookup") + 3 :] == [
        EOT,
        START,
        "tool weather.lookup",
        MESSAGE,
        '<tool_output name="weather.lookup">\nrain\n</tool_output>',
        EOT,
        START,
        "tool convert",
        MESSAGE,
        '<tool_output name="convert">\n52\n</tool_output>',
        EOT,
        START,
        "assistant",
    ]


def test_calls_another_assistant_message_follows_do_not_end_the_turn(formatter) -> None:
    messages = [
        QUESTION,
        {"role": "assistant", "tool_calls": [call("convert")]},
        {"role": "assistant", "content": "42"},
    ]
    prompt = formatter.format_prompt(messages)
    assert "</atem:function_calls><|eom|><|start|>assistant to=user<|message|>42<|eot|>" in prompt


@pytest.mark.parametrize(
    ("messages", "why"),
    [
        (
            [QUESTION, {"role": "assistant", "content": "Looking.", "end_turn": False}],
            "recipient or end_turn",
        ),
        (
            [QUESTION, {"role": "assistant", "content": "Hm.", "recipient": "self"}],
            "recipient or end_turn",
        ),
        ([QUESTION, {"role": "tool", "content": "rain"}], "does not say which tool"),
        ([QUESTION, {"role": "tool", "tool_call_id": "call_9", "content": "rain"}], "which tool"),
        ([{"role": "user", "content": [{"type": "image"}]}], "not text"),
        ([QUESTION, {"role": "narrator", "content": "Later."}], "role is 'narrator'"),
    ],
)
def test_a_conversation_the_format_cannot_hold_is_refused(formatter, messages, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        formatter.build_generation_prompt(messages)


def test_a_tool_without_a_description_is_refused(formatter) -> None:
    tool = {"type": "function", "function": {"name": "convert", "parameters": {}}}
    with pytest.raises(ValueError, match="convert has no description"):
        formatter.build_generation_prompt([QUESTION], [tool])


def test_a_turn_cut_off_while_it_reasons_has_no_content(formatter, muse_tokenizer) -> None:
    sampled = muse_tokenizer(" to=self<|message|>6 * 7 is")["input_ids"]
    assert formatter.parse_response(sampled) == (
        {"role": "assistant", "reasoning_content": "6 * 7 is", "content": ""},
        False,
    )


@pytest.mark.parametrize(
    "written",
    [
        "42",
        " to=convert<|message|><atem:function_calls>\n<atem:invoke name=",
        ' to=convert<|message|><atem:function_calls>\n<atem:invoke name="convert">\nstray\n'
        "</atem:invoke>\n</atem:function_calls>",
        # Text after a call, which a message cannot hold in that order.
        ' to=convert<|message|><atem:function_calls>\n<atem:invoke name="convert">\n'
        "</atem:invoke>\n</atem:function_calls><|eom|><|start|>assistant to=user<|message|>Done.",
        " to=user<|message|>One.<|eom|><|start|>assistant to=user<|message|>Two.",
    ],
)
def test_a_turn_that_cannot_be_read_stays_in_the_content(
    formatter, muse_tokenizer, written
) -> None:
    message, _ = formatter.parse_response(muse_tokenizer(written)["input_ids"])
    assert message == {"role": "assistant", "content": written}


def test_a_continued_prompt_keeps_the_sampled_tokens_and_adds_the_replies(
    formatter, muse_tokenizer
) -> None:
    turn = {
        "role": "assistant",
        "reasoning_content": "I should look.",
        "content": "",
        "tool_calls": [call("weather.lookup", city="Paris"), call("convert")],
    }
    replies = [
        {"role": "tool", "name": "weather.lookup", "content": "rain"},
        {"role": "tool", "name": "convert", "content": "52"},
    ]
    prompt = formatter.build_generation_prompt([QUESTION], [LOOKUP, CONVERT])
    sampled = muse_tokenizer(formatter.format_response(turn))["input_ids"]
    continued = formatter.continue_prompt(prompt + sampled, replies)
    assert continued == formatter.build_generation_prompt(
        [QUESTION, turn, *replies], [LOOKUP, CONVERT]
    )
    assert pieces(muse_tokenizer, continued[len(prompt) + len(sampled) :]) == [
        START,
        "tool weather.lookup",
        MESSAGE,
        '<tool_output name="weather.lookup">\nrain\n</tool_output>',
        EOT,
        START,
        "tool convert",
        MESSAGE,
        '<tool_output name="convert">\n52\n</tool_output>',
        EOT,
        START,
        "assistant",
    ]


def test_a_continued_prompt_closes_a_turn_the_sampler_left_open(formatter, muse_tokenizer) -> None:
    replies = [{"role": "tool", "name": "convert", "content": "52"}]
    continued = formatter.continue_prompt([7, 8], replies)
    assert pieces(muse_tokenizer, continued[2:])[:3] == [EOT, START, "tool convert"]


@pytest.mark.parametrize(
    "messages,why",
    [
        ([{"role": "tool", "content": "52"}], "does not say which tool it is from"),
        ([{"role": "user", "content": "And Rome?"}], "this message's role is 'user'"),
        ([], "continued with at least one tool reply"),
    ],
)
def test_a_continuation_the_format_cannot_write_is_refused(formatter, messages, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        formatter.continue_prompt([7, 8], messages)
