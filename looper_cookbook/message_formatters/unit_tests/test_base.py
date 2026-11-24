"""What every formatter does with its model's text: datums, their limits and
their refusals. Qwen3.5's formatter
stands for any."""

from __future__ import annotations

import pytest

from looper_cookbook.message_formatters.base import argument_value
from looper_cookbook.message_formatters.muse_glimmer import MuseGlimmerMessageFormatter
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter

CONVERSATION = [
    {"role": "system", "content": "Answer in one word."},
    {"role": "user", "content": "What is 6 times 7?"},
    {"role": "assistant", "content": "42"},
    {"role": "user", "content": "And 6 times 8?"},
    {"role": "assistant", "content": "48"},
]


@pytest.fixture
def formatter(qwen_tokenizer) -> Qwen3_5MessageFormatter:
    return Qwen3_5MessageFormatter(qwen_tokenizer, enable_thinking=False)


def test_each_assistant_turn_is_a_datum_trained_after_its_prompt(formatter) -> None:
    first, second = formatter.build_supervised_examples(CONVERSATION, max_seq_len=512)
    for datum, turn in ((first, 2), (second, 4)):
        prompt = formatter.build_generation_prompt(CONVERSATION[:turn])
        assert list(datum.tokens[: len(prompt)]) == prompt
        assert list(datum.weights) == [0] * len(prompt) + [1] * (len(datum.tokens) - len(prompt))
    assert formatter.targets(first) == ["42<|im_end|>"]
    assert formatter.targets(second) == ["48<|im_end|>"]
    # The first turn is the second datum's context, never its target.
    assert len(second.tokens) > len(first.tokens)


def test_a_tool_calling_turn_trains_its_calls(formatter) -> None:
    call = {"type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}
    conversation = [
        {"role": "user", "content": "Is it raining in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "content": "rain"},
        {"role": "assistant", "content": "Yes."},
    ]
    called, answered = formatter.build_supervised_examples(conversation, max_seq_len=512)
    assert formatter.targets(called) == [
        "<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n"
        "</tool_call><|im_end|>"
    ]
    assert formatter.targets(answered) == ["Yes.<|im_end|>"]
    # What the model samples for the first turn reads back as the turn.
    sampled = [token for token, weight in zip(called.tokens, called.weights, strict=True) if weight]
    assert formatter.parse_response(sampled) == (conversation[1], True)


def test_a_long_datum_keeps_its_head_and_a_turn_past_the_limit_is_refused(formatter) -> None:
    whole = formatter.build_supervised_examples(CONVERSATION[:3], max_seq_len=512)[0]
    start = whole.weights.index(1)
    cut = formatter.build_supervised_examples(CONVERSATION[:3], max_seq_len=start + 1)[0]
    assert cut.tokens == whole.tokens[: start + 1]
    assert list(cut.weights) == [0] * start + [1]
    with pytest.raises(ValueError, match=rf"starts at token {start}, past max_seq_len={start}"):
        formatter.build_supervised_examples(CONVERSATION[:3], max_seq_len=start)


def test_a_conversation_that_cannot_train_is_refused(formatter) -> None:
    with pytest.raises(ValueError, match="'What is 6 times 7\\?' has no assistant message"):
        formatter.build_supervised_examples(CONVERSATION[1:2], max_seq_len=512)


def test_a_refused_turn_names_its_message_and_its_conversation(formatter) -> None:
    conversation = [
        *CONVERSATION[:4],
        {"role": "assistant", "reasoning_content": "Hm.", "content": "48"},
    ]
    with pytest.raises(ValueError) as refused:
        formatter.build_supervised_examples(conversation, max_seq_len=512)
    assert str(refused.value).startswith(
        "messages[4] of the conversation starting 'What is 6 times 7?': this assistant turn "
        "carries reasoning"
    )


def test_a_tokenizer_of_another_model_is_refused(qwen_tokenizer, muse_tokenizer) -> None:
    with pytest.raises(
        ValueError, match="MuseGlimmerMessageFormatter writes <\\|begin_of_text\\|>"
    ):
        MuseGlimmerMessageFormatter(qwen_tokenizer)
    with pytest.raises(ValueError, match="Qwen3_5MessageFormatter writes <\\|im_start\\|>"):
        Qwen3_5MessageFormatter(muse_tokenizer)


def test_a_turn_cut_off_before_its_end_token_parses_as_far_as_it_goes(
    formatter, qwen_tokenizer
) -> None:
    sampled = qwen_tokenizer("42<|im_end|>")["input_ids"]
    assert formatter.parse_response(sampled) == ({"role": "assistant", "content": "42"}, True)
    assert formatter.parse_response(sampled[:1]) == ({"role": "assistant", "content": "4"}, False)
    assert formatter.stop_token_ids == (sampled[-1],)


@pytest.mark.parametrize(
    ("text", "schema", "value"),
    [
        # Without a schema, and for a string, the text as the model wrote it.
        ("42", None, "42"),
        ("42", {}, "42"),
        ("42", {"type": "string"}, "42"),
        ("true", {"type": "string"}, "true"),
        ("null", {"type": "string"}, "null"),
        (" 42", {"type": "string"}, " 42"),
        # The type the schema declares.
        ("42", {"type": "integer"}, 42),
        ("42", {"type": "number"}, 42),
        ("1.50", {"type": "number"}, 1.5),
        ("1e3", {"type": "number"}, 1000.0),
        ("true", {"type": "boolean"}, True),
        ("False", {"type": "boolean"}, False),
        ("[1, 2]", {"type": "array"}, [1, 2]),
        ('{"a": 1}', {"type": "object"}, {"a": 1}),
        ("null", {"type": ["integer", "null"]}, None),
        ("None", {"anyOf": [{"type": "integer"}, {"type": "null"}]}, None),
        ("42", {"anyOf": [{"type": "integer"}, {"type": "null"}]}, 42),
        ("null", {"type": ["string", "null"]}, None),
        ("no", {"type": ["string", "null"]}, "no"),
        # Text that does not fit its type stays text: nothing is invented.
        ("forty", {"type": "integer"}, "forty"),
        ("1.5", {"type": "integer"}, "1.5"),
        ("true", {"type": "integer"}, "true"),
        ("yes", {"type": "boolean"}, "yes"),
        ("[1, 2]", {"type": "object"}, "[1, 2]"),
        ("null", {"type": "integer"}, "null"),
    ],
)
def test_an_argument_reads_as_the_type_its_schema_declares(text, schema, value) -> None:
    read = argument_value(text, schema)
    assert read == value and type(read) is type(value)
