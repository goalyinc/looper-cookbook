"""A message formatter checked against its model's own chat template. Run by
hand, with the model's real tokenizer::

    python -m looper_cookbook.tests.message_formatters qwen3_5 Qwen/Qwen3.5-9B
    python -m looper_cookbook.tests.message_formatters muse_glimmer /path/to/tokenizer --data FILE

The formatters are written by hand, so this is what shows that one writes
what its model was trained on. For every conversation, every setting of the
formatter and every assistant turn, it checks that:

- ``format_prompt`` is the template's text with its generation prompt, and
  ``build_generation_prompt`` its tokens;
- the prompt and then ``format_response`` are the template's text of the
  conversation through that turn;
- ``parse_response``, given the tools, reads the turn's tokens back into a
  message with the calls and arguments the turn had, which formats to the
  same turn;
- when tools reply to the turn, ``format_continuation`` is the template's
  text from the turn's end token to the next generation prompt, and
  ``continue_prompt`` adds its tokens.

Muse Glimmer's template drops the content of a message that also has tool
calls, and writes the text when the text and the calls are two messages. The
formatter writes one message's text and calls, so the template is given the
two.

It also counts the datums whose tokens differ from the template's own
tokens of the conversation. They differ only where a prompt ends on a
newline and the turn starts with another, which one tokenization of the
whole text merges: the datum has what the model is sampled after.

And it counts the continued prompts whose template writes the turn before
the replies another way than the turn was written as the last: Qwen3's
drops an empty ``<think>`` block from a turn that is no longer the last. A
continued prompt keeps the turn as it was sampled, so only what follows the
turn is the template's there.

The unit tests need no template: they check each formatter's tokens against
what the format says (``message_formatters/unit_tests/``). Run this when a
formatter is added or changed, and when a model's template is.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from transformers import AutoTokenizer

from looper_cookbook.message_formatters.base import MessageFormatter, call_arguments
from looper_cookbook.message_formatters.muse_glimmer import (
    REASONING_STRENGTHS,
    MuseGlimmerMessageFormatter,
)
from looper_cookbook.message_formatters.qwen3 import Qwen3MessageFormatter
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter
from looper_cookbook.message_formatters.qwen3_8 import REASONING_EFFORTS, Qwen3_8MessageFormatter

Conversation = dict[str, Any]

# Each family's formatter, and the settings to check. A formatter's keywords
# are its template's, apart from the date Muse's template reads from the clock.
FAMILIES: dict[str, tuple[type[MessageFormatter], list[dict[str, Any]]]] = {
    "qwen3": (Qwen3MessageFormatter, [{"enable_thinking": on} for on in (True, False)]),
    "qwen3_5": (Qwen3_5MessageFormatter, [{"enable_thinking": on} for on in (True, False)]),
    "qwen3_8": (
        Qwen3_8MessageFormatter,
        [
            {"enable_thinking": on, "reasoning_effort": effort, "preserve_thinking": keep}
            for on, effort, keep in itertools.product(
                (True, False), REASONING_EFFORTS, (True, False)
            )
        ],
    ),
    "muse_glimmer": (
        MuseGlimmerMessageFormatter,
        [
            {"reasoning_strength": strength, "current_date": "2026-11-18"}
            for strength in REASONING_STRENGTHS
        ],
    ),
}

WEATHER = {
    "type": "function",
    "function": {
        "name": "weather.lookup",
        "description": "The weather in a city today.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}
CONVERT = {
    "type": "function",
    "function": {
        "name": "convert",
        "description": "Convert a temperature. Größe, 温度.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {"type": "number"},
                "exact": {"type": "boolean"},
                "unit": {"type": ["string", "null"]},
                "scales": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "object"},
                "text": {"type": "string"},
            },
        },
    },
}


def call(name: str, call_id: str, **arguments: Any) -> dict[str, Any]:
    return {"type": "function", "id": call_id, "function": {"name": name, "arguments": arguments}}


def reply(name: str, call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "name": name, "tool_call_id": call_id, "content": content}


def cases() -> list[Conversation]:
    """Conversations that exercise each part of a format."""
    ask = {"role": "user", "content": "What is the weather in Paris?"}
    looked_up = call("weather.lookup", "call_1", city="Paris", days=2)
    said_and_called = {
        "messages": [
            ask,
            {"role": "assistant", "content": "Let me look.", "tool_calls": [looked_up]},
            reply("weather.lookup", "call_1", "rain"),
            {"role": "assistant", "content": "Rain."},
        ],
        "tools": [WEATHER],
    }
    return [
        said_and_called,
        {"messages": [{"role": "user", "content": "What is 6 times 7?"}]},
        {
            "messages": [
                {"role": "system", "content": "Answer in one word."},
                {"role": "user", "content": "What is 6 times 7?"},
                {"role": "assistant", "content": "42"},
                {"role": "user", "content": "And 6 times 8?"},
                {"role": "assistant", "content": "48"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "  What is 6 times 7?\n"},
                {"role": "assistant", "reasoning_content": "6 * 7 = 42.", "content": "42"},
                {"role": "user", "content": "Größer als 40? 温度"},
                {"role": "assistant", "reasoning_content": "42 > 40.\n\nYes.", "content": "Yes."},
            ]
        },
        {"messages": [ask, {"role": "assistant", "content": "I cannot look."}], "tools": [WEATHER]},
        {
            "messages": [
                {"role": "system", "content": "Use the tools."},
                ask,
                {
                    "role": "assistant",
                    "reasoning_content": "Look it up.",
                    "tool_calls": [looked_up],
                },
                reply("weather.lookup", "call_1", '{"sky": "rain", "celsius": 11}'),
                {
                    "role": "assistant",
                    "reasoning_content": "Convert it.",
                    "tool_calls": [call("convert", "call_2", celsius=11)],
                },
                reply("convert", "call_2", "51.8"),
                {"role": "assistant", "content": "Rain, 52 degrees Fahrenheit."},
                {"role": "user", "content": "And tomorrow?"},
                {"role": "assistant", "content": "I only know today."},
            ],
            "tools": [WEATHER, CONVERT],
        },
        {
            "messages": [
                ask,
                {
                    "role": "assistant",
                    "tool_calls": [
                        looked_up,
                        call(
                            "convert",
                            "call_2",
                            celsius=11.5,
                            exact=True,
                            unit=None,
                            scales=["F", "K"],
                            note={"text": "two\nlines", "größe": 1},
                            text="a line\nand another",
                        ),
                    ],
                },
                reply("weather.lookup", "call_1", "rain"),
                reply("convert", "call_2", "52.7"),
                {"role": "assistant", "content": "Rain."},
            ],
            "tools": [WEATHER, CONVERT],
        },
    ]


def read(path: str) -> list[Conversation]:
    with open(path, encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def template(
    tokenizer: Any, messages: Sequence[Any], tools: Any, prompt: bool, **kwargs: Any
) -> str:
    return tokenizer.apply_chat_template(
        list(messages), tools=tools, add_generation_prompt=prompt, tokenize=False, **kwargs
    )


def as_muse_glimmers_template_takes(messages: Sequence[Any]) -> list[Any]:
    """``messages``, with each assistant message that has text and tool calls
    as two: its text, with ``end_turn`` false, and then its calls."""
    result = []
    for message in messages:
        text = message.get("content") or ""
        if message["role"] == "assistant" and message.get("tool_calls") and text.strip():
            said = {key: value for key, value in message.items() if key != "tool_calls"}
            result += [
                {**said, "end_turn": False},
                {"role": "assistant", "tool_calls": message["tool_calls"]},
            ]
        else:
            result.append(message)
    return result


def calls(message: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [call_arguments(made) for made in message.get("tool_calls") or ()]


def check(
    formatter: MessageFormatter, conversation: Conversation, kwargs: Mapping[str, Any]
) -> Iterator[str]:
    """What the formatter writes of ``conversation`` that its template does
    not, one line each; ``"split"`` alone for a datum whose tokens only split
    differently where its prompt ends, and ``"rewritten"`` alone for a
    continued prompt whose template writes the turn before it another way."""
    tokenizer = formatter.tokenizer
    messages, tools = conversation["messages"], conversation.get("tools")
    takes = list
    if isinstance(formatter, MuseGlimmerMessageFormatter):
        takes = as_muse_glimmers_template_takes
    turns = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    # Every assistant turn, and the prompt after the whole conversation.
    for index in [*turns, len(messages)]:
        want = template(tokenizer, takes(messages[:index]), tools, True, **kwargs)
        try:
            prompt = formatter.format_prompt(messages[:index], tools)
        except ValueError as error:
            yield f"prompt before messages[{index}] refused: {error}"
            continue
        if prompt != want:
            yield f"prompt before messages[{index}]:\n  ours     {prompt!r}\n  template {want!r}"
            continue
        ids = formatter.build_generation_prompt(messages[:index], tools)
        if ids != tokenizer(want, add_special_tokens=False)["input_ids"]:
            yield f"prompt before messages[{index}]: the tokens differ"
        if index == len(messages):
            continue
        message = messages[index]
        if kwargs.get("enable_thinking") is False and message.get("reasoning_content"):
            # A turn that reasons cannot follow a prompt that tells the model not to.
            continue
        try:
            response = formatter.format_response(message)
        except ValueError as error:
            yield f"messages[{index}] refused: {error}"
            continue
        whole = template(tokenizer, takes(messages[: index + 1]), tools, False, **kwargs)
        if not whole.startswith(prompt + response) or whole[len(prompt + response) :].strip("\n"):
            yield f"messages[{index}]:\n  ours     {prompt + response!r}\n  template {whole!r}"
            continue
        tokens = tokenizer(response, add_special_tokens=False)["input_ids"]
        parsed, ended = formatter.parse_response(tokens, tools)
        if not ended or formatter.format_response(parsed) != response:
            yield f"messages[{index}] reads back as {parsed!r}, written {response!r}"
        elif calls(parsed) != calls(message):
            yield f"messages[{index}] calls {calls(message)!r}, read back as {calls(parsed)!r}"
        whole_ids = tokenizer(whole, add_special_tokens=False)["input_ids"]
        if (ids + tokens) != whole_ids[: len(ids + tokens)]:
            yield "split"
        # The tools' replies to this turn, when the next messages are that.
        replies = list(
            itertools.takewhile(lambda later: later["role"] == "tool", messages[index + 1 :])
        )
        if not replies:
            continue
        after = template(
            tokenizer, takes(messages[: index + 1 + len(replies)]), tools, True, **kwargs
        )
        try:
            continuation = formatter.format_continuation(replies)
        except ValueError as error:
            yield f"continuation after messages[{index}] refused: {error}"
            continue
        ends = after.endswith(continuation)
        if not ends or not after[: -len(continuation)].endswith(formatter.end_of_turn):
            yield (
                f"continuation after messages[{index}]:\n  ours     {continuation!r}\n"
                f"  template {after!r}"
            )
            continue
        added = formatter.continue_prompt(ids + tokens, replies)[len(ids + tokens) :]
        if added != tokenizer(after, add_special_tokens=False)["input_ids"][-len(added) :]:
            yield f"continuation after messages[{index}]: the tokens differ"
        if not after.startswith(prompt + response):
            yield "rewritten"


def main() -> int:
    parser = argparse.ArgumentParser(description="A message formatter against its template.")
    parser.add_argument("family", choices=sorted(FAMILIES))
    parser.add_argument("tokenizer", help="the model's tokenizer: its name or a directory")
    parser.add_argument("--data", action="append", default=[], help="a JSONL file of conversations")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    conversations = cases() + [conversation for path in args.data for conversation in read(path)]
    formatter_class, settings = FAMILIES[args.family]
    failed = 0
    for kwargs in settings:
        formatter = formatter_class(tokenizer, **kwargs)
        found = [
            line
            for conversation in conversations
            for line in check(formatter, conversation, kwargs)
        ]
        differences = [line for line in found if line not in ("split", "rewritten")]
        failed += len(differences)
        print(
            f"{args.family} {kwargs}: {len(conversations)} conversations, "
            f"{len(differences)} differences, {found.count('split')} datums split differently "
            f"where their prompt ends, {found.count('rewritten')} continued prompts after a "
            "turn the template writes another way"
        )
        for line in differences:
            print(line)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
