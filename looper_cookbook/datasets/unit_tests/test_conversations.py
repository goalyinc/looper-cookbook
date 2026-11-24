"""Conversations as SFT data (looper_cookbook/datasets/conversations.py):
``ChatDataset``'s batches and passes and what it refuses, and a file of
conversations read, each line that holds none named."""

from __future__ import annotations

import json

import pytest

from looper_cookbook.datasets.base import batch_positions
from looper_cookbook.datasets.conversations import ChatDataset, read_conversations
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter
from looper_cookbook.message_formatters.unit_tests import tokenizers

PLAIN = [
    {"role": "user", "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris."},
]
REASONED = [
    {"role": "user", "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris.", "reasoning_content": "France: Paris."},
]
FORMATTER = Qwen3_5MessageFormatter(tokenizers.qwen(), enable_thinking=False)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look a capital up.",
            "parameters": {"type": "object", "properties": {"country": {"type": "string"}}},
        },
    }
]


def conversation(letter: str, turns: int = 1) -> dict:
    """A conversation of ``turns`` questions, each answered: as many datums."""
    messages = []
    for turn in range(turns):
        messages.append({"role": "user", "content": f"{letter}{turn}?"})
        messages.append({"role": "assistant", "content": f"{letter}{turn}."})
    return {"messages": messages}


# Seven conversations, the third of two turns: eight datums.
CONVERSATIONS = [conversation(letter, 2 if letter == "c" else 1) for letter in "abcdefg"]


def written(record: dict, max_seq_len: int = 64) -> list:
    """The datums the formatter makes of one conversation."""
    return FORMATTER.build_supervised_examples(
        record["messages"], record.get("tools"), max_seq_len=max_seq_len
    )


def test_a_batch_is_its_conversations_datums_a_turn_each() -> None:
    dataset = ChatDataset(CONVERSATIONS, FORMATTER, batch_size=3, max_seq_len=64, seed=1)
    assert len(dataset) == 3
    for index in range(6):
        positions = batch_positions(7, batch_size=3, seed=1, index=index)
        # Each conversation whole, its turns in order, in the order the tasks' dataset has.
        assert dataset.batch(index) == [
            datum for position in positions for datum in written(CONVERSATIONS[position])
        ]
    # A pass trains every turn once: eight datums of seven conversations.
    assert sum(len(dataset.batch(index)) for index in range(3)) == 8
    assert sum(len(dataset.batch(index)) for index in range(3, 6)) == 8
    other = ChatDataset(CONVERSATIONS, FORMATTER, batch_size=3, max_seq_len=64, seed=2)
    assert [other.batch(index) for index in range(3)] != [dataset.batch(i) for i in range(3)]


def test_a_conversations_tools_and_the_longest_length_reach_the_formatter() -> None:
    plain, with_tools = {"messages": PLAIN}, {"messages": PLAIN, "tools": TOOLS}
    # The byte-level tokenizer spends a token a byte, so the tools' schemas need room.
    dataset = ChatDataset([with_tools], FORMATTER, batch_size=1, max_seq_len=4096)
    assert dataset.batch(0) == written(with_tools, 4096) != written(plain, 4096)
    # A longer datum keeps its head.
    (whole,) = written(plain)
    shorter = len(whole.tokens) - 2
    (cut,) = ChatDataset([plain], FORMATTER, batch_size=1, max_seq_len=shorter).batch(0)
    assert list(cut.tokens) == list(whole.tokens)[:shorter]


def test_a_dataset_refuses_what_cannot_train_when_it_is_built() -> None:
    with pytest.raises(ValueError, match="conversations is empty"):
        ChatDataset([], FORMATTER, batch_size=2, max_seq_len=64)
    with pytest.raises(ValueError, match="batch_size must be at least 1, not 0"):
        ChatDataset(CONVERSATIONS, FORMATTER, batch_size=0, max_seq_len=64)
    # Before a run is created, and not at the conversation's step.
    unanswered = {"messages": [{"role": "user", "content": "Capital of France?"}]}
    with pytest.raises(ValueError, match="has no assistant message"):
        ChatDataset([*CONVERSATIONS, unanswered], FORMATTER, batch_size=2, max_seq_len=64)
    with pytest.raises(IndexError, match="0 or above, not -1"):
        ChatDataset(CONVERSATIONS, FORMATTER, batch_size=2, max_seq_len=64).batch(-1)


def test_conversations_are_read_one_per_line(tmp_path) -> None:
    path = tmp_path / "conversations.jsonl"
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    records = [{"messages": PLAIN}, {"messages": REASONED, "tools": tools}]
    path.write_text(json.dumps(records[0]) + "\n\n" + json.dumps(records[1]) + "\n")
    assert read_conversations(path) == records


@pytest.mark.parametrize(
    ("line", "error"),
    [
        ("{not json", r":2 is not JSON"),
        ('{"prompt": "hi"}', r':2 must be an object with "messages"'),
        ('{"messages": []}', r':2 must be an object with "messages"'),
        ('{"messages": ["hi"]}', r':2: messages\[0\] must be an object with a "role"'),
    ],
)
def test_a_line_that_is_no_conversation_is_named(tmp_path, line: str, error: str) -> None:
    path = tmp_path / "conversations.jsonl"
    path.write_text(json.dumps({"messages": PLAIN}) + "\n" + line + "\n")
    with pytest.raises(ValueError, match=error):
        read_conversations(path)
