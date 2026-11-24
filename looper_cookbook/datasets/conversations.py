"""Conversations as SFT data: the cookbook's ``SupervisedDataset``, and a file
of conversations read.

A conversation is an object with ``messages``, chat messages with a
``role`` and a ``content``, and optionally ``tools``, their JSON schemas.
``read_conversations`` reads a JSONL file with one a line and names the line
that holds none.
``ChatDataset`` gives the SFT loop a batch of them for each step of a run,
each written by the model's ``MessageFormatter`` as a datum per assistant
turn (message_formatters/).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

from looper_cookbook.datasets.base import batch_positions
from looper_cookbook.message_formatters.base import MessageFormatter
from looper_sdk import Datum

__all__ = ["ChatDataset", "read_conversations"]


class ChatDataset:
    """Conversations in batches, over as many passes as a run has steps for.

    A batch is ``batch_size`` conversations, each as a datum per assistant
    turn, so it may hold more datums than conversations. Each pass is in an
    order of its own, which ``seed`` and the pass fix, so a run called again
    reads the same batch at the same step. A pass's last batch holds what is
    left.

    Every conversation is written when the dataset is built. One its model's
    format cannot hold, or whose prompt leaves no room for a target, fails
    then, before a run is created, and not at its step.

    Args:
        conversations: The conversations, as ``read_conversations`` returns
            them.
        formatter: The model's message formatter.
        batch_size: The conversations of one step.
        max_seq_len: A datum's longest length, which the run's must hold. A
            longer datum keeps its head.
        seed: Fixes each pass's order.
    """

    def __init__(
        self,
        conversations: Sequence[Mapping[str, Any]],
        formatter: MessageFormatter,
        *,
        batch_size: int,
        max_seq_len: int,
        seed: int = 0,
    ) -> None:
        if not conversations:
            raise ValueError("conversations is empty: a step needs a target")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, not {batch_size}")
        self._batch_size = batch_size
        self._seed = seed
        self._datums = [
            formatter.build_supervised_examples(
                conversation["messages"], conversation.get("tools"), max_seq_len=max_seq_len
            )
            for conversation in conversations
        ]

    def __len__(self) -> int:
        return math.ceil(len(self._datums) / self._batch_size)

    def batch(self, index: int) -> list[Datum]:
        positions = batch_positions(
            len(self._datums), batch_size=self._batch_size, seed=self._seed, index=index
        )
        return [datum for position in positions for datum in self._datums[position]]


def read_conversations(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """The conversations of a JSONL file, one per line: an object with
    ``messages``, chat messages with a ``role`` and a ``content``, and
    optionally ``tools``, their JSON schemas. Blank lines are skipped."""
    conversations = []
    with open(path, encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            where = f"{os.fspath(path)}:{number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{where} is not JSON: {error}") from None
            messages = record.get("messages") if isinstance(record, dict) else None
            if not isinstance(messages, list) or not messages:
                raise ValueError(f'{where} must be an object with "messages", a non-empty list')
            for index, message in enumerate(messages):
                if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                    raise ValueError(f'{where}: messages[{index}] must be an object with a "role"')
            conversations.append(record)
    return conversations
