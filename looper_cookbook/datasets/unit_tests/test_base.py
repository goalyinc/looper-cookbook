"""``batch_positions`` (looper_cookbook/datasets/base.py): the order the
cookbook's datasets batch in."""

from __future__ import annotations

import pytest

from looper_cookbook.datasets.base import batch_positions


def batches(count: int, indexes: range, *, batch_size: int = 3, seed: int = 1) -> list[list[int]]:
    return [
        batch_positions(count, batch_size=batch_size, seed=seed, index=index) for index in indexes
    ]


def test_a_pass_holds_every_position_once_in_an_order_of_its_own() -> None:
    first, second = batches(7, range(3)), batches(7, range(3, 6))
    # The last batch of a pass holds what is left.
    assert [len(batch) for batch in first] == [3, 3, 1]
    for a_pass in (first, second):
        assert sorted(position for batch in a_pass for position in batch) == list(range(7))
    assert first != second
    assert batches(7, range(3), seed=2) != first


def test_the_index_alone_decides_a_batch() -> None:
    # Pinned: a run stopped under one release and called again under the next
    # must read the batch its step had.
    assert batches(7, range(6)) == [[0, 1, 6], [4, 2, 3], [5], [1, 6, 4], [0, 2, 3], [5]]
    assert batch_positions(7, batch_size=3, seed=1, index=4) == [0, 2, 3]


def test_one_batch_holds_a_whole_pass_when_it_fits() -> None:
    assert sorted(batch_positions(4, batch_size=9, seed=0, index=0)) == [0, 1, 2, 3]
    assert sorted(batch_positions(4, batch_size=9, seed=0, index=5)) == [0, 1, 2, 3]


def test_what_names_no_batch_is_refused() -> None:
    with pytest.raises(IndexError, match="0 or above, not -1"):
        batch_positions(7, batch_size=3, seed=1, index=-1)
    with pytest.raises(ValueError, match="not 0 items in batches of 3"):
        batch_positions(0, batch_size=3, seed=1, index=0)
    with pytest.raises(ValueError, match="not 7 items in batches of 0"):
        batch_positions(7, batch_size=0, seed=1, index=0)
