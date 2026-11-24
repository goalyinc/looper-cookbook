"""The process runner over real programs: what it keeps of an output. Its deadline and its kill are
``test_local.py``'s, through a local sandbox."""

from __future__ import annotations

from looper_cookbook.sandboxes.base import CommandResult
from looper_cookbook.sandboxes.process import run_process


async def test_an_output_longer_than_a_sandbox_keeps_loses_its_end() -> None:
    result = await run_process(
        ("/bin/sh", "-c", "head -c 100000 /dev/zero | tr '\\0' a; echo b >&2"),
        timeout_s=10,
        max_output_bytes=1000,
    )
    assert result == CommandResult(exit_code=0, stdout="a" * 1000, stderr="b\n", truncated=True)


async def test_output_that_is_not_text_is_read_as_far_as_it_is() -> None:
    result = await run_process(("/bin/sh", "-c", "printf 'ok \\377 done'"), timeout_s=10)
    assert result.stdout == "ok � done"
