"""The agent session against a real Looper server: episodes sampled from a
model, turn after turn.

The unit tests drive the session with turns written by hand. Only a real model
shows that a continued prompt is one it reads as its own format: here the
number it must answer with exists only in a tool's reply, so the right
answer means it read that reply from the tokens ``continue_prompt`` added.

It calls the ``/v1`` that ``$LOOPER_URL`` names, with a sampler of the base
model, and trains nothing. Opt-in, as every GPU test is; ``-s`` prints each
episode's turns::

    LOOPER_URL=https://api.looper.goaly.ai pytest -m gpu -s \\
        looper_cookbook/tests/e2e/test_agent_session.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import random

import pytest
import pytest_asyncio
from transformers import AutoTokenizer

from looper_cookbook.environments.bash import BashEnvironment
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.episode_sessions.base import Episode, EpisodeSession, StopReason
from looper_cookbook.message_formatters.models import message_formatter
from looper_cookbook.sandboxes.local import LocalSandbox
from looper_cookbook.verifiers.final_number import FinalNumber
from looper_sdk import (
    FinishReason,
    SamplerConfig,
    SamplingClient,
    SamplingParams,
    ServiceClient,
)

pytestmark = [pytest.mark.gpu, pytest.mark.asyncio(loop_scope="module")]

URL = os.environ.get("LOOPER_URL")
MODEL = "Qwen/Qwen3.5-9B"
# Placement, the image's pull and the weights' load.
READY_S = 20 * 60
MAX_EPISODE_TOKENS = 4096
READ_THE_SECRET = {
    "role": "user",
    "content": (
        "The file secret.txt in the working directory holds a number. Read it with the bash "
        "tool. Then reply with '#### ' and then that number."
    ),
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class WithSecret:
    """A local sandbox's config that leaves a number in ``secret.txt``."""

    secret: int

    async def build(self) -> LocalSandbox:
        sandbox = await LocalSandbox.Config().build()
        await sandbox.write_file("secret.txt", f"{self.secret}\n")
        return sandbox


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sampler():
    if URL is None:
        pytest.skip("Set LOOPER_URL to the /v1 of a Looper server with GPUs behind it")
    config = ServiceClient.Config(base_url=URL, project="e2e", display_name="agent-session")
    async with await config.build() as service:
        made = await service.create_sampling_client(
            SamplerConfig(
                checkpoint_name=f"base:{MODEL}",
                max_seq_len=MAX_EPISODE_TOKENS,
                max_concurrent_sequences=16,
            ),
            timeout_s=READY_S,
        )
        try:
            yield made
        finally:
            await made.stop()


@pytest.fixture(scope="module")
def runtime() -> EpisodeSession.Runtime:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    formatter = message_formatter(MODEL, tokenizer, enable_thinking=False)
    # Greedy, so a failure repeats.
    return EpisodeSession.Runtime(
        formatter=formatter, params=SamplingParams(max_tokens=512, temperature=0.0)
    )


def show(name: str, episode: Episode) -> None:
    print(f"{name}: {episode.stop_reason.value}, reward {episode.reward}, {dict(episode.metrics)}")
    for turn in episode.turns:
        sequence = turn.sequence
        print(
            f"  read {len(turn.context)} tokens, wrote {len(sequence.token_ids)} "
            f"({sequence.finish_reason.value}), {sequence.cached_tokens} of its prompt cached"
        )
    for message in episode.messages:
        print(f"  {message['role']}: {str(message.get('content'))[:200]!r}")


def check_tokens(episode: Episode, runtime: EpisodeSession.Runtime) -> None:
    """What every episode's tokens keep to, whatever the model wrote."""
    datum = episode.datum("group", 2)
    assert len(datum.tokens) <= MAX_EPISODE_TOKENS
    assert sum(datum.weights) == episode.metrics["sampled_tokens"]
    for turn in episode.turns:
        if turn.sequence.finish_reason is FinishReason.STOP:
            # The sampler returns the stop token, so a turn's end is trained with it.
            assert turn.sequence.token_ids[-1] == runtime.formatter.end_of_turn_id


async def test_a_one_turn_task_is_answered_and_verified(
    sampler: SamplingClient, runtime: EpisodeSession.Runtime
) -> None:
    config = AgentSession.Config(
        messages=[
            {
                "role": "user",
                "content": "What is 6 times 7? Reply with '#### ' and then the number.",
            }
        ],
        verifier=FinalNumber.Config(answer="42"),
        max_episode_tokens=MAX_EPISODE_TOKENS,
    )
    async with await config.build(runtime) as session:
        episode = await session.run(sampler)
    show("one turn", episode)
    check_tokens(episode, runtime)
    assert (episode.stop_reason, len(episode.turns)) == (StopReason.ANSWER, 1)
    assert episode.reward == 1.0


async def test_a_model_reads_a_tools_reply_from_the_continued_prompt(
    sampler: SamplingClient, runtime: EpisodeSession.Runtime
) -> None:
    secret = random.SystemRandom().randint(100_000, 999_999)
    config = AgentSession.Config(
        messages=[READ_THE_SECRET],
        verifier=FinalNumber.Config(answer=str(secret)),
        environment=BashEnvironment.Config(sandbox=WithSecret(secret=secret)),
        max_turns=4,
        max_episode_tokens=MAX_EPISODE_TOKENS,
    )
    async with await config.build(runtime) as session:
        episode = await session.run(sampler)
    show("bash", episode)
    check_tokens(episode, runtime)
    # The number is in no prompt of the task: only the tool's reply holds it.
    assert any(
        message["role"] == "tool" and str(secret) in message["content"]
        for message in episode.messages
    )
    assert episode.stop_reason is StopReason.ANSWER
    assert episode.reward == 1.0
    assert len(episode.turns) >= 2


async def test_attempts_run_at_once_each_in_a_sandbox_of_its_own(
    sampler: SamplingClient, runtime: EpisodeSession.Runtime
) -> None:
    secrets = [random.SystemRandom().randint(100_000, 999_999) for _ in range(4)]

    async def attempt(secret: int) -> Episode:
        config = AgentSession.Config(
            messages=[READ_THE_SECRET],
            verifier=FinalNumber.Config(answer=str(secret)),
            environment=BashEnvironment.Config(sandbox=WithSecret(secret=secret)),
            max_turns=4,
            max_episode_tokens=MAX_EPISODE_TOKENS,
        )
        async with await config.build(runtime) as session:
            return await session.run(sampler)

    episodes = await asyncio.gather(*(attempt(secret) for secret in secrets))
    for secret, episode in zip(secrets, episodes, strict=True):
        show(f"attempt for {secret}", episode)
        check_tokens(episode, runtime)
        # Each read its own file: no attempt saw another's sandbox.
        replies = [message for message in episode.messages if message["role"] == "tool"]
        assert [reply["content"] for reply in replies] == [f"{secret}\n"]
        assert episode.reward == 1.0
