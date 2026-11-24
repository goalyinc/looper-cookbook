"""The agent session, driven turn by turn with sequences written here: its
prompts, the actions it runs, what ends it and how it is scored.

No sampler: ``start`` and ``observe`` say what to sample, and each test
answers with a turn it wrote. ``run``, which samples between them, runs
against a real server (``tests/e2e/test_agent_session.py``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import pytest

from looper_cookbook.environments.base import Environment
from looper_cookbook.environments.bash import BashEnvironment
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.episode_sessions.base import Episode, EpisodeSession, StopReason, Turn
from looper_cookbook.message_formatters.base import Message
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter
from looper_cookbook.message_formatters.unit_tests import tokenizers
from looper_cookbook.sandboxes.base import CommandResult, SandboxError
from looper_cookbook.sandboxes.unit_tests.scripted import Prebuilt, ScriptedSandbox
from looper_cookbook.verifiers.base import Score, Verifier
from looper_cookbook.verifiers.final_number import FinalNumber
from looper_sdk import Datum, FinishReason, SampleRequest, SampleSequence, SamplingParams

TOKENIZER = tokenizers.qwen()
FORMATTER = Qwen3_5MessageFormatter(TOKENIZER, enable_thinking=False)
END = FORMATTER.end_of_turn_id
RUNTIME = EpisodeSession.Runtime(formatter=FORMATTER, params=SamplingParams(max_tokens=64))
QUESTION = [{"role": "user", "content": "What is 6 times 7? End on #### and the number."}]
RIGHT = FinalNumber.Config(answer="42")


def sampled(text: str, reason: FinishReason = FinishReason.STOP) -> SampleSequence:
    """A turn as a sampler returns it: its tokens, and a log probability each."""
    tokens = TOKENIZER(text)["input_ids"]
    logprobs = [-0.25 * (index + 1) for index in range(len(tokens))]
    return SampleSequence(token_ids=tokens, logprobs=logprobs, finish_reason=reason)


def bash(*commands: str) -> str:
    """A turn that calls ``bash`` once per command, as Qwen3.5 writes it."""
    calls = "\n".join(
        f"<tool_call>\n<function=bash>\n<parameter=command>\n{command}\n</parameter>\n"
        "</function>\n</tool_call>"
        for command in commands
    )
    return calls + "<|im_end|>"


def printed(text: str, exit_code: int = 0) -> CommandResult:
    return CommandResult(exit_code=exit_code, stdout=text)


def task(sandbox: ScriptedSandbox | None = None, **settings: Any) -> AgentSession.Config:
    """A task of ``QUESTION``, over a shell in ``sandbox`` when it is given."""
    environment = None
    if sandbox is not None:
        environment = BashEnvironment.Config(sandbox=Prebuilt(sandbox))
    settings = {"verifier": RIGHT, "max_episode_tokens": 4096, **settings}
    return AgentSession.Config(messages=QUESTION, environment=environment, **settings)


@pytest.mark.parametrize(
    "reply,reward,answered",
    [("#### 42", 1.0, 1.0), ("#### 41", 0.0, 1.0), ("Forty-two.", 0.0, 0.0)],
)
async def test_a_task_without_an_environment_is_one_turn_and_its_verifiers_score(
    reply: str, reward: float, answered: float
) -> None:
    async with await task().build(RUNTIME) as session:
        request = await session.start()
        prompt = tuple(FORMATTER.build_generation_prompt(QUESTION))
        # The session adds the formatter's stop token to the caller's params.
        assert request == SampleRequest(
            prompt=prompt, params=SamplingParams(max_tokens=64, stop_token_ids=(END,))
        )
        turn = sampled(f"{reply}<|im_end|>")
        assert await session.observe(turn) is None
        episode = await session.finish()
    assert episode == Episode(
        turns=(Turn(context=prompt, sequence=turn),),
        reward=reward,
        stop_reason=StopReason.ANSWER,
        messages=(*QUESTION, {"role": "assistant", "content": reply}),
        metrics={
            "answered": answered,
            "turns": 1.0,
            "sampled_tokens": float(len(turn.token_ids)),
            "truncated": 0.0,
        },
    )


async def test_a_turn_cut_off_ends_the_episode_unverified() -> None:
    async with await task().build(RUNTIME) as session:
        await session.start()
        # The right number, in a turn that never ended.
        assert await session.observe(sampled("#### 42", FinishReason.LENGTH)) is None
        episode = await session.finish()
    assert (episode.stop_reason, episode.reward) == (StopReason.MAX_TURN_TOKENS, 0.0)
    # No verifier ran, so none of its metrics is there.
    assert episode.metrics == {"turns": 1.0, "sampled_tokens": 7.0, "truncated": 1.0}
    assert episode.messages[-1] == {"role": "assistant", "content": "#### 42"}


async def test_tool_calls_run_in_order_and_their_replies_continue_the_prompt() -> None:
    sandbox = ScriptedSandbox(printed("a.txt\n"), printed("cat: b.txt: No such file\n", 1))
    async with await task(sandbox, max_turns=3).build(RUNTIME) as session:
        first = await session.start()
        assert session.environment is not None
        tools = session.environment.tools
        assert first.prompt == tuple(FORMATTER.build_generation_prompt(QUESTION, tools))
        called = sampled(bash("ls", "cat b.txt"))
        second = await session.observe(called)
        assert sandbox.commands == [
            ("bash -c ls 2>&1", 60.0, None),
            ("bash -c 'cat b.txt' 2>&1", 60.0, None),
        ]
        replies = [
            {"role": "tool", "name": "bash", "content": "a.txt\n"},
            {"role": "tool", "name": "bash", "content": "cat: b.txt: No such file\n[exit code 1]"},
        ]
        # Every token so far, as sampled, and then the replies: nothing is written again.
        transcript = [*first.prompt, *called.token_ids]
        assert second is not None
        assert second.prompt == tuple(FORMATTER.continue_prompt(transcript, replies))
        assert second.prompt[: len(transcript)] == tuple(transcript)
        answered = sampled("#### 42<|im_end|>")
        assert await session.observe(answered) is None
        episode = await session.finish()
    assert episode.turns == (
        Turn(context=first.prompt, sequence=called),
        Turn(context=second.prompt[len(transcript) :], sequence=answered),
    )
    assert (episode.stop_reason, episode.reward) == (StopReason.ANSWER, 1.0)
    assert episode.messages == (
        *QUESTION,
        FORMATTER.parse_response(called.token_ids, tools)[0],
        *replies,
        {"role": "assistant", "content": "#### 42"},
    )
    assert [call["function"]["arguments"] for call in episode.messages[1]["tool_calls"]] == [
        {"command": "ls"},
        {"command": "cat b.txt"},
    ]
    # The observations' counts add up over the episode's actions.
    assert episode.metrics == {
        "failed": 1.0,
        "timed_out": 0.0,
        "answered": 1.0,
        "turns": 2.0,
        "sampled_tokens": float(len(called.token_ids) + len(answered.token_ids)),
        "truncated": 0.0,
    }


async def test_an_episode_is_one_datum_with_weight_on_its_sampled_tokens() -> None:
    sandbox = ScriptedSandbox(printed("42\n"))
    async with await task(sandbox, max_turns=2).build(RUNTIME) as session:
        first = await session.start()
        called = sampled(bash("echo 42"))
        second = await session.observe(called)
        answered = sampled("#### 42<|im_end|>")
        assert second is not None and await session.observe(answered) is None
        episode = await session.finish()
    added = second.prompt[len(first.prompt) + len(called.token_ids) :]
    assert episode.datum("step-3-task-0", 8) == Datum(
        tokens=(*first.prompt, *called.token_ids, *added, *answered.token_ids),
        weights=(
            *[0] * len(first.prompt),
            *[1] * len(called.token_ids),
            *[0] * len(added),
            *[1] * len(answered.token_ids),
        ),
        columns={
            "logprobs": (
                *[0.0] * len(first.prompt),
                *called.logprobs,
                *[0.0] * len(added),
                *answered.logprobs,
            )
        },
        reward=1.0,
        group_id="step-3-task-0",
        group_size=8,
    )
    # The datum is the last prompt and the turn after it: what the model read and wrote.
    assert episode.datum("g", 2).tokens == (*second.prompt, *answered.token_ids)


async def test_a_call_the_model_got_wrong_is_answered_and_the_episode_goes_on() -> None:
    sandbox = ScriptedSandbox()
    wrong = (
        "<tool_call>\n<function=python>\n<parameter=code>\n1\n</parameter>\n</function>\n"
        "</tool_call><|im_end|>"
    )
    async with await task(sandbox, max_turns=2).build(RUNTIME) as session:
        await session.start()
        assert await session.observe(sampled(wrong)) is not None
        await session.observe(sampled("#### 42<|im_end|>"))
        episode = await session.finish()
    assert episode.messages[2] == {
        "role": "tool",
        "name": "python",
        "content": "Error: there is no tool named python. The tools are: bash.",
    }
    assert sandbox.commands == []
    assert episode.metrics["invalid_action"] == 1.0
    assert episode.reward == 1.0


async def test_the_turn_limit_ends_an_episode_after_its_last_calls_ran() -> None:
    sandbox = ScriptedSandbox(printed("42\n"))
    async with await task(sandbox, max_turns=1).build(RUNTIME) as session:
        await session.start()
        assert await session.observe(sampled(bash("echo 42"))) is None
        episode = await session.finish()
    assert len(sandbox.commands) == 1
    assert episode.stop_reason is StopReason.MAX_TURNS
    assert episode.messages[-1] == {"role": "tool", "name": "bash", "content": "42\n"}
    # Its last turn was whole, so it is verified: it gave no final number.
    assert (episode.reward, episode.metrics["answered"]) == (0.0, 0.0)
    assert episode.metrics["truncated"] == 0.0


async def test_a_turn_gets_no_more_tokens_than_the_episode_has_left() -> None:
    prompt = FORMATTER.build_generation_prompt(QUESTION)
    async with await task(max_episode_tokens=len(prompt) + 10).build(RUNTIME) as session:
        assert (await session.start()).params.max_tokens == 10


async def test_a_turn_cut_off_at_what_the_episode_had_left_names_the_episodes_budget() -> None:
    prompt = FORMATTER.build_generation_prompt(QUESTION)
    cut = sampled("#### 42", FinishReason.LENGTH)
    # The episode has fewer tokens left than a turn's 64, so its remainder is the turn's limit.
    limit = len(prompt) + len(cut.token_ids)
    async with await task(max_episode_tokens=limit).build(RUNTIME) as session:
        assert (await session.start()).params.max_tokens == len(cut.token_ids)
        assert await session.observe(cut) is None
        episode = await session.finish()
    assert (episode.stop_reason, episode.reward) == (StopReason.MAX_EPISODE_TOKENS, 0.0)
    assert episode.metrics == {"turns": 1.0, "sampled_tokens": 7.0, "truncated": 1.0}


async def test_a_later_turn_cut_off_at_the_turns_budget_names_the_turns_budget() -> None:
    sandbox = ScriptedSandbox(printed("42\n"))
    async with await task(sandbox, max_turns=4).build(RUNTIME) as session:
        await session.start()
        second = await session.observe(sampled(bash("echo 42")))
        # The episode has room: the second turn's limit is a turn's own 64.
        assert second is not None and second.params.max_tokens == 64
        assert await session.observe(sampled("#### 42", FinishReason.LENGTH)) is None
        episode = await session.finish()
    assert (episode.stop_reason, episode.reward) == (StopReason.MAX_TURN_TOKENS, 0.0)
    assert (episode.metrics["turns"], episode.metrics["truncated"]) == (2.0, 1.0)


async def test_an_episode_with_no_token_left_for_a_turn_ends_after_its_replies() -> None:
    shell = await BashEnvironment.Config(sandbox=Prebuilt(ScriptedSandbox())).build()
    prompt = FORMATTER.build_generation_prompt(QUESTION, shell.tools)
    called = sampled(bash("echo 42"))
    # Room for the first turn, and not for its replies and another.
    limit = len(prompt) + len(called.token_ids) + 5
    sandbox = ScriptedSandbox(printed("42\n"))
    async with await task(sandbox, max_turns=4, max_episode_tokens=limit).build(RUNTIME) as session:
        first = await session.start()
        assert await session.observe(called) is None
        episode = await session.finish()
    assert episode.stop_reason is StopReason.MAX_EPISODE_TOKENS
    assert episode.messages[-1]["role"] == "tool"
    assert len(episode.datum("g", 2).tokens) == len(first.prompt) + len(called.token_ids)


async def test_a_first_prompt_that_fills_the_episode_is_refused() -> None:
    prompt = FORMATTER.build_generation_prompt(QUESTION)
    async with await task(max_episode_tokens=len(prompt)).build(RUNTIME) as session:
        with pytest.raises(ValueError, match=f"has {len(prompt)} tokens.*no token is left"):
            await session.start()


async def test_a_tool_call_without_an_environment_is_the_models_answer() -> None:
    async with await task().build(RUNTIME) as session:
        await session.start()
        assert await session.observe(sampled(bash("echo 42"))) is None
        episode = await session.finish()
    assert episode.stop_reason is StopReason.ANSWER
    assert episode.reward == 0.0


class LeftInTheSandbox(Verifier):
    """Scores what the model left in its sandbox, where it worked."""

    @dataclasses.dataclass(frozen=True)
    class Config:
        async def build(self) -> LeftInTheSandbox:
            return LeftInTheSandbox()

    async def verify(self, messages: Sequence[Message], environment: Environment | None) -> Score:
        assert environment is not None and environment.sandbox is not None
        result = await environment.sandbox.run("cat answer.txt", timeout_s=5)
        return Score(reward=float(result.stdout.strip() == "42"))


async def test_a_verifier_reads_the_environment_before_the_loop_closes_it() -> None:
    sandbox = ScriptedSandbox(printed(""), printed("42\n"))
    config = task(sandbox, max_turns=2, verifier=LeftInTheSandbox.Config())
    async with await config.build(RUNTIME) as session:
        await session.start()
        await session.observe(sampled(bash("echo 42 > answer.txt")))
        await session.observe(sampled("Done.<|im_end|>"))
        episode = await session.finish()
        assert sandbox.commands[-1] == ("cat answer.txt", 5, None)
        assert sandbox.closes == 0
        # A second finish is the same episode: nothing is verified twice.
        assert await session.finish() is episode
    assert episode.reward == 1.0
    assert sandbox.closes == 1
    assert len(sandbox.commands) == 2


async def test_a_sandbox_that_fails_raises_from_its_step_and_the_loop_still_closes() -> None:
    class Broken(ScriptedSandbox):
        async def run(self, command, *, timeout_s, stdin=None):
            raise SandboxError("the container is gone", retryable=True)

    sandbox = Broken()
    async with await task(sandbox, max_turns=2).build(RUNTIME) as session:
        await session.start()
        with pytest.raises(SandboxError, match="the container is gone"):
            await session.observe(sampled(bash("ls")))
    assert sandbox.closes == 1


async def test_a_loop_runs_one_episode_and_its_steps_in_order() -> None:
    session = await task().build(RUNTIME)
    with pytest.raises(RuntimeError, match="observe takes a turn sampled for"):
        await session.observe(sampled("#### 42<|im_end|>"))
    await session.start()
    with pytest.raises(RuntimeError, match="finish scores an episode that observe has ended"):
        await session.finish()
    with pytest.raises(RuntimeError, match="a session runs one episode"):
        await session.start()
    await session.observe(sampled("#### 42<|im_end|>"))
    with pytest.raises(RuntimeError, match="observe takes a turn sampled for"):
        await session.observe(sampled("#### 42<|im_end|>"))
    await session.close()
    await session.close()


class Terse(AgentSession):
    """A harness of its own: it tells the model when a turn called no tool."""

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Config(AgentSession.Config):
        async def build(self, runtime: EpisodeSession.Runtime) -> Terse:
            return Terse(self, runtime, await self.verifier.build())

    async def respond(self, reply: Message) -> Sequence[Message]:
        replies = await super().respond(reply)
        if replies or "####" in reply["content"]:
            return replies
        return [{"role": "tool", "name": "bash", "content": "Call bash, or end on ####."}]


async def test_a_harness_of_its_own_decides_how_a_turn_is_answered() -> None:
    environment = BashEnvironment.Config(sandbox=Prebuilt(ScriptedSandbox()))
    config = Terse.Config(
        messages=QUESTION,
        verifier=RIGHT,
        environment=environment,
        max_turns=2,
        max_episode_tokens=4096,
    )
    async with await config.build(RUNTIME) as session:
        await session.start()
        nudged = await session.observe(sampled("Hmm.<|im_end|>"))
        assert nudged is not None
        assert "Call bash, or end on ####." in TOKENIZER.decode(list(nudged.prompt))
        assert await session.observe(sampled("#### 42<|im_end|>")) is None
        assert (await session.finish()).reward == 1.0


def test_a_harness_without_a_config_of_its_own_is_refused_when_it_is_defined() -> None:
    # AgentSession.Config builds an AgentSession, so its tasks would run the plain session.
    with pytest.raises(TypeError, match="Silent needs a Config of its own"):

        class Silent(AgentSession):
            async def respond(self, reply: Message) -> Sequence[Message]:
                return []


def test_the_agent_session_is_an_episode_session_and_its_task_is_every_sessions() -> None:
    assert issubclass(AgentSession, EpisodeSession)
    assert isinstance(task(), EpisodeSession.Config)
    # It adds no field of its own, and refuses what the contract refuses.
    assert dataclasses.fields(AgentSession.Config) == dataclasses.fields(EpisodeSession.Config)
    with pytest.raises(ValueError, match="max_turns must be at least 1, not 0"):
        task(max_turns=0)
