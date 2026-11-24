"""An RL job on math problems, through the cookbook's RL loop.

1. **The tasks.** By default GSM8K's train split, at the revision
   ``sft_basic`` pins: each problem asked through the model's message formatter,
   with an instruction to end on ``#### <number>``. With ``--data``, a JSONL
   file of ``{"question": ..., "answer": ...}`` lines instead: yours, or the
   bundled ``example_data/math_problems.jsonl``. A task is an ``AgentSession``'s
   config with no environment: one turn, which is the answer.
2. **The reward.** A ``FinalNumber`` verifier: 1 when the number after the
   reply's last ``####`` is the problem's answer, else 0. A turn cut off at
   ``--max-turn-tokens``, a turn's token budget, scores 0.
3. **The held-out problems.** GSM8K's test split, asked the same way and
   answered greedily; with ``--data``, the problems of ``--eval-data`` if
   given. The loop measures them before its first step and after its last,
   and every ``--eval-every`` steps between.
4. **The run.** A LoRA of rank ``--lora-rank``, or with ``--full`` every
   parameter, sized to an episode's token budget. An episode here is one
   turn, so that is the longest prompt plus ``--max-turn-tokens``.
5. **Its sampler.** Created by the run's training client, for sequences as
   long as the run's and ``--concurrency`` of them at once; the rest of a
   step's turns queue.
6. **The loop.** ``train_rl.train`` takes ``--steps`` steps. Each attempts
   each of ``--batch-size`` problems ``--group-size`` times, trains one
   ``Grpo`` step on the groups and publishes it to the sampler
   (``looper_cookbook/train_rl.py``).
7. **The checkpoint.** It saves one and waits until it is uploaded.

Each step's reward is the mean over episodes sampled before it trained, from
problems it has not trained on in the first pass, so a rising reward is not a
batch memorized. The held-out share is what the run's weights answer right
of problems it never trains.

    LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.rl_basic
    python -m looper_cookbook.recipes.rl_basic --steps 20 --batch-size 16

To check a deployment, use the bundled problems: no dataset to download.

    python -m looper_cookbook.recipes.rl_basic \\
        --data looper_cookbook/example_data/math_problems.jsonl \\
        --steps 4 --batch-size 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from looper_cookbook import train_rl
from looper_cookbook.datasets.tasks import TaskDataset
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.message_formatters.base import Message, MessageFormatter
from looper_cookbook.message_formatters.models import message_formatter
from looper_cookbook.verifiers.final_number import FinalNumber
from looper_sdk import (
    DEFAULT_BASE_URL,
    AdamW,
    Full,
    Lora,
    SamplingParams,
    ServiceClient,
    TrainingConfig,
)

GSM8K = "openai/gsm8k"
# sft_basic's revision: a later edit of the dataset changes neither.
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
INSTRUCTION = "Solve it step by step, then give the final answer as a number after '####'."
# The run's rate when --learning-rate names none: 8e-5 trains a rank-32 LoRA of
# Qwen3.5-9B on GSM8K; the full-weight rate is unmeasured.
LEARNING_RATES = {"full": 1e-6, "lora": 8e-5}


def gsm8k_problems(split: str) -> list[dict[str, str]]:
    """One of GSM8K's splits as problems, downloaded into Hugging Face's
    cache: each question, and the number its worked answer ends on."""
    records = load_dataset(GSM8K, "main", split=split, revision=GSM8K_REVISION)
    # Naming a split loads that one Dataset, not the dict of every split.
    assert isinstance(records, Dataset)
    return [
        {"question": record["question"], "answer": record["answer"].rpartition("####")[2].strip()}
        for record in records.to_list()
    ]


def read_problems(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def asked(problem: dict[str, Any]) -> list[Message]:
    """A problem as the model is asked it."""
    return [{"role": "user", "content": f"{problem['question']}\n{INSTRUCTION}"}]


def tasks(
    problems: Sequence[dict[str, Any]], *, max_episode_tokens: int
) -> list[AgentSession.Config]:
    """Every problem as a task: one turn, scored by its final number."""
    result = []
    for problem in problems:
        try:
            verifier = FinalNumber.Config(answer=str(problem["answer"]))
        except ValueError as error:
            raise SystemExit(f"the problem {problem['question'][:60]!r}: {error}") from None
        result.append(
            AgentSession.Config(
                messages=asked(problem), verifier=verifier, max_episode_tokens=max_episode_tokens
            )
        )
    return result


def longest_prompt(formatter: MessageFormatter, problems: Sequence[dict[str, Any]]) -> int:
    return max(len(formatter.build_generation_prompt(asked(problem))) for problem in problems)


def held_out(report: train_rl.StepReport) -> str:
    """A report's held-out share, with its standard error; empty without one."""
    share = report.metrics.get("eval/reward")
    if share is None:
        return ""
    return f"held out {share:.3f} ± {report.metrics['eval/reward_stderr']:.3f}"


def reward_over(reports: Sequence[train_rl.StepReport]) -> str:
    """The mean reward over ``reports``' steps, and the steps it covers."""
    reward = sum(report.metrics["reward"] for report in reports) / len(reports)
    return f"{reward:.3f} over steps {reports[0].step} to {reports[-1].step}"


async def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    formatter = message_formatter(args.model, tokenizer, enable_thinking=args.thinking)
    problems = read_problems(args.data) if args.data else gsm8k_problems("train")
    if not problems:
        raise SystemExit(f"{args.data} holds no problem, so there is nothing to train")
    unseen: list[dict[str, Any]] = []
    if args.eval_data:
        unseen = read_problems(args.eval_data)
    elif not args.data:
        unseen = gsm8k_problems("test")
    unseen = unseen[: args.eval_problems]
    # An episode is one turn, so its budget is the longest prompt and a turn's budget.
    # The run and its sampler are sized to hold it.
    longest = longest_prompt(formatter, [*problems, *unseen]) + args.max_turn_tokens
    dataset = TaskDataset(
        tasks(problems, max_episode_tokens=longest), batch_size=args.batch_size, seed=args.seed
    )
    print(f"{len(problems)} problems, {len(unseen)} held out. The first is asked as:")
    print(f"  {tokenizer.decode(formatter.build_generation_prompt(asked(problems[0])))!r}")
    # The loop attempts these greedily, so the share is of the model's best answers.
    held_out_tasks = tasks(unseen, max_episode_tokens=longest)
    kind = Full() if args.full else Lora(rank=args.lora_rank)
    rate = LEARNING_RATES["full" if args.full else "lora"]
    if args.learning_rate is not None:
        rate = args.learning_rate
    training = TrainingConfig(
        model=args.model,
        kind=kind,
        tokens_per_step=args.batch_size * args.group_size * longest,
        max_seq_len=longest,
        optimizer=AdamW(learning_rate=rate, betas=(0.9, 0.95), max_grad_norm=1.0),
    )
    loop_config = train_rl.Config(
        params=SamplingParams(max_tokens=args.max_turn_tokens, temperature=1.0),
        group_size=args.group_size,
        eval_every=args.eval_every,
    )

    def report(step: train_rl.StepReport) -> None:
        if step.result is None or step.version is None:
            print(f"before step {step.step + 1}: {held_out(step)}")
            return
        line = (
            f"step {step.step}/{args.steps}: reward {step.metrics['reward']:.3f} "
            f"on episodes of step {step.version.step}, lr {step.result.lr:.2e}"
        )
        print(", ".join(part for part in (line, held_out(step)) if part))

    config = ServiceClient.Config(base_url=args.url, project=args.project, display_name="rl-basic")
    async with await config.build() as service:
        print("starting the run and its sampler; each can take minutes")
        training_client = await service.create_training_client(training, timeout_s=20 * 60)
        sampler = await training_client.create_sampling_client(
            max_seq_len=longest,
            max_concurrent_sequences=min(args.concurrency, args.batch_size * args.group_size),
            timeout_s=20 * 60,
        )
        reports = await train_rl.train(
            training_client,
            sampler,
            dataset,
            loop_config,
            formatter=formatter,
            steps=args.steps,
            eval_tasks=held_out_tasks,
            on_step=report,
        )
        trained = [step for step in reports if step.result is not None]
        if trained:
            # A tenth of the run at each end: one step's reward is as noisy as its batch.
            window = max(1, len(trained) // 10)
            print(f"reward: {reward_over(trained[:window])}, {reward_over(trained[-window:])}")
        if held_out_tasks and reports:
            print(f"before: {held_out(reports[0])}; after: {held_out(reports[-1])}")
        name = args.checkpoint or f"rl-basic-{uuid4().hex[:8]}"
        saved = await (await training_client.save_state(name)).result(timeout_s=300)
        # A LoRA's upload takes seconds; a 9B full checkpoint is 107.5 GB and takes minutes.
        ready = await service.wait_checkpoint(saved.id, timeout_s=15 * 60)
        print(f"checkpoint {name!r}: {ready.state.value}, {ready.size_bytes} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An RL job: GSM8K, or your math problems.")
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--project", default="dev")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tokenizer", help="the tokenizer's name or directory; the model's")
    parser.add_argument("--data", help="a JSONL file of problems; GSM8K without it")
    parser.add_argument(
        "--eval-data", help="a JSONL file of held-out problems; GSM8K's test split without --data"
    )
    parser.add_argument("--eval-problems", type=int, help="the held-out problems measured; all")
    parser.add_argument("--eval-every", type=int, help="steps between held-out measurements")
    parser.add_argument("--thinking", action="store_true", help="let the model reason first")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--full", action="store_true", help="train every parameter instead")
    parser.add_argument("--max-turn-tokens", type=int, default=512, help="a turn's token budget")
    parser.add_argument("--steps", type=int, default=50, help="optimizer steps")
    parser.add_argument("--batch-size", type=int, default=32, help="problems per step")
    parser.add_argument("--group-size", type=int, default=16, help="attempts per problem")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=192,
        help="turns the sampler runs at once: one Qwen3.5-9B GPU's 192 by default",
    )
    parser.add_argument("--seed", type=int, default=0, help="fixes each pass's order")
    parser.add_argument("--learning-rate", type=float, help="the run's rate")
    parser.add_argument("--checkpoint", help="the checkpoint's name; rl-basic-… by default")
    asyncio.run(main(parser.parse_args()))
