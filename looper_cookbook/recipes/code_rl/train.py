"""RL on DeepCoder's programming problems, through the cookbook's RL loop.
The recipe's page is ``README.md``.

``main`` is these steps, in order:

1. **The sandboxes.** Docker containers on this machine by default, one for
   an attempt's tool and one for its verifier. ``--sandbox local`` runs the
   model's programs as you, with no isolation.
2. **The tasks.** DeepCoder's train subsets, each problem asked under
   LiveCodeBench's prompt (``tasks.py``). An attempt is an ``AgentSession`` of
   ``--max-turns`` turns over a ``CodingEnvironment``: the model may call
   ``check_solution`` on the problem's first ``--public-tests`` tests, and
   then answers with its program in a code block. A ``CodeTestsVerifier``
   scores it 1 when the program passes every test the problem keeps, in a
   sandbox the model never acted in, else 0.
3. **The held-out problems.** DeepCoder's test subsets, attempted greedily:
   the share whose final program passes. The loop measures it before its
   first step, after its last, and every ``--eval-every`` steps between.
4. **The run and its sampler.** A LoRA of rank ``--lora-rank``, sized for
   ``--batch-size`` problems of ``--group-size`` attempts of
   ``--max-episode-tokens`` tokens each.
5. **The loop.** ``train_rl.train`` takes ``--steps`` steps, and ``show``
   prints each of its reports as a line.
6. **The checkpoint.** The recipe saves one and waits until it is uploaded.

**Two token budgets.** ``--max-turn-tokens`` is a turn's: the most tokens the
sampler returns for one turn. ``--max-episode-tokens`` is an attempt's: its
prompt, every turn and every tool reply together. An attempt whose last turn
either cuts off scores 0, and each report's line says which budget ended how
many.

    LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.code_rl.train

A trial that downloads little and takes minutes:

    python -m looper_cookbook.recipes.code_rl.train --problems 32 \\
        --eval-problems 16 --steps 4 --batch-size 8
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from uuid import uuid4

from transformers import AutoTokenizer

from looper_cookbook import train_rl
from looper_cookbook.datasets.tasks import TaskDataset
from looper_cookbook.environments.coding import TOOLS
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.message_formatters.base import MessageFormatter
from looper_cookbook.message_formatters.models import message_formatter
from looper_cookbook.recipes.code_rl import tasks
from looper_cookbook.sandboxes.base import Sandbox
from looper_cookbook.sandboxes.docker import DockerSandbox
from looper_cookbook.sandboxes.local import LocalSandbox
from looper_sdk import (
    DEFAULT_BASE_URL,
    AdamW,
    Full,
    Lora,
    SamplingParams,
    ServiceClient,
    TrainingConfig,
)

# The run's rate when --learning-rate names none, for a rank-32 LoRA or full
# weights. Neither is tuned for this recipe yet.
LEARNING_RATES = {"full": 1e-6, "lora": 4e-5}

# The metrics a report's line shows, of those the report holds. A step's own are of the
# attempts it trained on, and the ``eval/`` ones are of the held-out problems.
SHOWN_METRICS = (
    "reward",
    "groups_trained",
    "turns",
    "stop/max_turn_tokens",
    "stop/max_episode_tokens",
    "eval/reward",
    "eval/reward_stderr",
    "eval/stop/max_turn_tokens",
    "eval/stop/max_episode_tokens",
)


def fitting(
    problems: Sequence[tasks.Problem], formatter: MessageFormatter, *, room: int
) -> list[tasks.Problem]:
    """The problems whose first prompt, the tool's schema included, takes at
    most ``room`` tokens: an episode needs the rest for a turn."""

    def prompt_tokens(problem: tasks.Problem) -> int:
        asked = [{"role": "user", "content": problem.question}]
        return len(formatter.build_generation_prompt(asked, TOOLS))

    return [problem for problem in problems if prompt_tokens(problem) <= room]


def show(report: train_rl.StepReport) -> None:
    """Prints a report as one line: its step, then each of ``SHOWN_METRICS``
    it holds. The first line, of the step the run starts at, is the held-out
    problems before any training."""
    shown = [
        f"{name} {report.metrics[name]:.3f}" for name in SHOWN_METRICS if name in report.metrics
    ]
    print(f"step {report.step}: {', '.join(shown)}")


async def main(args: argparse.Namespace) -> list[train_rl.StepReport]:
    """Runs the recipe, and returns the loop's reports."""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    formatter = message_formatter(args.model, tokenizer, enable_thinking=args.thinking)

    # 1. The sandboxes, and the Python that runs a program inside one.
    if args.sandbox == "local":
        sandbox: Sandbox.Config = LocalSandbox.Config()
        python = sys.executable
    else:
        sandbox = DockerSandbox.Config(image=args.image, memory_mb=args.memory_mb, cpus=1.0)
        python = "python3"

    def load_tasks(split: str, limit: int | None) -> list[AgentSession.Config]:
        """A split's problems that fit an episode, each as a task."""
        problems = tasks.load(split, limit=limit, max_tests=args.max_tests)
        # A problem fits when the episode's budget holds its prompt and one whole turn.
        kept = fitting(problems, formatter, room=args.max_episode_tokens - args.max_turn_tokens)
        print(f"{split}: {len(kept)} problems, of {len(problems)} read")
        return [
            tasks.task(
                problem,
                sandbox=sandbox,
                python=python,
                public_tests=args.public_tests,
                timeout_s=args.timeout_s,
                max_turns=args.max_turns,
                max_episode_tokens=args.max_episode_tokens,
            )
            for problem in kept
        ]

    # 2. The tasks the run trains on, a batch of them a step.
    train_tasks = load_tasks("train", args.problems)
    if not train_tasks:
        raise SystemExit("no problem fits --max-episode-tokens, so there is nothing to train")
    dataset = TaskDataset(train_tasks, batch_size=args.batch_size, seed=args.seed)

    # 3. The held-out problems, attempted greedily. Nothing sampled for them trains.
    held_out = load_tasks("test", args.eval_problems) if args.eval_problems != 0 else []

    # 4. The run and its sampler, each sized to hold an attempt's token budget.
    kind = Full() if args.full else Lora(rank=args.lora_rank)
    rate = args.learning_rate
    if rate is None:
        rate = LEARNING_RATES["full" if args.full else "lora"]
    training = TrainingConfig(
        model=args.model,
        kind=kind,
        tokens_per_step=args.batch_size * args.group_size * args.max_episode_tokens,
        max_seq_len=args.max_episode_tokens,
        optimizer=AdamW(learning_rate=rate, betas=(0.9, 0.95), max_grad_norm=1.0),
    )
    config = ServiceClient.Config(base_url=args.url, project=args.project, display_name="code-rl")
    async with await config.build() as service:
        print("starting the run and its sampler; each can take minutes")
        training_client = await service.create_training_client(training, timeout_s=20 * 60)
        sampling_client = await training_client.create_sampling_client(
            max_seq_len=args.max_episode_tokens,
            max_concurrent_sequences=min(args.concurrency, args.batch_size * args.group_size),
            timeout_s=20 * 60,
        )

        # 5. The loop: an optimizer step per batch, and a printed line per report.
        loop_config = train_rl.Config(
            params=SamplingParams(max_tokens=args.max_turn_tokens, temperature=1.0),
            group_size=args.group_size,
            eval_every=args.eval_every,
            save_every=args.save_every,
        )
        reports = await train_rl.train(
            training_client,
            sampling_client,
            dataset,
            loop_config,
            formatter=formatter,
            steps=args.steps,
            eval_tasks=held_out,
            on_step=show,
        )

        # 6. The checkpoint, saved and then waited for until it is uploaded.
        name = args.checkpoint or f"code-rl-{uuid4().hex[:8]}"
        saved = await (await training_client.save_state(name)).result(timeout_s=300)
        ready = await service.wait_checkpoint(saved.id, timeout_s=15 * 60)
        print(f"checkpoint {name!r}: {ready.state.value}, {ready.size_bytes} bytes")
    return reports


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL on DeepCoder's programming problems.")
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--project", default="dev")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tokenizer", help="the tokenizer's name or directory; the model's")
    parser.add_argument("--thinking", action="store_true", help="let the model reason first")
    parser.add_argument(
        "--problems", type=int, help="train problems read of each subset; all, downloaded whole"
    )
    parser.add_argument(
        "--eval-problems", type=int, default=64, help="held-out problems of each subset; 0 for none"
    )
    parser.add_argument("--eval-every", type=int, help="steps between held-out measurements")
    parser.add_argument("--save-every", type=int, help="steps between saves of the run's state")
    parser.add_argument("--max-tests", type=int, default=32, help="tests a problem keeps")
    parser.add_argument("--public-tests", type=int, default=2, help="tests check_solution runs")
    parser.add_argument("--timeout-s", type=float, default=6.0, help="a solution's time on a test")
    parser.add_argument("--sandbox", choices=("docker", "local"), default="docker")
    parser.add_argument("--image", default="python:3.12-slim", help="the Docker sandboxes' image")
    parser.add_argument("--memory-mb", type=int, default=1024, help="a Docker sandbox's memory")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--full", action="store_true", help="train every parameter instead")
    parser.add_argument("--max-turns", type=int, default=2, help="an attempt's turns")
    parser.add_argument(
        "--max-turn-tokens",
        type=int,
        default=2048,
        help="a turn's token budget: the most tokens sampled for one turn",
    )
    parser.add_argument(
        "--max-episode-tokens",
        type=int,
        default=8192,
        help="an attempt's token budget, over its prompt, turns and tool replies: "
        "Qwen3.5-9B's RL sampler holds 8,192",
    )
    parser.add_argument("--steps", type=int, default=100, help="optimizer steps")
    parser.add_argument("--batch-size", type=int, default=16, help="problems per step")
    parser.add_argument("--group-size", type=int, default=8, help="attempts per problem")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=192,
        help="turns the sampler runs at once: one Qwen3.5-9B GPU's 192 by default",
    )
    parser.add_argument("--seed", type=int, default=0, help="fixes each pass's order")
    parser.add_argument("--learning-rate", type=float, help="the run's rate")
    parser.add_argument("--checkpoint", help="the checkpoint's name; code-rl-… by default")
    return parser


if __name__ == "__main__":
    asyncio.run(main(arguments().parse_args()))
