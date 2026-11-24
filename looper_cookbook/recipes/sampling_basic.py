"""Standalone sampling against Looper's ``/v1``, through the SDK alone.

It starts a sampler of a base model, or with ``--checkpoint`` of a
checkpoint saved in the same project (``sft_basic.py`` names its checkpoints
``sft-basic-…``), sends every question at once, each for ``--n``
completions, waits for all of them and prints them with each request's token
counts, then the sampler's totals, then stops the sampler.
Each question is a grade-school math problem, as ``sft_basic.py``'s default
data is, prompted through the model's message formatter with thinking off:
the prompt ``sft_basic.py`` trains after.
The service runs the requests concurrently, in the order it accepted them,
and they may finish in any order. A LoRA checkpoint is served over its base;
name that base with ``--model``, for its tokenizer.

    LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.sampling_basic
    python -m looper_cookbook.recipes.sampling_basic --n 2
    python -m looper_cookbook.recipes.sampling_basic --checkpoint sft-basic-1a2b3c4d
    python -m looper_cookbook.recipes.sampling_basic --model Qwen/Qwen3.8-27B
"""

from __future__ import annotations

import argparse
import asyncio

from transformers import AutoTokenizer

from looper_cookbook.message_formatters.models import message_formatter
from looper_sdk import (
    DEFAULT_BASE_URL,
    SamplerConfig,
    SamplingParams,
    SamplingUsage,
    ServiceClient,
)

MODELS = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B")
QUESTIONS = [
    "A baker makes 24 muffins and packs them in boxes of 6. How many boxes does she fill?",
    "Tom has 15 marbles. He gives 4 to his sister, then buys 7 more. How many does he have?",
    "A train travels 60 miles an hour for 3 hours. How many miles does it travel?",
    "Pencils cost 2 dollars each. Maria buys 5 and pays with 20 dollars. What is her change?",
]


def format_usage(usage: SamplingUsage | None) -> str:
    if usage is None:
        return "(no token counts)"
    return (
        f"(prompt {usage.num_prompt_tokens}, generated {usage.num_generated_tokens}, "
        f"cached {usage.num_cached_tokens} tokens)"
    )


async def main(url: str, model: str, checkpoint: str | None, n: int, max_tokens: int) -> None:
    # The prompt sft_basic trains after: the model's format, in non-thinking mode.
    tokenizer = AutoTokenizer.from_pretrained(model)
    formatter = message_formatter(model, tokenizer, enable_thinking=False)
    prompts = [
        formatter.build_generation_prompt([{"role": "user", "content": question}])
        for question in QUESTIONS
    ]
    params = SamplingParams(
        max_tokens=max_tokens, temperature=0.7, stop_token_ids=formatter.stop_token_ids
    )
    # A sampler serves its context and concurrency for its life; n completions
    # of a request take n of its sequences while they generate.
    sampler_config = SamplerConfig(
        checkpoint_name=checkpoint or f"base:{model}",
        max_seq_len=4096,
        max_concurrent_sequences=16,
    )
    config = ServiceClient.Config(base_url=url, project="dev", display_name="sampling-basic")
    async with await config.build() as service:
        print("starting the sampler; the first one can take minutes")
        sampler = await service.create_sampling_client(sampler_config, timeout_s=20 * 60)
        try:
            # Each sample() returns once the request is queued; the results come later.
            operations = await asyncio.gather(
                *(sampler.sample(prompt, params=params, n=n) for prompt in prompts)
            )
            results = await asyncio.gather(*(op.result(timeout_s=600) for op in operations))
            usages = await asyncio.gather(*(op.usage() for op in operations))
            for question, result, usage in zip(QUESTIONS, results, usages, strict=True):
                print(f"{question} {format_usage(usage)}")
                for sequence in result.sequences:
                    reply, _ = formatter.parse_response(sequence.token_ids)
                    mean = sum(sequence.logprobs) / max(len(sequence.logprobs), 1)
                    reason = sequence.finish_reason.value
                    print(f"  -> {reply['content'].strip()!r} ({reason}, mean logprob {mean:.2f})")
            # The sampler's status sums the usage of every request it served.
            status = await sampler.status()
            print(f"sampler total: {format_usage(status.usage)}")
        finally:
            await sampler.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone sampling against Looper's /v1.")
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--checkpoint", help="a checkpoint's name in project dev")
    parser.add_argument("--n", type=int, default=2, help="completions of each question")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.model, args.checkpoint, args.n, args.max_tokens))
