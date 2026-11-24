# Recipes

A recipe is a whole job as a script, from its data to its checkpoint. Run one as it is to check a deployment or to see a loop work, then copy it when your job differs.

| Recipe | What it does |
|---|---|
| `sft_basic` | Fine-tunes a LoRA of Qwen3.5-9B with the [SFT loop](supervised-learning.md), on GSM8K's math problems, or with `--data` on your own JSONL file of conversations. It prints a line per step, then its checkpoint's name |
| `rl_basic` | Trains a LoRA of Qwen3.5-9B with the [RL loop](reinforcement-learning.md), on GSM8K's problems. Each problem is a task whose reward is 1 when the answer's final number is right. It measures GSM8K's test split before it trains and after, and prints each step's reward, the two held-out shares, then its checkpoint's name |
| [`code_rl`](../looper_cookbook/recipes/code_rl/README.md) | Trains a LoRA of Qwen3.5-9B on DeepCoder's programming problems. The model may check a program against two of a problem's tests with a tool, in a Docker container on your machine, and a verifier scores its answer on the problem's other tests. It has a page of its own |
| `sampling_basic` | Samples Qwen3.5-9B's base, Qwen3.8-27B's with `--model`, or a checkpoint of your project with `--checkpoint NAME`, on a few math questions. It prints the completions and each request's token counts |

## Run a recipe

With the cookbook installed, as [The cookbook](index.md#run-it) shows, run a recipe from its repository's root, with your API key set:

```bash
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.sft_basic --data my_conversations.jsonl
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.rl_basic --steps 20
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.code_rl.train --problems 32 --eval-problems 16 --steps 4 --batch-size 8
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.sampling_basic --checkpoint sft-basic-1a2b3c4d
```

- **The project.** `sft_basic` and `rl_basic` save into the project `--project` names, `dev` by default. Each `--help` lists the rest of its flags.
- **Another deployment.** Each recipe calls `https://api.looper.goaly.ai` unless `--url` names another API root, such as dev's `https://dev.api.looper.goaly.ai`.
- **Another model.** `--model` names it, as in `--model meta-models/Muse-Glimmer-30B`, and the recipe uses that model's message formatter.
- **Every parameter.** `--full` trains every parameter in place of a LoRA.

## Your own data

| Recipe | Flag | A line of the file |
|---|---|---|
| `sft_basic` | `--data` | `{"messages": [...]}`, a conversation of chat messages, with `tools` if it calls any |
| `rl_basic` | `--data` | `{"question": "...", "answer": "..."}`, a problem and the number that answers it |
| `rl_basic` | `--eval-data` | The same, for problems the run never trains |

`rl_basic` takes two more flags for its held-out problems: `--eval-problems` measures only that many of them, and `--eval-every` measures them every that many steps as well as before the first step and after the last.

The cookbook ships two small files in `looper_cookbook/example_data/`, `conversations.jsonl` and `math_problems.jsonl`, so a recipe can run with nothing to download:

```bash
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.rl_basic --data looper_cookbook/example_data/math_problems.jsonl --steps 4 --batch-size 8
```

## What a recipe does not do yet

No recipe checks its own result. Read the loss or the reward it prints, and for `rl_basic` the held-out share before and after: [Evaluation](evaluation.md) says how to read them.

## Next steps

- [Code RL](../looper_cookbook/recipes/code_rl/README.md) is the recipe over an environment and a verifier.
- [Supervised learning](supervised-learning.md) and [Reinforcement learning](reinforcement-learning.md) are the loops the recipes call.
- [The tutorials](../tutorials/index.md) build up to them from a first sample.
