# Recipes

A recipe is a ready-to-run script for a complete training or sampling job. It loads the data, trains or samples, and saves the result. Recipes are useful in two ways:

- **Run one as it is,** to see a cookbook loop working end to end, or to check that your API key and setup work.
- **Copy one and change it,** as the starting point for your own job.

This page covers:

- [The recipes](#the-recipes): what each one does.
- [Run a recipe](#run-a-recipe): the commands, and the flags most recipes share.
- [Your own data](#your-own-data): run a recipe on your own file instead of GSM8K.
- [What a recipe does not do yet](#what-a-recipe-does-not-do-yet): how to read a recipe's results.

## The recipes

The cookbook has four recipes. All of them use Qwen3.5-9B unless you name another model, and the training recipes train a LoRA unless you ask for full parameters.

| Recipe | What it does |
|---|---|
| `sft_basic` | Fine-tunes the model with the [SFT loop](supervised-learning.md). It trains on GSM8K's math problems, or on your own JSONL file of conversations with `--data`. It prints a line for each step, then the name of the checkpoint it saved. |
| `rl_basic` | Trains the model with the [RL loop](reinforcement-learning.md), on GSM8K's math problems. Each problem is a task whose reward is 1 when the answer's final number is right. It measures the model on GSM8K's test split before and after training. It prints each step's reward, the two test scores, then the name of the checkpoint it saved. |
| [`code_rl`](../looper_cookbook/recipes/code_rl/README.md) | Trains the model on DeepCoder's programming problems. While it answers, the model can test its program against two of the problem's tests, using a tool that runs in a Docker container on your machine. A verifier then scores its answer on the problem's other tests. It has a page of its own. |
| `sampling_basic` | Samples a model on a few math questions, without training. By default it samples Qwen3.5-9B. Use `--model` for Qwen3.8-27B, or `--checkpoint NAME` for a checkpoint saved in the project `dev`, such as one `sft_basic` or `rl_basic` saved with the default `--project`. It prints the answers, and each request's token counts. |

## Run a recipe

Run a recipe from the root of the repository, in its Python environment, with your API key set:

```bash
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.sft_basic --data my_conversations.jsonl
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.rl_basic --steps 20
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.code_rl.train --problems 32 --eval-problems 16 --steps 4 --batch-size 8
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.sampling_basic --checkpoint sft-basic-1a2b3c4d
```

- **Install the cookbook first,** from a clone of its repository, as [Run it](index.md#run-it) shows, and run the recipes from the clone's root.
- **`--project`** sets the project that `sft_basic` and `rl_basic` save into. It's `dev` by default.
- **`--url`** points the recipe at another Looper address. By default it uses `https://api.looper.goaly.ai`.
- **`--model`** picks another model, such as `--model meta-models/Muse-Glimmer-30B`. The recipe uses that model's message formatter.
- **`--full`** trains every parameter instead of a LoRA.
- **`--help`** lists each recipe's other flags.

## Your own data

`sft_basic` and `rl_basic` can train on your own JSONL file instead of GSM8K. Each line of the file is one example:

| Recipe | Flag | One line of the file |
|---|---|---|
| `sft_basic` | `--data` | `{"messages": [...]}`, which is a conversation of chat messages, with `tools` if it calls any. |
| `rl_basic` | `--data` | `{"question": "...", "answer": "..."}`, which holds a problem and the number that answers it. |
| `rl_basic` | `--eval-data` | The same, for held-out problems the run never trains on. |

`rl_basic` has two more flags for its held-out problems:

- **`--eval-problems N`** measures only N of them.
- **`--eval-every N`** also measures them every N steps. They're always measured before the first step and after the last.

The cookbook comes with two small example files in `looper_cookbook/example_data/`, `conversations.jsonl` and `math_problems.jsonl`. With them, you can run a recipe without downloading anything:

```bash
LOOPER_API_KEY=sk-… python -m looper_cookbook.recipes.rl_basic --data looper_cookbook/example_data/math_problems.jsonl --steps 4 --batch-size 8
```

## What a recipe does not do yet

No recipe checks its own result yet. Read the loss or the reward it prints and, for `rl_basic`, the held-out score before and after training. [Evaluation](evaluation.md#reading-the-numbers) explains how to read them.

## Next steps

- [Code RL](../looper_cookbook/recipes/code_rl/README.md) is the recipe that uses an environment and a verifier.
- [Supervised learning](supervised-learning.md) and [Reinforcement learning](reinforcement-learning.md) are the loops the recipes call.
- [The tutorials](../tutorials/index.md) build up to them, starting from a first sample.
