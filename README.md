# goaly-looper-cookbook

Training loops, recipes and tutorials for Looper, Goaly's hosted
post-training service, written over its Python SDK,
[`goaly-looper`](https://pypi.org/project/goaly-looper/). Run a recipe as it
is, or copy a loop and change it: your program holds the data, the rewards
and the loop, and Looper runs the training and sampling on its GPUs.

The documentation is at <https://looper.goaly.ai/wiki/>. Its
[Cookbook](https://looper.goaly.ai/wiki/cookbook.html) tab explains the
message formatters, the SFT and RL loops, episode sessions, environments,
verifiers and recipes, and its
[Tutorials](https://looper.goaly.ai/wiki/tutorials.html) tab goes from a
first sample to the loops.

This is a beta: the loops and recipes may change between releases.

## Install

Python 3.12 or newer. The cookbook is not on PyPI yet, so install it from a
clone, with the `recipes` extra, which adds `datasets` for the recipes'
data:

```bash
git clone https://github.com/goalyinc/looper-cookbook.git
cd looper-cookbook
pip install -e ".[recipes]"
```

This installs the SDK from PyPI. The import package is `looper_cookbook`.

## Run a recipe

Every call needs a Looper API key in `LOOPER_API_KEY`.
[Session](https://looper.goaly.ai/wiki/session.html#authenticate) says
where to create one.

```bash
export LOOPER_API_KEY=sk-...
python -m looper_cookbook.recipes.sft_basic
python -m looper_cookbook.recipes.rl_basic --steps 20
```

| Recipe | What it does |
|---|---|
| `sft_basic` | Fine-tunes a LoRA of Qwen3.5-9B on GSM8K, or on your own conversations with `--data` |
| `rl_basic` | Trains a LoRA of Qwen3.5-9B with RL on GSM8K's problems, and measures the test split before and after |
| `code_rl.train` | Trains a LoRA with RL on DeepCoder's programming problems, running the model's programs in Docker on your machine: [its page](https://looper.goaly.ai/wiki/code-rl.html) |
| `sampling_basic` | Samples a base model or one of your checkpoints on a few math questions |

Each `--help` lists its flags, and
[Recipes](https://looper.goaly.ai/wiki/recipes.html) has the rest: your own
data, another model, full-weight training.

## What is here

| Path | What it holds |
|---|---|
| `looper_cookbook/` | The package: the message formatters, the SFT and RL loops (`train_sft.py`, `train_rl.py`), environments, sandboxes, episode sessions and verifiers, the recipes and example data. Each subpackage's README shows its components in use |
| `tutorials/` | The tutorials as Jupyter notebooks, the Tutorials tab's pages |
| `docs/` | The Cookbook tab's pages |

## Writing your own loop

What a loop over the SDK's `TrainingClient` has to get right, from datums
and their two awaits to schedules, saves and an RL loop's `publish_weights`, is
in [Training](https://looper.goaly.ai/wiki/training.html). The cookbook's
loops follow it.

## License

Apache-2.0: see `LICENSE`.
