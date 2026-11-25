# Tutorials

The tutorials are Jupyter notebooks that teach Looper a step at a time, from a first sample to the cookbook's training loops. Each one is a page here, with what a real run printed under each cell, and a notebook you can run yourself.

## Before you start

- **Python 3.12 or newer, and Jupyter.** Run a notebook's cells in order, in a fresh kernel. It needs no GPU of yours: its training and sampling run on Looper's.
- **An API key** of an organization with credit. [Authenticate](../../server/docs/session.md#authenticate) says where to create one. Don't share or commit a notebook with your key in it.
- **The settings cell.** One cell near each notebook's top holds its settings, such as the model and the tokenizer, and, where it calls Looper, `BASE_URL`, `PROJECT` and `API_KEY`. `BASE_URL` is production's API root, `https://api.looper.goaly.ai`. For dev, set it to `https://dev.api.looper.goaly.ai` and use a dev key. `PROJECT` is any name you choose: the checkpoints a notebook saves go into it. Replace `API_KEY`'s `YOUR_API_KEY` with your key, or set `API_KEY = None` to have the SDK read the key from `LOOPER_API_KEY` in the environment Jupyter starts in.
- **What it costs.** Every notebook but Chat data runs on Looper's GPUs and spends your organization's credit. Training is billed by the tokens it trains, and sampling by the prompt tokens it reads and the tokens it generates. A notebook closes each session it opens, also when a cell fails, which retires the session's runs and samplers. The checkpoints it saved stay in your project.
- **What to install.** A 100-level notebook installs the SDK, `goaly-looper`, in its first cell. The others also use the cookbook's package, `looper_cookbook`, which is not on PyPI yet: install it from a clone of its repository, as [The cookbook](../docs/index.md#run-it) shows.
- **In Colab.** A 100-level tutorial's page has an **Open in Colab** button, which opens the notebook from the cookbook's repository on GitHub. Paste your key into `API_KEY`, or, to keep it out of the notebook, save it as a Colab secret named `LOOPER_API_KEY` (the key icon in Colab's left bar), set `API_KEY = None`, and run this in a new cell before the notebook's own:

  ```python
  import os

  from google.colab import userdata

  os.environ["LOOPER_API_KEY"] = userdata.get("LOOPER_API_KEY")
  ```

## 100: The basics

Read these in order. Each builds on the one before, with the SDK alone.

| # | Tutorial | What you'll learn | Read first |
|---|---|---|---|
| 101 | [Hello Looper](101_hello_looper.ipynb) | A key, a session, and a sampler of a base model: prompts as token ids, a sample's two awaits, several completions of one prompt | Nothing |
| 102 | [Your first fine-tune](102_first_sft.ipynb) | Question-and-answer pairs as tokens and weights; a LoRA of Qwen3.5-9B trained with `forward_backward` and `optim_step`; a checkpoint, and a sampler of the saved adapter | 101 |
| 103 | [Operations](103_operations.ipynb) | An operation's acceptance and its result; a run stepped one await at a time, then with every call submitted ahead; where an error surfaces | 102 |
| 104 | [Your first RL run](104_first_rl.ipynb) | A run's own sampler; rounds of RL written out: publish, sample a group of answers per problem, score them, train them with `Grpo`; why a task the model always solves trains nothing | 103 |

## 200: Core concepts

Each of these stands alone, once you have read the basics.

| # | Tutorial | What you'll learn | Read first |
|---|---|---|---|
| 201 | [Chat data](201_chat_data.ipynb) | The message formatter of a model: a prompt in the model's own format, conversations as a datum per assistant turn, what each one trains, the thinking switch, a reply read back, and a tool call. It runs on your machine alone, with no key | 102 |

## 300: The cookbook's loops

The cookbook's SFT and RL loops, on data of your own.

| # | Tutorial | What you'll learn | Read first |
|---|---|---|---|
| 301 | [The SFT loop](301_sft_loop.ipynb) | A file of conversations through the model's message formatter and `train_sft.train`, with the loop's learning-rate schedule, then a sampler of the adapter | 103, 201 |
| 302 | [The RL loop](302_rl_loop.ipynb) | A run's own sampler; math problems as tasks, each with a verifier; held-out tasks; then `train_rl.train`, with the baseline and the result measured | 104, 301 |

## Next steps

- [The cookbook](../docs/index.md): the loops, the message formatter and the recipes these tutorials use.
- [Training](../../server/docs/training.md) and [Sampling](../../server/docs/sampling.md): every call the tutorials make, in full.
- [The SDK reference](../../sdk/docs/api-reference.md): every client, value and error.
