# Tutorials

The tutorials are Jupyter notebooks that teach Looper one step at a time, from asking a model a first question to training with the cookbook's loops. Each one is both a page here, which shows what a real run printed under each cell, and a notebook you can run yourself.

## Before you start

- **Python 3.12 or newer, and Jupyter.** Run a notebook's cells in order, in a fresh kernel. You don't need a GPU, because the training and sampling run on Looper's GPUs.
- **An API key** for an organization with credit. [Authenticate](../../server/docs/session.md#authenticate) shows where to create one. Don't share or commit a notebook with your key in it.
- **The settings cell.** Near the top of each notebook, one cell holds its settings, such as the model and the tokenizer. Notebooks that call Looper also have:
  - `BASE_URL`: Looper's address, `https://api.looper.goaly.ai`.
  - `PROJECT`: any name you choose. The checkpoints a notebook saves are kept under it.
  - `API_KEY`: your key. Replace `YOUR_API_KEY` with it, or set `API_KEY = None` to have the SDK read the key from `LOOPER_API_KEY` in the environment Jupyter starts in.
- **What it costs.** Every notebook except Chat data runs on Looper's GPUs, and spends your organization's credit. Training is billed by the tokens it trains on. Sampling is billed by the prompt tokens it reads and the tokens it writes. Each notebook closes the sessions it opens, even when a cell fails, which shuts down their training runs and samplers. Any checkpoints it saved stay in your project.
- **What to install.** The 100-level notebooks install the SDK, `goaly-looper`, in their first cell. The others also use the cookbook's package, `looper_cookbook`, which isn't on PyPI yet, so install it from a clone of its repository, as [The cookbook](../docs/index.md#run-it) shows.
- **In Colab.** A 100-level tutorial's page has an **Open in Colab** button, which opens the notebook from the cookbook's repository on GitHub. Paste your key into `API_KEY`. Or, to keep it out of the notebook, save it as a Colab secret named `LOOPER_API_KEY`, using the key icon in Colab's left bar, set `API_KEY = None`, and run this in a cell before the others:

  ```python
  import os

  from google.colab import userdata

  os.environ["LOOPER_API_KEY"] = userdata.get("LOOPER_API_KEY")
  ```

## 100: The basics

Read these in order. Each builds on the one before, and uses the SDK alone, without the cookbook.

| # | Tutorial | What you'll learn | Read first |
|---|---|---|---|
| 101 | [Hello Looper](101_hello_looper.ipynb) | Connect to Looper, start a base model and ask it a question. You'll also see how prompts become token ids, why sampling takes two awaits, and how to get several answers to one prompt | Nothing |
| 102 | [Your first fine-tune](102_first_sft.ipynb) | Turn questions and answers into training data, train a LoRA of Qwen3.5-9B with `forward_backward` and `optim_step`, save it as a checkpoint, and sample it | 101 |
| 103 | [Operations](103_operations.ipynb) | Why each call has two awaits, how sending steps ahead keeps the GPUs busy, and where an error shows up when a call fails | 102 |
| 104 | [Your first RL run](104_first_rl.ipynb) | Write each round of RL yourself, which means publishing the weights to the run's sampler, getting a group of answers to each problem, scoring them, and training on the scores with `Grpo`. You'll also see why a problem the model always gets right trains nothing | 103 |

## 200: Core concepts

This one stands alone, once you've read the basics.

| # | Tutorial | What you'll learn | Read first |
|---|---|---|---|
| 201 | [Chat data](201_chat_data.ipynb) | Use a model's message formatter to write prompts in its chat format, turn conversations into training data, see which tokens each example trains, switch thinking on and off, read a reply back, and format a tool call. It runs on your machine alone, with no API key | 102 |

## 300: The cookbook's loops

Train on your own data with the cookbook's SFT and RL loops.

| # | Tutorial | What you'll learn | Read first |
|---|---|---|---|
| 301 | [The SFT loop](301_sft_loop.ipynb) | Fine-tune on a file of conversations with the message formatter and `train_sft.train`, with a learning rate that lowers as training goes, then sample the trained model | 103, 201 |
| 302 | [The RL loop](302_rl_loop.ipynb) | Turn math problems into tasks, each with a verifier, keep some aside to measure with, and train with `train_rl.train`, measuring the model before and after | 104, 301 |

## Next steps

- [The cookbook](../docs/index.md): the loops, message formatters and recipes these tutorials use.
- [Training](../../server/docs/training.md) and [Sampling](../../server/docs/sampling.md): every call the tutorials make, in full.
- [The SDK reference](../../sdk/docs/api-reference.md): every client, value and error.
