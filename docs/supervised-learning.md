# Supervised learning

Supervised fine-tuning (SFT) trains a model on examples of what it should write. You give it conversations with good assistant replies, and the model learns to write replies like them. For example, if your conversations ask about animals and the replies always list the animal's habitat, diet and a fun fact, the trained model learns to answer in that shape.

The cookbook's SFT loop, `looper_cookbook.train_sft`, runs this training for you. You provide the conversations and create the training run. The cookbook turns the conversations into training data, and runs the optimizer steps.

This page covers:

- [Conversations and the dataset](#conversations-and-the-dataset): load your conversations, and split them into batches.
  - [A dataset of your own](#a-dataset-of-your-own): feed the loop data that isn't a file of conversations.
- [The loop](#the-loop): run the training.
- [The config](#the-config): the learning rate and the loop's other settings.
- [What the loop leaves to you](#what-the-loop-leaves-to-you): the parts you write yourself.

Words used on this page:

| Word | Meaning |
|---|---|
| **Conversation** | A list of chat messages, such as a user's question and the assistant's reply. Your data is a file of these. |
| **Datum** | One training example. It holds a reply's prompt as context, and the reply itself as the tokens the model learns to write. See [Prepare data](../../server/docs/training.md#prepare-data). |
| **Batch** | The datums trained in one optimizer step. |
| **Step** | One optimizer step, which updates the model's weights once, from one batch. |
| **Pass** | One time through every conversation in your data. Also called an epoch. |

## Conversations and the dataset

Your data is a file of conversations, each a list of chat messages with a `role` and a `content`. To train on it, the cookbook needs to do two things:

1. **Turn each conversation into datums.** The model's [message formatter](message-formatting.md) does this. It writes each conversation in the model's own chat format, and makes one datum for each assistant reply.
2. **Split the datums into batches,** one batch for each optimizer step. A **dataset** does this. `ChatDataset` is the cookbook's dataset for conversations.

The code below builds a dataset from the example file that comes with the cookbook, which has 16 conversations about animals. It then prints how many datums each of the first four steps trains, and the text of the first datum's reply:

```python
from importlib import resources

from transformers import AutoTokenizer

from looper_cookbook.datasets.conversations import ChatDataset, read_conversations
from looper_cookbook.message_formatters.models import message_formatter

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
formatter = message_formatter("Qwen/Qwen3.5-9B", tokenizer, enable_thinking=False)
path = resources.files("looper_cookbook") / "example_data" / "conversations.jsonl"
dataset = ChatDataset(read_conversations(path), formatter, batch_size=8, max_seq_len=2048, seed=0)
print(len(dataset), "batches a pass")
for step in range(4):
    print(step, len(dataset.batch(step)), "datums")
print("the first trains", formatter.targets(dataset.batch(0)[0]))
```

```output
2 batches a pass
0 10 datums
1 8 datums
2 10 datums
3 8 datums
the first trains ['Habitat: the dry plains of inland Australia.\nDiet: grasses and shrubs.\nFun fact: it uses its tail as a fifth leg when it walks slowly.<|im_end|>']
```

The 16 conversations, at 8 per batch, make 2 batches per pass. Two of the conversations have a second assistant reply, so the whole pass is 18 datums: 10 in one batch and 8 in the other. Steps 2 and 3 are the second pass.

- **The file is JSONL,** with one conversation per line, as an object with `messages` and optionally `tools`. `read_conversations` reads it.
- **A batch is `batch_size` conversations,** each turned into one datum per assistant reply. So a batch can hold more datums than conversations.
- **`targets(datum)` shows the text a datum trains on,** so you can check your data before you train.
- **`max_seq_len` is the longest a datum can be.** Your training run's `max_seq_len` must be at least as long.
- **Bad conversations fail early.** Every conversation is written when the dataset is built. One the model's format can't hold fails then, before you create a run.
- **The run's step picks the batch.** Step `k`, counted from 0, trains `dataset.batch(k)`. So if the loop stops, after an error or a rollback, and you call it again on the same run, it carries on from where the run is. You don't need to keep track of where it was.
- **A dataset never runs out.** After one pass, the next step starts another pass. Each pass has its own order, set by the seed and the pass number. To train for several passes, ask the loop for that many steps.

### A dataset of your own

If your data isn't a file of conversations, you can write your own dataset. The loop only needs two methods:

- **`__len__`:** how many batches are in one pass.
- **`batch(index)`:** the datums for step `index`.

That's the `SupervisedDataset` contract, in `looper_cookbook.datasets.base`. The example below is a complete dataset of your own, which gives every step the same datums. It's useful for checking that a run can learn one batch by heart:

```python
from collections.abc import Sequence

from looper_cookbook.datasets.base import SupervisedDataset
from looper_sdk import Datum


class OneBatch:
    def __init__(self, datums: Sequence[Datum]) -> None:
        self.datums = datums

    def __len__(self) -> int:
        return 1

    def batch(self, index: int) -> Sequence[Datum]:
        return self.datums


overfit: SupervisedDataset = OneBatch(dataset.batch(0))
print(len(overfit), "batch a pass;", len(overfit.batch(7)), "datums at step 7")
```

```output
1 batch a pass; 10 datums at step 7
```

To split items into batches the way `ChatDataset` does, in a different seeded order for each pass, use `batch_positions`, from the same module. It tells you which items a step holds.

## The loop

Once you have a training client and a dataset, `train` runs the training. It takes the training client, the dataset and the loop's config:

```python
from looper_cookbook import train_sft

config = train_sft.Config(learning_rate=2e-4)
results = await train_sft.train(training_client, dataset, config, steps=4 * len(dataset))
print(results[-1].loss_sum / results[-1].weighted_targets)
```

`train(training_client, dataset, config, *, steps, on_step=None)` trains the run from its current step up to step `steps`, one step per batch, and returns each step's result. Here, `4 * len(dataset)` steps is four passes over the data. The last line prints the final step's loss per target token.

This assumes you've already created `training_client`, as [Create a training client](../../server/docs/training.md#create-a-training-client) shows.

## The config

| Field | Default | What it sets |
|---|---|---|
| `learning_rate` | No default, so you must set it | The learning rate of the run's first step. |
| `schedule` | `"linear"` | How the learning rate changes over the steps. `"linear"` and `"cosine"` lower it over the steps, so it would reach 0 at the step after the last, and `"constant"` keeps it the same. The run's optimizer doesn't keep a schedule itself, so the loop sets each step's rate. |
| `steps_ahead` | 1 | How many steps the loop sends ahead of the one it's waiting on, so the run never waits for your machine. See [Submit ahead](../../server/docs/training.md#submit-ahead). |
| `max_call_tokens` | 1,000,000 | The most tokens one call to Looper carries. A bigger batch is split over several calls, whose gradients are added up into the one step. |

## What the loop leaves to you

- **The training run,** created with its `TrainingConfig`, which sets the model, LoRA or all parameters, and its sizes.
- **The dataset.**
- **Saving the model,** once `train` returns. See [Checkpoint](../../server/docs/checkpoint.md).

`looper_cookbook/train_sft.py` is short enough to read. If your training needs to work differently, you can copy it into your project and change it.

## Next steps

- [The SFT loop](../tutorials/301_sft_loop.ipynb) trains a file of conversations with this loop, and samples the result.
- [Chat data](../tutorials/201_chat_data.ipynb) walks through datums on your machine, with no API key.
- [Recipes](recipes.md) has `sft_basic`, which is this loop as a ready-to-run script, on GSM8K or your own file.
