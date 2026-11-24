# Supervised learning

`looper_cookbook.train_sft` fine-tunes a run on examples of what the model should write. Each example is a datum: a prompt as context, and the assistant's turn as the targets the run learns to predict. The loop takes one optimizer step per batch of datums.

## Conversations and the dataset

Your data is conversations of chat messages, each a `role` and a `content`. A dataset puts them in batches, one per optimizer step. `ChatDataset` is the cookbook's. The model's [message formatter](message-formatting.md) turns each conversation into a datum per assistant turn, written as the model's own chat template writes it:

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

- **A file of conversations** is JSONL: one object a line, with `messages` and optionally `tools`. `read_conversations` reads it.
- **A batch is `batch_size` conversations,** each a datum per assistant turn. Two of these 16 conversations have a second turn, so a pass is 18 datums.
- **`targets(datum)`** shows the text a datum trains, so you can check your data before you train it.
- **`max_seq_len`** is a datum's longest length. Your run's `max_seq_len` must hold it.
- **Every conversation is written when the dataset is built.** One the model's format cannot hold fails then, before you create a run.
- **The run's step picks the batch.** Step `k`, counted from 0, trains `dataset.batch(k)`. So the loop called again on the same run, after an error or a rollback, goes on from where the run stands, with nothing for you to keep track of.
- **A dataset never ends.** Past one pass, an index names a batch of the next pass. Each pass is in an order of its own, which the seed and the pass fix. Train several passes by asking the loop for that many steps.

### A dataset of your own

A dataset is anything with `__len__`, the batches of one pass, and `batch(index)`, the datums of a step. That is the `SupervisedDataset` contract, in `looper_cookbook.datasets.base`. This one gives every step the same datums, to check that a run can fit one batch:

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

To batch as `ChatDataset` does, in a seeded order of its own for each pass, ask `batch_positions`, in the same module, which items a step holds.

## The loop

`train` takes the run, the dataset and the loop's config:

```python
from looper_cookbook import train_sft

config = train_sft.Config(learning_rate=2e-4)
results = await train_sft.train(training_client, dataset, config, steps=4 * len(dataset))
print(results[-1].loss_sum / results[-1].weighted_targets)
```

- `train(training_client, dataset, config, *, steps, on_step=None)` trains the run from its current step up to step `steps`, one step per batch, and returns the result of each step it trained. Here `4 * len(dataset)` steps are four passes.

## The config

| Field | Default | What it sets |
|---|---|---|
| `learning_rate` | none | The rate of the run's first step |
| `schedule` | `"linear"` | How the rate falls from there: to 0 by `"linear"` or `"cosine"`, or not at all by `"constant"`. A run's optimizer keeps no schedule, so the loop names each step's rate |
| `steps_ahead` | 1 | How many steps the loop submits beyond the one it awaits, so the run never waits for your machine |
| `max_call_tokens` | 1,000,000 | The most tokens one call carries. A larger batch goes over several calls, whose gradients add up into the one step |

## What the loop leaves to you

- **The run,** created with its `TrainingConfig`: its model, a LoRA or every parameter, and its sizes.
- **The dataset.**
- **The save,** two lines after `train` returns.

`looper_cookbook/train_sft.py` is short enough to read, and meant to be copied and changed when your training needs something else.

## Next steps

- [The SFT loop](../tutorials/301_sft_loop.ipynb) trains a file of conversations with it and samples the result.
- [Chat data](../tutorials/201_chat_data.ipynb) walks through datums on your machine, with no key.
- [Recipes](recipes.md): `sft_basic` is this loop as a script, on GSM8K or your own file.
