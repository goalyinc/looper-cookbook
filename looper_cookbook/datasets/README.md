# datasets/

Datasets: a run's data, a batch per optimizer
step. Each loop reads its batches through a contract, by the run's step:
the SFT loop reads datums through `SupervisedDataset`, and the RL loop
tasks through `RLDataset`. `ChatDataset` and `TaskDataset` are the
cookbook's.
[Supervised learning](../../docs/supervised-learning.md) and
[Reinforcement learning](../../docs/reinforcement-learning.md) are the
pages for users.

## Usage

`ChatDataset` batches conversations for the SFT loop. The model's
[message formatter](../message_formatters/README.md) writes each
conversation as a datum per assistant turn, and `read_conversations` reads
a JSONL file of them. It needs a tokenizer:

```python
from importlib import resources

from transformers import AutoTokenizer

from looper_cookbook.datasets.conversations import ChatDataset, read_conversations
from looper_cookbook.message_formatters.models import message_formatter

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
formatter = message_formatter("Qwen/Qwen3.5-9B", tokenizer, enable_thinking=False)
path = resources.files("looper_cookbook") / "example_data" / "conversations.jsonl"
dataset = ChatDataset(read_conversations(path), formatter, batch_size=8, max_seq_len=2048)
print(len(dataset), len(dataset.batch(0)))  # 2 10: the batches of a pass, the first one's datums
```

`TaskDataset` batches tasks for the RL loop. A task is an episode session's
config:

```python
from looper_cookbook.datasets.tasks import TaskDataset
from looper_cookbook.episode_sessions.agent_session import AgentSession
from looper_cookbook.verifiers.final_number import FinalNumber

problems = {"6 times 7?": "42", "9 times 8?": "72", "12 times 12?": "144", "7 times 8?": "56"}
tasks = [
    AgentSession.Config(
        messages=[{"role": "user", "content": question}],
        verifier=FinalNumber.Config(answer=answer),
        max_episode_tokens=1024,
    )
    for question, answer in problems.items()
]
dataset = TaskDataset(tasks, batch_size=3, seed=0)
print(len(dataset), [len(dataset.batch(step)) for step in range(4)])  # 2 [3, 1, 3, 1]
```

A loop trains `dataset.batch(k)` at the run's step `k`, pass after pass:
hand the dataset to `train_sft.train` or `train_rl.train`. The
[Supervised learning](../../docs/supervised-learning.md#conversations-and-the-dataset)
and [Reinforcement learning](../../docs/reinforcement-learning.md#tasks-and-the-dataset)
pages have the rest, and a dataset of your own on each: any class with
`__len__` and `batch`.

| File | What it holds |
|---|---|
| `base.py` | The contracts, `SupervisedDataset` and `RLDataset`: `len(dataset)`, the batches of one pass, and `batch(index)`, the datums or the tasks of a run's step. `batch_positions`, the order both of the cookbook's datasets batch in |
| `conversations.py` | `ChatDataset`: conversations in batches of `batch_size`, each written by the model's message formatter as a datum per assistant turn. `read_conversations`: a JSONL file of conversations, one a line, read as a list. It names a line that holds no conversation |
| `tasks.py` | `TaskDataset`: a list of tasks in batches of `batch_size` |
| `unit_tests/` | The order's batches and passes, each dataset's batches, and a file's lines read and refused |

Rules:

- **The run's step picks the batch.** `batch(index)` depends on its index
  alone. So a loop called again on the same run, after an error or a
  rollback, reads the batch of the step the run stands at.
- **A dataset never ends.** `len(dataset)` is the batches of one pass, and
  any index from 0 up names a batch: past one pass, the next pass's. A
  caller trains several passes by asking the loop for that many steps.
- **A dataset is ready when it is built.** `ChatDataset` writes every
  conversation's datums then, so one that cannot train fails before a run
  is created, and not at its step. `TaskDataset` holds configs and builds
  no session: the loop builds one for each attempt.
- **A loop names its contract alone.** `train_sft.py` and `train_rl.py`
  import `base.py`. A recipe, a tutorial or a test names `ChatDataset` or
  `TaskDataset`.
- **This package is `looper_cookbook.datasets`.** `datasets` alone is
  Hugging Face's library, which the recipes download with. Python started
  in `looper_cookbook/` itself finds this directory under that name, so
  run a recipe from the checkout's root, as its command shows.
- **A new dataset is a module here:** a class with `__len__` and `batch`,
  and its unit tests.
