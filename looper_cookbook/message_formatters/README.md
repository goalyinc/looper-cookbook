# message_formatters/

The message formatters: each model family's conversations
as its tokens, and its sampled tokens back as messages. A formatter is
written by hand from its model's chat template, so it can write a turn
apart from its prompt and read a sampled turn's reasoning and tool calls
back.

## Usage

Build the formatter of your run's model, over the model's tokenizer:

```python
from transformers import AutoTokenizer

from looper_cookbook.message_formatters.models import message_formatter

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
formatter = message_formatter("Qwen/Qwen3.5-9B", tokenizer, enable_thinking=False)
```

**SFT data.** A conversation becomes a datum per assistant turn: the turn's
prompt as context, and the turn as targets.

```python
conversation = [
    {"role": "user", "content": "What is 6 times 7?"},
    {"role": "assistant", "content": "42"},
]
datums = formatter.build_supervised_examples(conversation, max_seq_len=2048)
formatter.targets(datums[0])  # ['42<|im_end|>']
```

**RL data.** A prompt for the sampler, the token to stop on, and a sampled
turn read back as a message to score:

```python
from looper_sdk import SamplingParams

prompt = formatter.build_generation_prompt(conversation[:1])
params = SamplingParams(max_tokens=256, temperature=1.0, stop_token_ids=formatter.stop_token_ids)
# sampled = the token ids of one of the sampler's sequences
message, ended = formatter.parse_response(sampled)
# ({'role': 'assistant', 'content': '6 times 7 is 42.'}, True)
```

The [Message formatting](../../docs/message-formatting.md) page has the
rest, each with its output:
[each family's own class and settings](../../docs/message-formatting.md#build-your-models-formatter),
[tool calls](../../docs/message-formatting.md#tools),
[thinking](../../docs/message-formatting.md#thinking), and what a formatter
refuses.
`tutorials/201_chat_data.ipynb` walks through them.

| File | What it holds |
|---|---|
| `base.py` | `MessageFormatter`, the contract: `build_generation_prompt`, `build_supervised_examples` with a datum per assistant turn, `parse_response`, `stop_token_ids`, `continue_prompt`, which adds tool replies after an episode's tokens, and `targets`. A subclass writes text, `format_prompt`, `format_response`, `parse_text` and `format_continuation`, and the base tokenizes it. Also what the formats share: reading a message's content and a tool call's arguments, and reading an argument's text as the type its tool's schema declares |
| `qwen3.py` | `Qwen3MessageFormatter`: ChatML turns, `<think>` blocks, tool calls as JSON, `enable_thinking` |
| `qwen3_5.py` | `Qwen3_5MessageFormatter`, over Qwen3's: trimmed content, a `<think>` block the prompt opens, tool calls as `<function=…>` and `<parameter=…>` blocks |
| `qwen3_8.py` | `Qwen3_8MessageFormatter`, over Qwen3.5's: `reasoning_effort`, reasoning kept in history with `preserve_thinking`, arguments as JSON |
| `muse_glimmer.py` | `MuseGlimmerMessageFormatter`: the ATEM protocol, with `to=self` reasoning, `to=<tool>` calls and `<|eot|>` ending a turn; `reasoning_strength` and `current_date` |
| `models.py` | `MESSAGE_FORMATTERS`, the formatter of each model Looper serves, and `message_formatter(model, tokenizer)`, which a recipe calls with its `--model` |
| `unit_tests/` | Each format's tokens in each case, over small byte-level tokenizers built offline (`tokenizers.py`). They use no chat template |

## Rules


- **A formatter writes what its model's chat template writes.** Its module's
  docstring names the template it was written from, at a revision.
  `python -m looper_cookbook.tests.message_formatters FAMILY TOKENIZER` checks
  the formatter against that template with the model's real tokenizer. Run
  it when a formatter or a model's template changes, and say so in the PR.
- **A subclass writes text and never tokenizes.** The base tokenizes a
  prompt whole, as a chat template's text is tokenized, and a turn apart
  from its prompt, so a datum starts with exactly the tokens the model is
  sampled after.
- **A formatter's settings are its template's,** under the template's names:
  `enable_thinking`, `reasoning_effort`. One formatter formats every prompt
  and datum the same way.
- **What a format cannot hold is refused,** with the turn and the fix named.
  `parse_text` never raises on what a model wrote: a call it cannot read
  stays in the content.
- **A sampled turn is never written again.** `continue_prompt` keeps an
  episode's tokens and adds what `format_continuation` writes after them:
  the tool replies and the next turn's opening, as the template writes
  them after that turn. It takes tool replies alone, since a new user
  message can change how a format writes the turns before it.
- **What `parse_response` reads, `format_response` writes back the same.**
  Without the tools an argument is the text the model wrote; with them it
  has the type its tool declares, and nothing is guessed.
- **A new model** of a family here is a line in `models.py`. A new family is
  a module here, its unit tests, its entry in the check's `FAMILIES`, and
  its row on the Message formatting page.
