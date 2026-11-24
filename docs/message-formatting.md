# Message formatting

A chat model is trained and sampled in its own message format: the special tokens and layout that turn a list of messages into one token sequence. The cookbook has a message formatter for each model family. A formatter works in both directions: it writes conversations as the prompts a sampler takes and the datums a run trains, and it reads sampled turns back as messages, with their reasoning and their tool calls. Train, sample and evaluate a model through formatters with the same settings, and the model is sampled as it was trained.

## Build your model's formatter

```python
from transformers import AutoTokenizer

from looper_cookbook.message_formatters.models import message_formatter

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
formatter = message_formatter("Qwen/Qwen3.5-9B", tokenizer, enable_thinking=False)
```

`message_formatter(model, tokenizer, *, enable_thinking=True)` builds the formatter of a model Looper serves, named as your run names it. Each family's formatter is a class in `looper_cookbook.message_formatters`, which you can also build directly, with the family's own settings:

| Models | Formatter | Settings |
|---|---|---|
| `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-35B-A3B` | `qwen3_5.Qwen3_5MessageFormatter` | `enable_thinking` |
| `Qwen/Qwen3.8-27B` | `qwen3_8.Qwen3_8MessageFormatter` | `enable_thinking`, `reasoning_effort`, `preserve_thinking` |
| `meta-models/Muse-Glimmer-30B` | `muse_glimmer.MuseGlimmerMessageFormatter` | `reasoning_strength`, `current_date` |
| Qwen3's thinking models, such as `Qwen/Qwen3-8B` | `qwen3.Qwen3MessageFormatter` | `enable_thinking` |

Each formatter is written by hand from its model's chat template and writes the same text. It takes the model's Hugging Face tokenizer for its vocabulary, and refuses another model's.

Every formatter has the same calls:

| Call | What it gives |
|---|---|
| `build_generation_prompt(messages, tools=None)` | Token ids: the conversation, then the opening of the next assistant turn. What the model is given to write that turn |
| `build_supervised_examples(messages, tools=None, *, max_seq_len)` | A datum per assistant turn |
| `parse_response(tokens, tools=None)` | The assistant message a sampled turn holds, with its reasoning and tool calls, and whether the turn ended on its end token |
| `stop_token_ids` | The token a turn ends on, for the sampler to stop on |
| `targets(datum)` | The text a datum trains, to check your data before you train it |

**Messages** are chat messages: a `role` of `system`, `user`, `assistant` or `tool`, and a text `content`. An assistant message may carry `reasoning_content` and `tool_calls`.

## Prompts for a sampler

`build_generation_prompt` writes the conversation so far and the opening of the next assistant turn, as token ids, which is what a sampler takes. Stop the sampler on the formatter's end token:

```python
from looper_sdk import SamplingParams

prompt = formatter.build_generation_prompt([{"role": "user", "content": "What is 6 times 7?"}])
params = SamplingParams(max_tokens=256, stop_token_ids=formatter.stop_token_ids)
print(repr(tokenizer.decode(prompt)))
```

```output
'<|im_start|>user\nWhat is 6 times 7?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

[Sampling](../../server/docs/sampling.md#generate-answers) sends the prompt and its parameters.

## Datums for a run

`build_supervised_examples` makes one datum for each assistant turn of a conversation, in order. A datum is the prompt of its turn, exactly as the model is given it when sampled, and then the turn. The prompt's tokens are context, with weight 0. The turn's are the targets, with weight 1, through the end token, since stopping is part of what the model learns.

```python
conversation = [
    {"role": "user", "content": "Tell me about the red fox."},
    {"role": "assistant", "content": "Habitat: forests, grasslands and cities."},
    {"role": "user", "content": "And the Arctic fox?"},
    {"role": "assistant", "content": "Habitat: the tundra and sea ice."},
]
for datum in formatter.build_supervised_examples(conversation, max_seq_len=2048):
    print(len(datum.tokens), formatter.targets(datum))
```

```output
31 ['Habitat: forests, grasslands and cities.<|im_end|>']
57 ['Habitat: the tundra and sea ice.<|im_end|>']
```

**Why a datum per turn.** A format may write a turn differently once later turns follow it. Qwen3.5 keeps a turn's reasoning block only after the last user message, so an earlier turn loses its block once the conversation goes on. Trained as one sequence, each early turn would follow a prompt the model never meets when it is sampled. A datum per turn trains every turn after exactly its prompt. A conversation of T assistant turns costs T datums, each holding the turns before it as context.

**A file of conversations.** `read_conversations(path)`, in `looper_cookbook.datasets.conversations`, reads a JSONL file with one conversation a line: `{"messages": [...]}`, and optionally `"tools"`.

## Replies read back

`parse_response` reads a sampled turn's token ids as an assistant message, `{"role": "assistant", "content": ...}`, and says whether the turn ended on its end token. A turn that reasoned has its reasoning as `reasoning_content`, apart from its answer. A reply cut off at `max_tokens` did not end, and parses as far as it goes: cut off while it reasoned, it has no content yet.

```python
from looper_cookbook.message_formatters.qwen3_5 import Qwen3_5MessageFormatter

thinking = Qwen3_5MessageFormatter(tokenizer)
reply = tokenizer.encode("6 times 7 is 42.\n</think>\n\n42<|im_end|>", add_special_tokens=False)
print(thinking.parse_response(reply))
print(thinking.parse_response(reply[:4]))
```

```output
({'role': 'assistant', 'reasoning_content': '6 times 7 is 42.', 'content': '42'}, True)
({'role': 'assistant', 'reasoning_content': '6 times 7', 'content': ''}, False)
```

## Tools

Pass the tools a model may call as their JSON schemas, in the `tools` format chat APIs use. The formatter lists them where the model's format does and tells the model how to call one. `parse_response` gives a sampled turn's calls as the message's `tool_calls`, each with its arguments as a mapping. Give it the same tools, and each argument has the type its tool declares. Then continue the prompt with each tool's reply:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    }
]
messages = [{"role": "user", "content": "What is 6 times 7?"}]
prompt = formatter.build_generation_prompt(messages, tools)

# A turn the model sampled after that prompt, written here by hand.
sampled = tokenizer.encode(
    "<tool_call>\n<function=multiply>\n<parameter=a>\n6\n</parameter>\n"
    "<parameter=b>\n7\n</parameter>\n</function>\n</tool_call><|im_end|>",
    add_special_tokens=False,
)
message, ended = formatter.parse_response(sampled, tools)
print(message["tool_calls"])

# The next prompt: every token so far, then the tool's reply.
reply = {"role": "tool", "name": "multiply", "content": "42"}
continued = formatter.continue_prompt([*prompt, *sampled], [reply])
print(repr(tokenizer.decode(continued[len(prompt) + len(sampled) :])))
messages += [message, reply]
```

```output
[{'type': 'function', 'function': {'name': 'multiply', 'arguments': {'a': 6, 'b': 7}}}]
'\n<|im_start|>user\n<tool_response>\n42\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

**In a rollout, continue the prompt; do not format the conversation again.** `continue_prompt(transcript, replies)` keeps the tokens so far exactly as they were sampled and adds only the tools' replies and the opening of the next turn. So an episode is one token sequence, and the log probabilities your sampler recorded stay those of the tokens you train. The cookbook's episode session, [`AgentSession`](episode-sessions.md#run-an-episode), does this for you.

A conversation with tool calls trains the same way: pass its `tools` to `build_supervised_examples`, and a turn that calls a tool is a datum whose targets are its calls.

```python
messages.append({"role": "assistant", "content": "6 times 7 is 42."})
for datum in formatter.build_supervised_examples(messages, tools, max_seq_len=2048):
    print(formatter.targets(datum))
```

```output
['<tool_call>\n<function=multiply>\n<parameter=a>\n6\n</parameter>\n<parameter=b>\n7\n</parameter>\n</function>\n</tool_call><|im_end|>']
['6 times 7 is 42.<|im_end|>']
```

- **A call's arguments** are a mapping. A JSON string, as chat APIs carry them, is decoded.
- **An argument has the type its tool declares.** Qwen3.5, Qwen3.8 and Muse Glimmer write a string as it is and anything else as JSON, so the text `42` could be either. With the tools, `parse_response` reads it as the number or the string its parameter's schema says. Text that does not fit its type stays text.
- **Without the tools, every argument is the text the model wrote.** Nothing is guessed, and the message formats back to exactly the turn that was sampled. Qwen3 writes a call as JSON, which carries its own types.
- **A call that cannot be read stays in the content.** The message then has no `tool_calls`, and nothing the model wrote is lost. Your environment decides what a malformed call is worth.
- **A tool's reply** is a message with the role `tool`. Give it the tool's `name`: Muse Glimmer's format writes it. A `tool_call_id` that matches the `id` of a call names the tool too.
- **Text beside a call** is the message's `content`. Qwen writes it before the calls in the same turn. Muse Glimmer writes it to the user, and the turn goes on with the calls.

## Thinking

Train and sample a model with the same settings.

- **Qwen.** With `enable_thinking=True`, the default, the model is prompted to reason before it answers, and a turn's `reasoning_content` is part of its targets. A turn without reasoning trains an empty reasoning block. With `enable_thinking=False` the prompt closes an empty block itself, and the turn's answer follows it. Train conversations without reasoning this way, and sample the result the same way.
- **Qwen3.8** also takes `reasoning_effort`: `"xhigh"`, the default, `"medium"` or `"low"`. `xhigh` and `low` each put an instruction in the system turn. It keeps every earlier turn's reasoning in the prompt unless `preserve_thinking=False`.
- **Muse Glimmer** has no switch that turns reasoning off. `reasoning_strength` is `"low"`, `"medium"`, `"high"`, the default, or `"xhigh"`, and the system turn states it. Without a system message the format writes a default one that states the current date. Name a `current_date` to format the same tokens on any day.

**Qwen3.8 at low effort** puts its instruction in a system turn of its own, before the question:

```python
from looper_cookbook.message_formatters.qwen3_8 import Qwen3_8MessageFormatter

question = [{"role": "user", "content": "What is 6 times 7?"}]
qwen3_8_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
qwen3_8 = Qwen3_8MessageFormatter(qwen3_8_tokenizer, reasoning_effort="low")
print(repr(qwen3_8_tokenizer.decode(qwen3_8.build_generation_prompt(question))))
```

```output
'<|im_start|>system\nReasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.<|im_end|>\n<|im_start|>user\nWhat is 6 times 7?<|im_end|>\n<|im_start|>assistant\n<think>\n'
```

**Muse Glimmer** starts every conversation with a system turn, and writes each part of an assistant turn to a recipient. A turn's reasoning goes `to=self` before its answer goes `to=user`, and the turn ends on `<|eot|>`, which is what its formatter stops on:

```python
from looper_cookbook.message_formatters.muse_glimmer import MuseGlimmerMessageFormatter

muse_tokenizer = AutoTokenizer.from_pretrained("meta-models/Muse-Glimmer-30B")
muse = MuseGlimmerMessageFormatter(muse_tokenizer, current_date="2026-11-18")
print(repr(muse_tokenizer.decode(muse.build_generation_prompt(question))))

reasoned = [*question, {"role": "assistant", "reasoning_content": "6 * 7 = 42.", "content": "42"}]
[datum] = muse.build_supervised_examples(reasoned, max_seq_len=2048)
print(muse.targets(datum))
print("a turn ends on", muse_tokenizer.decode(muse.stop_token_ids))
```

```output
'<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\nCurrent date: 2026-11-18.\n\nReasoning strength: high.\n\n# Valid recipients: "self", "user".<|eot|><|start|>user<|message|>What is 6 times 7?<|eot|><|start|>assistant'
[' to=self<|message|>6 * 7 = 42.<|eom|><|start|>assistant to=user<|message|>42<|eot|>']
a turn ends on <|eot|>
```

A Muse Glimmer turn that calls a tool writes the call to the tool's name. Text beside the call goes to the user first, and the turn goes on:

```python
called = {
    "role": "assistant",
    "content": "Let me work it out.",
    "tool_calls": [{"type": "function", "function": {"name": "multiply", "arguments": {"a": 6, "b": 7}}}],
}
[datum] = muse.build_supervised_examples([*question, called], tools, max_seq_len=2048)
print(muse.targets(datum))
```

```output
[' to=user<|message|>Let me work it out.<|eom|><|start|>assistant to=multiply<|message|><atem:function_calls>\n<atem:invoke name="multiply">\n<atem:parameter name="a">6</atem:parameter>\n<atem:parameter name="b">7</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>']
```

## What is refused

A formatter fails a conversation, naming the turn and the fix, rather than build a prompt or a datum the model's format cannot hold:

- **A turn with reasoning, through a Qwen formatter with thinking off.** Its prompt tells the model not to reason. Build the formatter with thinking on, or drop the turn's reasoning.
- **A turn that starts past `max_seq_len`.** A turn that only ends past it is cut there: a datum keeps its head, so what the model conditions on never changes.
- **A conversation with no assistant turn,** since nothing in it trains, and one with no user message before a Qwen model's turn.
- **Content that is not text.** The formatters write text alone.
- **A system message after the first,** for Qwen3.5 and Qwen3.8.
- **A tool's reply that does not name its tool,** for Muse Glimmer, and a message that carries Muse Glimmer's own `recipient` or `end_turn` fields.
- **A tokenizer of another model,** when the formatter is built.
- **Anything but a tool's reply, given to `continue_prompt`.** After a new user message, Qwen3 and Qwen3.5 write the turns before it without their reasoning, so the sampled tokens could not stay. Start a new prompt with `build_generation_prompt` instead.

## Next steps

- [Chat data](../tutorials/201_chat_data.ipynb) walks through a formatter on your machine, with no key.
- [The cookbook](index.md): the loops and recipes that use it.
