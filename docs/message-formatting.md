# Message formatting

Looper trains and samples token ids, not text. A chat model also expects every conversation in its own format, with special tokens that mark where each message starts and ends, and who wrote it. For example, Qwen3.5 sees the question "What is 6 times 7?" as `<|im_start|>user\nWhat is 6 times 7?<|im_end|>`, followed by the start of its own reply.

A **message formatter** is the cookbook code that converts between your conversations and the model's tokens. The cookbook has one for each model family, and it works in both directions:

- **Messages to tokens:** it turns a conversation into a prompt for the sampler, or into datums for a training run.
- **Tokens to messages:** it reads the tokens the model sampled back as a message, with its reasoning and its tool calls separated out.

Use a formatter with the same settings to train, sample and evaluate a model. The model is then sampled in exactly the format it was trained in.

With the cookbook, you build the formatter once and pass it to `ChatDataset` for SFT, or to the RL loops. The cookbook then calls its methods for you. You call them yourself when you write your own loop, or want to check your data.

This page covers:

- [Build your model's formatter](#build-your-models-formatter): create the formatter for your model, and the calls every formatter has.
- [Prompts for a sampler](#prompts-for-a-sampler): turn a conversation into a prompt the model answers.
- [Datums for a run](#datums-for-a-run): turn conversations into training data.
- [Replies read back](#replies-read-back): turn sampled tokens back into a message.
- [Tools](#tools): let the model call tools, and train on conversations that use them.
- [Thinking](#thinking): control whether and how much the model reasons before it answers.
- [What is refused](#what-is-refused): conversations a formatter won't format, and why.

Words used on this page:

| Word | Meaning |
|---|---|
| **Message** | One entry in a conversation, such as the user's question or the model's reply. |
| **Turn** | One message from the assistant, which is what the model writes in one go. |
| **Prompt** | The token ids the sampler is given. They hold the conversation so far, then the start of the next assistant turn. |
| **Datum** | One training example, made of token ids plus a weight for each. See [Prepare data](../../server/docs/training.md#prepare-data). |
| **Target** | A token the model is trained to write, with weight 1. The rest of a datum is context, with weight 0. |
| **End token** | The special token that ends a turn, such as `<|im_end|>`. The sampler stops on it. |
| **Reasoning** | The thinking a reasoning model writes before its answer. It's kept apart from the answer as `reasoning_content`. |

## Build your model's formatter

Build a formatter with `message_formatter`, from the model's name and its Hugging Face tokenizer. The tokenizer runs on your machine, and only supplies the vocabulary, so no model weights are loaded.

```python
from transformers import AutoTokenizer

from looper_cookbook.message_formatters.models import message_formatter

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
formatter = message_formatter("Qwen/Qwen3.5-9B", tokenizer, enable_thinking=False)
```

`message_formatter(model, tokenizer, *, enable_thinking=True)` builds the formatter for a model Looper serves. Name the model as your training run names it. `enable_thinking` sets whether a Qwen model reasons before it answers (see [Thinking](#thinking)).

Each model family's formatter is also a class in `looper_cookbook.message_formatters`. You can build it directly when you need the family's other settings:

| Models | Formatter | Settings |
|---|---|---|
| `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-35B-A3B` | `qwen3_5.Qwen3_5MessageFormatter` | `enable_thinking` |
| `Qwen/Qwen3.8-27B` | `qwen3_8.Qwen3_8MessageFormatter` | `enable_thinking`, `reasoning_effort`, `preserve_thinking` |
| `meta-models/Muse-Glimmer-30B` | `muse_glimmer.MuseGlimmerMessageFormatter` | `reasoning_strength`, `current_date` |
| Qwen3's thinking models, such as `Qwen/Qwen3-8B` | `qwen3.Qwen3MessageFormatter` | `enable_thinking` |

Each formatter is written by hand from its model's chat template, and writes the same text the template does. It takes the model's own tokenizer, and refuses another model's.

Every formatter has the same calls:

| Call | What it gives you |
|---|---|
| `build_generation_prompt(messages, tools=None)` | A prompt, made of the conversation as token ids, then the start of the next assistant turn. The model writes that turn. |
| `build_supervised_examples(messages, tools=None, *, max_seq_len)` | One datum for each assistant turn in the conversation. |
| `parse_response(tokens, tools=None)` | The assistant message in a sampled turn, with its reasoning and tool calls, and whether the turn ended on its end token. |
| `stop_token_ids` | The end token, for the sampler to stop on. |
| `targets(datum)` | The text a datum trains on, so you can check your data before you train. |

**Messages** are chat messages. Each has a `role` of `system`, `user`, `assistant` or `tool`, and a text `content`. An assistant message can also have `reasoning_content` and `tool_calls`.

## Prompts for a sampler

To have the model answer a conversation, turn it into a prompt with `build_generation_prompt`. The prompt holds the conversation so far, then the start of the next assistant turn, so the model's next tokens are its reply. Tell the sampler to stop on the formatter's end token, so it stops when the reply ends:

```python
from looper_sdk import SamplingParams

prompt = formatter.build_generation_prompt([{"role": "user", "content": "What is 6 times 7?"}])
params = SamplingParams(max_tokens=256, stop_token_ids=formatter.stop_token_ids)
print(repr(tokenizer.decode(prompt)))
```

```output
'<|im_start|>user\nWhat is 6 times 7?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

The prompt ends with `<|im_start|>assistant`, where the model's reply begins. The empty `<think>` block is there because this formatter was built with thinking off.

[Sampling](../../server/docs/sampling.md#generate-answers) shows how to send the prompt and its parameters to a sampler.

## Datums for a run

To train on a conversation, turn it into datums with `build_supervised_examples`. You get one datum for each assistant turn, in order. Each datum is that turn's prompt, exactly as the model would see it when sampled, followed by the turn itself:

- **The prompt's tokens are context,** with weight 0. The model reads them, but isn't trained to write them.
- **The turn's tokens are the targets,** with weight 1, up to and including the end token. The model also learns when to stop.

This conversation about foxes has two assistant turns, so it gives two datums:

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

The second datum is longer, because its context includes the first exchange. `targets` shows that each datum trains only its own turn.

**Why one datum per turn.** Some formats write a turn differently once later turns follow it. For example, Qwen3.5 keeps a turn's reasoning only after the last user message, so an earlier turn loses its reasoning once the conversation goes on. If the whole conversation were trained as one sequence, each early turn would follow a prompt the model never sees when it's sampled. One datum per turn trains each turn after exactly the prompt it will be sampled with. The cost is that a conversation with T assistant turns gives T datums, each holding the turns before it as context.

**A file of conversations.** `read_conversations(path)`, in `looper_cookbook.datasets.conversations`, reads a JSONL file with one conversation per line, as `{"messages": [...]}`, with optional `"tools"`. [Supervised learning](supervised-learning.md) trains on such a file.

## Replies read back

When the sampler returns a turn, it's token ids. `parse_response` reads them back as an assistant message, `{"role": "assistant", "content": ...}`, and also says whether the turn ended on its end token.

- **Reasoning is kept apart from the answer.** A turn that reasoned has its reasoning as `reasoning_content`, and its answer as `content`.
- **A reply cut off at `max_tokens` didn't end.** It's read as far as it goes. A reply cut off while it was still reasoning has no `content` yet.

This reads one full reply, and the same reply cut off after 4 tokens:

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

A model can call tools, such as a calculator, while it answers. You describe the tools as JSON schemas, in the `tools` format that chat APIs use, and pass them to the formatter. The formatter lists them in the prompt where the model's format expects them, and tells the model how to call one.

When the model calls a tool, `parse_response` returns the calls as the message's `tool_calls`, each with its arguments as a mapping. Your code runs the tool, and then continues the prompt with the tool's reply, so the model can go on.

This example gives the model a `multiply` tool, reads a call the model made, and adds the tool's reply to the prompt:

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

**In a rollout, continue the prompt. Don't format the whole conversation again.** `continue_prompt(transcript, replies)` keeps the tokens so far exactly as they were sampled, and adds only the tools' replies and the start of the next turn. So an attempt stays one token sequence, and the log probabilities the sampler recorded still belong to the tokens you train on. The cookbook's episode session, [`AgentSession`](episode-sessions.md#run-an-episode), does this for you.

A conversation with tool calls trains the same way as any other. Pass its `tools` to `build_supervised_examples`, and a turn that calls a tool becomes a datum whose targets are its calls:

```python
messages.append({"role": "assistant", "content": "6 times 7 is 42."})
for datum in formatter.build_supervised_examples(messages, tools, max_seq_len=2048):
    print(formatter.targets(datum))
```

```output
['<tool_call>\n<function=multiply>\n<parameter=a>\n6\n</parameter>\n<parameter=b>\n7\n</parameter>\n</function>\n</tool_call><|im_end|>']
['6 times 7 is 42.<|im_end|>']
```

Things to know:

- **A call's arguments are a mapping.** A JSON string, as chat APIs send them, is decoded.
- **Each argument has the type its tool declares.** Qwen3.5, Qwen3.8 and Muse Glimmer write a string as it is and anything else as JSON, so the text `42` could be either a number or a string. Given the tools, `parse_response` reads it as the type the parameter's schema says. Text that doesn't fit its type stays text.
- **Without the tools, every argument is the text the model wrote.** Nothing is guessed, and the message formats back to exactly the turn that was sampled. Qwen3 writes a call as JSON, which carries its own types.
- **A call that can't be read stays in the content.** The message then has no `tool_calls`, and nothing the model wrote is lost. Your environment decides what a broken call is worth.
- **A tool's reply is a message with the role `tool`.** Give it the tool's `name`, because Muse Glimmer's format writes it. A `tool_call_id` that matches a call's `id` also names the tool.
- **Text next to a call is the message's `content`.** Qwen writes it before the calls in the same turn. Muse Glimmer writes it to the user, and the turn goes on with the calls.

## Thinking

Reasoning models can think before they answer. Each family has its own settings for this. Whatever you choose, train and sample the model with the same settings.

- **Qwen.** With `enable_thinking=True`, the default, the model is prompted to reason before it answers, and a turn's `reasoning_content` is part of what it's trained on. A turn without reasoning trains an empty reasoning block. With `enable_thinking=False`, the prompt closes an empty reasoning block itself, and the answer follows it. Use this for conversations without reasoning, and sample the trained model the same way.
- **Qwen3.8** also takes `reasoning_effort`, which can be `"xhigh"` (the default), `"medium"` or `"low"`. `xhigh` and `low` each add an instruction to the system turn. It keeps every earlier turn's reasoning in the prompt, unless you set `preserve_thinking=False`.
- **Muse Glimmer** can't turn reasoning off. `reasoning_strength` can be `"low"`, `"medium"`, `"high"` (the default) or `"xhigh"`, and the system turn states it. Without a system message, the format writes a default one that includes the current date. Set `current_date` to get the same tokens on any day.

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

**Muse Glimmer** starts every conversation with a system turn. It also addresses each part of an assistant turn to a recipient. The reasoning goes `to=self`, then the answer goes `to=user`. The turn ends on `<|eot|>`, which is what its formatter stops on:

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

When a Muse Glimmer turn calls a tool, it addresses the call to the tool's name. Any text next to the call goes to the user first, and the turn goes on:

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

Some conversations can't be written in a model's format. Rather than build a wrong prompt or datum, the formatter raises an error that names the turn and says how to fix it:

- **A turn with reasoning, through a Qwen formatter with thinking off.** The prompt tells the model not to reason. Build the formatter with thinking on, or remove the turn's reasoning.
- **A turn that starts past `max_seq_len`.** A turn that only ends past it is cut off there instead. The datum keeps the start of the turn, so the context the model learns from stays the same.
- **A conversation with no assistant turn,** since there's nothing in it to train. Also, for Qwen models, an assistant turn with no user message before it.
- **Content that isn't text.** The formatters only write text.
- **A second system message,** for Qwen3.5 and Qwen3.8.
- **A tool's reply that doesn't name its tool,** for Muse Glimmer. Also a message with Muse Glimmer's own `recipient` or `end_turn` fields.
- **A tokenizer from another model,** when you build the formatter.
- **Anything other than a tool's reply, passed to `continue_prompt`.** After a new user message, Qwen3 and Qwen3.5 write the earlier turns without their reasoning, so the sampled tokens couldn't stay as they are. Start a new prompt with `build_generation_prompt` instead.

## Next steps

- [Chat data](../tutorials/201_chat_data.ipynb) walks through a formatter on your machine, with no API key.
- [Supervised learning](supervised-learning.md) trains a model on a file of conversations.
- [The cookbook](index.md) lists the loops and recipes that use a formatter.
