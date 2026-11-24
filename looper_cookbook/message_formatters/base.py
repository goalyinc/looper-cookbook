"""The message formatter's contract, and what every model's formatter shares.

A ``MessageFormatter`` converts between one model's conversations and its
tokens, in both directions. Each model family has its own, in a module of
this package, written by hand from the model's chat template. Training,
sampling and evaluating through one formatter prompts the model as it was
trained.

Its methods:

- ``build_generation_prompt``: a conversation, then the opening of the next
  assistant turn. What the model is given to write that turn.
- ``build_supervised_examples``: a datum per assistant turn.
- ``parse_response``: a sampled turn read back as a message, with its
  reasoning and its tool calls.
- ``stop_token_ids``: the token a turn ends on, for ``SamplingParams``.
- ``continue_prompt``: an episode's next prompt, the tokens so far and then
  the tool replies a sampled turn led to.
- ``targets``: the text a datum trains, to check data before training it.

**Messages** are chat messages, as mappings: a ``role``, a text
``content``, and on an assistant message optionally ``reasoning_content``
and ``tool_calls``, each ``{"type": "function", "function": {"name": ...,
"arguments": {...}}}``. A tool's reply has the role ``tool``. **Tools** are
their JSON schemas, ``{"type": "function", "function": {...}}``.

**A model's module writes text; this class makes it tokens.** A subclass
writes four things: the prompt of the next assistant turn
(``format_prompt``), an assistant message as the model writes it after that
prompt (``format_response``), the reverse of the second (``parse_text``),
and what follows a sampled turn when tools reply to it
(``format_continuation``). This class tokenizes the prompt as the model's
own template is tokenized, whole and with its special tokens as single ids.

**An episode's tokens are only appended to**. A turn a model sampled
is never decoded and written again: ``continue_prompt`` keeps the tokens so
far and adds the new messages after them. So the log probabilities a
sampler recorded stay those of the tokens a datum trains.

**A datum per assistant turn**. A datum is the prompt of its turn,
then the turn: its reasoning, its content, its tool calls and the token
that ends it, since stopping is part of what SFT teaches. The prompt's
tokens are context, attended and never scored, and the turn's are the
targets. The two are tokenized apart, so a datum always starts with exactly
the tokens the model is sampled after. A model's format may write a turn
differently once later turns follow: Qwen's write ``<think>`` blocks only
after the last user message. A datum per turn trains every turn after the
prompt it is sampled with, under any format. A conversation of T assistant
turns costs T datums, each holding the turns before it as context.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from looper_sdk import Datum

__all__ = ["Message", "MessageFormatter", "Tool"]

Message = Mapping[str, Any]
Tool = Mapping[str, Any]


class MessageFormatter(abc.ABC):
    """One model family's message format, in both directions.

    A subclass names the tokens its format is written with and implements
    ``format_prompt``, ``format_response``, ``parse_text`` and
    ``format_continuation``.

    Attributes:
        end_of_turn: The token that ends an assistant turn: what a datum's
            targets end with and what sampling stops on.
        special_tokens: The tokens the format frames its turns with. Each
            must be one id of the tokenizer, or the tokenizer is another
            model's.

    Args:
        tokenizer: The model's Hugging Face tokenizer, at the revision the
            recipe pins. Only its vocabulary is used, never its chat
            template.
    """

    end_of_turn: ClassVar[str]
    special_tokens: ClassVar[tuple[str, ...]]

    def __init__(self, tokenizer: Any) -> None:
        for token in (*self.special_tokens, self.end_of_turn):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == tokenizer.unk_token_id:
                raise ValueError(
                    f"{type(self).__name__} writes {token} as one token, and the tokenizer "
                    "has no such token: it is another model's. Build the formatter of the "
                    "model the tokenizer belongs to"
                )
        self.tokenizer = tokenizer
        self.end_of_turn_id: int = tokenizer.convert_tokens_to_ids(self.end_of_turn)

    @abc.abstractmethod
    def format_prompt(
        self, messages: Sequence[Message], tools: Sequence[Tool] | None = None
    ) -> str:
        """The conversation, then the opening of the next assistant turn, as text.

        It is the model's chat template with its generation prompt, written
        out. A conversation the format cannot hold raises ``ValueError``.
        """

    @abc.abstractmethod
    def format_response(self, message: Message) -> str:
        """An assistant message as the model writes it after ``format_prompt``:
        its reasoning, content and tool calls, through ``end_of_turn``.

        A message the format cannot write after its prompt raises
        ``ValueError``, as one with reasoning does when thinking is off.
        """

    @abc.abstractmethod
    def parse_text(self, text: str, tools: Sequence[Tool] | None = None) -> dict[str, Any]:
        """The assistant message that the text of a sampled turn holds, its end
        token removed. The reverse of ``format_response``.

        It never raises on what a model wrote: a tool call it cannot read
        stays in the message's content, and the message has no
        ``tool_calls``. ``tools`` give each call's arguments their types
        (``argument_value``).
        """

    @abc.abstractmethod
    def format_continuation(self, messages: Sequence[Message]) -> str:
        """What follows a sampled assistant turn's end token when tools reply
        to it: ``messages``, then the opening of the next assistant turn, as
        text.

        It is what the model's chat template writes after that turn, so the
        tokens before it stay as they were sampled. Anything but a tool's
        reply raises ``ValueError``.
        """

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        """The token an assistant turn ends on: what sampling stops on."""
        return (self.end_of_turn_id,)

    def build_generation_prompt(
        self, messages: Sequence[Message], tools: Sequence[Tool] | None = None
    ) -> list[int]:
        """The conversation, then the opening of the next assistant turn, as
        token ids: what the model is given to write that turn.

        Args:
            messages: The conversation so far.
            tools: The JSON schemas of the tools the model may call.
        """
        return self._encode(self.format_prompt(messages, tools))

    def build_supervised_examples(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
        *,
        max_seq_len: int,
    ) -> list[Datum]:
        """One datum per assistant turn of the conversation, in order.

        Args:
            messages: The conversation.
            tools: The JSON schemas of the tools the conversation calls.
            max_seq_len: A datum's longest length. A longer one keeps its head,
                so what the model conditions on never changes; a turn whose
                prompt alone reaches it is refused.
        """
        messages = list(messages)
        turns = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
        if not turns:
            raise ValueError(
                f"the conversation starting {_opening(messages)!r} has no assistant message, "
                "so nothing in it trains"
            )
        return [self._turn_datum(messages, index, tools, max_seq_len) for index in turns]

    def parse_response(
        self, tokens: Sequence[int], tools: Sequence[Tool] | None = None
    ) -> tuple[dict[str, Any], bool]:
        """A sampled assistant turn as a message, and whether it ended on its
        end token.

        The message is ``{"role": "assistant", "content": ...}``, with the
        turn's ``reasoning_content`` and ``tool_calls`` when it has them. A
        turn cut off before its end token, at ``max_tokens``, parses as far
        as it goes.

        Args:
            tokens: The turn's token ids, as the sampler returned them: what
                follows the generation prompt.
            tools: The JSON schemas the prompt listed. A call's arguments
                are read as the types their tool declares; without them
                every argument is the text the model wrote.
        """
        tokens = list(tokens)
        ended = self.end_of_turn_id in tokens
        if ended:
            tokens = tokens[: tokens.index(self.end_of_turn_id)]
        return self.parse_text(self.tokenizer.decode(tokens), tools), ended

    def continue_prompt(
        self, transcript: Sequence[int], messages: Sequence[Message]
    ) -> list[int]:
        """An episode's next prompt, as token ids: ``transcript``, then
        ``messages`` and the opening of the next assistant turn.

        The transcript's tokens are kept as they are. A transcript that does
        not end on the end token gets it first, as context: a sampler may
        leave its stop token out of what it returns, and a model may end a
        turn with its end of text.

        Args:
            transcript: The last prompt, then the turn sampled after it.
            messages: The replies of the tools the turn called, in the
                calls' order.
        """
        tokens = list(transcript)
        if not tokens:
            raise ValueError(
                "the transcript is empty: a prompt is continued after a sampled turn, and "
                "build_generation_prompt writes an episode's first"
            )
        if tokens[-1] != self.end_of_turn_id:
            tokens.append(self.end_of_turn_id)
        return tokens + self._encode(self.format_continuation(messages))

    def targets(self, datum: Datum) -> list[str]:
        """The text of each run of the datum's targets, in order: what it trains."""
        runs: list[list[int]] = []
        previous = 0.0
        for token, weight in zip(datum.tokens, datum.weights, strict=True):
            if weight and not previous:
                runs.append([])
            if weight:
                runs[-1].append(token)
            previous = weight
        return [self.tokenizer.decode(run) for run in runs]

    def _turn_datum(
        self, messages: list[Message], index: int, tools: Sequence[Tool] | None, max_seq_len: int
    ) -> Datum:
        """The assistant turn ``messages[index]`` after the messages before it."""
        try:
            # What the model is given to write the turn, then the turn as it writes it.
            prompt = self._encode(self.format_prompt(messages[:index], tools))
            response = self._encode(self.format_response(messages[index]))
        except ValueError as error:
            raise ValueError(
                f"messages[{index}] of the conversation starting {_opening(messages)!r}: {error}"
            ) from error
        if len(prompt) >= max_seq_len:
            raise ValueError(
                f"messages[{index}]: this assistant turn starts at token {len(prompt)}, past "
                f"max_seq_len={max_seq_len}; raise max_seq_len or shorten the conversation "
                f"starting {_opening(messages)!r}"
            )
        tokens = (prompt + response)[:max_seq_len]
        return Datum(tokens=tokens, weights=[0] * len(prompt) + [1] * (len(tokens) - len(prompt)))

    def _encode(self, text: str) -> list[int]:
        # As a chat template's text is tokenized: it writes its own special tokens.
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])


def tool_replies(messages: Sequence[Message]) -> list[Message]:
    """The messages of a continuation, each a tool's reply; anything else,
    or none at all, is refused."""
    replies = list(messages)
    if not replies:
        raise ValueError(
            "a prompt is continued with at least one tool reply: without one the model "
            "would be asked for a second turn after nothing"
        )
    for message in replies:
        if message["role"] != "tool":
            raise ValueError(
                f"a prompt is continued with tool replies, and this message's role is "
                f"{message['role']!r}. A format may write the turns before a new user "
                "message differently, so the tokens sampled so far could not stay: "
                "start a new prompt with build_generation_prompt, or give the message "
                "as a tool's reply"
            )
    return replies


def text_content(message: Message) -> str:
    """A message's content as text; a message without content has none."""
    content = message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError(
            f"a {message['role']} message's content is {type(content).__name__}, not text: "
            "the formatters write text alone, so join its parts and leave images out"
        )
    return content


def reasoning_content(message: Message) -> str:
    """An assistant message's reasoning; a message without it has none."""
    reasoning = message.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else ""


def call_arguments(call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """A tool call's name, and its arguments as a mapping.

    Chat APIs carry the arguments as a JSON string, and datasets carry
    either, so a string is decoded.
    """
    function = call.get("function", call)
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"the arguments of the call to {function['name']} are not JSON: {error}"
            ) from None
    if not isinstance(arguments, Mapping):
        raise ValueError(f"the arguments of the call to {function['name']} are not an object")
    return function["name"], dict(arguments)


def tool_call(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """A parsed tool call, as an assistant message carries it."""
    return {"type": "function", "function": {"name": name, "arguments": dict(arguments)}}


def parameter_schemas(tools: Sequence[Tool] | None, name: str) -> Mapping[str, Any]:
    """The JSON schema of each parameter of the tool ``name``: none when the
    tool is not among ``tools``."""
    for tool in tools or ():
        function = tool.get("function", tool)
        if function.get("name") == name:
            return (function.get("parameters") or {}).get("properties") or {}
    return {}


def argument_value(text: str, schema: Mapping[str, Any] | None) -> Any:
    """A tool call's argument, read from the text a model wrote for it.

    Qwen3.5's, Qwen3.8's and Muse Glimmer's formats write a string as it is
    and anything else as JSON, without saying which, so the text alone cannot
    tell the string ``42`` from the number. The parameter's JSON schema does:
    the text is read as a type the schema declares, and stays text when it
    fits none. Without a schema it stays text, exactly as the model wrote it,
    as the sampler's parser for Qwen's format reads an argument.
    """
    types = _schema_types(schema or {})
    # Qwen3.5 writes Python's None, True and False where the others write JSON's.
    if "null" in types and text in ("null", "None"):
        return None
    if "string" in types:
        return text
    if "boolean" in types and text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    fits = (
        ("integer" in types and type(value) is int)
        or ("number" in types and type(value) in (int, float))
        or ("object" in types and isinstance(value, dict))
        or ("array" in types and isinstance(value, list))
    )
    return value if fits else text


def _schema_types(schema: Mapping[str, Any]) -> set[str]:
    """The JSON types a parameter's schema allows; none when it names none."""
    declared = schema.get("type")
    if isinstance(declared, str):
        return {declared}
    if isinstance(declared, list):
        return set(declared)
    options = schema.get("anyOf") or schema.get("oneOf") or ()
    return {kind for option in options for kind in _schema_types(option)}


def _opening(messages: Sequence[Message]) -> str:
    """The start of a conversation's first user message, to name it in errors."""
    for message in messages:
        if message["role"] == "user" and isinstance(message.get("content"), str):
            return message["content"][:60]
    return str(messages[0].get("content"))[:60] if messages else ""
