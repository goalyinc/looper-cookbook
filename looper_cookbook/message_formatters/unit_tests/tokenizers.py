"""Small byte-level Hugging Face tokenizers with each model's framing tokens,
built offline, so the formatters' unit tests check tokens without downloading
a tokenizer.

Every id is a byte or one of the model's own tokens, and two newlines are one
token, as in the models' vocabularies.
"""

from __future__ import annotations

from collections.abc import Sequence

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

QWEN_SPECIALS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
# Tokens of Qwen's vocabularies that are not special: they are text a model reads and writes.
QWEN_ADDED = (
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
)
MUSE_SPECIALS = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|eom|>",
    "<|eot|>",
    "<|finetune_right_pad|>",
    "<|start|>",
    "<|message|>",
)


def byte_level(
    specials: Sequence[str], added: Sequence[str] = (), **tokens: str
) -> PreTrainedTokenizerFast:
    """A tokenizer whose ids are bytes, one for two newlines, ``specials`` and ``added``."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {char: index for index, char in enumerate(alphabet)}
    vocab["ĊĊ"] = len(vocab)
    raw = Tokenizer(models.BPE(vocab=vocab, merges=[("Ċ", "Ċ")]))
    raw.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\n+"), "isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    raw.decoder = decoders.ByteLevel()
    raw.add_special_tokens([AddedToken(token, special=True) for token in specials])
    raw.add_tokens([AddedToken(token, special=False, normalized=False) for token in added])
    return PreTrainedTokenizerFast(tokenizer_object=raw, **tokens)


def qwen() -> PreTrainedTokenizerFast:
    """The framing tokens Qwen3's, Qwen3.5's and Qwen3.8's vocabularies share."""
    return byte_level(QWEN_SPECIALS, QWEN_ADDED, eos_token="<|im_end|>", pad_token="<|endoftext|>")


def muse_glimmer() -> PreTrainedTokenizerFast:
    """Muse Glimmer's framing tokens, with the hub tokenizer's default EOS,
    ``<|end_of_text|>``, which ends no turn."""
    return byte_level(
        MUSE_SPECIALS,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        pad_token="<|finetune_right_pad|>",
    )


def pieces(tokenizer: PreTrainedTokenizerFast, ids: Sequence[int]) -> list[str]:
    """Token ids as text: each of the model's own tokens by itself, and the
    bytes between two of them as one string."""
    own = set(tokenizer.get_added_vocab().values())
    result: list[str] = []
    run: list[int] = []
    for token in [*ids, None]:
        if token is not None and token not in own:
            run.append(token)
            continue
        if run:
            result.append(str(tokenizer.decode(run)))
            run = []
        if token is not None:
            result.append(str(tokenizer.convert_ids_to_tokens(token)))
    return result
