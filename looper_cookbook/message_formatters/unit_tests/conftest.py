"""The tokenizers the formatters' unit tests share."""

from __future__ import annotations

import pytest
from transformers import PreTrainedTokenizerFast

from looper_cookbook.message_formatters.unit_tests import tokenizers


@pytest.fixture(scope="session")
def qwen_tokenizer() -> PreTrainedTokenizerFast:
    return tokenizers.qwen()


@pytest.fixture(scope="session")
def muse_tokenizer() -> PreTrainedTokenizerFast:
    return tokenizers.muse_glimmer()
