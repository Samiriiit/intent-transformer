"""Tests for the whitespace tokenizer.

Run from the project directory:

    pytest tests/test_tokenizer.py
"""

import sys
from pathlib import Path

# The learning project is not an installed package. Put src/ on the path so
# `import tokenizer` works under pytest without a setup file.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from tokenizer import (  # noqa: E402
    build_vocab,
    clean_text,
    decode,
    encode,
    load_vocab,
    save_vocab,
    tokenize,
)


def test_lowercase_conversion():
    assert clean_text("Show me Nike Shoes") == "show me nike shoes"


def test_punctuation_removal_preserves_numbers():
    assert clean_text("Show me Nike Shoes under 5000!!!") == "show me nike shoes under 5000"


def test_punctuation_between_words_does_not_glue_them():
    assert clean_text("shoes,under") == "shoes under"


def test_whitespace_is_normalized():
    assert clean_text("  show   me\tnike  ") == "show me nike"


def test_tokenize_splits_on_whitespace():
    assert tokenize("show me nike shoes under 5000") == [
        "show",
        "me",
        "nike",
        "shoes",
        "under",
        "5000",
    ]


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_build_vocab_reserves_pad_and_unk_then_first_seen_order():
    vocab = build_vocab(["show me nike shoes"])
    assert vocab["<PAD>"] == 0
    assert vocab["<UNK>"] == 1
    assert vocab["show"] == 2
    assert vocab["me"] == 3
    assert vocab["nike"] == 4
    assert vocab["shoes"] == 5


def test_build_vocab_min_frequency_drops_rare_tokens():
    dataset = ["nike shoes", "nike socks", "rareword shoes"]
    vocab = build_vocab(dataset, min_frequency=2)
    assert "nike" in vocab
    assert "shoes" in vocab
    assert "socks" not in vocab
    assert "rareword" not in vocab
    assert vocab["<PAD>"] == 0
    assert vocab["<UNK>"] == 1


def test_unknown_token_handling():
    vocab = build_vocab(["show me nike shoes"])
    # "reebok" was never in the training queries, so it must be <UNK> (id 1).
    assert encode("show reebok shoes", vocab) == [
        vocab["show"],
        vocab["<UNK>"],
        vocab["shoes"],
    ]


def test_encode_cleans_before_lookup():
    vocab = build_vocab(["show me nike shoes under 5000"])
    assert encode("Show me Nike Shoes under 5000!!!", vocab) == encode(
        "show me nike shoes under 5000", vocab
    )


def test_encode_decode_roundtrip_for_known_words():
    raw = "Show me Nike Shoes under 5000!!!"
    vocab = build_vocab([raw, "more nike shoes"])
    token_ids = encode(raw, vocab)
    assert decode(token_ids, vocab) == tokenize(clean_text(raw))


def test_unknown_word_decodes_to_unk_token():
    vocab = build_vocab(["show me nike shoes"])
    assert decode(encode("show reebok shoes", vocab), vocab) == ["show", "<UNK>", "shoes"]


def test_save_and_load_vocab(tmp_path):
    vocab = build_vocab(["show me nike shoes"], min_frequency=1)
    path = tmp_path / "vocab.json"
    save_vocab(vocab, str(path))
    loaded = load_vocab(str(path))
    assert loaded == vocab
    assert encode("show nike", loaded) == encode("show nike", vocab)
