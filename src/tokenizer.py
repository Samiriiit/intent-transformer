"""Whitespace tokenizer for the intent-classification transformer.

This module is the first stage of the training pipeline. It turns a raw user
query into a list of integer ids. A later stage will use those ids to look up
rows in an embedding table. Nothing in this file creates embeddings, attention,
or a model.

Pipeline
--------
raw query
    -> clean_text        (lowercase, strip punctuation, keep numbers)
    -> tokenize          (split on spaces)
    -> encode            (replace each word with its vocab id)
    -> token ids         (what a transformer batch expects)

The vocabulary is built once from the training queries, then saved. Validation
and test queries must be encoded with that same vocabulary so ids mean the
same thing at train time and at eval time.

Special tokens
--------------
<PAD> is id 0.
    Queries in one batch have different lengths. Training code will append
    <PAD> until every sequence in the batch is the same length, because the
    model multiplies rectangular tensors. Those positions are not real words.
    Later, an attention mask will tell the model to ignore them. Id 0 is the
    usual place to put padding so a zero-filled tensor is already padded.

<UNK> is id 1.
    A word is unknown when it never appeared in training, or when it was too
    rare to keep (see min_frequency). Mapping every unknown word to one id
    lets encode finish instead of failing. The model learns a single vector
    for "some word I have not memorized."
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, Iterable, List

# Anything that is not a letter, a digit, or whitespace is punctuation for
# our purposes. Replacing it with a space (instead of deleting it) keeps
# "shoes,under" as two words. Digits are excluded so "5000" stays intact.
_PUNCTUATION = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE = re.compile(r"\s+")

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def clean_text(text: str) -> str:
    """Normalize a raw query into lowercase words and numbers.

    Lowercasing means "Nike" and "nike" share one vocab entry. Punctuation is
    turned into spaces and then collapsed, so extra spaces and marks like
    "!!!" do not become their own tokens. Numbers are kept because prices and
    counts ("under 5000") are meaningful in these queries.

    Args:
        text: The original user query.

    Returns:
        A single-spaced lowercase string with no punctuation.

    Example:
        >>> clean_text("Show me Nike Shoes under 5000!!!")
        'show me nike shoes under 5000'
    """
    lowered = text.lower()
    separated = _PUNCTUATION.sub(" ", lowered)
    return _WHITESPACE.sub(" ", separated).strip()


def tokenize(text: str) -> List[str]:
    """Split cleaned text into word tokens on whitespace.

    This is deliberately not BPE or WordPiece. A whitespace split is enough
    to see how ids flow into a transformer, and every token is still a word
    you can read. Call clean_text first. encode does that for you.

    Args:
        text: Text that has already been passed through clean_text.

    Returns:
        One string per whitespace-separated word. Empty input yields [].

    Example:
        >>> tokenize("show me nike shoes under 5000")
        ['show', 'me', 'nike', 'shoes', 'under', '5000']
    """
    if not text:
        return []
    return text.split()


def build_vocab(dataset: Iterable[str], min_frequency: int = 1) -> Dict[str, int]:
    """Build a token-to-id map from training queries.

    Each string is cleaned and tokenized, then counted. Tokens seen fewer
    than min_frequency times are left out. encode will later map them to
    <UNK>. Dropping rare words keeps the embedding table smaller and stops
    the model from memorizing one-off typos.

    Ids are assigned in first-seen order, starting at 2. <PAD> and <UNK> are
    always inserted first, even if they never appear in the text. First-seen
    order is stable as long as the training list is in a stable order.

    Args:
        dataset: Training queries as raw strings. For this project that is
            the "text" field of each row in data/intents.jsonl, not the
            whole JSON object.
        min_frequency: Minimum number of training occurrences required to
            receive a dedicated id. Must be at least 1.

    Returns:
        A dict mapping token string to integer id.

    Example:
        >>> build_vocab(["show me nike shoes"])
        {'<PAD>': 0, '<UNK>': 1, 'show': 2, 'me': 3, 'nike': 4, 'shoes': 5}
    """
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")

    counts: Counter[str] = Counter()
    first_seen: List[str] = []
    for query in dataset:
        for token in tokenize(clean_text(query)):
            if token not in counts:
                first_seen.append(token)
            counts[token] += 1

    vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    next_id = 2
    for token in first_seen:
        if counts[token] >= min_frequency:
            vocab[token] = next_id
            next_id += 1
    return vocab


def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    """Turn one raw query into token ids.

    The steps are fixed: clean_text, then tokenize, then a vocab lookup.
    A missing word uses the <UNK> id. The result is the sequence a later
    embedding layer will index. This function does not pad the sequence.
    Padding to a batch length happens when examples are grouped.

    Args:
        text: A raw user query.
        vocab: Map from build_vocab, save/load, or a hand-built dict.
            It must contain <UNK>.

    Returns:
        One id per token, in order.

    Example:
        With show=2, shoes=4, and reebok unknown:
        >>> encode("show reebok shoes", vocab)
        [2, 1, 4]
    """
    unknown_id = vocab[UNK_TOKEN]
    return [vocab.get(token, unknown_id) for token in tokenize(clean_text(text))]


def decode(token_ids: List[int], vocab: Dict[str, int]) -> List[str]:
    """Turn token ids back into token strings.

    This is the inverse of the vocab lookup, not of the original raw text.
    Punctuation and capitalization are already gone, and an unknown word
    comes back as "<UNK>" rather than the word that was typed. Decoding is
    how you inspect what the model actually received.

    Args:
        token_ids: Ids produced by encode, or a padded batch row.
        vocab: The same map that was used to encode.

    Returns:
        Token strings, including "<PAD>" and "<UNK>" when those ids appear.

    Raises:
        KeyError: An id is not one of the vocab values.

    Example:
        >>> decode([2, 3, 4], vocab)
        ['show', 'nike', 'shoes']
    """
    id_to_token = {index: token for token, index in vocab.items()}
    tokens: List[str] = []
    for index in token_ids:
        if index not in id_to_token:
            raise KeyError(f"token id {index} is not in the vocabulary")
        tokens.append(id_to_token[index])
    return tokens


def save_vocab(vocab: Dict[str, int], path: str) -> None:
    """Write a vocabulary dict to a JSON file.

    JSON is enough here: the object is a flat string-to-int map, and reading
    it does not require PyTorch. Training and evaluation both load this file
    so they share ids. The vocab is an argument, not hidden state, because
    this module is a set of functions rather than a tokenizer object.

    Args:
        vocab: Token-to-id map from build_vocab.
        path: Destination file path.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(vocab, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_vocab(path: str) -> Dict[str, int]:
    """Read a vocabulary JSON file written by save_vocab.

    Args:
        path: Path passed to save_vocab.

    Returns:
        Token-to-id map. Ids are ints even if a hand-edited file stored
        them in a form JSON would otherwise leave ambiguous.
    """
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(token): int(index) for token, index in raw.items()}
