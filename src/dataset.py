"""Prepare labeled queries for a future intent-classification transformer.

This module sits between the tokenizer and the model. It does not create
embeddings, attention layers, or a neural network. It only turns each JSONL
row into three parallel lists that training code can later stack into tensors:

    input_ids       token ids from tokenizer.encode, padded to max_length
    attention_mask  1 for a real token, 0 for a pad position
    label           integer class id (GENERIC=0 ... LEGAL=3)

Why padding and masks exist
---------------------------
Queries have different lengths. A transformer batch multiplies rectangular
tensors, so every sequence in a batch must share one length. We pad short
sequences with the tokenizer's <PAD> id (0). Those zeros are not words. The
attention mask marks them with 0 so a later model can ignore them when it
computes attention. Building the mask here keeps that contract visible before
any model code exists.

Pipeline
--------
intents.jsonl row
    -> read text and intent string
    -> map intent string to label id
    -> encode text with tokenizer.encode (clean, tokenize, vocab lookup)
    -> truncate or pad to max_length
    -> build attention_mask of the same length
    -> {"input_ids", "attention_mask", "label"}

JSONL fields
------------
Each line is a JSON object with an intent string and a query string. This
project's file uses the key "text". The key "query" is also accepted so the
same loader works with either name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

from tokenizer import PAD_TOKEN, encode

# Fixed label ids for the four intents. Training and evaluation must use the
# same mapping. Changing an id later would silently relabel the whole set.
INTENT_TO_ID: Dict[str, int] = {
    "GENERIC": 0,
    "ECOMMERCE": 1,
    "RETAIL": 2,
    "LEGAL": 3,
}

ID_TO_INTENT: Dict[int, str] = {index: name for name, index in INTENT_TO_ID.items()}

# Default pad id matches tokenizer.build_vocab: <PAD> is always 0.
DEFAULT_PAD_ID = 0


def label_to_id(intent: str) -> int:
    """Map an intent string to its integer class id.

    Args:
        intent: One of GENERIC, ECOMMERCE, RETAIL, LEGAL.

    Returns:
        The fixed class id used as the training target.

    Raises:
        KeyError: The intent string is not one of the four known labels.
    """
    if intent not in INTENT_TO_ID:
        known = ", ".join(INTENT_TO_ID)
        raise KeyError(f"unknown intent {intent!r}; expected one of: {known}")
    return INTENT_TO_ID[intent]


def load_dataset(path: Union[str, Path]) -> List[Dict[str, str]]:
    """Read intents.jsonl into a list of raw examples.

    Each non-empty line must be a JSON object with an intent and a query.
    The query may be stored under "text" (this project's file) or "query".
    Rows are returned in file order. No encoding or padding happens here so
    you can inspect the raw strings before they become ids.

    Args:
        path: Path to a JSONL file.

    Returns:
        A list of dicts with keys "text" and "intent". The query field is
        always normalized to "text" even when the file used "query".

    Raises:
        ValueError: A line is missing the query, missing the intent, or is
            not valid JSON.
        FileNotFoundError: The path does not exist.
    """
    path = Path(path)
    examples: List[Dict[str, str]] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on line {line_number} of {path}"
                ) from error

            if "text" in row:
                text = row["text"]
            elif "query" in row:
                text = row["query"]
            else:
                raise ValueError(
                    f"line {line_number} of {path} needs a 'text' or 'query' field"
                )

            if "intent" not in row:
                raise ValueError(
                    f"line {line_number} of {path} is missing the 'intent' field"
                )

            examples.append({"text": str(text), "intent": str(row["intent"])})

    return examples


def pad_sequence(
    token_ids: List[int],
    max_length: int,
    pad_id: int = DEFAULT_PAD_ID,
) -> List[int]:
    """Truncate or pad a token-id list to exactly max_length.

    Short sequences get pad_id appended on the right until they reach
    max_length. Long sequences keep the first max_length tokens and drop the
    rest. Keeping the start is the usual choice for short classification
    queries: the beginning of the sentence usually carries the intent.

    Args:
        token_ids: Ids from tokenizer.encode.
        max_length: Target length for every sequence in a batch.
        pad_id: Id written into empty slots. Must match <PAD> in the vocab
            (0 in this project).

    Returns:
        A list of length max_length.

    Raises:
        ValueError: max_length is less than 1.
    """
    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    if len(token_ids) > max_length:
        # Truncate from the right so the opening words stay.
        return token_ids[:max_length]

    padding_needed = max_length - len(token_ids)
    return token_ids + [pad_id] * padding_needed


def create_attention_mask(sequence_length: int, max_length: int) -> List[int]:
    """Build a mask that marks real tokens with 1 and pads with 0.

    sequence_length is how many tokens are real after truncation, before
    padding. The mask is always length max_length so it lines up with
    input_ids position by position.

    Example:
        sequence_length=4, max_length=6 -> [1, 1, 1, 1, 0, 0]

    Args:
        sequence_length: Number of non-pad tokens in the final sequence.
        max_length: Same max_length used by pad_sequence.

    Returns:
        A list of 1s followed by 0s, length max_length.

    Raises:
        ValueError: Lengths are inconsistent or non-positive.
    """
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    if sequence_length < 0 or sequence_length > max_length:
        raise ValueError(
            "sequence_length must be between 0 and max_length inclusive"
        )

    return [1] * sequence_length + [0] * (max_length - sequence_length)


def prepare_sample(
    text: str,
    intent: str,
    vocab: Dict[str, int],
    max_length: int,
) -> Dict[str, object]:
    """Encode one query into input_ids, attention_mask, and label.

    This is the unit a future DataLoader would yield. The attention mask is
    built from the truncated length, not by scanning for pad ids, so the
    meaning of each position stays explicit.

    Args:
        text: Raw user query.
        intent: Intent string from the JSONL row.
        vocab: Vocabulary from tokenizer.build_vocab or load_vocab.
        max_length: Fixed sequence length after truncate/pad.

    Returns:
        {
            "input_ids": [...],
            "attention_mask": [...],
            "label": int,
        }
    """
    token_ids = encode(text, vocab)
    # Real token count after the same truncation pad_sequence will apply.
    real_length = min(len(token_ids), max_length)
    pad_id = vocab.get(PAD_TOKEN, DEFAULT_PAD_ID)

    return {
        "input_ids": pad_sequence(token_ids, max_length, pad_id=pad_id),
        "attention_mask": create_attention_mask(real_length, max_length),
        "label": label_to_id(intent),
    }


def prepare_dataset(
    path: Union[str, Path],
    vocab: Dict[str, int],
    max_length: int,
) -> List[Dict[str, object]]:
    """Load a JSONL file and prepare every row for training.

    Args:
        path: Path to intents.jsonl (or a smaller split file).
        vocab: Shared vocabulary used for every row.
        max_length: Shared sequence length used for every row.

    Returns:
        A list of prepare_sample results, one per JSONL line, in order.
    """
    return [
        prepare_sample(
            text=example["text"],
            intent=example["intent"],
            vocab=vocab,
            max_length=max_length,
        )
        for example in load_dataset(path)
    ]
