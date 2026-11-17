"""Tests for dataset preparation.

Run from the project directory:

    pytest tests/test_dataset.py
"""

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from dataset import (  # noqa: E402
    INTENT_TO_ID,
    create_attention_mask,
    label_to_id,
    load_dataset,
    pad_sequence,
    prepare_dataset,
    prepare_sample,
)
from tokenizer import build_vocab  # noqa: E402


def test_label_mapping_matches_fixed_ids():
    assert label_to_id("GENERIC") == 0
    assert label_to_id("ECOMMERCE") == 1
    assert label_to_id("RETAIL") == 2
    assert label_to_id("LEGAL") == 3
    assert INTENT_TO_ID == {
        "GENERIC": 0,
        "ECOMMERCE": 1,
        "RETAIL": 2,
        "LEGAL": 3,
    }


def test_label_mapping_rejects_unknown_intent():
    try:
        label_to_id("SPORTS")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_pad_sequence_appends_zeros():
    assert pad_sequence([2, 3, 4, 5], max_length=6) == [2, 3, 4, 5, 0, 0]


def test_pad_sequence_exact_length_is_unchanged():
    assert pad_sequence([2, 3, 4], max_length=3) == [2, 3, 4]


def test_pad_sequence_empty_becomes_all_pads():
    assert pad_sequence([], max_length=4) == [0, 0, 0, 0]


def test_create_attention_mask_marks_pads_as_zero():
    assert create_attention_mask(sequence_length=4, max_length=6) == [
        1,
        1,
        1,
        1,
        0,
        0,
    ]


def test_create_attention_mask_all_real_tokens():
    assert create_attention_mask(sequence_length=3, max_length=3) == [1, 1, 1]


def test_create_attention_mask_all_pads():
    assert create_attention_mask(sequence_length=0, max_length=3) == [0, 0, 0]


def test_max_length_truncates_long_sequences():
    long_ids = [10, 11, 12, 13, 14, 15, 16]
    padded = pad_sequence(long_ids, max_length=4)
    assert padded == [10, 11, 12, 13]
    assert len(padded) == 4
    assert create_attention_mask(sequence_length=4, max_length=4) == [1, 1, 1, 1]


def test_prepare_sample_pads_and_masks_together():
    vocab = build_vocab(["show me nike shoes"])
    sample = prepare_sample(
        text="show me nike shoes",
        intent="ECOMMERCE",
        vocab=vocab,
        max_length=6,
    )

    assert sample["label"] == 1
    assert sample["input_ids"] == [
        vocab["show"],
        vocab["me"],
        vocab["nike"],
        vocab["shoes"],
        0,
        0,
    ]
    assert sample["attention_mask"] == [1, 1, 1, 1, 0, 0]
    assert len(sample["input_ids"]) == len(sample["attention_mask"]) == 6


def test_prepare_sample_truncates_when_longer_than_max_length():
    vocab = build_vocab(["show me nike shoes under five thousand dollars today"])
    sample = prepare_sample(
        text="show me nike shoes under five thousand dollars today",
        intent="ECOMMERCE",
        vocab=vocab,
        max_length=4,
    )

    assert sample["input_ids"] == [
        vocab["show"],
        vocab["me"],
        vocab["nike"],
        vocab["shoes"],
    ]
    assert sample["attention_mask"] == [1, 1, 1, 1]
    assert 0 not in sample["input_ids"]


def test_load_dataset_reads_text_field(tmp_path):
    path = tmp_path / "tiny.jsonl"
    path.write_text(
        '{"text": "show me nike shoes", "intent": "ECOMMERCE"}\n'
        '{"text": "store hours today", "intent": "RETAIL"}\n',
        encoding="utf-8",
    )

    rows = load_dataset(path)
    assert rows == [
        {"text": "show me nike shoes", "intent": "ECOMMERCE"},
        {"text": "store hours today", "intent": "RETAIL"},
    ]


def test_load_dataset_accepts_query_field(tmp_path):
    path = tmp_path / "query_style.jsonl"
    path.write_text(
        '{"query": "show me nike shoes", "intent": "ECOMMERCE"}\n',
        encoding="utf-8",
    )

    rows = load_dataset(path)
    assert rows[0]["text"] == "show me nike shoes"
    assert rows[0]["intent"] == "ECOMMERCE"


def test_prepare_dataset_end_to_end(tmp_path):
    path = tmp_path / "intents.jsonl"
    rows = [
        {"text": "show me nike shoes", "intent": "ECOMMERCE"},
        {"query": "what is photosynthesis", "intent": "GENERIC"},
        {"text": "fitting room line", "intent": "RETAIL"},
        {"text": "file a lien", "intent": "LEGAL"},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    vocab = build_vocab([row.get("text", row.get("query")) for row in rows])
    samples = prepare_dataset(path, vocab=vocab, max_length=8)

    assert len(samples) == 4
    assert [sample["label"] for sample in samples] == [1, 0, 2, 3]
    for sample in samples:
        assert len(sample["input_ids"]) == 8
        assert len(sample["attention_mask"]) == 8
        assert sum(sample["attention_mask"]) == sum(
            1 for token_id in sample["input_ids"] if token_id != 0
        )


def test_prepare_dataset_loads_project_intents_file():
    data_path = Path(__file__).resolve().parents[1] / "data" / "intents.jsonl"
    rows = load_dataset(data_path)
    assert len(rows) == 240

    vocab = build_vocab(example["text"] for example in rows)
    samples = prepare_dataset(data_path, vocab=vocab, max_length=32)

    assert len(samples) == 240
    assert {sample["label"] for sample in samples} == {0, 1, 2, 3}
    assert all(len(sample["input_ids"]) == 32 for sample in samples)
    assert all(len(sample["attention_mask"]) == 32 for sample in samples)
