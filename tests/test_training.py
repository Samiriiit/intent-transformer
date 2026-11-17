"""Tests for the training and validation pipeline.

Run from the project directory:

    pytest tests/test_training.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from tokenizer import build_vocab, encode, load_vocab  # noqa: E402
from train import (  # noqa: E402
    EarlyStopping,
    IntentDataset,
    build_model,
    build_vocab_from_texts,
    create_dataloaders,
    evaluate,
    prepare_examples,
    set_seed,
    stratified_split,
    train_one_epoch,
)
from trainable_intent_model import IntentClassificationTransformer  # noqa: E402


TINY_ROWS = [
    {"text": "weather in paris today", "intent": "GENERIC"},
    {"text": "how many cups in a liter", "intent": "GENERIC"},
    {"text": "tell me a joke please", "intent": "GENERIC"},
    {"text": "capital of france quiz", "intent": "GENERIC"},
    {"text": "sunrise time tomorrow morning", "intent": "GENERIC"},
    {"text": "meaning of the word serendipity", "intent": "GENERIC"},
    {"text": "buy nike shoes online now", "intent": "ECOMMERCE"},
    {"text": "track my package shipment", "intent": "ECOMMERCE"},
    {"text": "cancel subscription billing please", "intent": "ECOMMERCE"},
    {"text": "apply discount code checkout", "intent": "ECOMMERCE"},
    {"text": "refund my online order", "intent": "ECOMMERCE"},
    {"text": "cart total before tax", "intent": "ECOMMERCE"},
    {"text": "store hours on sunday", "intent": "RETAIL"},
    {"text": "fitting room near shoes", "intent": "RETAIL"},
    {"text": "aisle for laundry detergent", "intent": "RETAIL"},
    {"text": "price check at register", "intent": "RETAIL"},
    {"text": "curbside pickup at maple", "intent": "RETAIL"},
    {"text": "where is lost and found", "intent": "RETAIL"},
    {"text": "file a small claims case", "intent": "LEGAL"},
    {"text": "review this rental lease", "intent": "LEGAL"},
    {"text": "noncompete clause after leaving", "intent": "LEGAL"},
    {"text": "power of attorney form", "intent": "LEGAL"},
    {"text": "copyright the class syllabus", "intent": "LEGAL"},
    {"text": "contest a traffic ticket", "intent": "LEGAL"},
]


def _tiny_model(vocab_size: int, max_length: int = 16):
    torch.manual_seed(0)
    return IntentClassificationTransformer(
        vocab_size=vocab_size,
        embedding_dim=8,
        num_heads=2,
        hidden_dim=16,
        num_layers=1,
        num_classes=4,
        max_length=max_length,
        pad_token_id=0,
        dropout=0.0,
    )


def test_dataset_item_shapes_and_dtypes():
    vocab = build_vocab(["buy nike shoes online"])
    dataset = IntentDataset(
        texts=["buy nike shoes"],
        labels=[1],
        vocab=vocab,
        max_length=8,
    )
    item = dataset[0]
    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].shape == (8,)
    assert item["label"].shape == ()
    assert item["input_ids"].dtype == torch.long
    assert item["attention_mask"].dtype == torch.long
    assert item["label"].dtype == torch.long


def test_padding_behavior():
    vocab = build_vocab(["buy nike shoes"])
    dataset = IntentDataset(["buy nike"], [1], vocab, max_length=5)
    item = dataset[0]
    assert item["input_ids"].tolist()[:2] == encode("buy nike", vocab)
    assert item["input_ids"].tolist()[2:] == [0, 0, 0]
    assert item["attention_mask"].tolist() == [1, 1, 0, 0, 0]


def test_truncation_behavior():
    vocab = build_vocab(["one two three four five six"])
    dataset = IntentDataset(
        ["one two three four five six"], [0], vocab, max_length=4
    )
    item = dataset[0]
    assert item["input_ids"].tolist() == encode("one two three four", vocab)
    assert item["attention_mask"].tolist() == [1, 1, 1, 1]


def test_attention_mask_correctness():
    vocab = build_vocab(["hello world again"])
    dataset = IntentDataset(["hello world"], [0], vocab, max_length=6)
    mask = dataset[0]["attention_mask"].tolist()
    assert mask.count(1) == 2
    assert mask.count(0) == 4
    assert mask == [1, 1, 0, 0, 0, 0]


def test_unknown_token_behavior():
    vocab = build_vocab(["buy nike shoes"])
    dataset = IntentDataset(["buy reebok shoes"], [1], vocab, max_length=5)
    ids = dataset[0]["input_ids"].tolist()
    assert ids[0] == vocab["buy"]
    assert ids[1] == vocab["<UNK>"]
    assert ids[2] == vocab["shoes"]


def test_label_mapping():
    texts, labels = prepare_examples(TINY_ROWS)
    assert set(labels) == {0, 1, 2, 3}
    assert labels[0] == 0
    assert labels[6] == 1


def test_stratified_split_has_all_four_classes():
    train, val, test = stratified_split(TINY_ROWS, seed=42)
    for split in (train, val, test):
        intents = {row["intent"] for row in split}
        assert intents == {"GENERIC", "ECOMMERCE", "RETAIL", "LEGAL"}


def test_no_overlap_among_splits():
    train, val, test = stratified_split(TINY_ROWS, seed=42)
    train_texts = {row["text"] for row in train}
    val_texts = {row["text"] for row in val}
    test_texts = {row["text"] for row in test}
    assert train_texts.isdisjoint(val_texts)
    assert train_texts.isdisjoint(test_texts)
    assert val_texts.isdisjoint(test_texts)
    assert len(train_texts) + len(val_texts) + len(test_texts) == len(TINY_ROWS)


def test_vocabulary_uses_training_text_only():
    train, val, _ = stratified_split(TINY_ROWS, seed=7)
    train_texts = [row["text"] for row in train]
    vocab = build_vocab_from_texts(train_texts)
    assert set(vocab).issuperset({"<PAD>", "<UNK>"})
    larger = build_vocab(train_texts + [row["text"] for row in val])
    assert set(vocab).issubset(set(larger))


def test_dataloader_batch_shapes():
    train, val, test = stratified_split(TINY_ROWS, seed=1)
    vocab = build_vocab_from_texts(row["text"] for row in train)
    train_ds = IntentDataset(*prepare_examples(train), vocab=vocab, max_length=12)
    val_ds = IntentDataset(*prepare_examples(val), vocab=vocab, max_length=12)
    test_ds = IntentDataset(*prepare_examples(test), vocab=vocab, max_length=12)
    train_loader, _, _ = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=4
    )
    batch = next(iter(train_loader))
    assert batch["input_ids"].ndim == 2
    assert batch["input_ids"].shape[1] == 12
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["label"].ndim == 1
    assert batch["label"].shape[0] == batch["input_ids"].shape[0]


def test_train_one_epoch_updates_model_parameters():
    train, _, _ = stratified_split(TINY_ROWS, seed=2)
    vocab = build_vocab_from_texts(row["text"] for row in train)
    dataset = IntentDataset(*prepare_examples(train), vocab=vocab, max_length=12)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = _tiny_model(len(vocab), max_length=12)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    loss, acc = train_one_epoch(
        model, loader, optimizer, criterion, torch.device("cpu")
    )
    assert np.isfinite(loss)
    assert 0.0 <= acc <= 1.0
    changed = any(
        not torch.equal(before[name], param.detach())
        for name, param in model.named_parameters()
    )
    assert changed


def test_evaluate_does_not_update_parameters():
    train, val, _ = stratified_split(TINY_ROWS, seed=3)
    vocab = build_vocab_from_texts(row["text"] for row in train)
    dataset = IntentDataset(*prepare_examples(val), vocab=vocab, max_length=12)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = _tiny_model(len(vocab), max_length=12)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    criterion = nn.CrossEntropyLoss()
    result = evaluate(model, loader, criterion, torch.device("cpu"))
    assert np.isfinite(result["loss"])
    for name, param in model.named_parameters():
        assert torch.equal(before[name], param.detach())


def test_training_loss_is_finite():
    train, _, _ = stratified_split(TINY_ROWS, seed=4)
    vocab = build_vocab_from_texts(row["text"] for row in train)
    dataset = IntentDataset(*prepare_examples(train), vocab=vocab, max_length=12)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = _tiny_model(len(vocab), max_length=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss, _ = train_one_epoch(
        model, loader, optimizer, nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert np.isfinite(loss)


def test_evaluation_outputs_have_correct_lengths():
    train, val, _ = stratified_split(TINY_ROWS, seed=5)
    vocab = build_vocab_from_texts(row["text"] for row in train)
    dataset = IntentDataset(*prepare_examples(val), vocab=vocab, max_length=12)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = _tiny_model(len(vocab), max_length=12)
    result = evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    n = len(dataset)
    assert len(result["labels"]) == n
    assert len(result["predictions"]) == n
    assert len(result["probabilities"]) == n


def test_probability_rows_sum_approximately_to_one():
    train, val, _ = stratified_split(TINY_ROWS, seed=6)
    vocab = build_vocab_from_texts(row["text"] for row in train)
    dataset = IntentDataset(*prepare_examples(val), vocab=vocab, max_length=12)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = _tiny_model(len(vocab), max_length=12)
    result = evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    probs = np.asarray(result["probabilities"])
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(probs)), atol=1e-5)


def test_early_stopping_state_behavior():
    model = _tiny_model(vocab_size=20, max_length=8)
    stopper = EarlyStopping(patience=2, min_delta=1e-4)
    assert stopper.step(1.0, model, epoch=1) is False
    first_state = {k: v.clone() for k, v in stopper.best_state.items()}
    assert stopper.step(1.5, model, epoch=2) is False
    assert stopper.step(1.6, model, epoch=3) is True
    assert stopper.best_epoch == 1
    for key, value in first_state.items():
        assert torch.equal(value, stopper.best_state[key])


def test_checkpoint_contains_required_keys(tmp_path):
    model = _tiny_model(vocab_size=20, max_length=8)
    stopper = EarlyStopping(patience=1, min_delta=0.0)
    stopper.step(0.5, model, epoch=1)
    path = tmp_path / "best_model.pt"
    payload = {
        "model_state_dict": stopper.best_state,
        "model_config": {"vocab_size": 20, "embedding_dim": 8},
        "best_epoch": stopper.best_epoch,
        "best_validation_loss": stopper.best_loss,
    }
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert set(loaded) >= {
        "model_state_dict",
        "model_config",
        "best_epoch",
        "best_validation_loss",
    }


def test_saved_vocabulary_reloads_correctly(tmp_path):
    vocab = build_vocab(["buy nike shoes under 5000"])
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(vocab), encoding="utf-8")
    loaded = load_vocab(str(path))
    assert loaded == vocab
    assert encode("buy nike", loaded) == encode("buy nike", vocab)


def test_fixed_seed_produces_identical_splits():
    a_train, a_val, a_test = stratified_split(TINY_ROWS, seed=42)
    b_train, b_val, b_test = stratified_split(TINY_ROWS, seed=42)
    assert [row["text"] for row in a_train] == [row["text"] for row in b_train]
    assert [row["text"] for row in a_val] == [row["text"] for row in b_val]
    assert [row["text"] for row in a_test] == [row["text"] for row in b_test]


def test_set_seed_is_callable():
    set_seed(42)
    first = torch.rand(3)
    set_seed(42)
    second = torch.rand(3)
    assert torch.allclose(first, second)


def test_build_model_default_config_shape():
    model = build_model(vocab_size=30, max_length=16)
    assert isinstance(model, IntentClassificationTransformer)
    assert model.embedding_dim == 64
    assert model.num_heads == 4
    assert model.num_classes == 4
