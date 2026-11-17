"""Training and validation pipeline for intent classification.

Builds vocabulary from the **training split only**, trains
``IntentClassificationTransformer`` with AdamW + CrossEntropyLoss, selects
the best checkpoint by validation loss (early stopping), then evaluates once
on the held-out test set.

Default data file: ``data/intents_v3.jsonl`` (field ``text`` / intent labels).
Intent labels map to::

    GENERIC=0, ECOMMERCE=1, RETAIL=2, LEGAL=3

Run from the project root::

    python -m src.train
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

# Educational modules import each other as top-level names (``from tokenizer
# import ...``). Put ``src/`` on sys.path so ``python -m src.train`` and
# ``from train import ...`` both work without editing those files.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dataset import INTENT_TO_ID, ID_TO_INTENT, label_to_id, load_dataset
from tokenizer import (
    PAD_TOKEN,
    UNK_TOKEN,
    build_vocab,
    encode,
    load_vocab,
    save_vocab,
    tokenize,
    clean_text,
)
from trainable_intent_model import IntentClassificationTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "intents_v3.jsonl"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 64
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_PATIENCE = 8
DEFAULT_MIN_DELTA = 1e-4

LABEL_NAMES = [ID_TO_INTENT[i] for i in range(len(ID_TO_INTENT))]


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and CUDA when present)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stratified_split(
    examples: Sequence[Dict[str, Any]],
    seed: int = DEFAULT_SEED,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified 70/15/15 split. Every intent appears in each split."""
    by_intent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_intent[example["intent"]].append(example)

    missing = [name for name in INTENT_TO_ID if name not in by_intent]
    if missing:
        raise ValueError(f"dataset is missing intents: {missing}")

    rng = random.Random(seed)
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for intent, rows in sorted(by_intent.items()):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n < 3:
            raise ValueError(
                f"intent {intent} has only {n} examples; need at least 3 "
                "for a non-empty train/val/test split"
            )
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Give leftovers to test so all rows are used.
        n_test = n - n_train - n_val
        # Guarantee at least one example per split.
        if n_train < 1 or n_val < 1 or n_test < 1:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = n - n_train - n_val
            if n_test < 1:
                n_test = 1
                if n_train >= n_val and n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    raise ValueError(
                        f"cannot form three non-empty splits for intent {intent}"
                    )

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    _assert_all_intents_present(train, "train")
    _assert_all_intents_present(val, "validation")
    _assert_all_intents_present(test, "test")
    return train, val, test


def _assert_all_intents_present(
    split: Sequence[Dict[str, Any]], split_name: str
) -> None:
    present = {row["intent"] for row in split}
    missing = [name for name in INTENT_TO_ID if name not in present]
    if missing:
        raise ValueError(f"{split_name} split is missing intents: {missing}")


class IntentDataset(Dataset):
    """Encode texts with a fixed vocab into padded tensors for the model."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        vocab: Dict[str, int],
        max_length: int = DEFAULT_MAX_LENGTH,
        pad_token_id: int = 0,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if PAD_TOKEN not in vocab or UNK_TOKEN not in vocab:
            raise ValueError("vocab must contain <PAD> and <UNK>")

        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        for label in self.labels:
            if label not in ID_TO_INTENT:
                raise ValueError(f"label {label} is outside 0..3")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        token_ids = encode(self.texts[index], self.vocab)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]

        if len(token_ids) == 0:
            raise ValueError(
                f"sample at index {index} encodes to zero tokens "
                "(would be fully padded)"
            )

        attention_mask = [1] * len(token_ids)
        pad_count = self.max_length - len(token_ids)
        if pad_count:
            token_ids = token_ids + [self.pad_token_id] * pad_count
            attention_mask = attention_mask + [0] * pad_count

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }


def prepare_examples(
    rows: Sequence[Dict[str, str]],
) -> Tuple[List[str], List[int]]:
    """Convert raw JSONL rows to parallel text and label-id lists."""
    texts: List[str] = []
    labels: List[int] = []
    for row in rows:
        texts.append(row["text"])
        labels.append(label_to_id(row["intent"]))
    if any(label == -1 for label in labels):  # pragma: no cover
        raise ValueError("invalid labels detected")
    if not texts:
        raise ValueError("split contains zero examples")
    class_counts = Counter(labels)
    for intent_id in INTENT_TO_ID.values():
        if class_counts.get(intent_id, 0) == 0:
            raise ValueError(
                f"split has zero examples for class {ID_TO_INTENT[intent_id]}"
            )
    return texts, labels


def build_vocab_from_texts(texts: Iterable[str], min_frequency: int = 1) -> Dict[str, int]:
    """Build vocabulary from training texts only (no val/test leakage)."""
    return build_vocab(texts, min_frequency=min_frequency)


def create_dataloaders(
    train_dataset: IntentDataset,
    val_dataset: IntentDataset,
    test_dataset: IntentDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build loaders; shuffle only the training loader."""
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def build_model(vocab_size: int, max_length: int = DEFAULT_MAX_LENGTH) -> IntentClassificationTransformer:
    """Instantiate the default intent Transformer configuration."""
    return IntentClassificationTransformer(
        vocab_size=vocab_size,
        embedding_dim=64,
        num_heads=4,
        hidden_dim=256,
        num_layers=2,
        num_classes=4,
        max_length=max_length,
        pad_token_id=0,
        dropout=0.1,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (average_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("training dataloader yielded zero examples")
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate without updating parameters.

    Returns average loss, accuracy, true labels, predictions, and probabilities.
    """
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_probabilities: List[List[float]] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        probabilities = torch.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    if total_examples == 0:
        raise ValueError("evaluation dataloader yielded zero examples")

    accuracy = accuracy_score(all_labels, all_predictions)
    return {
        "loss": total_loss / total_examples,
        "accuracy": float(accuracy),
        "labels": all_labels,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
    }


class EarlyStopping:
    """Stop when validation loss stops improving; keep the best state dict."""

    def __init__(
        self,
        patience: int = DEFAULT_PATIENCE,
        min_delta: float = DEFAULT_MIN_DELTA,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.epochs_without_improvement = 0
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Update bookkeeping. Returns True when training should stop."""
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = float(val_loss)
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore(self, model: nn.Module) -> None:
        """Load the best weights into ``model``."""
        if self.best_state is None:
            raise RuntimeError("no best state to restore; call step() first")
        model.load_state_dict(self.best_state)


def compute_classification_metrics(
    labels: Sequence[int], predictions: Sequence[int]
) -> Dict[str, Any]:
    """Precision/recall/F1 per class, macro F1, and confusion matrix."""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(LABEL_NAMES))),
        average=None,
        zero_division=0,
    )
    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
        labels=list(range(len(LABEL_NAMES))),
    )
    matrix = confusion_matrix(
        labels, predictions, labels=list(range(len(LABEL_NAMES)))
    )
    per_class = {}
    for index, name in enumerate(LABEL_NAMES):
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    return {
        "per_class": per_class,
        "macro_f1": float(macro_f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(LABEL_NAMES))),
            target_names=LABEL_NAMES,
            zero_division=0,
            digits=4,
        ),
    }


def token_length(text: str) -> int:
    """Number of tokens after the project's clean/tokenize pipeline."""
    return len(tokenize(clean_text(text)))


def unk_token_percentage(texts: Sequence[str], vocab: Dict[str, int]) -> float:
    """Fraction of tokens that map to <UNK> under ``vocab``."""
    unk_id = vocab[UNK_TOKEN]
    total = 0
    unk = 0
    for text in texts:
        ids = encode(text, vocab)
        total += len(ids)
        unk += sum(1 for token_id in ids if token_id == unk_id)
    if total == 0:
        return 0.0
    return 100.0 * unk / total


def print_split_diagnostics(
    train_rows: Sequence[Dict[str, str]],
    val_rows: Sequence[Dict[str, str]],
    test_rows: Sequence[Dict[str, str]],
    vocab: Dict[str, int],
) -> Dict[str, float]:
    """Print class counts, split sizes, vocab size, and UNK rates.

    Returns UNK percentages for train, validation, and test.
    """
    all_rows = list(train_rows) + list(val_rows) + list(test_rows)
    overall = Counter(row["intent"] for row in all_rows)
    print("\n=== Dataset diagnostics ===")
    print("Class counts (full dataset):")
    for intent in LABEL_NAMES:
        print(f"  {intent}: {overall[intent]}")

    print(f"\nVocabulary size: {len(vocab)}")
    print(
        f"Split counts | train={len(train_rows)} | "
        f"validation={len(val_rows)} | test={len(test_rows)}"
    )

    for split_name, rows in (
        ("train", train_rows),
        ("validation", val_rows),
        ("test", test_rows),
    ):
        counts = Counter(row["intent"] for row in rows)
        print(f"\n[{split_name}] examples per intent:")
        for intent in LABEL_NAMES:
            intent_rows = [row for row in rows if row["intent"] == intent]
            lengths = [token_length(row["text"]) for row in intent_rows]
            mean_len = sum(lengths) / len(lengths) if lengths else 0.0
            print(
                f"  {intent}: count={counts[intent]} "
                f"avg_tokens={mean_len:.2f}"
            )

    train_texts = [row["text"] for row in train_rows]
    val_texts = [row["text"] for row in val_rows]
    test_texts = [row["text"] for row in test_rows]
    unk_rates = {
        "train": unk_token_percentage(train_texts, vocab),
        "validation": unk_token_percentage(val_texts, vocab),
        "test": unk_token_percentage(test_texts, vocab),
    }
    print(
        f"\nUNK token % | train={unk_rates['train']:.2f}% "
        f"| validation={unk_rates['validation']:.2f}% "
        f"| test={unk_rates['test']:.2f}%"
    )
    return unk_rates


def save_artifacts(
    artifacts_dir: Path,
    model: nn.Module,
    model_config: Dict[str, Any],
    vocab: Dict[str, int],
    early_stopping: EarlyStopping,
    history: List[Dict[str, Any]],
) -> Dict[str, Path]:
    """Write model checkpoint, vocab, label map, and training history."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "best_model.pt"
    vocab_path = artifacts_dir / "vocab.json"
    labels_path = artifacts_dir / "label_mapping.json"
    history_path = artifacts_dir / "training_history.json"

    torch.save(
        {
            "model_state_dict": early_stopping.best_state
            if early_stopping.best_state is not None
            else model.state_dict(),
            "model_config": model_config,
            "best_epoch": early_stopping.best_epoch,
            "best_validation_loss": early_stopping.best_loss,
        },
        model_path,
    )
    save_vocab(vocab, str(vocab_path))
    labels_path.write_text(
        json.dumps(INTENT_TO_ID, indent=2) + "\n", encoding="utf-8"
    )
    history_path.write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "best_model": model_path,
        "vocab": vocab_path,
        "label_mapping": labels_path,
        "training_history": history_path,
    }


def train_model(
    data_path: Path = DEFAULT_DATA_PATH,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = DEFAULT_SEED,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Full train / validate / test pipeline. Returns a results dictionary."""
    set_seed(seed)
    device = device or get_device()

    raw_examples = load_dataset(data_path)
    train_rows, val_rows, test_rows = stratified_split(raw_examples, seed=seed)

    train_texts, train_labels = prepare_examples(train_rows)
    val_texts, val_labels = prepare_examples(val_rows)
    test_texts, test_labels = prepare_examples(test_rows)

    # Vocabulary from training text only — never val/test.
    vocab = build_vocab_from_texts(train_texts)
    unk_rates = print_split_diagnostics(train_rows, val_rows, test_rows, vocab)

    train_dataset = IntentDataset(train_texts, train_labels, vocab, max_length)
    val_dataset = IntentDataset(val_texts, val_labels, vocab, max_length)
    test_dataset = IntentDataset(test_texts, test_labels, vocab, max_length)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )

    model = build_model(vocab_size=len(vocab), max_length=max_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    history: List[Dict[str, Any]] = []

    model_config = {
        "vocab_size": len(vocab),
        "embedding_dim": 64,
        "num_heads": 4,
        "hidden_dim": 256,
        "num_layers": 2,
        "num_classes": 4,
        "max_length": max_length,
        "pad_token_id": 0,
        "dropout": 0.1,
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_result = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "validation_loss": float(val_result["loss"]),
                "validation_accuracy": float(val_result["accuracy"]),
            }
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | train acc {train_accuracy:.4f} | "
            f"val loss {val_result['loss']:.4f} | "
            f"val acc {val_result['accuracy']:.4f}"
        )
        if early_stopping.step(val_result["loss"], model, epoch):
            print(
                f"Early stopping at epoch {epoch} "
                f"(best epoch {early_stopping.best_epoch})"
            )
            break

    early_stopping.restore(model)
    artifact_paths = save_artifacts(
        artifacts_dir,
        model,
        model_config,
        vocab,
        early_stopping,
        history,
    )

    # Test evaluation only after restoring the best validation checkpoint.
    test_result = evaluate(model, test_loader, criterion, device)
    metrics = compute_classification_metrics(
        test_result["labels"], test_result["predictions"]
    )

    best_history = next(
        (row for row in history if row["epoch"] == early_stopping.best_epoch),
        None,
    )
    best_train_acc = (
        best_history["train_accuracy"] if best_history is not None else None
    )
    best_val_acc = (
        best_history["validation_accuracy"] if best_history is not None else None
    )

    print("\n=== Test results (best validation checkpoint) ===")
    print(f"best epoch: {early_stopping.best_epoch}")
    print(f"best validation loss: {early_stopping.best_loss:.4f}")
    if best_train_acc is not None:
        print(f"train accuracy (best epoch): {best_train_acc:.4f}")
    if best_val_acc is not None:
        print(f"validation accuracy (best epoch): {best_val_acc:.4f}")
    print(f"test loss: {test_result['loss']:.4f}")
    print(f"test accuracy: {test_result['accuracy']:.4f}")
    print(f"macro F1: {metrics['macro_f1']:.4f}")
    print(f"vocabulary size: {len(vocab)}")
    print(f"train UNK rate: {unk_rates['train']:.2f}%")
    print(f"validation UNK rate: {unk_rates['validation']:.2f}%")
    print(f"test UNK rate: {unk_rates['test']:.2f}%")
    print("confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print(metrics["classification_report"])
    print("saved artifacts:")
    for name, path in artifact_paths.items():
        print(f"  {name}: {path}")

    return {
        "history": history,
        "best_epoch": early_stopping.best_epoch,
        "best_validation_loss": early_stopping.best_loss,
        "best_train_accuracy": best_train_acc,
        "best_validation_accuracy": best_val_acc,
        "test": test_result,
        "metrics": metrics,
        "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
        "split_counts": {
            "train": dict(Counter(row["intent"] for row in train_rows)),
            "validation": dict(Counter(row["intent"] for row in val_rows)),
            "test": dict(Counter(row["intent"] for row in test_rows)),
        },
        "vocab_size": len(vocab),
        "unk_rates": unk_rates,
        "device": str(device),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the intent-classification Transformer"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    return train_model(
        data_path=args.data,
        artifacts_dir=args.artifacts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        seed=args.seed,
        patience=args.patience,
        min_delta=args.min_delta,
    )


if __name__ == "__main__":
    main()
