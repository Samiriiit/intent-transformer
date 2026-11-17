"""Evaluate the current checkpoint on the OOD benchmark (baseline, no retrain).

Run from the project root::

    python scripts/eval_ood_baseline.py

Writes ``artifacts/ood_baseline_metrics.json`` and also records the previous
in-distribution test metrics on ``intents_v2.jsonl`` into
``artifacts/id_baseline_metrics.json`` for later comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
ARTIFACTS = PROJECT_ROOT / "artifacts"
sys.path.insert(0, str(SRC_DIR))

from dataset import load_dataset  # noqa: E402
from ood_eval import (  # noqa: E402
    evaluate_checkpoint_on_ood,
    evaluate_ood,
    load_checkpoint,
    run_manual_probes,
)
from train import (  # noqa: E402
    IntentDataset,
    compute_classification_metrics,
    evaluate,
    prepare_examples,
    stratified_split,
)


def evaluate_in_distribution_v2() -> dict:
    """Re-evaluate the current (v2-trained) checkpoint on the v2 ID test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, checkpoint = load_checkpoint(ARTIFACTS, device=device)
    rows = load_dataset(PROJECT_ROOT / "data" / "intents_v2.jsonl")
    _, _, test_rows = stratified_split(rows, seed=42)
    texts, labels = prepare_examples(test_rows)
    dataset = IntentDataset(texts, labels, vocab, max_length=model.max_length)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    result = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    metrics = compute_classification_metrics(result["labels"], result["predictions"])
    return {
        "dataset": "intents_v2.jsonl",
        "split": "held_out_test",
        "seed": 42,
        "n_examples": len(test_rows),
        "accuracy": float(result["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "best_epoch": checkpoint.get("best_epoch"),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "vocab_size": len(vocab),
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print("=== Baseline OOD evaluation (pre-augmentation checkpoint) ===")
    ood_report = evaluate_checkpoint_on_ood(
        artifacts_dir=ARTIFACTS,
        ood_path=PROJECT_ROOT / "data" / "intent_ood_test.jsonl",
        output_path=ARTIFACTS / "ood_baseline_metrics.json",
        include_probes=True,
    )
    print(f"OOD accuracy: {ood_report['accuracy']:.4f}")
    print(f"OOD macro F1: {ood_report['macro_f1']:.4f}")
    print(f"Overall UNK rate: {ood_report['overall_unk_rate']:.2f}%")
    print(f"Saved: {ARTIFACTS / 'ood_baseline_metrics.json'}")

    print("\n=== Baseline in-distribution (intents_v2 test split) ===")
    id_report = evaluate_in_distribution_v2()
    (ARTIFACTS / "id_baseline_metrics.json").write_text(
        json.dumps(id_report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"ID accuracy: {id_report['accuracy']:.4f}")
    print(f"ID macro F1: {id_report['macro_f1']:.4f}")
    print(f"Saved: {ARTIFACTS / 'id_baseline_metrics.json'}")

    print("\n=== Baseline manual probes ===")
    for probe in ood_report.get("manual_probes", []):
        print(
            f"  [{probe['predicted_intent']}] "
            f"unk={probe['unk_count']} conf={probe['confidence']:.3f} "
            f"| {probe['text']}"
        )


if __name__ == "__main__":
    main()
