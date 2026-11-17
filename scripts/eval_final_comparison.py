"""Final ID + OOD evaluation after training on intents_v3.

Run from the project root after ``python -m src.train``::

    python scripts/eval_final_comparison.py

Writes ``artifacts/ood_final_metrics.json`` and
``artifacts/final_comparison.json``.
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


def evaluate_in_distribution_v3() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, checkpoint = load_checkpoint(ARTIFACTS, device=device)
    rows = load_dataset(PROJECT_ROOT / "data" / "intents_v3.jsonl")
    _, _, test_rows = stratified_split(rows, seed=42)
    texts, labels = prepare_examples(test_rows)
    dataset = IntentDataset(texts, labels, vocab, max_length=model.max_length)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    result = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    metrics = compute_classification_metrics(result["labels"], result["predictions"])
    return {
        "dataset": "intents_v3.jsonl",
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

    print("=== Final OOD evaluation (post-augmentation checkpoint) ===")
    ood_report = evaluate_checkpoint_on_ood(
        artifacts_dir=ARTIFACTS,
        ood_path=PROJECT_ROOT / "data" / "intent_ood_test.jsonl",
        output_path=ARTIFACTS / "ood_final_metrics.json",
        include_probes=True,
    )
    print(f"OOD accuracy: {ood_report['accuracy']:.4f}")
    print(f"OOD macro F1: {ood_report['macro_f1']:.4f}")

    print("\n=== Final in-distribution (intents_v3 test split) ===")
    id_report = evaluate_in_distribution_v3()
    (ARTIFACTS / "id_final_metrics.json").write_text(
        json.dumps(id_report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"ID accuracy: {id_report['accuracy']:.4f}")
    print(f"ID macro F1: {id_report['macro_f1']:.4f}")

    id_baseline_path = ARTIFACTS / "id_baseline_metrics.json"
    ood_baseline_path = ARTIFACTS / "ood_baseline_metrics.json"
    id_baseline = (
        json.loads(id_baseline_path.read_text(encoding="utf-8"))
        if id_baseline_path.exists()
        else {}
    )
    ood_baseline = (
        json.loads(ood_baseline_path.read_text(encoding="utf-8"))
        if ood_baseline_path.exists()
        else {}
    )

    comparison = {
        "previous_in_distribution": {
            "accuracy": id_baseline.get("accuracy"),
            "macro_f1": id_baseline.get("macro_f1"),
            "source": "intents_v2 held-out test / pre-augmentation checkpoint",
        },
        "new_in_distribution": {
            "accuracy": id_report["accuracy"],
            "macro_f1": id_report["macro_f1"],
            "best_epoch": id_report["best_epoch"],
            "source": "intents_v3 held-out test / post-augmentation checkpoint",
        },
        "baseline_ood": {
            "accuracy": ood_baseline.get("accuracy"),
            "macro_f1": ood_baseline.get("macro_f1"),
        },
        "new_ood": {
            "accuracy": ood_report["accuracy"],
            "macro_f1": ood_report["macro_f1"],
        },
        "ood_per_class": {
            "baseline": ood_baseline.get("per_class"),
            "new": ood_report["per_class"],
        },
        "ood_confusion_matrices": {
            "baseline": ood_baseline.get("confusion_matrix"),
            "new": ood_report["confusion_matrix"],
        },
        "unk_rates": {
            "baseline_overall": ood_baseline.get("overall_unk_rate"),
            "new_overall": ood_report["overall_unk_rate"],
            "baseline_per_class": ood_baseline.get("unk_rate_per_class"),
            "new_per_class": ood_report["unk_rate_per_class"],
        },
        "accuracy_by_unk_count": {
            "baseline": ood_baseline.get("performance_by_unk_count"),
            "new": ood_report["performance_by_unk_count"],
        },
        "manual_probes": ood_report.get("manual_probes"),
        "artifact_paths": {
            "id_baseline": str(id_baseline_path),
            "id_final": str(ARTIFACTS / "id_final_metrics.json"),
            "ood_baseline": str(ood_baseline_path),
            "ood_final": str(ARTIFACTS / "ood_final_metrics.json"),
            "best_model": str(ARTIFACTS / "best_model.pt"),
            "vocab": str(ARTIFACTS / "vocab.json"),
        },
    }
    out = ARTIFACTS / "final_comparison.json"
    out.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved comparison: {out}")

    print("\n=== Manual probes (post-augmentation) ===")
    for probe in comparison["manual_probes"]:
        probs = probe["probability_distribution"]
        print(f"\n  text: {probe['text']}")
        print(f"  tokens: {probe['cleaned_tokens']}")
        print(f"  unk_count: {probe['unk_count']}")
        print(f"  predicted: {probe['predicted_intent']} ({probe['confidence']:.4f})")
        print(
            "  probs: "
            + ", ".join(f"{k}={v:.4f}" for k, v in probs.items())
        )


if __name__ == "__main__":
    main()
