"""Out-of-distribution evaluation utilities for the intent classifier.

The OOD benchmark (``data/intent_ood_test.jsonl``) is evaluation-only and must
never be used for training, vocabulary construction, early stopping, or
checkpoint selection.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dataset import ID_TO_INTENT, INTENT_TO_ID, label_to_id, load_dataset
from tokenizer import UNK_TOKEN, clean_text, encode, load_vocab, tokenize
from train import (
    LABEL_NAMES,
    IntentDataset,
    compute_classification_metrics,
    create_dataloaders,
    evaluate,
    prepare_examples,
)
from trainable_intent_model import IntentClassificationTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OOD_PATH = PROJECT_ROOT / "data" / "intent_ood_test.jsonl"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MANUAL_PROBES = [
    "show me nike shoes under 5000",
    "inventory level of store 12",
    "what is arbitration clause",
    "what is the capital of france",
    "compare sofas on the website",
    "compare sofas on the showroom floor",
    "they changed the agreement after I signed it",
    "how are sofas manufactured",
]

QUESTION_PREFIXES = (
    "what ",
    "why ",
    "how ",
    "when ",
    "where ",
    "which ",
    "who ",
    "can ",
    "is ",
    "are ",
    "do ",
    "does ",
    "did ",
    "should ",
    "could ",
    "would ",
)


def is_question(text: str) -> bool:
    """Heuristic: trailing '?' or leading interrogative word."""
    stripped = text.strip().lower()
    if "?" in stripped:
        return True
    return any(stripped.startswith(prefix) for prefix in QUESTION_PREFIXES)


def load_checkpoint(
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    device: Optional[torch.device] = None,
) -> Tuple[IntentClassificationTransformer, Dict[str, int], Dict[str, Any]]:
    """Load best_model.pt + vocab.json from artifacts."""
    device = device or torch.device("cpu")
    vocab = load_vocab(str(artifacts_dir / "vocab.json"))
    checkpoint = torch.load(
        artifacts_dir / "best_model.pt", map_location=device, weights_only=False
    )
    config = dict(checkpoint["model_config"])
    model = IntentClassificationTransformer(**config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, vocab, checkpoint


def count_unk_tokens(text: str, vocab: Dict[str, int]) -> int:
    unk_id = vocab[UNK_TOKEN]
    return sum(1 for tid in encode(text, vocab) if tid == unk_id)


def unk_bucket(unk_count: int) -> str:
    if unk_count <= 0:
        return "0"
    if unk_count == 1:
        return "1"
    return "2+"


def categorize_ood_example(text: str, intent: str) -> List[str]:
    """Tag examples into analysis subgroups used in the OOD report."""
    lower = text.lower().strip()
    tags: List[str] = []
    if intent == "LEGAL" and (
        lower.startswith("what is")
        or lower.startswith("define ")
        or lower.startswith("what does ")
    ):
        tags.append("short_legal_definition")
    if intent == "ECOMMERCE" and (
        any(ch.isdigit() for ch in text)
        or any(
            brand in lower
            for brand in (
                "nike",
                "adidas",
                "samsung",
                "apple",
                "sony",
                "puma",
                "boat",
                "realme",
                "vivo",
                "lenovo",
                "dell",
                "hp",
            )
        )
        or any(cur in lower for cur in ("rupee", "dollar", "euro", "rs ", "inr", "usd"))
    ):
        tags.append("brand_and_price_ecommerce")
    if intent == "RETAIL" and (
        len(tokenize(clean_text(text))) <= 10
        or any(
            w in lower
            for w in (
                "aisle",
                "backroom",
                "sku",
                "outlet",
                "fitting",
                "shelf",
                "inventory",
                "transfer",
                "restock",
            )
        )
    ):
        tags.append("short_retail_operational")
    if intent == "GENERIC" and lower.startswith("what is"):
        tags.append("generic_what_is")
    return tags


def predict_one(
    model: IntentClassificationTransformer,
    vocab: Dict[str, int],
    text: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run a single probe and return tokens, UNK count, prediction, probs."""
    device = device or next(model.parameters()).device
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    token_ids = encode(text, vocab)
    unk_id = vocab[UNK_TOKEN]
    unk_count = sum(1 for tid in token_ids if tid == unk_id)

    max_length = model.max_length
    truncated = token_ids[:max_length]
    attention = [1] * len(truncated)
    pad_count = max_length - len(truncated)
    padded_ids = truncated + [0] * pad_count
    attention = attention + [0] * pad_count

    input_ids = torch.tensor([padded_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attention], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(probs.argmax().item())

    probability_distribution = {
        ID_TO_INTENT[i]: float(probs[i].item()) for i in range(len(ID_TO_INTENT))
    }
    return {
        "text": text,
        "cleaned_tokens": tokens,
        "unk_count": unk_count,
        "predicted_intent": ID_TO_INTENT[pred_id],
        "confidence": float(probs[pred_id].item()),
        "probability_distribution": probability_distribution,
    }


def _accuracy_for_subset(
    labels: Sequence[int], predictions: Sequence[int], indices: Sequence[int]
) -> Optional[float]:
    if not indices:
        return None
    y_true = [labels[i] for i in indices]
    y_pred = [predictions[i] for i in indices]
    return float(accuracy_score(y_true, y_pred))


def _macro_f1_for_subset(
    labels: Sequence[int], predictions: Sequence[int], indices: Sequence[int]
) -> Optional[float]:
    if not indices:
        return None
    y_true = [labels[i] for i in indices]
    y_pred = [predictions[i] for i in indices]
    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
            labels=list(range(len(LABEL_NAMES))),
        )
    )


def evaluate_ood(
    model: IntentClassificationTransformer,
    vocab: Dict[str, int],
    ood_path: Path = DEFAULT_OOD_PATH,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate a loaded model on the isolated OOD benchmark."""
    device = device or next(model.parameters()).device
    rows = load_dataset(ood_path)
    texts, labels = prepare_examples(rows)
    dataset = IntentDataset(texts, labels, vocab, max_length=model.max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    result = evaluate(model, loader, criterion, device)
    metrics = compute_classification_metrics(result["labels"], result["predictions"])

    unk_counts = [count_unk_tokens(text, vocab) for text in texts]
    total_tokens = 0
    total_unk = 0
    per_class_token_unk: Dict[str, Dict[str, int]] = {
        name: {"tokens": 0, "unk": 0} for name in LABEL_NAMES
    }
    unk_id = vocab[UNK_TOKEN]
    for text, label in zip(texts, labels):
        ids = encode(text, vocab)
        n_unk = sum(1 for tid in ids if tid == unk_id)
        total_tokens += len(ids)
        total_unk += n_unk
        intent = ID_TO_INTENT[label]
        per_class_token_unk[intent]["tokens"] += len(ids)
        per_class_token_unk[intent]["unk"] += n_unk

    overall_unk_rate = 100.0 * total_unk / max(total_tokens, 1)
    unk_rate_per_class = {
        name: round(
            100.0
            * per_class_token_unk[name]["unk"]
            / max(per_class_token_unk[name]["tokens"], 1),
            4,
        )
        for name in LABEL_NAMES
    }

    by_unk: Dict[str, List[int]] = {"0": [], "1": [], "2+": []}
    for index, count in enumerate(unk_counts):
        by_unk[unk_bucket(count)].append(index)

    performance_by_unk_count = {}
    for bucket, indices in by_unk.items():
        performance_by_unk_count[bucket] = {
            "support": len(indices),
            "accuracy": _accuracy_for_subset(
                result["labels"], result["predictions"], indices
            ),
            "macro_f1": _macro_f1_for_subset(
                result["labels"], result["predictions"], indices
            ),
        }

    subgroup_indices: Dict[str, List[int]] = defaultdict(list)
    for index, (text, label) in enumerate(zip(texts, labels)):
        intent = ID_TO_INTENT[label]
        for tag in categorize_ood_example(text, intent):
            subgroup_indices[tag].append(index)

    performance_by_subgroup = {}
    for tag, indices in sorted(subgroup_indices.items()):
        performance_by_subgroup[tag] = {
            "support": len(indices),
            "accuracy": _accuracy_for_subset(
                result["labels"], result["predictions"], indices
            ),
            "macro_f1": _macro_f1_for_subset(
                result["labels"], result["predictions"], indices
            ),
        }

    return {
        "ood_path": str(ood_path),
        "n_examples": len(rows),
        "accuracy": float(result["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "loss": float(result["loss"]),
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "label_order": LABEL_NAMES,
        "overall_unk_rate": round(overall_unk_rate, 4),
        "unk_rate_per_class": unk_rate_per_class,
        "performance_by_unk_count": performance_by_unk_count,
        "performance_by_subgroup": performance_by_subgroup,
        "classification_report": metrics["classification_report"],
    }


def run_manual_probes(
    model: IntentClassificationTransformer,
    vocab: Dict[str, int],
    probes: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    probes = list(probes) if probes is not None else list(MANUAL_PROBES)
    return [predict_one(model, vocab, text, device=device) for text in probes]


def evaluate_checkpoint_on_ood(
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    ood_path: Path = DEFAULT_OOD_PATH,
    output_path: Optional[Path] = None,
    include_probes: bool = False,
) -> Dict[str, Any]:
    """Load artifacts, evaluate OOD, optionally probes, optionally save JSON."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, checkpoint = load_checkpoint(artifacts_dir, device=device)
    report = evaluate_ood(model, vocab, ood_path=ood_path, device=device)
    report["checkpoint"] = {
        "best_epoch": checkpoint.get("best_epoch"),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "vocab_size": len(vocab),
        "model_config": checkpoint.get("model_config"),
    }
    if include_probes:
        report["manual_probes"] = run_manual_probes(model, vocab, device=device)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


REQUIRED_OOD_METRIC_FIELDS = (
    "accuracy",
    "macro_f1",
    "per_class",
    "confusion_matrix",
    "overall_unk_rate",
    "unk_rate_per_class",
    "performance_by_unk_count",
    "performance_by_subgroup",
)

REQUIRED_PROBE_FIELDS = (
    "text",
    "cleaned_tokens",
    "unk_count",
    "predicted_intent",
    "confidence",
    "probability_distribution",
)


def validate_ood_metrics_schema(report: Dict[str, Any]) -> List[str]:
    missing = [field for field in REQUIRED_OOD_METRIC_FIELDS if field not in report]
    return missing


def validate_probe_schema(probe: Dict[str, Any]) -> List[str]:
    missing = [field for field in REQUIRED_PROBE_FIELDS if field not in probe]
    return missing
