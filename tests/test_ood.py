"""Tests for OOD isolation, v3 coverage, and OOD evaluation schemas.

Run from the project directory:

    pytest tests/test_ood.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dataset import load_dataset  # noqa: E402
from ood_eval import (  # noqa: E402
    REQUIRED_OOD_METRIC_FIELDS,
    REQUIRED_PROBE_FIELDS,
    categorize_ood_example,
    evaluate_ood,
    predict_one,
    unk_bucket,
    validate_ood_metrics_schema,
    validate_probe_schema,
)
from train import stratified_split  # noqa: E402
from tokenizer import build_vocab  # noqa: E402
from trainable_intent_model import IntentClassificationTransformer  # noqa: E402
import torch  # noqa: E402


OOD_PATH = ROOT / "data" / "intent_ood_test.jsonl"
V2_PATH = ROOT / "data" / "intents_v2.jsonl"
V3_PATH = ROOT / "data" / "intents_v3.jsonl"
ARTIFACTS = ROOT / "artifacts"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@pytest.fixture(scope="module")
def ood_rows():
    assert OOD_PATH.exists(), "OOD benchmark missing"
    return load_dataset(OOD_PATH)


@pytest.fixture(scope="module")
def v3_rows():
    assert V3_PATH.exists(), "intents_v3 missing"
    return load_dataset(V3_PATH)


@pytest.fixture(scope="module")
def v2_rows():
    assert V2_PATH.exists()
    return load_dataset(V2_PATH)


def test_ood_exact_class_counts(ood_rows):
    counts = Counter(r["intent"] for r in ood_rows)
    assert len(ood_rows) == 400
    assert counts == {
        "GENERIC": 100,
        "ECOMMERCE": 100,
        "RETAIL": 100,
        "LEGAL": 100,
    }


def test_v3_exact_class_counts(v3_rows):
    counts = Counter(r["intent"] for r in v3_rows)
    assert len(v3_rows) == 2800
    assert counts == {
        "GENERIC": 700,
        "ECOMMERCE": 700,
        "RETAIL": 700,
        "LEGAL": 700,
    }


def test_ood_never_included_in_v3_training_data(ood_rows, v3_rows):
    ood_norms = {_norm(r["text"]) for r in ood_rows}
    v3_norms = {_norm(r["text"]) for r in v3_rows}
    overlap = ood_norms & v3_norms
    assert not overlap, f"OOD leaked into v3: {list(overlap)[:5]}"


def test_ood_rows_do_not_overlap_v2(ood_rows, v2_rows):
    ood_norms = {_norm(r["text"]) for r in ood_rows}
    v2_norms = {_norm(r["text"]) for r in v2_rows}
    assert not (ood_norms & v2_norms)


def test_ood_no_exact_duplicates(ood_rows):
    norms = [_norm(r["text"]) for r in ood_rows]
    assert len(norms) == len(set(norms))


def test_v3_no_exact_duplicates(v3_rows):
    norms = [_norm(r["text"]) for r in v3_rows]
    assert len(norms) == len(set(norms))


def test_fixed_split_reproducibility(v3_rows):
    a_train, a_val, a_test = stratified_split(v3_rows, seed=42)
    b_train, b_val, b_test = stratified_split(v3_rows, seed=42)
    assert [r["text"] for r in a_train] == [r["text"] for r in b_train]
    assert [r["text"] for r in a_val] == [r["text"] for r in b_val]
    assert [r["text"] for r in a_test] == [r["text"] for r in b_test]
    # OOD must not appear in any split of v3
    ood_norms = {_norm(r["text"]) for r in load_dataset(OOD_PATH)}
    for split in (a_train, a_val, a_test):
        assert not ({_norm(r["text"]) for r in split} & ood_norms)


def test_grouped_metrics_by_unk_count():
    assert unk_bucket(0) == "0"
    assert unk_bucket(1) == "1"
    assert unk_bucket(2) == "2+"
    assert unk_bucket(5) == "2+"


def test_ood_evaluator_output_schema(ood_rows):
    vocab = build_vocab([r["text"] for r in ood_rows[:20]])
    model = IntentClassificationTransformer(
        vocab_size=len(vocab),
        embedding_dim=8,
        num_heads=2,
        hidden_dim=16,
        num_layers=1,
        num_classes=4,
        max_length=16,
        pad_token_id=0,
        dropout=0.0,
    )
    # Evaluate on a tiny slice written temporarily via existing file is heavy;
    # instead evaluate on full OOD with tiny model for schema only.
    report = evaluate_ood(model, vocab, ood_path=OOD_PATH, batch_size=16)
    missing = validate_ood_metrics_schema(report)
    assert missing == [], f"missing fields: {missing}"
    for field in REQUIRED_OOD_METRIC_FIELDS:
        assert field in report
    assert set(report["performance_by_unk_count"]) == {"0", "1", "2+"}


def test_manual_probe_output_schema():
    vocab = build_vocab(["what is the capital of france nike shoes"])
    model = IntentClassificationTransformer(
        vocab_size=len(vocab),
        embedding_dim=8,
        num_heads=2,
        hidden_dim=16,
        num_layers=1,
        num_classes=4,
        max_length=16,
        pad_token_id=0,
        dropout=0.0,
    )
    probe = predict_one(model, vocab, "what is the capital of france")
    missing = validate_probe_schema(probe)
    assert missing == []
    for field in REQUIRED_PROBE_FIELDS:
        assert field in probe
    assert set(probe["probability_distribution"]) == {
        "GENERIC",
        "ECOMMERCE",
        "RETAIL",
        "LEGAL",
    }


def test_artifacts_contain_required_fields():
    ood_report_path = ARTIFACTS / "ood_dataset_report.json"
    v3_report_path = ARTIFACTS / "intents_v3_report.json"
    assert ood_report_path.exists()
    assert v3_report_path.exists()
    ood_report = json.loads(ood_report_path.read_text(encoding="utf-8"))
    v3_report = json.loads(v3_report_path.read_text(encoding="utf-8"))
    for key in (
        "per_class_counts",
        "duplicate_count",
        "average_length_by_class",
        "question_ratio_by_class",
        "ambiguous_candidates",
    ):
        assert key in ood_report
    assert "unk_rate_overall_with_current_vocab" in ood_report
    for key in (
        "per_class_counts",
        "duplicate_count",
        "average_length_by_class",
        "question_ratio_by_class",
        "validation",
    ):
        assert key in v3_report
    assert ood_report["per_class_counts"] == {
        "GENERIC": 100,
        "ECOMMERCE": 100,
        "RETAIL": 100,
        "LEGAL": 100,
    }
    assert v3_report["per_class_counts"] == {
        "GENERIC": 700,
        "ECOMMERCE": 700,
        "RETAIL": 700,
        "LEGAL": 700,
    }


def test_categorize_ood_example_tags():
    assert "short_legal_definition" in categorize_ood_example(
        "what is an arbitration clause", "LEGAL"
    )
    assert "brand_and_price_ecommerce" in categorize_ood_example(
        "show me nike shoes under 5000", "ECOMMERCE"
    )
    assert "short_retail_operational" in categorize_ood_example(
        "inventory level of store 12", "RETAIL"
    )
    assert "generic_what_is" in categorize_ood_example(
        "what is the capital of france", "GENERIC"
    )
