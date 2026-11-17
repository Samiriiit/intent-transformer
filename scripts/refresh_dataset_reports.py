"""Refresh dataset reports without regenerating OOD/v3 rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_ood_and_v3 import (  # noqa: E402
    ARTIFACTS,
    OOD_PATH,
    OOD_REPORT,
    V2_PATH,
    V3_PATH,
    V3_REPORT,
    ambiguous_candidates,
    load_jsonl,
    norm,
    report_dataset,
)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    v2_rows = load_jsonl(V2_PATH)
    ood_rows = load_jsonl(OOD_PATH)
    v3_rows = load_jsonl(V3_PATH)
    v2_norms = {norm(r["text"]) for r in v2_rows}
    ood_norms = {norm(r["text"]) for r in ood_rows}

    ood_report = report_dataset(
        ood_rows,
        blocked_norms=v2_norms,
        name="intent_ood_test",
        extra={
            "purpose": "evaluation_only",
            "never_use_for": [
                "training",
                "validation",
                "vocabulary",
                "early_stopping",
                "checkpoint_selection",
            ],
            "examples_by_category": {
                intent: [r["text"] for r in ood_rows if r["intent"] == intent][:8]
                for intent in ("GENERIC", "ECOMMERCE", "RETAIL", "LEGAL")
            },
            "ambiguous_candidates": ambiguous_candidates(ood_rows),
        },
    )
    vocab_path = ARTIFACTS / "vocab.json"
    if vocab_path.exists():
        sys.path.insert(0, str(ROOT / "src"))
        from tokenizer import encode, load_vocab

        vocab = load_vocab(str(vocab_path))
        unk_id = vocab["<UNK>"]
        per_class_unk = {}
        total = unk = 0
        for intent in ("GENERIC", "ECOMMERCE", "RETAIL", "LEGAL"):
            texts = [r["text"] for r in ood_rows if r["intent"] == intent]
            t_tok = u_tok = 0
            for text in texts:
                ids = encode(text, vocab)
                t_tok += len(ids)
                u_tok += sum(i == unk_id for i in ids)
            per_class_unk[intent] = round(100 * u_tok / max(t_tok, 1), 2)
            total += t_tok
            unk += u_tok
        ood_report["unk_rate_overall_with_current_vocab"] = round(
            100 * unk / max(total, 1), 2
        )
        ood_report["unk_rate_by_class_with_current_vocab"] = per_class_unk

    OOD_REPORT.write_text(json.dumps(ood_report, indent=2) + "\n", encoding="utf-8")

    v3_report = report_dataset(
        v3_rows,
        blocked_norms=ood_norms,
        name="intents_v3",
        extra={
            "source": "intents_v2 + 800 targeted augmentations",
            "augmentation_counts": {
                "GENERIC": 200,
                "ECOMMERCE": 200,
                "RETAIL": 200,
                "LEGAL": 200,
            },
            "ambiguous_candidates": ambiguous_candidates(v3_rows),
            "confusion_candidates": ambiguous_candidates(v3_rows)[:25],
        },
    )
    added_rows = [r for r in v3_rows if norm(r["text"]) not in v2_norms]
    sys.path.insert(0, str(ROOT / "src"))
    from tokenizer import build_vocab, encode

    v2_vocab = build_vocab([r["text"] for r in v2_rows])
    unk_id = v2_vocab["<UNK>"]
    covered = total = 0
    for row in added_rows:
        ids = encode(row["text"], v2_vocab)
        total += len(ids)
        covered += sum(1 for i in ids if i != unk_id)
    v3_report["added_example_count"] = len(added_rows)
    v3_report["vocab_coverage_of_added_vs_v2_vocab"] = {
        "token_coverage_pct": round(100 * covered / max(total, 1), 2),
        "total_tokens": total,
        "known_tokens": covered,
        "unk_tokens": total - covered,
    }
    V3_REPORT.write_text(json.dumps(v3_report, indent=2) + "\n", encoding="utf-8")
    print("Refreshed", OOD_REPORT)
    print("Refreshed", V3_REPORT)


if __name__ == "__main__":
    main()
