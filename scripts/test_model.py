"""Quick smoke / debug test for the trained intent-classification model.

Run from the project root after training::

    python scripts/test_model.py
    python scripts/test_model.py --debug
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tokenizer import clean_text, encode, load_vocab, tokenize  # noqa: E402
from trainable_intent_model import IntentClassificationTransformer  # noqa: E402

_MODEL: Optional[IntentClassificationTransformer] = None
_VOCAB: Optional[Dict[str, int]] = None
_ID_TO_INTENT: Dict[int, str] = {}
_CHECKPOINT_META: Dict[str, Any] = {}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_artifacts() -> None:
    """Load vocab, labels, and best checkpoint once."""
    global _MODEL, _VOCAB, _ID_TO_INTENT, _CHECKPOINT_META

    if _MODEL is not None:
        return

    vocab_path = ARTIFACTS_DIR / "vocab.json"
    labels_path = ARTIFACTS_DIR / "label_mapping.json"
    model_path = ARTIFACTS_DIR / "best_model.pt"

    for path in (vocab_path, labels_path, model_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artifact {path}. Run `python -m src.train` first."
            )

    _VOCAB = load_vocab(str(vocab_path))
    label_to_id = json.loads(labels_path.read_text(encoding="utf-8"))
    _ID_TO_INTENT = {int(index): name for name, index in label_to_id.items()}

    checkpoint = torch.load(model_path, map_location=_DEVICE, weights_only=False)
    config = checkpoint["model_config"]
    _CHECKPOINT_META = {
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "labels_path": str(labels_path),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "model_config": config,
        "vocab_size_file": len(_VOCAB),
        "vocab_size_config": config["vocab_size"],
        "vocab_sizes_match": len(_VOCAB) == config["vocab_size"],
        "label_to_id": label_to_id,
        "id_to_intent": _ID_TO_INTENT,
    }

    _MODEL = IntentClassificationTransformer(**config)
    _MODEL.load_state_dict(checkpoint["model_state_dict"])
    _MODEL.to(_DEVICE)
    _MODEL.eval()


def _encode_query(
    text: str, max_length: int
) -> tuple[List[int], List[int], List[str], str, int]:
    """Return padded ids, mask, tokens, cleaned text, and UNK count."""
    assert _VOCAB is not None
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    token_ids = encode(text, _VOCAB)[:max_length]
    if not token_ids:
        raise ValueError(f"query encodes to zero tokens: {text!r}")

    unk_id = _VOCAB["<UNK>"]
    unk_count = sum(1 for tid in token_ids if tid == unk_id)
    attention = [1] * len(token_ids)
    pad_count = max_length - len(token_ids)
    if pad_count:
        token_ids = token_ids + [0] * pad_count
        attention = attention + [0] * pad_count
    return token_ids, attention, tokens, cleaned, unk_count


def predict(query: str, debug: bool = False) -> Dict[str, Any]:
    """Return intent name, confidence, and full probability vector."""
    _load_artifacts()
    assert _MODEL is not None and _VOCAB is not None

    padded_ids, attention_mask, tokens, cleaned, unk_count = _encode_query(
        query, _MODEL.max_length
    )
    input_ids = torch.tensor([padded_ids], dtype=torch.long, device=_DEVICE)
    attention_tensor = torch.tensor(
        [attention_mask], dtype=torch.long, device=_DEVICE
    )

    # Explicit eval + no_grad (predict_proba does this too; keep it visible).
    was_training = _MODEL.training
    _MODEL.eval()
    with torch.no_grad():
        logits = _MODEL(input_ids, attention_tensor)
        probabilities = F.softmax(logits, dim=-1)[0]
    _MODEL.train(was_training)

    predicted_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_id].item())
    result = {
        "intent": _ID_TO_INTENT[predicted_id],
        "confidence": confidence,
        "probabilities": {
            _ID_TO_INTENT[index]: float(probabilities[index].item())
            for index in range(len(_ID_TO_INTENT))
        },
        "logits": [float(x) for x in logits[0].tolist()],
        "cleaned": cleaned,
        "tokens": tokens,
        "token_ids_unpadded": [tid for tid, m in zip(padded_ids, attention_mask) if m],
        "attention_mask": attention_mask,
        "unk_count": unk_count,
        "model_training_flag": _MODEL.training,
    }

    if debug:
        print("\n=== Debug prediction ===")
        print(f"  raw query: {query!r}")
        print(f"  cleaned query: {cleaned!r}")
        print(f"  tokens: {tokens}")
        print(f"  token ids: {result['token_ids_unpadded']}")
        print(f"  attention mask (first 16): {attention_mask[:16]}")
        print(f"  UNK count: {unk_count}/{len(result['token_ids_unpadded'])}")
        print(f"  logits: {[round(x, 4) for x in result['logits']]}")
        print(
            "  probabilities:",
            {k: round(v, 4) for k, v in result["probabilities"].items()},
        )
        print(f"  predicted label: {result['intent']} (id={predicted_id})")
        print(f"  model.eval active during forward: {not was_training or True}")

    return result


def print_artifact_audit() -> None:
    _load_artifacts()
    print("=== Artifact audit ===")
    for key, value in _CHECKPOINT_META.items():
        print(f"  {key}: {value}")
    print(f"  model.training after load: {_MODEL.training if _MODEL else None}")


def print_prediction(query: str, prediction: Dict[str, Any], debug: bool = False) -> None:
    """Pretty-print one inference result."""
    print()
    print("Query:", query)
    print("Tokens:", prediction["tokens"])
    print("UNK count:", prediction["unk_count"])
    print("Prediction:", prediction["intent"])
    print("Confidence:", f"{prediction['confidence']:.4f}")
    print("Probabilities:")
    for intent, prob in prediction["probabilities"].items():
        marker = " <--" if intent == prediction["intent"] else ""
        print(f"  {intent:10s} {prob:.4f}{marker}")
    if debug:
        print("Logits:", [round(x, 4) for x in prediction["logits"]])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run intent classification on one or more queries"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Query text. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "-q",
        "--query-flag",
        dest="query_flag",
        action="append",
        default=[],
        help="Query text (repeatable). Alternative to positional args.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print tokens, ids, mask, UNK, logits, and probabilities",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the built-in demo queries instead of interactive mode",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print each result as a single JSON object",
    )
    args = parser.parse_args()

    queries: List[str] = list(args.query_flag)
    if args.query:
        # Support: python scripts/test_model.py show me nike shoes
        # and:     python scripts/test_model.py "show me nike shoes"
        if len(args.query) == 1:
            queries.append(args.query[0])
        else:
            queries.append(" ".join(args.query))

    if args.demo and not queries:
        queries = [
            "show me nike shoes under 5000",
            "inventory level of store 12",
            "what is arbitration clause",
            "what is the capital of france",
            "compare sofas on the website",
            "compare sofas on the showroom floor",
            "they changed the agreement after I signed it",
            "how are sofas manufactured",
        ]

    print_artifact_audit()

    if queries:
        for query in queries:
            prediction = predict(query, debug=args.debug)
            if args.json:
                print(json.dumps({
                    "query": query,
                    "intent": prediction["intent"],
                    "confidence": prediction["confidence"],
                    "unk_count": prediction["unk_count"],
                    "tokens": prediction["tokens"],
                    "probabilities": prediction["probabilities"],
                }, indent=2))
            else:
                print_prediction(query, prediction, debug=args.debug)
        return

    # Interactive REPL
    print()
    print("Interactive mode. Type a query and press Enter.")
    print("Commands: :quit / :exit / :demo / :debug on|off")
    debug = args.debug
    while True:
        try:
            line = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {":quit", ":exit", "quit", "exit"}:
            break
        if line.lower() == ":demo":
            for query in [
                "show me nike shoes under 5000",
                "what is arbitration clause",
                "inventory level of store 12",
            ]:
                prediction = predict(query, debug=debug)
                print_prediction(query, prediction, debug=debug)
            continue
        if line.lower().startswith(":debug"):
            parts = line.split()
            if len(parts) == 2 and parts[1].lower() in {"on", "off"}:
                debug = parts[1].lower() == "on"
                print(f"debug={'on' if debug else 'off'}")
            else:
                print("usage: :debug on|off")
            continue

        prediction = predict(line, debug=debug)
        if args.json:
            print(json.dumps({
                "query": line,
                "intent": prediction["intent"],
                "confidence": prediction["confidence"],
                "unk_count": prediction["unk_count"],
                "tokens": prediction["tokens"],
                "probabilities": prediction["probabilities"],
            }, indent=2))
        else:
            print_prediction(line, prediction, debug=debug)


if __name__ == "__main__":
    main()
