"""Debug inference audit for the trained intent classifier.

Run from the project root::

    python scripts/debug_inference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tokenizer import clean_text, encode, load_vocab, tokenize  # noqa: E402
from trainable_intent_model import IntentClassificationTransformer  # noqa: E402


def audit_artifacts() -> Dict[str, Any]:
    vocab_path = ARTIFACTS_DIR / "vocab.json"
    labels_path = ARTIFACTS_DIR / "label_mapping.json"
    model_path = ARTIFACTS_DIR / "best_model.pt"

    print("=== Artifact verification ===")
    for path in (model_path, vocab_path, labels_path):
        print(f"  exists {path.name}: {path.exists()} ({path})")

    vocab = load_vocab(str(vocab_path))
    label_to_id = json.loads(labels_path.read_text(encoding="utf-8"))
    id_to_intent = {int(i): name for name, i in label_to_id.items()}

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    state = checkpoint["model_state_dict"]

    print(f"  best_epoch: {checkpoint.get('best_epoch')}")
    print(f"  best_validation_loss: {checkpoint.get('best_validation_loss')}")
    print(f"  model_config: {config}")
    print(f"  vocab.json size: {len(vocab)}")
    print(f"  config vocab_size: {config['vocab_size']}")
    print(
        "  vocab sizes match:",
        len(vocab) == config["vocab_size"]
        == state["token_embedding.weight"].shape[0],
    )
    print(f"  label_mapping.json: {label_to_id}")
    print(f"  id_to_intent decode map: {id_to_intent}")
    print(f"  embedding shape: {tuple(state['token_embedding.weight'].shape)}")
    print(f"  classifier shape: {tuple(state['classifier.weight'].shape)}")
    print(f"  pad_token_id: {config.get('pad_token_id')}")
    print(f"  max_length: {config.get('max_length')}")

    return {
        "vocab": vocab,
        "label_to_id": label_to_id,
        "id_to_intent": id_to_intent,
        "checkpoint": checkpoint,
        "config": config,
    }


def load_model(config: Dict[str, Any], state_dict: Dict[str, torch.Tensor]):
    model = IntentClassificationTransformer(**config)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("\n=== Model load ===")
    print(f"  load_state_dict missing={missing} unexpected={unexpected}")
    print(f"  model.training after eval(): {model.training}")
    print(f"  model.vocab_size={model.vocab_size} embedding_dim={model.embedding_dim}")
    return model


def debug_predict(
    model: IntentClassificationTransformer,
    vocab: Dict[str, int],
    id_to_intent: Dict[int, str],
    query: str,
) -> Dict[str, Any]:
    cleaned = clean_text(query)
    tokens = tokenize(cleaned)
    token_ids = encode(query, vocab)
    unk_id = vocab["<UNK>"]
    unk_count = sum(1 for tid in token_ids if tid == unk_id)
    oov_tokens = [tok for tok in tokens if tok not in vocab]

    max_length = model.max_length
    truncated = token_ids[:max_length]
    attention = [1] * len(truncated)
    pad_count = max_length - len(truncated)
    padded_ids = truncated + [0] * pad_count
    attention_mask = attention + [0] * pad_count

    input_ids = torch.tensor([padded_ids], dtype=torch.long)
    attention_tensor = torch.tensor([attention_mask], dtype=torch.long)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_tensor)
        probabilities = F.softmax(logits, dim=-1)
    model.train(was_training)

    logits_row = logits[0]
    probs_row = probabilities[0]
    predicted_id = int(torch.argmax(probs_row).item())

    print("\n=== Debug prediction ===")
    print(f"  raw query: {query!r}")
    print(f"  cleaned query: {cleaned!r}")
    print(f"  tokens: {tokens}")
    print(f"  token ids: {token_ids}")
    print(f"  padded input_ids (first 16): {padded_ids[:16]}")
    print(f"  attention mask (first 16): {attention_mask[:16]}")
    print(f"  UNK count: {unk_count}/{len(token_ids)} OOV={oov_tokens}")
    print(f"  logits: {[round(float(x), 4) for x in logits_row.tolist()]}")
    print(
        "  probabilities:",
        {
            id_to_intent[i]: round(float(probs_row[i].item()), 4)
            for i in range(len(id_to_intent))
        },
    )
    print(f"  predicted label id: {predicted_id}")
    print(f"  predicted label: {id_to_intent[predicted_id]}")
    print(f"  model.training during predict: False (forced eval + no_grad)")

    return {
        "query": query,
        "cleaned": cleaned,
        "tokens": tokens,
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "unk_count": unk_count,
        "oov_tokens": oov_tokens,
        "logits": [float(x) for x in logits_row.tolist()],
        "probabilities": {
            id_to_intent[i]: float(probs_row[i].item())
            for i in range(len(id_to_intent))
        },
        "predicted_id": predicted_id,
        "predicted_intent": id_to_intent[predicted_id],
    }


def compare_to_nearest_training_examples(queries: List[str]) -> None:
    """Show whether debug queries resemble training distribution."""
    print("\n=== Training distribution notes (intents_v2) ===")
    rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / "data" / "intents_v2.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    vocab = load_vocab(str(ARTIFACTS_DIR / "vocab.json"))
    for query in queries:
        tokens = set(tokenize(clean_text(query)))
        # Count how many training rows share content tokens (excluding stop-ish)
        stop = {"the", "a", "an", "of", "to", "is", "me", "what", "for", "in", "on"}
        content = tokens - stop
        overlaps = []
        for row in rows:
            row_toks = set(tokenize(clean_text(row["text"])))
            shared = content & row_toks
            if shared:
                overlaps.append((len(shared), row["intent"], row["text"], sorted(shared)))
        overlaps.sort(reverse=True)
        print(f"\n  Query: {query!r}")
        print(f"  content tokens: {sorted(content)}")
        print(f"  in-vocab content: {[t for t in sorted(content) if t in vocab]}")
        print(f"  OOV content: {[t for t in sorted(content) if t not in vocab]}")
        print("  top overlapping training rows:")
        for score, intent, text, shared in overlaps[:5]:
            print(f"    [{intent}] shared={shared} :: {text[:90]}")


def main() -> None:
    info = audit_artifacts()
    model = load_model(info["config"], info["checkpoint"]["model_state_dict"])

    queries = [
        "show me nike shoes under 5000",
        "inventory level of store 12",
        "what is arbitration clause",
        "what is the capital of france",
    ]

    results = []
    for query in queries:
        results.append(
            debug_predict(
                model,
                info["vocab"],
                info["id_to_intent"],
                query,
            )
        )

    compare_to_nearest_training_examples(queries)

    # Sanity: run a clearly in-distribution LEGAL / ECOMMERCE line from v2
    print("\n=== In-distribution sanity checks ===")
    for query in [
        "Is arbitration locked in by this noncompete, or is court still available?",
        "Why is checkout blocking the wool coat because of an address check?",
        "Where do you keep paper towels in this store?",
        "What is gravity?",
    ]:
        debug_predict(model, info["vocab"], info["id_to_intent"], query)

    out_path = ARTIFACTS_DIR / "inference_audit.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
