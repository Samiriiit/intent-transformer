"""Tests for the trainable PyTorch intent-classification Transformer.

Run from the project directory:

    pytest tests/test_trainable_intent_model.py
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from trainable_intent_model import IntentClassificationTransformer  # noqa: E402


def _small_model(**overrides):
    params = dict(
        vocab_size=50,
        embedding_dim=8,
        num_heads=2,
        hidden_dim=32,
        num_layers=2,
        num_classes=4,
        max_length=16,
        pad_token_id=0,
        dropout=0.0,
    )
    params.update(overrides)
    torch.manual_seed(0)
    return IntentClassificationTransformer(**params)


def _batch(batch_size=2, seq_len=6, vocab_size=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), generator=g)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # Mark last two positions as PAD on every row.
    if seq_len >= 2:
        input_ids[:, -2:] = 0
        attention_mask[:, -2:] = 0
    return input_ids, attention_mask


def test_model_is_nn_module():
    model = _small_model()
    assert isinstance(model, nn.Module)


def test_token_embedding_shape():
    model = _small_model()
    assert model.token_embedding.weight.shape == (50, 8)


def test_registered_positional_encoding_shape():
    model = _small_model()
    assert model.positional_encoding.shape == (1, 16, 8)


def test_positional_encoding_is_not_a_trainable_parameter():
    model = _small_model()
    param_ids = {id(p) for p in model.parameters()}
    assert id(model.positional_encoding) not in param_ids
    assert "positional_encoding" in dict(model.named_buffers())


def test_forward_output_shape_is_batch_by_four():
    model = _small_model()
    model.eval()
    input_ids, attention_mask = _batch()
    logits = model(input_ids, attention_mask)
    assert logits.shape == (2, 4)


def test_different_batch_sizes_work():
    model = _small_model()
    model.eval()
    for batch_size in (1, 3, 5):
        input_ids, attention_mask = _batch(batch_size=batch_size, seed=batch_size)
        assert model(input_ids, attention_mask).shape == (batch_size, 4)


def test_shorter_sequence_lengths_work():
    model = _small_model()
    model.eval()
    input_ids, attention_mask = _batch(seq_len=3, seed=1)
    assert model(input_ids, attention_mask).shape == (2, 4)


def test_maximum_sequence_length_works():
    model = _small_model()
    model.eval()
    input_ids, attention_mask = _batch(seq_len=16, seed=2)
    assert model(input_ids, attention_mask).shape == (2, 4)


def test_sequence_length_above_max_length_is_rejected():
    model = _small_model(max_length=16)
    input_ids = torch.ones(1, 17, dtype=torch.long)
    attention_mask = torch.ones(1, 17, dtype=torch.long)
    with pytest.raises(ValueError, match="max_length"):
        model(input_ids, attention_mask)


def test_mask_aware_pooling_ignores_pad_positions():
    model = _small_model()
    model.eval()
    # Build encoded-like tensor and mask; call pooling helper directly.
    encoded = torch.tensor(
        [
            [[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0]])
    pooled = model._masked_mean_pool(encoded, mask)
    # Mean of first two rows only → [2, 2], not diluted by 100.
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 2.0]]))


def test_changing_pad_ids_does_not_change_logits_in_eval():
    model = _small_model(dropout=0.0)
    model.eval()
    input_ids, attention_mask = _batch(batch_size=1, seq_len=6, seed=3)
    # Ensure last two are PAD in the mask.
    assert torch.all(attention_mask[0, -2:] == 0)

    baseline = model(input_ids, attention_mask)

    altered = input_ids.clone()
    altered[0, -2:] = torch.tensor([7, 11])  # different PAD-slot ids
    altered_logits = model(altered, attention_mask)

    torch.testing.assert_close(baseline, altered_logits, rtol=1e-5, atol=1e-5)


def test_logits_contain_finite_values():
    model = _small_model()
    model.eval()
    logits = model(*_batch())
    assert torch.isfinite(logits).all()


def test_predict_proba_shape_and_rows_sum_to_one():
    model = _small_model()
    probs = model.predict_proba(*_batch(batch_size=3, seed=4))
    assert probs.shape == (3, 4)
    torch.testing.assert_close(
        probs.sum(dim=-1), torch.ones(3), rtol=1e-5, atol=1e-5
    )


def test_forward_returns_logits_not_probabilities():
    model = _small_model(dropout=0.0)
    model.eval()
    input_ids, attention_mask = _batch(seed=5)
    logits = model(input_ids, attention_mask)
    # Softmax rows sum to 1; raw logits generally do not.
    assert not torch.allclose(logits.sum(dim=-1), torch.ones(logits.size(0)))
    probs = torch.softmax(logits, dim=-1)
    assert not torch.allclose(logits, probs)


def test_invalid_rank_for_input_ids():
    model = _small_model()
    with pytest.raises(ValueError, match="input_ids must be rank 2"):
        model(torch.ones(4, dtype=torch.long), torch.ones(2, 4, dtype=torch.long))


def test_invalid_rank_for_attention_mask():
    model = _small_model()
    with pytest.raises(ValueError, match="attention_mask must be rank 2"):
        model(torch.ones(2, 4, dtype=torch.long), torch.ones(4, dtype=torch.long))


def test_mismatched_input_and_mask_shapes():
    model = _small_model()
    with pytest.raises(ValueError, match="does not match"):
        model(torch.ones(2, 4, dtype=torch.long), torch.ones(2, 5, dtype=torch.long))


def test_invalid_attention_mask_values():
    model = _small_model()
    input_ids = torch.ones(1, 4, dtype=torch.long)
    mask = torch.tensor([[1, 2, 0, 1]])
    with pytest.raises(ValueError, match="0 and 1"):
        model(input_ids, mask)


def test_fully_padded_sample_is_rejected():
    model = _small_model()
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])
    with pytest.raises(ValueError, match="non-PAD"):
        model(input_ids, attention_mask)


def test_negative_token_id_is_rejected():
    model = _small_model()
    input_ids = torch.tensor([[1, -1, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    with pytest.raises(ValueError, match=r"\[0,"):
        model(input_ids, attention_mask)


def test_token_id_greater_or_equal_vocab_size_is_rejected():
    model = _small_model(vocab_size=50)
    input_ids = torch.tensor([[1, 50, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    with pytest.raises(ValueError, match=r"\[0,"):
        model(input_ids, attention_mask)


def test_embedding_dim_not_divisible_by_num_heads_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        IntentClassificationTransformer(
            vocab_size=50, embedding_dim=10, num_heads=3, hidden_dim=32
        )


def test_invalid_dropout_is_rejected():
    with pytest.raises(ValueError, match="dropout"):
        IntentClassificationTransformer(vocab_size=50, dropout=-0.1)
    with pytest.raises(ValueError, match="dropout"):
        IntentClassificationTransformer(vocab_size=50, dropout=1.5)


def test_gradients_flow_through_trainable_components():
    model = _small_model(dropout=0.0)
    model.train()
    input_ids, attention_mask = _batch(batch_size=2, seq_len=6, seed=9)

    model.zero_grad(set_to_none=True)
    logits = model(input_ids, attention_mask)
    logits.sum().backward()

    trainable = [
        ("token_embedding", model.token_embedding.weight),
        ("classifier", model.classifier.weight),
    ]
    # At least one parameter inside the encoder stack must receive a grad.
    encoder_grads = [
        p.grad
        for p in model.encoder.parameters()
        if p.requires_grad
    ]
    assert any(g is not None for g in encoder_grads)

    for name, param in trainable:
        assert param.grad is not None, f"{name} missing grad"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"

    for grad in encoder_grads:
        if grad is not None:
            assert torch.isfinite(grad).all()

    # Positional encoding is a buffer — must not act like a trainable weight.
    assert model.positional_encoding.grad is None
