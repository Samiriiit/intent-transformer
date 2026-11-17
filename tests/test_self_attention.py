"""Tests for single-head scaled dot-product self-attention.

Run from the project directory:

    pytest tests/test_self_attention.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from self_attention import SelfAttention  # noqa: E402


def _tiny_attention():
    """Build a small layer and overwrite W_* with easy numbers."""
    attn = SelfAttention(embedding_dim=2, attention_dim=2, seed=0)
    # Deterministic projections so mental arithmetic stays easy.
    attn.W_query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    attn.W_key = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    attn.W_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return attn


def test_qkv_shape_correctness():
    attn = SelfAttention(embedding_dim=4, attention_dim=3, seed=1)
    inputs = np.ones((5, 4), dtype=np.float32)
    query, key, value = attn.project(inputs)
    assert query.shape == (5, 3)
    assert key.shape == (5, 3)
    assert value.shape == (5, 3)


def test_score_matrix_shape():
    attn = _tiny_attention()
    inputs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    query, key, _ = attn.project(inputs)
    scores = attn.calculate_scores(query, key)
    assert scores.shape == (3, 3)


def test_scaled_score_calculation():
    attn = _tiny_attention()
    # With identity W_*, Q=K=inputs. attention_dim=2 => scale = sqrt(2).
    inputs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query, key, _ = attn.project(inputs)
    scores = attn.calculate_scores(query, key)
    expected = (inputs @ inputs.T) / np.sqrt(2.0)
    np.testing.assert_allclose(scores, expected, rtol=1e-5)


def test_softmax_rows_sum_to_one():
    attn = SelfAttention(embedding_dim=2, attention_dim=2, seed=2)
    scores = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    weights = attn.softmax(scores)
    np.testing.assert_allclose(weights.sum(axis=1), [1.0, 1.0], rtol=1e-5)
    assert np.all(weights >= 0.0)


def test_output_shape():
    attn = SelfAttention(embedding_dim=4, attention_dim=3, seed=3)
    inputs = np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32)
    output = attn.forward(inputs)
    assert output.shape == (6, 3)


def test_deterministic_initialization_with_same_seed():
    a = SelfAttention(embedding_dim=4, attention_dim=4, seed=42)
    b = SelfAttention(embedding_dim=4, attention_dim=4, seed=42)
    c = SelfAttention(embedding_dim=4, attention_dim=4, seed=7)
    np.testing.assert_array_equal(a.W_query, b.W_query)
    np.testing.assert_array_equal(a.W_key, b.W_key)
    np.testing.assert_array_equal(a.W_value, b.W_value)
    assert not np.array_equal(a.W_query, c.W_query)


def test_different_tokens_can_have_different_attention_distributions():
    attn = SelfAttention(embedding_dim=4, attention_dim=4, seed=5)
    rng = np.random.default_rng(11)
    inputs = rng.normal(size=(4, 4)).astype(np.float32)
    _, weights = attn.forward(inputs, return_attention=True)
    # At least two query rows should disagree (not a hard-coded pattern).
    assert not np.allclose(weights[0], weights[1])


def test_pad_key_positions_receive_zero_attention_after_masking():
    attn = _tiny_attention()
    # Four positions; last two are PAD.
    inputs = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mask = [1, 1, 0, 0]
    _, weights = attn.forward(inputs, attention_mask=mask, return_attention=True)

    # Columns 2 and 3 are PAD keys: every query must put ~0 mass there.
    np.testing.assert_allclose(weights[:, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(weights[:, 3], 0.0, atol=1e-6)
    # Real key columns still receive the full probability mass per row.
    np.testing.assert_allclose(weights[:, :2].sum(axis=1), 1.0, rtol=1e-5)


def test_output_without_a_mask():
    attn = _tiny_attention()
    inputs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    output = attn.forward(inputs)
    assert output.shape == (2, 2)
    # Without a mask, both key positions may receive non-zero weight.
    _, weights = attn.forward(inputs, return_attention=True)
    assert np.all(weights > 0.0)


def test_invalid_input_dimensions():
    attn = SelfAttention(embedding_dim=3, attention_dim=3, seed=0)
    with pytest.raises(ValueError, match="2-D"):
        attn.forward(np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="embedding dimension"):
        attn.forward(np.ones((2, 4), dtype=np.float32))


def test_invalid_mask_length():
    attn = SelfAttention(embedding_dim=2, attention_dim=2, seed=0)
    inputs = np.ones((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="length"):
        attn.forward(inputs, attention_mask=[1, 1])


def test_invalid_mask_values():
    attn = SelfAttention(embedding_dim=2, attention_dim=2, seed=0)
    inputs = np.ones((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="0 and 1"):
        attn.forward(inputs, attention_mask=[1, 2, 0])


def test_fully_masked_sequence_rejected():
    attn = SelfAttention(embedding_dim=2, attention_dim=2, seed=0)
    inputs = np.ones((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="completely masked"):
        attn.forward(inputs, attention_mask=[0, 0, 0])


def test_return_attention_behavior():
    attn = SelfAttention(embedding_dim=2, attention_dim=2, seed=9)
    inputs = np.ones((3, 2), dtype=np.float32)

    only_output = attn.forward(inputs, return_attention=False)
    assert isinstance(only_output, np.ndarray)
    assert only_output.shape == (3, 2)

    output, weights = attn.forward(inputs, return_attention=True)
    assert output.shape == (3, 2)
    assert weights.shape == (3, 3)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-5)


def test_default_attention_dim_matches_embedding_dim():
    attn = SelfAttention(embedding_dim=5, seed=1)
    assert attn.attention_dim == 5
    assert attn.W_query.shape == (5, 5)
