"""Tests for multi-head self-attention.

Run from the project directory:

    pytest tests/test_multi_head_attention.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from multi_head_attention import MultiHeadAttention  # noqa: E402


def test_constructor_rejects_non_divisible_embedding_dim():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(embedding_dim=10, num_heads=3)
    with pytest.raises(ValueError, match="positive"):
        MultiHeadAttention(embedding_dim=0, num_heads=1)
    with pytest.raises(ValueError, match="positive"):
        MultiHeadAttention(embedding_dim=8, num_heads=0)


def test_head_dim_calculation():
    mha = MultiHeadAttention(embedding_dim=64, num_heads=4, seed=0)
    assert mha.head_dim == 16
    assert mha.W_query.shape == (64, 64)
    assert mha.W_output.shape == (64, 64)


def test_project_shapes():
    mha = MultiHeadAttention(embedding_dim=8, num_heads=2, seed=1)
    inputs = np.ones((5, 8), dtype=np.float32)
    query, key, value = mha.project(inputs)
    assert query.shape == (5, 8)
    assert key.shape == (5, 8)
    assert value.shape == (5, 8)


def test_split_heads_shape_and_values():
    mha = MultiHeadAttention(embedding_dim=8, num_heads=4, seed=0)
    # Fill with a simple range so reshape/transpose can be checked by hand.
    matrix = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    heads = mha.split_heads(matrix)
    assert heads.shape == (4, 6, 2)

    # Head 0 owns columns [0:2], head 1 owns [2:4], etc.
    np.testing.assert_array_equal(heads[0], matrix[:, 0:2])
    np.testing.assert_array_equal(heads[1], matrix[:, 2:4])
    np.testing.assert_array_equal(heads[2], matrix[:, 4:6])
    np.testing.assert_array_equal(heads[3], matrix[:, 6:8])


def test_combine_heads_inverts_split_heads():
    mha = MultiHeadAttention(embedding_dim=8, num_heads=4, seed=0)
    matrix = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    restored = mha.combine_heads(mha.split_heads(matrix))
    np.testing.assert_array_equal(restored, matrix)


def test_attention_weight_and_head_output_shapes():
    mha = MultiHeadAttention(embedding_dim=8, num_heads=2, seed=2)
    inputs = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    query, key, value = mha.project(inputs)
    head_outputs, weights = mha.calculate_attention(
        mha.split_heads(query),
        mha.split_heads(key),
        mha.split_heads(value),
    )
    assert head_outputs.shape == (2, 5, 4)
    assert weights.shape == (2, 5, 5)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, rtol=1e-5)


def test_scaling_uses_sqrt_head_dim():
    mha = MultiHeadAttention(embedding_dim=4, num_heads=2, seed=0)
    # Identity-like projections for a controllable check.
    mha.W_query = np.eye(4, dtype=np.float32)
    mha.W_key = np.eye(4, dtype=np.float32)
    mha.W_value = np.eye(4, dtype=np.float32)

    inputs = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    query_heads = mha.split_heads(inputs)  # with identity W, Q=inputs
    key_heads = mha.split_heads(inputs)

    # Manually: head 0 vectors are [1,0] and [0,1]; raw score[0,0]=1.
    # Scaled by sqrt(head_dim)=sqrt(2).
    scores = query_heads @ np.transpose(key_heads, (0, 2, 1))
    scaled = scores / np.sqrt(2.0)
    _, weights = mha.calculate_attention(query_heads, key_heads, query_heads)
    # Softmax of the manually scaled scores should match returned weights.
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(weights, expected, rtol=1e-5)


def test_forward_output_shape_matches_embedding_dim():
    mha = MultiHeadAttention(embedding_dim=16, num_heads=4, seed=3)
    inputs = np.ones((7, 16), dtype=np.float32)
    output = mha.forward(inputs)
    assert output.shape == (7, 16)


def test_return_attention_shape():
    mha = MultiHeadAttention(embedding_dim=8, num_heads=2, seed=4)
    inputs = np.ones((4, 8), dtype=np.float32)
    output, weights = mha.forward(inputs, return_attention=True)
    assert output.shape == (4, 8)
    assert weights.shape == (2, 4, 4)


def test_pad_keys_receive_zero_weight_on_every_head():
    mha = MultiHeadAttention(embedding_dim=4, num_heads=2, seed=5)
    inputs = np.random.default_rng(1).normal(size=(4, 4)).astype(np.float32)
    mask = [1, 1, 0, 0]
    _, weights = mha.forward(inputs, attention_mask=mask, return_attention=True)
    np.testing.assert_allclose(weights[:, :, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(weights[:, :, 3], 0.0, atol=1e-6)
    np.testing.assert_allclose(weights[:, :, :2].sum(axis=-1), 1.0, rtol=1e-5)


def test_heads_can_differ_without_manual_roles():
    mha = MultiHeadAttention(embedding_dim=8, num_heads=2, seed=6)
    inputs = np.random.default_rng(2).normal(size=(5, 8)).astype(np.float32)
    _, weights = mha.forward(inputs, return_attention=True)
    # Different heads are free to disagree; we do not assign meanings.
    assert not np.allclose(weights[0], weights[1])


def test_same_seed_is_deterministic():
    a = MultiHeadAttention(embedding_dim=8, num_heads=2, seed=99)
    b = MultiHeadAttention(embedding_dim=8, num_heads=2, seed=99)
    c = MultiHeadAttention(embedding_dim=8, num_heads=2, seed=100)
    np.testing.assert_array_equal(a.W_query, b.W_query)
    np.testing.assert_array_equal(a.W_output, b.W_output)
    assert not np.array_equal(a.W_query, c.W_query)


def test_invalid_input_dimensions():
    mha = MultiHeadAttention(embedding_dim=8, num_heads=2, seed=0)
    with pytest.raises(ValueError, match="2-D"):
        mha.forward(np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="embedding dimension"):
        mha.forward(np.ones((3, 4), dtype=np.float32))


def test_invalid_mask_length_and_values():
    mha = MultiHeadAttention(embedding_dim=4, num_heads=2, seed=0)
    inputs = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="length"):
        mha.forward(inputs, attention_mask=[1, 1])
    with pytest.raises(ValueError, match="0 and 1"):
        mha.forward(inputs, attention_mask=[1, 2, 0])


def test_fully_masked_sequence_rejected():
    mha = MultiHeadAttention(embedding_dim=4, num_heads=2, seed=0)
    inputs = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="completely masked"):
        mha.forward(inputs, attention_mask=[0, 0, 0])


def test_forward_without_mask():
    mha = MultiHeadAttention(embedding_dim=8, num_heads=2, seed=7)
    inputs = np.ones((3, 8), dtype=np.float32)
    output, weights = mha.forward(inputs, return_attention=True)
    assert output.shape == (3, 8)
    assert np.all(weights > 0.0)
