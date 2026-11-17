"""Tests for sinusoidal positional encoding.

Run from the project directory:

    pytest tests/test_positional_encoding.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from positional_encoding import PositionalEncoding  # noqa: E402


def test_encoding_table_shape():
    pe = PositionalEncoding(max_length=32, embedding_dim=16)
    assert pe.encoding_shape() == (32, 16)
    assert pe.encodings.shape == (32, 16)


def test_forward_shape_correctness():
    pe = PositionalEncoding(max_length=32, embedding_dim=8)
    output = pe.forward(sequence_length=5)
    assert output.shape == (5, 8)


def test_identical_positions_produce_identical_vectors():
    pe = PositionalEncoding(max_length=16, embedding_dim=8)
    first = pe.forward(4)
    second = pe.forward(4)
    np.testing.assert_array_equal(first, second)
    # Position 2 taken from two different slices must match.
    np.testing.assert_array_equal(pe.forward(6)[2], pe.forward(3)[2])


def test_different_positions_produce_different_vectors():
    pe = PositionalEncoding(max_length=16, embedding_dim=8)
    vectors = pe.forward(5)
    # At least one pair of neighboring positions must differ.
    assert not np.allclose(vectors[0], vectors[1])
    assert not np.allclose(vectors[1], vectors[2])
    assert not np.allclose(vectors[0], vectors[4])


def test_addition_preserves_dimensions():
    pe = PositionalEncoding(max_length=16, embedding_dim=8)
    token_embeddings = np.ones((5, 8), dtype=np.float32)
    combined = pe.add_position_information(token_embeddings)
    assert combined.shape == (5, 8)
    assert combined.shape == token_embeddings.shape


def test_addition_is_elementwise_sum_with_positional_rows():
    pe = PositionalEncoding(max_length=16, embedding_dim=4)
    token_embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    combined = pe.add_position_information(token_embeddings)
    expected = token_embeddings + pe.forward(3)
    np.testing.assert_allclose(combined, expected, rtol=1e-6)


def test_even_dims_are_sine_odd_dims_are_cosine_at_position_zero():
    # pos=0 -> sin(0)=0 on even dims, cos(0)=1 on odd dims.
    pe = PositionalEncoding(max_length=8, embedding_dim=6)
    row = pe.forward(1)[0]
    np.testing.assert_allclose(row[0::2], 0.0, atol=1e-7)
    np.testing.assert_allclose(row[1::2], 1.0, atol=1e-7)


def test_matches_paper_formula_for_one_cell():
    embedding_dim = 8
    pe = PositionalEncoding(max_length=10, embedding_dim=embedding_dim)
    pos = 3
    i = 1  # affects dimensions 2i=2 and 2i+1=3
    angle = pos / (10000 ** (2 * i / embedding_dim))
    assert pe.encodings[pos, 2 * i] == pytest.approx(np.sin(angle), rel=1e-5)
    assert pe.encodings[pos, 2 * i + 1] == pytest.approx(np.cos(angle), rel=1e-5)


def test_odd_embedding_dim_still_builds():
    pe = PositionalEncoding(max_length=4, embedding_dim=5)
    output = pe.forward(3)
    assert output.shape == (3, 5)


def test_empty_sequence_forward():
    pe = PositionalEncoding(max_length=4, embedding_dim=8)
    output = pe.forward(0)
    assert output.shape == (0, 8)


def test_sequence_longer_than_max_length_raises():
    pe = PositionalEncoding(max_length=4, embedding_dim=8)
    with pytest.raises(ValueError):
        pe.forward(5)
    with pytest.raises(ValueError):
        pe.add_position_information(np.zeros((5, 8), dtype=np.float32))


def test_embedding_dim_mismatch_raises():
    pe = PositionalEncoding(max_length=8, embedding_dim=4)
    with pytest.raises(ValueError):
        pe.add_position_information(np.zeros((3, 8), dtype=np.float32))


def test_invalid_constructor_sizes_raise():
    with pytest.raises(ValueError):
        PositionalEncoding(max_length=0, embedding_dim=8)
    with pytest.raises(ValueError):
        PositionalEncoding(max_length=8, embedding_dim=0)
