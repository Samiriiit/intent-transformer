"""Tests for the NumPy token embedding layer.

Run from the project directory:

    pytest tests/test_embedding.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from embedding import TokenEmbedding  # noqa: E402


def test_embedding_shape_matches_constructor():
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, seed=0)
    assert embedding.embedding_shape() == (10, 8)
    assert embedding.weight.shape == (10, 8)


def test_forward_dimensions_for_sequence():
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8, seed=0)
    output = embedding.forward([2, 3, 4, 5])
    assert output.shape == (4, 8)
    assert output.dtype == np.float32


def test_lookup_correctness_matches_weight_rows():
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=4, seed=1)
    output = embedding.forward([2, 3, 4, 5])
    np.testing.assert_array_equal(output[0], embedding.weight[2])
    np.testing.assert_array_equal(output[1], embedding.weight[3])
    np.testing.assert_array_equal(output[2], embedding.weight[4])
    np.testing.assert_array_equal(output[3], embedding.weight[5])


def test_get_embedding_matches_forward_for_one_id():
    embedding = TokenEmbedding(vocab_size=6, embedding_dim=3, seed=2)
    single = embedding.get_embedding(4)
    from_forward = embedding.forward([4])[0]
    np.testing.assert_array_equal(single, from_forward)
    assert single.shape == (3,)


def test_repeated_token_lookup_returns_same_vector():
    embedding = TokenEmbedding(vocab_size=8, embedding_dim=5, seed=3)
    output = embedding.forward([2, 2, 2])
    np.testing.assert_array_equal(output[0], output[1])
    np.testing.assert_array_equal(output[1], output[2])
    np.testing.assert_array_equal(output[0], embedding.weight[2])


def test_pad_token_lookup_uses_row_zero():
    # In this project <PAD> is always id 0. It still has a vector; a later
    # attention mask will decide whether that vector is ignored.
    embedding = TokenEmbedding(vocab_size=5, embedding_dim=4, seed=4)
    pad_vector = embedding.get_embedding(0)
    padded_sequence = embedding.forward([2, 3, 0, 0])

    np.testing.assert_array_equal(pad_vector, embedding.weight[0])
    np.testing.assert_array_equal(padded_sequence[2], embedding.weight[0])
    np.testing.assert_array_equal(padded_sequence[3], embedding.weight[0])
    assert padded_sequence.shape == (4, 4)


def test_empty_sequence_returns_zero_rows():
    embedding = TokenEmbedding(vocab_size=5, embedding_dim=7, seed=5)
    output = embedding.forward([])
    assert output.shape == (0, 7)


def test_numpy_input_ids_are_accepted():
    embedding = TokenEmbedding(vocab_size=5, embedding_dim=3, seed=6)
    ids = np.array([1, 2, 3], dtype=np.int64)
    output = embedding.forward(ids)
    assert output.shape == (3, 3)
    np.testing.assert_array_equal(output[1], embedding.weight[2])


def test_out_of_range_id_raises():
    embedding = TokenEmbedding(vocab_size=5, embedding_dim=3, seed=7)
    with pytest.raises(ValueError):
        embedding.forward([2, 5])
    with pytest.raises(ValueError):
        embedding.get_embedding(5)
    with pytest.raises(ValueError):
        embedding.forward([-1])


def test_invalid_constructor_sizes_raise():
    with pytest.raises(ValueError):
        TokenEmbedding(vocab_size=0, embedding_dim=4)
    with pytest.raises(ValueError):
        TokenEmbedding(vocab_size=4, embedding_dim=0)


def test_seed_makes_initialization_reproducible():
    a = TokenEmbedding(vocab_size=4, embedding_dim=3, seed=99)
    b = TokenEmbedding(vocab_size=4, embedding_dim=3, seed=99)
    c = TokenEmbedding(vocab_size=4, embedding_dim=3, seed=100)
    np.testing.assert_array_equal(a.weight, b.weight)
    assert not np.array_equal(a.weight, c.weight)


def test_weight_is_mutable_for_future_training_updates():
    # Training will write gradients into weight. Confirm the matrix is not a view
    # that forbids assignment.
    embedding = TokenEmbedding(vocab_size=4, embedding_dim=2, seed=8)
    embedding.weight[2] = np.array([1.0, -1.0], dtype=np.float32)
    np.testing.assert_array_equal(embedding.get_embedding(2), [1.0, -1.0])
