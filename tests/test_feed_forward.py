"""Tests for the position-wise feed-forward network.

Run from the project directory:

    pytest tests/test_feed_forward.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from feed_forward import FeedForward  # noqa: E402


def test_w1_shape():
    ffn = FeedForward(embedding_dim=8, hidden_dim=32, seed=0)
    assert ffn.W1.shape == (8, 32)


def test_b1_shape():
    ffn = FeedForward(embedding_dim=8, hidden_dim=32, seed=0)
    assert ffn.b1.shape == (32,)


def test_w2_shape():
    ffn = FeedForward(embedding_dim=8, hidden_dim=32, seed=0)
    assert ffn.W2.shape == (32, 8)


def test_b2_shape():
    ffn = FeedForward(embedding_dim=8, hidden_dim=32, seed=0)
    assert ffn.b2.shape == (8,)


def test_first_linear_output_shape():
    ffn = FeedForward(embedding_dim=8, hidden_dim=32, seed=1)
    inputs = np.ones((6, 8), dtype=np.float32)
    hidden = ffn.first_linear(inputs)
    assert hidden.shape == (6, 32)


def test_relu_preserves_shape():
    ffn = FeedForward(embedding_dim=4, hidden_dim=8, seed=0)
    values = np.array([[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]], dtype=np.float32)
    activated = ffn.relu(values)
    assert activated.shape == values.shape


def test_relu_replaces_negative_values_with_zero():
    ffn = FeedForward(embedding_dim=4, hidden_dim=8, seed=0)
    values = np.array([[-1.5, -0.1], [2.0, -9.0]], dtype=np.float32)
    activated = ffn.relu(values)
    assert activated[0, 0] == 0.0
    assert activated[0, 1] == 0.0
    assert activated[1, 1] == 0.0


def test_relu_preserves_positive_values():
    ffn = FeedForward(embedding_dim=4, hidden_dim=8, seed=0)
    values = np.array([[-1.0, 2.5], [3.0, -4.0]], dtype=np.float32)
    activated = ffn.relu(values)
    assert activated[0, 1] == pytest.approx(2.5)
    assert activated[1, 0] == pytest.approx(3.0)


def test_relu_does_not_modify_input_in_place():
    ffn = FeedForward(embedding_dim=4, hidden_dim=8, seed=0)
    values = np.array([[-1.0, 2.0]], dtype=np.float32)
    original = values.copy()
    _ = ffn.relu(values)
    np.testing.assert_array_equal(values, original)


def test_final_output_shape_equals_input_shape():
    ffn = FeedForward(embedding_dim=8, hidden_dim=32, seed=2)
    inputs = np.ones((6, 8), dtype=np.float32)
    output = ffn.forward(inputs)
    assert output.shape == inputs.shape == (6, 8)


def test_deterministic_initialization_with_same_seed():
    a = FeedForward(embedding_dim=8, hidden_dim=16, seed=42)
    b = FeedForward(embedding_dim=8, hidden_dim=16, seed=42)
    np.testing.assert_array_equal(a.W1, b.W1)
    np.testing.assert_array_equal(a.b1, b.b1)
    np.testing.assert_array_equal(a.W2, b.W2)
    np.testing.assert_array_equal(a.b2, b.b2)


def test_different_seeds_produce_different_weights():
    a = FeedForward(embedding_dim=8, hidden_dim=16, seed=1)
    b = FeedForward(embedding_dim=8, hidden_dim=16, seed=2)
    assert not np.array_equal(a.W1, b.W1)


def test_repeated_calls_with_fixed_parameters_are_identical():
    ffn = FeedForward(embedding_dim=4, hidden_dim=8, seed=3)
    inputs = np.array([[0.5, -0.25, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    first = ffn.forward(inputs)
    second = ffn.forward(inputs)
    np.testing.assert_array_equal(first, second)


def test_return_intermediates_behavior():
    ffn = FeedForward(embedding_dim=4, hidden_dim=8, seed=4)
    inputs = np.ones((3, 4), dtype=np.float32)

    only_output = ffn.forward(inputs, return_intermediates=False)
    assert isinstance(only_output, np.ndarray)
    assert only_output.shape == (3, 4)

    output, pre, post = ffn.forward(inputs, return_intermediates=True)
    assert output.shape == (3, 4)
    assert pre.shape == (3, 8)
    assert post.shape == (3, 8)
    np.testing.assert_array_equal(post, ffn.relu(pre))
    np.testing.assert_array_equal(output, ffn.second_linear(post))


def test_invalid_one_dimensional_input():
    ffn = FeedForward(embedding_dim=4, hidden_dim=8, seed=0)
    with pytest.raises(ValueError, match="2-D"):
        ffn.forward(np.ones(4, dtype=np.float32))


def test_invalid_three_dimensional_input():
    ffn = FeedForward(embedding_dim=4, hidden_dim=8, seed=0)
    with pytest.raises(ValueError, match="2-D"):
        ffn.forward(np.ones((2, 3, 4), dtype=np.float32))


def test_incorrect_embedding_dimension():
    ffn = FeedForward(embedding_dim=4, hidden_dim=8, seed=0)
    with pytest.raises(ValueError, match="embedding dimension"):
        ffn.forward(np.ones((3, 5), dtype=np.float32))


def test_empty_sequence_rejection():
    ffn = FeedForward(embedding_dim=4, hidden_dim=8, seed=0)
    with pytest.raises(ValueError, match="at least one token"):
        ffn.forward(np.ones((0, 4), dtype=np.float32))


def test_invalid_constructor_dimensions():
    with pytest.raises(ValueError, match="embedding_dim"):
        FeedForward(embedding_dim=0, hidden_dim=8)
    with pytest.raises(ValueError, match="hidden_dim"):
        FeedForward(embedding_dim=8, hidden_dim=0)


def test_row_independence():
    """Changing one token row must not change the output of other rows."""
    ffn = FeedForward(embedding_dim=4, hidden_dim=8, seed=5)
    inputs = np.array(
        [
            [1.0, 0.0, -0.5, 2.0],
            [0.0, 1.0, 0.5, -1.0],
            [0.25, -0.25, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    baseline = ffn.forward(inputs)

    modified = inputs.copy()
    modified[1] = np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32)
    after = ffn.forward(modified)

    np.testing.assert_array_equal(after[0], baseline[0])
    np.testing.assert_array_equal(after[2], baseline[2])
    assert not np.array_equal(after[1], baseline[1])


def test_shape_walkthrough_six_by_eight_with_hidden_thirty_two():
    ffn = FeedForward(embedding_dim=8, hidden_dim=32, seed=6)
    inputs = np.ones((6, 8), dtype=np.float32)
    output, pre, post = ffn.forward(inputs, return_intermediates=True)
    assert inputs.shape == (6, 8)
    assert pre.shape == (6, 32)
    assert post.shape == (6, 32)
    assert output.shape == (6, 8)
